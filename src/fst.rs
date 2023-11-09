// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::signals::{Signal, SignalSource, Time};
use crate::Waveform;
use fst_native::*;
use std::collections::HashMap;
use std::io::{BufRead, Seek};

pub fn read(filename: &str) -> Waveform {
    let input = std::fs::File::open(filename).expect("failed to open input file!");
    let mut reader = FstReader::open_and_read_time_table(std::io::BufReader::new(input)).unwrap();
    let hierarchy = read_hierarchy(&mut reader);
    let db = Box::new(FstWaveDatabase::new(reader));
    Waveform::new(hierarchy, db)
}

pub fn read_from_bytes(bytes: Vec<u8>) -> Waveform {
    let mut reader = FstReader::open_and_read_time_table(std::io::Cursor::new(bytes)).unwrap();
    let hierarchy = read_hierarchy(&mut reader);
    let db = Box::new(FstWaveDatabase::new(reader));
    Waveform::new(hierarchy, db)
}

struct FstWaveDatabase<R: BufRead + Seek> {
    reader: FstReader<R>,
    time_table: Vec<u64>,
}

impl<R: BufRead + Seek> FstWaveDatabase<R> {
    fn new(reader: FstReader<R>) -> Self {
        let time_table = reader.get_time_table().unwrap().to_vec();
        FstWaveDatabase { reader, time_table }
    }
}

impl<R: BufRead + Seek> SignalSource for FstWaveDatabase<R> {
    fn load_signals(&mut self, ids: &[(SignalRef, SignalLength)]) -> Vec<Signal> {
        // create a FST filter
        let fst_ids = ids
            .iter()
            .map(|(ii, _)| FstSignalHandle::from_index(ii.index()))
            .collect::<Vec<_>>();
        let filter = FstFilter::filter_signals(fst_ids);

        // lookup data structure for time table indices
        let mut time_table = self.time_table.iter().enumerate();
        let mut index_and_time = time_table.next().unwrap();

        // store signals
        let mut signals = Vec::new();
        let idx_to_pos: HashMap<usize, usize> =
            HashMap::from_iter(ids.iter().map(|(r, _)| r.index()).enumerate());
        let foo = |time: u64, handle: FstSignalHandle, _value: FstSignalValue| {
            // determine time index
            while *(index_and_time.1) < time {
                index_and_time = time_table.next().unwrap();
            }
            let time_idx = index_and_time.0;
            let signal_pos = idx_to_pos[&handle.get_index()];
            todo!()
        };

        self.reader.read_signals(&filter, foo).unwrap();
        signals
    }

    fn get_time_table(&self) -> Vec<Time> {
        self.time_table.clone()
    }
}

fn convert_scope_tpe(tpe: FstScopeType) -> ScopeType {
    match tpe {
        FstScopeType::Module => ScopeType::Module,
        _ => ScopeType::Todo,
    }
}

fn convert_var_tpe(tpe: FstVarType) -> VarType {
    match tpe {
        FstVarType::Wire => VarType::Wire,
        _ => VarType::Todo,
    }
}

fn convert_var_direction(tpe: FstVarDirection) -> VarDirection {
    match tpe {
        FstVarDirection::Input => VarDirection::Input,
        _ => VarDirection::Todo,
    }
}

fn convert_timescale(exponent: i8) -> Timescale {
    if exponent >= 0 {
        Timescale::new(10u32.pow(exponent as u32), TimescaleUnit::Seconds)
    } else if exponent >= -3 {
        Timescale::new(
            10u32.pow((exponent + 3) as u32),
            TimescaleUnit::MilliSeconds,
        )
    } else if exponent >= -6 {
        Timescale::new(
            10u32.pow((exponent + 6) as u32),
            TimescaleUnit::MicroSeconds,
        )
    } else if exponent >= -9 {
        Timescale::new(10u32.pow((exponent + 9) as u32), TimescaleUnit::NanoSeconds)
    } else if exponent >= -12 {
        Timescale::new(
            10u32.pow((exponent + 12) as u32),
            TimescaleUnit::PicoSeconds,
        )
    } else if exponent >= -15 {
        Timescale::new(
            10u32.pow((exponent + 15) as u32),
            TimescaleUnit::FemtoSeconds,
        )
    } else {
        panic!("Unexpected timescale exponent: {}", exponent);
    }
}

fn read_hierarchy<F: BufRead + Seek>(reader: &mut FstReader<F>) -> Hierarchy {
    let mut h = HierarchyBuilder::default();
    // load meta-data
    let fst_header = reader.get_header();
    h.set_version(fst_header.version.trim().to_string());
    h.set_date(fst_header.date.trim().to_string());
    h.set_timescale(convert_timescale(fst_header.timescale_exponent));

    let mut path_names = HashMap::new();

    let cb = |entry: FstHierarchyEntry| {
        match entry {
            FstHierarchyEntry::Scope { tpe, name, .. } => {
                h.add_scope(name, convert_scope_tpe(tpe));
            }
            FstHierarchyEntry::UpScope => h.pop_scope(),
            FstHierarchyEntry::Var {
                tpe,
                direction,
                name,
                length,
                handle,
                ..
            } => {
                h.add_var(
                    name,
                    convert_var_tpe(tpe),
                    convert_var_direction(direction),
                    length,
                    SignalRef::from_index(handle.get_index()).unwrap(),
                );
            }
            FstHierarchyEntry::PathName { id, name } => {
                path_names.insert(id, name);
            }
            FstHierarchyEntry::SourceStem { .. } => todo!(),
            FstHierarchyEntry::Comment { .. } => {} // ignored
            FstHierarchyEntry::EnumTable { .. } => todo!(),
            FstHierarchyEntry::EnumTableRef { .. } => todo!(),
            FstHierarchyEntry::AttributeEnd => todo!(),
        };
    };
    reader.read_hierarchy(cb).unwrap();
    h.finish()
}
