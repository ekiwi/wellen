// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::signals::{Signal, SignalSource};
use crate::values::*;
use fst_native::*;
use std::collections::HashMap;
use std::io::{BufRead, Seek};

pub fn read(filename: &str) -> (Hierarchy, Box<dyn SignalSource>) {
    let input = std::fs::File::open(filename).expect("failed to open input file!");
    let mut reader = fst_native::FstReader::open(std::io::BufReader::new(input)).unwrap();
    let hierarchy = read_hierarchy(&mut reader);
    let db = Box::new(FstWaveDatabase::new(reader));
    (hierarchy, db)
}

struct FstWaveDatabase<R: BufRead + Seek> {
    reader: FstReader<R>,
}

impl<R: BufRead + Seek> FstWaveDatabase<R> {
    fn new(reader: FstReader<R>) -> Self {
        FstWaveDatabase { reader }
    }
}

impl<R: BufRead + Seek> SignalSource for FstWaveDatabase<R> {
    fn load_signals(&mut self, ids: &[SignalIdx]) -> Vec<Signal> {
        let fst_ids = ids
            .iter()
            .map(|ii| FstSignalHandle::from_index(*ii as usize))
            .collect::<Vec<_>>();
        let filter = FstFilter::filter_signals(fst_ids);
        let mut signals = Vec::with_capacity(ids.len());
        let foo =
            |time: u64, handle: FstSignalHandle, value: FstSignalValue| todo!("implement callback");

        self.reader.read_signals(&filter, foo).unwrap();
        signals
    }

    fn get_time_table(&self) -> Vec<Time> {
        todo!("add time table reading to FST lib")
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

fn read_hierarchy<F: BufRead + Seek>(reader: &mut FstReader<F>) -> Hierarchy {
    let mut h = HierarchyBuilder::default();
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
                    handle.get_index() as u32,
                );
            }
            FstHierarchyEntry::PathName { id, name } => {
                path_names.insert(id, name);
            }
            FstHierarchyEntry::SourceStem { .. } => todo!(),
            FstHierarchyEntry::Comment { .. } => todo!(),
            FstHierarchyEntry::EnumTable { .. } => todo!(),
            FstHierarchyEntry::EnumTableRef { .. } => todo!(),
            FstHierarchyEntry::AttributeEnd => todo!(),
        };
    };
    reader.read_hierarchy(cb).unwrap();
    h.print_statistics();
    h.finish()
}

fn read_values<F: BufRead + Seek>(reader: &mut FstReader<F>, hierarchy: &Hierarchy) -> Values {
    let mut v = ValueBuilder::default();
    for var in hierarchy.iter_vars() {
        v.add_signal(var.handle(), var.length())
    }

    let cb = |time: u64, handle: FstSignalHandle, value: FstSignalValue| match value {
        FstSignalValue::String(value) => {
            v.value_and_time(handle.get_index() as SignalIdx, time, value.as_bytes());
        }
        FstSignalValue::Real(value) => {
            v.value_and_time(handle.get_index() as SignalIdx, time, &value.to_be_bytes());
        }
    };
    let filter = FstFilter::all();
    reader.read_signals(&filter, cb).unwrap();
    v.print_statistics();
    v.finish()
}
