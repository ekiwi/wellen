// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::signals::{Signal, SignalEncoding, SignalSource, Time, TimeTableIdx};
use crate::vcd::{parse_index, usize_div_ceil};
use crate::wavemem::{check_states, expand_states, write_n_state, States};
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
    fn load_signals(
        &mut self,
        ids: &[(SignalRef, SignalType)],
        _multi_threaded: bool,
    ) -> Vec<Signal> {
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
        let mut signals = ids
            .iter()
            .map(|(id, tpe)| SignalWriter::new(*id, *tpe))
            .collect::<Vec<_>>();
        let idx_to_pos: HashMap<usize, usize> =
            HashMap::from_iter(ids.iter().map(|(r, _)| r.index()).enumerate());
        let foo = |time: u64, handle: FstSignalHandle, value: FstSignalValue| {
            // determine time index
            while *(index_and_time.1) < time {
                index_and_time = time_table.next().unwrap();
            }
            let time_idx = index_and_time.0 as TimeTableIdx;
            debug_assert_eq!(*index_and_time.1, time);
            let signal_pos = idx_to_pos[&handle.get_index()];
            signals[signal_pos].add_change(time_idx, handle, value);
        };

        self.reader.read_signals(&filter, foo).unwrap();
        signals.into_iter().map(|w| w.finish()).collect()
    }

    fn get_time_table(&self) -> Vec<Time> {
        self.time_table.clone()
    }

    fn print_statistics(&self) {
        println!("FST backend currently has not statistics to print.");
    }
}

struct SignalWriter {
    tpe: SignalType,
    id: SignalRef,
    /// used to check that everything is going well
    handle: FstSignalHandle,
    data_bytes: Vec<u8>,
    strings: Vec<String>,
    time_indices: Vec<TimeTableIdx>,
    max_states: Option<States>,
}

impl SignalWriter {
    fn new(id: SignalRef, tpe: SignalType) -> Self {
        Self {
            tpe,
            id,
            handle: FstSignalHandle::from_index(id.index()),
            data_bytes: Vec::new(),
            strings: Vec::new(),
            time_indices: Vec::new(),
            max_states: None,
        }
    }

    fn add_change(
        &mut self,
        time_idx: TimeTableIdx,
        handle: FstSignalHandle,
        value: FstSignalValue,
    ) {
        debug_assert_eq!(handle, self.handle);
        if let Some(prev_idx) = self.time_indices.last() {
            debug_assert!(*prev_idx <= time_idx);
        }
        self.time_indices.push(time_idx);
        match value {
            FstSignalValue::String(value) => match self.tpe {
                SignalType::String => {
                    self.strings
                        .push(String::from_utf8_lossy(value).to_string());
                }
                SignalType::BitVector(len, _) => {
                    debug_assert_eq!(value.len(), len.get() as usize);
                    let new_states = check_states(value).unwrap_or_else(|| {
                        panic!(
                            "Unexpected signal value: {}",
                            String::from_utf8_lossy(value)
                        )
                    });
                    let encode_as = if let Some(max_states) = self.max_states {
                        let combined = States::join(max_states, new_states);
                        if combined != max_states {
                            // With FST we do not know how many states the signal needs, thus we first assume
                            // the minimal number of states. If that turns out to be wrong, we go back and
                            // expand the existing data.
                            let num_prev_entries = self.time_indices.len() - 1; // -1 to account for the new index
                            self.data_bytes = expand_entries(
                                max_states,
                                combined,
                                &self.data_bytes,
                                num_prev_entries,
                                len.get(),
                            );
                            self.max_states = Some(combined);
                        }
                        combined
                    } else {
                        self.max_states = Some(new_states);
                        new_states
                    };
                    write_n_state(encode_as, value, &mut self.data_bytes);
                }
                SignalType::Real => panic!(
                    "Expecting reals, but go: {}",
                    String::from_utf8_lossy(value)
                ),
            },
            FstSignalValue::Real(value) => {
                debug_assert_eq!(self.tpe, SignalType::Real);
                self.data_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    fn finish(self) -> Signal {
        match self.tpe {
            SignalType::String => {
                debug_assert!(self.data_bytes.is_empty());
                Signal::new_var_len(self.id, self.time_indices, self.strings)
            }
            SignalType::Real => {
                debug_assert!(self.strings.is_empty());
                Signal::new_fixed_len(
                    self.id,
                    self.time_indices,
                    SignalEncoding::Real,
                    8,
                    self.data_bytes,
                )
            }
            SignalType::BitVector(len, _) => {
                debug_assert!(self.strings.is_empty());
                let states = if let Some(max_states) = self.max_states {
                    debug_assert!(!self.data_bytes.is_empty());
                    max_states
                } else {
                    debug_assert!(self.data_bytes.is_empty());
                    States::Two
                };
                let encoding = SignalEncoding::BitVector(states, len.get());
                let bytes_per_entry =
                    usize_div_ceil(len.get() as usize, states.bits_in_a_byte()) as u32;
                Signal::new_fixed_len(
                    self.id,
                    self.time_indices,
                    encoding,
                    bytes_per_entry,
                    self.data_bytes,
                )
            }
        }
    }
}

fn expand_entries(from: States, to: States, old: &[u8], entries: usize, bits: u32) -> Vec<u8> {
    let expansion_factor = to.bits() / from.bits();
    debug_assert!(expansion_factor > 1);
    let bytes_per_entry = old.len() / entries;
    debug_assert_eq!(old.len(), bytes_per_entry * entries);
    let mut data = Vec::with_capacity(old.len() * expansion_factor);
    for value in old.chunks(bytes_per_entry) {
        expand_states(from, to, bits, value, &mut data);
    }
    data
}

fn convert_scope_tpe(tpe: FstScopeType) -> ScopeType {
    match tpe {
        FstScopeType::Module => ScopeType::Module,
        FstScopeType::Task => ScopeType::Task,
        FstScopeType::Function => ScopeType::Function,
        FstScopeType::Begin => ScopeType::Begin,
        FstScopeType::Fork => ScopeType::Fork,
        other => panic!("Unsupported scope type: {:?}", other),
    }
}

fn convert_var_tpe(tpe: FstVarType) -> VarType {
    match tpe {
        FstVarType::Wire => VarType::Wire,
        FstVarType::Event => VarType::Event,
        FstVarType::Integer => VarType::Integer,
        FstVarType::Parameter => VarType::Parameter,
        FstVarType::Real => VarType::Real,
        FstVarType::RealParameter => VarType::Parameter,
        FstVarType::Reg => VarType::Reg,
        FstVarType::Supply0 => VarType::Supply0,
        FstVarType::Supply1 => VarType::Supply1,
        FstVarType::Time => VarType::Time,
        FstVarType::Tri => VarType::Tri,
        FstVarType::TriAnd => VarType::TriAnd,
        FstVarType::TriOr => VarType::TriOr,
        FstVarType::TriReg => VarType::TriReg,
        FstVarType::Tri0 => VarType::Tri0,
        FstVarType::Tri1 => VarType::Tri1,
        FstVarType::Wand => VarType::WAnd,
        FstVarType::Wor => VarType::WOr,
        FstVarType::Port => VarType::Port,
        FstVarType::SparseArray => todo!("Implement support for SparseArray type!"),
        FstVarType::RealTime => todo!("Implement support for RealTime type!"),
        FstVarType::GenericString => VarType::String,
        FstVarType::Bit => VarType::Bit,
        FstVarType::Logic => VarType::Logic,
        FstVarType::Int => VarType::Int,
        FstVarType::ShortInt => VarType::Int,
        FstVarType::LongInt => VarType::Int,
        FstVarType::Byte => VarType::Int,
        FstVarType::Enum => VarType::Enum,
        FstVarType::ShortReal => VarType::Real,
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
    let mut h = HierarchyBuilder::new(FileType::Fst);
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
                // the fst name often contains the variable name + the index
                let (var_name, index) = if let Some((prefix, suffix)) = name.split_once(' ') {
                    (prefix.to_string(), parse_index(suffix.as_bytes()))
                } else {
                    (name, None)
                };

                h.add_var(
                    var_name,
                    convert_var_tpe(tpe),
                    convert_var_direction(direction),
                    length,
                    index,
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
