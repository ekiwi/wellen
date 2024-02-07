// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::signals::{Signal, SignalEncoding, SignalSource, Time, TimeTableIdx};
use crate::vcd::parse_index;
use crate::wavemem::{check_states, write_n_state, States};
use crate::{Waveform, WellenError};
use fst_native::*;
use std::collections::HashMap;
use std::io::{BufRead, Seek};

pub type Result<T> = std::result::Result<T, WellenError>;

pub fn read(filename: &str) -> Result<Waveform> {
    let input = std::fs::File::open(filename).expect("failed to open input file!");
    let mut reader = FstReader::open_and_read_time_table(std::io::BufReader::new(input)).unwrap();
    let hierarchy = read_hierarchy(&mut reader);
    let db = Box::new(FstWaveDatabase::new(reader));
    Ok(Waveform::new(hierarchy, db))
}

pub fn read_from_bytes(bytes: Vec<u8>) -> Result<Waveform> {
    let mut reader = FstReader::open_and_read_time_table(std::io::Cursor::new(bytes)).unwrap();
    let hierarchy = read_hierarchy(&mut reader);
    let db = Box::new(FstWaveDatabase::new(reader));
    Ok(Waveform::new(hierarchy, db))
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
        let idx_to_pos: HashMap<usize, usize> = HashMap::from_iter(
            ids.iter()
                .map(|(r, _)| r.index())
                .enumerate()
                .map(|(pos, idx)| (idx, pos)),
        );
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
    max_states: States,
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
            max_states: States::default(),
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
                    let bits = len.get();
                    debug_assert_eq!(value.len(), bits as usize);
                    let local_encoding = check_states(value).unwrap_or_else(|| {
                        panic!(
                            "Unexpected signal value: {}",
                            String::from_utf8_lossy(value)
                        )
                    });

                    let signal_states = States::join(self.max_states, local_encoding);
                    if signal_states != self.max_states {
                        // With FST we do not know how many states the signal needs, thus we first assume
                        // the minimal number of states. If that turns out to be wrong, we go back and
                        // expand the existing data.
                        let num_prev_entries = self.time_indices.len() - 1; // -1 to account for the new index
                        self.data_bytes = expand_entries(
                            self.max_states,
                            signal_states,
                            &self.data_bytes,
                            num_prev_entries,
                            bits,
                        );
                        self.max_states = signal_states;
                    }

                    let (len, has_meta) = get_len_and_meta(signal_states, bits);
                    let meta_data = (local_encoding as u8) << 6;
                    let (local_len, local_has_meta) = get_len_and_meta(local_encoding, bits);

                    if local_len == len && local_has_meta == has_meta {
                        // same meta-data location and length as the maximum
                        if has_meta {
                            self.data_bytes.push(meta_data);
                            write_n_state(local_encoding, value, &mut self.data_bytes, None);
                        } else {
                            write_n_state(
                                local_encoding,
                                value,
                                &mut self.data_bytes,
                                Some(meta_data),
                            );
                        }
                    } else {
                        // smaller encoding than the maximum
                        self.data_bytes.push(meta_data);
                        let (local_len, _) = get_len_and_meta(local_encoding, bits);
                        if has_meta {
                            push_zeros(&mut self.data_bytes, len - local_len);
                        } else {
                            push_zeros(&mut self.data_bytes, len - local_len - 1);
                        }
                        write_n_state(local_encoding, value, &mut self.data_bytes, None);
                    }
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
                let (bytes, meta_byte) = get_len_and_meta(self.max_states, len.get());
                let encoding = SignalEncoding::BitVector {
                    max_states: self.max_states,
                    bits: len.get(),
                    meta_byte,
                };
                Signal::new_fixed_len(
                    self.id,
                    self.time_indices,
                    encoding,
                    get_bytes_per_entry(bytes, meta_byte) as u32,
                    self.data_bytes,
                )
            }
        }
    }
}

#[inline]
pub(crate) fn get_len_and_meta(states: States, bits: u32) -> (usize, bool) {
    let len = (bits as usize).div_ceil(states.bits_in_a_byte());
    let has_meta = (states != States::Two) && ((bits as usize) % states.bits_in_a_byte() == 0);
    (len, has_meta)
}

#[inline]
pub(crate) fn get_bytes_per_entry(len: usize, has_meta: bool) -> usize {
    if has_meta {
        len + 1
    } else {
        len
    }
}

const META_MASK: u8 = 3 << 6;

fn expand_entries(from: States, to: States, old: &Vec<u8>, entries: usize, bits: u32) -> Vec<u8> {
    let (from_len, from_meta) = get_len_and_meta(from, bits);
    let from_bytes_per_entry = get_bytes_per_entry(from_len, from_meta);
    let (to_len, to_meta) = get_len_and_meta(to, bits);

    if from_len == to_len && from_meta == to_meta {
        return old.clone(); // no change necessary
    }

    let to_bytes_per_entry = get_bytes_per_entry(to_len, to_meta);
    debug_assert!(
        !from_meta || to_meta,
        "meta-bytes are only added, never removed when expanding!"
    );
    let padding_len = if !to_meta {
        to_len - from_len - 1 // subtract one to account for meta data
    } else {
        to_len - from_len
    };

    let mut data = Vec::with_capacity(entries * to_bytes_per_entry);
    for value in old.chunks(from_bytes_per_entry) {
        // meta handling
        let meta_data = if from == States::Two {
            (States::Two as u8) << 6
        } else {
            value[0] & META_MASK
        };
        // we can always push the meta byte, just need to adjust the padding
        data.push(meta_data);
        push_zeros(&mut data, padding_len);
        // copy over the actual values
        if from_meta {
            data.push(value[0] & !META_MASK);
            data.extend_from_slice(&value[1..]);
        } else {
            data.extend_from_slice(value);
        }
    }
    data
}

pub(crate) fn push_zeros(vec: &mut Vec<u8>, len: usize) {
    for _ in 0..len {
        vec.push(0);
    }
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
    let mut enums = HashMap::new();
    let mut next_var_has_enum = None;
    // let mut next_var_has_source_info = None;

    let cb = |entry: FstHierarchyEntry| {
        match entry {
            FstHierarchyEntry::Scope { tpe, name, .. } => {
                h.add_scope(name, convert_scope_tpe(tpe), false);
                println!("SCOPE");
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

                if let Some(handle) = next_var_has_enum {
                    next_var_has_enum = None;
                    let (name, mapping) = &enums[&handle];
                    println!("TODO: {var_name} is of enum type {name}: {mapping:?}!");
                }

                println!("VAR");

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
            FstHierarchyEntry::SourceStem {
                is_instantiation,
                path_id,
                line,
            } => {
                let path = &path_names[&path_id];
                println!("TODO: Deal with source info: {path}:{line} {is_instantiation}");
            }
            FstHierarchyEntry::Comment { .. } => {} // ignored
            FstHierarchyEntry::EnumTable {
                name,
                handle,
                mapping,
            } => {
                // remember enum table by handle
                enums.insert(handle, (name, mapping));
            }
            FstHierarchyEntry::EnumTableRef { handle } => {
                next_var_has_enum = Some(handle);
            }
            FstHierarchyEntry::AttributeEnd => todo!("{entry:?}"),
        };
    };
    reader.read_hierarchy(cb).unwrap();
    h.finish()
}
