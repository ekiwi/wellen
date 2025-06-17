// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::hierarchy::*;
use crate::signals::{
    FixedWidthEncoding, Signal, SignalSource, SignalSourceImplementation, TimeTableIdx,
};
use crate::vcd::parse_name;
use crate::wavemem::{check_if_changed_and_truncate, check_states, write_n_state, States};
use crate::{FileFormat, LoadOptions, TimeTable, WellenError};
use fst_reader::*;
use rustc_hash::FxHashMap;
use std::io::{BufRead, Seek};

pub type Result<T> = std::result::Result<T, WellenError>;

pub fn read_header<R: BufRead + Seek>(
    input: R,
    _options: &LoadOptions,
) -> Result<(Hierarchy, ReadBodyContinuation<R>)> {
    let mut reader = FstReader::open_and_read_time_table(input)?;
    let hierarchy = read_hierarchy(&mut reader)?;
    let cont = ReadBodyContinuation(reader);
    Ok((hierarchy, cont))
}
pub fn read_header_from_file<P: AsRef<std::path::Path>>(
    filename: P,
    _options: &LoadOptions,
) -> Result<(
    Hierarchy,
    ReadBodyContinuation<std::io::BufReader<std::fs::File>>,
)> {
    let input = std::io::BufReader::new(std::fs::File::open(filename.as_ref())?);
    let mut reader = match FstReader::open_and_read_time_table(input) {
        Ok(header) => header,
        Err(ReaderError::MissingGeometry() | ReaderError::MissingHierarchy()) => {
            // Geometric block or hierarchy block missing.
            // This generally indicates that the FST writing process was interrupted.
            // Thus, we try to load an external hierarchy file.
            let input = std::io::BufReader::new(std::fs::File::open(filename.as_ref())?);
            let mut hierarchy_filename = filename.as_ref().to_path_buf();
            hierarchy_filename.set_extension("fst.hier");
            let hierarchy = std::io::BufReader::new(std::fs::File::open(hierarchy_filename)?);
            FstReader::open_incomplete_and_read_time_table(input, hierarchy)?
        }
        Err(e) => return Err(e.into()),
    };
    let hierarchy = read_hierarchy(&mut reader)?;
    let cont = ReadBodyContinuation(reader);
    Ok((hierarchy, cont))
}
pub fn read_body<R: BufRead + Seek + Sync + Send + 'static>(
    data: ReadBodyContinuation<R>,
) -> Result<(SignalSource, TimeTable)> {
    let time_table = data.0.get_time_table().unwrap().to_vec();
    let reader = data.0;
    let db = FstWaveDatabase::new(reader);
    let boxed_db = Box::new(db);
    let source = SignalSource::new(boxed_db);
    Ok((source, time_table))
}

pub struct ReadBodyContinuation<R: BufRead + Seek>(FstReader<R>);

struct FstWaveDatabase<R: BufRead + Seek> {
    reader: FstReader<R>,
}

impl<R: BufRead + Seek> FstWaveDatabase<R> {
    fn new(reader: FstReader<R>) -> Self {
        FstWaveDatabase { reader }
    }
}

impl<R: BufRead + Seek + Sync + Send> SignalSourceImplementation for FstWaveDatabase<R> {
    fn load_signals(
        &mut self,
        ids: &[SignalRef],
        types: &[SignalEncoding],
        _multi_threaded: bool,
    ) -> Vec<Signal> {
        // create a FST filter
        let fst_ids = ids
            .iter()
            .zip(types.iter())
            .map(|(ii, _)| FstSignalHandle::from_index(ii.index()))
            .collect::<Vec<_>>();
        let filter = FstFilter::filter_signals(fst_ids);

        // lookup data structure for time table indices
        let tt = self.reader.get_time_table().unwrap().to_vec();
        let mut time_table = tt.iter().enumerate();
        let mut index_and_time = time_table.next().unwrap();

        // store signals
        let mut signals = ids
            .iter()
            .zip(types.iter())
            .map(|(id, tpe)| SignalWriter::new(*id, *tpe))
            .collect::<Vec<_>>();
        let idx_to_pos: FxHashMap<usize, usize> = FxHashMap::from_iter(
            ids.iter()
                .zip(types.iter())
                .map(|(r, _)| r.index())
                .enumerate()
                .map(|(pos, idx)| (idx, pos)),
        );
        let callback = |time: u64, handle: FstSignalHandle, value: FstSignalValue| {
            // determine time index
            while *(index_and_time.1) < time {
                index_and_time = time_table.next().unwrap();
            }
            let time_idx = index_and_time.0 as TimeTableIdx;
            debug_assert_eq!(*index_and_time.1, time);
            let signal_pos = idx_to_pos[&handle.get_index()];
            signals[signal_pos].add_change(time_idx, handle, value);
        };

        self.reader.read_signals(&filter, callback).unwrap();
        signals.into_iter().map(|w| w.finish()).collect()
    }
    fn print_statistics(&self) {
        println!("FST backend currently has not statistics to print.");
    }
}

struct SignalWriter {
    tpe: SignalEncoding,
    id: SignalRef,
    /// used to check that everything is going well
    handle: FstSignalHandle,
    data_bytes: Vec<u8>,
    strings: Vec<String>,
    time_indices: Vec<TimeTableIdx>,
    max_states: States,
}

impl SignalWriter {
    fn new(id: SignalRef, tpe: SignalEncoding) -> Self {
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
        match value {
            FstSignalValue::String(value) => match self.tpe {
                SignalEncoding::String => {
                    let str_value = String::from_utf8_lossy(value).to_string();
                    // check to see if the value actually changed
                    let changed = self
                        .strings
                        .last()
                        .map(|prev| prev != &str_value)
                        .unwrap_or(true);
                    if changed {
                        self.strings.push(str_value);
                        self.time_indices.push(time_idx);
                    }
                }
                SignalEncoding::BitVector(len) => {
                    let bits = len.get();

                    debug_assert_eq!(
                        value.len(),
                        bits as usize,
                        "{}",
                        String::from_utf8_lossy(value)
                    );
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
                        let num_prev_entries = self.time_indices.len();
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

                    let bytes_per_entry = get_bytes_per_entry(len, has_meta);
                    if check_if_changed_and_truncate(bytes_per_entry, &mut self.data_bytes) {
                        self.time_indices.push(time_idx);
                    }
                }
                SignalEncoding::Real => panic!(
                    "Expecting reals, but go: {}",
                    String::from_utf8_lossy(value)
                ),
            },
            FstSignalValue::Real(value) => {
                debug_assert_eq!(self.tpe, SignalEncoding::Real);
                self.data_bytes.extend_from_slice(&value.to_le_bytes());
                if check_if_changed_and_truncate(8, &mut self.data_bytes) {
                    self.time_indices.push(time_idx);
                }
            }
        }
    }

    fn finish(self) -> Signal {
        match self.tpe {
            SignalEncoding::String => {
                debug_assert!(self.data_bytes.is_empty());
                Signal::new_var_len(self.id, self.time_indices, self.strings)
            }
            SignalEncoding::Real => {
                debug_assert!(self.strings.is_empty());
                Signal::new_fixed_len(
                    self.id,
                    self.time_indices,
                    FixedWidthEncoding::Real,
                    8,
                    self.data_bytes,
                )
            }
            SignalEncoding::BitVector(len) => {
                debug_assert!(self.strings.is_empty());
                let (bytes, meta_byte) = get_len_and_meta(self.max_states, len.get());
                let encoding = FixedWidthEncoding::BitVector {
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
pub fn get_len_and_meta(states: States, bits: u32) -> (usize, bool) {
    let len = (bits as usize).div_ceil(states.bits_in_a_byte());
    let has_meta = (states != States::Two) && ((bits as usize) % states.bits_in_a_byte() == 0);
    (len, has_meta)
}

#[inline]
pub fn get_bytes_per_entry(len: usize, has_meta: bool) -> usize {
    if has_meta {
        len + 1
    } else {
        len
    }
}

const META_MASK: u8 = 3 << 6;

fn expand_entries(from: States, to: States, old: &[u8], entries: usize, bits: u32) -> Vec<u8> {
    let (from_len, from_meta) = get_len_and_meta(from, bits);
    let from_bytes_per_entry = get_bytes_per_entry(from_len, from_meta);
    let (to_len, to_meta) = get_len_and_meta(to, bits);

    if from_len == to_len && from_meta == to_meta {
        return Vec::from(old); // no change necessary
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

pub fn push_zeros(vec: &mut Vec<u8>, len: usize) {
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
        FstScopeType::Generate => ScopeType::Generate,
        FstScopeType::Struct => ScopeType::Struct,
        FstScopeType::Union => ScopeType::Union,
        FstScopeType::Class => ScopeType::Class,
        FstScopeType::Interface => ScopeType::Interface,
        FstScopeType::Package => ScopeType::Package,
        FstScopeType::Program => ScopeType::Program,
        FstScopeType::VhdlArchitecture => ScopeType::VhdlArchitecture,
        FstScopeType::VhdlProcedure => ScopeType::VhdlProcedure,
        FstScopeType::VhdlFunction => ScopeType::VhdlFunction,
        FstScopeType::VhdlRecord => ScopeType::VhdlRecord,
        FstScopeType::VhdlProcess => ScopeType::VhdlProcess,
        FstScopeType::VhdlBlock => ScopeType::VhdlBlock,
        FstScopeType::VhdlForGenerate => ScopeType::VhdlForGenerate,
        FstScopeType::VhdlIfGenerate => ScopeType::VhdlIfGenerate,
        FstScopeType::VhdlGenerate => ScopeType::VhdlGenerate,
        FstScopeType::VhdlPackage => ScopeType::VhdlPackage,
        FstScopeType::AttributeBegin
        | FstScopeType::AttributeEnd
        | FstScopeType::VcdScope
        | FstScopeType::VcdUpScope => unreachable!("unexpected scope type!"),
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
        FstVarType::SparseArray => VarType::SparseArray,
        FstVarType::RealTime => VarType::RealTime,
        FstVarType::GenericString => VarType::String,
        FstVarType::Bit => VarType::Bit,
        FstVarType::Logic => VarType::Logic,
        FstVarType::Int => VarType::Int,
        FstVarType::ShortInt => VarType::ShortInt,
        FstVarType::LongInt => VarType::LongInt,
        FstVarType::Byte => VarType::Byte,
        FstVarType::Enum => VarType::Enum,
        FstVarType::ShortReal => VarType::ShortReal,
    }
}

fn convert_var_direction(tpe: FstVarDirection) -> VarDirection {
    match tpe {
        FstVarDirection::Implicit => VarDirection::Implicit,
        FstVarDirection::Input => VarDirection::Input,
        FstVarDirection::Output => VarDirection::Output,
        FstVarDirection::InOut => VarDirection::InOut,
        FstVarDirection::Buffer => VarDirection::Buffer,
        FstVarDirection::Linkage => VarDirection::Linkage,
    }
}

/// GHDL does not seem to encode any actual information in the VHDL variable type.
/// Variables are always Signal or None.
pub fn deal_with_vhdl_var_type(tpe: FstVhdlVarType, var_name: &str) {
    if !matches!(tpe, FstVhdlVarType::None | FstVhdlVarType::Signal) {
        println!("INFO: detected a VHDL Var Type that is not Signal!: {tpe:?} for {var_name}");
    }
}

/// GHDL only uses a small combination of VCD variable and VHDL data types.
/// Here we merge them together into a single VarType.
pub fn merge_vhdl_data_and_var_type(vcd: VarType, vhdl: FstVhdlDataType) -> VarType {
    match vhdl {
        FstVhdlDataType::None => vcd,
        FstVhdlDataType::Boolean => VarType::Boolean,
        FstVhdlDataType::Bit => VarType::Bit,
        FstVhdlDataType::Vector => VarType::BitVector,
        FstVhdlDataType::ULogic => VarType::StdULogic,
        FstVhdlDataType::ULogicVector => VarType::StdULogicVector,
        FstVhdlDataType::Logic => VarType::StdLogic,
        FstVhdlDataType::LogicVector => VarType::StdLogicVector,
        FstVhdlDataType::Unsigned => {
            println!("TODO: handle {vcd:?} {vhdl:?} better!");
            vcd
        }
        FstVhdlDataType::Signed => {
            println!("TODO: handle {vcd:?} {vhdl:?} better!");
            vcd
        }
        FstVhdlDataType::Integer => VarType::Integer,
        FstVhdlDataType::Real => VarType::Real,
        FstVhdlDataType::Natural => {
            println!("TODO: handle {vcd:?} {vhdl:?} better!");
            vcd
        }
        FstVhdlDataType::Positive => {
            println!("TODO: handle {vcd:?} {vhdl:?} better!");
            vcd
        }
        FstVhdlDataType::Time => VarType::Time,
        FstVhdlDataType::Character => {
            println!("TODO: handle {vcd:?} {vhdl:?} better!");
            vcd
        }
        FstVhdlDataType::String => VarType::String,
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

#[derive(Debug)]
/// Represents an attribute which can from a FST or a VCD with extensions as generated by GTKWave or nvc.
pub enum Attribute {
    /// nvc: `misc 03 /home/oscar/test.vhdl 1`
    SourceLoc(HierarchyStringId, u64, bool),
    /// nvc: `misc 02 STD_LOGIC_VECTOR 1031`
    VhdlTypeInfo(String, FstVhdlVarType, FstVhdlDataType),
    Enum(EnumTypeId),
}

pub fn parse_var_attributes(
    attributes: &mut Vec<Attribute>,
    mut var_type: VarType,
    var_name: &str,
) -> crate::vcd::Result<(Option<String>, VarType, Option<EnumTypeId>)> {
    let mut type_name = None;
    let mut enum_type = None;
    while let Some(attr) = attributes.pop() {
        match attr {
            Attribute::SourceLoc(_, _, _) => {
                debug_assert!(false, "Unexpected attribute on a variable!");
            }
            Attribute::VhdlTypeInfo(name, vhdl_var_type, vhdl_data_type) => {
                type_name = Some(name);
                // For now we ignore the var type since GHDL seems to just always set it to Signal.
                // Their code does not use any other var type.
                deal_with_vhdl_var_type(vhdl_var_type, var_name);

                // We merge the info of the VCD var type and the vhdl data type
                var_type = merge_vhdl_data_and_var_type(var_type, vhdl_data_type);
            }
            Attribute::Enum(type_id) => enum_type = Some(type_id),
        }
    }
    Ok((type_name, var_type, enum_type))
}

pub fn parse_scope_attributes(
    attributes: &mut Vec<Attribute>,
    h: &mut HierarchyBuilder,
) -> crate::vcd::Result<(Option<SourceLocId>, Option<SourceLocId>)> {
    let mut declaration_source = None;
    let mut instance_source = None;
    while let Some(attr) = attributes.pop() {
        match attr {
            Attribute::SourceLoc(path, line, is_instantiation) => {
                if is_instantiation {
                    instance_source = Some(h.add_source_loc(path, line, true));
                } else {
                    declaration_source = Some(h.add_source_loc(path, line, false));
                }
            }
            Attribute::VhdlTypeInfo(_, _, _) | Attribute::Enum(_) => {
                debug_assert!(false, "Unexpected attribute on a scope!");
            }
        }
    }
    Ok((declaration_source, instance_source))
}

fn read_hierarchy<F: BufRead + Seek>(reader: &mut FstReader<F>) -> Result<Hierarchy> {
    let mut h = HierarchyBuilder::new(FileFormat::Fst);
    // load meta-data
    let fst_header = reader.get_header();
    h.set_version(fst_header.version.trim().to_string());
    h.set_date(fst_header.date.trim().to_string());
    h.set_timescale(convert_timescale(fst_header.timescale_exponent));

    let mut path_names = FxHashMap::default();
    let mut enums = FxHashMap::default();
    let mut attributes = Vec::new();

    let cb = |entry: FstHierarchyEntry| {
        match entry {
            FstHierarchyEntry::Scope {
                tpe,
                name,
                component,
            } => {
                let (declaration_source, instance_source) =
                    parse_scope_attributes(&mut attributes, &mut h).unwrap();
                let name_id = h.add_string(name);
                let component_id = h.add_string(component);
                h.add_scope(
                    name_id,
                    Some(component_id),
                    convert_scope_tpe(tpe),
                    declaration_source,
                    instance_source,
                    false,
                );
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
                let (var_name, index, scopes) = parse_name(name.as_bytes(), length).unwrap();
                let (type_name, var_type, enum_type) =
                    parse_var_attributes(&mut attributes, convert_var_tpe(tpe), &var_name).unwrap();
                let name_id = h.add_string(var_name);
                let type_name = type_name.map(|s| h.add_string(s));
                let num_scopes = scopes.len();
                // we derive the signal type from the fst tpe directly, the VHDL type should never factor in!
                let signal_tpe = match tpe {
                    FstVarType::GenericString => SignalEncoding::String,
                    FstVarType::Real
                    | FstVarType::RealTime
                    | FstVarType::RealParameter
                    | FstVarType::ShortReal => SignalEncoding::Real,
                    _ => SignalEncoding::bit_vec_of_len(length),
                };
                h.add_array_scopes(scopes);
                h.add_var(
                    name_id,
                    var_type,
                    signal_tpe,
                    convert_var_direction(direction),
                    index,
                    SignalRef::from_index(handle.get_index()).unwrap(),
                    enum_type,
                    type_name,
                );
                h.pop_scopes(num_scopes);
            }
            FstHierarchyEntry::PathName { id, name } => {
                let string_ref = h.add_string(name);
                path_names.insert(id, string_ref);
            }
            FstHierarchyEntry::SourceStem {
                is_instantiation,
                path_id,
                line,
            } => {
                let path = path_names[&path_id];
                attributes.push(Attribute::SourceLoc(path, line, is_instantiation));
            }
            FstHierarchyEntry::Comment { .. } => {} // ignored
            FstHierarchyEntry::EnumTable {
                name,
                handle,
                mapping,
            } => {
                let mapping = mapping
                    .into_iter()
                    .map(|(a, b)| (h.add_string(a), h.add_string(b)))
                    .collect::<Vec<_>>();
                let name = h.add_string(name);
                let enum_ref = h.add_enum_type(name, mapping);
                // remember enum table by handle
                enums.insert(handle, enum_ref);
            }
            FstHierarchyEntry::EnumTableRef { handle } => {
                attributes.push(Attribute::Enum(enums[&handle]));
            }
            FstHierarchyEntry::VhdlVarInfo {
                type_name,
                var_type,
                data_type,
            } => {
                attributes.push(Attribute::VhdlTypeInfo(type_name, var_type, data_type));
            }
            FstHierarchyEntry::AttributeEnd => {
                // ignore
                // So far the only simulator we know that uses this attribute is
                // `nvc` which calls `fstWriterSetAttrEnd` at the end of declaring an array.
                // This does not provide us with any additional information though since we
                // deduce array entries from the variable names.
            }
        };
    };
    reader.read_hierarchy(cb)?;
    Ok(h.finish())
}
