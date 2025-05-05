// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::fst::{parse_scope_attributes, parse_var_attributes, Attribute};
use crate::hierarchy::*;
use crate::signals::SignalSource;
use crate::viewers::ProgressCount;
use crate::wavemem::Encoder;
use crate::{FileFormat, LoadOptions, TimeTable};
use fst_reader::{FstVhdlDataType, FstVhdlVarType};
use num_enum::TryFromPrimitive;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fmt::Debug;
use std::io::{BufRead, Read, Seek, SeekFrom};
use std::sync::atomic::Ordering;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VcdParseError {
    #[error("[vcd] failed to parse length: `{0}` for variable `{1}`")]
    VcdVarLengthParsing(String, String),
    #[error("[vcd] failed to parse variable name: `{0}`")]
    VcdVarNameParsing(String),
    #[error("[vcd] expected command to start with `$`, not `{0}`")]
    VcdStartChar(String),
    #[error("[vcd] unknown or invalid command: `{0}`, valid are: {list:?}", list=get_vcd_command_str())]
    VcdInvalidCommand(String),
    #[error("[vcd] unexpected number of tokens for command {0}: {1}")]
    VcdUnexpectedNumberOfTokens(String, String),
    #[error("[vcd] encountered an attribute with an unsupported type: {0}")]
    VcdUnsupportedAttributeType(String),
    #[error("[vcd] failed to parse VHDL var type from attribute.")]
    VcdFailedToParseVhdlVarType(
        #[from] num_enum::TryFromPrimitiveError<fst_reader::FstVhdlVarType>,
    ),
    #[error("[vcd] failed to parse VHDL data type from attribute.")]
    VcdFailedToParseVhdlDataType(
        #[from] num_enum::TryFromPrimitiveError<fst_reader::FstVhdlDataType>,
    ),
    #[error("[vcd] unknown var type: {0}")]
    VcdUnknownVarType(String),
    #[error("[vcd] unknown scope type: {0}")]
    VcdUnknownScopeType(String),
    #[error("[vcd] unexpected token in VCD body: {0}")]
    VcdUnexpectedBodyToken(String),
    #[error("[vcd] expected an id for a value change, but did not find one")]
    VcdEmptyId,
    /// This is not really an error, but our parser has to terminate and start a new attempt
    /// at interpreting ids. This error should never reach any user.
    #[error("[vcd] non-contiguous ids detected, applying a work around.")]
    VcdNonContiguousIds,
    #[error("failed to decode string")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("failed to parse an integer")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, VcdParseError>;

pub fn read_header_from_file<P: AsRef<std::path::Path>>(
    filename: P,
    options: &LoadOptions,
) -> Result<(
    Hierarchy,
    ReadBodyContinuation<std::io::BufReader<std::fs::File>>,
    u64,
)> {
    let input_file = std::fs::File::open(filename)?;
    let mmap = unsafe { memmap2::Mmap::map(&input_file)? };
    let (header_len, hierarchy, lookup) =
        read_hierarchy(&mut std::io::Cursor::new(&mmap[..]), options)?;
    let body_len = (mmap.len() - header_len) as u64;
    let cont = ReadBodyContinuation {
        multi_thread: options.multi_thread,
        header_len,
        lookup,
        input: Input::Mmap(mmap),
    };
    Ok((hierarchy, cont, body_len))
}

pub fn read_header<R: BufRead + Seek>(
    mut input: R,
    options: &LoadOptions,
) -> Result<(Hierarchy, ReadBodyContinuation<R>, u64)> {
    // determine the length of the input
    let start = input.stream_position()?;
    input.seek(SeekFrom::End(0))?;
    let end = input.stream_position()?;
    input.seek(SeekFrom::Start(start))?;
    let input_len = end - start;

    // actually read the header
    let (header_len, hierarchy, lookup) = read_hierarchy(&mut input, options)?;
    let body_len = input_len - header_len as u64;
    let cont = ReadBodyContinuation {
        multi_thread: options.multi_thread,
        header_len,
        lookup,
        input: Input::Reader(input),
    };
    Ok((hierarchy, cont, body_len))
}

pub struct ReadBodyContinuation<R: BufRead + Seek> {
    multi_thread: bool,
    header_len: usize,
    lookup: IdLookup,
    input: Input<R>,
}

enum Input<R: BufRead + Seek> {
    Reader(R),
    Mmap(memmap2::Mmap),
}

pub fn read_body<R: BufRead + Seek>(
    data: ReadBodyContinuation<R>,
    hierarchy: &Hierarchy,
    progress: Option<ProgressCount>,
) -> Result<(SignalSource, TimeTable)> {
    let (source, time_table) = match data.input {
        Input::Reader(mut input) => {
            // determine binput length
            let start = input.stream_position()?;
            input.seek(SeekFrom::End(0))?;
            let end = input.stream_position()?;
            input.seek(SeekFrom::Start(start))?;

            // encode signals
            let encoder = read_single_stream_of_values(
                &mut input,
                end as usize, // end_pos includes the size of the header
                true,
                hierarchy,
                &data.lookup,
                progress,
            )?;
            encoder.finish()
        }
        Input::Mmap(mmap) => read_values(
            &mmap[data.header_len..],
            data.multi_thread,
            hierarchy,
            &data.lookup,
            progress,
        )?,
    };
    Ok((source, time_table))
}

const FST_SUP_VAR_DATA_TYPE_BITS: u32 = 10;
const FST_SUP_VAR_DATA_TYPE_MASK: u64 = (1 << FST_SUP_VAR_DATA_TYPE_BITS) - 1;

// VCD attributes are a GTKWave extension which is also used by nvc
fn parse_attribute(
    tokens: Vec<&[u8]>,
    path_names: &mut FxHashMap<u64, HierarchyStringId>,
    h: &mut HierarchyBuilder,
) -> Result<Option<Attribute>> {
    match tokens[1] {
        b"02" => {
            // FstHierarchyEntry::VhdlVarInfo
            if tokens.len() != 4 {
                return Err(unexpected_n_tokens("attribute", &tokens));
            }
            let type_name = std::str::from_utf8(tokens[2])?.to_string();
            let arg = std::str::from_utf8(tokens[3])?.parse::<u64>()?;
            let var_type =
                FstVhdlVarType::try_from_primitive((arg >> FST_SUP_VAR_DATA_TYPE_BITS) as u8)?;
            let data_type =
                FstVhdlDataType::try_from_primitive((arg & FST_SUP_VAR_DATA_TYPE_MASK) as u8)?;
            Ok(Some(Attribute::VhdlTypeInfo(
                type_name, var_type, data_type,
            )))
        }
        b"03" => {
            // FstHierarchyEntry::PathName
            if tokens.len() != 4 {
                return Err(unexpected_n_tokens("attribute", &tokens));
            }
            let path = std::str::from_utf8(tokens[2])?.to_string();
            let id = std::str::from_utf8(tokens[3])?.parse::<u64>()?;
            let string_ref = h.add_string(path);
            path_names.insert(id, string_ref);
            Ok(None)
        }
        b"04" => {
            // FstHierarchyEntry::SourceStem
            if tokens.len() != 4 {
                // TODO: GTKWave might actually generate 5 tokens in order to include whether it is the
                //       instance of the normal source path
                return Err(unexpected_n_tokens("attribute", &tokens));
            }
            let path_id = std::str::from_utf8(tokens[2])?.parse::<u64>()?;
            let line = std::str::from_utf8(tokens[3])?.parse::<u64>()?;
            let is_instance = false;
            Ok(Some(Attribute::SourceLoc(
                path_names[&path_id],
                line,
                is_instance,
            )))
        }
        _ => Err(VcdParseError::VcdUnsupportedAttributeType(
            iter_bytes_to_list_str(tokens.iter()),
        )),
    }
}

type IdLookup = Option<FxHashMap<Vec<u8>, SignalRef>>;

fn read_hierarchy(
    input: &mut (impl BufRead + Seek),
    options: &LoadOptions,
) -> Result<(usize, Hierarchy, IdLookup)> {
    // first we try to avoid using an id map
    let input_start = input.stream_position()?;
    match read_hierarchy_inner(input, false, options) {
        Ok(res) => Ok(res),
        Err(VcdParseError::VcdNonContiguousIds) => {
            // second try, this time with an id map
            input.seek(SeekFrom::Start(input_start))?;
            read_hierarchy_inner(input, true, options)
        }
        // non recoverable error
        Err(other) => Err(other),
    }
}

/// Collects statistics on VCD IDs used in order to decide whether we should be
/// using a hash map or a direct translation to indices.
#[derive(Debug, Clone, Default)]
struct IdTracker {
    var_count: u64,
    min_max_id: Option<(u64, u64)>,
    not_monotonic_inc: bool,
}

impl IdTracker {
    fn need_id_map(&mut self, id_value: u64) -> bool {
        // update statistics
        self.var_count += 1;
        if !self.not_monotonic_inc {
            // check to see if new increase is monotonic
            let is_monotonic = self
                .min_max_id
                .map(|(_, old_max)| old_max < id_value)
                .unwrap_or(true);
            self.not_monotonic_inc = !is_monotonic;
        }
        let (min_id, max_id) = match self.min_max_id {
            Some((min_id, max_id)) => (
                std::cmp::min(min_id, id_value),
                std::cmp::max(max_id, id_value),
            ),
            None => (id_value, id_value),
        };
        debug_assert!(min_id <= max_id);
        self.min_max_id = Some((min_id, max_id));

        if (id_value / self.var_count) > 1024 * 1024 {
            // a very large id value means that our dense strategy won't work
            // we are using 1MBi of addressable bytes as a threshold here
            return true;
        }

        // if there are big gaps between ids, our dense strategy probably won't work
        let inv_density = (max_id - min_id) / self.var_count;
        if inv_density > 1000 {
            // 1000 means only 0.1% of IDs are used, even if we employ and offset
            return true;
        }

        false
    }
}

fn read_hierarchy_inner(
    input: &mut (impl BufRead + Seek),
    use_id_map: bool,
    options: &LoadOptions,
) -> Result<(usize, Hierarchy, IdLookup)> {
    let start = input.stream_position().unwrap();
    let mut h = HierarchyBuilder::new(FileFormat::Vcd);
    let mut attributes = Vec::new();
    let mut path_names = FxHashMap::default();
    // this map is used to translate identifiers to signal references for cases where we detect ids that are too large
    let mut id_map: FxHashMap<Vec<u8>, SignalRef> = FxHashMap::default();
    // statistics to decide whether to switch to an ID map
    let mut id_tracker = IdTracker::default();

    let mut id_to_signal_ref = |id: &[u8]| -> Result<SignalRef> {
        // check to see if we should be using an id map
        if !use_id_map {
            if let Some(id_value) = id_to_int(id) {
                if id_tracker.need_id_map(id_value) {
                    return Err(VcdParseError::VcdNonContiguousIds); // restart with id map
                }
            } else {
                return Err(VcdParseError::VcdNonContiguousIds); // restart with id map
            }
        }

        // do the actual lookup / conversion
        if use_id_map {
            match id_map.get(id) {
                Some(signal_ref) => Ok(*signal_ref),
                None => {
                    let signal_ref = SignalRef::from_index(id_map.len() + 1).unwrap();
                    id_map.insert(id.to_vec(), signal_ref);
                    Ok(signal_ref)
                }
            }
        } else {
            Ok(SignalRef::from_index(id_to_int(id).unwrap() as usize).unwrap())
        }
    };

    let callback = |cmd: HeaderCmd| match cmd {
        HeaderCmd::Scope(tpe, name) => {
            let flatten = options.remove_scopes_with_empty_name && name.is_empty();
            let (declaration_source, instance_source) =
                parse_scope_attributes(&mut attributes, &mut h)?;
            let name = h.add_string(std::str::from_utf8(name)?.to_string());
            h.add_scope(
                name,
                None, // VCDs do not contain component names
                convert_scope_tpe(tpe)?,
                declaration_source,
                instance_source,
                flatten,
            );
            Ok(())
        }
        HeaderCmd::UpScope => {
            h.pop_scope();
            Ok(())
        }
        HeaderCmd::Var(tpe, size, id, name) => {
            let length = match std::str::from_utf8(size).unwrap().parse::<u32>() {
                Ok(len) => len,
                Err(_) => {
                    return Err(VcdParseError::VcdVarLengthParsing(
                        String::from_utf8_lossy(size).to_string(),
                        String::from_utf8_lossy(name).to_string(),
                    ));
                }
            };
            let (var_name, index, scopes) = parse_name(name, length)?;
            let raw_vcd_var_tpe = convert_var_tpe(tpe)?;
            // we derive the signal type from the vcd var directly, the VHDL type should never factor in!
            let signal_tpe = match raw_vcd_var_tpe {
                VarType::String => SignalEncoding::String,
                VarType::Real | VarType::RealTime | VarType::ShortReal => SignalEncoding::Real,
                _ => SignalEncoding::bit_vec_of_len(length),
            };
            // combine the raw variable type with VHDL type attributes
            let (type_name, var_type, enum_type) =
                parse_var_attributes(&mut attributes, raw_vcd_var_tpe, &var_name)?;
            let name = h.add_string(var_name);
            let type_name = type_name.map(|s| h.add_string(s));
            let num_scopes = scopes.len();
            h.add_array_scopes(scopes);

            h.add_var(
                name,
                var_type,
                signal_tpe,
                VarDirection::vcd_default(),
                index,
                id_to_signal_ref(id)?,
                enum_type,
                type_name,
            );
            h.pop_scopes(num_scopes);
            Ok(())
        }
        HeaderCmd::Date(value) => {
            h.set_date(String::from_utf8_lossy(value).to_string());
            Ok(())
        }
        HeaderCmd::Version(value) => {
            h.set_version(String::from_utf8_lossy(value).to_string());
            Ok(())
        }
        HeaderCmd::Comment(value) => {
            h.add_comment(String::from_utf8_lossy(value).to_string());
            Ok(())
        }
        HeaderCmd::Timescale(factor, unit) => {
            let factor_int = std::str::from_utf8(factor)?.parse::<u32>()?;
            let value = Timescale::new(factor_int, convert_timescale_unit(unit));
            h.set_timescale(value);
            Ok(())
        }
        HeaderCmd::MiscAttribute(tokens) => {
            if let Some(attr) = parse_attribute(tokens, &mut path_names, &mut h)? {
                attributes.push(attr);
            }
            Ok(())
        }
    };

    read_vcd_header(input, callback)?;
    let end = input.stream_position().unwrap();
    let hierarchy = h.finish();
    let lookup = if use_id_map { Some(id_map) } else { None };
    Ok(((end - start) as usize, hierarchy, lookup))
}

/// Tries to extract an index expression from the end of `value`. Ignores spaces.
/// Returns the index and the remaining bytes of `value` before the parsed index.
fn extract_suffix_index(value: &[u8]) -> (&[u8], Option<VarIndex>) {
    use ExtractSuffixIndexState as St;
    let mut state = St::SearchingForClosingBracket;

    for (ii, cc) in value.iter().enumerate().rev() {
        // skip whitespace
        if *cc == b' ' {
            continue;
        }

        state = match state {
            St::SearchingForClosingBracket => {
                if *cc == b']' {
                    St::ParsingLsb(ii, 0, 1)
                } else {
                    // our value does not end in `]`
                    return (&value[0..ii + 1], None);
                }
            }
            St::ParsingLsb(end, num, factor) => {
                if cc.is_ascii_digit() {
                    let digit = (*cc - b'0') as i64;
                    St::ParsingLsb(end, num + digit * factor, factor * 10)
                } else if *cc == b'-' {
                    St::ParsingLsb(end, -num, factor)
                } else if *cc == b':' {
                    St::ParsingMsb(end, num, 0, 1)
                } else if *cc == b'[' {
                    St::LookingForName(VarIndex::new(num, num))
                } else {
                    // not a valid number, give up
                    return (&value[0..end + 1], None);
                }
            }
            St::ParsingMsb(end, lsb, num, factor) => {
                if cc.is_ascii_digit() {
                    let digit = (*cc - b'0') as i64;
                    St::ParsingMsb(end, lsb, num + digit * factor, factor * 10)
                } else if *cc == b'-' {
                    St::ParsingMsb(end, lsb, -num, factor)
                } else if *cc == b'[' {
                    St::LookingForName(VarIndex::new(num, lsb))
                } else {
                    // not a valid number, give up
                    return (&value[0..end + 1], None);
                }
            }
            St::LookingForName(index) => {
                // any non-space character means that we found the name
                return (&value[0..ii + 1], Some(index));
            }
        };
    }

    // wasn't able to parse any index
    (value, None)
}

#[derive(Debug, Copy, Clone)]
enum ExtractSuffixIndexState {
    SearchingForClosingBracket,
    ParsingLsb(usize, i64, i64),
    ParsingMsb(usize, i64, i64, i64),
    LookingForName(VarIndex),
}

/// Splits a full name into:
/// 1. the variable name
/// 2. the bit index
/// 3. any extra scopes generated by a multidimensional arrays
/// `length` is used in order to distinguish bit-indices and array scopes.
pub fn parse_name(raw_name: &[u8], length: u32) -> Result<(String, Option<VarIndex>, Vec<String>)> {
    if raw_name.is_empty() {
        return Ok(("".to_string(), None, vec![]));
    }
    debug_assert!(
        raw_name[0] != b'[',
        "we assume that the first character is not `[`!"
    );

    // find the bit index from the back
    let (name, index) = extract_suffix_index(raw_name);

    // Check to see if the index makes sense with the length that was reported.
    // This is important in order to distinguish bit indices from array indices.
    let (mut name, index) = if let Some(index) = index {
        // index length does not match declared variable length
        if index.length() != length {
            // => go back to old name and no index
            (raw_name, None)
        } else {
            (name, Some(index))
        }
    } else {
        (name, index)
    };

    // see if there are any other indices from multidimensional arrays
    let mut indices = vec![];
    while name.last().cloned() == Some(b']') {
        let index_start = match find_last(name, b'[') {
            Some(s) => s,
            None => {
                return Err(VcdParseError::VcdVarNameParsing(
                    String::from_utf8_lossy(name).to_string(),
                ))
            }
        };
        let index = &name[index_start..(name.len())];
        indices.push(String::from_utf8_lossy(index).to_string());
        name = trim_right(&name[..index_start]);
    }

    let name = String::from_utf8_lossy(name).to_string();

    if indices.is_empty() {
        Ok((name, index, indices))
    } else {
        // if there are indices, the name actually becomes part of the scope
        let mut scopes = Vec::with_capacity(indices.len());
        scopes.push(name);
        while indices.len() > 1 {
            scopes.push(indices.pop().unwrap());
        }
        let final_name = indices.pop().unwrap();
        Ok((final_name, index, scopes))
    }
}

#[inline]
fn trim_right(mut name: &[u8]) -> &[u8] {
    while name.last().cloned() == Some(b' ') {
        name = &name[..(name.len() - 1)];
    }
    name
}

#[inline]
fn find_last(haystack: &[u8], needle: u8) -> Option<usize> {
    let from_back = haystack.iter().rev().position(|b| *b == needle)?;
    Some(haystack.len() - from_back - 1)
}

fn convert_timescale_unit(name: &[u8]) -> TimescaleUnit {
    match name {
        b"fs" => TimescaleUnit::FemtoSeconds,
        b"ps" => TimescaleUnit::PicoSeconds,
        b"ns" => TimescaleUnit::NanoSeconds,
        b"us" => TimescaleUnit::MicroSeconds,
        b"ms" => TimescaleUnit::MilliSeconds,
        b"s" => TimescaleUnit::Seconds,
        _ => TimescaleUnit::Unknown,
    }
}

fn convert_scope_tpe(tpe: &[u8]) -> Result<ScopeType> {
    match tpe {
        b"module" => Ok(ScopeType::Module),
        b"task" => Ok(ScopeType::Task),
        b"function" => Ok(ScopeType::Function),
        b"begin" => Ok(ScopeType::Begin),
        b"fork" => Ok(ScopeType::Fork),
        b"generate" => Ok(ScopeType::Generate),
        b"struct" => Ok(ScopeType::Struct),
        b"union" => Ok(ScopeType::Union),
        b"class" => Ok(ScopeType::Class),
        b"interface" => Ok(ScopeType::Interface),
        b"package" => Ok(ScopeType::Package),
        b"program" => Ok(ScopeType::Program),
        b"vhdl_architecture" => Ok(ScopeType::VhdlArchitecture),
        b"vhdl_procedure" => Ok(ScopeType::VhdlProcedure),
        b"vhdl_function" => Ok(ScopeType::VhdlFunction),
        b"vhdl_record" => Ok(ScopeType::VhdlRecord),
        b"vhdl_process" => Ok(ScopeType::VhdlProcess),
        b"vhdl_block" => Ok(ScopeType::VhdlBlock),
        b"vhdl_for_generate" => Ok(ScopeType::VhdlForGenerate),
        b"vhdl_if_generate" => Ok(ScopeType::VhdlIfGenerate),
        b"vhdl_generate" => Ok(ScopeType::VhdlGenerate),
        b"vhdl_package" => Ok(ScopeType::VhdlPackage),
        // questa sim produces "unknown" scopes
        b"unknown" => Ok(ScopeType::Unknown),
        _ => Err(VcdParseError::VcdUnknownScopeType(
            String::from_utf8_lossy(tpe).to_string(),
        )),
    }
}

fn convert_var_tpe(tpe: &[u8]) -> Result<VarType> {
    match tpe {
        b"wire" => Ok(VarType::Wire),
        b"reg" => Ok(VarType::Reg),
        b"parameter" => Ok(VarType::Parameter),
        b"integer" => Ok(VarType::Integer),
        b"string" => Ok(VarType::String),
        b"event" => Ok(VarType::Event),
        b"real" => Ok(VarType::Real),
        b"real_parameter" => Ok(VarType::Parameter),
        b"supply0" => Ok(VarType::Supply0),
        b"supply1" => Ok(VarType::Supply1),
        b"time" => Ok(VarType::Time),
        b"tri" => Ok(VarType::Tri),
        b"triand" => Ok(VarType::TriAnd),
        b"trior" => Ok(VarType::TriOr),
        b"trireg" => Ok(VarType::TriReg),
        b"tri0" => Ok(VarType::Tri0),
        b"tri1" => Ok(VarType::Tri1),
        b"wand" => Ok(VarType::WAnd),
        b"wor" => Ok(VarType::WOr),
        b"logic" => Ok(VarType::Logic),
        b"port" => Ok(VarType::Port),
        b"sparray" => Ok(VarType::SparseArray),
        b"realtime" => Ok(VarType::RealTime),
        b"bit" => Ok(VarType::Bit),
        b"int" => Ok(VarType::Int),
        b"shortint" => Ok(VarType::ShortInt),
        b"longint" => Ok(VarType::LongInt),
        b"byte" => Ok(VarType::Byte),
        b"enum" => Ok(VarType::Enum),
        b"shortread" => Ok(VarType::ShortReal),
        _ => Err(VcdParseError::VcdUnknownVarType(
            String::from_utf8_lossy(tpe).to_string(),
        )),
    }
}

const ID_CHAR_MIN: u8 = b'!';
const ID_CHAR_MAX: u8 = b'~';
const NUM_ID_CHARS: u64 = (ID_CHAR_MAX - ID_CHAR_MIN + 1) as u64;

/// Copied from https://github.com/kevinmehall/rust-vcd, licensed under MIT
#[inline]
fn id_to_int(id: &[u8]) -> Option<u64> {
    if id.is_empty() {
        return None;
    }
    let mut result = 0u64;
    for &i in id.iter().rev() {
        if !(ID_CHAR_MIN..=ID_CHAR_MAX).contains(&i) {
            return None;
        }
        let c = ((i - ID_CHAR_MIN) as u64) + 1;
        result = result
            .checked_mul(NUM_ID_CHARS)
            .and_then(|x| x.checked_add(c))?;
    }
    Some(result - 1)
}

#[inline]
fn unexpected_n_tokens(cmd: &str, tokens: &[&[u8]]) -> VcdParseError {
    VcdParseError::VcdUnexpectedNumberOfTokens(
        cmd.to_string(),
        iter_bytes_to_list_str(tokens.iter()),
    )
}

fn read_vcd_header(
    input: &mut impl BufRead,
    mut callback: impl FnMut(HeaderCmd) -> Result<()>,
) -> Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    loop {
        buf.clear();
        let (cmd, body) = read_command(input, &mut buf)?;
        let parsed = match cmd {
            VcdCmd::Scope => {
                let tokens = find_tokens(body);
                let name = tokens.get(1).cloned().unwrap_or(&[] as &[u8]);
                HeaderCmd::Scope(tokens[0], name)
            }
            VcdCmd::Var => {
                let tokens = find_tokens(body);
                // the actual variable name could be represented by a variable number of tokens,
                // thus we combine all trailing tokens together
                if tokens.len() < 4 {
                    return Err(unexpected_n_tokens("variable", &tokens));
                }
                // concatenate all trailing tokens
                let body_start = body.as_ptr() as u64;
                let name_start = tokens[3].as_ptr() as u64 - body_start;
                let last_token = tokens.last().unwrap();
                let name_end = last_token.as_ptr() as u64 - body_start + last_token.len() as u64;
                let name = &body[name_start as usize..name_end as usize];
                HeaderCmd::Var(tokens[0], tokens[1], tokens[2], name)
            }
            VcdCmd::UpScope => HeaderCmd::UpScope,
            VcdCmd::Date => HeaderCmd::Date(body),
            VcdCmd::Comment => HeaderCmd::Comment(body),
            VcdCmd::Version => HeaderCmd::Version(body),
            VcdCmd::Timescale => {
                let tokens = find_tokens(body);
                let (factor, unit) = match tokens.len() {
                    1 => {
                        // find the first non-numeric character
                        let token = tokens[0];
                        match token.iter().position(|c| *c < b'0' || *c > b'9') {
                            None => (token, &[] as &[u8]),
                            Some(pos) => (&token[..pos], &token[pos..]),
                        }
                    }
                    2 => (tokens[0], tokens[1]),
                    _ => {
                        return Err(VcdParseError::VcdUnexpectedNumberOfTokens(
                            "timescale".to_string(),
                            iter_bytes_to_list_str(tokens.iter()),
                        ))
                    }
                };
                HeaderCmd::Timescale(factor, unit)
            }
            VcdCmd::EndDefinitions => {
                // header is done
                return Ok(());
            }
            VcdCmd::Attribute => {
                let tokens = find_tokens(body);
                if tokens.len() < 3 {
                    return Err(VcdParseError::VcdUnexpectedNumberOfTokens(
                        "attribute".to_string(),
                        iter_bytes_to_list_str(tokens.iter()),
                    ));
                }
                match tokens[0] {
                    b"misc" => HeaderCmd::MiscAttribute(tokens),
                    _ => {
                        return Err(VcdParseError::VcdUnsupportedAttributeType(
                            iter_bytes_to_list_str(tokens.iter()),
                        ))
                    }
                }
            }
            VcdCmd::AttributeEnd => {
                // Empty command directly folloed by $end
                continue;
            }
        };
        callback(parsed)?;
    }
}

const VCD_DATE: &[u8] = b"date";
const VCD_TIMESCALE: &[u8] = b"timescale";
const VCD_VAR: &[u8] = b"var";
const VCD_SCOPE: &[u8] = b"scope";
const VCD_UP_SCOPE: &[u8] = b"upscope";
const VCD_COMMENT: &[u8] = b"comment";
const VCD_VERSION: &[u8] = b"version";
const VCD_END_DEFINITIONS: &[u8] = b"enddefinitions";
/// This might be an unofficial extension used by VHDL simulators.
const VCD_ATTRIBUTE_BEGIN: &[u8] = b"attrbegin";
/// Empty command that is generated in fst2vcd by e.g. NVCs VCD-generation
const VCD_ATTRIBUTE_END: &[u8] = b"attrend";
const VCD_COMMANDS: [&[u8]; 10] = [
    VCD_DATE,
    VCD_TIMESCALE,
    VCD_VAR,
    VCD_SCOPE,
    VCD_UP_SCOPE,
    VCD_COMMENT,
    VCD_VERSION,
    VCD_END_DEFINITIONS,
    VCD_ATTRIBUTE_BEGIN,
    VCD_ATTRIBUTE_END,
];

/// Used to show all commands when printing an error message.
fn get_vcd_command_str() -> String {
    iter_bytes_to_list_str(VCD_COMMANDS.iter())
}

fn iter_bytes_to_list_str<'a, I>(bytes: I) -> String
where
    I: Iterator<Item = &'a &'a [u8]>,
{
    bytes
        .map(|c| String::from_utf8_lossy(c))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, PartialEq)]
enum VcdCmd {
    Date,
    Timescale,
    Var,
    Scope,
    UpScope,
    Comment,
    Version,
    EndDefinitions,
    Attribute,
    AttributeEnd,
}

impl VcdCmd {
    fn from_bytes(name: &[u8]) -> Option<Self> {
        match name {
            VCD_VAR => Some(VcdCmd::Var),
            VCD_SCOPE => Some(VcdCmd::Scope),
            VCD_UP_SCOPE => Some(VcdCmd::UpScope),
            VCD_DATE => Some(VcdCmd::Date),
            VCD_TIMESCALE => Some(VcdCmd::Timescale),
            VCD_COMMENT => Some(VcdCmd::Comment),
            VCD_VERSION => Some(VcdCmd::Version),
            VCD_END_DEFINITIONS => Some(VcdCmd::EndDefinitions),
            VCD_ATTRIBUTE_BEGIN => Some(VcdCmd::Attribute),
            VCD_ATTRIBUTE_END => Some(VcdCmd::AttributeEnd),
            _ => None,
        }
    }
}

/// Tries to guess whether this input could be a VCD by looking at the first token.
pub fn is_vcd(input: &mut (impl BufRead + Seek)) -> bool {
    let is_vcd = matches!(internal_is_vcd(input), Ok(true));
    // try to reset input
    let _ = input.seek(std::io::SeekFrom::Start(0));
    is_vcd
}

/// Returns an error or false if not a vcd. Returns Ok(true) only if we think it is a vcd.
fn internal_is_vcd(input: &mut (impl BufRead + Seek)) -> Result<bool> {
    let mut buf = Vec::with_capacity(64);
    let (_cmd, _body) = read_command(input, &mut buf)?;
    Ok(true)
}

/// Reads in a command until the `$end`. Uses buf to store the read data.
/// Returns the name and the body of the command.
fn read_command<'a>(input: &mut impl BufRead, buf: &'a mut Vec<u8>) -> Result<(VcdCmd, &'a [u8])> {
    // start out with an empty buffer
    assert!(buf.is_empty());

    // skip over any preceding whitespace
    let start_char = skip_whitespace(input)?;

    if start_char != b'$' {
        return Err(VcdParseError::VcdStartChar(
            String::from_utf8_lossy(&[start_char]).to_string(),
        ));
    }

    // read the rest of the command into the buffer
    read_token(input, buf)?;

    // check to see if this is a valid command
    let cmd = VcdCmd::from_bytes(buf).ok_or_else(|| {
        VcdParseError::VcdInvalidCommand(String::from_utf8_lossy(buf).to_string())
    })?;
    buf.clear();

    // read until we find the end token
    read_until_end_token(input, buf)?;

    // return the name and body of the command
    Ok((cmd, &buf[..]))
}

#[inline]
fn find_tokens(line: &[u8]) -> Vec<&[u8]> {
    line.split(|c| matches!(*c, b' '))
        .filter(|e| !e.is_empty())
        .collect()
}

#[inline]
fn read_until_end_token(input: &mut impl BufRead, buf: &mut Vec<u8>) -> std::io::Result<()> {
    // count how many characters of the $end token we have recognized
    let mut end_index = 0;
    // we skip any whitespace at the beginning, but not between tokens
    let mut skipping_preceding_whitespace = true;
    loop {
        let byte = read_byte(input)?;
        if skipping_preceding_whitespace {
            match byte {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    continue;
                }
                _ => {
                    skipping_preceding_whitespace = false;
                }
            }
        }
        // we always append and then later drop the `$end` bytes.
        buf.push(byte);
        end_index = match (end_index, byte) {
            (0, b'$') => 1,
            (1, b'e') => 2,
            (2, b'n') => 3,
            (3, b'd') => {
                // we are done!
                buf.truncate(buf.len() - 4); // drop $end
                right_strip(buf);
                return Ok(());
            }
            _ => 0, // reset
        };
    }
}

#[inline]
fn read_token(input: &mut impl BufRead, buf: &mut Vec<u8>) -> std::io::Result<()> {
    loop {
        let byte = read_byte(input)?;
        match byte {
            b' ' | b'\n' | b'\r' | b'\t' => {
                return Ok(());
            }
            other => {
                buf.push(other);
            }
        }
    }
}

/// Advances the input until the first non-whitespace character which is then returned.
#[inline]
fn skip_whitespace(input: &mut impl BufRead) -> std::io::Result<u8> {
    loop {
        let byte = read_byte(input)?;
        match byte {
            b' ' | b'\n' | b'\r' | b'\t' => {}
            other => return Ok(other),
        }
    }
}

#[inline]
fn read_byte(input: &mut impl BufRead) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline]
fn right_strip(buf: &mut Vec<u8>) {
    while !buf.is_empty() {
        match buf.last().unwrap() {
            b' ' | b'\n' | b'\r' | b'\t' => buf.pop(),
            _ => break,
        };
    }
}

enum HeaderCmd<'a> {
    Date(&'a [u8]),
    Version(&'a [u8]),
    Comment(&'a [u8]),
    Timescale(&'a [u8], &'a [u8]), // factor, unit
    Scope(&'a [u8], &'a [u8]),     // tpe, name
    UpScope,
    Var(&'a [u8], &'a [u8], &'a [u8], &'a [u8]), // tpe, size, id, name
    /// Misc attributes are emitted by nvc (VHDL sim) and fst2vcd (included with GTKwave).
    MiscAttribute(Vec<&'a [u8]>),
}

/// The minimum number of bytes we want to read per thread.
const MIN_CHUNK_SIZE: usize = 8 * 1024;

/// Returns starting byte and read length for every thread. Note that read-length is just an
/// approximation and the thread might have to read beyond or might also run out of data before
/// reaching read length.
#[inline]
fn determine_thread_chunks(body_len: usize) -> Vec<(usize, usize)> {
    let max_threads = rayon::current_num_threads();
    let number_of_threads_for_min_chunk_size = body_len.div_ceil(MIN_CHUNK_SIZE);
    let num_threads = std::cmp::min(max_threads, number_of_threads_for_min_chunk_size);
    let chunk_size = body_len.div_ceil(num_threads);
    // TODO: for large file it might make sense to have more chunks than threads
    (0..num_threads)
        .map(|ii| (ii * chunk_size, chunk_size))
        .collect()
}

/// Reads the body of a VCD with multiple threads
fn read_values(
    input: &[u8],
    multi_thread: bool,
    hierarchy: &Hierarchy,
    lookup: &IdLookup,
    progress: Option<ProgressCount>,
) -> Result<(SignalSource, TimeTable)> {
    if multi_thread {
        let chunks = determine_thread_chunks(input.len());
        let encoders: Result<Vec<crate::wavemem::Encoder>> = chunks
            .par_iter()
            .map(|(start, len)| {
                let is_first = *start == 0;
                let mut inp = std::io::Cursor::new(&input[*start..]);
                read_single_stream_of_values(
                    &mut inp,
                    *len - 1,
                    is_first,
                    hierarchy,
                    lookup,
                    progress.clone(),
                )
            })
            .collect();
        let encoders = encoders?;

        // combine encoders
        let mut encoder_iter = encoders.into_iter();
        let mut encoder = encoder_iter.next().unwrap();
        for other in encoder_iter {
            encoder.append(other);
        }
        Ok(encoder.finish())
    } else {
        let mut inp = std::io::Cursor::new(input);
        let encoder = read_single_stream_of_values(
            &mut inp,
            input.len() - 1,
            true,
            hierarchy,
            lookup,
            progress,
        )?;
        Ok(encoder.finish())
    }
}

fn is_white_space(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

enum FirstTokenResult {
    Time(u64),
    OneBitValue,
    MultiBitValue,
    CommentStart,
    IgnoredCmd,
}

fn parse_first_token(token: &[u8]) -> Result<FirstTokenResult> {
    match token[0] {
        b'#' => {
            let value_str = std::str::from_utf8(&token[1..])?;
            // Try parsing as u64
            let value = match value_str.parse::<u64>() {
                Ok(val) => Ok(val),
                Err(e) => {
                    // Try parsing as f64
                    match value_str.parse::<f64>() {
                        Ok(val) => {
                            if val.fract() == 0.0 {
                                // Convert to u64 if no fractional part
                                Ok(val as u64)
                            } else {
                                Err(e)
                            }
                        }
                        Err(_) => Err(e),
                    }
                }
            }?;

            Ok(FirstTokenResult::Time(value))
        }
        b'0' | b'1' | b'z' | b'Z' | b'x' | b'X' | b'h' | b'H' | b'u' | b'U' | b'w' | b'W'
        | b'l' | b'L' | b'-' => Ok(FirstTokenResult::OneBitValue),
        b'b' | b'B' | b'r' | b'R' | b's' | b'S' => Ok(FirstTokenResult::MultiBitValue),
        _ => {
            match token {
                b"$dumpall" => {
                    // interpret dumpall as indicating timestep zero
                    Ok(FirstTokenResult::Time(0))
                }
                b"$comment" => Ok(FirstTokenResult::CommentStart),
                b"$dumpvars" | b"$end" | b"$dumpoff" | b"$dumpon" => {
                    // ignore dumpvars, dumpoff, dumpon, and end command
                    Ok(FirstTokenResult::IgnoredCmd)
                }
                _ => Err(VcdParseError::VcdUnexpectedBodyToken(
                    String::from_utf8_lossy(token).to_string(),
                )),
            }
        }
    }
}

/// wraps a wavemem encoder and adds vcd specific handling of input time
struct VcdEncoder<'a> {
    enc: Encoder,
    lookup: &'a IdLookup,
    is_first_part_of_vcd: bool,
    found_first_time_step: bool,
}

impl<'a> VcdEncoder<'a> {
    #[inline]
    fn new(hierarchy: &Hierarchy, lookup: &'a IdLookup, is_first_part_of_vcd: bool) -> Self {
        let found_first_time_step = false;
        Self {
            enc: Encoder::new(hierarchy),
            lookup,
            is_first_part_of_vcd,
            found_first_time_step,
        }
    }

    #[inline]
    fn into_inner(self) -> Encoder {
        self.enc
    }
}

impl ParseBodyOutput for VcdEncoder<'_> {
    #[inline]
    fn time(&mut self, value: u64) -> Result<()> {
        self.found_first_time_step = true;
        self.enc.time_change(value);
        Ok(())
    }

    #[inline]
    fn value(&mut self, value: &[u8], id: &[u8]) -> Result<()> {
        // In the first thread, we might encounter a dump values which dumps all initial values
        // without specifying a timestamp
        if self.is_first_part_of_vcd && !self.found_first_time_step {
            self.time(0)?;
        }
        // if we are not the first part of the VCD, we are skipping value changes until the
        // first timestep is found which serves as a synchronization point
        if self.found_first_time_step {
            let num_id = match self.lookup {
                None => match id_to_int(id) {
                    Some(ii) => ii,
                    None => {
                        debug_assert!(id.is_empty());
                        return Err(VcdParseError::VcdEmptyId);
                    }
                },
                Some(lookup) => lookup[id].index() as u64,
            };
            self.enc.vcd_value_change(num_id, value);
        }
        Ok(())
    }
}

struct ProgressReporter {
    progress: Option<ProgressCount>,
    last_reported_pos: usize,
    report_increments: usize,
}

impl ProgressReporter {
    #[inline]
    fn new(progress: Option<ProgressCount>, len: usize) -> Self {
        let last_reported_pos = 0;
        let report_increments = std::cmp::max(len / 1000, 512);
        Self {
            progress,
            last_reported_pos,
            report_increments,
        }
    }

    #[inline]
    fn report(&mut self, pos: usize, always_report: bool) {
        if let Some(p) = self.progress.as_ref() {
            let increment = pos - self.last_reported_pos;
            if always_report || increment > self.report_increments {
                p.fetch_add(increment as u64, Ordering::SeqCst);
                self.last_reported_pos = pos;
            }
        }
    }
}

fn read_single_stream_of_values<R: BufRead + Seek>(
    input: &mut R,
    stop_pos: usize,
    is_first: bool,
    hierarchy: &Hierarchy,
    lookup: &IdLookup,
    progress: Option<ProgressCount>,
) -> Result<crate::wavemem::Encoder> {
    let mut encoder = VcdEncoder::new(hierarchy, lookup, is_first);
    parse_body(input, &mut encoder, stop_pos, progress)?;
    Ok(encoder.into_inner())
}

trait ParseBodyOutput {
    fn time(&mut self, value: u64) -> Result<()>;
    fn value(&mut self, value: &[u8], id: &[u8]) -> Result<()>;
}

fn parse_body(
    input: &mut impl BufRead,
    out: &mut impl ParseBodyOutput,
    stop_pos: usize,
    progress: Option<ProgressCount>,
) -> Result<()> {
    let mut progress_report = ProgressReporter::new(progress, stop_pos);

    let mut state = BodyState::SkippingNewLine;

    let mut first = Vec::with_capacity(32);
    let mut id = Vec::with_capacity(32);
    let mut final_pos = 0;

    for (pos, b) in input.bytes().enumerate() {
        final_pos = pos;
        progress_report.report(pos, false);
        let b = b?;
        match state {
            BodyState::SkippingNewLine => {
                if b == b'\n' {
                    debug_assert!(first.is_empty());
                    state = BodyState::ParsingFirstToken;
                }
            }
            BodyState::ParsingFirstToken => {
                if is_white_space(b) {
                    if first.is_empty() {
                        // we are in front of the token => nothing to do
                    } else {
                        state = match parse_first_token(&first)? {
                            FirstTokenResult::Time(value) => {
                                // check to see if this time value is already fully past
                                // the stop position
                                let time_token_start = pos - first.len() - 1;
                                if time_token_start > stop_pos {
                                    // exit
                                    progress_report.report(pos, true);
                                    return Ok(());
                                }
                                // record time step if we aren't exiting
                                out.time(value)?;
                                BodyState::ParsingFirstToken
                            }
                            FirstTokenResult::OneBitValue => {
                                out.value(&first[0..1], &first[1..])?;
                                BodyState::ParsingFirstToken
                            }
                            FirstTokenResult::MultiBitValue => BodyState::ParsingIdToken,
                            FirstTokenResult::CommentStart => BodyState::LookingForEndToken,
                            FirstTokenResult::IgnoredCmd => BodyState::ParsingFirstToken,
                        };

                        // clear buffer to find next token
                        if state != BodyState::ParsingIdToken {
                            first.clear();
                        }
                    }
                } else {
                    first.push(b);
                }
            }

            BodyState::ParsingIdToken => {
                if is_white_space(b) {
                    if id.is_empty() {
                        // we are in front of the token => nothing to do
                    } else {
                        out.value(first.as_slice(), id.as_slice())?;
                        first.clear();
                        id.clear();
                        state = BodyState::ParsingFirstToken;
                    }
                } else {
                    id.push(b);
                }
            }
            BodyState::LookingForEndToken => {
                if is_white_space(b) {
                    if first.is_empty() {
                        // we are in front of the token => nothing to do
                    } else {
                        if first == b"$end" {
                            state = BodyState::ParsingFirstToken;
                        }
                        first.clear();
                    }
                } else {
                    first.push(b);
                }
            }
        }
    }

    // we reached the end of the file
    match state {
        BodyState::ParsingFirstToken => {
            if !first.is_empty() {
                match parse_first_token(&first)? {
                    FirstTokenResult::Time(value) => {
                        out.time(value)?;
                    }
                    FirstTokenResult::OneBitValue => {
                        out.value(&first[0..1], &first[1..])?;
                    }
                    _ => {} // nothing to do
                };
            }
        }
        BodyState::ParsingIdToken => {
            out.value(first.as_slice(), id.as_slice())?;
        }
        _ => {} // nothing to do
    }
    progress_report.report(final_pos, true);
    Ok(())
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum BodyState {
    /// initially the body parser might skip ahead to the next newline in order to synchronize
    SkippingNewLine,
    ParsingFirstToken,
    ParsingIdToken,
    LookingForEndToken,
}

#[cfg(test)]
mod tests {
    use super::*;

    impl ParseBodyOutput for Vec<String> {
        fn time(&mut self, value: u64) -> Result<()> {
            self.push(format!("Time({value})"));
            Ok(())
        }

        fn value(&mut self, value: &[u8], id: &[u8]) -> Result<()> {
            let desc = format!(
                "{} = {}",
                std::str::from_utf8(id)?,
                std::str::from_utf8(value)?
            );
            self.push(desc);
            Ok(())
        }
    }

    fn read_body_to_vec(input: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        parse_body(
            &mut std::io::Cursor::new(input),
            &mut out,
            input.len(),
            None,
        )
        .unwrap();
        out
    }

    #[test]
    fn test_read_body() {
        let input = r#"
1I,!
1J,!
1#2!
#2678437829
b00 D2!
b0000 d2!
b11 e2!
b00000 f2!
b10100 g2!
b00000 h2!
b00000 i2!
x(i"
x'i"
x&i"
x%i"
0j2!"#;
        let expected = vec![
            "I,! = 1",
            "J,! = 1",
            "#2! = 1",
            "Time(2678437829)",
            "D2! = b00",
            "d2! = b0000",
            "e2! = b11",
            "f2! = b00000",
            "g2! = b10100",
            "h2! = b00000",
            "i2! = b00000",
            "(i\" = x",
            "'i\" = x",
            "&i\" = x",
            "%i\" = x",
            "j2! = 0",
        ];
        let res = read_body_to_vec(input.as_bytes());
        assert_eq!(res, expected);
    }

    #[test]
    fn test_read_command() {
        let mut buf = Vec::with_capacity(128);
        let input_0 = b"$upscope $end";
        let (cmd_0, body_0) = read_command(&mut input_0.as_slice(), &mut buf).unwrap();
        assert_eq!(cmd_0, VcdCmd::UpScope);
        assert!(body_0.is_empty());

        // test with more whitespace
        buf.clear();
        let input_1 = b" \t $upscope \n $end  \n ";
        let (cmd_1, body_1) = read_command(&mut input_1.as_slice(), &mut buf).unwrap();
        assert_eq!(cmd_1, VcdCmd::UpScope);
        assert!(body_1.is_empty());
    }

    #[test]
    fn test_id_to_int() {
        assert_eq!(id_to_int(b""), None);
        assert_eq!(id_to_int(b"!"), Some(0));
        assert_eq!(id_to_int(b"#"), Some(2));
        assert_eq!(id_to_int(b"*"), Some(9));
        assert_eq!(id_to_int(b"c"), Some(66));
        assert_eq!(id_to_int(b"#%"), Some(472));
        assert_eq!(id_to_int(b"("), Some(7));
        assert_eq!(id_to_int(b")"), Some(8));
    }

    #[test]
    fn test_find_last() {
        assert_eq!(find_last(b"1234", b'1'), Some(0));
        assert_eq!(find_last(b"1234", b'5'), None);
        assert_eq!(find_last(b"12341", b'1'), Some(4));
    }

    fn do_test_parse_name(
        length: u32,
        full_name: &str,
        name: &str,
        index: Option<(i64, i64)>,
        scopes: &[&str],
    ) {
        let (a_name, a_index, a_scopes) = parse_name(full_name.as_bytes(), length).unwrap();
        assert_eq!(a_name, name);
        match index {
            None => assert!(a_index.is_none()),
            Some((msb, lsb)) => {
                assert_eq!(a_index.unwrap().msb(), msb);
                assert_eq!(a_index.unwrap().lsb(), lsb);
            }
        }
        assert_eq!(a_scopes, scopes);
    }

    #[test]
    fn test_parse_name() {
        do_test_parse_name(32, "test", "test", None, &[]);
        do_test_parse_name(1, "test[0]", "test", Some((0, 0)), &[]);
        do_test_parse_name(1, "test [0]", "test", Some((0, 0)), &[]);
        do_test_parse_name(2, "test[1:0]", "test", Some((1, 0)), &[]);
        do_test_parse_name(2, "test [1:0]", "test", Some((1, 0)), &[]);
        do_test_parse_name(3, "test[1:-1]", "test", Some((1, -1)), &[]);
        do_test_parse_name(3, "test [1:-1]", "test", Some((1, -1)), &[]);
        do_test_parse_name(1, "test[0][0]", "[0]", Some((0, 0)), &["test"]);
        do_test_parse_name(1, "test[0] [0]", "[0]", Some((0, 0)), &["test"]);
        do_test_parse_name(1, "test [0] [0]", "[0]", Some((0, 0)), &["test"]);
        do_test_parse_name(1, "test[3][2][0]", "[2]", Some((0, 0)), &["test", "[3]"]);
        do_test_parse_name(
            1,
            "test[0][3][2][0]",
            "[2]",
            Some((0, 0)),
            &["test", "[0]", "[3]"],
        );
        // if the length is not 1-bit, the suffix is parsed as an array index
        do_test_parse_name(
            32,
            "test[0][3][2][0]",
            "[0]",
            None,
            &["test", "[0]", "[3]", "[2]"],
        );
        do_test_parse_name(11, "test [10:0]", "test", Some((10, 0)), &[]);
    }
}
