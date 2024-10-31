// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::fst::{parse_scope_attributes, parse_var_attributes, Attribute};
use crate::hierarchy::*;
use crate::signals::SignalSource;
use crate::viewers::ProgressCount;
use crate::{FileFormat, LoadOptions, TimeTable};
use fst_reader::{FstVhdlDataType, FstVhdlVarType};
use num_enum::TryFromPrimitive;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
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
    #[error("[vcd] unexpected number of tokens for command {0}: {1}")]
    VcdUnexpectedNumberOfTokens(String, String),
    #[error("[vcd] encountered a attribute with an unsupported type: {0}")]
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
            // determine body length
            let start = input.stream_position()?;
            input.seek(SeekFrom::End(0))?;
            let end = input.stream_position()?;
            input.seek(SeekFrom::Start(start))?;
            let input_len = end - start;

            // encode signals
            let encoder = read_single_stream_of_values(
                &mut input,
                input_len - 1,
                true,
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
    path_names: &mut HashMap<u64, HierarchyStringId>,
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

type IdLookup = Option<HashMap<Vec<u8>, SignalRef>>;

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
    let mut path_names = HashMap::new();
    // this map is used to translate identifiers to signal references for cases where we detect ids that are too large
    let mut id_map: HashMap<Vec<u8>, SignalRef> = HashMap::new();
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
            let (var_name, index, scopes) = parse_name(name)?;
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
pub fn parse_name(name: &[u8]) -> Result<(String, Option<VarIndex>, Vec<String>)> {
    if name.is_empty() {
        return Ok(("".to_string(), None, vec![]));
    }
    debug_assert!(
        name[0] != b'[',
        "we assume that the first character is not `[`!"
    );

    // find the bit index from the back
    let (mut name, index) = extract_suffix_index(name);

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
        result = match result
            .checked_mul(NUM_ID_CHARS)
            .and_then(|x| x.checked_add(c))
        {
            None => return None,
            Some(value) => value,
        };
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
const VCD_COMMANDS: [&[u8]; 9] = [
    VCD_DATE,
    VCD_TIMESCALE,
    VCD_VAR,
    VCD_SCOPE,
    VCD_UP_SCOPE,
    VCD_COMMENT,
    VCD_VERSION,
    VCD_END_DEFINITIONS,
    VCD_ATTRIBUTE_BEGIN,
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
            _ => None,
        }
    }

    fn from_bytes_or_panic(name: &[u8]) -> Self {
        match Self::from_bytes(name) {
            None => {
                panic!(
                    "Unexpected VCD command {}. Supported commands are: {:?}",
                    String::from_utf8_lossy(name),
                    get_vcd_command_str()
                );
            }
            Some(cmd) => cmd,
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
    let cmd = VcdCmd::from_bytes_or_panic(buf);
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
                // check to see if the chunk start on a new line
                let starts_on_new_line = if is_first {
                    true
                } else {
                    let before = input[*start - 1];
                    // TODO: deal with \n\r
                    before == b'\n'
                };
                let mut inp = std::io::Cursor::new(&input[*start..]);
                read_single_stream_of_values(
                    &mut inp,
                    (*len - 1) as u64,
                    is_first,
                    starts_on_new_line,
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
            (input.len() - 1) as u64,
            true,
            true,
            hierarchy,
            lookup,
            progress,
        )?;
        Ok(encoder.finish())
    }
}

fn read_single_stream_of_values<R: BufRead + Seek>(
    input: &mut R,
    stop_pos: u64,
    is_first: bool,
    starts_on_new_line: bool,
    hierarchy: &Hierarchy,
    lookup: &IdLookup,
    progress: Option<ProgressCount>,
) -> Result<crate::wavemem::Encoder> {
    let mut encoder = crate::wavemem::Encoder::new(hierarchy);

    if !starts_on_new_line {
        // if we start in the middle of a line, we need to skip it
        let mut dummy = Vec::new();
        input.read_until(b'\n', &mut dummy)?;
    }
    let mut reader = BodyReader::new(input);
    // We only start recording once we have encountered our first time step
    let mut found_first_time_step = false;

    // progress tracking
    let mut last_reported_pos = 0;
    let report_increments = std::cmp::max(stop_pos as u64 / 1000, 512);

    loop {
        if let Some((pos, cmd)) = reader.next() {
            if pos > stop_pos {
                if let BodyCmd::Time(_to) = cmd {
                    if let Some(p) = progress.as_ref() {
                        let increment = (pos - last_reported_pos) as u64;
                        p.fetch_add(increment, Ordering::SeqCst);
                    }
                    break; // stop before the next time value when we go beyond the stop position
                }
            }
            if let Some(p) = progress.as_ref() {
                let increment = (pos - last_reported_pos) as u64;
                if increment >= report_increments {
                    last_reported_pos = pos;
                    p.fetch_add(increment, Ordering::SeqCst);
                }
            }
            match cmd {
                BodyCmd::Time(value) => {
                    found_first_time_step = true;
                    let int_value = std::str::from_utf8(value).unwrap().parse::<u64>().unwrap();
                    encoder.time_change(int_value);
                }
                BodyCmd::Value(value, id) => {
                    // In the first thread, we might encounter a dump values which dumps all initial values
                    // without specifying a timestamp
                    if is_first && !found_first_time_step {
                        encoder.time_change(0);
                        found_first_time_step = true;
                    }
                    if found_first_time_step {
                        let num_id = match lookup {
                            None => id_to_int(id).unwrap(),
                            Some(lookup) => lookup[id].index() as u64,
                        };
                        encoder.vcd_value_change(num_id, value);
                    }
                }
            };
        } else {
            if let Some(p) = progress.as_ref() {
                let increment = reader.get_pos()? - last_reported_pos;
                p.fetch_add(increment, Ordering::SeqCst);
            }
            break; // done, no more values to read
        }
    }

    Ok(encoder)
}

#[inline]
fn advance_to_first_newline(input: &[u8]) -> (&[u8], usize) {
    for (pos, byte) in input.iter().enumerate() {
        if *byte == b'\n' {
            return (&input[pos..], pos);
        }
    }
    (&[], 0) // no whitespaces found
}

struct BodyReader<'a, R: BufRead> {
    input: &'a mut R,
    // state
    token: Vec<u8>,
    prev_token: Vec<u8>,
    // statistics
    lines_read: u64,
}

const ASCII_ZERO: &[u8] = b"0";

impl<'a, R: BufRead + Seek> BodyReader<'a, R> {
    fn new(input: &'a mut R) -> Self {
        BodyReader {
            input,
            token: Vec::with_capacity(64),
            prev_token: Vec::with_capacity(64),
            lines_read: 0,
        }
    }

    #[inline]
    fn try_finish_token(&mut self, search_for_end: &mut bool) -> Option<BodyCmd<'_>> {
        // no token means that there is nothing to do
        if self.token.is_empty() {
            return None;
        }

        // if we are looking for the $end token, we discard everything else
        if *search_for_end {
            // did we find the end token?
            *search_for_end = self.token != b"$end";
            // consume token and return
            self.token.clear();
            return None;
        }

        // if there was no previous token
        if self.prev_token.is_empty() {
            if self.token.len() == 1 {
                // too short, wait for more input
                return None;
            }

            // 1-token commands are binary changes or time commands
            match self.token[0] {
                b'#' => Some(BodyCmd::Time(&self.token[1..])),
                b'0' | b'1' | b'z' | b'Z' | b'x' | b'X' | b'h' | b'H' | b'u' | b'U' | b'w'
                | b'W' | b'l' | b'L' | b'-' => {
                    Some(BodyCmd::Value(&self.token[0..1], &self.token[1..]))
                }
                _ => {
                    // parse command tokens
                    match self.token.as_slice() {
                        b"$dumpall" => {
                            // interpret dumpall as indicating timestep zero
                            self.token.clear();
                            return Some(BodyCmd::Time(ASCII_ZERO));
                        }
                        b"$comment" => {
                            // drop token, but start searching for $end in order to skip the comment
                            *search_for_end = true;
                        }
                        b"$dumpvars" | b"$end" | b"$dumpoff" | b"$dumpon" => {
                            // ignore dumpvars, dumpoff, dumpon, and end command
                            self.prev_token.copy_from_slice(self.token.as_slice());
                        }
                        _ => {} // do nothing
                    }
                    // wait for more input
                    None
                }
            }
        } else {
            let cmd = match self.prev_token[0] {
                b'b' | b'B' | b'r' | b'R' | b's' | b'S' => {
                    BodyCmd::Value(&self.prev_token[0..], self.token.as_slice())
                }
                _ => {
                    panic!(
                        "Unexpected tokens: `{}` and `{}` ({} lines after header)",
                        String::from_utf8_lossy(self.prev_token.as_slice()),
                        String::from_utf8_lossy(self.token.as_slice()),
                        self.lines_read
                    );
                }
            };
            Some(cmd)
        }
    }

    #[inline]
    fn get_pos(&mut self) -> Result<u64> {
        Ok(self.input.stream_position()?)
    }
}

#[inline]
fn try_read_u8(input: &mut impl BufRead) -> Option<u8> {
    let mut buf = [0u8; 1];
    match input.read_exact(&mut buf) {
        Ok(_) => Some(buf[0]),
        Err(_) => None,
    }
}

impl<'a, R: BufRead + Seek> Iterator for BodyReader<'a, R> {
    type Item = (u64, BodyCmd<'a>);

    /// returns the starting position and the body of the command
    #[inline]
    fn next(&mut self) -> Option<(u64, BodyCmd<'a>)> {
        debug_assert!(self.token.is_empty());
        debug_assert!(self.prev_token.is_empty());

        let mut pending_lines = 0u64;
        let mut start_pos = 0u64;
        // if we encounter a $comment, we will just be searching for a $end token
        let mut search_for_end = false;
        while let Some(b) = try_read_u8(self.input) {
            match b {
                // a white space indicates the end of a token
                b' ' | b'\n' | b'\r' | b'\t' => {
                    // skip whitespace if we haven't started a token yet
                    if self.token.is_empty() {
                        if b == b'\n' {
                            self.lines_read += 1;
                        }
                    } else {
                        match self.try_finish_token(&mut search_for_end) {
                            None => {
                                if b == b'\n' {
                                    pending_lines += 1;
                                }
                            }
                            Some(cmd) => {
                                // save state
                                self.lines_read += pending_lines;
                                if b == b'\n' {
                                    self.lines_read += 1;
                                }
                                return Some((start_pos, cmd));
                            }
                        }
                    }
                }
                _ => {
                    self.token.push(b);
                    if self.prev_token.is_empty() {
                        // remember the start of the first token
                        start_pos = self.input.stream_position().unwrap();
                    }
                }
            }
        }

        // check to see if there is a final token at the end
        match self.try_finish_token(&mut search_for_end) {
            None => {}
            Some(cmd) => {
                return Some((start_pos, cmd));
            }
        }
        // now we are done
        None
    }
}

enum BodyCmd<'a> {
    Time(&'a [u8]),
    Value(&'a [u8], &'a [u8]),
}

impl Debug for BodyCmd<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BodyCmd::Time(value) => {
                write!(f, "Time({})", String::from_utf8_lossy(value))
            }
            BodyCmd::Value(value, id) => {
                write!(
                    f,
                    "Value({}, {})",
                    String::from_utf8_lossy(id),
                    String::from_utf8_lossy(value)
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_body_to_vec(input: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let reader = BodyReader::new(input);
        for (_, cmd) in reader {
            let desc = match cmd {
                BodyCmd::Time(value) => {
                    format!("Time({})", std::str::from_utf8(value).unwrap())
                }
                BodyCmd::Value(value, id) => {
                    format!(
                        "{} = {}",
                        std::str::from_utf8(id).unwrap(),
                        std::str::from_utf8(value).unwrap()
                    )
                }
            };
            out.push(desc);
        }
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

    fn do_test_parse_name(full_name: &str, name: &str, index: Option<(i64, i64)>, scopes: &[&str]) {
        let (a_name, a_index, a_scopes) = parse_name(full_name.as_bytes()).unwrap();
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
        do_test_parse_name("test", "test", None, &[]);
        do_test_parse_name("test[0]", "test", Some((0, 0)), &[]);
        do_test_parse_name("test [0]", "test", Some((0, 0)), &[]);
        do_test_parse_name("test[1:0]", "test", Some((1, 0)), &[]);
        do_test_parse_name("test [1:0]", "test", Some((1, 0)), &[]);
        do_test_parse_name("test[1:-1]", "test", Some((1, -1)), &[]);
        do_test_parse_name("test [1:-1]", "test", Some((1, -1)), &[]);
        do_test_parse_name("test[0][0]", "[0]", Some((0, 0)), &["test"]);
        do_test_parse_name("test[0] [0]", "[0]", Some((0, 0)), &["test"]);
        do_test_parse_name("test [0] [0]", "[0]", Some((0, 0)), &["test"]);
        do_test_parse_name("test[3][2][0]", "[2]", Some((0, 0)), &["test", "[3]"]);
        do_test_parse_name(
            "test[0][3][2][0]",
            "[2]",
            Some((0, 0)),
            &["test", "[0]", "[3]"],
        );
        do_test_parse_name("test [10:0]", "test", Some((10, 0)), &[]);
    }
}
