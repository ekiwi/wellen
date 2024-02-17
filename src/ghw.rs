// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use num_enum::TryFromPrimitive;
use std::io::{BufRead, Read};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GhwParseError {
    #[error("[ghw] unsupported compression: {0}")]
    UnsupportedCompression(String),
    #[error("[ghw] unexpected header magic: {0}")]
    UnexpectedHeaderMagic(String),
    #[error("[ghw] unexpected value in header: {0:?}")]
    UnexpectedHeader(HeaderData),
    #[error("[ghw] unexpected section start: {0}")]
    UnexpectedSection(String),
    #[error("[ghw] failed to parse a {0} section: {1}")]
    FailedToParseSection(&'static str, String),
    #[error("[ghw] expected positive integer, not: {0}")]
    ExpectedPositiveInteger(i64),
    #[error("[ghw] failed to parse GHDL RTIK.")]
    FailedToParseGhdlRtik(#[from] num_enum::TryFromPrimitiveError<GhdlRtik>),
    #[error("[ghw] failed to parse a leb128 encoded number")]
    FailedToParsLeb128(#[from] leb128::read::Error),
    #[error("[ghw] failed to decode string")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("[ghw] failed to parse an integer")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("[ghw] I/O operation failed")]
    Io(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, GhwParseError>;

fn load(filename: &str) -> Result<()> {
    let f = std::fs::File::open(filename)?;
    let mut input = std::io::BufReader::new(f);

    let header = parse_header(&mut input)?;
    println!("{header:?}");

    let mut string_table = Vec::new();
    let mut type_table = Vec::new();

    while let Some(section) = read_section(&header, &mut input)? {
        match section {
            Section::StringTable(table) => {
                debug_assert!(
                    string_table.is_empty(),
                    "unexpected second string table:\n{:?}\n{:?}",
                    &string_table,
                    &table
                );
                string_table = table;
            }
            Section::TypeTable(table) => {
                debug_assert!(
                    type_table.is_empty(),
                    "unexpected second type table:\n{:?}\n{:?}",
                    &type_table,
                    &table
                );
                type_table = table;
            }
        }
    }

    Ok(())
}

fn parse_header(input: &mut impl BufRead) -> Result<HeaderData> {
    // check for compression
    let mut comp_header = [0u8; 2];
    input.read_exact(&mut comp_header)?;
    match &comp_header {
        b"GH" => {} // OK
        GHW_GZIP_HEADER => return Err(GhwParseError::UnsupportedCompression("gzip".to_string())),
        GHW_BZIP2_HEADER => return Err(GhwParseError::UnsupportedCompression("bzip2".to_string())),
        other => {
            return Err(GhwParseError::UnexpectedHeaderMagic(
                String::from_utf8_lossy(other).to_string(),
            ))
        }
    }

    // check full header
    let mut magic_rest = [0u8; GHW_HEADER_START.len() - 2];
    input.read_exact(&mut magic_rest)?;
    if &magic_rest != &GHW_HEADER_START[2..] {
        return Err(GhwParseError::UnexpectedHeaderMagic(format!(
            "{}{}",
            String::from_utf8_lossy(&comp_header),
            String::from_utf8_lossy(&magic_rest)
        )));
    }

    // parse the rest of the header
    let mut h = [0u8; 16 - GHW_HEADER_START.len()];
    input.read_exact(&mut h)?;
    let data = HeaderData {
        version: h[2],
        big_endian: h[3] == 2,
        word_len: h[4],
        word_offset: h[5],
    };

    if h[0] != 16 || h[1] != 0 {
        return Err(GhwParseError::UnexpectedHeader(data));
    }

    if data.version > 1 {
        return Err(GhwParseError::UnexpectedHeader(data));
    }

    if h[3] != 1 && h[3] != 2 {
        return Err(GhwParseError::UnexpectedHeader(data));
    }

    if h[6] != 0 {
        return Err(GhwParseError::UnexpectedHeader(data));
    }

    Ok(data)
}

const GHW_STRING_SECTION: &[u8; 4] = b"STR\x00";
const GHW_HIERARCHY_SECTION: &[u8; 4] = b"HIE\x00";
const GHW_TYPE_SECTION: &[u8; 4] = b"TYP\x00";
const GHW_WK_TYPE_SECTION: &[u8; 4] = b"WKT\x00";
const GHW_END_SECTION: &[u8; 4] = b"EOH\x00";

fn read_section(header: &HeaderData, input: &mut impl BufRead) -> Result<Option<Section>> {
    let mut mark = [0u8; 4];
    input.read_exact(&mut mark)?;

    match &mark {
        GHW_STRING_SECTION => Ok(Some(Section::StringTable(read_string_section(
            header, input,
        )?))),
        GHW_HIERARCHY_SECTION => todo!("parse hierarchy section"),
        GHW_TYPE_SECTION => Ok(Some(Section::TypeTable(read_type_section(header, input)?))),
        GHW_WK_TYPE_SECTION => todo!("parse wk type section"),
        GHW_END_SECTION => Ok(None),
        other => Err(GhwParseError::UnexpectedSection(
            String::from_utf8_lossy(other).to_string(),
        )),
    }
}

enum Section {
    StringTable(Vec<String>),
    TypeTable(Vec<GhwTypeInfo>),
}

#[inline]
fn read_u8(input: &mut impl BufRead) -> Result<u8> {
    let mut buf = [0u8];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_string_section(header: &HeaderData, input: &mut impl BufRead) -> Result<Vec<String>> {
    let mut h = [0u8; 12];
    input.read_exact(&mut h)?;

    if &h[..4] != b"\x00\x00\x00\x00" {
        return Err(GhwParseError::FailedToParseSection(
            "string",
            "first four bytes should be zero".to_string(),
        ));
    }

    let string_num = header.read_u32(&mut &h[4..8])? + 1;
    let _string_size = header.read_i32(&mut &h[8..12])? as u32;

    let mut string_table = Vec::with_capacity(string_num as usize);
    string_table.push("<anon>".to_string());

    let mut buf = Vec::with_capacity(64);

    for i in 1..(string_num + 1) {
        let mut c = 0;
        loop {
            c = read_u8(input)?;
            if c <= 31 || (c >= 128 && c <= 159) {
                break;
            } else {
                buf.push(c);
            }
        }

        // push the value to the string table
        let value = String::from_utf8_lossy(&buf).to_string();
        string_table.push(value);

        // determine the length of the shared prefix
        let mut prev_len = (c & 0x1f) as usize;
        let mut shift = 5;
        while c >= 128 {
            c = read_u8(input)?;
            prev_len |= ((c & 0x1f) as usize) << shift;
            shift += 5;
        }
        buf.truncate(prev_len);
    }

    Ok(string_table)
}

fn read_string_id(input: &mut impl BufRead) -> Result<StringId> {
    let value = leb128::read::unsigned(input)?;
    Ok(StringId(value as usize))
}

fn read_type_section(header: &HeaderData, input: &mut impl BufRead) -> Result<Vec<GhwTypeInfo>> {
    let mut h = [0u8; 8];
    input.read_exact(&mut h)?;

    if &h[..4] != b"\x00\x00\x00\x00" {
        return Err(GhwParseError::FailedToParseSection(
            "type",
            "first four bytes should be zero".to_string(),
        ));
    }

    let type_num = header.read_u32(&mut &h[4..8])?;
    let mut table = Vec::with_capacity(type_num as usize);

    for _ in 0..type_num {
        let t = read_u8(input)?;
        let kind = GhdlRtik::try_from_primitive(t)?;
        let name = read_string_id(input)?;
        let tpe = match kind {
            GhdlRtik::TypeE8 | GhdlRtik::TypeB2 => {
                let num_literals = leb128::read::unsigned(input)?;
                let mut literals = Vec::with_capacity(num_literals as usize);
                for _ in 0..num_literals {
                    literals.push(read_string_id(input)?);
                }

                GhwType::Enum {
                    wkt: GhwWellKnownType::Unknown,
                    literals,
                }
            }
            other => todo!("Support: {other:?}"),
        };
        let info = GhwTypeInfo { kind, name, tpe };
        println!("{info:?}");
        table.push(info);
    }

    Ok(table)
}

/// Pointer into the GHW string table.
#[derive(Debug, Copy, Clone, PartialEq)]
struct StringId(usize);
/// Pointer into the GHW type table.
#[derive(Debug, Copy, Clone, PartialEq)]
struct TypeId(usize);

#[derive(Debug)]
struct GhwTypeInfo {
    kind: GhdlRtik,
    name: StringId,
    tpe: GhwType,
}

#[derive(Debug)]
enum GhwType {
    Enum {
        wkt: GhwWellKnownType,
        literals: Vec<StringId>,
    },
    Scalar,
    Physical {
        units: Vec<GhwUnit>,
    },
    Array {
        elements: Vec<TypeId>,
        dimensions: Vec<TypeId>,
    },
    UnboundedArray {
        base: TypeId,
    },
    // TODO
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct GhwUnit {
    name: StringId,
    value: i64,
}

#[derive(Debug)]
struct HeaderData {
    version: u8,
    big_endian: bool,
    word_len: u8,
    word_offset: u8,
}

impl HeaderData {
    fn read_i32(&self, input: &mut impl BufRead) -> Result<i32> {
        let mut b = [0u8; 4];
        input.read_exact(&mut b)?;
        if self.big_endian {
            Ok(i32::from_be_bytes(b))
        } else {
            Ok(i32::from_le_bytes(b))
        }
    }
    fn read_u32(&self, input: &mut impl BufRead) -> Result<u32> {
        let ii = self.read_i32(input)?;
        if ii >= 0 {
            Ok(ii as u32)
        } else {
            Err(GhwParseError::ExpectedPositiveInteger(ii as i64))
        }
    }
}

const GHW_GZIP_HEADER: &[u8; 2] = &[0x1f, 0x8b];
const GHW_BZIP2_HEADER: &[u8; 2] = b"BZ";

/// This is the header of the uncompressed file.
const GHW_HEADER_START: &[u8] = b"GHDLwave\n";

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
enum GhwWellKnownType {
    Unknown = 0,
    Boolean = 1,
    Bit = 2,
    StdULogic = 3,
}

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone, TryFromPrimitive)]
enum GhdlRtik {
    Top = 0,
    Library = 1,
    Package = 2,
    PackageBody = 3,
    Entity = 4,
    Architecture = 5,
    Process = 6,
    Block = 7,
    IfGenerate = 8,
    ForGenerate = 9,
    Instance = 10,
    Constant = 11,
    Iterator = 12,
    Variable = 13,
    Signal = 14,
    File = 15,
    Port = 16,
    Generic = 17,
    Alias = 18,
    Guard = 19,
    Component = 20,
    Attribute = 21,
    TypeB2 = 22,
    TypeE8 = 23,
    TypeE32 = 24,
    TypeI32 = 25,
    TypeI64 = 26,
    TypeF64 = 27,
    TypeP32 = 28,
    TypeP64 = 29,
    TypeAccess = 30,
    TypeArray = 31,
    TypeRecord = 32,
    TypeFile = 33,
    SubtypeScalar = 34,
    SubtypeArray = 35,
    // obsolete array pointer = 36
    SubtypeUnboundedArray = 37,
    SubtypeRecord = 38,
    SubtypeUnboundedRecord = 39,
    SubtypeAccess = 40,
    TypeProtected = 41,
    Element = 42,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_load() {
        load("inputs/ghdl/tb_recv.ghw").unwrap();
        todo!()
    }
}
