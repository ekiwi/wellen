// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use num_enum::TryFromPrimitive;
use std::io::BufRead;
use std::num::NonZeroU32;
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
    #[error("[ghw] unexpected type: {0:?}, {1}")]
    UnexpectedType(GhdlRtik, &'static str),
    #[error("[ghw] failed to parse a {0} section: {1}")]
    FailedToParseSection(&'static str, String),
    #[error("[ghw] expected positive integer, not: {0}")]
    ExpectedPositiveInteger(i64),
    #[error("[ghw] failed to parse GHDL RTIK.")]
    FailedToParseGhdlRtik(#[from] num_enum::TryFromPrimitiveError<GhdlRtik>),
    #[error("[ghw] failed to parse well known type.")]
    FailedToParseWellKnownType(#[from] num_enum::TryFromPrimitiveError<GhwWellKnownType>),
    #[error("[ghw] failed to parse hierarchy kind.")]
    FailedToParseHierarchyKind(#[from] num_enum::TryFromPrimitiveError<GhwHierarchyKind>),
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
            Section::WellKnownTypes(wkts) => {
                debug_assert!(wkts.is_empty() || !type_table.is_empty());
                println!("TODO: {wkts:?}");
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
        GHW_HIERARCHY_SECTION => {
            read_hierarchy_section(header, input)?;
            todo!()
        }
        GHW_TYPE_SECTION => Ok(Some(Section::TypeTable(read_type_section(header, input)?))),
        GHW_WK_TYPE_SECTION => Ok(Some(Section::WellKnownTypes(
            read_well_known_types_section(input)?,
        ))),
        GHW_END_SECTION => Ok(None),
        other => Err(GhwParseError::UnexpectedSection(
            String::from_utf8_lossy(other).to_string(),
        )),
    }
}

enum Section {
    StringTable(Vec<String>),
    TypeTable(Vec<GhwTypeInfo>),
    WellKnownTypes(Vec<(TypeId, GhwWellKnownType)>),
}

#[inline]
fn read_u8(input: &mut impl BufRead) -> Result<u8> {
    let mut buf = [0u8];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn check_header_zeros(section: &'static str, header: &[u8]) -> Result<()> {
    if header.len() < 4 {
        return Err(GhwParseError::FailedToParseSection(
            section,
            "first four bytes should be zero".to_string(),
        ));
    }
    let zeros = &header[..4];
    if zeros == b"\x00\x00\x00\x00" {
        Ok(())
    } else {
        Err(GhwParseError::FailedToParseSection(
            section,
            "first four bytes should be zero".to_string(),
        ))
    }
}

fn read_string_section(header: &HeaderData, input: &mut impl BufRead) -> Result<Vec<String>> {
    let mut h = [0u8; 12];
    input.read_exact(&mut h)?;
    check_header_zeros("string", &h)?;

    let string_num = header.read_u32(&mut &h[4..8])? + 1;
    let _string_size = header.read_i32(&mut &h[8..12])? as u32;

    let mut string_table = Vec::with_capacity(string_num as usize);
    string_table.push("<anon>".to_string());

    let mut buf = Vec::with_capacity(64);

    for _ in 1..(string_num + 1) {
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

fn read_type_id(input: &mut impl BufRead) -> Result<TypeId> {
    let value = leb128::read::unsigned(input)?;
    Ok(TypeId(NonZeroU32::new(value as u32).unwrap()))
}

fn read_range(input: &mut impl BufRead) -> Result<GhwRange> {
    let t = read_u8(input)? & 0x7f;
    let kind = GhdlRtik::try_from_primitive(t)?;
    let to = (t & 0x80) != 0;
    let range = match kind {
        GhdlRtik::TypeE8 | GhdlRtik::TypeB2 => {
            let mut buf = [0u8; 2];
            input.read_exact(&mut buf)?;
            Range::U8(buf[0], buf[1])
        }
        GhdlRtik::TypeI32 | GhdlRtik::TypeP32 | GhdlRtik::TypeI64 | GhdlRtik::TypeP64 => {
            let left = leb128::read::signed(input)?;
            let right = leb128::read::signed(input)?;
            Range::I64(left, right)
        }
        GhdlRtik::TypeF64 => {
            todo!("float range!")
        }
        other => return Err(GhwParseError::UnexpectedType(other, "for range")),
    };

    Ok(GhwRange { kind, to, range })
}

fn read_type_section(header: &HeaderData, input: &mut impl BufRead) -> Result<Vec<GhwTypeInfo>> {
    let mut h = [0u8; 8];
    input.read_exact(&mut h)?;
    check_header_zeros("type", &h)?;

    let type_num = header.read_u32(&mut &h[4..8])?;
    let mut table: Vec<GhwTypeInfo> = Vec::with_capacity(type_num as usize);

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
            GhdlRtik::TypeI32 | GhdlRtik::TypeI64 | GhdlRtik::TypeF64 => GhwType::Scalar,
            GhdlRtik::SubtypeScalar => GhwType::SubtypeScalar {
                base: read_type_id(input)?,
                range: read_range(input)?,
            },
            GhdlRtik::TypeArray => {
                let element_tpe = read_type_id(input)?;
                let num_dims = leb128::read::unsigned(input)?;
                let mut dims = Vec::with_capacity(num_dims as usize);
                for _ in 0..num_dims {
                    dims.push(read_type_id(input)?);
                }
                GhwType::TypeArray { element_tpe, dims }
            }
            GhdlRtik::SubtypeArray => {
                let base = read_type_id(input)?;
                let base_tpe_id = table[base.index()].get_base_type_id(base);
                let array = &table[base_tpe_id.index()];
                if let GhwType::TypeArray { element_tpe, dims } = &array.tpe {
                    // one range per array dimension
                    let mut ranges = Vec::with_capacity(dims.len());
                    for _ in 0..dims.len() {
                        ranges.push(read_range(input)?);
                    }

                    let num_elements = table[element_tpe.index()].get_num_elements();
                    let element_tpe = match num_elements {
                        // for bounded number of elements, we just use the array element type
                        Some(_) => *element_tpe,
                        // for unbounded, we need to derive it
                        None => todo!("deal with unbounded!"),
                    };

                    GhwType::SubtypeArray {
                        base,
                        ranges,
                        element_tpe,
                    }
                } else {
                    return Err(GhwParseError::UnexpectedType(
                        array.kind,
                        "subtype array needs base to be an array!",
                    ));
                }
            }
            other => todo!("Support: {other:?}"),
        };
        let info = GhwTypeInfo { kind, name, tpe };
        table.push(info);
    }

    // the type section should end in zero
    if read_u8(input)? != 0 {
        Err(GhwParseError::FailedToParseSection(
            "type",
            "last byte should be 0".to_string(),
        ))
    } else {
        Ok(table)
    }
}

fn read_well_known_types_section(
    input: &mut impl BufRead,
) -> Result<Vec<(TypeId, GhwWellKnownType)>> {
    let mut h = [0u8; 4];
    input.read_exact(&mut h)?;
    check_header_zeros("well known types (WKT)", &h)?;

    let mut out = Vec::new();
    let mut t = read_u8(input)?;
    while t > 0 {
        let wkt = GhwWellKnownType::try_from_primitive(t)?;
        let type_id = read_type_id(input)?;
        out.push((type_id, wkt));
        t = read_u8(input)?;
    }

    Ok(out)
}

fn read_hierarchy_section(header: &HeaderData, input: &mut impl BufRead) -> Result<()> {
    let mut h = [0u8; 16];
    input.read_exact(&mut h)?;
    check_header_zeros("hierarchy", &h)?;

    let num_scopes = header.read_i32(&mut &h[4..8])?;
    // declared signals, may be composite
    let num_declared_vars = header.read_i32(&mut &h[8..12])?;
    let num_basic_vars = header.read_u32(&mut &h[12..16])?;

    loop {
        let kind = GhwHierarchyKind::try_from_primitive(read_u8(input)?)?;

        match kind {
            GhwHierarchyKind::End => break, // done
            GhwHierarchyKind::EndOfScope => {
                println!("UpScope");
            }
            _ => {
                let name = read_string_id(input)?;
                match kind {
                    GhwHierarchyKind::End
                    | GhwHierarchyKind::EndOfScope
                    | GhwHierarchyKind::Design => unreachable!(),
                    GhwHierarchyKind::Process => {
                        println!("TODO: process");
                    }
                    GhwHierarchyKind::Block
                    | GhwHierarchyKind::GenerateIf
                    | GhwHierarchyKind::GenerateFor
                    | GhwHierarchyKind::Instance
                    | GhwHierarchyKind::Generic
                    | GhwHierarchyKind::Package => {
                        todo!("scope")
                    }
                    GhwHierarchyKind::Signal
                    | GhwHierarchyKind::PortIn
                    | GhwHierarchyKind::PortOut
                    | GhwHierarchyKind::PortInOut
                    | GhwHierarchyKind::Buffer
                    | GhwHierarchyKind::Linkage => {
                        todo!("var")
                    }
                }
            }
        }
    }

    todo!()
}

/// Pointer into the GHW string table.
#[derive(Debug, Copy, Clone, PartialEq)]
struct StringId(usize);
/// Pointer into the GHW type table. Always positive!
#[derive(Debug, Copy, Clone, PartialEq)]
struct TypeId(NonZeroU32);

impl TypeId {
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// ???
#[derive(Debug, Copy, Clone, PartialEq)]
struct RangeId(usize);

#[derive(Debug)]
struct GhwTypeInfo {
    kind: GhdlRtik,
    name: StringId,
    tpe: GhwType,
}

impl GhwTypeInfo {
    fn get_base_type_id(&self, id: TypeId) -> TypeId {
        match self.tpe {
            GhwType::SubtypeScalar { base, .. } => base,
            GhwType::SubtypeArray { base, .. } => base,
            GhwType::SubtypeUnboundedArray { base, .. } => base,
            _ => id, // return our own id
        }
    }

    /// Returns `None` when the number is unbounded.
    fn get_num_elements(&self) -> Option<u32> {
        match self.tpe {
            GhwType::Array { .. }
            | GhwType::UnboundedArray { .. }
            | GhwType::UnboundedRecord { .. } => None,
            GhwType::SubtypeArray { .. } => todo!(""),
            GhwType::TypeRecord { .. } => todo!(""),
            GhwType::SubtypeRecord { .. } => todo!(""),
            _ => Some(1),
        }
    }
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
    UnboundedRecord {
        // TODO
    },
    SubtypeScalar {
        base: TypeId,
        range: GhwRange,
    },
    TypeArray {
        element_tpe: TypeId,
        dims: Vec<TypeId>,
    },
    SubtypeArray {
        base: TypeId,
        ranges: Vec<GhwRange>,
        element_tpe: TypeId,
    },
    SubtypeUnboundedArray {
        base: TypeId,
        // TODO
    },
    TypeRecord {
        // TODO
    },
    SubtypeRecord {
        // TODO
    },
    // TODO
}

#[derive(Debug)]
struct GhwRange {
    kind: GhdlRtik,
    /// `downto` if `false`
    to: bool,
    range: Range,
}

#[derive(Debug)]
enum Range {
    /// `b2` and `e8`
    U8(u8, u8),
    /// `i32` and `i64`
    I64(i64, i64),
    /// `f64`
    F64(f64, f64),
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
#[derive(Debug, PartialEq, Copy, Clone, TryFromPrimitive)]
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

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone, TryFromPrimitive)]
enum GhwHierarchyKind {
    /// indicates the end of the hierarchy
    End = 0,
    Design = 1,
    Block = 3,
    GenerateIf = 4,
    GenerateFor = 5,
    Instance = 6,
    Package = 7,
    Process = 13,
    Generic = 14,
    EndOfScope = 15,
    Signal = 16,
    PortIn = 17,
    PortOut = 18,
    PortInOut = 19,
    Buffer = 20,
    Linkage = 21,
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
