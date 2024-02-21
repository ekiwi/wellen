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
    #[error("[ghw] float range has no length: {0} .. {1}")]
    FloatRangeLen(f64, f64),
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

    let mut tables = GhwTables::default();

    while let Some(section) = read_section(&header, &tables, &mut input)? {
        match section {
            Section::StringTable(table) => {
                debug_assert!(
                    tables.strings.is_empty(),
                    "unexpected second string table:\n{:?}\n{:?}",
                    &tables.strings,
                    &table
                );
                tables.strings = table;
            }
            Section::TypeTable(table) => {
                debug_assert!(
                    tables.types.is_empty(),
                    "unexpected second type table:\n{:?}\n{:?}",
                    &tables.types,
                    &table
                );
                tables.types = table;
            }
            Section::WellKnownTypes(wkts) => {
                debug_assert!(wkts.is_empty() || !tables.types.is_empty());

                // we need to patch our type table with the well known types info
                for (type_id, wkt) in wkts.into_iter() {
                    let tpe = &mut tables.types[type_id.index()];
                    tpe.well_known = wkt;
                }
            }
            Section::Hierarchy => {}
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

fn read_section(
    header: &HeaderData,
    tables: &GhwTables,
    input: &mut impl BufRead,
) -> Result<Option<Section>> {
    let mut mark = [0u8; 4];
    input.read_exact(&mut mark)?;

    match &mark {
        GHW_STRING_SECTION => Ok(Some(Section::StringTable(read_string_section(
            header, input,
        )?))),
        GHW_HIERARCHY_SECTION => {
            read_hierarchy_section(header, tables, input)?;
            println!("TODO: actually do something with the hierarchy!");
            Ok(Some(Section::Hierarchy))
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
    Hierarchy,
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
    let t = read_u8(input)?;
    let kind = GhdlRtik::try_from_primitive(t & 0x7f)?;
    let downto = (t & 0x80) != 0;
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

    Ok(GhwRange {
        kind,
        downto,
        range,
    })
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
                GhwType::Enum { literals }
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

                    let num_elements = table[element_tpe.index()].get_num_elements(&table)?;
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
        let info = GhwTypeInfo {
            kind,
            name,
            well_known: GhwWellKnownType::Unknown,
            tpe,
        };
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

#[derive(Debug, Default)]
struct GhwTables {
    types: Vec<GhwTypeInfo>,
    strings: Vec<String>,
}

impl GhwTables {
    fn get_type(&self, type_id: TypeId) -> &GhwTypeInfo {
        &self.types[type_id.index()]
    }

    fn get_str(&self, string_id: StringId) -> &str {
        &self.strings[string_id.0]
    }
}

fn read_hierarchy_section(
    header: &HeaderData,
    tables: &GhwTables,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut h = [0u8; 16];
    input.read_exact(&mut h)?;
    check_header_zeros("hierarchy", &h)?;

    // it appears that this number is actually not always 100% accurate
    let _expected_num_scopes = header.read_u32(&mut &h[4..8])?;
    // declared signals, may be composite
    let expected_num_declared_vars = header.read_u32(&mut &h[8..12])?;
    let max_signal_id = header.read_u32(&mut &h[12..16])?;

    let mut num_declared_vars = 0;

    loop {
        let kind = GhwHierarchyKind::try_from_primitive(read_u8(input)?)?;

        match kind {
            GhwHierarchyKind::End => break, // done
            GhwHierarchyKind::EndOfScope => {
                println!("UpScope");
            }
            GhwHierarchyKind::Design => unreachable!(),
            GhwHierarchyKind::Process => {
                read_hierarchy_scope(tables, input, kind)?;
                // processes are always empty, thus an "upscope" is implied
                println!("UpScope");
            }
            GhwHierarchyKind::Block
            | GhwHierarchyKind::GenerateIf
            | GhwHierarchyKind::GenerateFor
            | GhwHierarchyKind::Instance
            | GhwHierarchyKind::Generic
            | GhwHierarchyKind::Package => {
                read_hierarchy_scope(tables, input, kind)?;
            }
            GhwHierarchyKind::Signal
            | GhwHierarchyKind::PortIn
            | GhwHierarchyKind::PortOut
            | GhwHierarchyKind::PortInOut
            | GhwHierarchyKind::Buffer
            | GhwHierarchyKind::Linkage => {
                read_hierarchy_var(tables, input, kind, max_signal_id)?;
                num_declared_vars += 1;
                if num_declared_vars > expected_num_declared_vars {
                    return Err(GhwParseError::FailedToParseSection(
                        "hierarchy",
                        format!(
                            "more declared variables than expected {expected_num_declared_vars}"
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

fn read_hierarchy_scope(
    tables: &GhwTables,
    input: &mut impl BufRead,
    kind: GhwHierarchyKind,
) -> Result<()> {
    let name = read_string_id(input)?;

    if kind == GhwHierarchyKind::GenerateFor {
        let iter_type = read_type_id(input)?;
        todo!("read value");
    }

    println!("SCOPE {kind:?} {}", tables.get_str(name));

    Ok(())
}

fn read_hierarchy_var(
    tables: &GhwTables,
    input: &mut impl BufRead,
    kind: GhwHierarchyKind,
    max_signal_id: u32,
) -> Result<usize> {
    let name = read_string_id(input)?;
    let tpe = tables.get_type(read_type_id(input)?);

    if let Some(num_elements) = tpe.get_num_elements(&tables.types)? {
        let mut signals = Vec::with_capacity(num_elements as usize);
        for _ in 0..num_elements {
            let index = leb128::read::unsigned(input)? as u32;
            if index > max_signal_id {
                return Err(GhwParseError::FailedToParseSection(
                    "hierarchy",
                    format!("SignalId too large {index} > {max_signal_id}"),
                ));
            }
            signals.push(SignalId(NonZeroU32::new(index as u32).unwrap()));
        }
        println!(
            "TODO: {kind:?} {} : {}\n{:?}",
            tables.get_str(name),
            tables.get_str(tpe.name),
            signals
        );
        Ok(signals.len())
    } else {
        Err(GhwParseError::FailedToParseSection(
            "hierarchy",
            format!(
                "var {} has unbound number of elements!",
                tables.get_str(name)
            ),
        ))
    }
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

#[derive(Debug, Copy, Clone, PartialEq)]
struct SignalId(NonZeroU32);

/// ???
#[derive(Debug, Copy, Clone, PartialEq)]
struct RangeId(usize);

#[derive(Debug)]
struct GhwTypeInfo {
    kind: GhdlRtik,
    name: StringId,
    well_known: GhwWellKnownType,
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
    fn get_num_elements(&self, type_table: &[GhwTypeInfo]) -> Result<Option<u64>> {
        match &self.tpe {
            GhwType::Array { .. }
            | GhwType::UnboundedArray { .. }
            | GhwType::UnboundedRecord { .. } => Ok(None),
            GhwType::SubtypeArray {
                base,
                ranges,
                element_tpe,
            } => {
                let num_elements = &type_table[element_tpe.index()]
                    .get_num_elements(type_table)?
                    .unwrap_or_else(|| todo!());
                let mut num_scalars = 1;
                for r in ranges.iter() {
                    num_scalars *= r.get_len()?;
                }
                Ok(Some(num_elements * num_scalars))
            }
            GhwType::TypeRecord { .. } => todo!(""),
            GhwType::SubtypeRecord { .. } => todo!(""),
            _ => Ok(Some(1)),
        }
    }
}

#[derive(Debug)]
enum GhwType {
    Enum {
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
    /// `to` instead of `downto` if `false`
    downto: bool,
    range: Range,
}

impl GhwRange {
    fn get_len(&self) -> Result<u64> {
        let (left, right) = match self.range {
            Range::U8(left, right) => (left as i64, right as i64),
            Range::I64(left, right) => (left, right),
            Range::F64(left, right) => return Err(GhwParseError::FloatRangeLen(left, right)),
        };
        let res = if self.downto {
            left - right + 1
        } else {
            right - left + 1
        };
        Ok(std::cmp::max(res, 0) as u64)
    }
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
fn render_type(tpe: &GhwTypeInfo, tables: &GhwTables) -> String {
    if tpe.well_known != GhwWellKnownType::Unknown {
        return format!("{:?}", tpe.well_known);
    }
    match &tpe.tpe {
        GhwType::Enum { literals } => todo!("enum"),
        GhwType::Scalar => todo!(""),
        GhwType::Physical { .. } => todo!(""),
        GhwType::Array { .. } => todo!(""),
        GhwType::UnboundedArray { .. } => todo!(""),
        GhwType::UnboundedRecord { .. } => todo!(""),
        // TODO: what is the range for?
        GhwType::SubtypeScalar { base, range } => render_type(tables.get_type(*base), tables),
        GhwType::TypeArray { .. } => todo!(""),
        GhwType::SubtypeArray { .. } => todo!(""),
        GhwType::SubtypeUnboundedArray { .. } => todo!(""),
        GhwType::TypeRecord { .. } => todo!(""),
        GhwType::SubtypeRecord { .. } => todo!(""),
    }
}

#[cfg(test)]
fn render_range(range: &GhwRange) -> String {
    let middle = if range.downto { "downto" } else { "to" };
    match &range.range {
        Range::U8(a, b) => format!("({a} {middle} {b})"),
        Range::I64(a, b) => format!("({a} {middle} {b})"),
        Range::F64(a, b) => format!("({a} {middle} {b})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_load() {
        load("inputs/ghdl/tb_recv.ghw").unwrap();
    }
}
