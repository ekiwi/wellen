// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::ghw::common::*;
use crate::hierarchy::HierarchyBuilder;
use crate::{
    FileFormat, Hierarchy, ScopeType, SignalRef, Timescale, TimescaleUnit, VarDirection, VarIndex,
    VarType,
};
use num_enum::TryFromPrimitive;
use std::io::{BufRead, Seek, SeekFrom};
use std::num::NonZeroU32;

pub(crate) fn read_ghw_header(input: &mut impl BufRead) -> Result<HeaderData> {
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
        return Err(GhwParseError::UnexpectedHeader(format!("{data:?}")));
    }

    if data.version > 1 {
        return Err(GhwParseError::UnexpectedHeader(format!("{data:?}")));
    }

    if h[3] != 1 && h[3] != 2 {
        return Err(GhwParseError::UnexpectedHeader(format!("{data:?}")));
    }

    if h[6] != 0 {
        return Err(GhwParseError::UnexpectedHeader(format!("{data:?}")));
    }

    Ok(data)
}

const GHW_TAILER_LEN: usize = 12;

/// The last 8 bytes of a finished, uncompressed file indicate where to find the directory which
/// contains the offset of all sections.
pub(crate) fn try_read_directory(
    header: &HeaderData,
    input: &mut (impl BufRead + Seek),
) -> Result<Option<Vec<SectionPos>>> {
    if input.seek(SeekFrom::End(-(GHW_TAILER_LEN as i64))).is_err() {
        // we treat a failure to seek as not being able to find the directory
        Ok(None)
    } else {
        let mut tailer = [0u8; GHW_TAILER_LEN];
        input.read_exact(&mut tailer)?;

        // check section start
        if &tailer[0..4] != GHW_TAILER_SECTION {
            Ok(None)
        } else {
            // note: the "tailer" section does not contain the normal 4 zeros
            let directory_offset = header.read_u32(&mut &tailer[8..12])?;
            input.seek(SeekFrom::Start(directory_offset as u64))?;

            // check directory marker
            let mut mark = [0u8; 4];
            input.read_exact(&mut mark)?;
            if &mark != GHW_DIRECTORY_SECTION {
                Err(GhwParseError::UnexpectedSection(
                    String::from_utf8_lossy(&mark).to_string(),
                ))
            } else {
                Ok(Some(read_directory(header, input)?))
            }
        }
    }
}

/// Parses the beginning of the GHW file until the end of the hierarchy.
pub(crate) fn read_hierarchy(
    header: &HeaderData,
    input: &mut impl BufRead,
) -> Result<(GhwDecodeInfo, usize, Hierarchy)> {
    let mut tables = GhwTables::default();
    let mut decode = GhwDecodeInfo::default();
    let mut hb = HierarchyBuilder::new(FileFormat::Ghw);
    let mut signal_ref_count = 0;

    // GHW seems to always uses fs
    hb.set_timescale(Timescale::new(1, TimescaleUnit::FemtoSeconds));

    loop {
        let mut mark = [0u8; 4];
        input.read_exact(&mut mark)?;

        match &mark {
            GHW_STRING_SECTION => {
                let table = read_string_section(header, input)?;
                debug_assert!(
                    tables.strings.is_empty(),
                    "unexpected second string table:\n{:?}\n{:?}",
                    &tables.strings,
                    &table
                );
                tables.strings = table;
            }
            GHW_TYPE_SECTION => {
                debug_assert!(tables.types.is_empty(), "unexpected second type table");
                read_type_section(header, &mut tables, input)?;
            }
            GHW_WK_TYPE_SECTION => {
                let wkts = read_well_known_types_section(input)?;
                debug_assert!(wkts.is_empty() || !tables.types.is_empty());

                // we should have already inferred the correct well know types, so we just check
                // that we did so correctly
                for (type_id, wkt) in wkts.into_iter() {
                    let tpe = &tables.types[type_id.index()];
                    match wkt {
                        GhwWellKnownType::Unknown => {} // does not matter
                        GhwWellKnownType::Boolean => todo!("add bool"),
                        GhwWellKnownType::Bit => todo!("add bit"),
                        GhwWellKnownType::StdULogic => {
                            debug_assert!(
                                matches!(tpe, VhdlType::NineValueBit(_)),
                                "{tpe:?} not recognized a std_ulogic!"
                            );
                        }
                    }
                }
            }
            GHW_HIERARCHY_SECTION => {
                let (dec, ref_count) = read_hierarchy_section(header, &mut tables, input, &mut hb)?;
                debug_assert!(
                    decode.signals.is_empty(),
                    "unexpected second hierarchy section:\n{:?}\n{:?}",
                    decode,
                    dec
                );
                decode = dec;
                signal_ref_count = ref_count;
            }
            GHW_END_OF_HEADER_SECTION => {
                break; // done
            }
            other => {
                return Err(GhwParseError::UnexpectedSection(
                    String::from_utf8_lossy(other).to_string(),
                ))
            }
        }
    }
    let hierarchy = hb.finish();
    Ok((decode, signal_ref_count, hierarchy))
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
        let mut c;
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

fn read_range(input: &mut impl BufRead) -> Result<Range> {
    let t = read_u8(input)?;
    let kind = GhwRtik::try_from_primitive(t & 0x7f)?;
    let dir = if (t & 0x80) != 0 {
        RangeDir::Downto
    } else {
        RangeDir::To
    };
    let range = match kind {
        GhwRtik::TypeE8 | GhwRtik::TypeB2 => {
            let mut buf = [0u8; 2];
            input.read_exact(&mut buf)?;
            Range::Int(IntRange(dir, buf[0] as i64, buf[1] as i64))
        }
        GhwRtik::TypeI32 | GhwRtik::TypeP32 | GhwRtik::TypeI64 | GhwRtik::TypeP64 => {
            let left = leb128::read::signed(input)?;
            let right = leb128::read::signed(input)?;
            Range::Int(IntRange(dir, left, right))
        }
        GhwRtik::TypeF64 => {
            todo!("float range!")
        }
        other => {
            return Err(GhwParseError::UnexpectedType(
                format!("{other:?}"),
                "for range",
            ))
        }
    };

    Ok(range)
}

fn read_type_section(
    header: &HeaderData,
    tables: &mut GhwTables,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut h = [0u8; 8];
    input.read_exact(&mut h)?;
    check_header_zeros("type", &h)?;

    let type_num = header.read_u32(&mut &h[4..8])?;
    tables.types = Vec::with_capacity(type_num as usize);

    for _ in 0..type_num {
        let t = read_u8(input)?;
        let kind = GhwRtik::try_from_primitive(t)?;
        let name = read_string_id(input)?;
        let tpe: VhdlType = match kind {
            GhwRtik::TypeE8 | GhwRtik::TypeB2 => {
                let num_literals = leb128::read::unsigned(input)?;
                let mut literals = Vec::with_capacity(num_literals as usize);
                for _ in 0..num_literals {
                    literals.push(read_string_id(input)?);
                }
                VhdlType::from_enum(tables, name, literals)
            }
            GhwRtik::TypeI32 => VhdlType::I32(name, None),
            GhwRtik::TypeI64 => VhdlType::I64(name, None),
            GhwRtik::TypeF64 => VhdlType::F64(name, None),
            GhwRtik::SubtypeScalar => {
                let base = read_type_id(input)?;
                let range = read_range(input)?;
                VhdlType::from_subtype_scalar(name, &tables.types, base, range)
            }
            GhwRtik::TypeArray => {
                let element_tpe = read_type_id(input)?;
                let num_dims = leb128::read::unsigned(input)?;
                let mut dims = Vec::with_capacity(num_dims as usize);
                for _ in 0..num_dims {
                    dims.push(read_type_id(input)?);
                }
                debug_assert!(!dims.is_empty());
                if dims.len() > 1 {
                    todo!("support multi-dimensional arrays!")
                }
                VhdlType::from_array(name, &tables.types, element_tpe, dims[0])
            }
            GhwRtik::SubtypeArray => {
                let base = read_type_id(input)?;
                let range = read_range(input)?;
                VhdlType::from_subtype_array(name, &tables.types, base, range)
            }
            GhwRtik::TypeRecord => {
                let num_fields = leb128::read::unsigned(input)?;
                let mut fields = Vec::with_capacity(num_fields as usize);
                for _ in 0..num_fields {
                    let field_name = read_string_id(input)?;
                    // important: we do not want to resolve aliases here!
                    let field_tpe = read_type_id(input)?;
                    fields.push((field_name, field_tpe));
                }
                VhdlType::from_record(name, fields)
            }
            other => todo!("Support: {other:?}"),
        };
        tables.types.push(tpe);
    }

    // the type section should end in zero
    if read_u8(input)? != 0 {
        Err(GhwParseError::FailedToParseSection(
            "type",
            "last byte should be 0".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Our own custom representation of VHDL Types.
/// During GHW parsing we convert the GHDL types to our own representation.
#[derive(Debug)]
enum VhdlType {
    /// std_logic or std_ulogic with lut to convert to the `wellen` 9-value representation.
    NineValueBit(StringId),
    /// std_logic_vector or std_ulogic_vector with lut to convert to the `wellen` 9-value representation.
    NineValueVec(StringId, IntRange),
    /// Type alias that does not restrict the underlying type in any way.
    TypeAlias(StringId, TypeId),
    /// Integer type with possible upper and lower bounds.
    I32(StringId, Option<IntRange>),
    /// Integer type with possible upper and lower bounds.
    I64(StringId, Option<IntRange>),
    /// Float type with possible upper and lower bounds.
    F64(StringId, Option<FloatRange>),
    /// Record with fields.
    Record(StringId, Vec<(StringId, TypeId)>),
    /// An enum that was not detected to be a 9-value bit.
    Enum(StringId, Vec<StringId>),
    /// Array
    Array(StringId, TypeId, Option<IntRange>),
}

/// resolves 1 layer of type aliases
fn lookup_concrete_type(types: &[VhdlType], type_id: TypeId) -> &VhdlType {
    match &types[type_id.index()] {
        VhdlType::TypeAlias(_, base_id) => {
            debug_assert!(!matches!(
                &types[base_id.index()],
                VhdlType::TypeAlias(_, _)
            ));
            &types[base_id.index()]
        }
        other => other,
    }
}

/// resolves 1 layer of type aliases
fn lookup_concrete_type_id(types: &[VhdlType], type_id: TypeId) -> TypeId {
    match &types[type_id.index()] {
        VhdlType::TypeAlias(_name, base_id) => {
            debug_assert!(!matches!(
                &types[base_id.index()],
                VhdlType::TypeAlias(_, _)
            ));
            *base_id
        }
        _ => type_id,
    }
}

impl VhdlType {
    fn from_enum(tables: &mut GhwTables, name: StringId, literals: Vec<StringId>) -> Self {
        if let Some(nine_value) = try_parse_nine_value_bit(tables, name, &literals) {
            nine_value
        } else {
            VhdlType::Enum(name, literals)
        }
    }

    fn from_array(name: StringId, types: &[VhdlType], element_tpe: TypeId, index: TypeId) -> Self {
        let element_tpe_id = lookup_concrete_type_id(types, element_tpe);
        let index_type = lookup_concrete_type(types, index);
        let index_range = index_type.int_range();
        if let (VhdlType::NineValueBit(_), Some(range)) =
            (&types[element_tpe_id.index()], index_range)
        {
            VhdlType::NineValueVec(name, range)
        } else {
            VhdlType::Array(name, element_tpe_id, index_range)
        }
    }

    fn from_record(name: StringId, fields: Vec<(StringId, TypeId)>) -> Self {
        VhdlType::Record(name, fields)
    }

    fn from_subtype_array(name: StringId, types: &[VhdlType], base: TypeId, range: Range) -> Self {
        let base_tpe = lookup_concrete_type(types, base);
        match (base_tpe, range) {
            (VhdlType::Array(base_name, element_tpe, maybe_base_range), Range::Int(int_range)) => {
                todo!(
                    "{name:?} of {base_name:?} : {:?} {maybe_base_range:?} {int_range:?}",
                    lookup_concrete_type(types, *element_tpe)
                )
            }
            (VhdlType::NineValueVec(base_name, base_range), Range::Int(int_range)) => {
                debug_assert!(
                    int_range.is_subset_of(&base_range),
                    "{int_range:?} {base_range:?}"
                );
                VhdlType::NineValueVec(pick_best_name(name, *base_name), int_range)
            }
            other => todo!("Currently unsupported combination: {other:?}"),
        }
    }

    fn from_subtype_scalar(name: StringId, types: &[VhdlType], base: TypeId, range: Range) -> Self {
        let base_tpe = lookup_concrete_type(types, base);
        match (base_tpe, range) {
            (VhdlType::Enum(_, lits), Range::Int(int_range)) => {
                let range = int_range.range();
                debug_assert!(range.start >= 0 && range.start <= lits.len() as i64);
                debug_assert!(range.end >= 0 && range.end <= lits.len() as i64);
                // check to see if this is just an alias or if we need to create a new enum
                if range.start == 0 && range.end == lits.len() as i64 {
                    VhdlType::TypeAlias(name, base)
                } else {
                    todo!("actual sub enum!")
                }
            }
            (VhdlType::NineValueBit(_), Range::Int(int_range)) => {
                let range = int_range.range();
                if range.start == 0 && range.end == 9 {
                    VhdlType::TypeAlias(name, base)
                } else {
                    todo!("actual sub enum!")
                }
            }
            (VhdlType::I32(_, maybe_base_range), Range::Int(int_range)) => {
                let base_range = IntRange::from_i32_option(*maybe_base_range);
                debug_assert!(
                    int_range.is_subset_of(&base_range),
                    "{int_range:?} {base_range:?}"
                );
                VhdlType::I32(name, Some(int_range))
            }
            other => todo!("Currently unsupported combination: {other:?}"),
        }
    }

    fn name(&self) -> StringId {
        match self {
            VhdlType::NineValueBit(name) => *name,
            VhdlType::NineValueVec(name, _) => *name,
            VhdlType::TypeAlias(name, _) => *name,
            VhdlType::I32(name, _) => *name,
            VhdlType::I64(name, _) => *name,
            VhdlType::F64(name, _) => *name,
            VhdlType::Record(name, _) => *name,
            VhdlType::Enum(name, _) => *name,
            VhdlType::Array(name, _, _) => *name,
        }
    }

    fn int_range(&self) -> Option<IntRange> {
        match self {
            VhdlType::NineValueBit(_) => Some(IntRange(RangeDir::To, 0, 8)),
            VhdlType::I32(_, range) => *range,
            VhdlType::I64(_, range) => *range,
            VhdlType::Enum(_, lits) => Some(IntRange(RangeDir::To, 0, lits.len() as i64)),
            _ => None,
        }
    }
}

/// Returns Some(VhdlType::NineValueBit(..)) if the enum corresponds to a 9-value bit type.
fn try_parse_nine_value_bit(
    tables: &mut GhwTables,
    name: StringId,
    literals: &[StringId],
) -> Option<VhdlType> {
    if literals.len() != 9 {
        return None;
    }

    // check to see if it matches the standard std_logic enum
    for (lit_id, expected) in literals.iter().zip(STD_LOGIC_VALUES.iter()) {
        let lit = tables.get_str(*lit_id).as_bytes();
        let cc = match lit.len() {
            1 => lit[0],
            3 => lit[1], // this is to account for GHDL encoding things as '0' (including the tick!)
            _ => return None,
        };
        if !cc.eq_ignore_ascii_case(expected) {
            return None; // encoding does not match
        }
    }

    Some(VhdlType::NineValueBit(name))
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
    types: Vec<VhdlType>,
    strings: Vec<String>,
}

/// Returns a if it is a non-anonymous string, otherwise b.
fn pick_best_name(a: StringId, b: StringId) -> StringId {
    if a.0 == 0 {
        b
    } else {
        a
    }
}

impl GhwTables {
    fn get_type_and_name(&self, type_id: TypeId) -> (&VhdlType, &str) {
        let top_name = self.types[type_id.index()].name();
        let tpe = lookup_concrete_type(&self.types, type_id);
        let name = pick_best_name(top_name, tpe.name());
        (tpe, self.get_str(name))
    }

    fn get_str(&self, string_id: StringId) -> &str {
        &self.strings[string_id.0]
    }
}

fn read_hierarchy_section(
    header: &HeaderData,
    tables: &mut GhwTables,
    input: &mut impl BufRead,
    h: &mut HierarchyBuilder,
) -> Result<(GhwDecodeInfo, usize)> {
    let mut hdr = [0u8; 16];
    input.read_exact(&mut hdr)?;
    check_header_zeros("hierarchy", &hdr)?;

    // it appears that this number is actually not always 100% accurate
    let _expected_num_scopes = header.read_u32(&mut &hdr[4..8])?;
    // declared signals, may be composite
    let expected_num_declared_vars = header.read_u32(&mut &hdr[8..12])?;
    let max_signal_id = header.read_u32(&mut &hdr[12..16])? as usize;

    let mut num_declared_vars = 0;
    let mut signals: Vec<Option<GhwSignal>> = Vec::with_capacity(max_signal_id + 1);
    signals.resize(max_signal_id + 1, None);
    let mut signal_ref_count = 0;

    loop {
        let kind = GhwHierarchyKind::try_from_primitive(read_u8(input)?)?;

        match kind {
            GhwHierarchyKind::End => break, // done
            GhwHierarchyKind::EndOfScope => {
                h.pop_scope();
            }
            GhwHierarchyKind::Design => unreachable!(),
            GhwHierarchyKind::Process => {
                // for now we ignore processes since they seem to only add noise!
                let _process_name = read_string_id(input)?;
            }
            GhwHierarchyKind::Block
            | GhwHierarchyKind::GenerateIf
            | GhwHierarchyKind::GenerateFor
            | GhwHierarchyKind::Instance
            | GhwHierarchyKind::Generic
            | GhwHierarchyKind::Package => {
                read_hierarchy_scope(tables, input, kind, h)?;
            }
            GhwHierarchyKind::Signal
            | GhwHierarchyKind::PortIn
            | GhwHierarchyKind::PortOut
            | GhwHierarchyKind::PortInOut
            | GhwHierarchyKind::Buffer
            | GhwHierarchyKind::Linkage => {
                read_hierarchy_var(tables, input, kind, &mut signals, &mut signal_ref_count, h)?;
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

    let decode = GhwDecodeInfo {
        signals: signals.into_iter().flatten().collect::<Vec<_>>(),
    };

    Ok((decode, signal_ref_count))
}

fn read_hierarchy_scope(
    tables: &GhwTables,
    input: &mut impl BufRead,
    kind: GhwHierarchyKind,
    h: &mut HierarchyBuilder,
) -> Result<()> {
    let name = read_string_id(input)?;

    if kind == GhwHierarchyKind::GenerateFor {
        let _iter_type = read_type_id(input)?;
        todo!("read value");
    }

    h.add_scope(
        // TODO: this does not take advantage of the string duplication done in GHW
        tables.get_str(name).to_string(),
        None, // TODO: do we know, e.g., the name of a module if we have an instance?
        convert_scope_type(kind),
        None, // no source info in GHW
        None, // no source info in GHW
        false,
    );

    Ok(())
}

fn convert_scope_type(kind: GhwHierarchyKind) -> ScopeType {
    match kind {
        GhwHierarchyKind::Block => ScopeType::VhdlBlock,
        GhwHierarchyKind::GenerateIf => ScopeType::VhdlIfGenerate,
        GhwHierarchyKind::GenerateFor => ScopeType::VhdlForGenerate,
        GhwHierarchyKind::Instance => ScopeType::VhdlArchitecture,
        GhwHierarchyKind::Package => ScopeType::VhdlPackage,
        GhwHierarchyKind::Generic => ScopeType::GhwGeneric,
        GhwHierarchyKind::Process => ScopeType::VhdlProcess,
        other => {
            unreachable!("this kind ({other:?}) should have been handled by a different code path")
        }
    }
}

fn read_hierarchy_var(
    tables: &mut GhwTables,
    input: &mut impl BufRead,
    kind: GhwHierarchyKind,
    signals: &mut [Option<GhwSignal>],
    signal_ref_count: &mut usize,
    h: &mut HierarchyBuilder,
) -> Result<()> {
    let name_id = read_string_id(input)?;
    let name = tables.get_str(name_id).to_string();
    let tpe = read_type_id(input)?;
    add_var(tables, input, kind, signals, signal_ref_count, h, name, tpe)
}

/// Creates a new signal ref unless one is already available (because of aliasing!)
fn get_signal_ref(
    signals: &[Option<GhwSignal>],
    signal_ref_count: &mut usize,
    index: SignalId,
) -> SignalRef {
    match &signals[index.0.get() as usize] {
        None => {
            let signal_ref = SignalRef::from_index(*signal_ref_count).unwrap();
            *signal_ref_count += 1;
            signal_ref
        }
        Some(signal) => signal.signal_ref,
    }
}

fn add_var(
    tables: &GhwTables,
    input: &mut impl BufRead,
    kind: GhwHierarchyKind,
    signals: &mut [Option<GhwSignal>],
    signal_ref_count: &mut usize,
    h: &mut HierarchyBuilder,
    name: String,
    type_id: TypeId,
) -> Result<()> {
    let (vhdl_tpe, type_name) = tables.get_type_and_name(type_id);
    let dir = convert_kind_to_dir(kind);
    let tpe_name = type_name.to_string();
    match vhdl_tpe {
        VhdlType::Enum(_, literals) => {
            // TODO: add enum type lazily
            let mapping = literals
                .iter()
                .enumerate()
                .map(|(ii, lit)| (format!("{ii}"), tables.get_str(*lit).to_string()))
                .collect::<Vec<_>>();
            let enum_type = h.add_enum_type(tpe_name.clone(), mapping);
            let index = read_signal_id(input, signals)?;
            let signal_ref = get_signal_ref(signals, signal_ref_count, index);
            let bits = 1;
            h.add_var(
                name,
                VarType::Enum,
                dir,
                bits,
                None,
                signal_ref,
                Some(enum_type),
                Some(tpe_name),
            );
            // meta date for writing the signal later
            let tpe = SignalType::U8(8); // TODO: try to find the actual number of bits used
            signals[index.0.get() as usize] = Some(GhwSignal { signal_ref, tpe })
        }
        VhdlType::NineValueBit(_) => {
            let index = read_signal_id(input, signals)?;
            let signal_ref = get_signal_ref(signals, signal_ref_count, index);
            let var_type = match type_name.to_ascii_lowercase().as_str() {
                "std_ulogic" => VarType::StdULogic,
                "std_logic" => VarType::StdLogic,
                _ => VarType::Wire,
            };
            h.add_var(
                name,
                var_type,
                dir,
                1,
                None,
                signal_ref,
                None,
                Some(tpe_name),
            );
            // meta date for writing the signal later
            let tpe = SignalType::NineState;
            signals[index.0.get() as usize] = Some(GhwSignal { signal_ref, tpe })
        }
        VhdlType::NineValueVec(_, range) => {
            let num_bits = range.len() as u32;
            let mut signal_ids = Vec::with_capacity(num_bits as usize);
            for _ in 0..num_bits {
                signal_ids.push(read_signal_id(input, signals)?);
            }
            let signal_ref = get_signal_ref(signals, signal_ref_count, signal_ids[0]);
            let var_type = match type_name.to_ascii_lowercase().as_str() {
                "std_ulogic_vector" => VarType::StdULogicVector,
                "std_logic_vector" => VarType::StdLogicVector,
                _ => VarType::Wire,
            };
            h.add_var(
                name,
                var_type,
                dir,
                num_bits,
                Some(range.as_var_index()),
                signal_ref,
                None,
                Some(tpe_name),
            );
            // meta date for writing the signal later
            for (bit, index) in signal_ids.iter().rev().enumerate() {
                let tpe = SignalType::NineStateBit(bit as u32, num_bits);
                signals[index.0.get() as usize] = Some(GhwSignal { signal_ref, tpe })
            }
        }
        VhdlType::Record(_, fields) => {
            h.add_scope(name, None, ScopeType::VhdlRecord, None, None, false);
            for (field_name, field_type) in fields.iter() {
                add_var(
                    tables,
                    input,
                    kind,
                    signals,
                    signal_ref_count,
                    h,
                    tables.get_str(*field_name).to_string(),
                    *field_type,
                )?;
            }
            h.pop_scope();
        }

        other => todo!("deal with {other:?}"),
    }
    Ok(())
}

fn read_signal_id(input: &mut impl BufRead, signals: &mut [Option<GhwSignal>]) -> Result<SignalId> {
    let index = leb128::read::unsigned(input)? as usize;
    if index >= signals.len() {
        Err(GhwParseError::FailedToParseSection(
            "hierarchy",
            format!("SignalId too large {index} > {}", signals.len()),
        ))
    } else {
        let id = SignalId(NonZeroU32::new(index as u32).unwrap());
        Ok(id)
    }
}

fn convert_kind_to_dir(kind: GhwHierarchyKind) -> VarDirection {
    match kind {
        GhwHierarchyKind::Signal => VarDirection::Implicit,
        GhwHierarchyKind::PortIn => VarDirection::Input,
        GhwHierarchyKind::PortOut => VarDirection::Output,
        GhwHierarchyKind::PortInOut => VarDirection::InOut,
        GhwHierarchyKind::Buffer => VarDirection::Buffer,
        GhwHierarchyKind::Linkage => VarDirection::Linkage,
        other => {
            unreachable!("this kind ({other:?}) should have been handled by a different code path")
        }
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

#[derive(Debug)]
enum Range {
    Int(IntRange),
    #[allow(dead_code)]
    Float(FloatRange),
}

#[derive(Debug, Copy, Clone)]
enum RangeDir {
    To,
    Downto,
}

#[derive(Debug, Copy, Clone)]
struct IntRange(RangeDir, i64, i64);

impl IntRange {
    fn range(&self) -> std::ops::Range<i64> {
        match self.0 {
            RangeDir::To => self.1..(self.2 + 1),
            RangeDir::Downto => self.2..(self.1 + 1),
        }
    }

    fn len(&self) -> i64 {
        match self.0 {
            RangeDir::To => self.2 - self.1 + 1,
            RangeDir::Downto => self.1 - self.2 + 1,
        }
    }

    fn as_var_index(&self) -> VarIndex {
        let msb = self.1 as i32;
        let lsb = self.2 as i32;
        VarIndex::new(msb, lsb)
    }

    fn from_i32_option(opt: Option<Self>) -> Self {
        opt.unwrap_or(Self(RangeDir::To, i32::MIN as i64, i32::MAX as i64))
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        let self_range = self.range();
        let other_range = other.range();
        self_range.start >= other_range.start && self_range.end <= other_range.end
    }
}

#[derive(Debug, Copy, Clone)]
struct FloatRange(RangeDir, f64, f64);

#[derive(Debug, Copy, Clone, PartialEq)]
struct GhwUnit {
    name: StringId,
    value: i64,
}
