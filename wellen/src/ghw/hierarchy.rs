// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::ghw::common::*;
use crate::hierarchy::{EnumTypeId, HierarchyBuilder, HierarchyStringId};
use crate::{
    FileFormat, Hierarchy, ScopeType, SignalRef, Timescale, TimescaleUnit, VarDirection, VarIndex,
    VarType,
};
use num_enum::TryFromPrimitive;
use rustc_hash::FxHashMap;
use std::io::{BufRead, Seek, SeekFrom};
use std::num::NonZeroU32;

pub fn read_ghw_header(input: &mut impl BufRead) -> Result<HeaderData> {
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
    if magic_rest != GHW_HEADER_START[2..] {
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

/// The last 8 bytes of a finished, uncompressed file indicate where to find the directory which
/// contains the offset of all sections.
pub fn try_read_directory(
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
pub fn read_hierarchy(
    header: &HeaderData,
    input: &mut impl BufRead,
) -> Result<(GhwDecodeInfo, Hierarchy)> {
    let mut tables = GhwTables::default();
    let mut strings = Vec::new();
    let mut decode: Option<GhwDecodeInfo> = None;
    let mut hb = HierarchyBuilder::new(FileFormat::Ghw);

    // GHW seems to always uses fs
    hb.set_timescale(Timescale::new(1, TimescaleUnit::FemtoSeconds));

    loop {
        let mut mark = [0u8; 4];
        input.read_exact(&mut mark)?;

        match &mark {
            GHW_STRING_SECTION => {
                let table = read_string_section(header, input)?;
                debug_assert!(
                    strings.is_empty(),
                    "unexpected second string table:\n{:?}\n{:?}",
                    &strings,
                    &table
                );
                strings = table;
                // add all strings to the hierarchy
                tables.strings = strings.iter().cloned().map(|s| hb.add_string(s)).collect();
            }
            GHW_TYPE_SECTION => {
                debug_assert!(tables.types.is_empty(), "unexpected second type table");
                let types = read_type_section(header, &strings, input)?;
                tables.types = types;

                // add enums to the table
                tables.enums = add_enums_to_wellen_hierarchy(&tables, &mut hb)?;
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
                        GhwWellKnownType::Boolean => {
                            // we expect the bool to be represented as an enum
                            debug_assert!(
                                matches!(tpe, VhdlType::Enum(_, _, _)),
                                "{tpe:?} not recognized a VHDL bool!"
                            );
                        }
                        GhwWellKnownType::Bit => {
                            debug_assert!(
                                matches!(tpe, VhdlType::Bit(_)),
                                "{tpe:?} not recognized a VHDL bit!"
                            );
                        }
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
                let dec = read_hierarchy_section(header, &mut tables, input, &mut hb)?;
                debug_assert!(
                    decode.is_none(),
                    "unexpected second hierarchy section:\n{:?}\n{:?}",
                    decode,
                    dec
                );
                decode = Some(dec);
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
    Ok((decode.unwrap(), hierarchy))
}

/// adds all enums to the hierarchy and
fn add_enums_to_wellen_hierarchy(
    tables: &GhwTables,
    hb: &mut HierarchyBuilder,
) -> Result<Vec<EnumTypeId>> {
    let mut string_cache: FxHashMap<(u16, u16), HierarchyStringId> = FxHashMap::default();
    let mut out = Vec::new();
    for tpe in tables.types.iter() {
        if let VhdlType::Enum(name, lits, enum_id) = tpe {
            let bits = get_enum_bits(lits) as u16;
            let literals: Vec<_> = lits
                .iter()
                .enumerate()
                .map(|(ii, str_id)| {
                    // generate a string representation of the enum encoding
                    let key = (bits, ii as u16);
                    let index_str_id = match string_cache.get(&key) {
                        Some(id) => *id,
                        None => {
                            let name = format!("{ii:0bits$b}", bits = bits as usize);
                            let id = hb.add_string(name);
                            string_cache.insert(key, id);
                            id
                        }
                    };
                    (index_str_id, tables.strings[str_id.0])
                })
                .collect();
            let enum_type_id = hb.add_enum_type(tables.strings[name.0], literals);
            assert_eq!(out.len(), (*enum_id) as usize);
            out.push(enum_type_id)
        }
    }
    Ok(out)
}

fn get_enum_bits(literals: &[StringId]) -> u32 {
    let max_value = literals.len() as u64 - 1;

    u64::BITS - max_value.leading_zeros()
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
            if c <= 31 || (128..=159).contains(&c) {
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
            let left = read_f64_le(input)?;
            let right = read_f64_le(input)?;
            Range::Float(FloatRange(dir, left, right))
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

fn read_unit_vec(input: &mut impl BufRead) -> Result<Vec<(StringId, i64)>> {
    let num_units = leb128::read::unsigned(input)?;
    let mut units = Vec::with_capacity(num_units as usize);
    for _ in 0..num_units {
        let name = read_string_id(input)?;
        let val = leb128::read::signed(input)?; // Read an lsleb, not sure relation to sleb
        units.push((name, val));
    }
    Ok(units)
}

fn read_type_section(
    header: &HeaderData,
    strings: &[String],
    input: &mut impl BufRead,
) -> Result<Vec<VhdlType>> {
    let mut h = [0u8; 8];
    input.read_exact(&mut h)?;
    check_header_zeros("type", &h)?;

    let type_num = header.read_u32(&mut &h[4..8])?;
    let mut types = vec![VhdlType::Missing; type_num as usize];
    let mut enum_count = 0;

    for new_tpe_id in 0..type_num {
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
                VhdlType::from_enum(strings, &mut enum_count, name, literals)
            }
            GhwRtik::TypeI32 => VhdlType::I32(name, None),
            GhwRtik::TypeI64 => VhdlType::I64(name, None),
            GhwRtik::TypeF64 => VhdlType::F64(name, None),
            GhwRtik::TypeP64 => {
                let units = match header.version {
                    0 => vec![],
                    _ => read_unit_vec(input)?,
                };
                VhdlType::P64(name, None, units)
            }
            GhwRtik::SubtypeScalar => {
                let base = read_type_id(input)?;
                let range = read_range(input)?;
                VhdlType::from_subtype_scalar(name, &types, base, range)
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
                VhdlType::from_array(name, &types, element_tpe, dims[0])
            }
            GhwRtik::SubtypeArray => {
                let base = read_type_id(input)?;
                debug_assert!(types[base.index()].is_array());
                // construct a finite version of the base type by recursively reading bounds
                read_subtype_bounds(input, &mut types, name, base)?
            }
            GhwRtik::SubtypeUnboundedArray => {
                let base = read_type_id(input)?;
                VhdlType::from_subtype_unbounded_array(name, &types, base)
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
            GhwRtik::SubtypeRecord => {
                // a subtype is a constraint version of the original type
                // like: https://stackoverflow.com/questions/61895716/what-is-the-point-of-a-subtype-when-a-type-can-be-constrained
                let base = read_type_id(input)?;
                debug_assert!(types[base.index()].is_record());
                // construct a finite version of the base type by recursively reading bounds
                read_subtype_bounds(input, &mut types, name, base)?
            }
            other => todo!("Support: {other:?}"),
        };
        debug_assert!(matches!(types[new_tpe_id as usize], VhdlType::Missing));
        types[new_tpe_id as usize] = tpe;
    }

    debug_assert!(
        types.iter().all(|t| !matches!(t, VhdlType::Missing)),
        "internal error: missing type!"
    );

    // the type section should end in zero
    if read_u8(input)? != 0 {
        Err(GhwParseError::FailedToParseSection(
            "type",
            "last byte should be 0".to_string(),
        ))
    } else {
        Ok(types)
    }
}

fn type_to_id(types: &mut Vec<VhdlType>, tpe: VhdlType) -> TypeId {
    types.push(tpe);
    TypeId::from_index(types.len() - 1)
}

/// Used to parse subtype array or subtype record entries which restrict an infinite subtype.
/// Returns a type_id if the type was already finite, a new type if the type had to be bound.
fn read_subtype_bounds(
    input: &mut impl BufRead,
    types: &mut Vec<VhdlType>,
    name: StringId,
    base: TypeId,
) -> Result<VhdlType> {
    match lookup_concrete_type(types.as_slice(), base).clone() {
        VhdlType::BitVec(_, _) | VhdlType::NineValueVec(_, _) => {
            // restrict array size along dimension zero (we only support 1D arrays at the moment)
            let range = read_range(input)?;
            // create new type with restricted range
            Ok(VhdlType::from_subtype_array(
                name, &types, base, range, None,
            ))
        }
        VhdlType::Array(_, el, _) => {
            // restrict array size along dimension zero (we only support 1D arrays at the moment)
            let range = read_range(input)?;
            // read in bounds iff the element type is unbound
            let element_tpe_is_finite = VhdlType::is_finite(el, types.as_slice());
            let sub_element_tpe = if element_tpe_is_finite {
                None
            } else {
                let restricted = read_subtype_bounds(input, types, StringId::none(), el)?;
                Some(type_to_id(types, restricted))
            };
            // create new type with restricted range and potentially a new, restricted element type
            let sub = VhdlType::from_subtype_array(name, &types, base, range, sub_element_tpe);
            Ok(sub)
        }
        VhdlType::Record(base_name, base_fields) => {
            // check to see if any of the base record fields need are missing a range
            let mut is_finite = true;
            let mut fields = Vec::with_capacity(base_fields.len());
            for (name, tpe_id) in base_fields {
                let is_field_finite = VhdlType::is_finite(tpe_id, types.as_slice());
                if !is_field_finite {
                    // the subtype is not an alias since it includes additional constraints
                    is_finite = false;
                    let constrained = read_subtype_bounds(input, types, StringId::none(), tpe_id)?;
                    fields.push((name, type_to_id(types, constrained)));
                } else {
                    fields.push((name, tpe_id));
                }
            }

            if is_finite {
                Ok(VhdlType::TypeAlias(pick_best_name(name, base_name), base))
            } else {
                Ok(VhdlType::Record(pick_best_name(name, base_name), fields))
            }
        }
        other => unreachable!(
            "This function is only expected to be called for arrays or records, not {other:?}"
        ),
    }
}

/// Our own custom representation of VHDL Types.
/// During GHW parsing we convert the GHDL types to our own representation.
#[derive(Debug, Clone)]
enum VhdlType {
    /// std_logic or std_ulogic
    NineValueBit(StringId),
    /// std_logic_vector or std_ulogic_vector
    NineValueVec(StringId, Option<IntRange>),
    /// bit (2-state)
    Bit(StringId),
    /// bit_vector (2-state)
    BitVec(StringId, Option<IntRange>),
    /// Type alias that does not restrict the underlying type in any way.
    TypeAlias(StringId, TypeId),
    /// Integer type with possible upper and lower bounds.
    I32(StringId, Option<IntRange>),
    /// Integer type with possible upper and lower bounds.
    I64(StringId, Option<IntRange>),
    /// Float type with possible upper and lower bounds.
    F64(StringId, Option<FloatRange>),
    /// Physical unit type, with upper and lower bounds.
    P64(StringId, Option<IntRange>, Vec<(StringId, i64)>),
    /// Record with fields.
    Record(StringId, Vec<(StringId, TypeId)>),
    /// An enum that was not detected to be a 9-value bit. The last entry is a unique ID, starting at 0.
    Enum(StringId, Vec<StringId>, u16),
    /// Array
    Array(StringId, TypeId, Option<IntRange>),
    /// A type that has not been read in yet. Should never be encountered.
    Missing,
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
    fn from_enum(
        strings: &[String],
        enum_count: &mut u16,
        name: StringId,
        literals: Vec<StringId>,
    ) -> Self {
        if let Some(nine_value) = try_parse_nine_value_bit(strings, name, &literals) {
            nine_value
        } else if let Some(two_value) = try_parse_two_value_bit(strings, name, &literals) {
            two_value
        } else {
            let e = VhdlType::Enum(name, literals, *enum_count);
            *enum_count += 1;
            e
        }
    }

    fn from_array(name: StringId, types: &[VhdlType], element_tpe: TypeId, index: TypeId) -> Self {
        let element_tpe_id = lookup_concrete_type_id(types, element_tpe);
        let index_type = lookup_concrete_type(types, index);
        let index_range = index_type.int_range();
        // this one wont be an alias!
        let concrete_element_type = &types[element_tpe_id.index()];
        match (concrete_element_type, index_range) {
            (VhdlType::NineValueBit(_), Some(range)) => VhdlType::NineValueVec(name, Some(range)),
            (VhdlType::Bit(_), Some(range)) => VhdlType::BitVec(name, Some(range)),
            _ => VhdlType::Array(name, element_tpe_id, index_range),
        }
    }

    fn from_record(name: StringId, fields: Vec<(StringId, TypeId)>) -> Self {
        VhdlType::Record(name, fields)
    }

    fn from_subtype_unbounded_array(name: StringId, types: &[VhdlType], base: TypeId) -> Self {
        // removes the range of the base array
        let base_tpe = lookup_concrete_type(types, base);
        match base_tpe {
            VhdlType::Array(_, element_tpe, _) => VhdlType::Array(name, *element_tpe, None),
            VhdlType::NineValueVec(_, _) => VhdlType::NineValueVec(name, None),
            VhdlType::BitVec(_, _) => VhdlType::BitVec(name, None),
            _ => panic!("unexpected base type {base_tpe:?}, expected array"),
        }
    }

    fn from_subtype_array(
        name: StringId,
        types: &[VhdlType],
        base: TypeId,
        range: Range,
        new_element_tpe: Option<TypeId>,
    ) -> Self {
        let base_tpe = lookup_concrete_type(types, base);
        match (base_tpe, range) {
            (VhdlType::Array(base_name, element_tpe, maybe_base_range), Range::Int(int_range)) => {
                if let Some(base_range) = maybe_base_range {
                    debug_assert!(
                        int_range.is_subset_of(base_range),
                        "{int_range:?} {base_range:?}"
                    );
                }
                VhdlType::Array(
                    pick_best_name(name, *base_name),
                    new_element_tpe.unwrap_or(*element_tpe),
                    Some(int_range),
                )
            }
            (VhdlType::NineValueVec(base_name, maybe_base_range), Range::Int(int_range)) => {
                if let Some(base_range) = maybe_base_range {
                    debug_assert!(
                        int_range.is_subset_of(base_range),
                        "{int_range:?} {base_range:?}"
                    );
                }
                debug_assert!(new_element_tpe.is_none());
                VhdlType::NineValueVec(pick_best_name(name, *base_name), Some(int_range))
            }
            (VhdlType::BitVec(base_name, maybe_base_range), Range::Int(int_range)) => {
                if let Some(base_range) = maybe_base_range {
                    debug_assert!(
                        int_range.is_subset_of(base_range),
                        "{int_range:?} {base_range:?}"
                    );
                }
                debug_assert!(new_element_tpe.is_none());
                VhdlType::BitVec(pick_best_name(name, *base_name), Some(int_range))
            }
            other => todo!("Currently unsupported combination: {other:?}"),
        }
    }

    fn from_subtype_scalar(name: StringId, types: &[VhdlType], base: TypeId, range: Range) -> Self {
        let base_tpe = lookup_concrete_type(types, base);
        match (base_tpe, range) {
            (VhdlType::Enum(_, lits, _), Range::Int(int_range)) => {
                debug_assert!(int_range.start() >= 0 && int_range.start() < lits.len() as i64);
                debug_assert!(int_range.end() >= 0 && int_range.end() < lits.len() as i64);
                // check to see if this is just an alias or if we need to create a new enum
                if int_range.start() == 0 && int_range.end() == (lits.len() - 1) as i64 {
                    VhdlType::TypeAlias(name, base)
                } else {
                    todo!("actual sub enum!")
                }
            }
            (VhdlType::NineValueBit(_), Range::Int(int_range)) => {
                if int_range.start() == 0 && int_range.end() == 8 {
                    VhdlType::TypeAlias(name, base)
                } else {
                    todo!("actual sub enum!")
                }
            }
            (VhdlType::P64(_, maybe_base_range, units), Range::Int(int_range)) => {
                let base_range = IntRange::from_i64_option(*maybe_base_range);
                debug_assert!(
                    int_range.is_subset_of(&base_range),
                    "{int_range:?} {base_range:?}"
                );
                VhdlType::P64(name, Some(int_range), units.clone())
            }
            (VhdlType::I32(_, maybe_base_range), Range::Int(int_range)) => {
                let base_range = IntRange::from_i32_option(*maybe_base_range);
                debug_assert!(
                    int_range.is_subset_of(&base_range),
                    "{int_range:?} {base_range:?}"
                );
                VhdlType::I32(name, Some(int_range))
            }
            (VhdlType::F64(_, maybe_base_range), Range::Float(float_range)) => {
                let base_range = FloatRange::from_f64_option(*maybe_base_range);
                debug_assert!(
                    float_range.is_subset_of(&base_range),
                    "{float_range:?} {base_range:?}"
                );
                VhdlType::F64(name, Some(float_range))
            }
            other => todo!("Currently unsupported combination: {other:?}"),
        }
    }

    fn name(&self) -> StringId {
        match self {
            VhdlType::NineValueBit(name) => *name,
            VhdlType::NineValueVec(name, _) => *name,
            VhdlType::Bit(name) => *name,
            VhdlType::BitVec(name, _) => *name,
            VhdlType::TypeAlias(name, _) => *name,
            VhdlType::I32(name, _) => *name,
            VhdlType::I64(name, _) => *name,
            VhdlType::F64(name, _) => *name,
            VhdlType::P64(name, _, _) => *name,
            VhdlType::Record(name, _) => *name,
            VhdlType::Enum(name, _, _) => *name,
            VhdlType::Array(name, _, _) => *name,
            VhdlType::Missing => panic!("missing type, no name"),
        }
    }

    fn int_range(&self) -> Option<IntRange> {
        match self {
            VhdlType::NineValueBit(_) => Some(IntRange(RangeDir::To, 0, 8)),
            VhdlType::I32(_, range) => *range,
            VhdlType::I64(_, range) => *range,
            VhdlType::P64(_, range, _) => *range,
            VhdlType::Enum(_, lits, _) => Some(IntRange(RangeDir::To, 0, lits.len() as i64)),
            VhdlType::NineValueVec(_, range) => *range,
            VhdlType::BitVec(_, range) => *range,
            VhdlType::Array(_, _, range) => *range,
            _ => None,
        }
    }

    fn is_array(&self) -> bool {
        matches!(
            self,
            // while we make a distinction between regular and "special" arrays, GHDL
            // treats all three types as just arrays
            Self::NineValueVec(_, _) | Self::BitVec(_, _) | Self::Array(_, _, _)
        )
    }

    fn is_record(&self) -> bool {
        matches!(self, Self::Record(_, _))
    }

    /// Determines whether the number of elements contained in the given type is finite.
    fn is_finite(tpe_id: TypeId, types: &[VhdlType]) -> bool {
        match &types[tpe_id.index()] {
            VhdlType::NineValueBit(_) => true,
            VhdlType::NineValueVec(_, None) => false,
            VhdlType::NineValueVec(_, Some(_)) => true,
            VhdlType::Bit(_) => true,
            VhdlType::BitVec(_, None) => false,
            VhdlType::BitVec(_, Some(_)) => true,
            &VhdlType::TypeAlias(_, underlying) => Self::is_finite(underlying, types),
            VhdlType::I32(_, None) => false,
            VhdlType::I32(_, Some(_)) => true,
            VhdlType::I64(_, None) => false,
            VhdlType::I64(_, Some(_)) => true,
            VhdlType::F64(_, None) => false,
            VhdlType::F64(_, Some(_)) => true,
            VhdlType::P64(_, None, _) => false,
            VhdlType::P64(_, Some(_), _) => true,
            VhdlType::Record(_, fields) => fields.iter().all(|&(_, f)| Self::is_finite(f, types)),
            VhdlType::Enum(_, _, _) => true,
            VhdlType::Array(_, _, None) => false,
            &VhdlType::Array(_, et, Some(_)) => Self::is_finite(et, types),
            VhdlType::Missing => unreachable!(""),
        }
    }
}

/// Returns Some(VhdlType::NineValueBit(..)) if the enum corresponds to a 9-value bit type.
fn try_parse_nine_value_bit(
    strings: &[String],
    name: StringId,
    literals: &[StringId],
) -> Option<VhdlType> {
    if check_literals_match(strings, literals, &STD_LOGIC_VALUES) {
        Some(VhdlType::NineValueBit(name))
    } else {
        None
    }
}

/// Returns Some(VhdlType::Bit(..)) if the enum corresponds to a 2-value bit type.
fn try_parse_two_value_bit(
    strings: &[String],
    name: StringId,
    literals: &[StringId],
) -> Option<VhdlType> {
    if check_literals_match(strings, literals, &VHDL_BIT_VALUES) {
        Some(VhdlType::Bit(name))
    } else {
        None
    }
}

#[inline]
fn check_literals_match(strings: &[String], literals: &[StringId], expected: &[u8]) -> bool {
    if literals.len() != expected.len() {
        return false;
    }

    // check to see if it matches the standard std_logic enum
    for (lit_id, expected) in literals.iter().zip(expected.iter()) {
        let lit = strings[lit_id.0].as_bytes();
        let cc = match lit.len() {
            1 => lit[0],
            3 => lit[1], // this is to account for GHDL encoding things as '0' (including the tick!)
            _ => return false,
        };
        if !cc.eq_ignore_ascii_case(expected) {
            return false; // encoding does not match
        }
    }

    true
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
    strings: Vec<HierarchyStringId>,
    enums: Vec<EnumTypeId>,
}

/// Returns a if it is a non-anonymous string, otherwise b.
fn pick_best_name(a: StringId, b: StringId) -> StringId {
    if a.is_none() {
        b
    } else {
        a
    }
}

impl GhwTables {
    fn get_type_and_name(&self, type_id: TypeId) -> (&VhdlType, HierarchyStringId) {
        let top_name = self.types[type_id.index()].name();
        let tpe = lookup_concrete_type(&self.types, type_id);
        let name = pick_best_name(top_name, tpe.name());
        (tpe, self.get_str(name))
    }

    fn get_str(&self, string_id: StringId) -> HierarchyStringId {
        self.strings[string_id.0]
    }
}

fn read_hierarchy_section(
    header: &HeaderData,
    tables: &mut GhwTables,
    input: &mut impl BufRead,
    h: &mut HierarchyBuilder,
) -> Result<GhwDecodeInfo> {
    let mut hdr = [0u8; 16];
    input.read_exact(&mut hdr)?;
    check_header_zeros("hierarchy", &hdr)?;

    // it appears that this number is actually not always 100% accurate
    let _expected_num_scopes = header.read_u32(&mut &hdr[4..8])?;
    // declared signals, may be composite
    let expected_num_declared_vars = header.read_u32(&mut &hdr[8..12])?;
    let max_signal_id = header.read_u32(&mut &hdr[12..16])?;

    let mut num_declared_vars = 0;
    let mut index_string_cache = IndexCache::default();
    let mut signal_info = GhwSignalTracker::new(max_signal_id);

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
                read_hierarchy_var(
                    tables,
                    input,
                    kind,
                    &mut signal_info,
                    &mut index_string_cache,
                    h,
                )?;
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

    let (decode_info, aliases) = signal_info.into_decode_info();
    for alias in aliases.into_iter() {
        h.add_slice(alias.signal_ref, alias.msb, alias.lsb, alias.sliced_signal);
    }

    Ok(decode_info)
}

/// used in order to get the value of an iterator variable in a generate for
fn read_signal_value_to_str(
    _tables: &GhwTables,
    tpe: &VhdlType,
    input: &mut impl BufRead,
) -> Result<String> {
    let s = match tpe {
        VhdlType::NineValueBit(_) | VhdlType::Bit(_) | VhdlType::Enum(_, _, _) => {
            format!("{}", read_u8(input)?)
        }
        VhdlType::I32(_, _) | VhdlType::I64(_, _) => {
            format!("{}", leb128::read::signed(input)?)
        }
        VhdlType::F64(_, _) => {
            format!("{}", read_f64_le(input)?)
        }
        VhdlType::TypeAlias(_, _) => unreachable!("type should have been resolved"),
        other => todo!("add support for {other:?}"),
    };
    Ok(s)
}

fn read_hierarchy_scope(
    tables: &GhwTables,
    input: &mut impl BufRead,
    kind: GhwHierarchyKind,
    h: &mut HierarchyBuilder,
) -> Result<()> {
    let name = read_string_id(input)?;
    let name = tables.get_str(name);

    let name = if kind == GhwHierarchyKind::GenerateFor {
        let iter_tpe_id = read_type_id(input)?;
        let (iter_tpe, _) = tables.get_type_and_name(iter_tpe_id);
        let value = read_signal_value_to_str(tables, iter_tpe, input)?;
        // create a new name based on the value
        let name = format!("{}({})", h.get_str(name), value);
        h.add_string(name)
    } else {
        name
    };

    h.add_scope(
        name,
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
    signals: &mut GhwSignalTracker,
    index_string_cache: &mut IndexCache,
    h: &mut HierarchyBuilder,
) -> Result<()> {
    let name_id = read_string_id(input)?;
    let name = tables.get_str(name_id);
    let tpe = read_type_id(input)?;
    add_var(
        tables,
        input,
        kind,
        signals,
        index_string_cache,
        h,
        name,
        tpe,
    )
}

type IndexCache = FxHashMap<i64, HierarchyStringId>;

fn get_index_string(
    index_string_cache: &mut IndexCache,
    h: &mut HierarchyBuilder,
    element_id: i64,
) -> HierarchyStringId {
    match index_string_cache.get(&element_id) {
        Some(id) => *id,
        None => {
            let id = h.add_string(format!("[{element_id}]"));
            index_string_cache.insert(element_id, id);
            id
        }
    }
}

/// Collects information for every signal id in the GHW file.
/// We store the type information needed to decode the signal as well as
/// information that is needed in order to map the bit-wise signal encoding
/// in a GHW to the bit-vector signal encoding used in our `wellen` wavemem.
#[derive(Debug)]
struct GhwSignalTracker {
    signals: Vec<Option<GhwSignalInfo>>,
    signal_ref_count: usize,
    vectors: Vec<GhwVecInfo>,
    aliases: Vec<AliasInfo>,
}

/// Used to define aliased signals in the `wellen` wavemem.
#[derive(Debug, Clone, Copy)]
struct AliasInfo {
    msb: u32,
    lsb: u32,
    signal_ref: SignalRef,
    sliced_signal: SignalRef,
    next: Option<NonZeroU32>,
}

impl GhwSignalTracker {
    fn new(max_signal_id: u32) -> Self {
        let signals = vec![None; max_signal_id as usize];
        Self {
            signals,
            signal_ref_count: 0,
            vectors: vec![],
            aliases: vec![],
        }
    }

    fn into_decode_info(self) -> (GhwDecodeInfo, Vec<AliasInfo>) {
        let mut signals: Vec<_> = self.signals.into_iter().flatten().collect();
        signals.shrink_to_fit();
        ((GhwSignals::new(signals), self.vectors), self.aliases)
    }

    fn max_signal_id(&self) -> usize {
        self.signals.len()
    }

    /// used for f64, i32, i64 and u8
    fn register_scalar(&mut self, signal_id: GhwSignalId, tpe: SignalType) -> SignalRef {
        if let Some(prev) = &self.signals[signal_id.index()] {
            debug_assert_eq!(prev.tpe(), tpe, "{signal_id:?}");
            prev.signal_ref()
        } else {
            // create new id
            let id = self.new_signal_ref();
            // we do not need to store any extra aliasing info
            self.signals[signal_id.index()] = Some(GhwSignalInfo::new(tpe, id, None));
            id
        }
    }

    fn new_signal_ref(&mut self) -> SignalRef {
        let id = SignalRef::from_index(self.signal_ref_count).unwrap();
        self.signal_ref_count += 1;
        id
    }

    fn find_or_add_alias(&mut self, vec_id: GhwVecId, msb: u32, lsb: u32) -> SignalRef {
        // first we try to see if we can find an existing alias
        let sliced_signal = self.vectors[vec_id.index()].signal_ref();
        if let Some(mut alias_id) = self.vectors[vec_id.index()].alias() {
            loop {
                let alias = self.aliases[alias_id.get() as usize - 1];
                if alias.msb == msb && alias.lsb == lsb {
                    return alias.signal_ref;
                }
                match alias.next {
                    Some(next) => alias_id = next,
                    None => {
                        // create new entry
                        let signal_ref = self.new_signal_ref();
                        let new_alias_id = self.aliases.len() + 1;
                        self.aliases.push(AliasInfo {
                            msb,
                            lsb,
                            signal_ref,
                            sliced_signal,
                            next: None,
                        });
                        self.aliases[alias_id.get() as usize - 1].next =
                            Some(NonZeroU32::new(new_alias_id as u32).unwrap());
                        return signal_ref;
                    }
                }
            }
        } else {
            // create entry
            let signal_ref = self.new_signal_ref();
            let alias_id = self.aliases.len() + 1;
            self.aliases.push(AliasInfo {
                msb,
                lsb,
                signal_ref,
                sliced_signal,
                next: None,
            });
            self.vectors[vec_id.index()].set_alias(NonZeroU32::new(alias_id as u32).unwrap());
            signal_ref
        }
    }

    fn register_bit_vec(
        &mut self,
        min_id: GhwSignalId,
        max_id: GhwSignalId,
        is_binary: bool,
    ) -> SignalRef {
        let (min, max) = (min_id.index(), max_id.index());
        debug_assert!(max >= min);

        // check to see if there is already a vector that this signal aliases with
        let maybe_vector_id = self.find_vec(min, max);
        if let Some(vector_id) = maybe_vector_id {
            let vector = &self.vectors[vector_id.index()];
            let (prev_min, prev_max) = (vector.min().index(), vector.max().index());
            if max == prev_max && min == prev_min {
                // perfect alias -> just return the signal ref
                self.signals[min].unwrap().signal_ref()
            } else if prev_min <= min && prev_max >= max {
                // this means that a larger vector already exists and our signal is a slice of this vector
                let (msb, lsb) = (max - prev_min, min - prev_min);
                self.find_or_add_alias(vector_id, msb as u32, lsb as u32)
            } else if prev_min >= min && prev_max <= max {
                todo!("super signal alias!")
            } else {
                todo!("overlapped aliased vectors")
            }
        } else {
            // no existing vector to alias with
            if min == max {
                // single bit with no aliased vector behaves essentially like a scalar
                if is_binary {
                    self.register_scalar(min_id, SignalType::TwoState)
                } else {
                    self.register_scalar(min_id, SignalType::NineState)
                }
            } else {
                // for a vector we want to make sure that there are no bits that alias this vector
                for ii in min..=max {
                    // this situation is very unlikely since assigning bit ids first would make it harder for GHDL
                    // to assign contiguous vec ids
                    assert!(
                        self.signals[ii].is_none(),
                        "TODO: deal with bit that aliases with vector"
                    );
                }

                // create a new vec entry
                let vec_id = self.vectors.len();
                let id = self.new_signal_ref();
                self.vectors
                    .push(GhwVecInfo::new(min_id, max_id, is_binary, id));
                let tpe = if is_binary {
                    SignalType::TwoStateVec
                } else {
                    SignalType::NineStateVec
                };

                // update all bits
                for ii in min..=max {
                    self.signals[ii] = Some(GhwSignalInfo::new(tpe, id, Some(vec_id)));
                }
                id
            }
        }
    }

    /// Searches for vector info in the specified range.
    #[inline]
    fn find_vec(&self, min: usize, max: usize) -> Option<GhwVecId> {
        let mut res = None;
        for ii in min..=max {
            if let Some(info) = &self.signals[ii] {
                let vec_id = info.vec_id();
                debug_assert!(
                    res.is_none() || vec_id.is_none() || res == vec_id,
                    "Signal spans several sub-vectors"
                );
                res = vec_id;
                // early returns in release mode
                if !cfg!(debug_assertions) && res.is_some() {
                    return res;
                }
            }
        }
        res
    }
}

#[allow(clippy::too_many_arguments)]
fn add_var(
    tables: &GhwTables,
    input: &mut impl BufRead,
    kind: GhwHierarchyKind,
    signals: &mut GhwSignalTracker,
    index_string_cache: &mut IndexCache,
    h: &mut HierarchyBuilder,
    name: HierarchyStringId,
    type_id: TypeId,
) -> Result<()> {
    let (vhdl_tpe, tpe_name) = tables.get_type_and_name(type_id);
    let dir = convert_kind_to_dir(kind);
    match vhdl_tpe {
        VhdlType::Enum(_, literals, enum_id) => {
            let enum_type = tables.enums[*enum_id as usize];
            let index = read_signal_id(input, signals.max_signal_id())?;
            let bits = get_enum_bits(literals);
            let signal_ref = signals.register_scalar(index, SignalType::U8);
            h.add_var(
                name,
                VarType::Enum,
                crate::hierarchy::SignalEncoding::bit_vec_of_len(bits),
                dir,
                None,
                signal_ref,
                Some(enum_type),
                Some(tpe_name),
            );
        }
        VhdlType::NineValueBit(_) | VhdlType::Bit(_) => {
            let index = read_signal_id(input, signals.max_signal_id())?;
            let is_binary = matches!(vhdl_tpe, VhdlType::Bit(_));
            let signal_ref = signals.register_bit_vec(index, index, is_binary);
            let var_type = match h.get_str(tpe_name).to_ascii_lowercase().as_str() {
                "std_ulogic" => VarType::StdULogic,
                "std_logic" => VarType::StdLogic,
                "bit" => VarType::Bit,
                _ => VarType::Wire,
            };
            h.add_var(
                name,
                var_type,
                crate::hierarchy::SignalEncoding::bit_vec_of_len(1),
                dir,
                None,
                signal_ref,
                None,
                Some(tpe_name),
            );
        }
        VhdlType::I32(_, maybe_range) => {
            // TODO: we could use the range to deduce indices and tighter widths
            let _range = IntRange::from_i32_option(*maybe_range);
            let bits = 32;
            let index = read_signal_id(input, signals.max_signal_id())?;
            let signal_ref = signals.register_scalar(index, SignalType::Leb128Signed);
            let var_type = VarType::Integer;
            h.add_var(
                name,
                var_type,
                crate::hierarchy::SignalEncoding::bit_vec_of_len(bits),
                dir,
                None,
                signal_ref,
                None,
                Some(tpe_name),
            );
        }
        VhdlType::P64(_, maybe_range, _) => {
            // Tentatively treat them as normal integers
            let _range = IntRange::from_i64_option(*maybe_range);
            let bits = 64;
            let index = read_signal_id(input, signals.max_signal_id())?;
            let signal_ref = signals.register_scalar(index, SignalType::Leb128Signed);
            let var_type = VarType::Integer;
            h.add_var(
                name,
                var_type,
                crate::hierarchy::SignalEncoding::bit_vec_of_len(bits),
                dir,
                None,
                signal_ref,
                None,
                Some(tpe_name),
            );
        }
        VhdlType::F64(_, maybe_range) => {
            // TODO: we could use the range to deduce indices and tighter widths
            let _range = FloatRange::from_f64_option(*maybe_range);
            let index = read_signal_id(input, signals.max_signal_id())?;
            let signal_ref = signals.register_scalar(index, SignalType::F64);
            let var_type = VarType::Real;
            h.add_var(
                name,
                var_type,
                crate::hierarchy::SignalEncoding::Real,
                dir,
                None,
                signal_ref,
                None,
                Some(tpe_name),
            );
        }
        VhdlType::NineValueVec(_, Some(range)) | VhdlType::BitVec(_, Some(range)) => {
            let num_bits = range.len().unsigned_abs() as u32;
            if num_bits == 0 {
                // TODO: how should we correctly deal with an empty vector?
                return Ok(());
            }

            let mut signal_ids = Vec::with_capacity(num_bits as usize);
            for _ in 0..num_bits {
                signal_ids.push(read_signal_id(input, signals.max_signal_id())?);
            }

            // check assumption that all IDs are continuous
            for (prev, cur) in signal_ids
                .iter()
                .take(signal_ids.len() - 1)
                .zip(signal_ids.iter().skip(1))
            {
                debug_assert_eq!(
                    prev.index() + 1,
                    cur.index(),
                    "We expected signal ids increasing by exactly 1, not {prev:?} -> {cur:?}"
                );
            }

            let is_binary = matches!(vhdl_tpe, VhdlType::BitVec(_, _));
            let min = *signal_ids.first().unwrap();
            let max = *signal_ids.last().unwrap();
            let signal_ref = signals.register_bit_vec(min, max, is_binary);
            let var_type = match h.get_str(tpe_name).to_ascii_lowercase().as_str() {
                "std_ulogic_vector" => VarType::StdULogicVector,
                "std_logic_vector" => VarType::StdLogicVector,
                "bit_vector" => VarType::BitVector,
                _ => VarType::Wire,
            };
            h.add_var(
                name,
                var_type,
                crate::hierarchy::SignalEncoding::bit_vec_of_len(num_bits),
                dir,
                Some(range.as_var_index()),
                signal_ref,
                None,
                Some(tpe_name),
            );
        }
        VhdlType::Record(_, fields) => {
            h.add_scope(name, None, ScopeType::VhdlRecord, None, None, false);
            for (field_name, field_type) in fields.iter() {
                add_var(
                    tables,
                    input,
                    kind,
                    signals,
                    index_string_cache,
                    h,
                    tables.get_str(*field_name),
                    *field_type,
                )?;
            }
            h.pop_scope();
        }
        // we treat arrays like records
        VhdlType::Array(_, element_tpe, maybe_range) => {
            let range = IntRange::from_i32_option(*maybe_range);
            h.add_scope(name, None, ScopeType::VhdlArray, None, None, false);
            for element_id in range.range() {
                let name = get_index_string(index_string_cache, h, element_id);
                add_var(
                    tables,
                    input,
                    kind,
                    signals,
                    index_string_cache,
                    h,
                    name,
                    *element_tpe,
                )?;
            }
            h.pop_scope();
        }
        other => todo!("deal with {other:?}"),
    }
    Ok(())
}

fn read_signal_id(input: &mut impl BufRead, max_signal_id: usize) -> Result<GhwSignalId> {
    let index = leb128::read::unsigned(input)? as usize;
    if index > max_signal_id {
        Err(GhwParseError::FailedToParseSection(
            "hierarchy",
            format!("SignalId too large {index} > {}", max_signal_id),
        ))
    } else {
        let id = GhwSignalId::new(index as u32);
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

impl StringId {
    fn none() -> Self {
        Self(0)
    }

    fn is_none(&self) -> bool {
        self.0 == 0
    }
}

/// Pointer into the GHW type table. Always positive!
#[derive(Debug, Copy, Clone, PartialEq)]
struct TypeId(NonZeroU32);

impl TypeId {
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
    fn from_index(index: usize) -> Self {
        Self(NonZeroU32::new(index as u32 + 1).unwrap())
    }
}

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
    fn range(&self) -> Box<dyn Iterator<Item = i64>> {
        match self.0 {
            RangeDir::To => Box::new(self.1..=self.2),
            RangeDir::Downto => Box::new((self.2..=self.1).rev()),
        }
    }

    fn len(&self) -> i64 {
        match self.0 {
            RangeDir::To => self.2 - self.1 + 1,
            RangeDir::Downto => self.1 - self.2 + 1,
        }
    }

    fn as_var_index(&self) -> VarIndex {
        VarIndex::new(self.1, self.2)
    }

    fn from_i64_option(opt: Option<Self>) -> Self {
        opt.unwrap_or(Self(RangeDir::To, i64::MIN, i64::MAX))
    }

    fn from_i32_option(opt: Option<Self>) -> Self {
        opt.unwrap_or(Self(RangeDir::To, i32::MIN as i64, i32::MAX as i64))
    }

    fn start(&self) -> i64 {
        match self.0 {
            RangeDir::To => self.1,
            RangeDir::Downto => self.2,
        }
    }

    fn end(&self) -> i64 {
        match self.0 {
            RangeDir::To => self.2,
            RangeDir::Downto => self.1,
        }
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.start() >= other.start() && self.end() <= other.end()
    }
}

#[derive(Debug, Copy, Clone)]
struct FloatRange(RangeDir, f64, f64);

impl FloatRange {
    fn range(&self) -> std::ops::RangeInclusive<f64> {
        match self.0 {
            RangeDir::To => self.1..=(self.2),
            RangeDir::Downto => self.2..=(self.1),
        }
    }

    fn from_f64_option(opt: Option<Self>) -> Self {
        opt.unwrap_or(Self(RangeDir::To, f64::MIN, f64::MAX))
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        let self_range = self.range();
        let other_range = other.range();
        self_range.start() >= other_range.start() && self_range.end() <= other_range.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_range() {
        let r = IntRange(RangeDir::To, 1, 3);
        assert_eq!(r.start(), 1);
        assert_eq!(r.end(), 3);
        assert_eq!(r.len(), 3);
        let r_seq: Vec<i64> = r.range().collect();
        assert_eq!(r_seq, [1, 2, 3]);

        let r = IntRange(RangeDir::Downto, 4, 0);
        assert_eq!(r.start(), 0);
        assert_eq!(r.end(), 4);
        assert_eq!(r.len(), 5);
        let r_seq: Vec<i64> = r.range().collect();
        assert_eq!(r_seq, [4, 3, 2, 1, 0]);
    }
}
