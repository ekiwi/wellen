// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::{HierarchyBuilder, HierarchyStringId};
use crate::wavemem::{bit_char_to_num, Encoder, States};
use crate::{
    FileFormat, FileType, Hierarchy, ScopeType, SignalRef, Timescale, TimescaleUnit, VarDirection,
    VarIndex, VarType, Waveform, WellenError,
};
use num_enum::TryFromPrimitive;
use std::io::{BufRead, Seek, SeekFrom};
use std::num::NonZeroU32;
use thiserror::Error;

#[derive(Debug, Error)]
enum GhwParseError {
    #[error("[ghw] unsupported compression: {0}")]
    UnsupportedCompression(String),
    #[error("[ghw] unexpected header magic: {0}")]
    UnexpectedHeaderMagic(String),
    #[error("[ghw] unexpected value in header: {0:?}")]
    UnexpectedHeader(HeaderData),
    #[error("[ghw] unexpected section start: {0}")]
    UnexpectedSection(String),
    #[error("[ghw] unexpected type: {0:?}, {1}")]
    UnexpectedType(GhwRtik, &'static str),
    #[error("[ghw] failed to parse a {0} section: {1}")]
    FailedToParseSection(&'static str, String),
    #[error("[ghw] expected positive integer, not: {0}")]
    ExpectedPositiveInteger(i64),
    #[error("[ghw] float range has no length: {0} .. {1}")]
    FloatRangeLen(f64, f64),
    #[error("[ghw] failed to parse GHDL RTIK.")]
    FailedToParseGhdlRtik(#[from] num_enum::TryFromPrimitiveError<GhwRtik>),
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

impl From<GhwParseError> for WellenError {
    fn from(value: GhwParseError) -> Self {
        WellenError::FailedToLoad(FileFormat::Ghw, value.to_string())
    }
}

/// Checks header to see if we are dealing with a GHW file.
pub(crate) fn is_ghw(input: &mut (impl BufRead + Seek)) -> bool {
    let is_ghw = read_ghw_header(input).is_ok();
    // try to reset input
    let _ = input.seek(std::io::SeekFrom::Start(0));
    is_ghw
}

pub fn read(filename: &str) -> std::result::Result<Waveform, WellenError> {
    let f = std::fs::File::open(filename)?;
    let mut input = std::io::BufReader::new(f);
    read_internal(&mut input)
}

pub fn read_from_bytes(bytes: Vec<u8>) -> std::result::Result<Waveform, WellenError> {
    let mut input = std::io::Cursor::new(bytes);
    read_internal(&mut input)
}

fn read_internal(input: &mut (impl BufRead + Seek)) -> std::result::Result<Waveform, WellenError> {
    let header = read_ghw_header(input)?;
    let header_len = input.stream_position()?;

    // currently we do read the directory, however we are not using it yet
    let _sections = try_read_directory(&header, input)?;
    input.seek(SeekFrom::Start(header_len))?;
    // TODO: use actual section positions

    let (decode_info, signal_ref_count, hierarchy) = read_hierarchy(&header, input)?;
    let wave_mem = read_signals(&header, &decode_info, signal_ref_count, &hierarchy, input)?;
    Ok(Waveform::new(hierarchy, wave_mem))
}

fn read_ghw_header(input: &mut impl BufRead) -> Result<HeaderData> {
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

const GHW_TAILER_LEN: usize = 12;

/// The last 8 bytes of a finished, uncompressed file indicate where to find the directory which
/// contains the offset of all sections.
fn try_read_directory(
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

fn read_directory(header: &HeaderData, input: &mut impl BufRead) -> Result<Vec<SectionPos>> {
    let mut h = [0u8; 8];
    input.read_exact(&mut h)?;
    // note: the directory section does not contain the normal 4 zeros
    let num_entries = header.read_u32(&mut &h[4..8])?;

    let mut sections = Vec::new();
    for _ in 0..num_entries {
        let mut id = [0u8; 4];
        input.read_exact(&mut id)?;
        let mut buf = [0u8; 4];
        input.read_exact(&mut buf)?;
        let pos = header.read_u32(&mut &buf[..])?;
        sections.push(SectionPos { id, pos });
    }

    check_magic_end(input, "directory", GHW_END_DIRECTORY_SECTION)?;
    Ok(sections)
}

#[derive(Debug)]
struct SectionPos {
    id: [u8; 4],
    pos: u32,
}

/// Reads the GHW signal values. `input` should be advanced until right after the end of hierarchy
fn read_signals(
    header: &HeaderData,
    info: &GhwDecodeInfo,
    signal_ref_count: usize,
    hierarchy: &Hierarchy,
    input: &mut impl BufRead,
) -> Result<Box<crate::wavemem::Reader>> {
    // TODO: multi-threading
    let mut encoder = Encoder::new(hierarchy);
    let mut vecs = VecBuffer::from_decode_info(info, signal_ref_count);

    // loop over signal sections
    loop {
        let mut mark = [0u8; 4];
        input.read_exact(&mut mark)?;

        // read_sm_hdr
        match &mark {
            GHW_SNAPSHOT_SECTION => {
                read_snapshot_section(header, info, &mut vecs, &mut encoder, input)?
            }
            GHW_CYCLE_SECTION => read_cycle_section(header, info, &mut vecs, &mut encoder, input)?,
            GHW_DIRECTORY_SECTION => {
                // skip the directory by reading it
                let _ = read_directory(header, input)?;
            }
            GHW_TAILER_SECTION => {
                // the "tailer" means that we are done reading the file
                break;
            }
            other => {
                return Err(GhwParseError::UnexpectedSection(
                    String::from_utf8_lossy(other).to_string(),
                ))
            }
        }
    }
    Ok(Box::new(encoder.finish()))
}

fn read_snapshot_section(
    header: &HeaderData,
    info: &GhwDecodeInfo,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut h = [0u8; 12];
    input.read_exact(&mut h)?;
    check_header_zeros("snapshot", &h)?;

    // time in femto seconds
    let start_time = header.read_i64(&mut &h[4..12])? as u64;
    enc.time_change(start_time);

    for sig in info.signals.iter() {
        read_signal_value(info, sig, vecs, enc, input)?;
    }

    // check for correct end magic
    check_magic_end(input, "snapshot", GHW_END_SNAPSHOT_SECTION)?;
    Ok(())
}

fn check_magic_end(input: &mut impl BufRead, section: &'static str, expected: &[u8]) -> Result<()> {
    let mut end_magic = [0u8; 4];
    input.read_exact(&mut end_magic)?;
    if &end_magic == expected {
        Ok(())
    } else {
        Err(GhwParseError::UnexpectedSection(format!(
            "expected {section} section to end in {}, not {}",
            String::from_utf8_lossy(expected),
            String::from_utf8_lossy(&end_magic)
        )))
    }
}

fn read_cycle_section(
    header: &HeaderData,
    info: &GhwDecodeInfo,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut h = [0u8; 8];
    input.read_exact(&mut h)?;
    // note: cycle sections do not have the four zero bytes!

    // time in femto seconds
    let mut start_time = header.read_i64(&mut &h[..])? as u64;

    loop {
        enc.time_change(start_time);
        read_cycle_signals(info, vecs, enc, input)?;

        let time_delta = leb128::read::signed(input)?;
        if time_delta < 0 {
            break; // end of cycle
        } else {
            start_time += time_delta as u64;
        }
    }

    // check cycle end
    check_magic_end(input, "cycle", GHW_END_CYCLE_SECTION)?;

    Ok(())
}

fn read_cycle_signals(
    info: &GhwDecodeInfo,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut pos_signal_index = 0;
    loop {
        let delta = leb128::read::unsigned(input)? as usize;
        if delta == 0 {
            break;
        }
        pos_signal_index += delta;
        if pos_signal_index == 0 {
            return Err(GhwParseError::FailedToParseSection(
                "cycle",
                "Expected a first delta > 0".to_string(),
            ));
        }
        let sig = &info.signals[pos_signal_index - 1];
        read_signal_value(info, sig, vecs, enc, input)?;
    }
    Ok(())
}

fn read_signal_value(
    info: &GhwDecodeInfo,
    signal: &GhwSignal,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    match signal.tpe {
        SignalType::NineState(lut) => {
            let value = [info.decode(read_u8(input)?, lut)];
            enc.raw_value_change(signal.signal_ref, &value, States::Nine);
        }
        SignalType::NineStateBit(lut, bit, _) => {
            let value = info.decode(read_u8(input)?, lut);

            // check to see if we already had a change to this same bit in the current time step
            if vecs.is_second_change(signal.signal_ref, bit, value) {
                // immediately dispatch the change to properly reflect the delta cycle
                let data = vecs.get_full_value_and_clear_changes(signal.signal_ref);
                enc.raw_value_change(signal.signal_ref, data, States::Nine);
            }

            // update value
            vecs.update_value(signal.signal_ref, bit, value);

            // check to see if we need to report a change
            if vecs.full_signal_has_changed(signal.signal_ref) {
                let data = vecs.get_full_value_and_clear_changes(signal.signal_ref);
                enc.raw_value_change(signal.signal_ref, data, States::Nine);
            }
        }
        SignalType::U8(bits) => {
            let value = [read_u8(input)?];
            if bits < 8 {
                debug_assert!(value[0] < (1u8 << bits));
            }
            enc.raw_value_change(signal.signal_ref, &value, States::Two);
        }
        SignalType::Leb128Signed(bits) => {
            let value = leb128::read::signed(input)? as u64;
            if bits < u64::BITS {
                debug_assert!(value < (1u64 << bits));
            }
            enc.raw_value_change(signal.signal_ref, &value.to_be_bytes(), States::Two);
        }

        SignalType::F64 => {
            // we need to figure out the endianes here
            let mut buf = [0u8; 8];
            input.read_exact(&mut buf)?;
            todo!(
                "float values: {} or {}?",
                f64::from_le_bytes(buf.clone()),
                f64::from_be_bytes(buf)
            )
        }
    }
    Ok(())
}

/// Keeps track of individual bits and combines them into a full bit vector.
#[derive(Debug)]
struct VecBuffer {
    info: Vec<Option<VecBufferInfo>>,
    data: Vec<u8>,
    change: Vec<u8>,
}

#[derive(Debug, Clone)]
struct VecBufferInfo {
    /// Offset in bits
    offset: u64,
    bits: u32,
}

impl VecBufferInfo {
    fn change_range(&self) -> std::ops::Range<usize> {
        // whether a bit has been changed is stored with 8 bits per byte
        let start = self.offset.div_ceil(8) as usize;
        let len = self.bits.div_ceil(8) as usize;
        start..(start + len)
    }
    fn data_range(&self) -> std::ops::Range<usize> {
        // data is stored with 2 bits per byte
        let start = self.offset.div_ceil(2) as usize;
        let len = self.bits.div_ceil(2) as usize;
        start..(start + len)
    }
}

impl VecBuffer {
    fn from_decode_info(decode_info: &GhwDecodeInfo, signal_ref_count: usize) -> Self {
        let mut info = Vec::with_capacity(signal_ref_count);
        info.resize(signal_ref_count, None);
        let mut offset = 0;

        for signal in decode_info.signals.iter() {
            if let SignalType::NineStateBit(_, 0, bits) = signal.tpe {
                if info[signal.signal_ref.index()].is_none() {
                    info[signal.signal_ref.index()] = Some(VecBufferInfo { offset, bits });
                    // pad offset to ensure that each value starts with its own byte
                    let offset_delta = bits.div_ceil(8) * 8;
                    offset += offset_delta as u64;
                }
            }
        }

        let data_bytes = offset.div_ceil(2) as usize;
        let change_bytes = offset.div_ceil(8) as usize;
        let mut data = Vec::with_capacity(data_bytes);
        data.resize(data_bytes, 0);
        let mut change = Vec::with_capacity(change_bytes);
        change.resize(change_bytes, 0);

        Self { info, data, change }
    }

    #[inline]
    fn is_second_change(&self, signal_ref: SignalRef, bit: u32, value: u8) -> bool {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();
        self.has_changed(info, bit) && self.get_value(info, bit) != value
    }

    #[inline]
    fn update_value(&mut self, signal_ref: SignalRef, bit: u32, value: u8) {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();
        let is_a_real_change = self.get_value(info, bit) != value;
        if is_a_real_change {
            Self::mark_changed(&mut self.change, info, bit);
            Self::set_value(&mut self.data, info, bit, value);
        }
    }

    /// Used in order to dispatch full signal changes as soon as possible
    #[inline]
    fn full_signal_has_changed(&self, signal_ref: SignalRef) -> bool {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();

        // check changes
        let changes = &self.change[info.change_range()];
        let skip = if info.bits % 8 == 0 { 0 } else { 1 };
        for e in changes.iter().skip(skip) {
            if *e != 0xff {
                return false;
            }
        }

        // check valid msb (in case where the number of bits is not a multiple of 8)
        if skip > 0 {
            let msb_mask = (1u8 << (info.bits % 8)) - 1;
            if changes[0] != msb_mask {
                return false;
            }
        }

        true
    }

    #[inline]
    fn get_full_value_and_clear_changes(&mut self, signal_ref: SignalRef) -> &[u8] {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();
        let changes = &mut self.change[info.change_range()];

        // clear changes
        for e in changes.iter_mut() {
            *e = 0;
        }

        // return reference to value
        let data = &self.data[info.data_range()];
        data
    }

    #[inline]
    fn has_changed(&self, info: &VecBufferInfo, bit: u32) -> bool {
        debug_assert!(bit < info.bits);
        let valid = &self.change[info.change_range()];
        (valid[(bit / 8) as usize] >> (bit % 8)) & 1 == 1
    }

    #[inline]
    fn mark_changed(change: &mut [u8], info: &VecBufferInfo, bit: u32) {
        debug_assert!(bit < info.bits);
        let index = (bit / 8) as usize;
        let changes = &mut change[info.change_range()][index..(index + 1)];
        let mask = 1u8 << (bit % 8);
        changes[0] |= mask;
    }

    #[inline]
    fn get_value(&self, info: &VecBufferInfo, bit: u32) -> u8 {
        debug_assert!(bit < info.bits);
        let data = &self.data[info.data_range()];
        let byte = data[(bit / 2) as usize];
        if bit % 2 == 0 {
            byte & 0xf
        } else {
            (byte >> 4) & 0xf
        }
    }

    #[inline]
    fn set_value(data: &mut [u8], info: &VecBufferInfo, bit: u32, value: u8) {
        debug_assert!(bit < info.bits);
        debug_assert!(value <= 0xf);
        let index = (bit / 2) as usize;
        let data = &mut data[info.data_range()][index..(index + 1)];
        if bit % 2 == 0 {
            data[0] = (data[0] & 0xf0) | value;
        } else {
            data[0] = (data[0] & 0x0f) | (value << 4);
        }
    }
}

/// Contains information needed in order to decode value changes.
#[derive(Debug, Default)]
struct GhwDecodeInfo {
    signals: Vec<GhwSignal>,
    luts: Vec<NineValueLut>,
}

impl GhwDecodeInfo {
    fn decode(&self, value: u8, lut_id: NineValueLutId) -> u8 {
        self.luts[lut_id.0 as usize][value as usize]
    }
}

/// Holds information from the header needed in order to read the corresponding data in the signal section.
#[derive(Debug, Clone)]
struct GhwSignal {
    /// Signal ID in the wavemem Encoder.
    signal_ref: SignalRef,
    tpe: SignalType,
}

#[derive(Debug, PartialEq, Copy, Clone)]
struct NineValueLutId(u8);

/// Specifies the signal type info that is needed in order to read it.
#[derive(Debug, PartialEq, Copy, Clone)]
enum SignalType {
    /// Nine value signal encoded as a single byte.
    NineState(NineValueLutId),
    /// A single bit in a nine value bit vector. bit N / M bits.
    NineStateBit(NineValueLutId, u32, u32),
    /// Binary signal encoded as a single byte with N valid bits.
    U8(u32),
    /// Binary signal encoded as a variable number of bytes with N valid bits.
    Leb128Signed(u32),
    /// F64 (real)
    F64,
}

/// Parses the beginning of the GHW file until the end of the hierarchy.
fn read_hierarchy(
    header: &HeaderData,
    input: &mut impl BufRead,
) -> Result<(GhwDecodeInfo, usize, Hierarchy)> {
    let mut tables = GhwTables::default();
    let mut decode = GhwDecodeInfo::default();
    let mut hb = HierarchyBuilder::new(FileType::Vcd);
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
                                matches!(tpe, VhdlType::NineValueBit(_, _)),
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

const GHW_STRING_SECTION: &[u8; 4] = b"STR\x00";
const GHW_HIERARCHY_SECTION: &[u8; 4] = b"HIE\x00";
const GHW_TYPE_SECTION: &[u8; 4] = b"TYP\x00";
const GHW_WK_TYPE_SECTION: &[u8; 4] = b"WKT\x00";
const GHW_END_OF_HEADER_SECTION: &[u8; 4] = b"EOH\x00";
const GHW_SNAPSHOT_SECTION: &[u8; 4] = b"SNP\x00";
const GHW_CYCLE_SECTION: &[u8; 4] = b"CYC\x00";
const GHW_DIRECTORY_SECTION: &[u8; 4] = b"DIR\x00";
const GHW_TAILER_SECTION: &[u8; 4] = b"TAI\x00";
const GHW_END_SNAPSHOT_SECTION: &[u8; 4] = b"ESN\x00";
const GHW_END_CYCLE_SECTION: &[u8; 4] = b"ECY\x00";
const GHW_END_DIRECTORY_SECTION: &[u8; 4] = b"EOD\x00";

#[derive(Debug)]
enum Section {
    StringTable(Vec<String>),
    TypeTable(Vec<VhdlType>),
    WellKnownTypes(Vec<(TypeId, GhwWellKnownType)>),
    Hierarchy(Vec<GhwSignal>),
    EndOfHeader,
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
            format!(
                "first four bytes should be zero and not {}",
                String::from_utf8_lossy(&zeros)
            ),
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
        other => return Err(GhwParseError::UnexpectedType(other, "for range")),
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
                    let field_tpe = lookup_concrete_type_id(&tables.types, read_type_id(input)?);
                    fields.push((field_name, field_tpe));
                }
                VhdlType::from_record(name, &tables.types, fields)
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

type NineValueLut = [u8; 9];

/// Our own custom representation of VHDL Types.
/// During GHW parsing we convert the GHDL types to our own representation.
#[derive(Debug)]
enum VhdlType {
    /// std_logic or std_ulogic with lut to convert to the `wellen` 9-value representation.
    NineValueBit(StringId, NineValueLutId),
    /// std_logic_vector or std_ulogic_vector with lut to convert to the `wellen` 9-value representation.
    NineValueVec(StringId, NineValueLutId, IntRange),
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
        VhdlType::TypeAlias(name, base_id) => {
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
        if let (VhdlType::NineValueBit(_, lut), Some(range)) =
            (&types[element_tpe_id.index()], index_range)
        {
            VhdlType::NineValueVec(name, lut.clone(), range)
        } else {
            VhdlType::Array(name, element_tpe_id, index_range)
        }
    }

    fn from_record(name: StringId, types: &[VhdlType], fields: Vec<(StringId, TypeId)>) -> Self {
        if cfg!(debug_assertions) {
            for (_, tpe) in fields.iter() {
                debug_assert!(!types[tpe.index()].is_alias());
            }
        }
        VhdlType::Record(name, fields)
    }

    fn from_subtype_array(name: StringId, types: &[VhdlType], base: TypeId, range: Range) -> Self {
        let base_tpe = lookup_concrete_type(types, base);
        match (base_tpe, range) {
            (VhdlType::Array(_, element_tpe, maybe_base_range), Range::Int(int_range)) => {
                todo!()
            }
            (VhdlType::NineValueVec(_, lut, base_range), Range::Int(int_range)) => {
                debug_assert!(
                    int_range.is_subset_of(&base_range),
                    "{int_range:?} {base_range:?}"
                );
                VhdlType::NineValueVec(name, lut.clone(), int_range)
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
            (VhdlType::NineValueBit(_, _), Range::Int(int_range)) => {
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
            VhdlType::NineValueBit(name, _) => *name,
            VhdlType::NineValueVec(name, _, _) => *name,
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
            VhdlType::NineValueBit(_, _) => Some(IntRange(RangeDir::To, 0, 8)),
            VhdlType::I32(_, range) => *range,
            VhdlType::I64(_, range) => *range,
            VhdlType::Enum(_, lits) => Some(IntRange(RangeDir::To, 0, lits.len() as i64)),
            _ => None,
        }
    }

    fn is_alias(&self) -> bool {
        matches!(self, VhdlType::TypeAlias(_, _))
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

    // try to build a translation table
    let mut lut = [0u8; 9];
    let mut out_covered = [false; 9];

    // map to the wellen 9-state lookup: ['0', '1', 'x', 'z', 'h', 'u', 'w', 'l', '-'];
    for (ii, lit_id) in literals.iter().enumerate() {
        let lit = tables.get_str(*lit_id).as_bytes();
        let cc = match lit.len() {
            1 => lit[0],
            3 => lit[1], // this is to account for GHDL encoding things as '0' (including the tick!)
            _ => return None,
        };
        if let Some(out) = bit_char_to_num(cc) {
            if out_covered[out as usize] {
                return None; // duplicate detected
            }
            out_covered[out as usize] = true;
            lut[ii] = out;
        } else {
            return None; // invalid character
        }
    }
    let lut_id = tables.get_lut_id(lut);
    Some(VhdlType::NineValueBit(name, lut_id))
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
    luts: Vec<NineValueLut>,
    /// keps track of whether we have already added a string to the hierarchy
    hier_string_ids: Vec<Option<HierarchyStringId>>,
}

impl GhwTables {
    fn get_type(&self, type_id: TypeId) -> &VhdlType {
        lookup_concrete_type(&self.types, type_id)
    }

    fn get_type_and_name(&self, type_id: TypeId) -> (&VhdlType, &str) {
        let name = self.get_str(self.types[type_id.index()].name());
        (lookup_concrete_type(&self.types, type_id), name)
    }

    fn get_str(&self, string_id: StringId) -> &str {
        &self.strings[string_id.0]
    }

    fn get_hier_str_id(
        &mut self,
        h: &mut HierarchyBuilder,
        string_id: StringId,
    ) -> HierarchyStringId {
        if self.hier_string_ids.len() < self.strings.len() {
            self.hier_string_ids.resize(self.strings.len(), None);
        }
        if let Some(id) = self.hier_string_ids[string_id.0] {
            id
        } else {
            let id = h.add_string(self.strings[string_id.0].to_string());
            self.hier_string_ids[string_id.0] = Some(id);
            id
        }
    }

    fn get_lut_id(&mut self, lut: NineValueLut) -> NineValueLutId {
        let id = NineValueLutId(self.luts.len() as u8);
        self.luts.push(lut);
        id
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
        luts: tables.luts.clone(),
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
        let iter_type = read_type_id(input)?;
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
        GhwHierarchyKind::Instance => ScopeType::Interface,
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
    let (var_tpe, dir) = convert_var_kind(kind);
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
                var_tpe,
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
        VhdlType::NineValueBit(_, lut) => {
            let index = read_signal_id(input, signals)?;
            let signal_ref = get_signal_ref(signals, signal_ref_count, index);
            h.add_var(
                name,
                var_tpe,
                dir,
                1,
                None,
                signal_ref,
                None,
                Some(tpe_name),
            );
            // meta date for writing the signal later
            let tpe = SignalType::NineState(*lut);
            signals[index.0.get() as usize] = Some(GhwSignal { signal_ref, tpe })
        }
        VhdlType::NineValueVec(_, lut, range) => {
            let num_bits = range.len() as u32;
            let mut signal_ids = Vec::with_capacity(num_bits as usize);
            for _ in 0..num_bits {
                signal_ids.push(read_signal_id(input, signals)?);
            }
            let signal_ref = get_signal_ref(signals, signal_ref_count, signal_ids[0]);
            h.add_var(
                name,
                var_tpe,
                dir,
                num_bits,
                Some(range.as_var_index()),
                signal_ref,
                None,
                Some(tpe_name),
            );
            // meta date for writing the signal later
            for (bit, index) in signal_ids.iter().enumerate() {
                // TODO: are we iterating in the correct order?
                let tpe = SignalType::NineStateBit(*lut, bit as u32, num_bits);
                signals[index.0.get() as usize] = Some(GhwSignal { signal_ref, tpe })
            }
        }
        VhdlType::Record(_, fields) => {
            h.add_scope(name, None, ScopeType::Module, None, None, false);
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

fn convert_var_kind(kind: GhwHierarchyKind) -> (VarType, VarDirection) {
    match kind {
        GhwHierarchyKind::Signal => (VarType::Wire, VarDirection::Implicit),
        GhwHierarchyKind::PortIn => (VarType::Port, VarDirection::Input),
        GhwHierarchyKind::PortOut => (VarType::Port, VarDirection::Output),
        GhwHierarchyKind::PortInOut => (VarType::Port, VarDirection::InOut),
        GhwHierarchyKind::Buffer => (VarType::Wire, VarDirection::Buffer),
        GhwHierarchyKind::Linkage => (VarType::Wire, VarDirection::Linkage),
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

    fn from_i64_option(opt: Option<Self>) -> Self {
        opt.unwrap_or(Self(RangeDir::To, i64::MIN, i64::MAX))
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

#[derive(Debug)]
struct HeaderData {
    version: u8,
    big_endian: bool,
    #[allow(dead_code)]
    word_len: u8,
    #[allow(dead_code)]
    word_offset: u8,
}

impl HeaderData {
    #[inline]
    fn read_i32(&self, input: &mut impl BufRead) -> Result<i32> {
        let mut b = [0u8; 4];
        input.read_exact(&mut b)?;
        if self.big_endian {
            Ok(i32::from_be_bytes(b))
        } else {
            Ok(i32::from_le_bytes(b))
        }
    }
    #[inline]
    fn read_u32(&self, input: &mut impl BufRead) -> Result<u32> {
        let ii = self.read_i32(input)?;
        if ii >= 0 {
            Ok(ii as u32)
        } else {
            Err(GhwParseError::ExpectedPositiveInteger(ii as i64))
        }
    }

    #[inline]
    fn read_i64(&self, input: &mut impl BufRead) -> Result<i64> {
        let mut b = [0u8; 8];
        input.read_exact(&mut b)?;
        if self.big_endian {
            Ok(i64::from_be_bytes(b))
        } else {
            Ok(i64::from_le_bytes(b))
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

/// This enum used to be the same than the internal Ghdl rtik,
/// however in order to maintain backwards compatibility it was cloned.
#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone, TryFromPrimitive)]
enum GhwRtik {
    Error = 0,
    EndOfScope = 15,
    Signal = 16,
    PortIn = 17,
    PortOut = 18,
    PortInOut = 19,
    PortBuffer = 20,
    PortLinkage = 21,
    TypeB2 = 22,
    TypeE8 = 23,
    TypeI32 = 25,
    TypeI64 = 26,
    TypeF64 = 27,
    TypeP32 = 28,
    TypeP64 = 29,
    TypeArray = 31,
    TypeRecord = 32,
    SubtypeScalar = 34,
    SubtypeArray = 35,
    SubtypeUnboundedArray = 37,
    SubtypeRecord = 38,
    SubtypeUnboundedRecord = 39,
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
