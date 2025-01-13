// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::SignalRef;
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
    #[error("[ghw] unexpected value in header: {0}")]
    UnexpectedHeader(String),
    #[error("[ghw] unexpected section start: {0}")]
    UnexpectedSection(String),
    #[error("[ghw] unexpected type: {0}, {1}")]
    UnexpectedType(String, &'static str),
    #[error("[ghw] failed to parse a {0} section: {1}")]
    FailedToParseSection(&'static str, String),
    #[error("[ghw] expected positive integer, not: {0}")]
    ExpectedPositiveInteger(i64),
    #[error("[ghw] float range has no length: {0} .. {1}")]
    #[allow(dead_code)]
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

pub const GHW_STRING_SECTION: &[u8; 4] = b"STR\x00";
pub const GHW_HIERARCHY_SECTION: &[u8; 4] = b"HIE\x00";
pub const GHW_TYPE_SECTION: &[u8; 4] = b"TYP\x00";
pub const GHW_WK_TYPE_SECTION: &[u8; 4] = b"WKT\x00";
pub const GHW_END_OF_HEADER_SECTION: &[u8; 4] = b"EOH\x00";
pub const GHW_DIRECTORY_SECTION: &[u8; 4] = b"DIR\x00";
pub const GHW_TAILER_SECTION: &[u8; 4] = b"TAI\x00";
pub const GHW_END_DIRECTORY_SECTION: &[u8; 4] = b"EOD\x00";
pub const GHW_SNAPSHOT_SECTION: &[u8; 4] = b"SNP\x00";
pub const GHW_CYCLE_SECTION: &[u8; 4] = b"CYC\x00";
pub const GHW_END_SNAPSHOT_SECTION: &[u8; 4] = b"ESN\x00";
pub const GHW_END_CYCLE_SECTION: &[u8; 4] = b"ECY\x00";

pub const GHW_TAILER_LEN: usize = 12;

pub type Result<T> = std::result::Result<T, GhwParseError>;

pub fn read_directory(header: &HeaderData, input: &mut impl BufRead) -> Result<Vec<SectionPos>> {
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
pub struct SectionPos {
    #[allow(dead_code)]
    pub id: [u8; 4],
    #[allow(dead_code)]
    pub pos: u32,
}

pub fn check_header_zeros(section: &'static str, header: &[u8]) -> Result<()> {
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
                String::from_utf8_lossy(zeros)
            ),
        ))
    }
}

pub fn check_magic_end(
    input: &mut impl BufRead,
    section: &'static str,
    expected: &[u8],
) -> Result<()> {
    let mut end_magic = [0u8; 4];
    input.read_exact(&mut end_magic)?;
    if end_magic == expected {
        Ok(())
    } else {
        Err(GhwParseError::UnexpectedSection(format!(
            "expected {section} section to end in {}, not {}",
            String::from_utf8_lossy(expected),
            String::from_utf8_lossy(&end_magic)
        )))
    }
}

#[inline]
pub fn read_u8(input: &mut impl BufRead) -> Result<u8> {
    let mut buf = [0u8];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline]
pub fn read_f64_le(input: &mut impl BufRead) -> Result<f64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

#[derive(Debug)]
pub struct HeaderData {
    pub version: u8,
    pub big_endian: bool,
    #[allow(dead_code)]
    pub word_len: u8,
    #[allow(dead_code)]
    pub word_offset: u8,
}

impl HeaderData {
    #[inline]
    pub fn read_i32(&self, input: &mut impl BufRead) -> Result<i32> {
        let mut b = [0u8; 4];
        input.read_exact(&mut b)?;
        if self.big_endian {
            Ok(i32::from_be_bytes(b))
        } else {
            Ok(i32::from_le_bytes(b))
        }
    }
    #[inline]
    pub fn read_u32(&self, input: &mut impl BufRead) -> Result<u32> {
        let ii = self.read_i32(input)?;
        if ii >= 0 {
            Ok(ii as u32)
        } else {
            Err(GhwParseError::ExpectedPositiveInteger(ii as i64))
        }
    }

    #[inline]
    pub fn read_i64(&self, input: &mut impl BufRead) -> Result<i64> {
        let mut b = [0u8; 8];
        input.read_exact(&mut b)?;
        if self.big_endian {
            Ok(i64::from_be_bytes(b))
        } else {
            Ok(i64::from_le_bytes(b))
        }
    }
}

/// Information needed to read a signal value.
#[derive(Debug, Clone, Copy)]
pub struct GhwSignalInfo {
    tpe_and_vec: NonZeroU32,
    signal_ref: SignalRef,
}

impl GhwSignalInfo {
    pub fn new(tpe: SignalType, signal_ref: SignalRef, vector: Option<usize>) -> Self {
        let raw_tpe = ((tpe as u8) as u32) + 1;
        debug_assert_eq!(raw_tpe & 0x7, raw_tpe);
        let tpe_and_vec = if let Some(vector) = vector {
            let raw_vector_id = (vector as u32) + 1;
            debug_assert_eq!((raw_vector_id << 3) >> 3, raw_vector_id);
            NonZeroU32::new((raw_vector_id << 3) | raw_tpe).unwrap()
        } else {
            NonZeroU32::new(raw_tpe).unwrap()
        };
        Self {
            tpe_and_vec,
            signal_ref,
        }
    }

    pub fn signal_ref(&self) -> SignalRef {
        self.signal_ref
    }

    pub fn vec_id(&self) -> Option<GhwVecId> {
        let vec_id = self.tpe_and_vec.get() >> 3;
        if vec_id == 0 {
            None
        } else {
            Some(GhwVecId::new(vec_id as usize))
        }
    }

    pub fn tpe(&self) -> SignalType {
        let value = self.tpe_and_vec.get();
        let raw_tpe = (value & 0x7) as u8;

        SignalType::try_from_primitive(raw_tpe - 1).unwrap()
    }
}

pub type GhwDecodeInfo = (GhwSignals, Vec<GhwVecInfo>);

/// Contains information needed in order to decode value changes.
#[derive(Debug, Default)]
pub struct GhwSignals {
    /// Type and signal reference info. Indexed by `GhwSignalId`
    signals: Vec<GhwSignalInfo>,
}

impl GhwSignals {
    pub fn new(signals: Vec<GhwSignalInfo>) -> Self {
        Self { signals }
    }
    pub fn get_info(&self, signal_id: GhwSignalId) -> &GhwSignalInfo {
        &self.signals[signal_id.index()]
    }
    pub fn signal_len(&self) -> usize {
        self.signals.len()
    }
}

/// Pointer to a `GhwVecInfo`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GhwVecId(NonZeroU32);

impl GhwVecId {
    pub fn new(pos_id: usize) -> Self {
        Self(NonZeroU32::new(pos_id as u32).unwrap())
    }

    pub fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GhwSignalId(NonZeroU32);

impl GhwSignalId {
    #[inline]
    pub fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
    pub fn new(pos_id: u32) -> Self {
        Self(NonZeroU32::new(pos_id).unwrap())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GhwVecInfo {
    max: GhwSignalId,
    min: GhwSignalId,
    two_state: bool,
    signal_ref: SignalRef,
    /// pointer to an alias list
    alias: Option<NonZeroU32>,
}

impl GhwVecInfo {
    pub fn new(min: GhwSignalId, max: GhwSignalId, two_state: bool, signal_ref: SignalRef) -> Self {
        Self {
            min,
            max,
            two_state,
            signal_ref,
            alias: None,
        }
    }
    pub fn bits(&self) -> u32 {
        (self.max.index() - self.min.index() + 1) as u32
    }
    pub fn is_two_state(&self) -> bool {
        self.two_state
    }
    pub fn min(&self) -> GhwSignalId {
        self.min
    }
    pub fn max(&self) -> GhwSignalId {
        self.max
    }
    pub fn signal_ref(&self) -> SignalRef {
        self.signal_ref
    }
    pub fn alias(&self) -> Option<NonZeroU32> {
        self.alias
    }

    pub fn set_alias(&mut self, alias: NonZeroU32) {
        self.alias = Some(alias);
    }
}

/// Specifies the signal type info that is needed in order to read it.
#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone, TryFromPrimitive)]
pub enum SignalType {
    /// Nine value signal encoded as a single byte.
    NineState,
    /// A single bit in a 9 value bit vector.
    NineStateVec,
    /// Two value signal encoded as a single byte.
    TwoState,
    /// A single bit in a 2 value bit vector. bit N / M bits.
    TwoStateVec,
    /// Binary signal encoded as a single byte with N valid bits.
    U8,
    /// Binary signal encoded as a variable number of bytes with N valid bits.
    Leb128Signed,
    /// F64 (real)
    F64,
}

pub const GHW_GZIP_HEADER: &[u8; 2] = &[0x1f, 0x8b];
pub const GHW_BZIP2_HEADER: &[u8; 2] = b"BZ";

/// This is the header of the uncompressed file.
pub const GHW_HEADER_START: &[u8] = b"GHDLwave\n";

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone, TryFromPrimitive)]
pub enum GhwWellKnownType {
    Unknown = 0,
    Boolean = 1,
    Bit = 2,
    StdULogic = 3,
}

/// This enum used to be the same than the internal Ghdl rtik,
/// however in order to maintain backwards compatibility it was cloned.
#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone, TryFromPrimitive)]
pub enum GhwRtik {
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
pub enum GhwHierarchyKind {
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

/// The order in which the nine values appear in the STD_LOGIC enum.
pub const STD_LOGIC_VALUES: [u8; 9] = [b'u', b'x', b'0', b'1', b'z', b'w', b'l', b'h', b'-'];
/// Mapping from STD_LOGIC value to the `wellen` nine state encoding: ['0', '1', 'x', 'z', 'h', 'u', 'w', 'l', '-']
pub const STD_LOGIC_LUT: [u8; 9] = [5, 2, 0, 1, 3, 6, 7, 4, 8];

/// The order in which the two values appear in the BIT enum.
pub const VHDL_BIT_VALUES: [u8; 2] = [b'0', b'1'];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        // We store 8-bytes per GHW signal to have the SignalRef mapping, type and vector info available.
        // This is in comparison to ghwdump which stores at least two 8-bit pointers per signal.
        assert_eq!(std::mem::size_of::<GhwSignalInfo>(), 8);

        assert_eq!(std::mem::size_of::<GhwVecInfo>(), 20);
    }
}
