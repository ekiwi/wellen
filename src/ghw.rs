// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

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

#[derive(Debug)]
struct HeaderData {
    version: u8,
    big_endian: bool,
    word_len: u8,
    word_offset: u8,
}

#[inline]
pub(crate) fn read_bytes(input: &mut impl Read, len: usize) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    input.take(len as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

const GHW_GZIP_HEADER: &[u8; 2] = &[0x1f, 0x8b];
const GHW_BZIP2_HEADER: &[u8; 2] = b"BZ";

/// This is the header of the uncompressed file.
const GHW_HEADER_START: &[u8] = b"GHDLwave\n";

#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
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
    CaseGenerate = 9,
    ForGenerate = 10,
    GenerateBody = 11,
    Instance = 12,
    Constant = 13,
    Iterator = 14,
    Variable = 15,
    Signal = 16,
    File = 17,
    Port = 18,
    Generic = 19,
    Alias = 20,
    Guard = 21,
    Component = 22,
    Attribute = 23,
    TypeB1 = 24,
    TypeE8 = 25,
    TypeE32 = 26,
    TypeI32 = 27,
    TypeI64 = 28,
    TypeF64 = 29,
    TypeP32 = 30,
    TypeP64 = 31,
    TypeAccess = 32,
    TypeArray = 33,
    TypeRecord = 34,
    TypeUnboundedRecord = 35,
    TypeFile = 36,
    SubtypeScalar = 37,
    SubtypeArray = 38,
    SubtypeUnboundedArray = 39,
    SubtypeRecord = 40,
    SubtypeUnboundedRecord = 41,
    SubtypeAccess = 42,
    TypeProtected = 43,
    Element = 44,
    Unit64 = 45,
    Unitptr = 46,
    AttributeTransaction = 47,
    AttributeQuiet = 48,
    AttributeStable = 49,
    PslAssert = 50,
    PslAssume = 51,
    PslCover = 52,
    PslEndpoint = 53,
    Error = 54,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_load() {
        load("inputs/ghdl/tb_recv.ghw").unwrap();
    }
}
