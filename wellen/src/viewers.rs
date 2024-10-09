// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Interface for waveform viewers

use crate::{FileFormat, Hierarchy, LoadOptions, Result, SignalSource, TimeTable, WellenError};
use std::io::{BufRead, Seek};

impl From<crate::ghw::GhwParseError> for WellenError {
    fn from(value: crate::ghw::GhwParseError) -> Self {
        WellenError::FailedToLoad(FileFormat::Ghw, value.to_string())
    }
}

impl From<crate::vcd::VcdParseError> for WellenError {
    fn from(value: crate::vcd::VcdParseError) -> Self {
        WellenError::FailedToLoad(FileFormat::Vcd, value.to_string())
    }
}

impl From<fst_reader::ReaderError> for WellenError {
    fn from(value: fst_reader::ReaderError) -> Self {
        WellenError::FailedToLoad(FileFormat::Fst, value.to_string())
    }
}

pub struct HeaderResult {
    pub hierarchy: Hierarchy,
    pub file_format: FileFormat,
    /// Body length in bytes.
    pub body_len: u64,
    pub body: ReadBodyContinuation,
}

pub fn read_header(filename: &str, options: &LoadOptions) -> Result<HeaderResult> {
    let file_format = open_and_detect_file_format(filename);
    match file_format {
        FileFormat::Unknown => Err(WellenError::UnknownFileFormat),
        FileFormat::Vcd => {
            let (hierarchy, body, body_len) = crate::vcd::read_header(filename, options)?;
            let body = ReadBodyContinuation::new(ReadBodyData::Vcd(body));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len,
                body,
            })
        }
        FileFormat::Ghw => {
            let (hierarchy, body, body_len) = crate::ghw::read_header(filename, options)?;
            let body = ReadBodyContinuation::new(ReadBodyData::Ghw(body));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len,
                body,
            })
        }
        FileFormat::Fst => {
            let input = std::io::BufReader::new(std::fs::File::open(filename)?);
            let (hierarchy, body) = crate::fst::read_header(input, options)?;
            let body = ReadBodyContinuation::new(ReadBodyData::Fst(body));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len: 0, // fst never reads the full body (unless all signals are displayed)
                body,
            })
        }
    }
}

pub fn read_header_from_bytes(bytes: Vec<u8>, options: &LoadOptions) -> Result<HeaderResult> {
    read_header_from_reader(std::io::Cursor::new(bytes), options)
}

pub fn read_header_from_reader<R>(mut input: R, options: &LoadOptions) -> Result<HeaderResult>
where
    R: std::io::BufRead + std::io::Seek,
{
    // remember where we are supposed to start reading
    let start = input.stream_position()?;
    let file_format = { detect_file_format(&mut input) };
    match file_format {
        FileFormat::Unknown => Err(WellenError::UnknownFileFormat),
        FileFormat::Vcd => {
            todo!()
            // let (hierarchy, body, body_len) = crate::vcd::read_header_from_bytes(bytes, options)?;
            // let body = ReadBodyContinuation::new(ReadBodyData::Vcd(body));
            // Ok(HeaderResult {
            //     hierarchy,
            //     file_format,
            //     body_len,
            //     body,
            // })
        }
        FileFormat::Ghw => {
            todo!()
            // let (hierarchy, body, body_len) = crate::ghw::read_header_from_bytes(bytes, options)?;
            // let body = ReadBodyContinuation::new(ReadBodyData::Ghw(body));
            // Ok(HeaderResult {
            //     hierarchy,
            //     file_format,
            //     body_len,
            //     body,
            // })
        }
        FileFormat::Fst => {
            let (hierarchy, body) = crate::fst::read_header(input, options)?;
            let body = ReadBodyContinuation::new(ReadBodyData::Fst(body));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len: 0, // fst never reads the full body (unless all signals are displayed)
                body,
            })
        }
    }
}

pub struct ReadBodyContinuation {
    data: ReadBodyData,
}

impl ReadBodyContinuation {
    fn new(data: ReadBodyData) -> Self {
        Self { data }
    }
}

enum ReadBodyData {
    Vcd(crate::vcd::ReadBodyContinuation),
    Fst(crate::fst::ReadBodyContinuation),
    Ghw(crate::ghw::ReadBodyContinuation),
}

pub struct BodyResult {
    pub source: SignalSource,
    pub time_table: TimeTable,
}

pub type ProgressCount = std::sync::Arc<std::sync::atomic::AtomicU64>;

pub fn read_body(
    body: ReadBodyContinuation,
    hierarchy: &Hierarchy,
    progress: Option<ProgressCount>,
) -> Result<BodyResult> {
    match body.data {
        ReadBodyData::Vcd(data) => {
            let (source, time_table) = crate::vcd::read_body(data, hierarchy, progress)?;
            Ok(BodyResult { source, time_table })
        }
        ReadBodyData::Fst(data) => {
            // fst does not support a progress count since it is no actually reading the body
            let (source, time_table) = crate::fst::read_body(data)?;
            Ok(BodyResult { source, time_table })
        }
        ReadBodyData::Ghw(data) => {
            let (source, time_table) = crate::ghw::read_body(data, hierarchy, progress)?;
            Ok(BodyResult { source, time_table })
        }
    }
}

/// Tries to guess the format of the file.
pub fn open_and_detect_file_format(filename: &str) -> FileFormat {
    let input_file = std::fs::File::open(filename).expect("failed to open input file!");
    let mut reader = std::io::BufReader::new(input_file);
    detect_file_format(&mut reader)
}

/// Tries to guess the file format used by the input.
fn detect_file_format(input: &mut (impl BufRead + Seek)) -> FileFormat {
    if crate::vcd::is_vcd(input) {
        FileFormat::Vcd
    } else if fst_reader::is_fst_file(input) {
        FileFormat::Fst
    } else if crate::ghw::is_ghw(input) {
        FileFormat::Ghw
    } else {
        FileFormat::Unknown
    }
}
