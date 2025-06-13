// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
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

pub struct HeaderResult<R: BufRead + Seek> {
    pub hierarchy: Hierarchy,
    pub file_format: FileFormat,
    /// Body length in bytes.
    pub body_len: u64,
    pub body: ReadBodyContinuation<R>,
}

pub fn read_header_from_file<P: AsRef<std::path::Path>>(
    filename: P,
    options: &LoadOptions,
) -> Result<HeaderResult<std::io::BufReader<std::fs::File>>> {
    let file_format = open_and_detect_file_format(filename.as_ref());
    match file_format {
        FileFormat::Unknown => Err(WellenError::UnknownFileFormat),
        FileFormat::Vcd => {
            let (hierarchy, body, body_len) = crate::vcd::read_header_from_file(filename, options)?;
            let body = ReadBodyContinuation(ReadBodyData::Vcd(Box::new(body)));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len,
                body,
            })
        }
        FileFormat::Ghw => {
            let input = std::io::BufReader::new(std::fs::File::open(filename)?);
            let (hierarchy, body, body_len) = crate::ghw::read_header(input, options)?;
            let body = ReadBodyContinuation(ReadBodyData::Ghw(Box::new(body)));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len,
                body,
            })
        }
        FileFormat::Fst => {
            let (hierarchy, body) = crate::fst::read_header_from_file(filename, options)?;
            let body = ReadBodyContinuation(ReadBodyData::Fst(Box::new(body)));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len: 0, // fst never reads the full body (unless all signals are displayed)
                body,
            })
        }
    }
}

pub fn read_header<R: BufRead + Seek>(
    mut input: R,
    options: &LoadOptions,
) -> Result<HeaderResult<R>> {
    let file_format = detect_file_format(&mut input);
    match file_format {
        FileFormat::Unknown => Err(WellenError::UnknownFileFormat),
        FileFormat::Vcd => {
            let (hierarchy, body, body_len) = crate::vcd::read_header(input, options)?;
            let body = ReadBodyContinuation(ReadBodyData::Vcd(Box::new(body)));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len,
                body,
            })
        }
        FileFormat::Ghw => {
            let (hierarchy, body, body_len) = crate::ghw::read_header(input, options)?;
            let body = ReadBodyContinuation(ReadBodyData::Ghw(Box::new(body)));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len,
                body,
            })
        }
        FileFormat::Fst => {
            let (hierarchy, body) = crate::fst::read_header(input, options)?;
            let body = ReadBodyContinuation(ReadBodyData::Fst(Box::new(body)));
            Ok(HeaderResult {
                hierarchy,
                file_format,
                body_len: 0, // fst never reads the full body (unless all signals are displayed)
                body,
            })
        }
    }
}

pub struct ReadBodyContinuation<R: BufRead + Seek>(ReadBodyData<R>);

enum ReadBodyData<R: BufRead + Seek> {
    Vcd(Box<crate::vcd::ReadBodyContinuation<R>>),
    Fst(Box<crate::fst::ReadBodyContinuation<R>>),
    Ghw(Box<crate::ghw::ReadBodyContinuation<R>>),
}

pub struct BodyResult {
    pub source: SignalSource,
    pub time_table: TimeTable,
}

pub type ProgressCount = std::sync::Arc<std::sync::atomic::AtomicU64>;

pub fn read_body<R: BufRead + Seek + Sync + Send + 'static>(
    body: ReadBodyContinuation<R>,
    hierarchy: &Hierarchy,
    progress: Option<ProgressCount>,
) -> Result<BodyResult> {
    match body.0 {
        ReadBodyData::Vcd(data) => {
            let (source, time_table) = crate::vcd::read_body(*data, hierarchy, progress)?;
            Ok(BodyResult { source, time_table })
        }
        ReadBodyData::Fst(data) => {
            // fst does not support a progress count since it is not actually reading the body
            let (source, time_table) = crate::fst::read_body(*data)?;
            Ok(BodyResult { source, time_table })
        }
        ReadBodyData::Ghw(data) => {
            let (source, time_table) = crate::ghw::read_body(*data, hierarchy, progress)?;
            Ok(BodyResult { source, time_table })
        }
    }
}

/// Tries to guess the format of the file.
pub fn open_and_detect_file_format<P: AsRef<std::path::Path>>(filename: P) -> FileFormat {
    let input_file = std::fs::File::open(filename).expect("failed to open input file!");
    let mut reader = std::io::BufReader::new(input_file);
    detect_file_format(&mut reader)
}

/// Tries to guess the file format used by the input.
pub fn detect_file_format(input: &mut (impl BufRead + Seek)) -> FileFormat {
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
