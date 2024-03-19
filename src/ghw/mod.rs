// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod common;
mod hierarchy;
mod signals;

pub(crate) use crate::ghw::common::GhwParseError;
use crate::ghw::common::{GhwDecodeInfo, HeaderData};
use crate::signals::SignalSource;
use crate::{Hierarchy, LoadOptions, TimeTable};
use std::io::{BufRead, Seek, SeekFrom};

/// Checks header to see if we are dealing with a GHW file.
pub(crate) fn is_ghw(input: &mut (impl BufRead + Seek)) -> bool {
    let is_ghw = hierarchy::read_ghw_header(input).is_ok();
    // try to reset input
    let _ = input.seek(SeekFrom::Start(0));
    is_ghw
}

pub(crate) type Result<T> = std::result::Result<T, GhwParseError>;

pub(crate) fn read_header(
    filename: &str,
    options: &LoadOptions,
) -> Result<(Hierarchy, ReadBodyContinuation)> {
    let f = std::fs::File::open(filename)?;
    let mut input = std::io::BufReader::new(f);
    let (hierarchy, header, decode_info) = read_header_internal(&mut input, options)?;
    let cont = ReadBodyContinuation {
        header,
        decode_info,
        input: Input::File(input),
    };
    Ok((hierarchy, cont))
}

pub(crate) fn read_header_from_bytes(
    bytes: Vec<u8>,
    options: &LoadOptions,
) -> Result<(Hierarchy, ReadBodyContinuation)> {
    let mut input = std::io::Cursor::new(bytes);
    let (hierarchy, header, decode_info) = read_header_internal(&mut input, options)?;
    let cont = ReadBodyContinuation {
        header,
        decode_info,
        input: Input::Bytes(input),
    };
    Ok((hierarchy, cont))
}

pub(crate) fn read_body(
    data: ReadBodyContinuation,
    hierarchy: &Hierarchy,
) -> Result<(SignalSource, TimeTable)> {
    let (source, time_table) = match data.input {
        Input::Bytes(mut input) => {
            signals::read_signals(&data.header, data.decode_info, hierarchy, &mut input)?
        }
        Input::File(mut input) => {
            signals::read_signals(&data.header, data.decode_info, hierarchy, &mut input)?
        }
    };
    Ok((source, time_table))
}

pub(crate) struct ReadBodyContinuation {
    header: HeaderData,
    decode_info: GhwDecodeInfo,
    input: Input,
}

enum Input {
    Bytes(std::io::Cursor<Vec<u8>>),
    File(std::io::BufReader<std::fs::File>),
}

fn read_header_internal(
    input: &mut (impl BufRead + Seek),
    _options: &LoadOptions,
) -> Result<(Hierarchy, HeaderData, GhwDecodeInfo)> {
    let header = hierarchy::read_ghw_header(input)?;
    let header_len = input.stream_position()?;

    // currently we do read the directory, however we are not using it yet
    let _sections = hierarchy::try_read_directory(&header, input)?;
    input.seek(SeekFrom::Start(header_len))?;
    // TODO: use actual section positions

    let (decode_info, hierarchy) = hierarchy::read_hierarchy(&header, input)?;
    Ok((hierarchy, header, decode_info))
}
