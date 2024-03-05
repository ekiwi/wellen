// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod common;
mod hierarchy;
mod signals;

use crate::{Waveform, WellenError};
use std::io::{BufRead, Seek, SeekFrom};

/// Checks header to see if we are dealing with a GHW file.
pub(crate) fn is_ghw(input: &mut (impl BufRead + Seek)) -> bool {
    let is_ghw = hierarchy::read_ghw_header(input).is_ok();
    // try to reset input
    let _ = input.seek(SeekFrom::Start(0));
    is_ghw
}

pub fn read(filename: &str) -> Result<Waveform, WellenError> {
    let f = std::fs::File::open(filename)?;
    let mut input = std::io::BufReader::new(f);
    read_internal(&mut input)
}

pub fn read_from_bytes(bytes: Vec<u8>) -> Result<Waveform, WellenError> {
    let mut input = std::io::Cursor::new(bytes);
    read_internal(&mut input)
}

fn read_internal(input: &mut (impl BufRead + Seek)) -> std::result::Result<Waveform, WellenError> {
    let header = hierarchy::read_ghw_header(input)?;
    let header_len = input.stream_position()?;

    // currently we do read the directory, however we are not using it yet
    let _sections = hierarchy::try_read_directory(&header, input)?;
    input.seek(SeekFrom::Start(header_len))?;
    // TODO: use actual section positions

    let (decode_info, hierarchy) = hierarchy::read_hierarchy(&header, input)?;
    let wave_mem = signals::read_signals(&header, decode_info, &hierarchy, input)?;
    Ok(Waveform::new(hierarchy, wave_mem))
}
