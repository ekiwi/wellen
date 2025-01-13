// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

mod common;
mod hierarchy;
mod signals;

pub use crate::ghw::common::GhwParseError;
use crate::ghw::common::{GhwDecodeInfo, HeaderData};
use crate::signals::SignalSource;
use crate::viewers::ProgressCount;
use crate::{Hierarchy, LoadOptions, TimeTable};
use std::io::{BufRead, Seek, SeekFrom};
use std::sync::atomic::Ordering;

/// Checks header to see if we are dealing with a GHW file.
pub fn is_ghw(input: &mut (impl BufRead + Seek)) -> bool {
    let is_ghw = hierarchy::read_ghw_header(input).is_ok();
    // try to reset input
    let _ = input.seek(SeekFrom::Start(0));
    is_ghw
}

pub type Result<T> = std::result::Result<T, GhwParseError>;

pub fn read_header<R: BufRead + Seek>(
    mut input: R,
    options: &LoadOptions,
) -> Result<(Hierarchy, ReadBodyContinuation<R>, u64)> {
    let (hierarchy, header, decode_info, body_len) = read_header_internal(&mut input, options)?;
    let cont = ReadBodyContinuation {
        header,
        decode_info,
        input,
    };
    Ok((hierarchy, cont, body_len))
}

pub fn read_body<R: BufRead + Seek>(
    data: ReadBodyContinuation<R>,
    hierarchy: &Hierarchy,
    progress: Option<ProgressCount>,
) -> Result<(SignalSource, TimeTable)> {
    let mut input = data.input;
    match progress {
        Some(p) => {
            let mut wrapped = ProgressTracker::new(input, p);
            signals::read_signals(&data.header, data.decode_info, hierarchy, &mut wrapped)
        }
        None => signals::read_signals(&data.header, data.decode_info, hierarchy, &mut input),
    }
}

pub struct ReadBodyContinuation<R: BufRead + Seek> {
    header: HeaderData,
    decode_info: GhwDecodeInfo,
    input: R,
}

fn read_header_internal(
    input: &mut (impl BufRead + Seek),
    _options: &LoadOptions,
) -> Result<(Hierarchy, HeaderData, GhwDecodeInfo, u64)> {
    let header = hierarchy::read_ghw_header(input)?;
    let header_len = input.stream_position()?;

    // currently we do read the directory, however we are not using it yet
    let _sections = hierarchy::try_read_directory(&header, input)?;
    input.seek(SeekFrom::Start(header_len))?;
    // TODO: use actual section positions

    let (decode_info, hierarchy) = hierarchy::read_hierarchy(&header, input)?;

    // determine body length
    let body_start = input.stream_position()?;
    input.seek(SeekFrom::End(0))?;
    let file_size = input.stream_position()?;
    input.seek(SeekFrom::Start(body_start))?;
    let body_len = file_size - body_start;

    Ok((hierarchy, header, decode_info, body_len))
}

struct ProgressTracker<T: BufRead> {
    inner: T,
    progress: ProgressCount,
}

impl<T: BufRead> ProgressTracker<T> {
    fn new(inner: T, progress: ProgressCount) -> Self {
        Self { inner, progress }
    }
}

impl<T: BufRead> std::io::Read for ProgressTracker<T> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = self.inner.read(buf)?;
        self.progress.fetch_add(len as u64, Ordering::SeqCst);
        Ok(len)
    }
}

impl<T: BufRead> BufRead for ProgressTracker<T> {
    #[inline]
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    #[inline]
    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt);
        self.progress.fetch_add(amt as u64, Ordering::SeqCst);
    }
}
