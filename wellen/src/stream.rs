// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//! # Stream Interface
//! This interface is useful when you want to batch process waveform data instead
//! of displaying it in a waveform viewer.

use crate::{
    viewers, FileFormat, Hierarchy, LoadOptions, Result, SignalRef, SignalValue, Time, WellenError,
};
use std::fmt::{Debug, Formatter};
use std::io::{BufRead, Seek};

/// Read a waveform file. Reads only the header.
pub fn read_from_file<P: AsRef<std::path::Path>>(
    filename: P,
    options: &LoadOptions,
) -> Result<StreamingWaveform<std::io::BufReader<std::fs::File>>> {
    let file_format = viewers::open_and_detect_file_format(filename.as_ref());
    match file_format {
        FileFormat::Unknown => Err(WellenError::UnknownFileFormat),
        FileFormat::Vcd => {
            let (hierarchy, body, _body_len) =
                crate::vcd::read_header_from_file(filename, options)?;
            Ok(StreamingWaveform {
                hierarchy,
                body: StreamBody::Vcd(body),
            })
        }
        FileFormat::Ghw => {
            todo!("streaming for ghw")
        }
        FileFormat::Fst => {
            todo!("streaming for fst")
        }
    }
}

/// Read from something that is not a file. Reads only the header.
pub fn read<R: BufRead + Seek + Send + Sync + 'static>(
    input: R,
    options: &LoadOptions,
) -> Result<StreamingWaveform<R>> {
    todo!("support streaming read from things that are not files")
}

/// Represents a waveform that was loaded for streaming.
pub struct StreamingWaveform<R: BufRead + Seek> {
    hierarchy: Hierarchy,
    body: StreamBody<R>,
}

enum StreamBody<R: BufRead + Seek> {
    Vcd(crate::vcd::ReadBodyContinuation<R>),
}

impl<R: BufRead + Seek> Debug for StreamingWaveform<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream::Waveform(...)")
    }
}

/// Defines which time interval and signals to include when streaming value changes.
pub struct Filter<'a> {
    /// First time change.
    pub start: Time,
    /// Last time change. `None` means until the end of the file.
    pub end: Option<Time>,
    /// `None` means all signals.
    pub signals: Option<&'a [SignalRef]>,
}

impl<'a> Filter<'a> {
    /// Include all value changes.
    pub fn all() -> Self {
        Filter {
            start: 0,
            end: None,
            signals: None,
        }
    }

    pub fn new(start: u64, end: u64, signals: &'a [SignalRef]) -> Self {
        Filter {
            start,
            end: Some(end),
            signals: Some(signals),
        }
    }
}

impl<R: BufRead + Seek> StreamingWaveform<R> {
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    pub fn stream(
        &mut self,
        filter: &Filter,
        callback: impl FnMut(Time, SignalRef, SignalValue<'_>),
    ) -> Result<()> {
        match &mut self.body {
            StreamBody::Vcd(data) => {
                crate::vcd::stream_body(data, &self.hierarchy, filter, callback)?
            }
        }
        Ok(())
    }
}
