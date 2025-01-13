// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// A simpler interface to load waves. Use this instead of `wellen::viewers` if you are building
// a batch processing instead of a GUI app.

use crate::{
    viewers, Hierarchy, LoadOptions, Result, Signal, SignalRef, SignalSource, Time, TimeTable,
};
use rustc_hash::FxHashMap;
use std::fmt::{Debug, Formatter};
use std::io::{BufRead, Seek};

/// Read a waveform file with the default options. Reads in header and body at once.
pub fn read<P: AsRef<std::path::Path>>(filename: P) -> Result<Waveform> {
    read_with_options(filename, &LoadOptions::default())
}

/// Read a waveform file. Reads in header and body at once.
pub fn read_with_options<P: AsRef<std::path::Path>>(
    filename: P,
    options: &LoadOptions,
) -> Result<Waveform> {
    let header = viewers::read_header_from_file(filename, options)?;
    let body = viewers::read_body(header.body, &header.hierarchy, None)?;
    Ok(Waveform::new(
        header.hierarchy,
        body.source,
        body.time_table,
    ))
}

/// Read from something that is not a file.
pub fn read_from_reader<R: BufRead + Seek + Send + Sync + 'static>(input: R) -> Result<Waveform> {
    let options = LoadOptions::default();
    let header = viewers::read_header(input, &options)?;
    let body = viewers::read_body(header.body, &header.hierarchy, None)?;
    Ok(Waveform::new(
        header.hierarchy,
        body.source,
        body.time_table,
    ))
}

/// Provides file format independent access to a waveform file.
pub struct Waveform {
    hierarchy: Hierarchy,
    source: SignalSource,
    time_table: TimeTable,
    /// Signals are stored in a HashMap since we expect only a small subset of signals to be
    /// loaded at a time.
    signals: FxHashMap<SignalRef, Signal>,
}

impl Debug for Waveform {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Waveform(...)")
    }
}

impl Waveform {
    fn new(hierarchy: Hierarchy, source: SignalSource, time_table: TimeTable) -> Self {
        Waveform {
            hierarchy,
            source,
            time_table,
            signals: FxHashMap::default(),
        }
    }

    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    pub fn time_table(&self) -> &[Time] {
        &self.time_table
    }

    fn load_signals_internal(&mut self, ids: &[SignalRef], multi_threaded: bool) {
        // make sure that we do not load signals that have already been loaded
        let filtered_ids = ids
            .iter()
            .filter(|id| !self.signals.contains_key(id))
            .cloned()
            .collect::<Vec<_>>();

        let res = self
            .source
            .load_signals(&filtered_ids, &self.hierarchy, multi_threaded);
        for (id, signal) in res.into_iter() {
            self.signals.insert(id, signal);
        }
    }

    pub fn load_signals(&mut self, ids: &[SignalRef]) {
        self.load_signals_internal(ids, false)
    }

    pub fn load_signals_multi_threaded(&mut self, ids: &[SignalRef]) {
        self.load_signals_internal(ids, true)
    }

    pub fn unload_signals(&mut self, ids: &[SignalRef]) {
        for id in ids.iter() {
            self.signals.remove(id);
        }
    }

    pub fn get_signal(&self, id: SignalRef) -> Option<&Signal> {
        self.signals.get(&id)
    }

    pub fn print_backend_statistics(&self) {
        self.source.print_statistics();
    }
}
