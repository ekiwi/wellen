// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::SignalIdx;
use crate::values::Time;

/// Specifies the encoding of a signal.
enum SignalEncoding {
    /// Each bit is encoded as a single bit.
    Binary,
    /// Each bit is encoded as two bits.
    FourValue,
    /// Each value is encoded as an 8-byte f64 in little endian.
    Float,
}

pub struct Signal {
    idx: SignalIdx,
    time_indices: Vec<u32>,
    data: SignalChangeData,
}

enum SignalChangeData {
    FixedLength {
        encoding: SignalEncoding,
        width: usize,
        bytes: Vec<u8>,
    },
    VariableLength(Vec<String>),
}

pub trait WaveDatabase {
    /// Loads new signals into the database.
    /// Many implementations take advantage of loading multiple signals at a time.
    fn load_signals(&mut self, ids: &[SignalIdx]);

    /// Access a signal.
    fn get_signal(&self, idx: SignalIdx) -> &Signal;
    /// Access the global time table which stores the time at each value change.
    fn get_time_table(&self) -> Vec<Time>;
}
