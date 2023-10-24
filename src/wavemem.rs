// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Fast and compact wave-form representation inspired by the FST on disk format.

use crate::hierarchy::SignalIdx;
use crate::signals::{Signal, SignalSource};
use crate::values::Time;

/// Holds queryable waveform data. Use the `Encoder` to generate.
pub struct Reader {
    blocks: Vec<Block>,
}

impl SignalSource for Reader {
    fn load_signals(&mut self, ids: &[SignalIdx]) -> Vec<Signal> {
        todo!()
    }

    fn get_time_table(&self) -> Vec<Time> {
        todo!()
    }
}

/// A block that contains all value changes in a certain time segment.
/// Note that while in FST blocks can be skipped, here we only use blocks
/// in order to combine data from different threads and to compress partial data.
struct Block {
    start_time: Time,
    time_table: Vec<Time>,
}

impl Block {
    fn load_signals(&self, ids: u32) {}
}

/// Encodes value and time changes into a compressed in-memory representation.
pub struct Encoder {}

impl Default for Encoder {
    fn default() -> Self {
        Encoder {}
    }
}

impl Encoder {
    pub fn time_change(&mut self, time: u64) {
        todo!()
    }

    /// Call with an unaltered VCD value.
    pub fn vcd_value_change(&mut self, id: u64, value: &[u8]) {}

    pub fn finish(self) -> Reader {
        todo!()
    }

    // appends the contents of the other encoder to this one
    pub fn append(&mut self, other: Encoder) {
        todo!()
    }
}

/// Encodes changes for a single signal.
struct SignalEncoder {}

impl Default for SignalEncoder {
    fn default() -> Self {
        SignalEncoder {}
    }
}

impl SignalEncoder {
    fn add_change(&mut self, time_index: u64, value: &[u8]) {
        todo!()
    }

    /// returns a compressed signal representation
    fn finish(self) -> Vec<u8> {
        todo!()
    }
}
