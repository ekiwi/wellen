// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Fast and compact wave-form representation inspired by the FST on disk format.

/// Holds queryable waveform data. Use the `Encoder` to generate.
pub struct Reader {}

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
