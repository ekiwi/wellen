// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::{Hierarchy, SignalLength, SignalRef};
use crate::vcd::usize_div_ceil;
use std::collections::HashMap;
use std::fmt::Debug;

pub type Time = u64;

#[derive(Debug, Clone, Copy)]
pub enum SignalValue<'a> {
    Binary(&'a [u8]),
    String(&'a str),
}
/// Specifies the encoding of a signal.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SignalEncoding {
    /// Each bit is encoded as a single bit.
    Binary(u32),
    /// Each bit is encoded as two bits.
    FourValue(u32),
    /// Fixed length ASCII string.
    FixedLength,
    /// Each value is encoded as an 8-byte f64 in little endian.
    Float,
    /// Variable length string.
    VariableLength,
}

pub(crate) struct Signal {
    idx: SignalRef,
    time_indices: Vec<u32>,
    data: SignalChangeData,
}

impl Signal {
    pub(crate) fn new_fixed_len(
        idx: SignalRef,
        time_indices: Vec<u32>,
        encoding: SignalEncoding,
        width: u32,
        bytes: Vec<u8>,
    ) -> Self {
        assert_eq!(time_indices.len(), bytes.len() / width as usize);
        let data = SignalChangeData::FixedLength {
            encoding,
            width,
            bytes,
        };
        Signal {
            idx,
            time_indices,
            data,
        }
    }

    pub(crate) fn new_var_len(
        idx: SignalRef,
        time_indices: Vec<u32>,
        strings: Vec<String>,
    ) -> Self {
        assert_eq!(time_indices.len(), strings.len());
        let data = SignalChangeData::VariableLength(strings);
        Signal {
            idx,
            time_indices,
            data,
        }
    }

    pub(crate) fn size_in_memory(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let time = self.time_indices.len() * std::mem::size_of::<u32>();
        let data = match &self.data {
            SignalChangeData::FixedLength { bytes, .. } => bytes.len(),
            SignalChangeData::VariableLength(strings) => strings
                .iter()
                .map(|s| s.as_bytes().len() + std::mem::size_of::<String>())
                .sum::<usize>(),
        };
        base + time + data
    }
}

/// Provides file format independent access to a waveform file.
pub struct Waveform {
    hierarchy: Hierarchy,
    source: Box<dyn SignalSource + Send>,
    time_table: Vec<Time>,
    /// Signals are stored in a HashMap since we expect only a small subset of signals to be
    /// loaded at a time.
    signals: HashMap<SignalRef, Signal>,
}

impl Debug for Waveform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Waveform(...)")
    }
}

impl Waveform {
    pub(crate) fn new(hierarchy: Hierarchy, source: Box<dyn SignalSource + Send>) -> Self {
        let time_table = source.get_time_table();
        Waveform {
            hierarchy,
            source,
            time_table,
            signals: HashMap::new(),
        }
    }

    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    pub fn load_signals(&mut self, ids: &[SignalRef]) {
        let ids_with_len = ids
            .iter()
            .map(|i| (*i, self.hierarchy.get_signal_length(*i).unwrap()))
            .collect::<Vec<_>>();
        let signals = self.source.load_signals(&ids_with_len);
        // the signal source must always return the correct number of signals!
        assert_eq!(signals.len(), ids.len());
        for (id, signal) in ids.iter().zip(signals.into_iter()) {
            self.signals.insert(*id, signal);
        }
    }

    pub fn unload_signals(&mut self, ids: &[SignalRef]) {
        for id in ids.iter() {
            self.signals.remove(id);
        }
    }

    pub fn get_signal_size_in_memory(&self, id: SignalRef) -> Option<usize> {
        Some((self.signals.get(&id)?).size_in_memory())
    }
}

enum SignalChangeData {
    FixedLength {
        encoding: SignalEncoding,
        width: u32, // bytes per entry
        bytes: Vec<u8>,
    },
    VariableLength(Vec<String>),
}

pub(crate) trait SignalSource {
    /// Loads new signals.
    /// Many implementations take advantage of loading multiple signals at a time.
    fn load_signals(&mut self, ids: &[(SignalRef, SignalLength)]) -> Vec<Signal>;
    /// Returns the global time table which stores the time at each value change.
    fn get_time_table(&self) -> Vec<Time>;
}

#[inline]
fn byte_and_bit_index(ii: usize, max_byte_ii: usize, bits_per_byte: usize) -> (usize, usize) {
    (max_byte_ii - ii / bits_per_byte, ii % bits_per_byte)
}

fn binary_to_four_value(bits: usize, value: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; usize_div_ceil(bits, 4)];
    let max_value_ii = value.len() - 1;
    let max_out_ii = out.len() - 1;
    for ii in 0..bits {
        let (in_byte_index, in_bit_index) = byte_and_bit_index(ii, max_value_ii, 8);
        let is_active = (value[in_byte_index] >> in_bit_index) & 1 == 1;
        let (out_byte_index, out_bit_index) = byte_and_bit_index(ii, max_out_ii, 4);
        if is_active {
            out[out_byte_index] |= 1 << out_bit_index;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        assert_eq!(std::mem::size_of::<SignalRef>(), 4);

        // 4 bytes for length + tag + padding
        assert_eq!(std::mem::size_of::<SignalEncoding>(), 8);

        assert_eq!(std::mem::size_of::<SignalChangeData>(), 40);
        assert_eq!(std::mem::size_of::<Signal>(), 72);

        // since there is some empty space in the Signal struct, we can make it an option for free!
        assert_eq!(std::mem::size_of::<Option<Signal>>(), 72);
    }
}
