// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::{SignalIdx, SignalLength};
use crate::vcd::int_div_ceil;

pub type Time = u64;

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

pub struct Signal {
    idx: SignalIdx,
    time_indices: Vec<u32>,
    data: SignalChangeData,
}

impl Signal {
    pub(crate) fn new_fixed_len(idx: SignalIdx, time_indices: Vec<u32>, encoding: SignalEncoding, width: u32, bytes: Vec<u8>) -> Self {
        assert_eq!(time_indices.len(), bytes.len() / width as usize);
        let data = SignalChangeData::FixedLength { encoding, width, bytes };
        Signal { idx, time_indices, data }
    }

    pub(crate) fn new_var_len(idx: SignalIdx, time_indices: Vec<u32>, strings: Vec<String>) -> Self {
        assert_eq!(time_indices.len(), strings.len());
        let data = SignalChangeData::VariableLength(strings);
        Signal { idx, time_indices, data }
    }
}

/// Holds all loaded signals and facilitates access to them.
pub struct SignalDatabase {}

enum SignalChangeData {
    FixedLength {
        encoding: SignalEncoding,
        width: u32, // bytes per entry
        bytes: Vec<u8>,
    },
    VariableLength(Vec<String>),
}

pub trait SignalSource {
    /// Loads new signals.
    /// Many implementations take advantage of loading multiple signals at a time.
    fn load_signals(&mut self, ids: &[(SignalIdx, SignalLength)]) -> Vec<Signal>;
    /// Returns the global time table which stores the time at each value change.
    fn get_time_table(&self) -> Vec<Time>;
}

#[inline]
fn byte_and_bit_index(ii: usize, max_byte_ii: usize, bits_per_byte: usize) -> (usize, usize) {
    (max_byte_ii - ii / bits_per_byte, ii % bits_per_byte)
}

fn binary_to_four_value(bits: usize, value: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; int_div_ceil(bits, 4)];
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
