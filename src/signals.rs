// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::{Hierarchy, SignalLength, SignalRef};
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};

pub type Time = u64;
pub type TimeTableIdx = u32;

#[derive(Debug, Clone, Copy)]
pub enum SignalValue<'a> {
    Binary(&'a [u8], u32),
    FourValue(&'a [u8], u32),
    String(&'a str),
    Float(f64),
}

impl<'a> Display for SignalValue<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            SignalValue::Binary(data, bits) => {
                write!(f, "{}", two_state_to_bit_string(data, *bits))
            }
            SignalValue::FourValue(data, bits) => {
                write!(f, "{}", four_state_to_bit_string(data, *bits))
            }
            SignalValue::String(value) => write!(f, "{}", value),
            SignalValue::Float(value) => write!(f, "{}", value),
        }
    }
}

impl<'a> SignalValue<'a> {
    pub fn to_bit_string(&self) -> Option<String> {
        match &self {
            SignalValue::Binary(data, bits) => Some(two_state_to_bit_string(data, *bits)),
            SignalValue::FourValue(data, bits) => Some(four_state_to_bit_string(data, *bits)),
            _ => None,
        }
    }
}

fn two_state_to_bit_string(data: &[u8], bits: u32) -> String {
    let mut out = String::with_capacity(bits as usize);
    if bits == 0 {
        return out;
    }

    // the first byte might not contain a full 8 bits
    let byte0_bits = bits - ((bits / 8) * 8);
    let byte0_is_special = byte0_bits > 0;
    if byte0_is_special {
        let byte0 = data[0];
        for ii in (0..byte0_bits).rev() {
            let value = (byte0 >> ii) & 1;
            let char = ['0', '1'][value as usize];
            out.push(char);
        }
    }

    for byte in data.iter().skip(if byte0_is_special { 1 } else { 0 }) {
        for ii in (0..8).rev() {
            let value = (byte >> ii) & 1;
            let char = ['0', '1'][value as usize];
            out.push(char);
        }
    }
    out
}

fn four_state_to_bit_string(data: &[u8], bits: u32) -> String {
    let mut out = String::with_capacity(bits as usize);
    if bits == 0 {
        return out;
    }

    // the first byte might not contain a full 4 bits
    let byte0_bits = bits - ((bits / 4) * 4);
    let byte0_is_special = byte0_bits > 0;
    if byte0_is_special {
        let byte = data[0];
        for ii in (0..byte0_bits).rev() {
            let value = (byte >> (ii * 2)) & 3;
            let char = ['0', '1', 'x', 'z'][value as usize];
            out.push(char);
        }
    }

    for byte in data.iter().skip(if byte0_is_special { 1 } else { 0 }) {
        for ii in (0..4).rev() {
            let value = (byte >> (ii * 2)) & 3;
            let char = ['0', '1', 'x', 'z'][value as usize];
            out.push(char);
        }
    }
    out
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
    time_indices: Vec<TimeTableIdx>,
    data: SignalChangeData,
}

impl Signal {
    pub(crate) fn new_fixed_len(
        idx: SignalRef,
        time_indices: Vec<TimeTableIdx>,
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
        time_indices: Vec<TimeTableIdx>,
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
        let time = self.time_indices.len() * std::mem::size_of::<TimeTableIdx>();
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

    pub fn time_table(&self) -> &[Time] {
        &self.time_table
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

    pub fn get_signal_value_at(
        &self,
        signal_ref: SignalRef,
        time_table_idx: TimeTableIdx,
    ) -> SignalValue {
        assert!((time_table_idx as usize) < self.time_table.len());
        let signal: &Signal = &self.signals[&signal_ref];
        let offset = find_offset_from_time_table_idx(&signal.time_indices, time_table_idx);
        signal.data.get_value_at(offset)
    }
}

/// Finds the index that is the same or less than the needle and returns the position of it.
/// Note that `indices` needs to sorted from smallest to largest.
/// Essentially implements a binary search!
fn find_offset_from_time_table_idx(indices: &[TimeTableIdx], needle: TimeTableIdx) -> usize {
    let mut lower_idx = 0usize;
    let mut upper_idx = indices.len() - 1;
    while lower_idx <= upper_idx {
        let mid_idx = lower_idx + ((upper_idx - lower_idx) / 2);

        match indices[mid_idx].cmp(&needle) {
            std::cmp::Ordering::Less => {
                lower_idx = mid_idx + 1;
            }
            std::cmp::Ordering::Equal => {
                return mid_idx;
            }
            std::cmp::Ordering::Greater => {
                upper_idx = mid_idx - 1;
            }
        }
    }
    lower_idx - 1
}

enum SignalChangeData {
    FixedLength {
        encoding: SignalEncoding,
        width: u32, // bytes per entry
        bytes: Vec<u8>,
    },
    VariableLength(Vec<String>),
}

impl SignalChangeData {
    fn get_value_at(&self, offset: usize) -> SignalValue {
        match &self {
            SignalChangeData::FixedLength {
                encoding,
                width,
                bytes,
            } => {
                let start = offset * (*width as usize);
                let data = &bytes[start..(start + (*width as usize))];
                match encoding {
                    SignalEncoding::Binary(bits) => SignalValue::Binary(data, *bits),
                    SignalEncoding::FourValue(bits) => SignalValue::FourValue(data, *bits),
                    SignalEncoding::FixedLength => {
                        SignalValue::String(std::str::from_utf8(data).unwrap())
                    }
                    SignalEncoding::Float => {
                        SignalValue::Float(f64::from_le_bytes(<[u8; 8]>::try_from(data).unwrap()))
                    }
                    SignalEncoding::VariableLength => {
                        panic!("Variable length signals need to be variable length encoded!")
                    }
                }
            }
            SignalChangeData::VariableLength(strings) => SignalValue::String(&strings[offset]),
        }
    }
}

pub(crate) trait SignalSource {
    /// Loads new signals.
    /// Many implementations take advantage of loading multiple signals at a time.
    fn load_signals(&mut self, ids: &[(SignalRef, SignalLength)]) -> Vec<Signal>;
    /// Returns the global time table which stores the time at each value change.
    fn get_time_table(&self) -> Vec<Time>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::usize_div_ceil;

    #[test]
    fn test_sizes() {
        assert_eq!(std::mem::size_of::<SignalRef>(), 4);

        // 4 bytes for length + tag + padding
        assert_eq!(std::mem::size_of::<SignalEncoding>(), 8);

        assert_eq!(std::mem::size_of::<SignalChangeData>(), 40);
        assert_eq!(std::mem::size_of::<Signal>(), 72);

        // since there is some empty space in the Signal struct, we can make it an option for free!
        assert_eq!(std::mem::size_of::<Option<Signal>>(), 72);

        // signal values contain a slice (ptr + len) as well as a tag and potentially a length
        assert_eq!(std::mem::size_of::<&[u8]>(), 16);
        assert_eq!(std::mem::size_of::<SignalValue>(), 16 + 8);
    }

    #[test]
    fn test_to_bit_string_binary() {
        let data0 = [0b11100101u8, 0b00110010];
        let full_str = "1110010100110010";
        let full_str_len = full_str.len();

        for bits in 0..(full_str_len + 1) {
            let expected: String = full_str.chars().skip(full_str_len - bits).collect();
            let number_of_bytes = usize_div_ceil(bits, 8);
            let drop_bytes = data0.len() - number_of_bytes;
            let data = &data0[drop_bytes..];
            assert_eq!(
                SignalValue::Binary(data, bits as u32)
                    .to_bit_string()
                    .unwrap(),
                expected,
                "bits={}",
                bits
            );
        }
    }

    #[test]
    fn test_to_bit_string_four_state() {
        let data0 = [0b11100101u8, 0b00110010];
        let full_str = "zx110z0x";
        let full_str_len = full_str.len();

        for bits in 0..(full_str_len + 1) {
            let expected: String = full_str.chars().skip(full_str_len - bits).collect();
            let number_of_bytes = usize_div_ceil(bits, 4);
            let drop_bytes = data0.len() - number_of_bytes;
            let data = &data0[drop_bytes..];
            assert_eq!(
                SignalValue::FourValue(data, bits as u32)
                    .to_bit_string()
                    .unwrap(),
                expected,
                "bits={}",
                bits
            );
        }
    }
}
