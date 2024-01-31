// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::{Hierarchy, SignalRef, SignalType};
use crate::wavemem::States;
use num_enum::TryFromPrimitive;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::num::NonZeroU32;

pub type Real = f64;
pub type Time = u64;
pub type TimeTableIdx = u32;
pub type NonZeroTimeTableIdx = NonZeroU32;

#[derive(Debug, Clone, Copy)]
pub enum SignalValue<'a> {
    Binary(&'a [u8], u32),
    FourValue(&'a [u8], u32),
    NineValue(&'a [u8], u32),
    String(&'a str),
    Real(Real),
}

impl<'a> Display for SignalValue<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            SignalValue::Binary(data, bits) => {
                write!(f, "{}", n_state_to_bit_string(States::Two, data, *bits))
            }
            SignalValue::FourValue(data, bits) => {
                write!(f, "{}", n_state_to_bit_string(States::Four, data, *bits))
            }
            SignalValue::NineValue(data, bits) => {
                write!(f, "{}", n_state_to_bit_string(States::Nine, data, *bits))
            }
            SignalValue::String(value) => write!(f, "{}", value),
            SignalValue::Real(value) => write!(f, "{}", value),
        }
    }
}

impl<'a> SignalValue<'a> {
    pub fn to_bit_string(&self) -> Option<String> {
        match &self {
            SignalValue::Binary(data, bits) => {
                Some(n_state_to_bit_string(States::Two, data, *bits))
            }
            SignalValue::FourValue(data, bits) => {
                Some(n_state_to_bit_string(States::Four, data, *bits))
            }
            SignalValue::NineValue(data, bits) => {
                Some(n_state_to_bit_string(States::Nine, data, *bits))
            }
            other => panic!("Cannot convert {other:?} to bit string"),
        }
    }
}

const TWO_STATE_LOOKUP: [char; 2] = ['0', '1'];
const FOUR_STATE_LOOKUP: [char; 4] = ['0', '1', 'x', 'z'];
const NINE_STATE_LOOKUP: [char; 9] = ['0', '1', 'x', 'z', 'h', 'u', 'w', 'l', '-'];

#[inline]
fn n_state_to_bit_string(states: States, data: &[u8], bits: u32) -> String {
    let lookup = match states {
        States::Two => TWO_STATE_LOOKUP.as_slice(),
        States::Four => FOUR_STATE_LOOKUP.as_slice(),
        States::Nine => NINE_STATE_LOOKUP.as_slice(),
    };
    let bits_per_byte = states.bits_in_a_byte() as u32;
    let states_bits = states.bits() as u32;
    let mask = (1u8 << states_bits) - 1;

    let mut out = String::with_capacity(bits as usize);
    if bits == 0 {
        return out;
    }

    // the first byte might not contain a full N bits
    let byte0_bits = bits - ((bits / bits_per_byte) * bits_per_byte);
    let byte0_is_special = byte0_bits > 0;
    if byte0_is_special {
        let byte0 = data[0];
        for ii in (0..byte0_bits).rev() {
            let value = (byte0 >> (ii * states_bits)) & mask;
            let char = lookup[value as usize];
            out.push(char);
        }
    }

    for byte in data.iter().skip(if byte0_is_special { 1 } else { 0 }) {
        for ii in (0..bits_per_byte).rev() {
            let value = (byte >> (ii * states_bits)) & mask;
            let char = lookup[value as usize];
            out.push(char);
        }
    }
    out
}

/// Specifies the encoding of a signal.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SignalEncoding {
    /// Bitvector of length N (u32) with 2, 4 or 9 states.
    /// If `meta_byte` is `true`, each sequence of data bytes is preceded by a meta-byte indicating whether the states
    /// are reduced by 1 (Four -> Two, Nine -> Four) or by 2 (Nine -> Two).
    BitVector {
        max_states: States,
        bits: u32,
        meta_byte: bool,
    },
    /// Each value is encoded as an 8-byte f64 in little endian.
    Real,
}

pub struct Signal {
    #[allow(unused)]
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
        debug_assert_eq!(time_indices.len(), bytes.len() / width as usize);
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

    pub fn size_in_memory(&self) -> usize {
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

    pub fn get_offset(&self, time_table_idx: TimeTableIdx) -> DataOffset {
        find_offset_from_time_table_idx(&self.time_indices, time_table_idx)
    }

    pub fn get_time_idx_at(&self, offset: &DataOffset) -> TimeTableIdx {
        self.time_indices[offset.start]
    }

    pub fn get_value_at(&self, offset: &DataOffset, element: u16) -> SignalValue {
        assert!(element < offset.elements);
        self.data.get_value_at(offset.start + element as usize)
    }
}

/// Provides file format independent access to a waveform file.
pub struct Waveform {
    hierarchy: Hierarchy,
    source: Box<dyn SignalSource + Send + Sync>,
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
    pub(crate) fn new(hierarchy: Hierarchy, source: Box<dyn SignalSource + Send + Sync>) -> Self {
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

    fn load_signals_internal(&mut self, ids: &[SignalRef], multi_threaded: bool) {
        let ids_with_len = ids
            .iter()
            .map(|i| (*i, self.hierarchy.get_signal_tpe(*i).unwrap()))
            .collect::<Vec<_>>();
        let signals = self.source.load_signals(&ids_with_len, multi_threaded);
        // the signal source must always return the correct number of signals!
        assert_eq!(signals.len(), ids.len());
        for (id, signal) in ids.iter().zip(signals.into_iter()) {
            self.signals.insert(*id, signal);
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

/// Finds the index that is the same or less than the needle and returns the position of it.
/// Note that `indices` needs to sorted from smallest to largest.
/// Essentially implements a binary search!
fn find_offset_from_time_table_idx(indices: &[TimeTableIdx], needle: TimeTableIdx) -> DataOffset {
    // find the index of a matching time
    let res = binary_search(indices, needle);
    let res_index = indices[res];

    // find start
    let mut start = res;
    while start > 0 && indices[start - 1] == res_index {
        start -= 1;
    }
    // find number of elements
    let mut elements = 1;
    while start + elements < indices.len() && indices[start + elements] == res_index {
        elements += 1;
    }

    // find next index
    let next_index = if start + elements < indices.len() {
        NonZeroTimeTableIdx::new(indices[start + elements])
    } else {
        None
    };

    DataOffset {
        start,
        elements: elements as u16,
        time_match: res_index == needle,
        next_index,
    }
}

#[inline]
fn binary_search(indices: &[TimeTableIdx], needle: TimeTableIdx) -> usize {
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

pub struct DataOffset {
    /// Offset of the first data value at the time requested (or earlier).
    pub start: usize,
    /// Number of elements that have the same time index. This is usually 1. Greater when there are delta cycles.
    pub elements: u16,
    /// Indicates that the offset exactly matches the time requested. If false, then we are matching an earlier time step.
    pub time_match: bool,
    /// Indicates the time table index of the next change.
    pub next_index: Option<NonZeroTimeTableIdx>,
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
                let raw_data = &bytes[start..(start + (*width as usize))];
                match encoding {
                    SignalEncoding::BitVector {
                        max_states,
                        bits,
                        meta_byte,
                    } => {
                        let data = if *meta_byte { &raw_data[1..] } else { raw_data };
                        match max_states {
                            States::Two => {
                                debug_assert!(!meta_byte);
                                // if the max state is 2, then all entries must be binary
                                SignalValue::Binary(data, *bits)
                            }
                            States::Four | States::Nine => {
                                // otherwise the actual number of states is encoded in the meta data
                                let meta_value = (raw_data[0] >> 6) & 0x3;
                                let states = States::try_from_primitive(meta_value).unwrap();
                                let ratio = states.bits_in_a_byte() / max_states.bits_in_a_byte();
                                let signal_bytes = match ratio {
                                    1 => data,
                                    2 => &data[(data.len() / 2)..],
                                    4 => &data[(data.len() / 4 * 3)..],
                                    other => unreachable!("Ratio of: {other}"),
                                };
                                match states {
                                    States::Two => SignalValue::Binary(signal_bytes, *bits),
                                    States::Four => SignalValue::FourValue(signal_bytes, *bits),
                                    States::Nine => SignalValue::NineValue(signal_bytes, *bits),
                                }
                            }
                        }
                    }
                    SignalEncoding::Real => SignalValue::Real(Real::from_le_bytes(
                        <[u8; 8]>::try_from(raw_data).unwrap(),
                    )),
                }
            }
            SignalChangeData::VariableLength(strings) => SignalValue::String(&strings[offset]),
        }
    }
}

pub(crate) trait SignalSource {
    /// Loads new signals.
    /// Many implementations take advantage of loading multiple signals at a time.
    fn load_signals(
        &mut self,
        ids: &[(SignalRef, SignalType)],
        multi_threaded: bool,
    ) -> Vec<Signal>;
    /// Returns the global time table which stores the time at each value change.
    fn get_time_table(&self) -> Vec<Time>;
    /// Print memory size / speed statistics.
    fn print_statistics(&self);
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

    #[test]
    fn test_long_2_state_to_string() {
        let data = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b110001, 0b11, 0b10110011,
        ];
        let out = SignalValue::Binary(data.as_slice(), 153)
            .to_bit_string()
            .unwrap();
        let expected = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100010000001110110011";
        assert_eq!(out, expected);
    }
}
