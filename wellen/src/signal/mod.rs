// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::{SignalEncoding, SignalRef};
use num_enum::TryFromPrimitive;
use std::fmt::{Debug, Formatter};
use std::num::NonZeroU32;

mod compressed;
pub use compressed::{CompressedSignal, CompressedTimeTable, Compression};
mod value;
pub use value::{Bit, BitVecRef, Real, SignalValue, SignalValueRef, States, bit_char_to_num};
pub(crate) use value::BitVecValue;
mod map;
mod source;
mod transform;

pub use map::SignalMap;
pub use source::{SignalSource, SignalSourceImplementation};
pub use transform::DerivedBitVecSignal;

pub type Time = u64;
pub type TimeTableIdx = u32;
pub type NonZeroTimeTableIdx = NonZeroU32;

/// Specifies the encoding of a signal.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub enum FixedWidthEncoding {
    /// Bitvector of length N (u32) with 2, 4 or 9 states.
    /// If `meta_byte` is `true`, each sequence of data bytes is preceded by a meta-byte indicating whether the states
    /// are reduced by 1 (Four -> Two, Nine -> Four) or by 2 (Nine -> Two).
    BitVector {
        max_states: States,
        width: u32,
        meta_byte: bool,
    },
    /// Each value is encoded as an 8-byte f64 in little endian.
    Real,
    /// No values, just the timetable to indicate when the event occurred.
    Event,
}

impl FixedWidthEncoding {
    pub fn signal_encoding(&self) -> SignalEncoding {
        match self {
            FixedWidthEncoding::BitVector { width: bits, .. } => SignalEncoding::BitVector(*bits),
            FixedWidthEncoding::Real => SignalEncoding::Real,
            FixedWidthEncoding::Event => SignalEncoding::BitVector(0),
        }
    }
}

#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct Signal {
    idx: SignalRef,
    time_indices: Vec<TimeTableIdx>,
    data: SignalChangeData,
}

impl Debug for Signal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Signal({:?}, {} changes, {:?})",
            self.idx,
            self.time_indices.len(),
            self.data
        )
    }
}

impl PartialEq for Signal {
    fn eq(&self, other: &Self) -> bool {
        if self.idx == other.idx {
            let mut our_iter = self.iter_changes();
            let mut other_iter = other.iter_changes();
            loop {
                if let Some((our_time, our_value)) = our_iter.next() {
                    if let Some((other_time, other_value)) = other_iter.next() {
                        if our_time != other_time || our_value != other_value {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return other_iter.next().is_none();
                }
            }
        } else {
            false
        }
    }
}

impl Eq for Signal {}

impl Signal {
    pub fn new_fixed_len(
        idx: SignalRef,
        time_indices: Vec<TimeTableIdx>,
        encoding: FixedWidthEncoding,
        width: u32,
        bytes: Vec<u8>,
    ) -> Self {
        if width > 0 {
            debug_assert_eq!(time_indices.len(), bytes.len() / width as usize);
        }
        let data = ChangeData::FixedLength {
            encoding,
            bytes_per_entry: width,
            bytes,
        };
        Signal {
            idx,
            time_indices,
            data: SignalChangeData(data),
        }
    }

    pub fn new_var_len(
        idx: SignalRef,
        time_indices: Vec<TimeTableIdx>,
        strings: Vec<String>,
    ) -> Self {
        assert_eq!(time_indices.len(), strings.len());
        let data = SignalChangeData(ChangeData::VariableLength(strings));
        Signal {
            idx,
            time_indices,
            data,
        }
    }

    pub fn size_in_memory(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let time = self.time_indices.len() * std::mem::size_of::<TimeTableIdx>();
        let data = match &self.data.0 {
            ChangeData::FixedLength { bytes, .. } => bytes.len(),
            ChangeData::VariableLength(strings) => strings
                .iter()
                .map(|s| s.len() + std::mem::size_of::<String>())
                .sum::<usize>(),
        };
        base + time + data
    }

    /// Returns the data offset for the nearest change at or before the provided idx.
    /// Returns `None` of not data is available for this signal at or before the idx.
    pub fn get_offset(&self, time_table_idx: TimeTableIdx) -> Option<DataOffset> {
        match self.time_indices.first() {
            None => None,
            Some(first) if *first > time_table_idx => None,
            _ => Some(find_offset_from_time_table_idx(
                &self.time_indices,
                time_table_idx,
            )),
        }
    }

    pub fn get_time_idx_at(&self, offset: &DataOffset) -> TimeTableIdx {
        self.time_indices[offset.start]
    }

    pub fn get_value_at(&self, offset: &DataOffset, element: u16) -> SignalValueRef<'_> {
        assert!(element < offset.elements);
        self.data.get_value_at(offset.start + element as usize)
    }

    pub fn get_first_time_idx(&self) -> Option<TimeTableIdx> {
        self.time_indices.first().cloned()
    }

    pub fn time_indices(&self) -> &[TimeTableIdx] {
        &self.time_indices
    }

    pub fn iter_changes(&self) -> impl Iterator<Item = (TimeTableIdx, SignalValueRef<'_>)> {
        SignalChangeIterator::new(self)
    }

    pub fn signal_ref(&self) -> SignalRef {
        self.idx
    }

    pub(crate) fn signal_encoding(&self) -> SignalEncoding {
        self.data.signal_encoding()
    }

    pub fn max_states(&self) -> Option<States> {
        if let ChangeData::FixedLength { encoding, .. } = self.data.0
            && let FixedWidthEncoding::BitVector { max_states, .. } = encoding
        {
            Some(max_states)
        } else {
            None
        }
    }

    pub fn data(&self) -> &SignalChangeData {
        &self.data
    }
}

pub struct SignalChangeIterator<'a> {
    signal: &'a Signal,
    offset: usize,
}

impl<'a> SignalChangeIterator<'a> {
    fn new(signal: &'a Signal) -> Self {
        Self { signal, offset: 0 }
    }
}

impl<'a> Iterator for SignalChangeIterator<'a> {
    type Item = (TimeTableIdx, SignalValueRef<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(time_idx) = self.signal.time_indices.get(self.offset) {
            let data = self.signal.data.get_value_at(self.offset);
            self.offset += 1;
            Some((*time_idx, data))
        } else {
            None
        }
    }
}

/// Finds the index that is the same or less than the needle and returns the position of it.
/// Note that `indices` needs to be sorted from smallest to largest.
/// Essentially implements a binary search!
fn find_offset_from_time_table_idx(indices: &[TimeTableIdx], needle: TimeTableIdx) -> DataOffset {
    debug_assert!(!indices.is_empty(), "empty time table");

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
    debug_assert!(!indices.is_empty(), "empty time table!");
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

#[derive(Debug, Eq, PartialEq, Clone)]
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

#[derive(Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct SignalChangeData(ChangeData);

#[derive(Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
enum ChangeData {
    FixedLength {
        encoding: FixedWidthEncoding,
        bytes_per_entry: u32,
        bytes: Vec<u8>,
    },
    VariableLength(Vec<String>),
}

impl SignalChangeData {
    fn signal_encoding(&self) -> SignalEncoding {
        match self.0 {
            ChangeData::FixedLength { encoding, .. } => encoding.signal_encoding(),
            ChangeData::VariableLength(_) => SignalEncoding::String,
        }
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        match &self.0 {
            ChangeData::FixedLength { bytes, .. } => bytes.is_empty(),
            ChangeData::VariableLength(data) => data.is_empty(),
        }
    }
}

impl Debug for SignalChangeData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            ChangeData::FixedLength {
                encoding, bytes, ..
            } => {
                write!(
                    f,
                    "SignalChangeData({encoding:?}, {} data bytes)",
                    bytes.len()
                )
            }
            ChangeData::VariableLength(values) => {
                write!(f, "SignalChangeData({} strings)", values.len())
            }
        }
    }
}

impl SignalChangeData {
    pub fn get_value_at(&self, offset: usize) -> SignalValueRef<'_> {
        match &self.0 {
            ChangeData::FixedLength {
                encoding,
                bytes_per_entry,
                bytes,
            } => {
                let start = offset * (*bytes_per_entry as usize);
                let raw_data = &bytes[start..(start + (*bytes_per_entry as usize))];
                match encoding {
                    FixedWidthEncoding::Event => SignalValueRef::Event,
                    FixedWidthEncoding::BitVector {
                        max_states,
                        width,
                        meta_byte,
                    } => {
                        let data = if *meta_byte { &raw_data[1..] } else { raw_data };
                        match max_states {
                            States::Two => {
                                debug_assert!(!meta_byte);
                                // if the max state is 2, then all entries must be binary
                                SignalValueRef::bit_vec(*max_states, *width, data)
                            }
                            States::Four | States::Nine => {
                                // otherwise the actual number of states is encoded in the meta data
                                let meta_value = (raw_data[0] >> 6) & 0x3;
                                debug_assert!(
                                    States::try_from_primitive(meta_value).is_ok(),
                                    "ERROR: offset={offset}, encoding={encoding:?}, width={bytes_per_entry}, raw_data[0]={}",
                                    raw_data[0]
                                );
                                let states = States::try_from_primitive(meta_value).unwrap();
                                let num_out_bytes = states.bytes_required(*width);
                                debug_assert!(num_out_bytes <= data.len());
                                let signal_bytes = if num_out_bytes == data.len() {
                                    data
                                } else {
                                    debug_assert!(states != *max_states);
                                    &data[(data.len() - num_out_bytes)..]
                                };
                                SignalValueRef::bit_vec(states, *width, signal_bytes)
                            }
                        }
                    }
                    FixedWidthEncoding::Real => SignalValueRef::Real(Real::from_le_bytes(
                        <[u8; 8]>::try_from(raw_data).unwrap(),
                    )),
                }
            }
            ChangeData::VariableLength(strings) => SignalValueRef::String(&strings[offset]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        assert_eq!(std::mem::size_of::<SignalRef>(), 4);

        // 4 bytes for length + tag + padding
        assert_eq!(std::mem::size_of::<FixedWidthEncoding>(), 8);

        assert_eq!(std::mem::size_of::<ChangeData>(), 40);
        assert_eq!(std::mem::size_of::<Signal>(), 72);

        // since there is some empty space in the Signal struct, we can make it an option for free!
        assert_eq!(std::mem::size_of::<Option<Signal>>(), 72);
    }
}
