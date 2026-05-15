// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use super::States;
use crate::fst::{get_bytes_per_entry, get_len_and_meta, push_zeros};
use crate::signal::{FixedWidthEncoding, SignalChangeData};
use crate::wavemem::check_if_changed_and_truncate;
use crate::{Signal, SignalEncoding, SignalRef, SignalValueRef, Time, TimeTableIdx};

/// Generates a new signal based on other signals.
/// The on_change method takes a mutable reference and thus the transform must be
/// executed sequentially over all changes in order.
pub trait SignalTransform {
    type SignalRefType;
    /// The encoding of the output signal.
    fn output_encoding(&self) -> SignalEncoding;
    /// The signals that this transform depends on.
    fn inputs(&self) -> &[SignalRef];
    /// Process a change in at least one of the input signals.
    fn on_change(&mut self, time: Time, values: &[Self::SignalRefType]) -> Self::SignalRefType;
}

/// Captures a signal which is derived from other bit-vector signals by slice and concat operations.
pub struct DerivedBitVecSignal {
    width: u32,
    inputs: Vec<SignalRef>,
    bits: Vec<u64>,
}

impl DerivedBitVecSignal {
    /// Creates a new signal by concatenating an existing one.
    pub fn new(signal: SignalRef, enc: SignalEncoding, msb: u32, lsb: u32) -> Self {
        let mut out = Self {
            width: 0,
            inputs: vec![],
            bits: vec![],
        };
        out.concat_left(signal, enc, msb, lsb);
        out
    }

    /// Concatenates a signal to the left, i.e., on the most significant side.
    pub fn concat_left(&mut self, signal: SignalRef, enc: SignalEncoding, msb: u32, lsb: u32) {
        let mut extract = self.make_extract(signal, enc, msb, lsb);
        self.width += extract.width();
        if let Some(other_num) = self.bits.first_mut() {
            let other: Extract = (*other_num).into();
            // can the two extracts be merged?
            if other.signal == extract.signal && extract.lsb == other.msb + 1 {
                extract.lsb = other.lsb;
                *other_num = extract.into();
            } else {
                self.bits.insert(0, extract.into());
            }
        } else {
            debug_assert!(self.bits.is_empty());
            self.bits.push(extract.into());
        }
    }

    /// Concatenates a signal to the right, i.e., on the least significant side.
    pub fn concat_right(&mut self, signal: SignalRef, enc: SignalEncoding, msb: u32, lsb: u32) {
        let mut extract = self.make_extract(signal, enc, msb, lsb);
        self.width += extract.width();
        if let Some(other_num) = self.bits.last_mut() {
            let other: Extract = (*other_num).into();
            // can the two extracts be merged?
            if other.signal == extract.signal && extract.msb + 1 == other.lsb {
                extract.msb = other.msb;
                *other_num = extract.into();
            } else {
                self.bits.push(extract.into());
            }
        } else {
            debug_assert!(self.bits.is_empty());
            self.bits.push(extract.into());
        }
    }

    /// Converts a signal description into an [[Extract]] op.
    fn make_extract(
        &mut self,
        signal: SignalRef,
        enc: SignalEncoding,
        msb: u32,
        lsb: u32,
    ) -> Extract {
        debug_assert!(msb >= lsb);
        if let SignalEncoding::BitVector(len) = enc {
            debug_assert!(msb < len.get());
            // check to see if we already have this signal as an input
            let signal = if let Some(ii) = self.inputs.iter().position(|i| *i == signal) {
                ii as u16
            } else {
                let ii = self.inputs.len() as u16;
                self.inputs.push(signal);
                ii
            };
            Extract { signal, msb, lsb }
        } else {
            unreachable!("Can only derive from a bit-vector signal!");
        }
    }
}

impl SignalTransform for DerivedBitVecSignal {
    type SignalRefType = ();

    fn output_encoding(&self) -> SignalEncoding {
        todo!()
    }

    fn inputs(&self) -> &[SignalRef] {
        &self.inputs
    }

    fn on_change(&mut self, time: Time, values: &[Self::SignalRefType]) -> Self::SignalRefType {
        todo!()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Extract {
    signal: u16,
    msb: u32,
    lsb: u32,
}

impl Extract {
    fn width(&self) -> u32 {
        self.msb - self.lsb + 1
    }
}

impl From<u64> for Extract {
    fn from(value: u64) -> Self {
        let signal = (value >> (64 - 16)) as u16;
        let lsb = (value & (u32::MAX as u64)) as u32;
        let width = ((value >> 32) & (u16::MAX as u64)) as u16;
        let msb = (width as u32) - 1 + lsb;
        Self { signal, msb, lsb }
    }
}

impl From<Extract> for u64 {
    fn from(value: Extract) -> Self {
        let width = (value.msb - value.lsb + 1) as u16;
        ((value.signal as u64) << (64 - 16)) | ((width as u64) << 32) | (value.lsb as u64)
    }
}

struct BitVectorBuilder {
    max_states: States,
    bits: u32,
    len: usize,
    has_meta: bool,
    bytes_per_entry: usize,
    data: Vec<u8>,
    time_indices: Vec<TimeTableIdx>,
}

impl BitVectorBuilder {
    fn new(max_states: States, bits: u32) -> Self {
        assert!(bits > 0);
        let (len, has_meta) = get_len_and_meta(max_states, bits);
        let bytes_per_entry = get_bytes_per_entry(len, has_meta);
        let data = vec![];
        let time_indices = vec![];
        Self {
            max_states,
            bits,
            len,
            has_meta,
            bytes_per_entry,
            data,
            time_indices,
        }
    }

    fn add_change(&mut self, time_idx: TimeTableIdx, value: SignalValueRef) {
        debug_assert_eq!(value.width().unwrap(), self.bits);
        let local_encoding = value.states().unwrap();
        debug_assert!(local_encoding.bits() >= self.max_states.bits());
        if self.bits == 1 {
            let (value, mask) = value.data_and_mask().unwrap();
            let value = value[0] & mask;
            let meta_data = (local_encoding as u8) << 6;
            self.data.push(value | meta_data);
        } else {
            let (data, mask) = value.data_and_mask().unwrap();
            let (local_len, local_has_meta) = get_len_and_meta(local_encoding, self.bits);
            assert_eq!(data.len(), local_len);

            // append data
            let meta_data = (local_encoding as u8) << 6;
            if local_len == self.len && local_has_meta == self.has_meta {
                // same meta-data location and length as the maximum
                if self.has_meta {
                    self.data.push(meta_data);
                    self.data.push(data[0] & mask);
                } else {
                    self.data.push(meta_data | (data[0] & mask));
                }
                self.data.extend_from_slice(&data[1..]);
            } else {
                // smaller encoding than the maximum
                self.data.push(meta_data);
                if self.has_meta {
                    push_zeros(&mut self.data, self.len - local_len);
                } else {
                    push_zeros(&mut self.data, self.len - local_len - 1);
                }
                self.data.push(data[0] & mask);
                self.data.extend_from_slice(&data[1..]);
            }
        }
        // see if there actually was a change and revert if there was not
        if check_if_changed_and_truncate(self.bytes_per_entry, &mut self.data) {
            self.time_indices.push(time_idx);
        }
    }

    fn finish(self, id: SignalRef) -> Signal {
        debug_assert_eq!(
            self.data.len(),
            self.time_indices.len() * self.bytes_per_entry
        );
        let encoding = FixedWidthEncoding::BitVector {
            max_states: self.max_states,
            bits: self.bits,
            meta_byte: self.has_meta,
        };
        Signal::new_fixed_len(
            id,
            self.time_indices,
            encoding,
            self.bytes_per_entry as u32,
            self.data,
        )
    }
}

fn slice_signal(id: SignalRef, signal: &Signal, msb: u32, lsb: u32) -> Signal {
    debug_assert!(msb >= lsb);
    if let SignalChangeData::FixedLength {
        encoding: FixedWidthEncoding::BitVector { max_states, .. },
        ..
    } = &signal.data
    {
        slice_bit_vector(id, signal, msb, lsb, *max_states)
    } else {
        panic!("Cannot slice signal with data: {:?}", signal.data);
    }
}

fn slice_bit_vector(
    id: SignalRef,
    signal: &Signal,
    msb: u32,
    lsb: u32,
    max_states: States,
) -> Signal {
    debug_assert!(msb >= lsb);
    let result_bits = msb - lsb + 1;
    let mut builder = BitVectorBuilder::new(max_states, result_bits);
    let mut buf = Vec::with_capacity(result_bits.div_ceil(2) as usize);
    for (time_idx, value) in signal.iter_changes() {
        let out_value = match value {
            SignalValueRef::Binary(data, in_bits) => {
                slice_n_states(States::Two, data, &mut buf, msb, lsb, in_bits);
                SignalValueRef::Binary(&buf, result_bits)
            }
            SignalValueRef::FourValue(data, in_bits) => {
                slice_n_states(States::Four, data, &mut buf, msb, lsb, in_bits);
                SignalValueRef::FourValue(&buf, result_bits)
            }
            SignalValueRef::NineValue(data, in_bits) => {
                slice_n_states(States::Nine, data, &mut buf, msb, lsb, in_bits);
                SignalValueRef::NineValue(&buf, result_bits)
            }
            _ => unreachable!("expected a bit vector"),
        };
        builder.add_change(time_idx, out_value);
        buf.clear();
    }
    builder.finish(id)
}

#[inline]
fn slice_n_states(
    states: States,
    data: &[u8],
    out: &mut Vec<u8>,
    msb: u32,
    lsb: u32,
    in_bits: u32,
) {
    let out_bits = msb - lsb + 1;
    debug_assert!(in_bits > out_bits);
    let mut working_byte = 0u8;
    for (out_bit, in_bit) in (lsb..(msb + 1)).enumerate().rev() {
        let rev_in_bit = in_bits - in_bit - 1;
        let in_byte = data[(rev_in_bit / states.bits_in_a_byte()) as usize];
        let in_value =
            (in_byte >> ((in_bit % states.bits_in_a_byte()) * states.bits())) & states.mask();

        working_byte = (working_byte << states.bits()) + in_value;
        if out_bit as u32 % states.bits_in_a_byte() == 0 {
            out.push(working_byte);
            working_byte = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_signal() {
        let mut out = vec![];
        slice_n_states(States::Two, &[0b001001], &mut out, 3, 3, 7);
        assert_eq!(out[0], 1);
        out.clear();
    }

    use proptest::prelude::*;

    fn do_test_extract_conversion(signal: u16, width: u16, lsb: u32) {
        let msb = lsb + width as u32 - 1;
        assert_eq!(width as u32, msb - lsb + 1);
        let extract = Extract { signal, msb, lsb };
        let as_num: u64 = extract.into();
        let and_back: Extract = as_num.into();
        assert_eq!(extract, and_back);
    }

    proptest! {
        #[test]
        fn test_extract_conversion(signal: u16, width: u16, lsb: u32) {
            do_test_extract_conversion(signal, width, lsb);
        }
    }
}
