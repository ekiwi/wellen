// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use super::{Bit, FixedWidthEncoding};
use crate::fst::push_zeros;
use crate::fst::{get_bytes_per_entry, get_len_and_meta};
use crate::signal::value::BitVecValue;
use crate::wavemem::check_if_changed_and_truncate;
use crate::{BitVecRef, Signal, SignalEncoding, SignalRef, States, TimeTableIdx};
use std::num::NonZeroU32;

pub fn transform_signal(transform: &DerivedBitVecSignal, inputs: &[&Signal]) -> Signal {
    match transform.output_encoding() {
        SignalEncoding::BitVector(width) => transform_bv_signal(transform, width.get(), inputs),
        other => todo!("Add support for generating a {:?} signal.", other),
    }
}

fn transform_bv_signal(
    transform: &DerivedBitVecSignal,
    out_width: u32,
    inputs: &[&Signal],
) -> Signal {
    let max_states = inputs
        .iter()
        .map(|i| {
            i.max_states()
                .expect("inputs to a bit-vec transform mut be bit-vec")
        })
        .reduce(States::join)
        .unwrap();
    let mut out = BitVectorBuilder::new(max_states, out_width);
    // iteration - todo: split into its own function
    let mut time = inputs
        .iter()
        .map(|i| i.time_indices.first().cloned().unwrap_or(0))
        .min()
        .unwrap_or(0);
    let mut offsets: Vec<Option<u32>> = inputs
        .iter()
        .map(|i| {
            for (ii, i_time) in i.time_indices.iter().enumerate() {
                if *i_time > time {
                    return if ii > 0 {
                        Some(i.time_indices[ii - 1])
                    } else {
                        None
                    };
                }
            }
            i.time_indices
                .first()
                .and_then(|first| if *first == time { Some(0) } else { None })
        })
        .collect();
    let mut values: Vec<Option<BitVecRef>> = inputs
        .iter()
        .zip(offsets.iter())
        .map(|(i, offset)| {
            offset.map(|o| {
                let value = i.data.get_value_at(o as usize);

                value.as_bit_vec().unwrap()
            })
        })
        .collect();

    while !offsets.iter().all(|o| o.is_none()) {
        out.add_change(time, (&transform.on_change(&values)).into());

        // more iteration logic
        let maybe_next_time = inputs
            .iter()
            .zip(offsets.iter())
            // find the next time index
            .flat_map(|(i, offset)| offset.and_then(|o| i.time_indices.get(o as usize + 1)))
            // pick the smallest
            .min();
        if let Some(next_time) = maybe_next_time {
            for ((i, offset), value) in inputs.iter().zip(offsets.iter_mut()).zip(values.iter_mut())
            {
                // we only advance the signals which have a matching next time
                if let Some(o) = *offset {
                    let next_o = o + 1;
                    if i.time_indices.get(next_o as usize) == Some(next_time) {
                        *offset = Some(next_o);
                        let any_value = i.data.get_value_at(next_o as usize);
                        *value = Some(any_value.as_bit_vec().unwrap());
                    }
                }
            }
            time = *next_time;
        } else {
            break; // done
        }
    }

    out.finish(SignalRef::derived_max())
}

/// Captures a signal which is derived from other bit-vector signals by slice and concat operations.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct DerivedBitVecSignal {
    width: u32,
    inputs: Vec<SignalRef>,
    bits: Vec<u64>,
}

impl DerivedBitVecSignal {
    /// Creates a new signal by slicing an existing one.
    pub fn new_slice(signal: SignalRef, enc: SignalEncoding, msb: u32, lsb: u32) -> Self {
        let mut out = Self {
            width: 0,
            inputs: vec![],
            bits: vec![],
        };
        out.concat_left(signal, enc, msb, lsb);
        out
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    /// Creates a new signal that is just the identity (full slice) of an existing one.
    pub fn new_identity(signal: SignalRef, enc: SignalEncoding) -> Self {
        if let SignalEncoding::BitVector(width) = enc {
            Self::new_slice(signal, enc, width.get() - 1, 0)
        } else {
            unreachable!("This function only works for bit vector signals")
        }
    }

    /// Concatenates a full signal to the left, i.e., on the most significant side.
    pub fn concat_left_full(&mut self, signal: SignalRef, enc: SignalEncoding) {
        if let SignalEncoding::BitVector(width) = enc {
            self.concat_left(signal, enc, width.get() - 1, 0)
        } else {
            unreachable!("This function only works for bit vector signals")
        }
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

    /// Concatenates a full signal to the right, i.e., on the least significant side.
    pub fn concat_right_full(&mut self, signal: SignalRef, enc: SignalEncoding) {
        if let SignalEncoding::BitVector(width) = enc {
            self.concat_right(signal, enc, width.get() - 1, 0)
        } else {
            unreachable!("This function only works for bit vector signals")
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

/// TODO: turn this into more of a trait!
impl DerivedBitVecSignal {
    pub fn output_encoding(&self) -> SignalEncoding {
        SignalEncoding::BitVector(NonZeroU32::new(self.width).unwrap())
    }

    pub fn inputs(&self) -> &[SignalRef] {
        &self.inputs
    }

    pub fn on_change(&self, values: &[Option<BitVecRef<'_>>]) -> BitVecValue {
        let max_states = values
            .iter()
            .flat_map(|v| v.map(|v| v.states()))
            .reduce(States::join)
            .unwrap();
        let mut out = BitVecValue::zero(max_states, self.width);

        let mut bit = self.width;
        for extract in self.bits.iter().map(|b| Extract::from(*b)) {
            if let Some(value) = values[extract.signal as usize] {
                for other_bit in (extract.lsb..(extract.msb + 1)).rev() {
                    bit -= 1;
                    let other_value = value.get_bit(other_bit);
                    out.set_bit(bit, other_value);
                }
            } else {
                // set bits to X if not value is available
                for _ in 0..extract.width() {
                    bit -= 1;
                    let other_value = Bit::X;
                    out.set_bit(bit, other_value);
                }
            }
        }
        debug_assert_eq!(bit, 0);

        out
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
    width: u32,
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
            width: bits,
            len,
            has_meta,
            bytes_per_entry,
            data,
            time_indices,
        }
    }

    fn add_change(&mut self, time_idx: TimeTableIdx, value: BitVecRef) {
        debug_assert_eq!(value.width(), self.width);
        let local_encoding = value.states();
        debug_assert!(local_encoding.bits() >= self.max_states.bits());
        if self.width == 1 {
            let value = u8::from(value.get_bit(0));
            let meta_data = (local_encoding as u8) << 6;
            self.data.push(value | meta_data);
        } else {
            let (local_len, local_has_meta) = get_len_and_meta(local_encoding, self.width);

            // append data
            let meta_data = (local_encoding as u8) << 6;
            if local_len == self.len && local_has_meta == self.has_meta {
                // same meta-data location and length as the maximum
                if self.has_meta {
                    self.data.push(meta_data);
                    value.append_to_vec(&mut self.data);
                } else {
                    let meta_data_index = self.data.len();
                    value.append_to_vec(&mut self.data);
                    self.data[meta_data_index] |= meta_data;
                }
            } else {
                // smaller encoding than the maximum
                self.data.push(meta_data);
                if self.has_meta {
                    push_zeros(&mut self.data, self.len - local_len);
                } else {
                    push_zeros(&mut self.data, self.len - local_len - 1);
                }
                value.append_to_vec(&mut self.data);
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
            width: self.width,
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

#[cfg(test)]
mod tests {
    use super::*;
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
