// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use super::States;
use crate::fst::{get_bytes_per_entry, get_len_and_meta, push_zeros};
use crate::signal::{FixedWidthEncoding, SignalChangeData};
use crate::wavemem::check_if_changed_and_truncate;
use crate::{Signal, SignalRef, SignalValueRef, TimeTableIdx};

pub struct BitVectorBuilder {
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
        debug_assert_eq!(value.bits().unwrap(), self.bits);
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

pub fn slice_signal(id: SignalRef, signal: &Signal, msb: u32, lsb: u32) -> Signal {
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
                slice_n_states(
                    States::Two,
                    data,
                    &mut buf,
                    msb as usize,
                    lsb as usize,
                    in_bits as usize,
                );
                SignalValueRef::Binary(&buf, result_bits)
            }
            SignalValueRef::FourValue(data, in_bits) => {
                slice_n_states(
                    States::Four,
                    data,
                    &mut buf,
                    msb as usize,
                    lsb as usize,
                    in_bits as usize,
                );
                SignalValueRef::FourValue(&buf, result_bits)
            }
            SignalValueRef::NineValue(data, in_bits) => {
                slice_n_states(
                    States::Nine,
                    data,
                    &mut buf,
                    msb as usize,
                    lsb as usize,
                    in_bits as usize,
                );
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
    msb: usize,
    lsb: usize,
    in_bits: usize,
) {
    let out_bits = msb - lsb + 1;
    debug_assert!(in_bits > out_bits);
    let mut working_byte = 0u8;
    for (out_bit, in_bit) in (lsb..(msb + 1)).enumerate().rev() {
        let rev_in_bit = in_bits - in_bit - 1;
        let in_byte = data[rev_in_bit / states.bits_in_a_byte()];
        let in_value =
            (in_byte >> ((in_bit % states.bits_in_a_byte()) * states.bits())) & states.mask();

        working_byte = (working_byte << states.bits()) + in_value;
        if out_bit % states.bits_in_a_byte() == 0 {
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
}
