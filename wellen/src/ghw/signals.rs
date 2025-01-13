// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::ghw::common::*;
use crate::signals::SignalSource;
use crate::wavemem::{Encoder, States};
use crate::{Hierarchy, SignalRef, TimeTable};
use std::io::BufRead;

/// Reads the GHW signal values. `input` should be advanced until right after the end of hierarchy
pub fn read_signals(
    header: &HeaderData,
    decode_info: GhwDecodeInfo,
    hierarchy: &Hierarchy,
    input: &mut impl BufRead,
) -> Result<(SignalSource, TimeTable)> {
    let (info, vectors) = decode_info;
    // TODO: multi-threading
    let mut encoder = Encoder::new(hierarchy);
    let mut vecs = VecBuffer::from_vec_info(vectors);

    // loop over signal sections
    loop {
        let mut mark = [0u8; 4];
        input.read_exact(&mut mark)?;

        // read_sm_hdr
        match &mark {
            GHW_SNAPSHOT_SECTION => {
                read_snapshot_section(header, &info, &mut vecs, &mut encoder, input)?
            }
            GHW_CYCLE_SECTION => read_cycle_section(header, &info, &mut vecs, &mut encoder, input)?,
            GHW_DIRECTORY_SECTION => {
                // skip the directory by reading it
                let _ = read_directory(header, input)?;
            }
            GHW_TAILER_SECTION => {
                // the "tailer" means that we are done reading the file
                // we still read the tailer in order to make sure our progress indicator ends at
                // 100%
                let mut tailer_body = [0u8; GHW_TAILER_LEN - GHW_TAILER_SECTION.len()];
                input.read_exact(&mut tailer_body)?;
                break;
            }
            other => {
                return Err(GhwParseError::UnexpectedSection(
                    String::from_utf8_lossy(other).to_string(),
                ))
            }
        }
    }
    Ok(encoder.finish())
}

fn read_snapshot_section(
    header: &HeaderData,
    info: &GhwSignals,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut h = [0u8; 12];
    input.read_exact(&mut h)?;
    check_header_zeros("snapshot", &h)?;

    // time in femto seconds
    let start_time = header.read_i64(&mut &h[4..12])? as u64;
    enc.time_change(start_time);

    for sig_index in 0..(info.signal_len() as u32) {
        read_signal_value(info, GhwSignalId::new(sig_index + 1), vecs, enc, input)?;
    }
    finish_time_step(vecs, enc);

    // check for correct end magic
    check_magic_end(input, "snapshot", GHW_END_SNAPSHOT_SECTION)?;
    Ok(())
}

fn read_cycle_section(
    header: &HeaderData,
    info: &GhwSignals,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut h = [0u8; 8];
    input.read_exact(&mut h)?;
    // note: cycle sections do not have the four zero bytes!

    // time in femto seconds
    let mut start_time = header.read_i64(&mut &h[..])? as u64;

    loop {
        enc.time_change(start_time);
        read_cycle_signals(info, vecs, enc, input)?;
        finish_time_step(vecs, enc);

        let time_delta = leb128::read::signed(input)?;
        if time_delta < 0 {
            break; // end of cycle
        } else {
            start_time += time_delta as u64;
        }
    }

    // check cycle end
    check_magic_end(input, "cycle", GHW_END_CYCLE_SECTION)?;

    Ok(())
}

fn read_cycle_signals(
    info: &GhwSignals,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    let mut pos_signal_index = 0;
    loop {
        let delta = leb128::read::unsigned(input)? as usize;
        if delta == 0 {
            break;
        }
        pos_signal_index += delta;
        if pos_signal_index == 0 {
            return Err(GhwParseError::FailedToParseSection(
                "cycle",
                "Expected a first delta > 0".to_string(),
            ));
        }
        let sig_id = GhwSignalId::new(pos_signal_index as u32);
        read_signal_value(info, sig_id, vecs, enc, input)?;
    }
    Ok(())
}

/// This dispatches any remaining vector changes.
fn finish_time_step(vecs: &mut VecBuffer, enc: &mut Encoder) {
    vecs.process_changed_signals(|signal_ref, data, states| {
        enc.raw_value_change(signal_ref, data, states);
    })
}

fn read_signal_value(
    info: &GhwSignals,
    signal_id: GhwSignalId,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    let signal_info = info.get_info(signal_id);
    let (tpe, signal_ref) = (signal_info.tpe(), signal_info.signal_ref());
    match tpe {
        SignalType::NineState => {
            let ghdl_value = read_u8(input)?;
            let value = [STD_LOGIC_LUT[ghdl_value as usize]];
            enc.raw_value_change(signal_ref, &value, States::Nine);
        }
        SignalType::TwoState => {
            let value = [read_u8(input)?];
            debug_assert!(value[0] <= 1);
            enc.raw_value_change(signal_ref, &value, States::Two);
        }
        SignalType::NineStateVec | SignalType::TwoStateVec => {
            let ghdl_value = read_u8(input)?;
            let (value, states) = if tpe == SignalType::NineStateVec {
                (STD_LOGIC_LUT[ghdl_value as usize], States::Nine)
            } else {
                debug_assert!(ghdl_value <= 1);
                (ghdl_value, States::Two)
            };

            let vec_id = signal_info.vec_id().unwrap();

            // check to see if we already had a change to this same bit in the current time step
            if vecs.is_second_change(vec_id, signal_id, value) {
                // immediately dispatch the change to properly reflect the delta cycle
                let data = vecs.get_full_value_and_clear_changes(vec_id);
                enc.raw_value_change(signal_ref, data, states);
            }

            // update value
            vecs.update_value(vec_id, signal_id, value);

            // check to see if we need to report a change
            if vecs.full_signal_has_changed(vec_id) {
                let data = vecs.get_full_value_and_clear_changes(vec_id);
                enc.raw_value_change(signal_ref, data, states);
            }
        }
        SignalType::U8 => {
            let value = [read_u8(input)?];
            enc.raw_value_change(signal_ref, &value, States::Two);
        }
        SignalType::Leb128Signed => {
            let signed_value = leb128::read::signed(input)?;
            let value = signed_value as u64;
            let bytes = &value.to_be_bytes();
            enc.raw_value_change(signal_ref, bytes, States::Two);
        }
        SignalType::F64 => {
            // we need to figure out the endianes here
            let value = read_f64_le(input)?;
            enc.real_change(signal_ref, value);
        }
    }
    Ok(())
}

/// Keeps track of individual bits and combines them into a full bit vector.
#[derive(Debug)]
struct VecBuffer {
    info: Vec<VecBufferInfo>,
    data: Vec<u8>,
    bit_change: Vec<u8>,
    change_list: Vec<GhwVecId>,
    signal_change: Vec<u8>,
}

#[derive(Debug, Clone)]
struct VecBufferInfo {
    /// start as byte index
    data_start: u32,
    /// start as byte index
    bit_change_start: u32,
    bits: u32,
    states: States,
    signal_ref: SignalRef,
    max_index: u32,
}

impl VecBufferInfo {
    fn change_range(&self) -> std::ops::Range<usize> {
        // whether a bit has been changed is stored with 8 bits per byte
        let start = self.bit_change_start as usize;
        let len = self.bits.div_ceil(8) as usize;
        start..(start + len)
    }
    fn data_range(&self) -> std::ops::Range<usize> {
        // data is stored with N bits per byte depending on the states
        let start = self.data_start as usize;
        let len = (self.bits as usize).div_ceil(self.states.bits_in_a_byte());
        start..(start + len)
    }
}

impl VecBuffer {
    fn from_vec_info(vectors: Vec<GhwVecInfo>) -> Self {
        let mut data_start = 0;
        let mut bit_change_start = 0;

        let mut info = vectors
            .into_iter()
            .map(|vector| {
                let bits = vector.bits();
                let states = if vector.is_two_state() {
                    States::Two
                } else {
                    States::Nine
                };
                let info = VecBufferInfo {
                    data_start: data_start as u32,
                    bit_change_start: bit_change_start as u32,
                    bits,
                    states,
                    signal_ref: vector.signal_ref(),
                    max_index: vector.max().index() as u32,
                };
                data_start += (bits as usize).div_ceil(states.bits_in_a_byte());
                bit_change_start += (bits as usize).div_ceil(8);
                info
            })
            .collect::<Vec<_>>();
        info.shrink_to_fit();

        let data = vec![0; data_start];
        let bit_change = vec![0; bit_change_start];
        let change_list = vec![];
        let signal_change = vec![0; info.len().div_ceil(8)];

        Self {
            info,
            data,
            bit_change,
            change_list,
            signal_change,
        }
    }

    fn process_changed_signals(&mut self, mut callback: impl FnMut(SignalRef, &[u8], States)) {
        let change_list = std::mem::take(&mut self.change_list);
        for vec_id in change_list.into_iter() {
            if self.has_signal_changed(vec_id) {
                let states = self.info[vec_id.index()].states;
                let signal_ref = self.info[vec_id.index()].signal_ref;
                let data = self.get_full_value_and_clear_changes(vec_id);
                (callback)(signal_ref, data, states);
            }
        }
    }

    #[inline]
    fn is_second_change(&self, vector_id: GhwVecId, signal_id: GhwSignalId, value: u8) -> bool {
        let info = &self.info[vector_id.index()];
        let bit = info.max_index - signal_id.index() as u32;
        self.has_bit_changed(info, bit) && self.get_value(info, bit) != value
    }

    #[inline]
    fn update_value(&mut self, vector_id: GhwVecId, signal_id: GhwSignalId, value: u8) {
        let info = &self.info[vector_id.index()];
        let bit = info.max_index - signal_id.index() as u32;
        Self::mark_bit_changed(&mut self.bit_change, info, bit);
        Self::set_value(&mut self.data, info, bit, value);
        // add signal to change list if it has not already been added
        if !self.has_signal_changed(vector_id) {
            self.mark_signal_changed(vector_id);
        }
    }

    /// Used in order to dispatch full signal changes as soon as possible
    #[inline]
    fn full_signal_has_changed(&self, vector_id: GhwVecId) -> bool {
        let info = &self.info[vector_id.index()];

        // check changes
        let changes = &self.bit_change[info.change_range()];
        let skip = if info.bits % 8 == 0 { 0 } else { 1 };
        for e in changes.iter().skip(skip) {
            if *e != 0xff {
                return false;
            }
        }

        // check valid msb (in case where the number of bits is not a multiple of 8)
        if skip > 0 {
            let msb_mask = (1u8 << (info.bits % 8)) - 1;
            if changes[0] != msb_mask {
                return false;
            }
        }

        true
    }

    #[inline]
    fn get_full_value_and_clear_changes(&mut self, vector_id: GhwVecId) -> &[u8] {
        let info = &self.info[vector_id.index()];

        // clear bit changes
        let changes = &mut self.bit_change[info.change_range()];
        for e in changes.iter_mut() {
            *e = 0;
        }

        // clear signal change
        let byte = vector_id.index() / 8;
        let bit = vector_id.index() % 8;
        self.signal_change[byte] &= !(1u8 << bit);
        // note, we keep the signal on the change list

        // return reference to value
        &self.data[info.data_range()]
    }

    #[inline]
    fn has_bit_changed(&self, info: &VecBufferInfo, bit: u32) -> bool {
        debug_assert!(bit < info.bits);
        let valid = &self.bit_change[info.change_range()];
        (valid[(bit / 8) as usize] >> (bit % 8)) & 1 == 1
    }

    #[inline]
    fn mark_bit_changed(change: &mut [u8], info: &VecBufferInfo, bit: u32) {
        debug_assert!(bit < info.bits);
        let index = (bit / 8) as usize;
        let changes = &mut change[info.change_range()][index..(index + 1)];
        let mask = 1u8 << (bit % 8);
        changes[0] |= mask;
    }

    #[inline]
    fn has_signal_changed(&self, vec_id: GhwVecId) -> bool {
        let byte = vec_id.index() / 8;
        let bit = vec_id.index() % 8;
        (self.signal_change[byte] >> bit) & 1 == 1
    }

    #[inline]
    fn mark_signal_changed(&mut self, vec_id: GhwVecId) {
        let byte = vec_id.index() / 8;
        let bit = vec_id.index() % 8;
        self.signal_change[byte] |= 1u8 << bit;
        self.change_list.push(vec_id);
    }

    fn get_value(&self, info: &VecBufferInfo, bit: u32) -> u8 {
        debug_assert!(bit < info.bits);
        let data = &self.data[info.data_range()];
        // specialize depending on states
        match info.states {
            States::Two => get_value(info.bits, States::Two, data, bit),
            States::Four => get_value(info.bits, States::Four, data, bit),
            States::Nine => get_value(info.bits, States::Nine, data, bit),
        }
    }

    #[inline]
    fn set_value(data: &mut [u8], info: &VecBufferInfo, bit: u32, value: u8) {
        debug_assert!(value <= 0xf);
        let data = &mut data[info.data_range()];
        // specialize depending on states
        match info.states {
            States::Two => set_value(info.bits, States::Two, data, bit, value),
            States::Four => set_value(info.bits, States::Four, data, bit, value),
            States::Nine => set_value(info.bits, States::Nine, data, bit, value),
        }
    }
}

#[inline]
fn get_value(bits: u32, states: States, data: &[u8], bit: u32) -> u8 {
    debug_assert!(bit < bits);
    let (index, shift) = get_data_index(bits, bit, states);
    let byte = data[index];
    (byte >> shift) & states.mask()
}

#[inline]
fn set_value(bits: u32, states: States, data: &mut [u8], bit: u32, value: u8) {
    debug_assert!(value <= 0xf);
    let (index, shift) = get_data_index(bits, bit, states);
    let data = &mut data[index..(index + 1)];
    let old_data = data[0] & !(states.mask() << shift);
    data[0] = old_data | (value << shift);
}

#[inline]
fn get_data_index(bits: u32, bit: u32, states: States) -> (usize, usize) {
    debug_assert!(bit < bits);
    let bits_in_a_byte = states.bits_in_a_byte() as u32;
    let bytes = bits.div_ceil(bits_in_a_byte);
    let index = bytes - 1 - (bit / bits_in_a_byte);
    let shift = (bit % bits_in_a_byte) * states.bits() as u32;
    (index as usize, shift as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_data_index() {
        // big endian and right aligned
        assert_eq!(get_data_index(4, 0, States::Nine), (1, 0));
        assert_eq!(get_data_index(4, 1, States::Nine), (1, 4));
        assert_eq!(get_data_index(4, 2, States::Nine), (0, 0));
        assert_eq!(get_data_index(4, 3, States::Nine), (0, 4));

        assert_eq!(get_data_index(3, 0, States::Nine), (1, 0));
        assert_eq!(get_data_index(3, 1, States::Nine), (1, 4));
        assert_eq!(get_data_index(3, 2, States::Nine), (0, 0));

        assert_eq!(get_data_index(4, 0, States::Two), (0, 0));
        assert_eq!(get_data_index(4, 1, States::Two), (0, 1));
        assert_eq!(get_data_index(4, 2, States::Two), (0, 2));
        assert_eq!(get_data_index(4, 3, States::Two), (0, 3));
    }
}
