// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::ghw::common::*;
use crate::wavemem::{Encoder, States};
use crate::{Hierarchy, SignalRef};
use std::io::BufRead;

/// Reads the GHW signal values. `input` should be advanced until right after the end of hierarchy
pub(crate) fn read_signals(
    header: &HeaderData,
    info: &GhwDecodeInfo,
    signal_ref_count: usize,
    hierarchy: &Hierarchy,
    input: &mut impl BufRead,
) -> Result<Box<crate::wavemem::Reader>> {
    // TODO: multi-threading
    let mut encoder = Encoder::new(hierarchy);
    let mut vecs = VecBuffer::from_decode_info(info, signal_ref_count);

    // loop over signal sections
    loop {
        let mut mark = [0u8; 4];
        input.read_exact(&mut mark)?;

        // read_sm_hdr
        match &mark {
            GHW_SNAPSHOT_SECTION => {
                read_snapshot_section(header, info, &mut vecs, &mut encoder, input)?
            }
            GHW_CYCLE_SECTION => read_cycle_section(header, info, &mut vecs, &mut encoder, input)?,
            GHW_DIRECTORY_SECTION => {
                // skip the directory by reading it
                let _ = read_directory(header, input)?;
            }
            GHW_TAILER_SECTION => {
                // the "tailer" means that we are done reading the file
                break;
            }
            other => {
                return Err(GhwParseError::UnexpectedSection(
                    String::from_utf8_lossy(other).to_string(),
                ))
            }
        }
    }
    Ok(Box::new(encoder.finish()))
}

fn read_snapshot_section(
    header: &HeaderData,
    info: &GhwDecodeInfo,
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

    for sig in info.signals.iter() {
        read_signal_value(info, sig, vecs, enc, input)?;
    }
    finish_time_step(vecs, enc);

    // check for correct end magic
    check_magic_end(input, "snapshot", GHW_END_SNAPSHOT_SECTION)?;
    Ok(())
}

fn read_cycle_section(
    header: &HeaderData,
    info: &GhwDecodeInfo,
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
    info: &GhwDecodeInfo,
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
        let sig = &info.signals[pos_signal_index - 1];
        read_signal_value(info, sig, vecs, enc, input)?;
    }
    Ok(())
}

/// This dispatches any remaining vector changes.
fn finish_time_step(vecs: &mut VecBuffer, enc: &mut Encoder) {
    vecs.process_changed_signals(|signal_ref, data| {
        enc.raw_value_change(signal_ref, data, States::Nine);
    })
}

fn read_signal_value(
    info: &GhwDecodeInfo,
    signal: &GhwSignal,
    vecs: &mut VecBuffer,
    enc: &mut Encoder,
    input: &mut impl BufRead,
) -> Result<()> {
    match signal.tpe {
        SignalType::NineState(lut) => {
            let value = [info.decode(read_u8(input)?, lut)];
            enc.raw_value_change(signal.signal_ref, &value, States::Nine);
        }
        SignalType::NineStateBit(lut, bit, _) => {
            let value = info.decode(read_u8(input)?, lut);

            // check to see if we already had a change to this same bit in the current time step
            if vecs.is_second_change(signal.signal_ref, bit, value) {
                // immediately dispatch the change to properly reflect the delta cycle
                let data = vecs.get_full_value_and_clear_changes(signal.signal_ref);
                enc.raw_value_change(signal.signal_ref, data, States::Nine);
            }

            // update value
            vecs.update_value(signal.signal_ref, bit, value);

            // check to see if we need to report a change
            if vecs.full_signal_has_changed(signal.signal_ref) {
                let data = vecs.get_full_value_and_clear_changes(signal.signal_ref);
                enc.raw_value_change(signal.signal_ref, data, States::Nine);
            }
        }
        SignalType::U8(bits) => {
            let value = [read_u8(input)?];
            if bits < 8 {
                debug_assert!(value[0] < (1u8 << bits));
            }
            enc.raw_value_change(signal.signal_ref, &value, States::Two);
        }
        SignalType::Leb128Signed(bits) => {
            let value = leb128::read::signed(input)? as u64;
            if bits < u64::BITS {
                debug_assert!(value < (1u64 << bits));
            }
            enc.raw_value_change(signal.signal_ref, &value.to_be_bytes(), States::Two);
        }

        SignalType::F64 => {
            // we need to figure out the endianes here
            let mut buf = [0u8; 8];
            input.read_exact(&mut buf)?;
            todo!(
                "float values: {} or {}?",
                f64::from_le_bytes(buf.clone()),
                f64::from_be_bytes(buf)
            )
        }
    }
    Ok(())
}

/// Keeps track of individual bits and combines them into a full bit vector.
#[derive(Debug)]
struct VecBuffer {
    info: Vec<Option<VecBufferInfo>>,
    data: Vec<u8>,
    bit_change: Vec<u8>,
    change_list: Vec<SignalRef>,
    signal_change: Vec<u8>,
}

#[derive(Debug, Clone)]
struct VecBufferInfo {
    /// Offset in bits
    offset: u64,
    bits: u32,
}

impl VecBufferInfo {
    fn change_range(&self) -> std::ops::Range<usize> {
        // whether a bit has been changed is stored with 8 bits per byte
        let start = self.offset.div_ceil(8) as usize;
        let len = self.bits.div_ceil(8) as usize;
        start..(start + len)
    }
    fn data_range(&self) -> std::ops::Range<usize> {
        // data is stored with 2 bits per byte
        let start = self.offset.div_ceil(2) as usize;
        let len = self.bits.div_ceil(2) as usize;
        start..(start + len)
    }
}

impl VecBuffer {
    fn from_decode_info(decode_info: &GhwDecodeInfo, signal_ref_count: usize) -> Self {
        let mut info = Vec::with_capacity(signal_ref_count);
        info.resize(signal_ref_count, None);
        let mut offset = 0;

        for signal in decode_info.signals.iter() {
            if let SignalType::NineStateBit(_, 0, bits) = signal.tpe {
                if info[signal.signal_ref.index()].is_none() {
                    info[signal.signal_ref.index()] = Some(VecBufferInfo { offset, bits });
                    // pad offset to ensure that each value starts with its own byte
                    let offset_delta = bits.div_ceil(8) * 8;
                    offset += offset_delta as u64;
                }
            }
        }

        let data = vec![0; offset.div_ceil(2) as usize];
        let bit_change = vec![0; offset.div_ceil(8) as usize];
        let change_list = vec![];
        let signal_change = vec![0; signal_ref_count.div_ceil(8)];

        Self {
            info,
            data,
            bit_change,
            change_list,
            signal_change,
        }
    }

    fn process_changed_signals(&mut self, mut callback: impl FnMut(SignalRef, &[u8])) {
        let change_list = std::mem::take(&mut self.change_list);
        for signal_ref in change_list.into_iter() {
            if self.has_signal_changed(signal_ref) {
                let data = self.get_full_value_and_clear_changes(signal_ref);
                (callback)(signal_ref, data);
            }
        }
    }

    #[inline]
    fn is_second_change(&self, signal_ref: SignalRef, bit: u32, value: u8) -> bool {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();
        self.has_bit_changed(info, bit) && self.get_value(info, bit) != value
    }

    #[inline]
    fn update_value(&mut self, signal_ref: SignalRef, bit: u32, value: u8) {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();
        let is_a_real_change = self.get_value(info, bit) != value;
        if is_a_real_change {
            Self::mark_bit_changed(&mut self.bit_change, info, bit);
            Self::set_value(&mut self.data, info, bit, value);
            // add signal to change list if it has not already been added
            if !self.has_signal_changed(signal_ref) {
                self.mark_signal_changed(signal_ref);
            }
        }
    }

    /// Used in order to dispatch full signal changes as soon as possible
    #[inline]
    fn full_signal_has_changed(&self, signal_ref: SignalRef) -> bool {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();

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
    fn get_full_value_and_clear_changes(&mut self, signal_ref: SignalRef) -> &[u8] {
        let info = (&self.info[signal_ref.index()].as_ref()).unwrap();

        // clear bit changes
        let changes = &mut self.bit_change[info.change_range()];
        for e in changes.iter_mut() {
            *e = 0;
        }

        // clear signal change
        let byte = signal_ref.index() / 8;
        let bit = signal_ref.index() % 8;
        self.signal_change[byte] = self.signal_change[byte] & !(1u8 << bit);
        // note, we keep the signal on the change list

        // return reference to value
        let data = &self.data[info.data_range()];
        data
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
    fn has_signal_changed(&self, signal_ref: SignalRef) -> bool {
        let byte = signal_ref.index() / 8;
        let bit = signal_ref.index() % 8;
        (self.signal_change[byte] >> bit) & 1 == 1
    }

    #[inline]
    fn mark_signal_changed(&mut self, signal_ref: SignalRef) {
        let byte = signal_ref.index() / 8;
        let bit = signal_ref.index() % 8;
        self.signal_change[byte] |= 1u8 << bit;
        self.change_list.push(signal_ref);
    }

    #[inline]
    fn get_value(&self, info: &VecBufferInfo, bit: u32) -> u8 {
        debug_assert!(bit < info.bits);
        let data = &self.data[info.data_range()];
        let (index, is_lsb) = Self::get_data_index(info.bits, bit);
        let byte = data[index];
        if is_lsb {
            byte & 0xf
        } else {
            (byte >> 4) & 0xf
        }
    }

    #[inline]
    fn set_value(data: &mut [u8], info: &VecBufferInfo, bit: u32, value: u8) {
        debug_assert!(value <= 0xf);
        let (index, is_lsb) = Self::get_data_index(info.bits, bit);
        let data = &mut data[info.data_range()][index..(index + 1)];
        if is_lsb {
            data[0] = (data[0] & 0xf0) | value;
        } else {
            data[0] = (data[0] & 0x0f) | (value << 4);
        }
    }

    /// We need to store "left-aligned", big endian 9-value bits.
    #[inline]
    fn get_data_index(bits: u32, bit: u32) -> (usize, bool) {
        debug_assert!(bit < bits);
        let mirrored = bits - 1 - bit;
        let index = (mirrored / 2) as usize;
        let is_lsb = mirrored % 2 == 1;
        (index, is_lsb)
    }
}
