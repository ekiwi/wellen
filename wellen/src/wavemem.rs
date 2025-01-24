// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// Fast and compact wave-form representation inspired by the FST on disk format.

use crate::compressed::Compression;
use crate::fst::{get_bytes_per_entry, get_len_and_meta, push_zeros};
use crate::hierarchy::{Hierarchy, SignalRef};
use crate::signals::{
    FixedWidthEncoding, Real, Signal, SignalSource, SignalSourceImplementation, Time, TimeTableIdx,
};
use crate::{SignalEncoding, SignalValue, TimeTable};
use num_enum::TryFromPrimitive;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::io::Read;
use std::num::NonZeroU32;

/// Holds queryable waveform data. Use the `Encoder` to generate.
pub struct Reader {
    blocks: Vec<Block>,
}

impl SignalSourceImplementation for Reader {
    fn load_signals(
        &mut self,
        ids: &[SignalRef],
        types: &[SignalEncoding],
        multi_threaded: bool,
    ) -> Vec<Signal> {
        if multi_threaded {
            ids.par_iter()
                .zip(types.par_iter())
                .map(|(id, len)| self.load_signal(*id, *len))
                .collect::<Vec<_>>()
        } else {
            ids.iter()
                .zip(types.iter())
                .map(|(id, len)| self.load_signal(*id, *len))
                .collect::<Vec<_>>()
        }
    }

    fn print_statistics(&self) {
        println!("[wavemem] size in memory: {} bytes", self.size_in_memory());
        self.print_statistics();
    }
}

impl Reader {
    pub fn size_in_memory(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let blocks = self
            .blocks
            .iter()
            .map(|b| b.size_in_memory())
            .sum::<usize>();
        base + blocks
    }

    pub fn print_statistics(&self) {
        println!("[wavemem] there are {} blocks.", self.blocks.len());
        let max_time_table_size = self
            .blocks
            .iter()
            .map(|b| b.time_table.len())
            .max()
            .unwrap();
        println!(
            "[wavemem] the maximum time table size is {}.",
            max_time_table_size
        );
        let total_data_size = self
            .blocks
            .iter()
            .map(|b| b.data.len() * std::mem::size_of::<u8>())
            .sum::<usize>();
        let total_offset_size = self
            .blocks
            .iter()
            .map(|b| b.offsets.len() * std::mem::size_of::<SignalDataOffset>())
            .sum::<usize>();
        let total_time_table_size = self
            .blocks
            .iter()
            .map(|b| b.time_table.len() * std::mem::size_of::<Time>())
            .sum::<usize>();
        println!(
            "[wavemem] data across all blocks takes up {} bytes.",
            total_data_size
        );
        println!(
            "[wavemem] offsets across all blocks take up {} bytes.",
            total_offset_size
        );
        println!(
            "[wavemem] time table data across all blocks takes up {} bytes.",
            total_time_table_size
        );
    }

    fn collect_signal_meta_data(&self, id: SignalRef) -> SignalMetaData {
        let mut time_idx_offset = 0;
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.iter() {
            if let Some((start_ii, data_len)) = block.get_offset_and_length(id) {
                let end_ii = start_ii + data_len;
                // uncompress if necessary
                let mut reader = std::io::Cursor::new(&block.data[start_ii..end_ii]);
                let meta_data_raw = leb128::read::unsigned(&mut reader).unwrap();
                let meta_data = SignalEncodingMetaData::decode(meta_data_raw);
                let data_block = &block.data[start_ii + reader.position() as usize..end_ii];
                blocks.push((time_idx_offset, data_block, meta_data));
            }
            time_idx_offset += block.time_table.len() as u32;
        }
        let max_states = blocks
            .iter()
            .map(|b| b.2.max_states)
            .reduce(States::join)
            .unwrap_or(States::Nine);
        SignalMetaData { max_states, blocks }
    }

    fn load_signal(&self, id: SignalRef, tpe: SignalEncoding) -> Signal {
        let meta = self.collect_signal_meta_data(id);
        load_compressed_signal(meta, id, tpe)
    }
}

pub(crate) fn load_compressed_signal(
    meta: SignalMetaData,
    id: SignalRef,
    tpe: SignalEncoding,
) -> Signal {
    let mut time_indices: Vec<TimeTableIdx> = Vec::new();
    let mut data_bytes: Vec<u8> = Vec::new();
    let mut strings: Vec<String> = Vec::new();
    for (time_idx_offset, data_block, meta_data) in meta.blocks.into_iter() {
        let data = match meta_data.compression {
            Compression::Lz4(uncompressed_len) => {
                let data = lz4_flex::decompress(data_block, uncompressed_len).unwrap();
                Cow::Owned(data)
            }
            Compression::None => Cow::Borrowed(data_block),
        };

        match tpe {
            SignalEncoding::String => {
                load_signal_strings(
                    &mut data.as_ref(),
                    time_idx_offset,
                    &mut time_indices,
                    &mut strings,
                );
            }
            SignalEncoding::BitVector(signal_len) => {
                load_fixed_len_signal(
                    &mut data.as_ref(),
                    time_idx_offset,
                    signal_len.get(),
                    meta.max_states,
                    &mut time_indices,
                    &mut data_bytes,
                    id,
                );
            }
            SignalEncoding::Real => {
                load_reals(
                    &mut data.as_ref(),
                    time_idx_offset,
                    &mut time_indices,
                    &mut data_bytes,
                );
            }
        }
    }

    match tpe {
        SignalEncoding::String => {
            debug_assert!(data_bytes.is_empty());
            Signal::new_var_len(id, time_indices, strings)
        }
        SignalEncoding::BitVector(len) => {
            debug_assert!(strings.is_empty());
            let (bytes, meta_byte) = get_len_and_meta(meta.max_states, len.get());
            let encoding = FixedWidthEncoding::BitVector {
                max_states: meta.max_states,
                bits: len.get(),
                meta_byte,
            };
            Signal::new_fixed_len(
                id,
                time_indices,
                encoding,
                get_bytes_per_entry(bytes, meta_byte) as u32,
                data_bytes,
            )
        }
        SignalEncoding::Real => {
            assert!(strings.is_empty());
            Signal::new_fixed_len(id, time_indices, FixedWidthEncoding::Real, 8, data_bytes)
        }
    }
}

/// Data about a single signal inside a Reader.
/// Only used internally by `collect_signal_meta_data`
pub(crate) struct SignalMetaData<'a> {
    pub(crate) max_states: States,
    /// For every block that contains the signal: time_idx_offset, data and meta-data
    pub(crate) blocks: Vec<(u32, &'a [u8], SignalEncodingMetaData)>,
}

#[inline]
fn load_reals(
    data: &mut impl Read,
    time_idx_offset: u32,
    time_indices: &mut Vec<TimeTableIdx>,
    out: &mut Vec<u8>,
) {
    let mut last_time_idx = time_idx_offset;

    while let Ok(value) = leb128::read::unsigned(data) {
        let time_idx_delta = value as u32;
        last_time_idx += time_idx_delta;

        // read 8 bytes of reald
        let mut buf = vec![0u8; 8];
        data.read_exact(buf.as_mut()).unwrap();

        // check to see if the value actually changed
        let changed = if out.is_empty() {
            true
        } else {
            out[out.len() - 8..] != buf
        };
        if changed {
            out.append(&mut buf);
            time_indices.push(last_time_idx)
        }
    }
}

#[inline]
fn load_fixed_len_signal(
    data: &mut impl Read,
    time_idx_offset: u32,
    bits: u32,
    signal_states: States,
    time_indices: &mut Vec<TimeTableIdx>,
    out: &mut Vec<u8>,
    _signal_id: SignalRef, // for debugging
) {
    let mut last_time_idx = time_idx_offset;
    let (len, has_meta) = get_len_and_meta(signal_states, bits);
    let bytes_per_entry = get_bytes_per_entry(len, has_meta);

    while let Ok(value) = leb128::read::unsigned(data) {
        let time_idx_delta_raw = value as u32;
        // now the decoding depends on the size and whether it is two state
        let time_idx_delta = match bits {
            1 => {
                let value = (time_idx_delta_raw & 0xf) as u8;
                let states = States::from_value(value);
                let meta_data = (states as u8) << 6;
                out.push(value | meta_data);
                // time delta is encoded together with the value
                time_idx_delta_raw >> 4
            }
            other_len => {
                // the lower 2 bits of the time idx delta encode how many state bits are encoded in the local signal
                let local_encoding =
                    States::try_from_primitive((time_idx_delta_raw & 0x3) as u8).unwrap();
                let num_bytes = (other_len as usize).div_ceil(local_encoding.bits_in_a_byte());
                let mut buf = vec![0u8; num_bytes];
                data.read_exact(buf.as_mut()).unwrap();
                let (local_len, local_has_meta) = get_len_and_meta(local_encoding, bits);

                // append data
                let meta_data = (local_encoding as u8) << 6;
                if local_len == len && local_has_meta == has_meta {
                    // same meta-data location and length as the maximum
                    if has_meta {
                        out.push(meta_data);
                        out.append(&mut buf);
                    } else {
                        if meta_data > 0 {
                            debug_assert_eq!(buf[0] & 0x3f, buf[0], "unexpected data in upper 2-bits of buf[0]={:x} {_signal_id:?} {len} {signal_states:?}", buf[0]);
                        }
                        out.push(meta_data | buf[0]);
                        out.extend_from_slice(&buf[1..]);
                    }
                } else {
                    // smaller encoding than the maximum
                    out.push(meta_data);
                    if has_meta {
                        push_zeros(out, len - local_len);
                    } else {
                        push_zeros(out, len - local_len - 1);
                    }
                    out.append(&mut buf);
                }
                //
                time_idx_delta_raw >> 2
            }
        };
        // see if there actually was a change and revert if there was not
        last_time_idx += time_idx_delta;
        if check_if_changed_and_truncate(bytes_per_entry, out) {
            time_indices.push(last_time_idx);
        }
    }

    debug_assert_eq!(out.len(), time_indices.len() * bytes_per_entry);
}

pub fn check_if_changed_and_truncate(bytes_per_entry: usize, out: &mut Vec<u8>) -> bool {
    let changed = if out.len() < 2 * bytes_per_entry {
        true
    } else {
        let prev_start = out.len() - 2 * bytes_per_entry;
        let new_start = out.len() - bytes_per_entry;
        out[prev_start..new_start] != out[new_start..]
    };

    if !changed {
        // remove new value
        out.truncate(out.len() - bytes_per_entry);
    }

    changed
}

#[inline]
fn load_signal_strings(
    data: &mut impl Read,
    time_idx_offset: u32,
    time_indices: &mut Vec<TimeTableIdx>,
    out: &mut Vec<String>,
) {
    let mut last_time_idx = time_idx_offset;

    while let Ok(value) = leb128::read::unsigned(data) {
        let time_idx_delta = value as u32;
        last_time_idx += time_idx_delta;

        // read variable length string
        let len = leb128::read::unsigned(data).unwrap() as usize;
        let mut buf = vec![0u8; len];
        data.read_exact(&mut buf).unwrap();
        let str_value = String::from_utf8_lossy(&buf).to_string();

        // check to see if the value actually changed
        let changed = out.last().map(|prev| prev != &str_value).unwrap_or(true);
        if changed {
            out.push(str_value);
            time_indices.push(last_time_idx);
        }
    }
}

/// A block that contains all value changes in a certain time segment.
/// Note that while in FST blocks can be skipped, here we only use blocks
/// in order to combine data from different threads and to compress partial data.
struct Block {
    start_time: Time,
    time_table: Vec<Time>,
    /// Offsets of (potentially compressed) signal data.
    offsets: Vec<Option<SignalDataOffset>>,
    /// Data for all signals in block
    data: Vec<u8>,
}

impl Block {
    fn size_in_memory(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let time = self.time_table.len() * std::mem::size_of::<Time>();
        let offsets = self.offsets.len() * std::mem::size_of::<SignalDataOffset>();
        let data = self.data.len() * std::mem::size_of::<u8>();
        base + time + offsets + data
    }

    fn end_time(&self) -> Time {
        *self.time_table.last().unwrap()
    }

    fn get_offset_and_length(&self, id: SignalRef) -> Option<(usize, usize)> {
        let offset = match self.offsets[id.index()] {
            None => return None,
            Some(offset) => offset.get_index(),
        };
        // find the next offset or take the data len
        let next_offset = self
            .offsets
            .iter()
            .skip(id.index() + 1)
            .find(|o| o.is_some())
            .map(|o| o.unwrap().get_index())
            .unwrap_or(self.data.len());
        Some((offset, next_offset - offset))
    }
}

/// Position of first byte of a signal in the block data.
#[derive(Debug, Clone, Copy)]
struct SignalDataOffset(NonZeroU32);

impl SignalDataOffset {
    fn new(index: usize) -> Self {
        SignalDataOffset(NonZeroU32::new((index as u32) + 1).unwrap())
    }
    fn get_index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Encodes value and time changes into a compressed in-memory representation.
pub struct Encoder {
    /// Time table under construction
    time_table: Vec<Time>,
    /// Signals under construction
    signals: Vec<SignalEncoder>,
    /// Tracks if there has been any new data that would require us to create another block.
    has_new_data: bool,
    /// Tracks if we are skipping a timestep because it came with an invalid time.
    skipping_time_step: bool,
    /// Finished blocks
    blocks: Vec<Block>,
}

/// Indexes the time table inside a block.
type BlockTimeIdx = u16;

impl Encoder {
    pub fn new(hierarchy: &Hierarchy) -> Self {
        let mut signals = Vec::with_capacity(hierarchy.num_unique_signals());
        for var in hierarchy.get_unique_signals_vars() {
            let tpe = match var {
                None => SignalEncoding::String, // we do not know!
                Some(var) => var.signal_encoding(),
            };
            let pos = signals.len();
            signals.push(SignalEncoder::new(tpe, pos));
        }

        Encoder {
            time_table: Vec::default(),
            signals,
            has_new_data: false,
            skipping_time_step: false,
            blocks: Vec::default(),
        }
    }

    pub fn time_change(&mut self, time: u64) {
        // sanity check to make sure that time is increasing
        if let Some(prev_time) = self.time_table.last() {
            match prev_time.cmp(&time) {
                Ordering::Equal => {
                    return; // ignore calls to time_change that do not actually change anything
                }
                Ordering::Greater => {
                    println!(
                        "WARN: time decreased from {} to {}. Skipping!",
                        *prev_time, time
                    );
                    self.skipping_time_step = true;
                    return;
                }
                Ordering::Less => {
                    // this is the normal situation where we actually increment the time
                }
            }
        }
        // if we run out of time indices => start a new block
        if self.time_table.len() >= BlockTimeIdx::MAX as usize {
            self.finish_block();
        }
        self.time_table.push(time);
        self.has_new_data = true;
        self.skipping_time_step = false;
    }

    /// Call with an unaltered VCD value.
    pub fn vcd_value_change(&mut self, id: u64, value: &[u8]) {
        assert!(
            !self.time_table.is_empty(),
            "We need a call to time_change first!"
        );
        if !self.skipping_time_step {
            let time_idx = (self.time_table.len() - 1) as TimeTableIdx;
            self.signals[id as usize].add_vcd_change(time_idx, value);
            self.has_new_data = true;
        }
    }

    /// Call with a value that is already encoded in our internal format.
    pub fn raw_value_change(&mut self, id: SignalRef, value: &[u8], states: States) {
        assert!(
            !self.time_table.is_empty(),
            "We need a call to time_change first!"
        );
        if !self.skipping_time_step {
            let time_idx = (self.time_table.len() - 1) as TimeTableIdx;
            self.signals[id.index()].add_n_bit_change(time_idx, value, states);
            self.has_new_data = true;
        }
    }

    pub fn real_change(&mut self, id: SignalRef, value: f64) {
        assert!(
            !self.time_table.is_empty(),
            "We need a call to time_change first!"
        );
        if !self.skipping_time_step {
            let time_idx = (self.time_table.len() - 1) as TimeTableIdx;
            self.signals[id.index()].add_real_change(time_idx, value);
            self.has_new_data = true;
        }
    }

    pub fn finish(mut self) -> (SignalSource, TimeTable) {
        // ensure that we have no open blocks
        self.finish_block();
        // create a new reader with the blocks that we have
        let reader = Reader {
            blocks: self.blocks,
        };
        let time_table = Self::combine_time_tables(&reader.blocks);
        (SignalSource::new(Box::new(reader)), time_table)
    }

    fn combine_time_tables(blocks: &[Block]) -> TimeTable {
        // create a combined time table from all blocks
        let len = blocks.iter().map(|b| b.time_table.len()).sum::<usize>();
        let mut table = Vec::with_capacity(len);
        for block in blocks.iter() {
            table.extend_from_slice(&block.time_table);
        }
        table
    }

    // appends the contents of the other encoder to this one
    pub fn append(&mut self, mut other: Encoder) {
        // ensure that we have no open blocks
        self.finish_block();
        // ensure that the other encoder is also done
        other.finish_block();
        // if the other encoder has no blocks, there is nothing for us to do
        if let Some(other_first_block) = other.blocks.first() {
            // make sure the timeline fits
            let us_end_time = self.blocks.last().unwrap().end_time();
            let other_start = other_first_block.start_time;
            assert!(
                us_end_time <= other_start,
                "Can only append encoders in chronological order!"
            );
            // append all blocks from the other encoder
            self.blocks.append(&mut other.blocks);
        }
    }

    fn finish_block(&mut self) {
        if !self.has_new_data {
            return; // nothing to do!
        }
        let signal_count = self.signals.len();
        let mut offsets = Vec::with_capacity(signal_count);
        let mut data: Vec<u8> = Vec::with_capacity(128);
        for signal in self.signals.iter_mut() {
            if let Some((mut signal_data, is_compressed)) = signal.finish() {
                let offset = SignalDataOffset::new(data.len());
                offsets.push(Some(offset));
                let meta_data = is_compressed.encode();
                leb128::write::unsigned(&mut data, meta_data).unwrap();
                data.append(&mut signal_data);
            } else {
                offsets.push(None);
            }
        }
        let start_time = *self.time_table.first().unwrap();
        // the next block might continue the time step
        let end_time = *self.time_table.last().unwrap();
        let new_time_table = vec![end_time];
        let mut time_table = std::mem::replace(&mut self.time_table, new_time_table);
        time_table.shrink_to_fit();
        offsets.shrink_to_fit();
        data.shrink_to_fit();
        let block = Block {
            start_time,
            time_table,
            offsets,
            data,
        };
        self.blocks.push(block);
        self.has_new_data = false;
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct SignalEncodingMetaData {
    pub(crate) compression: Compression,
    pub(crate) max_states: States,
}

/// We divide the decompressed size by this number and round up.
/// This is OK, since it will just make us allocate a slightly too larger buffer.
const SIGNAL_DECOMPRESSED_LEN_DIV: u32 = 32;

impl SignalEncodingMetaData {
    fn uncompressed(max_states: States) -> Self {
        SignalEncodingMetaData {
            compression: Compression::None,
            max_states,
        }
    }

    fn compressed(max_states: States, uncompressed_len: usize) -> Self {
        // turn the length into a value that we can actually encode
        let uncompressed_len_approx = (uncompressed_len as u32)
            .div_ceil(SIGNAL_DECOMPRESSED_LEN_DIV)
            * SIGNAL_DECOMPRESSED_LEN_DIV;
        SignalEncodingMetaData {
            compression: Compression::Lz4(uncompressed_len_approx as usize),
            max_states,
        }
    }

    fn decode(data: u64) -> Self {
        let max_states = States::try_from_primitive((data & 3) as u8).unwrap();
        let is_compressed = (data >> 2) & 1 == 1;
        let compression = if is_compressed {
            let decompressed_len_bits = ((data >> 3) & u32::MAX as u64) as u32;
            let decompressed_len = decompressed_len_bits * SIGNAL_DECOMPRESSED_LEN_DIV;
            Compression::Lz4(decompressed_len as usize)
        } else {
            Compression::None
        };
        SignalEncodingMetaData {
            compression,
            max_states,
        }
    }
    fn encode(&self) -> u64 {
        match &self.compression {
            Compression::Lz4(decompressed_len) => {
                let decompressed_len_bits =
                    ((*decompressed_len) as u32).div_ceil(SIGNAL_DECOMPRESSED_LEN_DIV);

                ((decompressed_len_bits as u64) << 3) | (1 << 2) | (self.max_states as u64)
            }
            Compression::None => self.max_states as u64,
        }
    }
}

/// Encodes changes for a single signal.
#[derive(Debug, Clone)]
struct SignalEncoder {
    data: Vec<u8>,
    tpe: SignalEncoding,
    prev_time_idx: TimeTableIdx,
    max_states: States,
    /// Same as the index of this encoder in a Vec<_>. Used for debugging purposes.
    #[allow(unused)]
    signal_idx: u32,
}

impl SignalEncoder {
    fn new(tpe: SignalEncoding, pos: usize) -> Self {
        SignalEncoder {
            data: Vec::default(),
            tpe,
            prev_time_idx: 0,
            max_states: States::Two, // we start out assuming we are dealing with a two state signal
            signal_idx: pos as u32,
        }
    }
}

/// Minimum number of bytes for a signal to warrant an attempt at LZ4 compression.
const MIN_SIZE_TO_COMPRESS: usize = 32;
/// Flag to turn off compression.
const SKIP_COMPRESSION: bool = false;

impl SignalEncoder {
    /// Adds a 2, 4 or 9-value change that has already been converted into our internal format.
    fn add_n_bit_change(&mut self, time_index: TimeTableIdx, value: &[u8], states: States) {
        let time_idx_delta = time_index - self.prev_time_idx;
        self.max_states = States::join(self.max_states, states);
        match self.tpe {
            SignalEncoding::BitVector(len) => {
                let bits = len.get();
                if bits == 1 {
                    debug_assert_eq!(value.len(), 1);
                    let value = value[0];
                    debug_assert_eq!(value & 0xf, value, "leading bits are not zero: {value:x}");
                    let write_value = ((time_idx_delta as u64) << 4) + value as u64;
                    leb128::write::unsigned(&mut self.data, write_value).unwrap();
                } else {
                    // sometimes we might include some leading zeros that are not necessary
                    let required_bytes = (bits as usize).div_ceil(states.bits_in_a_byte());
                    debug_assert!(value.len() >= required_bytes);
                    let value = &value[(value.len() - required_bytes)..];

                    // we automatically compress the signal to its minimum states encoding
                    let min_states = check_min_state(value, states);
                    // write time and meta data
                    let time_and_meta = (time_idx_delta as u64) << 2 | (min_states as u64);
                    leb128::write::unsigned(&mut self.data, time_and_meta).unwrap();
                    let data_start_index = self.data.len();
                    if min_states == states {
                        // raw data
                        self.data.extend_from_slice(value);
                    } else {
                        compress(value, states, min_states, bits as usize, &mut self.data);
                    }

                    // make sure the leading bits are 0
                    if cfg!(debug_assertions) {
                        let first_byte = self.data[data_start_index];
                        let first_byte_mask = min_states.first_byte_mask(bits);
                        debug_assert_eq!(
                            first_byte & first_byte_mask,
                            first_byte,
                            "{first_byte:x} & {first_byte_mask:x} {bits} {min_states:?}\n{value:?}"
                        );
                    }
                }
            }
            other => unreachable!("Cannot call add_n_bit_change on signal of type: {other:?}"),
        }
        // update time index to calculate next delta
        self.prev_time_idx = time_index;
    }

    fn add_real_change(&mut self, time_index: TimeTableIdx, value: f64) {
        let time_idx_delta = time_index - self.prev_time_idx;

        // write var-length time index + fixed little endian float bytes
        leb128::write::unsigned(&mut self.data, time_idx_delta as u64).unwrap();
        self.data.extend_from_slice(&value.to_le_bytes());

        // update time index to calculate next delta
        self.prev_time_idx = time_index;
    }

    fn add_str_change(&mut self, time_index: TimeTableIdx, value: &str) {
        let time_idx_delta = time_index - self.prev_time_idx;

        // string: var-length time index + var-len length + content
        leb128::write::unsigned(&mut self.data, time_idx_delta as u64).unwrap();
        leb128::write::unsigned(&mut self.data, value.len() as u64).unwrap();
        self.data.extend_from_slice(value.as_bytes());

        // update time index to calculate next delta
        self.prev_time_idx = time_index;
    }

    /// Adds a change from a VCD string.
    fn add_vcd_change(&mut self, time_index: TimeTableIdx, value: &[u8]) {
        let time_idx_delta = time_index - self.prev_time_idx;
        match self.tpe {
            SignalEncoding::BitVector(len) => {
                let value_bits: &[u8] = match value[0] {
                    b'b' | b'B' => &value[1..],
                    _ => value,
                };
                // special detection for pymtl3 which adds an extra `0b` for all bit vectors
                let value_bits: &[u8] = if value_bits.len() <= 2 {
                    value_bits
                } else {
                    match &value_bits[0..2] {
                        b"0b" => &value_bits[2..],
                        _ => value_bits,
                    }
                };
                if len.get() == 1 {
                    let value_char = match value_bits {
                        // special handling for empty values which we always treat as zero
                        [] => b'0',
                        [v] => *v,
                        _ => unreachable!(
                            "value bits are too long for 0-bit or 1-bit signal: {}",
                            String::from_utf8_lossy(value)
                        ),
                    };
                    let states =
                        try_write_1_bit_9_state(time_idx_delta, value_char, &mut self.data)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Failed to parse four state value: {} for signal of size 1",
                                    String::from_utf8_lossy(value)
                                )
                            });
                    self.max_states = States::join(self.max_states, states);
                } else {
                    let states = check_states(value_bits).unwrap_or_else(|| {
                        panic!(
                            "Bit-vector contains invalid character. Only 2, 4 and 9-state signals are supported: {}",
                            String::from_utf8_lossy(value)
                        )
                    });
                    self.max_states = States::join(self.max_states, states);

                    // write time delta + num-states meta-data
                    let time_and_meta = (time_idx_delta as u64) << 2 | (states as u64);
                    leb128::write::unsigned(&mut self.data, time_and_meta).unwrap();
                    // write actual data
                    let bits = len.get() as usize;
                    let data_to_write = if value_bits.len() == bits {
                        Cow::Borrowed(value_bits)
                    } else {
                        let expanded = expand_special_vector_cases(value_bits, bits)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Failed to parse four state value: {} for signal of size {}",
                                    String::from_utf8_lossy(value),
                                    bits
                                )
                            });
                        assert_eq!(expanded.len(), bits);
                        Cow::Owned(expanded)
                    };
                    write_n_state(states, &data_to_write, &mut self.data, None);
                }
            }
            SignalEncoding::String => {
                assert!(
                    matches!(value[0], b's' | b'S'),
                    "expected a string, not {}",
                    String::from_utf8_lossy(value)
                );
                // string: var-length time index + var-len length + content
                leb128::write::unsigned(&mut self.data, time_idx_delta as u64).unwrap();
                leb128::write::unsigned(&mut self.data, (value.len() - 1) as u64).unwrap();
                self.data.extend_from_slice(&value[1..]);
            }
            SignalEncoding::Real => {
                assert!(
                    matches!(value[0], b'r' | b'R'),
                    "expected a real, not {}",
                    String::from_utf8_lossy(value)
                );
                // parse float
                let float_value: Real = std::str::from_utf8(&value[1..])
                    .unwrap()
                    .parse::<Real>()
                    .unwrap();
                // write var-length time index + fixed little endian float bytes
                leb128::write::unsigned(&mut self.data, time_idx_delta as u64).unwrap();
                self.data.extend_from_slice(&float_value.to_le_bytes());
            }
        }
        self.prev_time_idx = time_index;
    }

    /// returns a compressed signal representation
    fn finish(&mut self) -> Option<(Vec<u8>, SignalEncodingMetaData)> {
        // reset time index for the next block
        self.prev_time_idx = 0;

        // no updates
        if self.data.is_empty() {
            return None;
        }
        // replace data, the actual meta data stays the same
        let data = std::mem::take(&mut self.data);

        // is there so little data that compression does not make sense?
        if data.len() < MIN_SIZE_TO_COMPRESS || SKIP_COMPRESSION {
            return Some((data, SignalEncodingMetaData::uncompressed(self.max_states)));
        }
        // attempt a compression
        let compressed = lz4_flex::compress(&data);
        if (compressed.len() + 1) >= data.len() {
            Some((data, SignalEncodingMetaData::uncompressed(self.max_states)))
        } else {
            Some((
                compressed,
                SignalEncodingMetaData::compressed(self.max_states, data.len()),
            ))
        }
    }
}

/// Compress a Signal by replaying all changes on our SignalEncoder.
pub(crate) fn compress_signal(signal: &Signal) -> Option<(Vec<u8>, SignalEncodingMetaData)> {
    let mut enc = SignalEncoder::new(signal.signal_encoding(), signal.signal_ref().index());
    let mut scratch = vec![];
    for (time, value) in signal.iter_changes() {
        if let Some((data, mask)) = value.data_and_mask() {
            let states = value.states().unwrap();
            if mask == u8::MAX {
                enc.add_n_bit_change(time, data, states);
            } else {
                // make a copy to allow us to mask out bits
                scratch.extend_from_slice(data);
                scratch[0] &= mask;
                enc.add_n_bit_change(time, &scratch, states);
                scratch.clear();
            }
        } else if let SignalValue::Real(data) = value {
            enc.add_real_change(time, data);
        } else if let SignalValue::String(data) = value {
            enc.add_str_change(time, data);
        } else {
            unreachable!()
        }
    }
    enc.finish()
}

#[inline]
fn expand_special_vector_cases(value: &[u8], len: usize) -> Option<Vec<u8>> {
    // if the value is actually longer than expected, there is nothing we can do
    if value.len() >= len {
        return None;
    }

    // zero, x or z extend
    match value[0] {
        b'1' | b'0' => {
            let mut extended = Vec::with_capacity(len);
            extended.resize(len - value.len(), b'0');
            extended.extend_from_slice(value);
            Some(extended)
        }
        b'x' | b'X' | b'z' | b'Z' => {
            let mut extended = Vec::with_capacity(len);
            extended.resize(len - value.len(), value[0]);
            extended.extend_from_slice(value);
            Some(extended)
        }
        _ => None, // failed
    }
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub enum States {
    #[default]
    Two = 0,
    Four = 1,
    Nine = 2,
}

impl States {
    fn from_value(value: u8) -> Self {
        if value <= 1 {
            States::Two
        } else if value <= 3 {
            States::Four
        } else {
            States::Nine
        }
    }
    pub fn join(a: Self, b: Self) -> Self {
        let num = std::cmp::max(a as u8, b as u8);
        Self::try_from_primitive(num).unwrap()
    }
    /// Returns how many bits are needed in order to encode one bit of state.
    #[inline]
    pub fn bits(&self) -> usize {
        match self {
            States::Two => 1,
            States::Four => 2,
            States::Nine => 4,
        }
    }

    #[inline]
    pub fn mask(&self) -> u8 {
        match self {
            States::Two => 0x1,
            States::Four => 0x3,
            States::Nine => 0xf,
        }
    }

    /// Returns how many signal bits can be encoded in a u8.
    #[inline]
    pub fn bits_in_a_byte(&self) -> usize {
        8 / self.bits()
    }

    /// Returns how many bits the first byte would contain.
    #[inline]
    fn bits_in_first_byte(&self, bits: u32) -> u32 {
        (bits * self.bits() as u32) % u8::BITS
    }

    /// Creates a mask that will only leave the relevant bits in the first byte.
    #[inline]
    pub(crate) fn first_byte_mask(&self, bits: u32) -> u8 {
        let n = self.bits_in_first_byte(bits);
        if n > 0 {
            (1u8 << n) - 1
        } else {
            u8::MAX
        }
    }
}

#[cfg(feature = "benchmark")]
pub fn check_states_pub(value: &[u8]) -> Option<usize> {
    check_states(value).map(|s| s.bits())
}

#[inline]
fn check_min_state(value: &[u8], states: States) -> States {
    if states == States::Two {
        return States::Two;
    }

    let mut union = 0;
    for v in value.iter() {
        for ii in 0..states.bits_in_a_byte() {
            union |= ((*v) >> (ii * states.bits())) & states.mask();
        }
    }
    States::from_value(union)
}

/// picks a specialized compress implementation
fn compress(value: &[u8], in_states: States, out_states: States, bits: usize, out: &mut Vec<u8>) {
    match (in_states, out_states) {
        (States::Nine, States::Two) => {
            compress_template(value, States::Nine, States::Two, bits, out)
        }
        (States::Four, States::Two) => {
            compress_template(value, States::Four, States::Two, bits, out)
        }
        (States::Nine, States::Four) => {
            compress_template(value, States::Nine, States::Four, bits, out)
        }
        _ => unreachable!("Cannot compress {in_states:?} => {out_states:?}"),
    }
}

#[inline]
fn compress_template(
    value: &[u8],
    in_states: States,
    out_states: States,
    bits: usize,
    out: &mut Vec<u8>,
) {
    debug_assert!(in_states.bits_in_a_byte() < out_states.bits_in_a_byte());
    let mut working_byte = 0u8;
    let max_bits = value.len() * in_states.bits_in_a_byte();
    for bit in (0..bits).rev() {
        let rev_bit = max_bits - bit - 1;
        let in_byte = value[rev_bit / in_states.bits_in_a_byte()];
        let in_value =
            (in_byte >> ((bit % in_states.bits_in_a_byte()) * in_states.bits())) & in_states.mask();
        debug_assert!(in_value <= out_states.mask(), "{in_value:?}");

        working_byte = (working_byte << out_states.bits()) + in_value;
        if bit % out_states.bits_in_a_byte() == 0 {
            out.push(working_byte);
            working_byte = 0;
        }
    }
}

#[inline]
pub fn check_states(value: &[u8]) -> Option<States> {
    let mut union = 0;
    for cc in value.iter() {
        union |= bit_char_to_num(*cc)?;
    }
    Some(States::from_value(union))
}

#[inline]
pub fn bit_char_to_num(value: u8) -> Option<u8> {
    match value {
        // Value shared with 2 and 4-state logic
        b'0' | b'1' => Some(value - b'0'), // strong 0 / strong 1
        // Values shared with Verilog 4-state logic
        b'x' | b'X' => Some(2), // strong o or 1 (unknown)
        b'z' | b'Z' => Some(3), // high impedance
        // Values unique to the IEEE Standard Logic Type
        b'h' | b'H' => Some(4), // weak 1
        b'u' | b'U' => Some(5), // uninitialized
        b'w' | b'W' => Some(6), // weak 0 or 1 (unknown)
        b'l' | b'L' => Some(7), // weak 1
        b'-' => Some(8),        // don't care
        _ => None,
    }
}

#[inline]
fn try_write_1_bit_9_state(
    time_index_delta: TimeTableIdx,
    value: u8,
    data: &mut Vec<u8>,
) -> Option<States> {
    if let Some(bit_value) = bit_char_to_num(value) {
        let write_value = ((time_index_delta as u64) << 4) + bit_value as u64;
        leb128::write::unsigned(data, write_value).unwrap();
        let states = States::from_value(bit_value);
        Some(states)
    } else {
        None
    }
}

#[inline]
pub fn write_n_state(states: States, value: &[u8], data: &mut Vec<u8>, meta_data: Option<u8>) {
    let states_bits = states.bits();
    debug_assert!(states_bits == 1 || states_bits == 2 || states_bits == 4);
    let bits = value.len() * states_bits;
    let bit_values = value.iter().map(|b| bit_char_to_num(*b).unwrap());
    let mut working_byte = 0u8;
    let mut first_push = true;
    for (ii, value) in bit_values.enumerate() {
        let bit_id = bits - (ii * states_bits) - states_bits;
        working_byte = (working_byte << states_bits) + value;
        // Is there old data to push?
        // we use the bit_id here instead of just testing ii % bits_in_a_byte == 0
        // because for e.g. a 7-bit signal, the push needs to happen after 3 iterations!
        if bit_id % 8 == 0 {
            // this allows us to add some meta-data to the first byte.
            if let Some(meta_data) = meta_data {
                debug_assert_eq!(meta_data & (0b11 << 6), meta_data);
                if first_push {
                    first_push = false;
                    working_byte |= meta_data;
                }
            }
            data.push(working_byte);
            working_byte = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_data_encoding() {
        do_test_meta_data_round_trip(SignalEncodingMetaData::uncompressed(States::Two));
        do_test_meta_data_round_trip(SignalEncodingMetaData::uncompressed(States::Four));
        do_test_meta_data_round_trip(SignalEncodingMetaData::uncompressed(States::Nine));
        do_test_meta_data_round_trip(SignalEncodingMetaData::compressed(States::Two, 12345));
        do_test_meta_data_round_trip(SignalEncodingMetaData::compressed(States::Four, 12345));
        do_test_meta_data_round_trip(SignalEncodingMetaData::compressed(States::Nine, 12345));
    }

    fn do_test_meta_data_round_trip(data: SignalEncodingMetaData) {
        let encoded = data.encode();
        let decoded = SignalEncodingMetaData::decode(encoded);
        assert_eq!(data, decoded);
        assert_eq!(encoded, decoded.encode())
    }

    #[test]
    fn test_try_write_4_state() {
        // write all ones
        do_test_try_write_4_state(b"1111".as_slice(), Some([0b01010101].as_slice()), true);
        do_test_try_write_4_state(
            b"11111".as_slice(),
            Some([0b01, 0b01010101].as_slice()),
            true,
        );
        do_test_try_write_4_state(
            b"111111".as_slice(),
            Some([0b0101, 0b01010101].as_slice()),
            true,
        );
        do_test_try_write_4_state(
            b"111111".as_slice(),
            Some([0b0101, 0b01010101].as_slice()),
            true,
        );
        do_test_try_write_4_state(
            b"1111111".as_slice(),
            Some([0b010101, 0b01010101].as_slice()),
            true,
        );
        do_test_try_write_4_state(
            b"11111111".as_slice(),
            Some([0b01010101, 0b01010101].as_slice()),
            true,
        );
        // write some zeros, including leading zeros
        do_test_try_write_4_state(
            b"011111111".as_slice(),
            Some([0, 0b01010101, 0b01010101].as_slice()),
            true,
        );
        do_test_try_write_4_state(
            b"1011001".as_slice(),
            Some([0b010001, 0b01000001].as_slice()),
            true,
        );
        // write some X/Z
        do_test_try_write_4_state(b"xz01".as_slice(), Some([0b10110001].as_slice()), false);

        // write a long value
        do_test_try_write_4_state(
            b"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100010000001110110011".as_slice(),
           Some([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b0101, 0b01, 0, 0b0101, 0b01000101, 0b0101].as_slice()),
           true,
        );
    }

    fn do_test_try_write_4_state(value: &[u8], expected: Option<&[u8]>, is_two_state: bool) {
        let mut out = vec![5u8, 7u8];
        let out_starting_len = out.len();
        let identified_state = check_states(value).unwrap();
        if is_two_state {
            assert_eq!(identified_state, States::Two);
        }
        write_n_state(States::Four, value, &mut out, None);
        match expected {
            None => {}
            Some(expect) => {
                assert_eq!(&out[out_starting_len..], expect);
            }
        }
    }

    use proptest::prelude::*;

    fn convert_to_bits(states: States, chars: &str) -> Vec<u8> {
        let mut out = Vec::new();
        write_n_state(states, chars.as_bytes(), &mut out, None);
        out
    }

    fn do_test_compress(value: String, max_states: States) {
        let min_states = check_states(value.as_bytes()).unwrap();
        let bits = value.len();
        // convert string to bit vector
        let max_value = convert_to_bits(max_states, &value);
        // compress
        let mut out = Vec::new();
        compress(&max_value, max_states, min_states, bits, &mut out);
        // check
        let direct_conversion = convert_to_bits(min_states, &value);
        assert_eq!(direct_conversion, out, "{value} - write_n_states -> {max_value:?} - compress -> {out:?} != {direct_conversion:?}");
    }

    proptest! {
        #[test]
        fn compress_from_nine_state(value in "[01xz]{0,127}") {
            do_test_compress(value, States::Nine);
        }

        #[test]
        fn compress_from_four_state(value in "[01]{0,127}") {
            do_test_compress(value, States::Four);
        }
    }
}
