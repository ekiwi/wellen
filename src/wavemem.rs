// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Fast and compact wave-form representation inspired by the FST on disk format.

use crate::hierarchy::{Hierarchy, SignalRef, SignalType};
use crate::signals::{Signal, SignalEncoding, SignalSource, Time};
use crate::vcd::{u32_div_ceil, usize_div_ceil};
use bytesize::ByteSize;
use std::borrow::Cow;
use std::io::Read;
use std::num::NonZeroU32;

/// Holds queryable waveform data. Use the `Encoder` to generate.
pub struct Reader {
    blocks: Vec<Block>,
}

impl SignalSource for Reader {
    fn load_signals(&mut self, ids: &[(SignalRef, SignalType)]) -> Vec<Signal> {
        let mut signals = Vec::with_capacity(ids.len());
        for (id, len) in ids.iter() {
            let sig = self.load_signal(*id, *len);
            signals.push(sig);
        }
        signals
    }

    fn get_time_table(&self) -> Vec<Time> {
        // create a combined time table from all blocks
        let len = self
            .blocks
            .iter()
            .map(|b| b.time_table.len())
            .sum::<usize>();
        let mut table = Vec::with_capacity(len);
        for block in self.blocks.iter() {
            table.extend_from_slice(&block.time_table);
        }
        table
    }

    fn print_statistics(&self) {
        println!(
            "[wavemem] size in memory: {}",
            ByteSize::b(self.size_in_memory() as u64)
        );
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
            "[wavemem] data across all blocks takes up {}.",
            ByteSize::b(total_data_size as u64)
        );
        println!(
            "[wavemem] offsets across all blocks take up {}.",
            ByteSize::b(total_offset_size as u64)
        );
        println!(
            "[wavemem] time table data across all blocks takes up {}.",
            ByteSize::b(total_time_table_size as u64)
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
        let is_two_state = blocks
            .iter()
            .map(|b| b.2.is_two_state)
            .reduce(|a, b| a && b)
            .unwrap_or(false);
        SignalMetaData {
            is_two_state,
            blocks,
        }
    }

    fn load_signal(&self, id: SignalRef, tpe: SignalType) -> Signal {
        let meta = self.collect_signal_meta_data(id);
        let mut time_indices: Vec<u32> = Vec::new();
        let mut data_bytes: Vec<u8> = Vec::new();
        let mut strings: Vec<String> = Vec::new();
        let mut reals: Vec<f32> = Vec::new();
        for (time_idx_offset, data_block, meta_data) in meta.blocks.into_iter() {
            let data = match meta_data.compression {
                SignalCompression::Compressed(uncompressed_len) => {
                    let data = lz4_flex::decompress(data_block, uncompressed_len).unwrap();
                    Cow::Owned(data)
                }
                SignalCompression::Uncompressed => Cow::Borrowed(data_block),
            };

            match tpe {
                SignalType::String => {
                    let (mut new_strings, mut new_time_indices) =
                        load_signal_strings(&mut data.as_ref(), time_idx_offset);
                    time_indices.append(&mut new_time_indices);
                    strings.append(&mut new_strings);
                }
                SignalType::BitVector(signal_len, _) => {
                    let (mut new_data, mut new_time_indices) = load_fixed_len_signal(
                        &mut data.as_ref(),
                        time_idx_offset,
                        signal_len.get(),
                        meta.is_two_state,
                    );
                    time_indices.append(&mut new_time_indices);
                    data_bytes.append(&mut new_data);
                }
                SignalType::Real => {
                    todo!("reals");
                }
            }
        }

        match tpe {
            SignalType::String => {
                assert!(data_bytes.is_empty());
                Signal::new_var_len(id, time_indices, strings)
            }
            SignalType::BitVector(len, _) => {
                assert!(strings.is_empty());
                let (encoding, bytes_per_entry) = if meta.is_two_state {
                    (
                        SignalEncoding::Binary(len.get()),
                        usize_div_ceil(len.get() as usize, 8) as u32,
                    )
                } else {
                    (
                        SignalEncoding::FourValue(len.get()),
                        usize_div_ceil(len.get() as usize, 4) as u32,
                    )
                };
                Signal::new_fixed_len(id, time_indices, encoding, bytes_per_entry, data_bytes)
            }
            SignalType::Real => {
                todo!("Implement real!")
            }
        }
    }
}

/// Data about a single signal inside a Reader.
/// Only used internally by `collect_signal_meta_data`
struct SignalMetaData<'a> {
    is_two_state: bool,
    /// For every block that contains the signal: time_idx_offset, data and meta-data
    blocks: Vec<(u32, &'a [u8], SignalEncodingMetaData)>,
}

#[inline]
fn load_fixed_len_signal(
    data: &mut impl Read,
    time_idx_offset: u32,
    signal_len: u32,
    is_two_state: bool,
) -> (Vec<u8>, Vec<u32>) {
    let mut out = Vec::new();
    let mut time_indices = Vec::new();
    let mut last_time_idx = time_idx_offset;

    loop {
        // read time index
        let time_idx_delta_raw = match leb128::read::unsigned(data) {
            Ok(value) => value as u32,
            Err(_) => break, // presumably there is no more data to be read
        };
        // now the decoding depends on the size and whether it is two state
        let time_idx_delta = match signal_len {
            1 => {
                let value = (time_idx_delta_raw & 0x3) as u8;
                // for a 1-bit signal we do not need to distinguish between 2 and 4 state!
                out.push(value);
                // time delta is encoded together with the value
                time_idx_delta_raw >> 2
            }
            other => {
                let num_bytes = usize_div_ceil(other as usize, 4);
                let mut buf = vec![0u8; num_bytes];
                data.read_exact(&mut buf.as_mut()).unwrap();
                if is_two_state {
                    four_state_to_two_state(&mut buf);
                }
                out.append(&mut buf);
                //
                time_idx_delta_raw
            }
        };
        last_time_idx += time_idx_delta;
        time_indices.push(last_time_idx)
    }

    (out, time_indices)
}

#[inline]
fn four_state_to_two_state(buf: &mut Vec<u8>) {
    let result_len = usize_div_ceil(buf.len(), 2);
    let is_uneven = result_len * 2 > buf.len();
    for ii in 0..result_len {
        let (msb, lsb) = if is_uneven {
            if ii == 0 {
                (0u8, buf[0])
            } else {
                (buf[ii * 2 - 1], buf[ii * 2])
            }
        } else {
            (buf[ii * 2], buf[ii * 2 + 1])
        };
        buf[ii] = compress_four_state_to_two_state(msb, lsb);
    }
    buf.truncate(result_len);
}

#[inline]
fn compress_four_state_to_two_state(msb: u8, lsb: u8) -> u8 {
    (msb & (1 << 6)) << 1 | // msb[6] -> out[7]
    (msb & (1 << 4)) << 2 | // msb[4] -> out[6]
    (msb & (1 << 2)) << 3 | // msb[2] -> out[5]
    (msb & (1 << 0)) << 4 | // msb[0] -> out[4]
    (lsb & (1 << 6)) >> 3 | // lsb[6] -> out[3]
    (lsb & (1 << 4)) >> 2 | // lsb[4] -> out[2]
    (lsb & (1 << 2)) >> 1 | // lsb[2] -> out[1]
    (lsb & (1 << 0)) >> 0 // lsb[0] -> out[0]
}

#[inline]
fn load_signal_strings(data: &mut impl Read, time_idx_offset: u32) -> (Vec<String>, Vec<u32>) {
    let mut out = Vec::new();
    let mut time_indices = Vec::new();
    let mut last_time_idx = time_idx_offset;

    loop {
        // read time index
        let time_idx_delta = match leb128::read::unsigned(data) {
            Ok(value) => value as u32,
            Err(_) => break, // presumably there is no more data to be read
        };
        last_time_idx += time_idx_delta;
        time_indices.push(last_time_idx);
        // read variable length string
        let len = leb128::read::unsigned(data).unwrap() as usize;
        let mut buf = vec![0u8; len];
        data.read_exact(&mut buf).unwrap();
        let str_value = String::from_utf8_lossy(&buf).to_string();
        out.push(str_value);
    }
    (out, time_indices)
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
                None => SignalType::String, // we do not know!
                Some(var) => var.signal_tpe(),
            };
            let pos = signals.len();
            signals.push(SignalEncoder::new(tpe, pos));
        }

        Encoder {
            time_table: Vec::default(),
            signals,
            has_new_data: false,
            blocks: Vec::default(),
        }
    }

    pub fn time_change(&mut self, time: u64) {
        // sanity check to make sure that time is increasing
        if let Some(prev_time) = self.time_table.last() {
            if *prev_time == time {
                return; // ignore calls to time_change that do not actually change anything
            }
            assert!(*prev_time < time, "Time can only increase!");
        }
        // if we run out of time indices => start a new block
        if self.time_table.len() >= BlockTimeIdx::MAX as usize {
            self.finish_block();
        }
        self.time_table.push(time);
        self.has_new_data = true;
    }

    /// Call with an unaltered VCD value.
    pub fn vcd_value_change(&mut self, id: u64, value: &[u8]) {
        assert!(
            !self.time_table.is_empty(),
            "We need a call to time_change first!"
        );
        if id == 32 {
            println!();
        }
        let time_idx = (self.time_table.len() - 1) as u16;
        self.signals[id as usize].add_vcd_change(time_idx, value);
        self.has_new_data = true;
    }

    pub fn finish(mut self) -> Reader {
        // ensure that we have no open blocks
        self.finish_block();
        // create a new reader with the blocks that we have
        Reader {
            blocks: self.blocks,
        }
    }

    // appends the contents of the other encoder to this one
    pub fn append(&mut self, mut other: Encoder) {
        // ensure that we have no open blocks
        self.finish_block();
        // ensure that the other encoder is also done
        other.finish_block();
        // make sure the timeline fits
        let us_end_time = self.blocks.last().unwrap().end_time();
        let other_start = other.blocks.first().unwrap().start_time;
        assert!(
            us_end_time <= other_start,
            "Can only append encoders in chronological order!"
        );
        // append all blocks from the other encoder
        self.blocks.append(&mut other.blocks);
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
        let time_table = std::mem::replace(&mut self.time_table, new_time_table);
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
enum SignalCompression {
    /// signal is compressed and the output is at max `inner` bytes long
    Compressed(usize),
    Uncompressed,
}

#[derive(Debug, Clone, PartialEq)]
struct SignalEncodingMetaData {
    compression: SignalCompression,
    is_two_state: bool,
}

/// We divide the decompressed size by this number and round up.
/// This is OK, since it will just make us allocate a slightly too larger buffer.
const SIGNAL_DECOMPRESSED_LEN_DIV: u32 = 32;

impl SignalEncodingMetaData {
    fn uncompressed(is_two_state: bool) -> Self {
        SignalEncodingMetaData {
            compression: SignalCompression::Uncompressed,
            is_two_state,
        }
    }

    fn compressed(is_two_state: bool, uncompressed_len: usize) -> Self {
        // turn the length into a value that we can actually encode
        let uncompressed_len_approx =
            u32_div_ceil(uncompressed_len as u32, SIGNAL_DECOMPRESSED_LEN_DIV)
                * SIGNAL_DECOMPRESSED_LEN_DIV;
        SignalEncodingMetaData {
            compression: SignalCompression::Compressed(uncompressed_len_approx as usize),
            is_two_state,
        }
    }

    fn decode(data: u64) -> Self {
        let is_two_state = data & 1 == 1;
        let is_compressed = (data >> 1) & 1 == 1;
        let compression = if is_compressed {
            let decompressed_len_bits = ((data >> 2) & u32::MAX as u64) as u32;
            let decompressed_len = decompressed_len_bits * SIGNAL_DECOMPRESSED_LEN_DIV;
            SignalCompression::Compressed(decompressed_len as usize)
        } else {
            SignalCompression::Uncompressed
        };
        SignalEncodingMetaData {
            compression,
            is_two_state,
        }
    }
    fn encode(&self) -> u64 {
        match &self.compression {
            SignalCompression::Compressed(decompressed_len) => {
                let decompressed_len_bits =
                    u32_div_ceil((*decompressed_len) as u32, SIGNAL_DECOMPRESSED_LEN_DIV);
                let data =
                    ((decompressed_len_bits as u64) << 2) | (1 << 1) | (self.is_two_state as u64);
                data
            }
            SignalCompression::Uncompressed => self.is_two_state as u64,
        }
    }
}

/// Encodes changes for a single signal.
#[derive(Debug, Clone)]
struct SignalEncoder {
    data: Vec<u8>,
    tpe: SignalType,
    prev_time_idx: u16,
    is_two_state: bool,
    /// Same as the index of this encoder in a Vec<_>. Used for debugging purposes.
    signal_idx: u32,
}

impl SignalEncoder {
    fn new(tpe: SignalType, pos: usize) -> Self {
        SignalEncoder {
            data: Vec::default(),
            tpe,
            prev_time_idx: 0,
            is_two_state: true,
            signal_idx: pos as u32,
        }
    }
}

/// Minimum number of bytes for a signal to warrant an attempt at LZ4 compression.
const MIN_SIZE_TO_COMPRESS: usize = 32;
/// Flag to turn off compression.
const SKIP_COMPRESSION: bool = false;

impl SignalEncoder {
    fn add_vcd_change(&mut self, time_index: u16, value: &[u8]) {
        let time_idx_delta = time_index - self.prev_time_idx;
        match self.tpe {
            SignalType::BitVector(len, _) => {
                if len.get() == 1 {
                    let digit = if value.len() == 1 { value[0] } else { value[1] };
                    self.is_two_state &=
                        try_write_1_bit_4_state(time_idx_delta, digit, &mut self.data)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Failed to parse four state value: {} for signal of size 1",
                                    String::from_utf8_lossy(value)
                                )
                            });
                } else {
                    let value_bits: &[u8] = match value[0] {
                        b'b' | b'B' => &value[1..],
                        b'1' | b'0' => value,
                        _ => panic!(
                            "expected a bit vector, not {} for signal of size {}",
                            String::from_utf8_lossy(value),
                            len.get()
                        ),
                    };
                    // write time delta
                    leb128::write::unsigned(&mut self.data, time_idx_delta as u64).unwrap();
                    // write actual data
                    let bits = len.get() as usize;
                    if value_bits.len() == bits {
                        self.is_two_state &= try_write_4_state(value_bits, &mut self.data)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Failed to parse four state value: {}",
                                    String::from_utf8_lossy(value)
                                )
                            });
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
                        self.is_two_state &= try_write_4_state(&expanded, &mut self.data)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Failed to parse four state value: {}",
                                    String::from_utf8_lossy(value)
                                )
                            });
                    };
                }
            }
            SignalType::String => {
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
            SignalType::Real => {
                assert!(
                    matches!(value[0], b'r' | b'R'),
                    "expected a real, not {}",
                    String::from_utf8_lossy(value)
                );
                // parse float
                let float_value = std::str::from_utf8(&value[1..])
                    .unwrap()
                    .parse::<f64>()
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
        let data = std::mem::replace(&mut self.data, Vec::new());

        // is there so little data that compression does not make sense?
        if data.len() < MIN_SIZE_TO_COMPRESS || SKIP_COMPRESSION {
            return Some((
                data,
                SignalEncodingMetaData::uncompressed(self.is_two_state),
            ));
        }
        // attempt a compression
        let compressed = lz4_flex::compress(&data);
        if (compressed.len() + 1) >= data.len() {
            Some((
                data,
                SignalEncodingMetaData::uncompressed(self.is_two_state),
            ))
        } else {
            Some((
                compressed,
                SignalEncodingMetaData::compressed(self.is_two_state, data.len()),
            ))
        }
    }
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

#[inline]
fn four_state_to_num(value: u8) -> Option<u8> {
    match value {
        b'0' | b'1' => Some(value - b'0'),
        b'x' | b'X' => Some(2),
        b'z' | b'Z' => Some(3),
        _ => None,
    }
}

#[inline]
fn try_write_1_bit_4_state(time_index: u16, value: u8, data: &mut Vec<u8>) -> Option<bool> {
    if let Some(bit_value) = four_state_to_num(value) {
        let write_value = ((time_index as u64) << 2) + bit_value as u64;
        leb128::write::unsigned(data, write_value).unwrap();
        Some(bit_value <= 1)
    } else {
        None
    }
}

#[inline]
fn try_write_4_state(value: &[u8], data: &mut Vec<u8>) -> Option<bool> {
    let bits = value.len() * 2;
    let bit_values = value.iter().map(|b| four_state_to_num(*b));
    let mut working_byte = 0u8;
    let mut is_two_state = true;
    for (ii, digit_option) in bit_values.enumerate() {
        let bit_id = bits - (ii * 2) - 2;
        if let Some(value) = digit_option {
            is_two_state &= value <= 1;
            working_byte = (working_byte << 2) + value;
            // Is there old data to push?
            // we use the bit_id here instead of just testing ii % 4 == 0
            // because for e.g. a 7-bit signal, the push needs to happen after 3 iterations!
            if bit_id % 8 == 0 {
                data.push(working_byte);
                working_byte = 0;
            }
        } else {
            // remove added data
            let total_bytes = usize_div_ceil(bits, 8);
            let bytes_pushed = total_bytes - 1 - (bit_id / 8);
            for _ in 0..bytes_pushed {
                data.pop();
            }
            return None;
        }
    }
    Some(is_two_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_data_encoding() {
        do_test_meta_data_round_trip(SignalEncodingMetaData::uncompressed(false));
        do_test_meta_data_round_trip(SignalEncodingMetaData::uncompressed(true));
        do_test_meta_data_round_trip(SignalEncodingMetaData::compressed(false, 12345));
        do_test_meta_data_round_trip(SignalEncodingMetaData::compressed(true, 12345));
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

        match (try_write_4_state(value, &mut out), expected) {
            (Some(actual_two_state), Some(expect)) => {
                assert_eq!(&out[out_starting_len..], expect);
                assert_eq!(actual_two_state, is_two_state);
            }
            (None, Some(expect)) => {
                panic!("Expected: {expect:?}, but got error");
            }
            (Some(_), None) => {
                panic!("Expected error, but got: {out:?}")
            }
            (None, None) => {
                assert_eq!(out.len(), out_starting_len);
            } // great!
        }
    }

    #[test]
    fn test_four_state_to_two_state() {
        let mut input0 = vec![0b01010001u8, 0b00010100u8, 0b00010000u8];
        let expected0 = [0b1101u8, 0b01100100u8];
        four_state_to_two_state(&mut input0);
        assert_eq!(input0, expected0);

        // example from the try_write_4_state test
        let mut input1 = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0b0101, 0b01, 0, 0b0101, 0b01000101, 0b0101,
        ];
        let expected1 = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b110001, 0b11, 0b10110011,
        ];
        four_state_to_two_state(&mut input1);
        assert_eq!(input1, expected1);
    }
}
