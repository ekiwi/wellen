// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Fast and compact wave-form representation inspired by the FST on disk format.

use crate::dense::DenseHashMap;
use crate::hierarchy::{Hierarchy, SignalIdx, SignalLength};
use crate::signals::{Signal, SignalSource};
use crate::values::Time;
use crate::vcd::int_div_ceil;
use bytesize::ByteSize;
use std::str::FromStr;

/// Holds queryable waveform data. Use the `Encoder` to generate.
pub struct Reader {
    blocks: Vec<Block>,
}

impl SignalSource for Reader {
    fn load_signals(&mut self, ids: &[SignalIdx]) -> Vec<Signal> {
        todo!()
    }

    fn get_time_table(&self) -> Vec<Time> {
        todo!()
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
}

/// A block that contains all value changes in a certain time segment.
/// Note that while in FST blocks can be skipped, here we only use blocks
/// in order to combine data from different threads and to compress partial data.
struct Block {
    start_time: Time,
    time_table: Vec<Time>,
    /// Offsets of (potentially compressed) signal data.
    offsets: Vec<SignalDataOffset>,
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
}

/// 31-bit byte offset + info about compression
struct SignalDataOffset(u32);
impl SignalDataOffset {
    fn new(index: usize, is_compressed: bool) -> Self {
        let data = (index << 1) as u32 + is_compressed as u32;
        SignalDataOffset(data)
    }
    fn is_compressed(&self) -> bool {
        self.0 & 1 == 1
    }
    fn get_index(&self) -> usize {
        (self.0 >> 1) as usize
    }
}

/// Encodes value and time changes into a compressed in-memory representation.
pub struct Encoder {
    /// Time table under construction
    time_table: Vec<Time>,
    /// Signals under construction
    signals: Vec<SignalEncoder>,
    /// Finished blocks
    blocks: Vec<Block>,
}

/// Indexes the time table inside a block.
type BlockTimeIdx = u16;

impl Encoder {
    pub fn new(hierarchy: &Hierarchy) -> Self {
        let mut signals = Vec::with_capacity(hierarchy.num_unique_signals());
        for var in hierarchy.get_unique_signals_vars() {
            let len = match var {
                None => SignalLength::Variable, // we do not know!
                Some(var) => var.length(),
            };
            signals.push(SignalEncoder::new(len));
        }

        Encoder {
            time_table: Vec::default(),
            signals,
            blocks: Vec::default(),
        }
    }

    pub fn time_change(&mut self, time: u64) {
        // if we run out of time indices => start a new block
        if self.time_table.len() >= BlockTimeIdx::MAX as usize {
            self.finish_block();
        }
        // sanity check to make sure that time is increasing
        if let Some(prev_time) = self.time_table.last() {
            assert!(*prev_time < time, "Time can only increase!");
        }
        self.time_table.push(time);
    }

    /// Call with an unaltered VCD value.
    pub fn vcd_value_change(&mut self, id: u64, value: &[u8]) {
        assert!(
            !self.time_table.is_empty(),
            "We need a call to time_change first!"
        );
        let time_idx = (self.time_table.len() - 1) as u16;
        self.signals[id as usize].add_vcd_change(time_idx, value);
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
        let signal_count = self.signals.len();
        let mut offsets: Vec<SignalDataOffset> = Vec::with_capacity(signal_count);
        let mut data: Vec<u8> = Vec::with_capacity(128);
        for signal in self.signals.iter_mut() {
            if let Some((mut signal_data, is_compressed)) = signal.finish() {
                offsets.push(SignalDataOffset::new(data.len(), is_compressed));
                data.append(&mut signal_data);
            } else {
                let prev_index = offsets.last().map(|o| o.get_index()).unwrap_or(0);
                offsets.push(SignalDataOffset::new(prev_index, false));
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
    }
}

/// Encodes changes for a single signal.
#[derive(Debug, Clone)]
struct SignalEncoder {
    data: Vec<u8>,
    is_four_state: bool,
    len: SignalLength,
}

impl SignalEncoder {
    fn new(len: SignalLength) -> Self {
        SignalEncoder {
            data: Vec::default(),
            is_four_state: false,
            len,
        }
    }
}

/// Minimum number of bytes for a signal to warrant an attempt at LZ4 compression.
const MIN_SIZE_TO_COMPRESS: usize = 32;
/// Flag to turn off compression.
const SKIP_COMPRESSION: bool = true;

impl SignalEncoder {
    fn add_vcd_change(&mut self, time_index: u16, value: &[u8]) {
        match self.len {
            SignalLength::Fixed(len) => {
                if len.get() == 1 {
                    let digit = if value.len() == 1 { value[0] } else { value[1] };
                    let two_state =
                        try_write_1_bit_4_state(time_index, digit, &mut self.data).unwrap();
                    self.is_four_state = self.is_four_state | !two_state;
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
                    leb128::write::unsigned(&mut self.data, time_index as u64).unwrap();
                    let bits = len.get() as usize;
                    let two_state: bool = if value_bits.len() == bits {
                        try_write_4_state(value_bits, &mut self.data).unwrap_or_else(|| {
                            panic!(
                                "Failed to parse four state value: {}",
                                String::from_utf8_lossy(value)
                            )
                        })
                    } else {
                        let expanded = expand_special_vector_cases(value_bits, bits)
                            .unwrap_or_else(|| {
                                panic!(
                                    "Failed to parse four state value: {} for signal of size {}",
                                    String::from_utf8_lossy(value),
                                    bits
                                )
                            });
                        try_write_4_state(&expanded, &mut self.data).unwrap_or_else(|| {
                            panic!(
                                "Failed to parse four state value: {}",
                                String::from_utf8_lossy(value)
                            )
                        })
                    };

                    self.is_four_state = self.is_four_state | !two_state;
                }
            }
            SignalLength::Variable => {
                assert!(
                    matches!(value[0], b's' | b'S'),
                    "expected a string, not {}",
                    String::from_utf8_lossy(value)
                );
                // string: var-length time index + var-len length + content
                leb128::write::unsigned(&mut self.data, time_index as u64).unwrap();
                leb128::write::unsigned(&mut self.data, value.len() as u64).unwrap();
                self.data.extend_from_slice(value);
            }
        }
    }

    /// returns a compressed signal representation
    fn finish(&mut self) -> Option<(Vec<u8>, bool)> {
        // no updates
        if self.data.is_empty() {
            return None;
        }
        // replace data, the actual meta data stays the same
        let data = std::mem::replace(&mut self.data, Vec::new());

        // is there so little data that compression does not make sense?
        if data.len() < MIN_SIZE_TO_COMPRESS || SKIP_COMPRESSION {
            return Some((data, false));
        }
        // attempt a compression
        let compressed = lz4_flex::compress(&data);
        if compressed.len() >= data.len() {
            Some((data, false))
        } else {
            Some((compressed, true))
        }
    }
}

#[inline]
fn expand_special_vector_cases(value: &[u8], len: usize) -> Option<Vec<u8>> {
    // if the value is actually longer than expected, there is nothing we can do
    if value.len() >= len {
        return None;
    }

    // sometimes an all z or all x vector is written with only a single digit
    if value.len() == 1 && matches!(value[0], b'x' | b'X' | b'z' | b'Z') {
        let mut repeated = Vec::with_capacity(len);
        repeated.resize(len, value[0]);
        return Some(repeated);
    }

    // check if we might want to zero extend the value
    if matches!(value[0], b'1' | b'0') {
        let mut zero_extended = Vec::with_capacity(len);
        zero_extended.resize(len - value.len(), b'0');
        zero_extended.extend_from_slice(value);
        return Some(zero_extended);
    }

    None // failed
}

#[inline]
fn two_state_to_num(value: u8) -> Option<u8> {
    match value {
        b'0' | b'1' => Some(value - b'0'),
        _ => None,
    }
}

#[inline]
fn try_write_1_bit_2_state(time_index: u16, value: u8, data: &mut Vec<u8>) -> Option<()> {
    if let Some(bit_value) = two_state_to_num(value) {
        let write_value = ((time_index as u64) << 1) + bit_value as u64;
        leb128::write::unsigned(data, write_value).unwrap();
        Some(())
    } else {
        None
    }
}

#[inline]
fn try_write_2_state(value: &[u8], data: &mut Vec<u8>) -> Option<()> {
    let bits = value.len();
    let bit_values = value.iter().map(|b| two_state_to_num(*b));
    let mut working_byte = 0u8;
    for (ii, digit_option) in bit_values.enumerate() {
        let bit_id = bits - ii - 1;
        if let Some(value) = digit_option {
            // is there old data to push?
            if bit_id == 7 && ii > 0 {
                data.push(working_byte);
                working_byte = 0;
            }
            working_byte = (working_byte << 1) + value;
        } else {
            // remove added data
            let total_bytes = int_div_ceil(bits, 8);
            let bytes_pushed = total_bytes - 1 - int_div_ceil(bit_id, 8);
            for _ in 0..bytes_pushed {
                data.pop();
            }
            return None;
        }
    }
    data.push(working_byte);
    Some(())
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
    let mut all_two_state = true;
    for (ii, digit_option) in bit_values.enumerate() {
        let bit_id = bits - (ii * 2) - 1;
        if let Some(value) = digit_option {
            // is there old data to push?
            if bit_id == 6 && ii > 0 {
                data.push(working_byte);
                working_byte = 0;
            }
            working_byte = (working_byte << 2) + value;
            all_two_state = all_two_state && value <= 1;
        } else {
            // remove added data
            let total_bytes = int_div_ceil(bits, 8);
            let bytes_pushed = total_bytes - 1 - (bit_id / 8);
            for _ in 0..bytes_pushed {
                data.pop();
            }
            return None;
        }
    }
    data.push(working_byte);
    Some(all_two_state)
}

#[inline]
fn try_decode_two_state(value: &[u8]) {}
