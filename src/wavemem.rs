// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Fast and compact wave-form representation inspired by the FST on disk format.

use crate::dense::DenseHashMap;
use crate::hierarchy::SignalIdx;
use crate::signals::{Signal, SignalSource};
use crate::values::Time;
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
        let time = self.time_table.capacity() * std::mem::size_of::<Time>();
        let offsets = self.offsets.capacity() * std::mem::size_of::<SignalDataOffset>();
        let data = self.data.capacity() * std::mem::size_of::<u8>();
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
    signals: DenseHashMap<SignalEncoder>,
    /// Finished blocks
    blocks: Vec<Block>,
}

impl Default for Encoder {
    fn default() -> Self {
        Encoder {
            time_table: Vec::default(),
            signals: DenseHashMap::default(),
            blocks: Vec::default(),
        }
    }
}

/// Indexes the time table inside a block.
type BlockTimeIdx = u16;

impl Encoder {
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
        self.signals
            .get_or_else_create_mut(id as usize)
            .add_vcd_change(time_idx, value);
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
        let mut signals =
            std::mem::replace(&mut self.signals, DenseHashMap::with_capacity(signal_count));
        for signal in signals.into_vec().into_iter() {
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
    tpe: SignalType,
}

#[derive(Debug, Clone)]
enum SignalType {
    Unknown,
    /// this is how we start of
    Real,
    /// encoded in VCD with `r`
    String,
    /// encoded in VCD with `s`
    Bits(u32),
    /// binary number of size N
    FourState(u32),
    // TODO: 9-state for VHDL
}

impl Default for SignalEncoder {
    fn default() -> Self {
        SignalEncoder {
            data: Vec::default(),
            tpe: SignalType::Unknown,
        }
    }
}

/// Minimum number of bytes for a signal to warrant an attempt at LZ4 compression.
const MIN_SIZE_TO_COMPRESS: usize = 32;
/// Flag to turn off compression.
const SKIP_COMPRESSION: bool = true;

impl SignalEncoder {
    fn add_vcd_change(&mut self, time_index: u16, value: &[u8]) {
        // save time index: (TODO: var length)
        self.data.extend_from_slice(&u16::to_be_bytes(time_index));
        // save data depending on signal type
        let (new_tpe, decoded) = decode_vcd_signal(value, &self.tpe);
        self.tpe = new_tpe;
        self.data.extend_from_slice(decoded);
    }

    /// returns a compressed signal representation
    fn finish(self) -> Option<(Vec<u8>, bool)> {
        // no updates
        if self.data.is_empty() {
            return None;
        }
        // is there so little data that compression does not make sense?
        if self.data.len() < MIN_SIZE_TO_COMPRESS || SKIP_COMPRESSION {
            return Some((self.data, false));
        }
        // attempt a compression
        let compressed = lz4_flex::compress(&self.data);
        if compressed.len() >= self.data.len() {
            Some((self.data, false))
        } else {
            Some((compressed, true))
        }
    }
}

#[inline]
fn decode_vcd_signal<'a>(value: &'a [u8], expected_tpe: &SignalType) -> (SignalType, &'a [u8]) {
    match expected_tpe {
        SignalType::Unknown => {
            todo!()
        }
        SignalType::Real => {
            let b0 = value[0];
            let is_real = b0 == b'r' || b0 == b'R';
            assert!(is_real, "Expected a real value!");
            let value = f64::from_str(&String::from_utf8_lossy(value));
            todo!()
        }
        SignalType::String => {
            todo!()
        }
        SignalType::Bits(_) => {
            todo!()
        }
        SignalType::FourState(_) => {
            todo!()
        }
    }
}
