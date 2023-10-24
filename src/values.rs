// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Space efficient but also fast format for keeping value changes in memory or on disk.

use crate::dense::DenseHashMap;
use crate::hierarchy::{SignalIdx, SignalLength};
use bytesize::ByteSize;
use std::io::Write;
use std::num::NonZeroU32;

pub struct Values {}

#[derive(Debug, Clone, Copy)]
struct SignalInfo {
    length: SignalLength,
    prev_value_change: ValueChangePos,
}

/// Space efficient list of value changes. Inspired by FST's value change preprocessing buffer.
#[derive(Debug, Default)]
struct ValueChangeList {
    data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ValueChangePos(u32);

#[inline]
pub(crate) fn write_u16(output: &mut impl Write, value: u16) -> std::io::Result<()> {
    let buf = value.to_be_bytes();
    output.write_all(&buf)?;
    Ok(())
}

impl ValueChangeList {
    fn add(&mut self, prev: ValueChangePos, time_index: u16, value: &[u8]) -> ValueChangePos {
        let new_pos = ValueChangePos(self.data.len() as u32);
        leb128::write::unsigned(&mut self.data, prev.0 as u64).unwrap();
        write_u16(&mut self.data, time_index).unwrap();
        self.data.extend_from_slice(value);
        new_pos
    }
    fn clear(&mut self) {
        self.data.clear();
    }
}

pub struct ValueBuilder<W: Write> {
    output: W,
    signals: DenseHashMap<Option<SignalInfo>>,
    /// Similar to the Time Chain used in the FST format.
    time_table: Vec<Time>,
    current_time: Time,
    /// A record of all value changes that have not been written to disk yet.
    value_changes: ValueChangeList,
}

impl Default for ValueBuilder<std::io::Cursor<Vec<u8>>> {
    fn default() -> Self {
        let storage: Vec<u8> = Vec::new();
        ValueBuilder {
            output: std::io::Cursor::new(storage),
            signals: DenseHashMap::default(),
            time_table: vec![0; 1],
            current_time: 0,
            value_changes: ValueChangeList::default(),
        }
    }
}

pub type Time = u64;

impl<W: Write> ValueBuilder<W> {
    pub fn add_signal(&mut self, handle: SignalIdx, length: SignalLength) {
        let signal = SignalInfo {
            length,
            prev_value_change: ValueChangePos(0),
        };
        self.signals.insert(handle as usize, Some(signal));
    }
    pub fn value_change(&mut self, handle: SignalIdx, value: &[u8]) {
        if let Some(info) = self.signals.get_mut(handle as usize).unwrap() {
            let vc =
                self.value_changes
                    .add(info.prev_value_change, self.time_table.len() as u16, value);
            info.prev_value_change = vc;
        }
    }
    pub fn time_change(&mut self, time: Time) {
        assert!(time >= self.current_time);
        if time > self.current_time {
            self.current_time = time;
            self.time_table.push(time);
        }
    }
    pub fn value_and_time(&mut self, handle: SignalIdx, time: Time, value: &[u8]) {
        assert!(time >= self.current_time);
        if time > self.current_time {
            self.time_change(time);
        }
        self.value_change(handle, value);
    }
    pub fn finish(self) -> Values {
        Values {}
    }
    pub fn print_statistics(&self) {
        println!(
            "Time table size: {}",
            ByteSize::b((self.time_table.len() * std::mem::size_of::<Time>()) as u64)
        );
        println!(
            "Value change size: {}",
            ByteSize::b((self.value_changes.data.len()) as u64)
        );
    }
}
