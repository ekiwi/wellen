// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

//! Compression for individual signals.

use crate::{Signal, SignalRef, SignalValue, TimeTableIdx};

/// A compressed version of a Signal. Uses a compression scheme very similar to what [crate::wavemem] uses.
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressedSignal {
    idx: SignalRef,
    /// variable length encoded signal data
    data: Vec<u8>,
    /// additional compression performed on the data
    compression: Compression,
}

#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Compression {
    None,
    Lz4(usize),
}

impl CompressedSignal {
    fn new(idx: SignalRef) -> Self {
        Self {
            idx,
            data: vec![],
            compression: Compression::None,
        }
    }

    fn add_value_change(&mut self, time: TimeTableIdx, change: SignalValue) {
        debug_assert_eq!(
            self.compression,
            Compression::None,
            "signal is already compressed!"
        );
        todo!()
    }

    fn compress_lz4(&mut self) {
        assert_eq!(
            self.compression,
            Compression::None,
            "signal is already compressed"
        );
        todo!()
    }
}

impl From<&Signal> for CompressedSignal {
    fn from(value: &Signal) -> Self {
        let mut out = CompressedSignal::new(value.idx());
        for (time, change) in value.iter_changes() {
            out.add_value_change(time, change);
        }
        out.compress_lz4();
        out
    }
}

impl From<&CompressedSignal> for Signal {
    fn from(value: &CompressedSignal) -> Self {
        todo!()
    }
}
