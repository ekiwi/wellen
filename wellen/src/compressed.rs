// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

//! Compression for individual signals.
//! Makes the `[crate::wavemem]` signal compression available for individual signals.

use crate::wavemem::{
    compress_signal, load_compressed_signal, SignalEncodingMetaData, SignalMetaData,
};
use crate::{Signal, SignalEncoding, SignalRef};

/// A compressed version of a Signal. Uses a compression scheme very similar to what [crate::wavemem] uses.
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressedSignal {
    idx: SignalRef,
    tpe: SignalEncoding,
    /// variable length encoded signal data
    data: Vec<u8>,
    /// information on how the signal is compressed
    encoding: SignalEncodingMetaData,
}

#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Compression {
    None,
    Lz4(usize),
}

impl From<&Signal> for CompressedSignal {
    fn from(signal: &Signal) -> Self {
        if let Some((data, meta)) = compress_signal(signal) {
            CompressedSignal {
                idx: signal.idx(),
                tpe: signal.signal_encoding(),
                data,
                encoding: meta,
            }
        } else {
            // empty
            CompressedSignal {
                idx: signal.idx(),
                tpe: signal.signal_encoding(),
                data: vec![],
                encoding: SignalEncodingMetaData {
                    compression: Compression::None,
                    max_states: Default::default(),
                },
            }
        }
    }
}

impl From<&CompressedSignal> for Signal {
    fn from(signal: &CompressedSignal) -> Self {
        let block = (0, signal.data.as_ref(), signal.encoding.clone());
        let meta = SignalMetaData {
            max_states: signal.encoding.max_states,
            blocks: vec![block],
        };
        load_compressed_signal(meta, signal.idx, signal.tpe)
    }
}
