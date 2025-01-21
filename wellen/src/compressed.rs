// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

//! Compression for individual signals.
//! Makes the `[crate::wavemem]` signal compression available for individual signals.

use crate::wavemem::{
    compress_signal, load_compressed_signal, SignalEncodingMetaData, SignalMetaData,
};
use crate::{Signal, SignalEncoding, SignalRef, Time};
use std::borrow::Cow;

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

impl CompressedSignal {
    pub fn compress(signal: &Signal) -> Self {
        if let Some((data, meta)) = compress_signal(signal) {
            CompressedSignal {
                idx: signal.signal_ref(),
                tpe: signal.signal_encoding(),
                data,
                encoding: meta,
            }
        } else {
            // empty
            CompressedSignal {
                idx: signal.signal_ref(),
                tpe: signal.signal_encoding(),
                data: vec![],
                encoding: SignalEncodingMetaData {
                    compression: Compression::None,
                    max_states: Default::default(),
                },
            }
        }
    }

    pub fn uncompress(&self) -> Signal {
        let block = (0, self.data.as_ref(), self.encoding.clone());
        let meta = SignalMetaData {
            max_states: self.encoding.max_states,
            blocks: vec![block],
        };
        load_compressed_signal(meta, self.idx, self.tpe)
    }
}

/// Compressed form of the timetable. Uses delta compression with variable length integers and gzip.
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct CompressedTimeTable {
    data: Vec<u8>,
    zlib_compressed: bool,
}

/// by unscientific experiment, we observed that this level might be good enough :)
const ZLIB_LEVEL: u8 = 3;

impl CompressedTimeTable {
    pub fn compress(table: &[Time]) -> Self {
        let mut prev = 0;
        let mut tmp = vec![];
        leb128::write::unsigned(&mut tmp, table.len() as u64).unwrap();
        for &time in table.iter() {
            debug_assert!(time >= prev);
            let delta = time - prev;
            prev = time;
            leb128::write::unsigned(&mut tmp, delta).unwrap();
        }

        let mut compressed = miniz_oxide::deflate::compress_to_vec_zlib(&tmp, ZLIB_LEVEL);
        // is compression worth it?
        if compressed.len() + 4 > tmp.len() {
            // it is more space efficient to stick with the uncompressed version
            Self {
                data: tmp,
                zlib_compressed: false,
            }
        } else {
            // add uncompressed length
            let len = (tmp.len() as u32).to_be_bytes();
            compressed.extend_from_slice(len.as_slice());
            Self {
                data: compressed,
                zlib_compressed: true,
            }
        }
    }
    pub fn uncompress(&self) -> Vec<Time> {
        let data = if self.zlib_compressed {
            let len_bytes = &self.data[self.data.len() - 4..];
            let len = u32::from_be_bytes(len_bytes.try_into().unwrap());
            let uncompressed = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(
                &self.data[..self.data.len() - 4],
                len as usize,
            )
            .unwrap();
            Cow::Owned(uncompressed)
        } else {
            Cow::Borrowed(self.data.as_slice())
        };

        let mut reader = std::io::Cursor::new(data);
        let num_entries = leb128::read::unsigned(&mut reader).unwrap();
        let mut out = Vec::with_capacity(num_entries as usize);
        let mut prev = 0;
        for _ in 0..num_entries {
            prev += leb128::read::unsigned(&mut reader).unwrap();
            out.push(prev);
        }
        out
    }
}
