// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::{SignalEncoding, SignalRef, Time};
use std::num::NonZeroU32;

/// Generates a new signal based on other signals.
/// The on_change method takes a mutable reference and thus the transform must be
/// executed sequentially over all changes in order.
pub trait SignalTransform {
    type SignalRefType;
    /// The encoding of the output signal.
    fn output_encoding(&self) -> SignalEncoding;
    /// The signals that this transform depends on.
    fn inputs(&self) -> &[SignalRef];
    /// Process a change in at least one of the input signals.
    fn on_change(&mut self, time: Time, values: &[Self::SignalRefType]) -> Self::SignalRefType;
}

/// Captures a signal which is derived from other bit-vector signals by slice and concat operations.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct DerivedBitVecSignal {
    width: u32,
    inputs: Vec<SignalRef>,
    bits: Vec<u64>,
}

impl DerivedBitVecSignal {
    /// Creates a new signal by slicing an existing one.
    pub fn new_slice(signal: SignalRef, enc: SignalEncoding, msb: u32, lsb: u32) -> Self {
        let mut out = Self {
            width: 0,
            inputs: vec![],
            bits: vec![],
        };
        out.concat_left(signal, enc, msb, lsb);
        out
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    /// Creates a new signal that is just the identity (full slice) of an existing one.
    pub fn new_identity(signal: SignalRef, enc: SignalEncoding) -> Self {
        if let SignalEncoding::BitVector(width) = enc {
            Self::new_slice(signal, enc, width.get() - 1, 0)
        } else {
            unreachable!("This function only works for bit vector signals")
        }
    }

    /// Concatenates a full signal to the left, i.e., on the most significant side.
    pub fn concat_left_full(&mut self, signal: SignalRef, enc: SignalEncoding) {
        if let SignalEncoding::BitVector(width) = enc {
            self.concat_left(signal, enc, width.get() - 1, 0)
        } else {
            unreachable!("This function only works for bit vector signals")
        }
    }

    /// Concatenates a signal to the left, i.e., on the most significant side.
    pub fn concat_left(&mut self, signal: SignalRef, enc: SignalEncoding, msb: u32, lsb: u32) {
        let mut extract = self.make_extract(signal, enc, msb, lsb);
        self.width += extract.width();
        if let Some(other_num) = self.bits.first_mut() {
            let other: Extract = (*other_num).into();
            // can the two extracts be merged?
            if other.signal == extract.signal && extract.lsb == other.msb + 1 {
                extract.lsb = other.lsb;
                *other_num = extract.into();
            } else {
                self.bits.insert(0, extract.into());
            }
        } else {
            debug_assert!(self.bits.is_empty());
            self.bits.push(extract.into());
        }
    }

    /// Concatenates a full signal to the right, i.e., on the least significant side.
    pub fn concat_right_full(&mut self, signal: SignalRef, enc: SignalEncoding) {
        if let SignalEncoding::BitVector(width) = enc {
            self.concat_right(signal, enc, width.get() - 1, 0)
        } else {
            unreachable!("This function only works for bit vector signals")
        }
    }

    /// Concatenates a signal to the right, i.e., on the least significant side.
    pub fn concat_right(&mut self, signal: SignalRef, enc: SignalEncoding, msb: u32, lsb: u32) {
        let mut extract = self.make_extract(signal, enc, msb, lsb);
        self.width += extract.width();
        if let Some(other_num) = self.bits.last_mut() {
            let other: Extract = (*other_num).into();
            // can the two extracts be merged?
            if other.signal == extract.signal && extract.msb + 1 == other.lsb {
                extract.msb = other.msb;
                *other_num = extract.into();
            } else {
                self.bits.push(extract.into());
            }
        } else {
            debug_assert!(self.bits.is_empty());
            self.bits.push(extract.into());
        }
    }

    /// Converts a signal description into an [[Extract]] op.
    fn make_extract(
        &mut self,
        signal: SignalRef,
        enc: SignalEncoding,
        msb: u32,
        lsb: u32,
    ) -> Extract {
        debug_assert!(msb >= lsb);
        if let SignalEncoding::BitVector(len) = enc {
            debug_assert!(msb < len.get());
            // check to see if we already have this signal as an input
            let signal = if let Some(ii) = self.inputs.iter().position(|i| *i == signal) {
                ii as u16
            } else {
                let ii = self.inputs.len() as u16;
                self.inputs.push(signal);
                ii
            };
            Extract { signal, msb, lsb }
        } else {
            unreachable!("Can only derive from a bit-vector signal!");
        }
    }
}

impl SignalTransform for DerivedBitVecSignal {
    type SignalRefType = ();

    fn output_encoding(&self) -> SignalEncoding {
        SignalEncoding::BitVector(NonZeroU32::new(self.width).unwrap())
    }

    fn inputs(&self) -> &[SignalRef] {
        &self.inputs
    }

    fn on_change(&mut self, _time: Time, _values: &[Self::SignalRefType]) -> Self::SignalRefType {
        todo!()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Extract {
    signal: u16,
    msb: u32,
    lsb: u32,
}

impl Extract {
    fn width(&self) -> u32 {
        self.msb - self.lsb + 1
    }
}

impl From<u64> for Extract {
    fn from(value: u64) -> Self {
        let signal = (value >> (64 - 16)) as u16;
        let lsb = (value & (u32::MAX as u64)) as u32;
        let width = ((value >> 32) & (u16::MAX as u64)) as u16;
        let msb = (width as u32) - 1 + lsb;
        Self { signal, msb, lsb }
    }
}

impl From<Extract> for u64 {
    fn from(value: Extract) -> Self {
        let width = (value.msb - value.lsb + 1) as u16;
        ((value.signal as u64) << (64 - 16)) | ((width as u64) << 32) | (value.lsb as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn do_test_extract_conversion(signal: u16, width: u16, lsb: u32) {
        let msb = lsb + width as u32 - 1;
        assert_eq!(width as u32, msb - lsb + 1);
        let extract = Extract { signal, msb, lsb };
        let as_num: u64 = extract.into();
        let and_back: Extract = as_num.into();
        assert_eq!(extract, and_back);
    }

    proptest! {
        #[test]
        fn test_extract_conversion(signal: u16, width: u16, lsb: u32) {
            do_test_extract_conversion(signal, width, lsb);
        }
    }
}
