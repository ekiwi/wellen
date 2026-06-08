// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::wavemem::write_n_state_from_ascii;
use num_enum::TryFromPrimitive;
use smallvec::{SmallVec, ToSmallVec, smallvec};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

pub type Real = f64;

#[derive(Debug, Clone, Copy)]
pub enum SignalValueRef<'a> {
    Event,
    BitVec(BitVecRef<'a>),
    String(&'a str),
    Real(Real),
}

impl<'a> SignalValueRef<'a> {
    pub fn bit_vec(states: States, width: u32, data: &'a [u8]) -> Self {
        Self::BitVec(BitVecRef::new(states, width, data))
    }
}

impl<'a> From<BitVecRef<'a>> for SignalValueRef<'a> {
    fn from(value: BitVecRef<'a>) -> Self {
        Self::BitVec(value)
    }
}

impl<'a> Eq for SignalValueRef<'a> {}

/// Owns a signal value.
#[derive(Debug, Clone)]
pub struct SignalValue(SignalValueE);

/// Private enum to hide the internals of [[SignalValue]].
#[derive(Debug, Clone)]
enum SignalValueE {
    Event,
    BitVec(BitVecValue),
    String(String),
    Real(Real),
}

impl<'a> From<SignalValueRef<'a>> for SignalValue {
    fn from(value: SignalValueRef<'a>) -> Self {
        let e = match value {
            SignalValueRef::Event => SignalValueE::Event,
            SignalValueRef::BitVec(v) => SignalValueE::BitVec(v.into()),
            SignalValueRef::String(v) => SignalValueE::String(v.into()),
            SignalValueRef::Real(v) => SignalValueE::Real(v),
        };
        Self(e)
    }
}

impl<'a> From<&'a SignalValue> for SignalValueRef<'a> {
    fn from(value: &'a SignalValue) -> Self {
        match &value.0 {
            SignalValueE::Event => SignalValueRef::Event,
            SignalValueE::BitVec(v) => SignalValueRef::BitVec(v.into()),
            SignalValueE::String(v) => SignalValueRef::String(v),
            SignalValueE::Real(v) => SignalValueRef::Real(*v),
        }
    }
}

impl From<BitVecValue> for SignalValue {
    fn from(value: BitVecValue) -> Self {
        Self(SignalValueE::BitVec(value))
    }
}

/// References the value of a (2/4/9 value) bit vector signal.
#[derive(Debug, Clone, Copy)]
pub struct BitVecRef<'a> {
    states: States,
    width: u32,
    data: &'a [u8],
}

/// Represents a single bit.
/// This is how the values map to ASCII:
/// ` '0', '1', 'x', 'z', 'h', 'u', 'w', 'l', '-' `
#[derive(Debug, Clone, Copy)]
pub struct Bit(u8);

impl Bit {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const X: Self = Self(2);
    pub const Z: Self = Self(3);

    /// Checks to make sure that the value is in range (in debug mode).
    #[inline]
    pub const fn new(value: u8) -> Self {
        debug_assert!((value as usize) < NINE_STATE_LOOKUP.len());
        Bit(value)
    }

    #[inline]
    pub fn as_ascii(&self) -> char {
        NINE_STATE_LOOKUP[self.0 as usize]
    }
}

impl From<Bit> for u8 {
    fn from(value: Bit) -> Self {
        value.0
    }
}

impl From<Bit> for u64 {
    fn from(value: Bit) -> Self {
        value.0 as u64
    }
}

impl<'a> BitVecRef<'a> {
    pub fn new(states: States, width: u32, data: &'a [u8]) -> Self {
        // TODO: can we enforce that states == min_states?
        debug_assert_eq!(states.bytes_required(width), data.len());
        Self {
            states,
            width,
            data,
        }
    }

    /// The number of bits in the bit-vector.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The number of values/states that each bit can represent.
    pub fn states(&self) -> States {
        self.states
    }

    /// Provides access to big-endian bytes iff `self` is a 2-state value.
    pub fn be_bytes(&self) -> Option<&[u8]> {
        if self.states == States::Two {
            Some(self.data)
        } else {
            None
        }
    }

    /// Returns the numeric (not ASCII!) value of a bit.
    pub fn get_bit(&self, bit: u32) -> Bit {
        self.states.get_bit(self.data, bit)
    }

    /// Iterate over bits, starting with the most significant bit.
    pub fn iter_msb_to_lsb(&self) -> impl Iterator<Item = Bit> + '_ {
        (0..self.width()).rev().map(move |bit| self.get_bit(bit))
    }

    /// Iterate over bits, starting with the least significant bit.
    pub fn iter_lsb_to_msb(&self) -> impl Iterator<Item = Bit> + '_ {
        (0..self.width()).map(move |bit| self.get_bit(bit))
    }

    /// Returns a string with one ASCII character for each bit.
    pub fn bit_string(&self) -> String {
        String::from_iter(self.iter_msb_to_lsb().map(|b| b.as_ascii()))
    }

    /// Find the minimum number of states required to represent all bits.
    pub fn find_min_states(&self) -> States {
        if self.states == States::Two {
            // No need to scan, since we already know by construction that all bits are two state.
            States::Two
        } else {
            States::from_bits(self.iter_lsb_to_msb())
        }
    }

    /// Append data to a vec, making sure to properly mask the leading byte.
    pub fn append_to_vec(&self, out: &mut Vec<u8>) {
        let mask = self.states.first_byte_mask(self.width);
        if mask == u8::MAX {
            out.extend_from_slice(self.data);
        } else {
            out.push(self.data[0] & mask);
            out.extend_from_slice(&self.data[1..]);
        }
    }
}

impl PartialEq for BitVecRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.states == other.states {
            debug_assert_eq!(self.data.len(), other.data.len());
            self.data == other.data
        } else {
            self.bit_string() == other.bit_string()
        }
    }
}

impl Display for SignalValueRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            SignalValueRef::Event => write!(f, "Event"),
            SignalValueRef::BitVec(..) => write!(f, "{}", self.to_bit_string().unwrap()),
            SignalValueRef::String(value) => write!(f, "{value}"),
            SignalValueRef::Real(value) => write!(f, "{value}"),
        }
    }
}

impl PartialEq for SignalValueRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SignalValueRef::Real(a), SignalValueRef::Real(b)) => a == b,
            (SignalValueRef::Event, SignalValueRef::Event) => true,
            (SignalValueRef::String(a), SignalValueRef::String(b)) => a == b,
            (SignalValueRef::BitVec(a), SignalValueRef::BitVec(b)) => a == b,
            _ => false,
        }
    }
}

impl<'a> SignalValueRef<'a> {
    pub fn is_event(&self) -> bool {
        matches!(self, SignalValueRef::Event)
    }

    pub fn to_bit_string(&self) -> Option<String> {
        self.as_bit_vec().map(|bv| bv.bit_string())
    }

    /// Returns the number of bits in the signal value. Returns None if the value is a real or string.
    pub fn width(&self) -> Option<u32> {
        match self {
            SignalValueRef::Event => Some(0),
            SignalValueRef::BitVec(b) => Some(b.width),
            _ => None,
        }
    }

    /// Returns the states per bit. Returns None if the value is a real or string.
    pub fn states(&self) -> Option<States> {
        self.as_bit_vec().map(|bv| bv.states())
    }

    /// Returns a reference to the raw data, bits and states
    pub(crate) fn as_bit_vec(&self) -> Option<BitVecRef<'a>> {
        match self {
            SignalValueRef::BitVec(b) => Some(*b),
            _ => None,
        }
    }
}

/// Owns the value of a (2/4/9 value) bit vector signal.
#[derive(Debug, Clone)]
pub struct BitVecValue {
    width: u32,
    states: States,
    data: SmallVec<[u8; 16]>,
}

impl BitVecValue {
    #[inline]
    pub fn zero(states: States, width: u32) -> Self {
        Self {
            width,
            states,
            data: smallvec![0; states.bytes_required(width)],
        }
    }

    pub fn from_u64(value: u64, width: u32) -> Self {
        if width < u64::BITS {
            let mask = (1u64 << width) - 1;
            assert_eq!(
                value,
                value & mask,
                "Cannot represent {value:x} in {width} bits"
            );
        }
        let mut out = Self::zero(States::Two, width);
        let be_bytes = value.to_be_bytes();
        let iter_len = std::cmp::min(out.data.len(), be_bytes.len());
        let data_range = out.data.len() - iter_len..out.data.len();
        let value_range = be_bytes.len() - iter_len..be_bytes.len();
        for (oo, vv) in out.data[data_range]
            .iter_mut()
            .zip(be_bytes[value_range].iter())
        {
            *oo = *vv;
        }
        out
    }

    pub fn repeat(states: States, width: u32, bit: Bit) -> Self {
        debug_assert!(States::from_bit(bit).bytes_required(width) <= states.bytes_required(width));
        let mut value = 0;
        for _ in 0..states.bits_in_a_byte() {
            value <<= states.bits();
            value |= u8::from(bit);
        }
        Self {
            width,
            states,
            data: smallvec![value; states.bytes_required(width)],
        }
    }

    /// Sets the numeric (not ASCII!) value of a bit.
    pub fn set_bit(&mut self, bit: u32, value: Bit) {
        self.states.set_bit(&mut self.data, bit, value);
    }

    pub fn try_from_ascii_chars(value_bits: &[u8]) -> Result<Self, ()> {
        let states = States::from_ascii(value_bits).ok_or(())?;
        let width = value_bits.len() as u32;
        let mut data = smallvec![];
        write_n_state_from_ascii(states, value_bits, |v| data.push(v), None);
        Ok(Self {
            width,
            states,
            data,
        })
    }
}

impl FromStr for BitVecValue {
    type Err = ();

    fn from_str(value_bits: &str) -> Result<Self, Self::Err> {
        Self::try_from_ascii_chars(value_bits.as_bytes())
    }
}

impl FromStr for SignalValue {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        BitVecValue::from_str(s).map(|v| v.into())
    }
}

impl<'a> From<BitVecRef<'a>> for BitVecValue {
    fn from(value: BitVecRef<'a>) -> Self {
        Self {
            width: value.width,
            states: value.states,
            data: value.data.to_smallvec(),
        }
    }
}

impl<'a> From<&'a BitVecValue> for BitVecRef<'a> {
    fn from(value: &'a BitVecValue) -> Self {
        Self {
            width: value.width,
            states: value.states,
            data: &value.data,
        }
    }
}

impl<'a> From<&'a BitVecValue> for SignalValueRef<'a> {
    fn from(value: &'a BitVecValue) -> Self {
        let bv: BitVecRef = value.into();
        bv.into()
    }
}

const NINE_STATE_LOOKUP: [char; 9] = ['0', '1', 'x', 'z', 'h', 'u', 'w', 'l', '-'];

#[repr(u8)]
#[derive(TryFromPrimitive, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub enum States {
    #[default]
    Two = 0,
    Four = 1,
    Nine = 2,
}

impl States {
    pub fn from_bit(value: Bit) -> Self {
        if value.0 <= 1 {
            States::Two
        } else if value.0 <= 3 {
            States::Four
        } else if value.0 < NINE_STATE_LOOKUP.len() as u8 {
            States::Nine
        } else {
            unreachable!(
                "Bit should never contain a value greater {}",
                NINE_STATE_LOOKUP.len() - 1
            );
        }
    }

    pub fn from_ascii_bit(bit: u8) -> Option<(Self, Bit)> {
        let num = bit_char_to_num(bit)?;
        Some((Self::from_bit(num), num))
    }

    pub fn from_ascii(string: &[u8]) -> Option<Self> {
        let mut union = 0;
        for cc in string {
            union |= u8::from(bit_char_to_num(*cc)?);
        }
        Some(Self::from_union(union))
    }

    #[inline]
    fn from_union(union: u8) -> Self {
        // invalid bits are only possible when there was at least one 9-state signal involved
        // since 2 and 4 states signal bits are closed under bit-wise or
        if union >= 4 {
            Self::Nine
        } else {
            Self::from_bit(Bit::new(union))
        }
    }

    pub fn from_bits(bits: impl Iterator<Item = Bit>) -> Self {
        // We combine all values in a single u8, this can result in an invalid bit value,
        // since we might end up with something like (8 | 7) = 15.
        let union = bits.map(u8::from).reduce(|a, b| a | b).unwrap_or(0);
        Self::from_union(union)
    }

    pub fn join(a: Self, b: Self) -> Self {
        let num = std::cmp::max(a as u8, b as u8);
        Self::try_from_primitive(num).unwrap()
    }
    /// Returns how many bits are needed in order to encode one bit of state.
    #[inline]
    pub fn bits(self) -> u32 {
        match self {
            States::Two => 1,
            States::Four => 2,
            States::Nine => 4,
        }
    }

    #[inline]
    pub fn mask(self) -> u8 {
        match self {
            States::Two => 0x1,
            States::Four => 0x3,
            States::Nine => 0xf,
        }
    }

    /// Returns how many signal bits can be encoded in a u8.
    #[inline]
    pub fn bits_in_a_byte(self) -> u32 {
        8 / self.bits()
    }

    /// Returns how many bits the first byte would contain.
    #[inline]
    fn bits_in_first_byte(self, width: u32) -> u32 {
        (width * self.bits()) % u8::BITS
    }

    /// Creates a mask that will only leave the relevant bits in the first byte.
    #[inline]
    pub(crate) fn first_byte_mask(self, width: u32) -> u8 {
        let n = self.bits_in_first_byte(width);
        if n > 0 { (1u8 << n) - 1 } else { u8::MAX }
    }

    /// Returns how many bytes are required to store bits.
    #[inline]
    pub fn bytes_required(self, bits: u32) -> usize {
        // (bits as usize).div_ceil(self.bits_in_a_byte())
        match self {
            States::Two => (bits as usize + 7) >> 3,
            States::Four => (bits as usize + 3) >> 2,
            States::Nine => (bits as usize + 1) >> 1,
        }
    }

    /// Extracts a single bit from a n-state encoding.
    #[inline]
    fn get_bit(&self, data: &[u8], bit: u32) -> Bit {
        debug_assert!(data.len() >= self.bytes_required(bit));
        let bit_in_byte = bit % self.bits_in_a_byte();
        let little_endian_byte_index = (bit / self.bits_in_a_byte()) as usize;
        let big_endian_byte_index = data.len() - 1 - little_endian_byte_index;
        let byte = data[big_endian_byte_index];
        Bit::new(self.mask() & (byte >> (bit_in_byte * self.bits())))
    }

    /// Sets a single bit of an n-state encoded bit-vector {
    #[inline]
    fn set_bit(&self, data: &mut [u8], bit: u32, value: Bit) {
        debug_assert!(data.len() >= self.bytes_required(bit));
        let bit_in_byte = bit % self.bits_in_a_byte();
        let little_endian_byte_index = (bit / self.bits_in_a_byte()) as usize;
        let big_endian_byte_index = data.len() - 1 - little_endian_byte_index;
        let raw_value = u8::from(value);
        let shift_by = bit_in_byte * self.bits();
        let other_bits = !(self.mask() << shift_by);
        data[big_endian_byte_index] =
            (data[big_endian_byte_index] & other_bits) | (raw_value << shift_by);
    }
}

impl std::fmt::Debug for States {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            States::Two => write!(f, "2-state"),
            States::Four => write!(f, "4-state"),
            States::Nine => write!(f, "9-state"),
        }
    }
}

const BIT_CHAR_TO_NUM: [u8; 256] = {
    let mut table = [u8::MAX; 256];
    table[b'0' as usize] = 0;
    table[b'1' as usize] = 1;
    table[b'x' as usize] = 2;
    table[b'X' as usize] = 2;
    table[b'z' as usize] = 3;
    table[b'Z' as usize] = 3;
    table[b'h' as usize] = 4;
    table[b'H' as usize] = 4;
    table[b'u' as usize] = 5;
    table[b'U' as usize] = 5;
    table[b'w' as usize] = 6;
    table[b'W' as usize] = 6;
    table[b'l' as usize] = 7;
    table[b'L' as usize] = 7;
    table[b'-' as usize] = 8;
    table
};

#[inline]
pub fn bit_char_to_num(value: u8) -> Option<Bit> {
    let mapped = BIT_CHAR_TO_NUM[value as usize];
    if mapped == u8::MAX {
        None
    } else {
        Some(Bit(mapped))
    }
}

impl From<Real> for SignalValue {
    fn from(value: Real) -> Self {
        Self(SignalValueE::Real(value))
    }
}

impl From<Real> for SignalValueRef<'_> {
    fn from(value: Real) -> Self {
        Self::Real(value)
    }
}

impl TryFrom<SignalValueRef<'_>> for Real {
    type Error = ();

    fn try_from(value: SignalValueRef<'_>) -> Result<Self, Self::Error> {
        match value {
            SignalValueRef::Event => Err(()),
            SignalValueRef::BitVec(b) => {
                let uint: u64 = b.try_into()?;
                let candidate = uint as f64;
                if (candidate as u64) == uint {
                    Ok(candidate)
                } else {
                    Err(())
                }
            }
            SignalValueRef::String(s) => s.parse::<Real>().map_err(|_| ()),
            SignalValueRef::Real(v) => Ok(v),
        }
    }
}

impl From<String> for SignalValue {
    fn from(value: String) -> Self {
        Self(SignalValueE::String(value))
    }
}

impl<'a> From<&'a str> for SignalValueRef<'a> {
    fn from(value: &'a str) -> Self {
        SignalValueRef::String(value)
    }
}

impl TryFrom<BitVecRef<'_>> for u64 {
    type Error = ();

    fn try_from(value: BitVecRef) -> Result<Self, Self::Error> {
        let mut out = 0;
        for (idx, bit) in value.iter_lsb_to_msb().enumerate() {
            if bit.0 > 1 {
                // not a 2-state value
                return Err(());
            } else if idx < u64::BITS as usize {
                out |= (bit.0 as u64) << idx;
            }
        }
        Ok(out)
    }
}

impl TryFrom<BitVecRef<'_>> for bool {
    type Error = ();

    fn try_from(value: BitVecRef<'_>) -> Result<Self, Self::Error> {
        if value.width == 1 {
            match value.get_bit(0).0 {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(()),
            }
        } else {
            Err(())
        }
    }
}

impl TryFrom<SignalValueRef<'_>> for u64 {
    type Error = ();

    fn try_from(value: SignalValueRef<'_>) -> Result<Self, Self::Error> {
        match value {
            SignalValueRef::Event => Err(()),
            SignalValueRef::BitVec(b) => b.try_into(),
            SignalValueRef::String(s) => s.parse::<u64>().map_err(|_| ()),
            SignalValueRef::Real(v) => {
                let candidate = v as u64;
                if candidate as Real == v {
                    Ok(candidate)
                } else {
                    Err(())
                }
            }
        }
    }
}

impl TryFrom<SignalValueRef<'_>> for bool {
    type Error = ();

    fn try_from(value: SignalValueRef<'_>) -> Result<Self, Self::Error> {
        match value {
            SignalValueRef::Event => Err(()),
            SignalValueRef::BitVec(b) => b.try_into(),
            SignalValueRef::String(s) => s.parse::<bool>().map_err(|_| ()),
            SignalValueRef::Real(_) => Err(()),
        }
    }
}

impl From<SignalValueRef<'_>> for String {
    fn from(value: SignalValueRef<'_>) -> Self {
        format!("{}", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        // signal values contain a slice (ptr + len) as well as a tag and potentially a length
        assert_eq!(std::mem::size_of::<&[u8]>(), 2 * 8);
        assert_eq!(std::mem::size_of::<SignalValueRef>(), 3 * 8);
        // BitVecRef is the same size as SignalValueRef
        assert_eq!(std::mem::size_of::<BitVecRef>(), 3 * 8);
        // A BitVecValue has a Vec (3 pointer sized values) + meta-data
        assert_eq!(std::mem::size_of::<BitVecValue>(), 4 * 8);
        // SignalValue is just as big as a BitVecValue
        assert_eq!(std::mem::size_of::<SignalValue>(), 4 * 8);
    }

    #[test]
    fn test_bit_constants() {
        assert_eq!(Bit::ZERO.as_ascii(), '0');
        assert_eq!(Bit::ONE.as_ascii(), '1');
        assert_eq!(Bit::X.as_ascii(), 'x');
        assert_eq!(Bit::Z.as_ascii(), 'z');
    }

    #[test]
    fn test_to_bit_string_binary() {
        let data0 = [0b11100101u8, 0b00110010];
        let full_str = "1110010100110010";
        let full_str_len = full_str.len();

        for bits in 0..(full_str_len + 1) {
            let expected: String = full_str.chars().skip(full_str_len - bits).collect();
            let number_of_bytes = bits.div_ceil(8);
            let drop_bytes = data0.len() - number_of_bytes;
            let data = &data0[drop_bytes..];
            assert_eq!(
                BitVecRef {
                    states: States::Two,
                    width: bits as u32,
                    data,
                }
                .bit_string(),
                expected,
                "bits={bits}"
            );
        }
    }

    #[test]
    fn test_to_bit_string_four_state() {
        let data0 = [0b11100101u8, 0b00110010];
        let full_str = "zx110z0x";
        let full_str_len = full_str.len();

        for bits in 0..(full_str_len + 1) {
            let expected: String = full_str.chars().skip(full_str_len - bits).collect();
            let number_of_bytes = bits.div_ceil(4);
            let drop_bytes = data0.len() - number_of_bytes;
            let data = &data0[drop_bytes..];
            assert_eq!(
                BitVecRef {
                    states: States::Four,
                    width: bits as u32,
                    data,
                }
                .bit_string(),
                expected,
                "bits={bits}"
            );
        }
    }

    #[test]
    fn test_long_2_state_to_string() {
        let data = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b110001, 0b11, 0b10110011,
        ];
        let out = BitVecRef {
            states: States::Two,
            width: 153,
            data: data.as_slice(),
        }
        .bit_string();
        let expected = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100010000001110110011";
        assert_eq!(out, expected);
    }

    #[test]
    fn test_conversions_real() {
        let inp: Real = 0.345;
        // f64 -> SignalValue
        let value: SignalValue = inp.into();
        // &SignalValue -> SignalValueRef
        let value_ref: SignalValueRef = (&value).into();
        // f64 -> SignalValueRef
        let direct_value_ref: SignalValueRef = inp.into();
        // SignalValueRef -> f64
        assert_eq!(inp, value_ref.try_into().unwrap());
        assert_eq!(inp, direct_value_ref.try_into().unwrap());

        // string value
        let str_value: SignalValueRef = "0.123".into();
        assert_eq!(Real::from(0.123), str_value.try_into().unwrap());
        let str_value_2: SignalValueRef = "bla".into();
        let maybe_real: Result<Real, _> = str_value_2.try_into();
        assert!(maybe_real.is_err());

        // bit vec value
        let value: SignalValue = BitVecValue::from_u64(12345, 32).into();
        let value_ref: SignalValueRef = (&value).into();
        assert_eq!(Real::from(12345), value_ref.try_into().unwrap());
    }

    #[test]
    fn test_string_conversions() {
        let real_value: SignalValueRef = 0.345.into();
        let str_value: SignalValueRef = "1234".into();
        let bit_vec_value: SignalValue = BitVecValue::from_u64(12345, 32).into();
        assert_eq!(String::from(real_value), "0.345");
        assert_eq!(String::from(str_value), "1234");
        assert_eq!(
            String::from(SignalValueRef::from(&bit_vec_value)),
            "00000000000000000011000000111001"
        );
    }

    #[test]
    fn test_bv_conversions() {
        let state_2: SignalValue = "00010101010".parse().unwrap();
        assert_eq!(u64::try_from(SignalValueRef::from(&state_2)).unwrap(), 170);
        let state_2_true: SignalValue = "1".parse().unwrap();
        assert_eq!(
            u64::try_from(SignalValueRef::from(&state_2_true)).unwrap(),
            1
        );
        assert!(bool::try_from(SignalValueRef::from(&state_2_true)).unwrap());
        let state_2_false: SignalValue = "0".parse().unwrap();
        assert_eq!(
            u64::try_from(SignalValueRef::from(&state_2_false)).unwrap(),
            0
        );
        assert!(!bool::try_from(SignalValueRef::from(&state_2_false)).unwrap());
        let state_4: SignalValue = "0zz1010101x".parse().unwrap();
        assert!(u64::try_from(SignalValueRef::from(&state_4)).is_err());
        let state_9: SignalValue = "0--1010101x".parse().unwrap();
        assert!(u64::try_from(SignalValueRef::from(&state_9)).is_err());
    }
}
