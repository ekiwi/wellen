// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use num_enum::TryFromPrimitive;
use std::fmt::{Display, Formatter};

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
    /// Checks to make sure that the value is in range (in debug mode).
    #[inline]
    pub fn new(value: u8) -> Self {
        debug_assert!((value as usize) < NINE_STATE_LOOKUP.len());
        Bit(value)
    }

    #[inline]
    pub fn to_ascii(&self) -> char {
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
        String::from_iter(self.iter_msb_to_lsb().map(|b| b.to_ascii()))
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
            (SignalValueRef::String(a), SignalValueRef::String(b)) => a == b,
            (SignalValueRef::Real(a), SignalValueRef::Real(b)) => a == b,
            _ => self.to_bit_string().unwrap() == other.to_bit_string().unwrap(),
        }
    }
}

impl SignalValueRef<'_> {
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
    pub(crate) fn as_bit_vec(&self) -> Option<BitVecRef<'_>> {
        match self {
            SignalValueRef::BitVec(b) => Some(*b),
            _ => None,
        }
    }
}

// #[derive(Debug, Clone)]
// pub struct SignalValue(SignalValueKind);
//
// /// Private enum to protect [[SignalValue]] internals.
// #[derive(Debug, Clone)]
// enum SignalValueKind {
//     Event,
//     Binary(Vec<u8>, u32),
//     FourValue(Vec<u8>, u32),
//     NineValue(Vec<u8>, u32),
//     String(String),
//     Real(Real),
// }
//
// impl<'a> From<&'a SignalValue> for SignalValueRef<'a> {
//     fn from(value: &'a SignalValue) -> Self {
//         match &value.0 {
//             SignalValueKind::Event => SignalValueRef::Event,
//             SignalValueKind::Binary(data, bits) => SignalValueRef::Binary(data, *bits),
//             SignalValueKind::FourValue(data, bits) => SignalValueRef::FourValue(data, *bits),
//             SignalValueKind::NineValue(data, bits) => SignalValueRef::NineValue(data, *bits),
//             SignalValueKind::String(s) => SignalValueRef::String(s.as_str()),
//             SignalValueKind::Real(data) => SignalValueRef::Real(*data),
//         }
//     }
// }
//
// impl<'a> From<SignalValueRef<'a>> for SignalValue {
//     fn from(value: SignalValueRef<'a>) -> Self {
//         Self(match value {
//             SignalValueRef::Event => SignalValueKind::Event,
//             SignalValueRef::Binary(data, bits) => SignalValueKind::Binary(data.to_vec(), bits),
//             SignalValueRef::FourValue(data, bits) => {
//                 SignalValueKind::FourValue(data.to_vec(), bits)
//             }
//             SignalValueRef::NineValue(data, bits) => {
//                 SignalValueKind::NineValue(data.to_vec(), bits)
//             }
//             SignalValueRef::String(data) => SignalValueKind::String(data.to_string()),
//             SignalValueRef::Real(data) => SignalValueKind::Real(data),
//         })
//     }
// }
//
// impl Display for SignalValue {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let sig_ref: SignalValueRef = self.into();
//         sig_ref.fmt(f)
//     }
// }

const NINE_STATE_LOOKUP: [char; 9] = ['0', '1', 'x', 'z', 'h', 'u', 'w', 'l', '-'];

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
        let union = bits.map(|b| u8::from(b)).reduce(|a, b| a | b).unwrap_or(0);
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
}

#[inline]
pub fn bit_char_to_num(value: u8) -> Option<Bit> {
    match value {
        // Value shared with 2 and 4-state logic
        b'0' | b'1' => Some(Bit(value - b'0')), // strong 0 / strong 1
        // Values shared with Verilog 4-state logic
        b'x' | b'X' => Some(Bit(2)), // strong o or 1 (unknown)
        b'z' | b'Z' => Some(Bit(3)), // high impedance
        // Values unique to the IEEE Standard Logic Type
        b'h' | b'H' => Some(Bit(4)), // weak 1
        b'u' | b'U' => Some(Bit(5)), // uninitialized
        b'w' | b'W' => Some(Bit(6)), // weak 0 or 1 (unknown)
        b'l' | b'L' => Some(Bit(7)), // weak 1
        b'-' => Some(Bit(8)),        // don't care
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        // signal values contain a slice (ptr + len) as well as a tag and potentially a length
        assert_eq!(std::mem::size_of::<&[u8]>(), 16);
        assert_eq!(std::mem::size_of::<SignalValueRef>(), 16 + 8);
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
}
