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
    Binary(&'a [u8], u32),
    FourValue(&'a [u8], u32),
    NineValue(&'a [u8], u32),
    String(&'a str),
    Real(Real),
}

impl Display for SignalValueRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            SignalValueRef::Event => write!(f, "Event"),
            SignalValueRef::Binary(..)
            | SignalValueRef::FourValue(..)
            | SignalValueRef::NineValue(..) => {
                write!(f, "{}", self.to_bit_string().unwrap())
            }
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
        self.iter_msb_to_lsb()
            .map(|bits| String::from_iter(bits.flat_map(bit_num_to_char)))
    }

    /// Returns the number of bits in the signal value. Returns None if the value is a real or string.
    pub fn width(&self) -> Option<u32> {
        match self {
            SignalValueRef::Event => Some(0),
            SignalValueRef::Binary(_, bits) => Some(*bits),
            SignalValueRef::FourValue(_, bits) => Some(*bits),
            SignalValueRef::NineValue(_, bits) => Some(*bits),
            _ => None,
        }
    }

    /// Returns the states per bit. Returns None if the value is a real or string.
    pub fn states(&self) -> Option<States> {
        match self {
            SignalValueRef::Binary(_, _) => Some(States::Two),
            SignalValueRef::FourValue(_, _) => Some(States::Four),
            SignalValueRef::NineValue(_, _) => Some(States::Nine),
            _ => None,
        }
    }

    /// Iterate over bits, starting with the most significant bit.
    pub fn iter_msb_to_lsb(&self) -> Option<impl Iterator<Item = u8> + '_> {
        let (data, bits, states) = self.data_bits_and_states()?;
        Some((0..bits).rev().map(move |bit| states.get_bit(data, bit)))
    }

    /// Iterate over bits, starting with the least significant bit.
    pub fn iter_lsb_to_msb(&self) -> Option<impl Iterator<Item = u8> + '_> {
        let (data, bits, states) = self.data_bits_and_states()?;
        Some((0..bits).map(move |bit| states.get_bit(data, bit)))
    }

    /// Returns a reference to the raw data, bits and states
    pub(crate) fn data_bits_and_states(&self) -> Option<(&[u8], u32, States)> {
        match self {
            SignalValueRef::Binary(data, bits) => Some((*data, *bits, States::Two)),
            SignalValueRef::FourValue(data, bits) => Some((*data, *bits, States::Four)),
            SignalValueRef::NineValue(data, bits) => Some((*data, *bits, States::Nine)),
            _ => None,
        }
    }

    /// Returns a reference to the raw data and a mask. Returns None if the value is a real or string.
    pub(crate) fn data_and_mask(&self) -> Option<(&[u8], u8)> {
        let (data, bits, states) = self.data_bits_and_states()?;
        Some((data, states.first_byte_mask(bits)))
    }
}

#[derive(Debug, Clone)]
pub struct SignalValue(SignalValueKind);

/// Private enum to protect [[SignalValue]] internals.
#[derive(Debug, Clone)]
enum SignalValueKind {
    Event,
    Binary(Vec<u8>, u32),
    FourValue(Vec<u8>, u32),
    NineValue(Vec<u8>, u32),
    String(String),
    Real(Real),
}

impl<'a> From<&'a SignalValue> for SignalValueRef<'a> {
    fn from(value: &'a SignalValue) -> Self {
        match &value.0 {
            SignalValueKind::Event => SignalValueRef::Event,
            SignalValueKind::Binary(data, bits) => SignalValueRef::Binary(data, *bits),
            SignalValueKind::FourValue(data, bits) => SignalValueRef::FourValue(data, *bits),
            SignalValueKind::NineValue(data, bits) => SignalValueRef::NineValue(data, *bits),
            SignalValueKind::String(s) => SignalValueRef::String(s.as_str()),
            SignalValueKind::Real(data) => SignalValueRef::Real(*data),
        }
    }
}

impl<'a> From<SignalValueRef<'a>> for SignalValue {
    fn from(value: SignalValueRef<'a>) -> Self {
        Self(match value {
            SignalValueRef::Event => SignalValueKind::Event,
            SignalValueRef::Binary(data, bits) => SignalValueKind::Binary(data.to_vec(), bits),
            SignalValueRef::FourValue(data, bits) => {
                SignalValueKind::FourValue(data.to_vec(), bits)
            }
            SignalValueRef::NineValue(data, bits) => {
                SignalValueKind::NineValue(data.to_vec(), bits)
            }
            SignalValueRef::String(data) => SignalValueKind::String(data.to_string()),
            SignalValueRef::Real(data) => SignalValueKind::Real(data),
        })
    }
}

impl Display for SignalValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let sig_ref: SignalValueRef = self.into();
        sig_ref.fmt(f)
    }
}

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
    pub fn from_value(value: u8) -> Option<Self> {
        if value <= 1 {
            Some(States::Two)
        } else if value <= 3 {
            Some(States::Four)
        } else if value <= 15 {
            Some(States::Nine)
        } else {
            None
        }
    }

    pub fn from_ascii_bit(bit: u8) -> Option<(Self, u8)> {
        let num = bit_char_to_num(bit)?;
        Some((Self::from_value(num).unwrap(), num))
    }

    pub fn from_ascii(string: &[u8]) -> Option<Self> {
        let mut union = 0;
        for cc in string {
            union |= bit_char_to_num(*cc)?;
        }
        Self::from_value(union)
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
    fn bits_in_first_byte(self, bits: u32) -> u32 {
        (bits * self.bits() as u32) % u8::BITS
    }

    /// Creates a mask that will only leave the relevant bits in the first byte.
    #[inline]
    pub(crate) fn first_byte_mask(self, bits: u32) -> u8 {
        let n = self.bits_in_first_byte(bits);
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
    pub fn get_bit(&self, data: &[u8], bit: u32) -> u8 {
        debug_assert!(data.len() >= self.bytes_required(bit));
        let bit_in_byte = bit % self.bits_in_a_byte();
        let little_endian_byte_index = (bit / self.bits_in_a_byte()) as usize;
        let big_endian_byte_index = data.len() - 1 - little_endian_byte_index;
        let byte = data[big_endian_byte_index];
        self.mask() & (byte >> (bit_in_byte * self.bits()))
    }
}

#[inline]
pub fn bit_char_to_num(value: u8) -> Option<u8> {
    match value {
        // Value shared with 2 and 4-state logic
        b'0' | b'1' => Some(value - b'0'), // strong 0 / strong 1
        // Values shared with Verilog 4-state logic
        b'x' | b'X' => Some(2), // strong o or 1 (unknown)
        b'z' | b'Z' => Some(3), // high impedance
        // Values unique to the IEEE Standard Logic Type
        b'h' | b'H' => Some(4), // weak 1
        b'u' | b'U' => Some(5), // uninitialized
        b'w' | b'W' => Some(6), // weak 0 or 1 (unknown)
        b'l' | b'L' => Some(7), // weak 1
        b'-' => Some(8),        // don't care
        _ => None,
    }
}

#[inline]
fn bit_num_to_char(num: u8) -> Option<char> {
    NINE_STATE_LOOKUP.get(num as usize).cloned()
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
                SignalValueRef::Binary(data, bits as u32)
                    .to_bit_string()
                    .unwrap(),
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
                SignalValueRef::FourValue(data, bits as u32)
                    .to_bit_string()
                    .unwrap(),
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
        let out = SignalValueRef::Binary(data.as_slice(), 153)
            .to_bit_string()
            .unwrap();
        let expected = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100010000001110110011";
        assert_eq!(out, expected);
    }
}
