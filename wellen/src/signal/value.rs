// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::wavemem::States;
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
            SignalValueRef::Binary(data, bits) => {
                write!(f, "{}", two_state_to_bit_string(data, *bits))
            }
            SignalValueRef::FourValue(data, bits) => {
                write!(f, "{}", four_state_to_bit_string(data, *bits))
            }
            SignalValueRef::NineValue(data, bits) => {
                write!(f, "{}", nine_state_to_bit_string(data, *bits))
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
        match &self {
            SignalValueRef::Binary(data, bits) => Some(two_state_to_bit_string(data, *bits)),
            SignalValueRef::FourValue(data, bits) => Some(four_state_to_bit_string(data, *bits)),
            SignalValueRef::NineValue(data, bits) => Some(nine_state_to_bit_string(data, *bits)),
            other => panic!("Cannot convert {other:?} to bit string"),
        }
    }

    /// Returns the number of bits in the signal value. Returns None if the value is a real or string.
    pub fn bits(&self) -> Option<u32> {
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

    /// Returns a reference to the raw data and a mask. Returns None if the value is a real or string.
    pub(crate) fn data_and_mask(&self) -> Option<(&[u8], u8)> {
        match self {
            SignalValueRef::Binary(data, bits) => Some((*data, States::Two.first_byte_mask(*bits))),
            SignalValueRef::FourValue(data, bits) => {
                Some((*data, States::Four.first_byte_mask(*bits)))
            }
            SignalValueRef::NineValue(data, bits) => {
                Some((*data, States::Nine.first_byte_mask(*bits)))
            }
            _ => None,
        }
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

const TWO_STATE_LOOKUP: [char; 2] = ['0', '1'];
const FOUR_STATE_LOOKUP: [char; 4] = ['0', '1', 'x', 'z'];
const NINE_STATE_LOOKUP: [char; 9] = ['0', '1', 'x', 'z', 'h', 'u', 'w', 'l', '-'];

fn two_state_to_bit_string(data: &[u8], bits: u32) -> String {
    n_state_to_bit_string(States::Two, data, bits)
}

fn four_state_to_bit_string(data: &[u8], bits: u32) -> String {
    n_state_to_bit_string(States::Four, data, bits)
}

fn nine_state_to_bit_string(data: &[u8], bits: u32) -> String {
    n_state_to_bit_string(States::Nine, data, bits)
}

#[inline]
fn n_state_to_bit_string(states: States, data: &[u8], bits: u32) -> String {
    let lookup = match states {
        States::Two => TWO_STATE_LOOKUP.as_slice(),
        States::Four => FOUR_STATE_LOOKUP.as_slice(),
        States::Nine => NINE_STATE_LOOKUP.as_slice(),
    };
    let bits_per_byte = states.bits_in_a_byte() as u32;
    let states_bits = states.bits() as u32;
    let mask = states.mask();

    let mut out = String::with_capacity(bits as usize);
    if bits == 0 {
        return out;
    }

    // the first byte might not contain a full N bits
    let byte0_bits = bits - ((bits / bits_per_byte) * bits_per_byte);
    let byte0_is_special = byte0_bits > 0;
    if byte0_is_special {
        let byte0 = data[0];
        for ii in (0..byte0_bits).rev() {
            let value = (byte0 >> (ii * states_bits)) & mask;
            let char = lookup[value as usize];
            out.push(char);
        }
    }

    for byte in data.iter().skip(if byte0_is_special { 1 } else { 0 }) {
        for ii in (0..bits_per_byte).rev() {
            let value = (byte >> (ii * states_bits)) & mask;
            let char = lookup[value as usize];
            out.push(char);
        }
    }
    out
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
