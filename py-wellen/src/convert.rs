use wellen::SignalValue;

/// Trait to easily convert between existing data types
pub trait Mappable: Sized {
    fn try_from_signal(signal_value: SignalValue<'_>) -> Option<Self>;
    fn from_signal(signal_value: SignalValue<'_>) -> Self {
        Self::try_from_signal(signal_value).unwrap()
    }

    fn into_signal(&self) -> SignalValue;

    fn bit_width(&self) -> u32 {
        (std::mem::size_of::<Self>() * 8) as u32
    }
}

macro_rules! impl_mappable_basic {
    ($t:ty) => {
        impl Mappable for $t {
            fn try_from_signal(signal_value: SignalValue<'_>) -> Option<Self> {
                match signal_value {
                    SignalValue::Binary(val, bits) => {
                        if bits <= std::mem::size_of::<Self>() as u32 {
                            let val = val.try_into().ok().map(|val| <$t>::from_be_bytes(val));
                            val
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }

            fn into_signal(&self) -> SignalValue {
                const ARRAY_SIZE: usize = std::mem::size_of::<$t>();
                let value: &[u8; ARRAY_SIZE] = bytemuck::cast_ref(self);
                SignalValue::Binary(value.as_slice(), self.bit_width())
            }
        }
    };
}

impl_mappable_basic!(u8);
impl_mappable_basic!(u16);
impl_mappable_basic!(u32);
impl_mappable_basic!(u64);
impl_mappable_basic!(i8);
impl_mappable_basic!(i16);
impl_mappable_basic!(i32);
impl_mappable_basic!(i64);
//NOTE: we should also cover reals here
impl_mappable_basic!(f32);
impl_mappable_basic!(f64);

#[cfg(test)]
mod tests {
    use super::Mappable;
    use wellen::SignalValue;

    #[test]
    fn test_long_2_state_to_string() {
        let data = [0b0, 0b110001, 0b10110011];

        let out = SignalValue::Binary(data.as_slice(), 12);
        let _ = u16::try_from_signal(out).unwrap();
    }
}
