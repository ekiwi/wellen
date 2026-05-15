use num_bigint::BigUint;
use wellen::SignalValueRef;

/// Trait to easily convert between existing data types
pub trait Mappable: Sized {
    fn try_from_signal(signal_value: SignalValueRef<'_>) -> Option<Self>;
}

macro_rules! impl_mappable_basic {
    ($t:ty) => {
        impl Mappable for $t {
            fn try_from_signal(signal_value: SignalValueRef<'_>) -> Option<Self> {
                match signal_value {
                    SignalValueRef::BitVec(bv)
                        if bv.width() <= std::mem::size_of::<Self>() as u32 =>
                    {
                        bv.be_bytes()
                            .and_then(|b| b.try_into().ok())
                            .map(<$t>::from_be_bytes)
                    }
                    _ => None,
                }
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
impl_mappable_basic!(f32);
impl_mappable_basic!(f64);

impl Mappable for BigUint {
    fn try_from_signal(signal_value: SignalValueRef<'_>) -> Option<Self> {
        match signal_value {
            SignalValueRef::BitVec(bv) => bv.be_bytes().map(BigUint::from_bytes_be),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Mappable;
    use wellen::SignalValueRef;

    #[test]
    fn test_long_2_state_to_string() {
        let data = [0b0, 0b110001, 0b10110011];

        let out = SignalValueRef::Binary(data.as_slice(), 12);
        let _ = u16::try_from_signal(out).unwrap();
    }
}
