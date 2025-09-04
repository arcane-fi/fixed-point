
use crate::{construct_sint, error::FixedPointError};
use ::uint::construct_uint;
use std::fmt;

// these have scuffed doc comments because the macro codegens the beginning of them
construct_uint! {
    /// with 256-bits of precision, consisting of four 64-bit words.
    pub struct U256(4);
}

construct_uint! {
    /// with 192-bits of precision, consisting of three 64-bit words.
    pub struct U192(3);
}

construct_uint! {
    /// with 128-bits of precision, consisting of two 64-bit words.
    pub struct U128(2);
}

impl U256 {
    #[inline]
    pub fn wrapping_add(&self, other: U256) -> U256 {
        let (result, _) = self.overflowing_add(other);

        result
    }

    #[inline]
    pub fn wrapping_sub(&self, other: U256) -> U256 {
        let (result, _) = self.overflowing_sub(other);

        result
    }

    #[inline]
    pub fn wrapping_mul(&self, other: U256) -> U256 {
        let (result, _) = self.overflowing_mul(other);

        result
    }
}

impl U192 {
    #[inline]
    pub fn wrapping_add(&self, other: U192) -> U192 {
        let (result, _) = self.overflowing_add(other);

        result
    }

    #[inline]
    pub fn wrapping_sub(&self, other: U192) -> U192 {
        let (result, _) = self.overflowing_sub(other);

        result
    }

    #[inline]
    pub fn wrapping_mul(&self, other: U192) -> U192 {
        let (result, _) = self.overflowing_mul(other);

        result
    }
}

impl U128 {
    #[inline]
    pub fn wrapping_add(&self, other: U128) -> U128 {
        let (result, _) = self.overflowing_add(other);

        result
    }

    #[inline]
    pub fn wrapping_sub(&self, other: U128) -> U128 {
        let (result, _) = self.overflowing_sub(other);

        result
    }

    #[inline]
    pub fn wrapping_mul(&self, other: U128) -> U128 {
        let (result, _) = self.overflowing_mul(other);

        result
    }
}

impl From<U192> for U256 {
    fn from(value: U192) -> U256 {
        let U192(ref arr) = value;

        U256([arr[0], arr[1], arr[2], 0])
    }
}

impl TryFrom<U256> for U192 {
    type Error = FixedPointError;

    fn try_from(value: U256) -> Result<U192, Self::Error> {
        let U256(ref arr) = value;
        if arr[3] != 0 {
            return Err(FixedPointError::IntegerConversionError);
        }

        Ok(U192([arr[0], arr[1], arr[2]]))
    }
}

impl From<U128> for U192 {
    fn from(value: U128) -> U192 {
        let U128(ref arr) = value;

        U192([arr[0], arr[1], 0])
    }
}

/* Signed Integers */

construct_sint! {
    pub struct I256(U256);
}

construct_sint! {
    pub struct I192(U192);
}

construct_sint! {
    pub struct I128(U128);
}

// I192 -> I256 (sign-extend one limb)
impl From<I192> for I256 {
    fn from(v: I192) -> Self {
        let neg = v.is_negative();
        let U192(ref a) = v.to_unsigned();
        // zero-extend then fill the top limb if negative
        let mut hi = 0u64;
        if neg { hi = u64::MAX; }
        I256::from_unsigned(U256([a[0], a[1], a[2], hi]))
    }
}

// I256 -> I192 (only if value fits in 192-bit signed range)
impl TryFrom<I256> for I192 {
    type Error = FixedPointError;
    fn try_from(v: I256) -> Result<Self, Self::Error> {
        let neg = v.is_negative();
        let U256(ref a) = v.to_unsigned();

        // For a value to fit in signed 192:
        //  - if non-negative: bits 192..255 must be zero (a[3] == 0)
        //    and bit191 must be 0 (a[2] top bit == 0).
        //  - if negative: bits 192..255 must be ones (a[3] == u64::MAX).
        if (!neg && (a[3] != 0 || (a[2] >> 63) != 0))
            || (neg && a[3] != u64::MAX)
        {
            return Err(FixedPointError::IntegerConversionError);
        }

        Ok(I192::from_unsigned(U192([a[0], a[1], a[2]])))
    }
}

impl TryFrom<I256> for u128 {
    type Error = FixedPointError;

    fn try_from(v: I256) -> Result<Self, Self::Error> {
        if v.is_negative() { return Err(FixedPointError::IntegerConversionError); }
        let U256(ref a) = v.to_unsigned();
        // must fit in lower 128 bits
        if a[2] != 0 || a[3] != 0 { return Err(FixedPointError::IntegerConversionError); }
        Ok(((a[1] as u128) << 64) | (a[0] as u128))
    }
}

impl TryFrom<I192> for u128 {
    type Error = FixedPointError;

    fn try_from(v: I192) -> Result<Self, Self::Error> {
        if v.is_negative() { return Err(FixedPointError::IntegerConversionError); }
        let U192(ref a) = v.to_unsigned();
        if a[2] != 0 { return Err(FixedPointError::IntegerConversionError); }
        Ok(((a[1] as u128) << 64) | (a[0] as u128))
    }
}

impl TryFrom<I256> for u64 {
    type Error = FixedPointError;
    
    fn try_from(v: I256) -> Result<Self, Self::Error> {
        if v.is_negative() { return Err(FixedPointError::IntegerConversionError); }
        let U256(ref a) = v.to_unsigned();
        if a[3] != 0 || a[2] != 0 || a[1] != 0 { return Err(FixedPointError::IntegerConversionError); }
        Ok(a[0])
    }
}

impl TryFrom<I256> for i64 {
    type Error = FixedPointError;

    fn try_from(v: I256) -> Result<Self, Self::Error> {
        let neg = v.is_negative();
        let U256(ref a) = v.to_unsigned();

        if !neg {
            // non-negative must have all bits >=64 clear and bit 63 clear
            if a[3] != 0 || a[2] != 0 || a[1] != 0 || (a[0] >> 63) != 0 {
                return Err(FixedPointError::IntegerConversionError);
            }
        } else {
            // negative must be proper sign extension: bits 64..255 all ones
            // and bit 63 set (>= i64::MIN)
            if a[3] != u64::MAX || a[2] != u64::MAX || a[1] != u64::MAX || (a[0] >> 63) == 0 {
                return Err(FixedPointError::IntegerConversionError);
            }
        }

        Ok(a[0] as i64)
    }
}

// I256 -> i128
impl TryFrom<I256> for i128 {
    type Error = FixedPointError;

    fn try_from(v: I256) -> Result<Self, Self::Error> {
        let neg = v.is_negative();
        let U256(ref a) = v.to_unsigned(); // LE limbs: [lo, mid1, mid2, hi]

        if !neg {
            // non-negative must have all bits >=128 clear AND bit127 clear
            if a[3] != 0 || a[2] != 0 || (a[1] >> 63) != 0 {
                return Err(FixedPointError::IntegerConversionError);
            }
        } else {
            // negative must be proper sign-extension: bits 128..255 all ones
            // and bit127 set (>= i128::MIN)
            if a[3] != u64::MAX || a[2] != u64::MAX || (a[1] >> 63) == 0 {
                return Err(FixedPointError::IntegerConversionError);
            }
        }

        let lo128 = ((a[1] as u128) << 64) | (a[0] as u128);
        Ok(lo128 as i128) // safe: we just proved it fits
    }
}

// I192 -> i128
impl TryFrom<I192> for i128 {
    type Error = FixedPointError;

    fn try_from(v: I192) -> Result<Self, Self::Error> {
        let neg = v.is_negative();
        let U192(ref a) = v.to_unsigned(); // LE limbs: [lo, mid, hi]

        if !neg {
            // non-negative: bits 128..191 must be zero AND bit127 clear
            if a[2] != 0 || (a[1] >> 63) != 0 {
                return Err(FixedPointError::IntegerConversionError);
            }
        } else {
            // negative: bits 128..191 must be ones AND bit127 set
            if a[2] != u64::MAX || (a[1] >> 63) == 0 {
                return Err(FixedPointError::IntegerConversionError);
            }
        }

        let lo128 = ((a[1] as u128) << 64) | (a[0] as u128);
        Ok(lo128 as i128)
    }
}