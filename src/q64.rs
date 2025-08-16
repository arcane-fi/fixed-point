use crate::{error::FixedPointError, integers::{I192, I256, U192, U256}, utils::extract_from_raw_bytes};
use std::ops::*;
#[cfg(feature = "anchor")]
use anchor_lang::prelude::{AnchorDeserialize, AnchorSerialize};

macro_rules! impl_q64 {
    ( $(#[$attr:meta])* $visibility:vis struct $name:ident ( $int_type:ty, $intermediate_type:ident ); ) => {
        #[repr(transparent)]
        $(#[$attr])*
        #[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Default)]
        #[cfg_attr(feature = "anchor", derive(AnchorDeserialize, AnchorSerialize))]
        $visibility struct $name(pub $int_type);

        impl $name {
            pub const ONE: Self = Self(1 << 64);
            pub const ZERO: Self = Self(0);
            pub const MAX: Self = Self(<$int_type>::MAX);
            pub const MIN: Self = Self(<$int_type>::MIN);
            pub const U64_SCALE: u64 = 1_000_000_000;

            /// ## Create a new Q64.64
            /// 
            /// ### Arguments
            /// 
            /// * `value` - The Q64.64 value
            /// 
            /// ### Returns
            /// 
            /// The Q64.64 wrapper of the integer type
            pub fn new(value: $int_type) -> Self {
                Self(value)
            }

            /// ## Convert a scaled 9-decimal u64 to a Q64.64
            /// 
            /// i.e. X * 10^9 -> (X << 64) / 10^9
            /// 
            /// ### Arguments
            /// 
            /// * `value` - The scaled value to convert to Q64.64
            /// 
            /// ### Returns
            /// 
            /// The Q64.64 representation of the scaled value
            pub fn from_scaled_u64(value: u64) -> Self {
                let value_q64 = ((value as $int_type) << 64) / (Self::U64_SCALE as $int_type);

                Self(value_q64)
            }

            /// ## Convert a x-decimal scaled u64 to a Q64.64
            /// 
            /// i.e. X * 10^x -> (X << 64) / 10^x
            /// 
            /// ### Arguments
            /// 
            /// * `value` - The scaled value to convert to Q64.64
            /// * `decimals` - The number of decimals in the scaled value
            /// 
            /// ### Returns
            /// 
            /// The Q64.64 representation of the scaled value
            pub fn from_x_scaled_u64(value: u64, decimals: u32) -> Self {
                let value_q64 = ((value as $int_type) << 64) / (10u64.pow(decimals) as $int_type);

                Self(value_q64)
            }

            #[inline]
            pub fn leading_zeros(&self) -> u32 {
                self.0.leading_zeros()
            }

            #[inline]
            pub fn checked_shl(self, rhs: u32) -> Option<Self> {
                self.0.checked_shl(rhs).map(|value| Self(value))
            }

            #[inline]
            pub fn checked_shr(self, rhs: u32) -> Option<Self> {
                self.0.checked_shr(rhs).map(|value| Self(value))
            }

            #[inline]
            pub fn square(self) -> Self {
                let intermediate = ($intermediate_type::from(self.0) * $intermediate_type::from(self.0)) >> 64usize;

                let result = <$int_type>::try_from(intermediate).expect("square overflow");

                Self(result)
            }

            #[inline]
            pub fn saturating_add(self, other: Self) -> Self {
                Self(self.0.saturating_add(other.0))
            }

            #[inline]
            pub fn saturating_sub(self, other: Self) -> Self {
                Self(self.0.saturating_sub(other.0))
            }

            #[inline]
            pub fn checked_add(self, other: Self) -> Option<Self> {
                let result = self.0.checked_add(other.0);

                result.map(|value| Self(value))
            }

            #[inline]
            pub fn checked_sub(self, other: Self) -> Option<Self> {
                let result = self.0.checked_sub(other.0);

                result.map(|value| Self(value))
            }

            #[inline]
            pub fn count_ones(&self) -> u32 {
                self.0.count_ones()
            }

            #[inline]
            pub fn count_zeros(&self) -> u32 {
                self.0.count_zeros()
            }

            #[inline]
            pub fn leading_ones(&self) -> u32 {
                self.0.leading_ones()
            }

            #[inline]
            pub fn trailing_ones(&self) -> u32 {
                self.0.trailing_ones()
            }

            #[inline]
            pub fn trailing_zeros(&self) -> u32 {
                self.0.trailing_zeros()
            }

            #[inline]
            pub fn rotate_left(self, bits: u32) -> Self {
                Self(self.0.rotate_left(bits))
            }

            #[inline]
            pub fn rotate_right(self, bits: u32) -> Self {
                Self(self.0.rotate_right(bits))
            }

            #[inline]
            pub fn reverse_bits(self) -> Self {
                Self(self.0.reverse_bits())
            }

            #[inline]
            pub fn swap_bytes(self) -> Self {
                Self(self.0.swap_bytes())
            }

            /// Little endian bytes to Q64
            #[inline]
            pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, FixedPointError> {
                let value = <$int_type>::from_le_bytes(extract_from_raw_bytes(bytes, 0..16)?);

                Ok(Self(value))
            }

            /// ## Convert a Q64.64 to a scaled u64
            /// 
            /// i.e. (X * 10^9) >> 64 -> X * 10^9
            /// 
            /// ### Arguments
            /// 
            /// * `self` - The Q64.64 value being consumed to create the u64
            /// 
            /// ### Returns
            /// 
            /// The scaled u64 representation of the Q64.64 value
            pub fn try_to_scaled_u64(self) -> Result<u64, FixedPointError> {
                let value = (self.0 * Self::U64_SCALE as $int_type) >> 64;
                value.try_into().map_err(|_| FixedPointError::IntegerConversionError)
            }
        }
        
        impl Add<$name> for $name {
            type Output = Self;

            #[inline]
            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }
        }

        impl Sub<$name> for $name {
            type Output = Self;

            #[inline]
            fn sub(self, other: Self) -> Self {
                Self(self.0 - other.0)
            }
        }

        impl Mul<$name> for $name {
            type Output = Self;

            #[inline]
            fn mul(self, other: Self) -> Self {
                let intermediate = ($intermediate_type::from(self.0) * $intermediate_type::from(other.0)) >> 64usize;

                let result = <$int_type>::try_from(intermediate).expect("multiplication overflow");

                Self(result)
            }
        }

        impl Div<$name> for $name {
            type Output = Self;

            #[inline]
            fn div(self, other: Self) -> Self {
                let intermediate = ($intermediate_type::from(self.0) << 64usize) / $intermediate_type::from(other.0);

                let result = <$int_type>::try_from(intermediate).expect("division overflow");

                Self(result)
            }
        }

        impl Rem<$name> for $name {
            type Output = Self;

            #[inline]
            fn rem(self, other: Self) -> Self {
                Self(self.0 % other.0)
            }
        }
        
        impl AddAssign<$name> for $name {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                self.0 += other.0;
            }
        }

        impl SubAssign<$name> for $name {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                self.0 -= other.0;
            }
        }

        impl MulAssign<$name> for $name {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                *self = *self * other;
            }
        }

        impl DivAssign<$name> for $name {
            #[inline]
            fn div_assign(&mut self, other: Self) {
                *self = *self * other;
            }
        }

        impl RemAssign<$name> for $name {
            #[inline]
            fn rem_assign(&mut self, other: Self) {
                self.0 %= other.0;
            }
        }

        impl Shl<usize> for $name {
            type Output = Self;

            #[inline]
            fn shl(self, rhs: usize) -> Self {
                Self(self.0 << rhs)
            }
        }

        impl ShlAssign<usize> for $name {
            #[inline]
            fn shl_assign(&mut self, rhs: usize) {
                self.0 <<= rhs;
            }
        }

        impl Shr<usize> for $name {
            type Output = Self;

            #[inline]
            fn shr(self, rhs: usize) -> Self {
                Self(self.0 >> rhs)
            }
        }

        impl ShrAssign<usize> for $name {
            #[inline]
            fn shr_assign(&mut self, rhs: usize) {
                self.0 >>= rhs;
            }
        }

        impl BitAnd<$name> for $name {
            type Output = Self;

            #[inline]
            fn bitand(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }
        }

        impl BitAndAssign<$name> for $name {
            #[inline]
            fn bitand_assign(&mut self, other: Self) {
                self.0 &= other.0;
            }
        }

        impl BitOr<$name> for $name {
            type Output = Self;

            #[inline]
            fn bitor(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }
        }

        impl BitOrAssign<$name> for $name {
            #[inline]
            fn bitor_assign(&mut self, other: Self) {
                self.0 |= other.0;
            }
        }

        impl BitXor<$name> for $name {
            type Output = Self;

            #[inline]
            fn bitxor(self, other: Self) -> Self {
                Self(self.0 ^ other.0)
            }
        }

        impl BitXorAssign<$name> for $name {
            #[inline]
            fn bitxor_assign(&mut self, other: Self) {
                self.0 ^= other.0;
            }
        }

        impl Not for $name {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                Self(!self.0)
            }
        }

        impl TryFrom<i32> for $name {
            type Error = FixedPointError;

            /// Converts from an unscaled i32 to a Q64.64
            fn try_from(value: i32) -> Result<Self, Self::Error> {
                let value = <$int_type>::try_from(value).map_err(|_| FixedPointError::IntegerConversionError)?;
                Ok(Self(value << 64))
            }
        }

        impl From<u32> for $name {
            /// Converts from an unscaled u32 to a Q64.64
            fn from(value: u32) -> Self {
                Self(<$int_type>::from(value) << 64)
            }
        }

        impl TryFrom<i64> for $name {
            type Error = FixedPointError;

            /// Converts from an unscaled i64 to a Q64.64
            fn try_from(value: i64) -> Result<Self, Self::Error> {
                let value = <$int_type>::try_from(value).map_err(|_| FixedPointError::IntegerConversionError)?;
                Ok(Self(value << 64))
            }
        }

        impl From<u64> for $name {
            /// Converts from an unscaled u64 to a Q64.64
            fn from(value: u64) -> Self {
                Self(<$int_type>::from(value) << 64)
            }
        }

        impl TryFrom<i128> for $name {
            type Error = FixedPointError;

            /// Converts from an unscaled i128 to a Q64.64
            fn try_from(value: i128) -> Result<Self, Self::Error> {
                let value = <$int_type>::try_from(value).map_err(|_| FixedPointError::IntegerConversionError)?;
                Ok(Self(value << 64))
            }
        }

        impl TryFrom<u128> for $name {
            type Error = FixedPointError;

            /// Converts from an unscaled u128 to a Q64.64
            fn try_from(value: u128) -> Result<Self, Self::Error> {
                let value = <$int_type>::try_from(value).map_err(|_| FixedPointError::IntegerConversionError)?;
                Ok(Self(value << 64))
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Q64({}), scaled_9_decimal={}", self.0, (*self).try_to_scaled_u64().map_err(|_| std::fmt::Error)?)
            }
        }
    };
}

impl_q64! {
    /// Unsigned Q64.64 fixed-point number type, with 192-bit intermediate type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a u128
    pub struct Q64(u128, U192);
}

impl_q64! {
    /// Signed Q64.64 fixed-point number type, with 192-bit intermediate type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a i128
    pub struct SQ64(i128, I192);
}

impl_q64! {
    /// Unsigned Q64.64 fixed-point number type, with 256-bit intermediate type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a u128
    pub struct Q64Large(u128, U256);
}

impl_q64! {
    /// Signed Q64.64 fixed-point number type, with 256-bit intermediate type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a i128
    pub struct SQ64Large(i128, I256);
}

impl Q64 {
    #[inline]
    pub fn abs(self) -> Self {
        self
    }

    #[inline]
    pub fn from_int<T: Into<u128>>(value: T) -> Self {
        Q64(value.into() << 64)
    }

    #[inline]
    pub fn try_from_int<T: TryInto<u128>>(value: T) -> Result<Self, FixedPointError> {
        Ok(Q64(value.try_into().map_err(|_| FixedPointError::IntegerConversionError)? << 64))
    }
}

impl SQ64 {
    #[inline]
    pub const fn abs(self) -> SQ64 {
        Self(self.0.abs())
    }
    
    #[inline]
    pub const fn unsigned_abs(self) -> Q64 {
        Q64((self.0).unsigned_abs())
    }

    #[inline]
    pub fn from_int<T: Into<i128>>(value: T) -> Self {
        SQ64((value.into()) << 64)
    }

    #[inline]
    pub fn try_from_int<T: TryInto<i128>>(value: T) -> Result<Self, FixedPointError> {
        Ok(SQ64(value.try_into().map_err(|_| FixedPointError::IntegerConversionError)? << 64))
    }
}

impl Neg for SQ64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl SQ64Large {
    #[inline]
    pub const fn abs(self) -> SQ64Large {
        Self(self.0.abs())
    }
    
    #[inline]
    pub const fn unsigned_abs(self) -> Q64Large {
        Q64Large((self.0).unsigned_abs())
    }

    #[inline]
    pub fn from_int<T: Into<i128>>(value: T) -> Self {
        SQ64Large((value.into()) << 64)
    }

    #[inline]
    pub fn try_from_int<T: TryInto<i128>>(value: T) -> Result<Self, FixedPointError> {
        Ok(SQ64Large(value.try_into().map_err(|_| FixedPointError::IntegerConversionError)? << 64))
    }
}

impl Neg for SQ64Large {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl From<SQ64> for SQ64Large {
    fn from(value: SQ64) -> Self {
        Self(value.0)
    }
}

impl From<SQ64Large> for SQ64 {
    fn from(value: SQ64Large) -> Self {
        Self(value.0)
    }
}

impl From<Q64Large> for Q64 {
    fn from(value: Q64Large) -> Self {
        Self(value.0)
    }
}

impl From<Q64> for Q64Large {
    fn from(value: Q64) -> Self {
        Self(value.0)
    }
}

impl From<Q64> for SQ64Large {
    fn from(value: Q64) -> Self {
        Self(value.0 as i128)
    }
}

impl From<Q64Large> for SQ64Large {
    fn from(value: Q64Large) -> Self {
        Self(value.0 as i128)
    }
}

impl From<Q64> for SQ64 {
    fn from(value: Q64) -> Self {
        Self(value.0 as i128)
    }
}

impl TryFrom<SQ64> for Q64 {
    type Error = FixedPointError;

    fn try_from(value: SQ64) -> Result<Q64, Self::Error> {
        if value.0.is_negative() {
            return Err(FixedPointError::IntegerConversionError);
        }

        Ok(Q64(value.0 as u128))
    } 
}

impl TryFrom<SQ64Large> for Q64 {
    type Error = FixedPointError;

    fn try_from(value: SQ64Large) -> Result<Q64, Self::Error> {
        if value.0.is_negative() {
            return Err(FixedPointError::IntegerConversionError);
        }

        Ok(Q64(value.0 as u128))
    }
}

impl TryFrom<SQ64> for Q64Large {
    type Error = FixedPointError;

    fn try_from(value: SQ64) -> Result<Q64Large, Self::Error> {
        if value.0.is_negative() {
            return Err(FixedPointError::IntegerConversionError);
        }

        Ok(Q64Large(value.0 as u128))
    }
}

impl TryFrom<SQ64Large> for Q64Large {
    type Error = FixedPointError;

    fn try_from(value: SQ64Large) -> Result<Q64Large, Self::Error> {
        if value.0.is_negative() {
            return Err(FixedPointError::IntegerConversionError);
        }

        Ok(Q64Large(value.0 as u128))
    }
}