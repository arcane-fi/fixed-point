use crate::{error::FixedPointError, integers::{U256, I256}};
use std::ops::*;

/// Standard scale of 9 decimals
pub const SCALE: u128 = 1_000_000_000;

macro_rules! impl_q64 {
    ( $(#[$attr:meta])* $visibility:vis struct $name:ident ( $int_type:ty, $intermediate_type:ident ); ) => {
        #[repr(transparent)]
        $(#[$attr])*
        #[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Hash)]
        $visibility struct $name(pub $int_type);

        impl $name {
            pub const ONE: Self = Self(1 << 64);
            pub const ZERO: Self = Self(0);
            pub const MAX: Self = Self(<$int_type>::MAX);
            pub const MIN: Self = Self(<$int_type>::MIN);

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

            /// ## Convert a scaled u64 to a Q64.64
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
                let value_q64 = ((value as $int_type) << 64) / (SCALE as $int_type);

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
                let intermediate = ($intermediate_type::from(self.0) * $intermediate_type::from(other.0)) >> 64;

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
    };
}

impl_q64! {
    /// Unsigned Q64.64 fixed-point number
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a u128
    pub struct Q64(u128, U256);
}

impl_q64! {
    /// Signed Q64.64 fixed-point number
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a i128
    pub struct SQ64(i128, I256);
}

impl Q64 {
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
        let value = (self.0 * SCALE) >> 64;
        value.try_into().map_err(|_| FixedPointError::IntegerConversionError)
    }

    #[inline]
    pub fn abs(self) -> Self {
        self
    }
}

impl SQ64 {
    #[inline]
    pub fn abs(self) -> SQ64 {
        Self(self.0.abs())
    }
    
    #[inline]
    pub fn unsigned_abs(self) -> Q64 {
        Q64((self.0).unsigned_abs())
    }
}