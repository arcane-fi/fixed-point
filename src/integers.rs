
use crate::error::FixedPointError;
use ::uint::construct_uint;
use std::fmt;

/// Macro to construct signed integer types that wrap unsigned types
#[macro_export]
macro_rules! construct_sint {
    ( $(#[$attr:meta])* $visibility:vis struct $sname:ident ( $uname:ident ); ) => {
        /// Signed integer type wrapping an unsigned type
        #[repr(transparent)]
        $(#[$attr])*
        #[derive(Copy, Clone, Eq, PartialEq, Hash)]
        $visibility struct $sname($uname);

        impl $sname {
            /// The number of bits in this signed integer type
            pub const BITS: u32 = $uname::WORD_BITS as u32 * Self::WORDS as u32;
            /// The number of words in the underlying unsigned type
            const WORDS: usize = core::mem::size_of::<$uname>() / 8;
            /// Position of the sign bit (MSB)
            const SIGN_BIT: usize = Self::BITS as usize - 1;
            
            /// Maximum positive value (2^(n-1) - 1)
            pub const MAX: $sname = {
                let mut max_unsigned = $uname::MAX;
                // Clear the sign bit to get maximum positive value
                let word_idx = (Self::SIGN_BIT) / 64;
                let bit_idx = (Self::SIGN_BIT) % 64;
                max_unsigned.0[word_idx] &= !(1u64 << bit_idx);
                $sname(max_unsigned)
            };
            
            /// Minimum negative value (-2^(n-1))
            pub const MIN: $sname = {
                let mut min_val = $uname::zero();
                // Set only the sign bit
                let word_idx = (Self::SIGN_BIT) / 64;
                let bit_idx = (Self::SIGN_BIT) % 64;
                min_val.0[word_idx] = 1u64 << bit_idx;
                $sname(min_val)
            };

            /// Zero value
            pub const fn zero() -> Self {
                Self($uname::zero())
            }

            /// One value
            pub const fn one() -> Self {
                Self($uname::one())
            }

            /// Negative one value
            pub const fn minus_one() -> Self {
                Self($uname::MAX) // All bits set = -1 in two's complement
            }

            /// Check if this number is negative
            pub const fn is_negative(&self) -> bool {
                self.0.bit(Self::SIGN_BIT)
            }

            /// Check if this number is positive (> 0)
            pub const fn is_positive(&self) -> bool {
                !self.is_negative() && !self.is_zero()
            }

            /// Check if this number is zero
            pub const fn is_zero(&self) -> bool {
                self.0.is_zero()
            }

            /// Get the absolute value as an unsigned integer
            #[inline]
            pub fn abs(&self) -> $uname {
                if self.is_negative() {
                    self.wrapping_neg().0
                } else {
                    self.0
                }
            }

            /// Wrapping negation using two's complement
            #[inline]
            pub fn wrapping_neg(&self) -> Self {
                Self((!self.0).wrapping_add($uname::one()))
            }

            /// Checked negation
            #[inline]
            pub fn checked_neg(self) -> Option<Self> {
                if self == Self::MIN {
                    None // -MIN would overflow
                } else {
                    Some(self.wrapping_neg())
                }
            }

            /// Saturating negation
            #[inline]
            pub fn saturating_neg(self) -> Self {
                if self == Self::MIN {
                    Self::MAX
                } else {
                    self.wrapping_neg()
                }
            }

            /// Overflowing negation
            #[inline]
            pub fn overflowing_neg(self) -> (Self, bool) {
                (self.wrapping_neg(), self == Self::MIN)
            }

            /// Wrapping addition
            #[inline]
            pub fn wrapping_add(self, other: Self) -> Self {
                Self(self.0.wrapping_add(other.0))
            }

            /// Checked addition
            #[inline]
            pub fn checked_add(self, other: Self) -> Option<Self> {
                let (result, overflow) = self.overflowing_add(other);
                if overflow { None } else { Some(result) }
            }

            /// Saturating addition
            #[inline]
            pub fn saturating_add(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_add(other);
                if overflow {
                    if self.is_negative() { Self::MIN } else { Self::MAX }
                } else {
                    result
                }
            }

            /// Overflowing addition
            #[inline]
            pub fn overflowing_add(self, other: Self) -> (Self, bool) {
                let (result, _) = self.0.overflowing_add(other.0);
                let wrapped = Self(result);
                
                // Check for signed overflow:
                // - Both positive, result negative -> overflow
                // - Both negative, result positive -> overflow
                let overflow = (self.is_positive() && other.is_positive() && wrapped.is_negative()) ||
                              (self.is_negative() && other.is_negative() && wrapped.is_positive());
                
                (wrapped, overflow)
            }

            /// Wrapping subtraction
            #[inline]
            pub fn wrapping_sub(self, other: Self) -> Self {
                Self(self.0.wrapping_sub(other.0))
            }

            /// Checked subtraction
            #[inline]
            pub fn checked_sub(self, other: Self) -> Option<Self> {
                let (result, overflow) = self.overflowing_sub(other);
                if overflow { None } else { Some(result) }
            }

            /// Saturating subtraction
            #[inline]
            pub fn saturating_sub(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_sub(other);
                if overflow {
                    if other.is_negative() { Self::MAX } else { Self::MIN }
                } else {
                    result
                }
            }

            /// Overflowing subtraction
            #[inline]
            pub fn overflowing_sub(self, other: Self) -> (Self, bool) {
                let (result, _) = self.0.overflowing_sub(other.0);
                let wrapped = Self(result);
                
                // Check for signed overflow:
                // - Positive - negative = negative -> overflow
                // - Negative - positive = positive -> overflow  
                let overflow = (self.is_positive() && other.is_negative() && wrapped.is_negative()) ||
                              (self.is_negative() && other.is_positive() && wrapped.is_positive());
                
                (wrapped, overflow)
            }

            /// Wrapping multiplication
            #[inline]
            pub fn wrapping_mul(self, other: Self) -> Self {
                Self(self.0.wrapping_mul(other.0))
            }

            /// Checked multiplication
            #[inline]
            pub fn checked_mul(self, other: Self) -> Option<Self> {
                let (result, overflow) = self.overflowing_mul(other);
                if overflow { None } else { Some(result) }
            }

            /// Saturating multiplication
            #[inline]
            pub fn saturating_mul(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_mul(other);
                if overflow {
                    if (self.is_negative()) ^ (other.is_negative()) {
                        Self::MIN
                    } else {
                        Self::MAX
                    }
                } else {
                    result
                }
            }

            /// Overflowing multiplication (simplified check)
            #[inline]
            pub fn overflowing_mul(self, other: Self) -> (Self, bool) {
                // Convert to positive values for unsigned multiplication
                let (abs_self, self_neg) = if self.is_negative() {
                    (self.wrapping_neg().0, true)
                } else {
                    (self.0, false)
                };
                
                let (abs_other, other_neg) = if other.is_negative() {
                    (other.wrapping_neg().0, true)
                } else {
                    (other.0, false)
                };

                let (unsigned_result, unsigned_overflow) = abs_self.overflowing_mul(abs_other);
                let result_negative = self_neg ^ other_neg;
                
                let result = if result_negative {
                    Self((!unsigned_result).wrapping_add($uname::one()))
                } else {
                    Self(unsigned_result)
                };

                // Check for signed overflow
                let overflow = unsigned_overflow || 
                              (!result_negative && result.is_negative()) ||
                              (result_negative && !result.is_negative() && !result.is_zero());

                (result, overflow)
            }

            /// Wrapping division (trunc toward zero).
            /// Panics on division by zero (to mirror Rust primitives).
            #[inline]
            pub fn wrapping_div(self, other: Self) -> Self {
                if other.is_zero() { panic!("division by zero"); }

                // Special two's-complement overflow: MIN / -1 wraps to MIN
                if self == Self::MIN && other == Self::minus_one() {
                    return Self::MIN;
                }

                // Divide magnitudes as unsigned, then apply sign
                let (a, an) = if self.is_negative()  { (self.wrapping_neg().0, true)  } else { (self.0, false) };
                let (b, bn) = if other.is_negative() { (other.wrapping_neg().0, true) } else { (other.0, false) };
                let q = a / b;
                let neg = an ^ bn;

                if neg { Self((!q).wrapping_add($uname::one())) } else { Self(q) }
            }

            /// Checked division. Returns None on /0 or MIN / -1.
            #[inline]
            pub fn checked_div(self, other: Self) -> Option<Self> {
                if other.is_zero() { return None; }
                if self == Self::MIN && other == Self::minus_one() { return None; }
                Some(self.wrapping_div(other))
            }

            /// Saturating division.
            /// On /0: saturates to MIN for negative lhs, MAX for non-negative lhs.
            /// On MIN / -1: saturates to MAX.
            #[inline]
            pub fn saturating_div(self, other: Self) -> Self {
                if other.is_zero() {
                    return if self.is_negative() { Self::MIN } else { Self::MAX };
                }
                if self == Self::MIN && other == Self::minus_one() {
                    return Self::MAX;
                }
                self.wrapping_div(other)
            }

            /// Overflowing division. Returns (result, overflow_flag).
            /// Overflow only for MIN / -1. Division by zero sets overflow=true and returns 0.
            #[inline]
            pub fn overflowing_div(self, other: Self) -> (Self, bool) {
                if other.is_zero() { return (Self::zero(), true); }
                if self == Self::MIN && other == Self::minus_one() { return (Self::MIN, true); }
                (self.wrapping_div(other), false)
            }

            /// Convert to the underlying unsigned type (reinterpret cast)
            pub const fn to_unsigned(self) -> $uname {
                self.0
            }

            /// Create from unsigned type (reinterpret cast)
            pub const fn from_unsigned(value: $uname) -> Self {
                Self(value)
            }

            /// Convert from string in given radix
            pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ::uint::FromStrRadixErr> {
                let src = src.trim();
                if src.starts_with('-') {
                    let abs_part = &src[1..];
                    let abs_val = $uname::from_str_radix(abs_part, radix)?;
                    Ok(Self::from_unsigned((!abs_val).wrapping_add($uname::one())))
                } else {
                    let positive_part = if src.starts_with('+') { &src[1..] } else { src };
                    let val = $uname::from_str_radix(positive_part, radix)?;
                    Ok(Self::from_unsigned(val))
                }
            }

            /// Logical left shift by `rhs` bits
            #[inline]
            pub fn logical_shl(self, rhs: usize) -> Self {
                if rhs == 0 { return self; }
                if rhs >= Self::BITS as usize { panic!("shift overflow"); }

                Self(self.0 << rhs)
            }

            /// Logical right shift by `rhs` bits (sign-extending)
            #[inline]
            pub fn logical_shr(self, rhs: usize) -> Self {
                if rhs == 0 { return self; }
                if rhs >= Self::BITS as usize { panic!("shift overflow"); }

                let logical = self.0 >> rhs;
                if !self.is_negative() {
                    Self(logical)
                } else {
                    // fill the top `rhs` bits with 1s to preserve the sign
                    let mask = (! $uname::zero()) << (Self::BITS as usize - rhs);
                    Self(logical | mask)
                }
            }
        }

        // Implement standard traits
        impl Default for $sname {
            fn default() -> Self {
                Self::zero()
            }
        }

        impl From<u8> for $sname {
            fn from(value: u8) -> Self {
                Self::from_unsigned($uname::from(value as u64))
            }
        }

        impl From<u16> for $sname {
            fn from(value: u16) -> Self {
                Self::from_unsigned($uname::from(value as u64))
            }
        }

        impl From<u32> for $sname {
            fn from(value: u32) -> Self {
                Self::from_unsigned($uname::from(value as u64))
            }
        }

        impl From<u64> for $sname {
            fn from(value: u64) -> Self {
                Self::from_unsigned($uname::from(value as u64))
            }
        }

        impl From<u128> for $sname {
            fn from(value: u128) -> Self {
                Self::from_unsigned($uname::from(value as u64))
            }
        }

        // Conversions from smaller signed types
        impl From<i8> for $sname {
            fn from(value: i8) -> Self {
                if value >= 0 {
                    Self::from_unsigned($uname::from(value as u64))
                } else {
                    let abs_val = value.wrapping_neg() as u64;
                    let unsigned_abs = $uname::from(abs_val);
                    Self::from_unsigned((!unsigned_abs).wrapping_add($uname::one()))
                }
            }
        }

        impl From<i16> for $sname {
            fn from(value: i16) -> Self {
                if value >= 0 {
                    Self::from_unsigned($uname::from(value as u64))
                } else {
                    let abs_val = value.wrapping_neg() as u64;
                    let unsigned_abs = $uname::from(abs_val);
                    Self::from_unsigned((!unsigned_abs).wrapping_add($uname::one()))
                }
            }
        }

        impl From<i32> for $sname {
            fn from(value: i32) -> Self {
                if value >= 0 {
                    Self::from_unsigned($uname::from(value as u64))
                } else {
                    let abs_val = value.wrapping_neg() as u64;
                    let unsigned_abs = $uname::from(abs_val);
                    Self::from_unsigned((!unsigned_abs).wrapping_add($uname::one()))
                }
            }
        }

        impl From<i64> for $sname {
            fn from(value: i64) -> Self {
                if value >= 0 {
                    Self::from_unsigned($uname::from(value as u64))
                } else {
                    let abs_val = value.wrapping_neg() as u64;
                    let unsigned_abs = $uname::from(abs_val);
                    Self::from_unsigned((!unsigned_abs).wrapping_add($uname::one()))
                }
            }
        }

        impl From<i128> for $sname {
            fn from(value: i128) -> Self {
                // two's-complement bits of the source
                let bits: u128 = value as u128;
        
                // pack lower 128 bits into the unsigned backing type
                let lo: u64 = bits as u64;
                let hi: u64 = (bits >> 64) as u64;
                let mut u = $uname::from(lo) | ($uname::from(hi) << 64);
        
                // if the target is wider than 128 bits and the value is negative,
                // sign-extend by filling the high bits with 1s
                if value < 0 && (Self::BITS as usize) > 128 {
                    u |= (! $uname::zero()) << 128;
                }
        
                Self::from_unsigned(u)
            }
        }

        // Arithmetic operators
        impl core::ops::Add for $sname {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_add(other);
                if overflow { panic!("arithmetic overflow"); }
                result
            }
        }

        impl core::ops::Sub for $sname {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_sub(other);
                if overflow { panic!("arithmetic overflow"); }
                result
            }
        }

        impl core::ops::Mul for $sname {
            type Output = Self;
            fn mul(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_mul(other);
                if overflow { panic!("arithmetic overflow"); }
                result
            }
        }

        impl core::ops::Div for $sname {
            type Output = Self;

            #[inline]
            fn div(self, other: Self) -> Self {
                if other.is_zero() { panic!("division by zero"); }
                let (res, ovf) = self.overflowing_div(other);
                if ovf { panic!("arithmetic overflow"); }
                res
            }
        }

        impl core::ops::Neg for $sname {
            type Output = Self;
            fn neg(self) -> Self {
                let (result, overflow) = self.overflowing_neg();
                if overflow { panic!("arithmetic overflow"); }
                result
            }
        }

        impl core::ops::Shl<usize> for $sname {
            type Output = Self;

            #[inline]
            fn shl(self, rhs: usize) -> Self { self.logical_shl(rhs) }
        }

        impl core::ops::ShlAssign<usize> for $sname {
            #[inline]
            fn shl_assign(&mut self, rhs: usize) { *self = (*self).logical_shl(rhs); }
        }

        impl core::ops::Shl<u32> for $sname {
            type Output = Self;

            #[inline]
            fn shl(self, rhs: u32) -> Self { self.logical_shl(rhs as usize) }
        }

        impl core::ops::ShlAssign<u32> for $sname {
            #[inline]
            fn shl_assign(&mut self, rhs: u32) { *self = (*self).logical_shl(rhs as usize); }
        }

        impl core::ops::Shr<usize> for $sname { 
            type Output = Self;

            #[inline]
            fn shr(self, rhs: usize) -> Self { self.logical_shr(rhs) }
        }

        impl core::ops::ShrAssign<usize> for $sname {
            #[inline]
            fn shr_assign(&mut self, rhs: usize) { *self = (*self).logical_shr(rhs); }
        }

        impl core::ops::Shr<u32> for $sname {
            type Output = Self;

            #[inline]
            fn shr(self, rhs: u32) -> Self { self.logical_shr(rhs as usize) }
        }

        impl core::ops::ShrAssign<u32> for $sname {
            #[inline]
            fn shr_assign(&mut self, rhs: u32) { *self = (*self).logical_shr(rhs as usize); }
        }

        // Comparison - delegate to signed comparison
        impl PartialOrd for $sname {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $sname {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                use core::cmp::Ordering;
                
                match (self.is_negative(), other.is_negative()) {
                    (true, false) => Ordering::Less,    // negative < positive
                    (false, true) => Ordering::Greater, // positive > negative
                    (false, false) => self.0.cmp(&other.0), // both positive: compare unsigned
                    (true, true) => other.0.cmp(&self.0),   // both negative: reverse unsigned comparison
                }
            }
        }

        impl core::ops::BitAnd<$sname> for $sname {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl core::ops::BitOr<$sname> for $sname {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl core::ops::BitXor<$sname> for $sname {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }

        impl core::ops::Not for $sname {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                Self(!self.0)
            }
        }

        impl core::ops::BitAndAssign<$sname> for $sname {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl core::ops::BitOrAssign<$sname> for $sname {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl core::ops::BitXorAssign<$sname> for $sname {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl core::ops::AddAssign<$sname> for $sname {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl core::ops::SubAssign<$sname> for $sname {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl core::ops::MulAssign<$sname> for $sname {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0;
            }
        }

        impl core::ops::DivAssign<$sname> for $sname {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0;
            }
        }

        impl core::ops::Rem<$sname> for $sname {
            type Output = Self;

            #[inline]
            fn rem(self, rhs: Self) -> Self {
                Self(self.0 % rhs.0)
            }
        }

        impl core::ops::RemAssign<$sname> for $sname {
            #[inline]
            fn rem_assign(&mut self, rhs: Self) {
                self.0 %= rhs.0;
            }
        }

        // Display formatting
        impl fmt::Display for $sname {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                if self.is_negative() {
                    write!(f, "-{}", self.abs())
                } else {
                    write!(f, "{}", self.0)
                }
            }
        }

        impl fmt::Debug for $sname {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                fmt::Display::fmt(self, f)
            }
        }

        impl fmt::LowerHex for $sname {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                if self.is_negative() {
                    write!(f, "-{:x}", self.abs())
                } else {
                    write!(f, "{:x}", self.0)
                }
            }
        }

        impl fmt::UpperHex for $sname {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                if self.is_negative() {
                    write!(f, "-{:X}", self.abs())
                } else {
                    write!(f, "{:X}", self.0)
                }
            }
        }
    };
}

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