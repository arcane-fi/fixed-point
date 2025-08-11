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
        impl Add for $sname {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_add(other);
                if overflow { panic!("arithmetic overflow"); }
                result
            }
        }

        impl Sub for $sname {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_sub(other);
                if overflow { panic!("arithmetic overflow"); }
                result
            }
        }

        impl Mul for $sname {
            type Output = Self;
            fn mul(self, other: Self) -> Self {
                let (result, overflow) = self.overflowing_mul(other);
                if overflow { panic!("arithmetic overflow"); }
                result
            }
        }

        impl Div for $sname {
            type Output = Self;

            #[inline]
            fn div(self, other: Self) -> Self {
                if other.is_zero() { panic!("division by zero"); }
                let (res, ovf) = self.overflowing_div(other);
                if ovf { panic!("arithmetic overflow"); }
                res
            }
        }

        impl Neg for $sname {
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