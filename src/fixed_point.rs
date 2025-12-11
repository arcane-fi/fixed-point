// Copyright (c) 2025, Arcane Labs
// SPDX-License-Identifier: Apache-2.0

use crate::{error::FixedPointError, integers::{I256, U256}};

/// Usage example:
/// 
/// fixed_point! {
///     pub struct Q1x63(u64, u128, 63, false, None); // unsigned Q1.63
/// }
#[macro_export]
macro_rules! fixed_point {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:tt ( $storage:tt, $wide:ty, $frac_bits:expr, $signed:tt, $unsigned_type:tt, $gen_one:tt );
    ) => {
        #[repr(transparent)]
        $(#[$attr])*
        #[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug, Default)]
        $vis struct $name(pub $storage);

        impl $name {
            pub const FRAC_BITS: u32 = $frac_bits;
            pub const MAX: Self = Self(<$storage>::MAX);
            pub const MIN: Self = Self(<$storage>::MIN);
            pub const ZERO: Self = Self(0 as $storage);
            
            $crate::fixed_point::__private::__gen_one_const!($gen_one, $storage);

            // --- compile-time guards ---
            const __ASSERTS: () = {
                const S_BITS: usize = core::mem::size_of::<$storage>() * 8;
                const W_BITS: usize = core::mem::size_of::<$wide>() * 8;
            
                // fractional bits must fit in storage
                if !($frac_bits < S_BITS) {
                    panic!("FRAC_BITS must be < storage bit-width");
                }
            
                // wide must be >= storage
                if !(W_BITS >= S_BITS) {
                    panic!("Wide bit-width must be >= storage bit-width");
                }
            };

            $crate::fixed_point::__private::__impl_rne_div_for_signedness!($signed, $wide, rne_div_wide);

            #[inline]
            fn storage_from_u64(v: u64) -> $storage {
                <$storage as core::convert::TryFrom<u64>>
                    ::try_from(v)
                    .unwrap_or_else(|_| panic!("u64 -> storage out of range"))
            }

            #[inline] pub const fn new(value: $storage) -> Self { Self(value) }
            #[inline] pub const fn into_raw(self) -> $storage { self.0 }

            /// Convert an integer scaled by 10^decimals to fixed-point:
            ///     X * 10^decimals -> (X << FRAC_BITS) / 10^decimals
            /// xdp → Q (RNE): q = round((value << FRAC_BITS) / 10^dec)
            #[track_caller]
            #[inline]
            pub fn from_x_scaled_u64(value: u64, decimals: u32) -> Self {
                // den = 10^dec in storage, then lift
                let ten_s: $storage = <$storage as core::convert::From<u8>>::from(10u8);
                let den_s: $storage = ten_s.pow(decimals);
                let den: $wide = <$wide as core::convert::From<$storage>>::from(den_s);

                let v_s: $storage = <$storage as core::convert::TryFrom<u64>>::try_from(value)
                    .unwrap_or_else(|_| panic!("from_x_scaled_u64: value out of storage range"));
                let num: $wide = (<$wide as core::convert::From<$storage>>::from(v_s)) << Self::FRAC_BITS;

                let q: $wide = Self::rne_div_wide(num, den);
                let out: $storage = <$storage as core::convert::TryFrom<$wide>>::try_from(q)
                    .unwrap_or_else(|_| panic!("from_x_scaled_u64: narrowing overflow"));
                Self(out)
            }

            /// Convenience: decimals = 9 (common on-chain scale for Solana)
            pub const U64_SCALE: u64 = 1_000_000_000;
            /// 9dp → Q (RNE): q = round((value << FRAC_BITS) / 1e9)
            #[track_caller]
            #[inline]
            pub fn from_scaled_u64(value: u64) -> Self {
                // den = 1e9 in storage, then lift
                let den_s: $storage = Self::storage_from_u64(Self::U64_SCALE);
                let den: $wide = <$wide as core::convert::From<$storage>>::from(den_s);

                // num = (value << FRAC_BITS) in wide (route via storage to avoid From<u64> for signed storage)
                let v_s: $storage = <$storage as core::convert::TryFrom<u64>>::try_from(value)
                    .unwrap_or_else(|_| panic!("from_scaled_u64: value out of storage range"));
                let num: $wide = (<$wide as core::convert::From<$storage>>::from(v_s)) << Self::FRAC_BITS;

                // RNE divide and narrow
                let q: $wide = Self::rne_div_wide(num, den);
                let out: $storage = <$storage as core::convert::TryFrom<$wide>>::try_from(q)
                    .unwrap_or_else(|_| panic!("from_scaled_u64: narrowing overflow"));
                Self(out)
            }

            // --- arithmetic helpers, storage domain ---
            #[inline] pub fn saturating_add(self, rhs: Self) -> Self { Self(self.0.saturating_add(rhs.0)) }
            #[inline] pub fn saturating_sub(self, rhs: Self) -> Self { Self(self.0.saturating_sub(rhs.0)) }
            #[inline] pub fn checked_add(self, rhs: Self) -> Option<Self> { self.0.checked_add(rhs.0).map(|v| Self(v)) }
            #[inline] pub fn checked_sub(self, rhs: Self) -> Option<Self> { self.0.checked_sub(rhs.0).map(|v| Self(v)) }
            #[inline] pub fn wrapping_add(self, rhs: Self) -> Self { Self(self.0.wrapping_add(rhs.0)) }
            #[inline] pub fn wrapping_sub(self, rhs: Self) -> Self { Self(self.0.wrapping_sub(rhs.0)) }

            // --- widening arithmetic, into $wide domain ---

            /// (a * b) >> FRAC_BITS (trunc towards zero)
            #[track_caller]
            #[inline]
            pub fn mul_trunc(self, rhs: Self) -> Self {
                let a: $wide = <_ as core::convert::From<$storage>>::from(self.0);
                let b: $wide = <_ as core::convert::From<$storage>>::from(rhs.0);
                let p: $wide = a * b;
                let q: $wide = p >> Self::FRAC_BITS;
                let out: $storage = <_ as core::convert::TryFrom<$wide>>::try_from(q)
                    .unwrap_or_else(|_| panic!("multiplication overflow"));
                Self(out)
            }

            /// (a << FRAC_BITS) / b
            #[track_caller]
            #[inline]
            pub fn div_trunc(self, rhs: Self) -> Self {
                debug_assert!(rhs.0 != <$storage as core::default::Default>::default(), "division by zero");

                let num: $wide = <$wide as core::convert::From<$storage>>::from(self.0) << Self::FRAC_BITS;
                let den: $wide = <_ as core::convert::From<$storage>>::from(rhs.0);
                let q: $wide = num / den;
                let out: $storage = <_ as core::convert::TryFrom<$wide>>::try_from(q)
                    .unwrap_or_else(|_| panic!("division overflow"));
                Self(out)
            }

            /// (x * x) >> FRAC_BITS
            #[track_caller]
            #[inline]
            pub fn square(self) -> Self {
                let a: $wide = <_ as core::convert::From<$storage>>::from(self.0);
                let p: $wide = a * a;
                let q: $wide = p >> Self::FRAC_BITS;
                let out: $storage = <_ as core::convert::TryFrom<$wide>>::try_from(q)
                    .unwrap_or_else(|_| panic!("square overflow"));
                Self(out)
            }

            /// Raw storage remainder, not fixed-point modulo
            #[inline] pub fn rem_raw(self, rhs: Self) -> Self { Self(self.0 % rhs.0) }

            /// (X * 10^decimals) >> FRAC_BITS -> u64
            /// Q → xdp (RNE): q = round((raw * 10^dec) / (1<<FRAC_BITS))
            #[inline]
            pub fn try_to_x_scaled_u64(self, decimals: u32) -> Result<u64, FixedPointError> {
                let ten_s: $storage = <$storage as core::convert::From<u8>>::from(10u8);
                let mul_s: $storage = ten_s.pow(decimals);
                let mul: $wide = <$wide as core::convert::From<$storage>>::from(mul_s);

                let val: $wide = <$wide as core::convert::From<$storage>>::from(self.0);
                let num: $wide = val * mul;

                let one_w: $wide = <$wide as core::convert::From<u8>>::from(1u8);
                let den: $wide = one_w << Self::FRAC_BITS;

                let q: $wide = Self::rne_div_wide(num, den);
                <u64 as core::convert::TryFrom<$wide>>::try_from(q)
                    .map_err(|_| FixedPointError::IntegerConversionError)
            }

            /// Q → 9dp (RNE): q = round((raw * 1e9) / (1<<FRAC_BITS))
            #[inline]
            pub fn try_to_scaled_u64(self) -> Result<u64, FixedPointError> {
                let mul_s: $storage = Self::storage_from_u64(Self::U64_SCALE);
                let mul: $wide = <$wide as core::convert::From<$storage>>::from(mul_s);

                let val: $wide = <$wide as core::convert::From<$storage>>::from(self.0);
                let num: $wide = val * mul;

                let one_w: $wide = <$wide as core::convert::From<u8>>::from(1u8);
                let den: $wide = one_w << Self::FRAC_BITS;

                let q: $wide = Self::rne_div_wide(num, den);
                <u64 as core::convert::TryFrom<$wide>>::try_from(q)
                    .map_err(|_| FixedPointError::IntegerConversionError)
            }

            #[inline] pub fn leading_zeros(&self) -> u32 { self.0.leading_zeros() }
            #[inline] pub fn leading_ones(&self) -> u32 { self.0.leading_ones() }
            #[inline] pub fn trailing_zeros(&self) -> u32 { self.0.trailing_zeros() }
            #[inline] pub fn trailing_ones(&self) -> u32 { self.0.trailing_ones() }
            #[inline] pub fn count_zeros(&self) -> u32 { self.0.count_zeros() }
            #[inline] pub fn count_ones(&self) -> u32 { self.0.count_ones() }
            #[inline] pub fn reverse_bits(self) -> Self { Self(self.0.reverse_bits()) }
            #[inline] pub fn swap_bytes(self) -> Self { Self(self.0.swap_bytes()) }
            #[inline] pub fn rotate_left(self, n: u32) -> Self { Self(self.0.rotate_left(n)) }
            #[inline] pub fn rotate_right(self, n: u32) -> Self { Self(self.0.rotate_right(n)) }
            #[inline] pub fn checked_shl(self, rhs: u32) -> Option<Self> { self.0.checked_shl(rhs).map(Self) }
            #[inline] pub fn checked_shr(self, rhs: u32) -> Option<Self> { self.0.checked_shr(rhs).map(Self) }
        }

        // ---- operator impls (PANIC on overflow for add/sub) ----
        impl core::ops::Add<$name> for $name {
            type Output = Self;

            #[track_caller]
            #[inline]
            fn add(self, rhs: Self) -> Self {
                let sum = self.0
                    .checked_add(rhs.0)
                    .unwrap_or_else(|| panic!("addition overflow"));
                Self(sum)
            }
        }

        impl core::ops::Sub<$name> for $name {
            type Output = Self;

            #[track_caller]
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                let diff = self.0
                    .checked_sub(rhs.0)
                    .unwrap_or_else(|| panic!("subtraction overflow"));
                Self(diff)
            }
        }

        // Multiplication/division still go through the widened, panicking helpers you already have.
        impl core::ops::Mul<$name> for $name {
            type Output = Self;
            #[inline] fn mul(self, rhs: Self) -> Self { self.mul_trunc(rhs) }
        }
        impl core::ops::Div<$name> for $name {
            type Output = Self;
            #[inline] fn div(self, rhs: Self) -> Self { self.div_trunc(rhs) }
        }
        impl core::ops::Rem<$name> for $name {
            type Output = Self;
            #[inline] fn rem(self, rhs: Self) -> Self { Self(self.0 % rhs.0) }
        }

        // ---- Assign variants (also PANIC on overflow for add/sub) ----

        impl core::ops::AddAssign<$name> for $name {
            #[track_caller]
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0
                    .checked_add(rhs.0)
                    .unwrap_or_else(|| panic!("addition overflow"));
            }
        }

        impl core::ops::SubAssign<$name> for $name {
            #[track_caller]
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 = self.0
                    .checked_sub(rhs.0)
                    .unwrap_or_else(|| panic!("subtraction overflow"));
            }
        }

        impl core::ops::MulAssign<$name> for $name {
            #[inline] fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
        }
        impl core::ops::DivAssign<$name> for $name {
            #[inline] fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
        }
        impl core::ops::RemAssign<$name> for $name {
            #[inline] fn rem_assign(&mut self, rhs: Self) { self.0 %= rhs.0; }
        }

        impl core::ops::BitAnd<$name> for $name { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { Self(self.0 & o.0) } }
        impl core::ops::BitAndAssign<$name> for $name { #[inline] fn bitand_assign(&mut self, o: Self) { self.0 &= o.0; } }
        impl core::ops::BitOr<$name>  for $name { type Output = Self; #[inline] fn bitor (self, o: Self) -> Self { Self(self.0 | o.0) } }
        impl core::ops::BitOrAssign<$name>  for $name { #[inline] fn bitor_assign(&mut self, o: Self) { self.0 |= o.0; } }
        impl core::ops::BitXor<$name> for $name { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { Self(self.0 ^ o.0) } }
        impl core::ops::BitXorAssign<$name> for $name { #[inline] fn bitxor_assign(&mut self, o: Self) { self.0 ^= o.0; } }
        impl core::ops::Not for $name { type Output = Self; #[inline] fn not(self) -> Self { Self(!self.0) } }

        impl core::ops::Shl<usize> for $name {
            type Output = Self;

            #[inline]
            fn shl(self, shift: usize) -> Self {
                Self(self.0 << shift)
            }
        }

        impl core::ops::Shr<usize> for $name {
            type Output = Self;

            #[inline]
            fn shr(self, shift: usize) -> Self {
                Self(self.0 >> shift)
            }
        }

        impl core::ops::Shl<u32> for $name {
            type Output = Self;

            #[inline]
            fn shl(self, shift: u32) -> Self {
                Self(self.0 << shift)
            }
        }

        impl core::ops::Shr<u32> for $name {
            type Output = Self;

            #[inline]
            fn shr(self, shift: u32) -> Self {
                Self(self.0 >> shift)
            }
        }

        impl core::ops::ShlAssign<usize> for $name {
            #[inline]
            fn shl_assign(&mut self, shift: usize) { self.0 <<= shift; }
        }

        impl core::ops::ShrAssign<usize> for $name {
            #[inline]
            fn shr_assign(&mut self, shift: usize) { self.0 >>= shift; }
        }

        impl core::ops::ShlAssign<u32> for $name {
            #[inline]
            fn shl_assign(&mut self, shift: u32) { self.0 <<= shift; }
        }

        impl core::ops::ShrAssign<u32> for $name {
            #[inline]
            fn shr_assign(&mut self, shift: u32) { self.0 >>= shift; }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.try_to_scaled_u64() {
                    Ok(scaled) => {
                        let int_part = scaled / Self::U64_SCALE;
                        let frac_part = scaled % Self::U64_SCALE;

                        let frac_str = format!("{:09}", frac_part);
                        let trimmed = frac_str.trim_end_matches('0');

                        if trimmed.is_empty() {
                            write!(f, "{}", int_part)
                        } else {
                            write!(f, "{}.{}", int_part, trimmed)
                        }
                    }
                    Err(_) => {
                        write!(f, "0x{:x}", self.0)
                    }
                }
            }
        }

        $crate::fixed_point::__private::__impl_signed_fixed_point_ops!($name, $unsigned_type, $signed);
        $crate::fixed_point::__private::__impl_fixed_point_from_base_int!($name, $storage, $signed);

        // Optional: bytemuck
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Zeroable for $name {}
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Pod for $name {}

        // Optional: serde
        #[cfg(feature = "serde")]
        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where S: serde::Serializer
            {
                self.0.serialize(serializer)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                where D: serde::Deserializer<'de>
            {
                <$storage>::deserialize(deserializer).map(Self)
            }
        }

        #[cfg(feature = "anchor")]
        impl anchor_lang::AnchorDeserialize for $name {
            fn deserialize_reader<R: std::io::Read>(reader: &mut R) -> Result<Self, std::io::Error> {
                let storage = borsh::BorshDeserialize::deserialize_reader(reader)?;
                Ok(Self(storage))
            }
        }

        #[cfg(feature = "anchor")]
        impl anchor_lang::AnchorSerialize for $name {
            fn serialize<W: std::io::Write>(&self, writer: &mut W) -> Result<(), std::io::Error> {
                self.into_raw().serialize(writer)?;
                Ok(())
            }
        }

        // Represent this transparent newtype as its storage primitive in IDL.
        #[cfg(feature = "idl-build")]
        impl anchor_lang::IdlBuild for $name {
            fn create_type() -> Option<anchor_lang::idl::types::IdlTypeDef> {
                use anchor_lang::idl::types::*;

                Some(IdlTypeDef {
                    name: stringify!($name).to_string(),
                    docs: vec![],
                    serialization: IdlSerialization::Borsh,
                    repr: None,
                    ty: IdlTypeDefTy::Type {
                        alias: $crate::fixed_point::__private::__idl_type_for_storage!($storage),
                    },
                    generics: vec![], // v0.31.1 expects this field
                })
            }

            fn insert_types(
                _types: &mut std::collections::BTreeMap<String, anchor_lang::idl::types::IdlTypeDef>
            ) {
                // no nested types to register for a primitive alias
            }

            fn get_full_path() -> String {
                // used for disambiguation; stable enough
                stringify!($name).to_string()
            }
        }
    };
}

mod __private {
    macro_rules! __gen_one_const {
        (true, $storage:ty) => {
            pub const ONE: Self = Self((1 as $storage) << Self::FRAC_BITS);
        };
        (false, $storage:ty) => {};
    }

    // Helper that expands differently for signed vs unsigned
    macro_rules! __impl_rne_div_for_signedness {
        // $signed == true -> signed version
        (true, $wide:ty, $fn_name:ident) => {
            #[inline]
            fn $fn_name(num: $wide, den: $wide) -> $wide {
                // Preconditions: den > 0
                let q0: $wide = num / den;
                let r:  $wide = num - (q0 * den);

                let zero: $wide = <$wide as core::convert::From<u8>>::from(0u8);
                let one:  $wide = <$wide as core::convert::From<u8>>::from(1u8);
                let two:  $wide = <$wide as core::convert::From<u8>>::from(2u8);

                // |r| and sign(num)
                let neg_r = zero - r;
                let abs_r = if r < zero { neg_r } else { r };
                let twice = abs_r * two;

                let gt  = twice > den;
                let tie = twice == den;
                let odd = (q0 & one) == one;

                // +1 if num>=0 else -1
                let sign = if num < zero { zero - one } else { one };

                if gt || (tie && odd) { q0 + sign } else { q0 }
            }
        };

        // $signed == false -> unsigned version
        (false, $wide:ty, $fn_name:ident) => {
            #[inline]
            fn $fn_name(num: $wide, den: $wide) -> $wide {
                // Preconditions: den > 0
                let q0: $wide = num / den;
                let r:  $wide = num - (q0 * den);

                let one: $wide = <$wide as core::convert::From<u8>>::from(1u8);
                let two: $wide = <$wide as core::convert::From<u8>>::from(2u8);

                let twice = r * two;
                let round_up = (twice > den) || (twice == den && ((q0 & one) == one));
                if round_up { q0 + one } else { q0 }
            }
        };
        ($other:tt, $wide:ty, $fn_name:ident) => {
            compile_error!("$signed must be the literal `true` or `false`");
        };
    }

    macro_rules! __impl_signed_fixed_point_ops {
        ($name:ident, $unsigned:ty, true) => {
            impl core::ops::Neg for $name {
                type Output = Self;
                #[track_caller]
                #[inline]
                fn neg(self) -> Self {
                    let val = self.0.checked_neg()
                        .unwrap_or_else(|| panic!("unary negation overflow"));
    
                    Self(val)
                }
            }
        
            impl core::ops::Neg for &$name {
                type Output = $name;
                #[inline]
                fn neg(self) -> $name { (*self).neg() }
            }
        
            impl $name {
                #[inline] pub fn abs(self) -> Self { if self.0 < 0 { -self } else { self } }
                #[inline] pub fn unsigned_abs(self) -> $unsigned { <$unsigned>::from(self.abs()) }
                #[inline] pub fn saturating_neg(self) -> Self { Self(self.0.saturating_neg()) }
                #[inline] pub fn wrapping_neg(self) -> Self { Self(self.0.wrapping_neg()) }
                #[inline] pub fn is_negative(&self) -> bool { self.0.is_negative() }
                #[inline] pub fn is_positive(&self) -> bool { self.0.is_positive() }
            }
        };
        ($name:ident, $unsigned:tt, false) => {};
    }

    macro_rules! __impl_fixed_point_from_base_int {
        // types where the base integer is i64
        ($name:ident, i64, true) => {
            impl core::convert::From<i64> for $name {
                #[inline]
                fn from(v: i64) -> Self {
                    Self(v << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<i32> for $name {
                #[inline]
                fn from(v: i32) -> Self {
                    Self((v as i64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<i16> for $name {
                #[inline]
                fn from(v: i16) -> Self {
                    Self((v as i64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<i8> for $name {
                #[inline]
                fn from(v: i8) -> Self {
                    Self((v as i64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u32> for $name {
                #[inline]
                fn from(v: u32) -> Self {
                    Self((v as i64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u16> for $name {
                #[inline]
                fn from(v: u16) -> Self {
                    Self((v as i64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u8> for $name {
                #[inline]
                fn from(v: u8) -> Self {
                    Self((v as i64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::TryFrom<i128> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i128) -> Result<Self, Self::Error> {
                    let short: i64 = <_ as core::convert::TryFrom<i128>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(short << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<u128> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: u128) -> Result<Self, Self::Error> {
                    let short: i64 = <_ as core::convert::TryFrom<u128>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(short << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<u64> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: u64) -> Result<Self, Self::Error> {
                    let short: i64 = <_ as core::convert::TryFrom<u64>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(short << Self::FRAC_BITS))
                }
            }
        };
        // for types where the base integer is u64
        ($name:ident, u64, false) => {
            impl core::convert::From<u64> for $name {
                #[inline]
                fn from(v: u64) -> Self {
                    Self(v << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u32> for $name {
                #[inline]
                fn from(v: u32) -> Self {
                    Self((v as u64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u16> for $name {
                #[inline]
                fn from(v: u16) -> Self {
                    Self((v as u64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u8> for $name {
                #[inline]
                fn from(v: u8) -> Self {
                    Self((v as u64) << Self::FRAC_BITS)
                }
            }

            impl core::convert::TryFrom<u128> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: u128) -> Result<Self, Self::Error> {
                    let short: u64 = <_ as core::convert::TryFrom<u128>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(short << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i128> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i128) -> Result<Self, Self::Error> {
                    let short: u64 = <_ as core::convert::TryFrom<i128>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(short << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i64> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i64) -> Result<Self, Self::Error> {
                    let unsigned: u64 = <_ as core::convert::TryFrom<i64>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i32> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i32) -> Result<Self, Self::Error> {
                    let unsigned: u64 = <_ as core::convert::TryFrom<i32>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i16> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i16) -> Result<Self, Self::Error> {
                    let unsigned: u64 = <_ as core::convert::TryFrom<i16>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i8> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i8) -> Result<Self, Self::Error> {
                    let unsigned: u64 = <_ as core::convert::TryFrom<i8>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }
        };
        // for types where the base integer is i128
        ($name:ident, i128, true) => {
            impl core::convert::From<i128> for $name {
                #[inline]
                fn from(v: i128) -> Self {
                    Self(v << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<i64> for $name {
                #[inline]
                fn from(v: i64) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<i32> for $name {
                #[inline]
                fn from(v: i32) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<i16> for $name {
                #[inline]
                fn from(v: i16) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<i8> for $name {
                #[inline]
                fn from(v: i8) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u64> for $name {
                #[inline]
                fn from(v: u64) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u32> for $name {
                #[inline]
                fn from(v: u32) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u16> for $name {
                #[inline]
                fn from(v: u16) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u8> for $name {
                #[inline]
                fn from(v: u8) -> Self {
                    Self((v as i128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::TryFrom<u128> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: u128) -> Result<Self, Self::Error> {
                    let short: i128 = <_ as core::convert::TryFrom<u128>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(short << Self::FRAC_BITS))
                }
            }
        };
        // for types where base int is u128
        ($name:ident, u128, false) => {
            impl core::convert::From<u128> for $name {
                #[inline]
                fn from(v: u128) -> Self {
                    Self(v << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u64> for $name {
                #[inline]
                fn from(v: u64) -> Self {
                    Self((v as u128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u32> for $name {
                #[inline]
                fn from(v: u32) -> Self {
                    Self((v as u128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u16> for $name {
                #[inline]
                fn from(v: u16) -> Self {
                    Self((v as u128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::From<u8> for $name {
                #[inline]
                fn from(v: u8) -> Self {
                    Self((v as u128) << Self::FRAC_BITS)
                }
            }

            impl core::convert::TryFrom<i128> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i128) -> Result<Self, Self::Error> {
                    let unsigned: u128 = <_ as core::convert::TryFrom<i128>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i64> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i64) -> Result<Self, Self::Error> {
                    let unsigned: u128 = <_ as core::convert::TryFrom<i64>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i32> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i32) -> Result<Self, Self::Error> {
                    let unsigned: u128 = <_ as core::convert::TryFrom<i32>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i16> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i16) -> Result<Self, Self::Error> {
                    let unsigned: u128 = <_ as core::convert::TryFrom<i16>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }

            impl core::convert::TryFrom<i8> for $name {
                type Error = FixedPointError;

                #[inline]
                fn try_from(v: i8) -> Result<Self, Self::Error> {
                    let unsigned: u128 = <_ as core::convert::TryFrom<i8>>::try_from(v)
                        .map_err(|_| FixedPointError::IntegerConversionError)?;
                    Ok(Self(unsigned << Self::FRAC_BITS))
                }
            }
        };
    }

    macro_rules! __idl_type_for_storage {
        (u64)  => { anchor_lang::idl::types::IdlType::U64 };
        (i64)  => { anchor_lang::idl::types::IdlType::I64 };
        (u128) => { anchor_lang::idl::types::IdlType::U128 };
        (i128) => { anchor_lang::idl::types::IdlType::I128 };
        (u32)  => { anchor_lang::idl::types::IdlType::U32 };
        (i32)  => { anchor_lang::idl::types::IdlType::I32 };
        (u16)  => { anchor_lang::idl::types::IdlType::U16 };
        (i16)  => { anchor_lang::idl::types::IdlType::I16 };
        (u8)   => { anchor_lang::idl::types::IdlType::U8 };
        (i8)   => { anchor_lang::idl::types::IdlType::I8 };
        ($other:tt) => {
            compile_error!("Unsupported storage type for Anchor IDL");
        };
    }

    pub(crate) use __gen_one_const;
    pub(crate) use __impl_rne_div_for_signedness;
    pub(crate) use __impl_signed_fixed_point_ops;
    pub(crate) use __impl_fixed_point_from_base_int;
    #[cfg(feature = "idl-build")]
    pub(crate) use __idl_type_for_storage;
}

fixed_point! {
    /// Unsigned Q1.63 fixed-point numerical type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q1.63 value represented as a u64
    /// 
    /// ## Notes
    /// 
    /// * Uses a u128 intermediate type for multiplication and division
    /// * 1 integer bit, 63 fractional bits
    /// * Range: integer = [0, 2), fractional resolution = 2^-63 ≈ 1.0842e-19
    pub struct Q1x63(u64, u128, 63, false, None, true);
}

fixed_point! {
    /// Signed Q0.63 fixed-point numerical type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q0.63 value represented as a i64
    /// 
    /// ## Notes
    /// 
    /// * Uses a i128 intermediate type for multiplication and division
    /// * 0 integer bits (sign bit), 63 fractional bits
    /// * Range: integer = [-1, 1), fractional resolution = 2^-63 ≈ 1.0842 * 10^-19
    pub struct SQ0x63(i64, i128, 63, true, Q1x63, false);
}

fixed_point! {
    /// Unsigned Q2.62 fixed-point numerical type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q2.62 value represented as a u64
    /// 
    /// ## Notes
    /// 
    /// * Uses a u128 intermediate type for multiplication and division
    /// * 2 integer bits, 62 fractional bits
    /// * Range: integer = [0, 4), fractional resolution = 2^-62 ≈ 2.168 * 10^-19
    pub struct Q2x62(u64, u128, 62, false, None, true);
}

fixed_point! {
    /// Signed Q1.62 fixed-point numerical type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Signed Q1.62 value represented as a i64
    /// 
    /// ## Notes
    /// 
    /// * Uses a i128 intermediate type for multiplication and division
    /// * (sign bit) +1 integer bits, 62 fractional bits
    /// * Range: integer = [-2, 2), fractional resolution = 2^-62 ≈ 2.168 * 10^-19
    pub struct SQ1x62(i64, i128, 62, true, Q2x62, true);
}

fixed_point! {
    /// Unsigned Q64.64 fixed-point numerical type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a u128
    /// 
    /// ## Notes
    /// 
    /// * Uses a U256 intermediate type for multiplication and division
    /// * 64 integer bits, 64 fractional bits
    /// * Range: integer = [0, 2^64), fractional resolution = 2^-64 ≈ 5.421 * 10^-20
    pub struct Q64x64(u128, U256, 64, false, None, true);
}

fixed_point! {
    /// Unsigned Q64.64 fixed-point numerical type, used in places where widening to U256 is not desired
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q64.64 value represented as a u128
    /// 
    /// ## Notes
    /// 
    /// * Uses a u128 intermediate type for multiplication and division
    /// * 64 integer bits, 64 fractional bits
    /// * Range: integer = [0, 2^64), fractional resolution = 2^-64 ≈ 5.421 * 10^-20
    pub struct ShortQ64x64(u128, u128, 64, false, None, true);
}

fixed_point! {
    /// Signed Q63.64 fixed-point numerical type
    /// 
    /// ## Fields
    ///
    /// * `0` - The Q63.64 value represented as a i128
    /// 
    /// ## Notes
    /// 
    /// * Uses a I256 intermediate type for multiplication and division
    /// * sign bit, 63 integer bits, 64 fractional bits
    /// * Range: integer = [-2^63, 2^63), fractional resolution = 2^-64 ≈ 5.421 * 10^-20
    pub struct SQ63x64(i128, I256, 64, true, Q64x64, true);
}

fixed_point! {
    /// Unsigned Q32.96 fixed-point numerical type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q32.96 value represented as a u128
    /// 
    /// ## Notes
    /// 
    /// * Uses U256 intermediate type for multiplication and division
    /// * 32 integer bits, 96 fractional bits
    /// * Range: integer = [0, 2^32), fractional resolution = 2^-96 ≈ 1.262 * 10^-29
    pub struct Q32x96(u128, U256, 96, false, None, true);
}

fixed_point! {
    /// Unsigned Q0.64 fixed-point numerical type
    /// 
    /// ## Fields
    /// 
    /// * `0` - The Q0.64 value represented as a u64
    /// 
    /// ## Notes
    /// 
    /// * Uses a u128 intermediate type for multiplication and division
    /// * 0 integer bits, 64 fractional bits
    /// * Range: integer = [0, 1), fractional resolution = 2^-64 ≈ 5.421 * 10^-20
    /// 
    /// This type is pretty cursed and can't represent 1.0, the maximum value is always one ULP below 1.0
    pub struct Q0x64(u64, u128, 64, false, None, false);
}

#[macro_export]
macro_rules! q64x64 {
    ($num:literal / $den:literal) => {{
        const S: u128 = 1u128 << 64;

        const N: u128 = $num as u128;
        const D: u128 = $den as u128;

        const VAL: u128 = ((N * S) + (D / 2)) / D;

        $crate::fixed_point::Q64x64::new(VAL)
    }};
    ($bp:literal bp) => {{
        const BP: u128 = $bp as u128;
        const S: u128 = 1u128 << 64;

        const VAL: u128 = (BP * S) / 10_000;

        $crate::fixed_point::Q64x64::new(VAL)
    }};
    ($int:tt) => {{
        const S: u128 = 1u128 << 64;

        const VAL: u128 = ($int as u128) * S;

        $crate::fixed_point::Q64x64::new(VAL)
    }};
}

pub use q64x64;

impl core::convert::From<Q0x64> for Q64x64 {
    #[inline]
    fn from(value: Q0x64) -> Self {
        let raw = value.into_raw();
        Q64x64::new(raw as u128)
    }
}

impl core::convert::From<Q32x96> for Q64x64 {
    #[inline]
    fn from(value: Q32x96) -> Self {
        let raw = value.into_raw() >> 32; // trim from 96 to 64 frac bits
        Q64x64::new(raw)
    }
}

impl core::convert::From<Q1x63> for Q64x64 {
    #[inline]
    fn from(value: Q1x63) -> Self {
        let raw = (value.into_raw() as u128) << 1;
        Q64x64::new(raw)
    }
}

impl From<ShortQ64x64> for Q64x64 {
    #[inline]
    fn from(value: ShortQ64x64) -> Self {
        Q64x64::new(value.into_raw())
    }
}

impl From<Q64x64> for ShortQ64x64 {
    #[inline]
    fn from(value: Q64x64) -> Self {
        ShortQ64x64::new(value.into_raw())
    }
}


impl core::convert::From<Q1x63> for SQ0x63 {
    #[track_caller]
    #[inline]
    fn from(value: Q1x63) -> Self {
        let raw = value.into_raw();
        
        if raw > i64::MAX as u64 {
            panic!("overflow converting Q1x63 to SQ0x63");
        }

        SQ0x63::new(raw as i64)
    }
}

impl core::convert::From<SQ0x63> for Q1x63 {
    #[track_caller]
    #[inline]
    fn from(value: SQ0x63) -> Self {
        let raw = value.into_raw();

        if raw.is_negative() {
            panic!("can not convert negative SQ0x63 to Q1x63");
        }
        
        Q1x63::new(raw as u64)
    }
}

impl core::convert::From<SQ1x62> for SQ63x64 {
    #[inline]
    fn from(value: SQ1x62) -> Self {
        let raw = (value.into_raw() as i128) << 2;
        SQ63x64::new(raw)
    }
}

impl core::convert::TryFrom<Q64x64> for Q1x63 {
    type Error = FixedPointError;
    
    #[inline]
    fn try_from(value: Q64x64) -> Result<Self, Self::Error> {
        // shr 1 to convert to 63 frac bits
        let raw: u64 = (value.into_raw() >> 1).try_into().map_err(|_| FixedPointError::IntegerConversionError)?;
        Ok(Q1x63::new(raw))
    }
}

impl core::convert::TryFrom<SQ63x64> for SQ1x62 {
    type Error = FixedPointError;
    
    #[inline]
    fn try_from(value: SQ63x64) -> Result<Self, Self::Error> {
        // shr 2 to convert to 62 frac bits
        let raw: i64 = (value.into_raw() >> 2).try_into().map_err(|_| FixedPointError::IntegerConversionError)?;
        Ok(SQ1x62::new(raw))
    }
}

impl core::convert::TryFrom<SQ63x64> for SQ0x63 {
    type Error = FixedPointError;

    #[inline]
    fn try_from(value: SQ63x64) -> Result<Self, Self::Error> {
        // shr 1 to convert to 63 frac bits
        let raw: i64 = (value.into_raw() >> 1).try_into().map_err(|_| FixedPointError::IntegerConversionError)?;
        Ok(SQ0x63::new(raw))
    }
}

impl TryFrom<Q64x64> for SQ63x64 {
    type Error = FixedPointError;

    #[inline]
    fn try_from(value: Q64x64) -> Result<Self, Self::Error> {
        let raw = value.into_raw().try_into().map_err(|_| FixedPointError::IntegerConversionError)?;
        Ok(SQ63x64::new(raw))
    }
}

impl core::convert::From<SQ63x64> for Q64x64 {
    #[track_caller]
    #[inline]
    fn from(value: SQ63x64) -> Self {
        let raw = value.into_raw();

        if raw.is_negative() {
            panic!("can not convert negative SQ63x64 to Q64x64");
        }

        Q64x64::new(raw as u128)
    }
}

impl core::convert::From<SQ1x62> for Q2x62 {
    #[track_caller]
    #[inline]
    fn from(value: SQ1x62) -> Self {
        let raw = value.into_raw();

        if raw.is_negative() {
            panic!("can not convert negative SQ1x62 to Q2x62");
        }

        Q2x62::new(raw as u64)
    }
}

impl core::convert::From<SQ0x63> for SQ63x64 {
    #[inline]
    fn from(value: SQ0x63) -> Self {
        let raw = (value.into_raw() as i128) << 1;
        SQ63x64::new(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consts_and_one() {
        assert_eq!(Q1x63::FRAC_BITS, 63);
        assert_eq!(Q1x63::ZERO.into_raw(), 0);
        assert_eq!(Q1x63::ONE.into_raw(), 1u64 << 63);
        assert_eq!(Q1x63::MAX.into_raw(), u64::MAX);
        assert_eq!(Q1x63::MIN.into_raw(), 0); // unsigned
    }

    #[test]
    fn from_x_scaled_u64_roundtrip_9dp() {
        // 1.500000000 (9dp) -> Q1.63 -> back to 9dp
        let a = Q1x63::from_x_scaled_u64(1_500_000_000, 9);
        // expected raw = ((value << 63) / 10^9)
        let expected = ((1_500_000_000u128 << 63) / 1_000_000_000u128) as u64;
        assert_eq!(a.into_raw(), expected);

        let test = Q64x64::from(100u64);

        println!("test: {:?}", test);

        let back = a.try_to_scaled_u64().unwrap();
        assert_eq!(back, 1_500_000_000);
    }

    #[test]
    fn from_x_scaled_u64_other_decimals() {
        let a = Q1x63::from_x_scaled_u64(1_234_000, 6);
        let back = a.try_to_x_scaled_u64(6).unwrap();

        assert_eq!(back, 1_234_000, "undesired truncation in conversion: {} != {}", back, 1_234_000);
    }

    #[test]
    fn mul_trunc_matches_formula() {
        // a = 1.5, b = 0.25
        let one   = Q1x63::ONE.into_raw();
        let half  = 1u64 << 62; // 0.5
        let quart = 1u64 << 61; // 0.25

        let a = Q1x63::new(one + half);
        let b = Q1x63::new(quart);

        let c = a * b;
        // expected: ((a*b) >> 63)
        let expected = (((a.into_raw() as u128) * (b.into_raw() as u128)) >> 63) as u64;
        assert_eq!(c.into_raw(), expected);

        // 1.5 * 0.25 = 0.375 = 3/8; raw = (3/8)*2^63 = 3 << 60
        assert_eq!(c.into_raw(), 3u64 << 60);
    }

    #[test]
    fn div_trunc_identities_in_range() {
        // pick values well within [0,2)
        let a = Q1x63::new((1u64 << 63) + (1u64 << 62)); // 1.5
        let one = Q1x63::ONE;
        assert_eq!((a / one).into_raw(), a.into_raw());

        // also verify formula on a/b < 2 case, e.g., 1.5 / 1.0 = 1.5
        let q = a / one;
        let expected = (((a.into_raw() as u128) << 63) / (one.into_raw() as u128)) as u64;
        assert_eq!(q.into_raw(), expected);
    }

    #[test]
    #[should_panic(expected = "division overflow")]
    fn div_trunc_overflow_at_two() {
        // 1.5 / 0.75 = 2.0 -> not representable in UQ1.63 (range [0,2))
        let a = Q1x63::new((1u64 << 63) + (1u64 << 62)); // 1.5
        let b = Q1x63::new(3u64 << 61);                  // 0.75
        let _ = a / b; // should panic narrowing to u64
    }

    #[test]
    fn square_trunc_in_range() {
        // 1.25^2 = 1.5625 < 2.0 -> safe
        let a = Q1x63::new((1u64 << 63) + (1u64 << 61)); // 1.25
        let s = a.square();
        let expected = (((a.into_raw() as u128) * (a.into_raw() as u128)) >> 63) as u64;
        assert_eq!(s.into_raw(), expected);
    }

    #[test]
    #[should_panic(expected = "square overflow")]
    fn square_trunc_overflow() {
        let a = Q1x63::new((1u64 << 63) + (1u64 << 62)); // 1.5 -> 2.25
        let _ = a.square();
    }

    #[test]
    #[should_panic(expected = "addition overflow")]
    fn add_overflow_panics() {
        // With explicit "panic on overflow" policy in Add
        let _ = Q1x63::MAX + Q1x63::ONE;
    }

    #[test]
    #[should_panic(expected = "subtraction overflow")]
    fn sub_underflow_panics() {
        let _ = Q1x63::ZERO - Q1x63::ONE;
    }

    #[test]
    fn saturating_add_and_sub() {
        assert_eq!(Q1x63::MAX.saturating_add(Q1x63::ONE).into_raw(), u64::MAX);
        assert_eq!(Q1x63::ZERO.saturating_sub(Q1x63::ONE).into_raw(), 0);
    }

    #[test]
    fn sq0x63_consts_and_one() {
        assert_eq!(SQ0x63::FRAC_BITS, 63);
        assert_eq!(SQ0x63::MAX.into_raw(), i64::MAX);
        assert_eq!(SQ0x63::MIN.into_raw(), i64::MIN);
        assert_eq!(SQ0x63::ZERO.into_raw(), 0);
    }

    #[test]
    fn sq0x63_range_edges() {
        // min = -1.0 exactly
        let min = SQ0x63::MIN;
        // Convert -1.0 to 9dp should be -1_000_000_000 (RNE exact)
        assert!(min.try_to_scaled_u64().is_err());
        // Use a value in range: -0.5
        let neg_half = SQ0x63::new(-(1i64 << 62));
        let nine = neg_half.try_to_x_scaled_u64(9);
        // negative to u64 should fail (can't represent negative in u64)
        assert!(nine.is_err());

        // +max to 9dp fits
        let almost_one = SQ0x63::MAX;
        let back = almost_one.try_to_x_scaled_u64(9).unwrap();
        // close to 1e9 - 1 ulp at 9dp after RNE; allow off-by-1 due to rounding boundary
        assert!(back <= 1_000_000_000);
        assert!(back >= 1_000_000_000 - 1);
    }

    #[test]
    fn sq0x63_from_integers_and_sign_ops() {
        let z = SQ0x63::try_from(0i32).unwrap();
        assert_eq!(z.into_raw(), 0);

        let one = SQ0x63::try_from(1i32).unwrap();
        assert_eq!(one.into_raw(), 1i64 << 63);

        let minus_one = SQ0x63::try_from(-1i32).unwrap();
        assert_eq!(minus_one.into_raw(), i64::MIN);

        // neg/abs
        let a = SQ0x63::new(-(1i64 << 62)); // -0.5
        assert!(a.is_negative());
        assert!(!a.is_positive());
        assert_eq!(a.abs().into_raw(), 1i64 << 62);

        // saturating/wrapping neg are available too
        assert_eq!(a.saturating_neg().into_raw(), 1i64 << 62);
        assert_eq!(a.wrapping_neg().into_raw(), 1i64 << 62);
    }

    #[test]
    #[should_panic(expected = "unary negation overflow")]
    fn sq0x63_neg_overflow_panics() {
        // -MIN would overflow for two's-complement signed
        let _ = -SQ0x63::MIN;
    }

    #[test]
    fn sq0x63_mul_div_square_small() {
        // 0.5 * 0.5 = 0.25
        let half = SQ0x63::new(1i64 << 62);
        let q = half * half;
        assert_eq!(q.into_raw(), 1i64 << 61);
    }

    #[test]
    #[should_panic(expected = "division overflow")]
    fn sq0x63_div_equal_halves_panics() {
        // 0.5 / 0.5 = 1.0, which is not representable in Q0.63
        let half = SQ0x63::new(1i64 << 62);
        let _ = half / half;
    }

    // ---------- SQ1x62 (signed Q2.62) ----------

    #[test]
    fn sq1x62_consts_and_range() {
        // Check FRAC_BITS and ONE
        assert_eq!(SQ1x62::FRAC_BITS, 62);
        assert_eq!(SQ1x62::ONE.into_raw(), 1i64 << 62);

        // Minimum value is -2.0 (i64::MIN)
        assert_eq!(SQ1x62::MIN.into_raw(), i64::MIN);

        // Maximum value is 2 − 2^-62 (i64::MAX)
        assert_eq!(SQ1x62::MAX.into_raw(), i64::MAX);

        // Roundtrip via scaled u64 at 9 decimals
        let near_two = SQ1x62::new(i64::MAX);
        let back = near_two.try_to_scaled_u64().unwrap();
        assert_eq!(back, 2_000_000_000); // RNE gives 2.000000000
    }

    #[test]
    fn sq1x62_from_scaled_and_roundtrip() {
        // 1.500000000 -> Q2.62 -> back to 9dp
        let a = SQ1x62::from_scaled_u64(1_500_000_000);
        let back = a.try_to_scaled_u64().unwrap();
        assert_eq!(back, 1_500_000_000);

        // From i32/i64/i128
        let one = SQ1x62::try_from(1i32).unwrap();
        assert_eq!(one.into_raw(), 1i64 << 62);
        let m1 = SQ1x62::try_from(-1i64).unwrap();
        assert_eq!(m1.into_raw(), -(1i64 << 62));
        let big = SQ1x62::try_from(0i128).unwrap();
        assert_eq!(big.into_raw(), 0);
    }

    #[test]
    fn sq1x62_arith_ok_and_overflow_paths() {
        // 1.5 * 0.25 = 0.375
        let one = SQ1x62::ONE.into_raw();
        let half = 1i64 << 61;  // 0.5
        let quart = 1i64 << 60; // 0.25
        let a = SQ1x62::new(one + half); // 1.5
        let b = SQ1x62::new(quart);
        let c = a * b;
        let expected = (((a.into_raw() as i128) * (b.into_raw() as i128)) >> 62) as i64;
        assert_eq!(c.into_raw(), expected);
        assert_eq!(c.into_raw(), 3i64 << 59); // 0.375 * 2^62

        // division identity within range
        let q = a / SQ1x62::new(one);
        assert_eq!(q.into_raw(), a.into_raw());
    }

    #[test]
    #[should_panic(expected = "multiplication overflow")]
    fn sq1x62_mul_overflow_panics() {
        // (2 - ε) * (2 - ε) ≈ 4 - … → > max representable (>=2) after >>FRAC_BITS
        let a = SQ1x62::new(i64::MAX);
        let _ = a * a;
    }

    #[test]
    #[should_panic(expected = "division overflow")]
    fn sq1x62_div_overflow_panics() {
        // 1.5 / 0.75 = 2.0 → narrowing overflow in i64 domain (raw == 2<<62)
        let a = SQ1x62::new((1i64 << 62) + (1i64 << 61)); // 1.5
        let b = SQ1x62::new(3i64 << 60);                  // 0.75
        let _ = a / b;
    }

    #[test]
    fn sq1x62_signed_helpers() {
        let x = SQ1x62::new(-(1i64 << 61)); // -0.5
        assert!(x.is_negative());
        assert_eq!(x.abs().into_raw(), 1i64 << 61);

        let y = SQ1x62::new(1i64 << 60); // +0.25
        assert!(y.is_positive());
        assert_eq!((-y).into_raw(), -(1i64 << 60));
    }

    // ---------- Q64x64 (unsigned) ----------

    #[test]
    fn q64x64_consts_and_conversions() {
        assert_eq!(Q64x64::FRAC_BITS, 64);
        assert_eq!(Q64x64::ZERO.into_raw(), 0u128);
        assert_eq!(Q64x64::ONE.into_raw(), 1u128 << 64);
        assert_eq!(Q64x64::MIN.into_raw(), 0u128);
        assert_eq!(Q64x64::MAX.into_raw(), u128::MAX);

        // from_x_scaled_u64 and roundtrip (1.5)
        let a = Q64x64::from_x_scaled_u64(1_500_000_000, 9);
        let back = a.try_to_x_scaled_u64(9).unwrap();
        assert_eq!(back, 1_500_000_000);

        // from_scaled_u64 shortcut
        let b = Q64x64::from_scaled_u64(750_000_000); // 0.75
        let back2 = b.try_to_scaled_u64().unwrap();
        assert_eq!(back2, 750_000_000);
    }

    #[test]
    fn q64x64_mul_div_square() {
        // 1.5 * 0.25 = 0.375
        let one   = Q64x64::ONE.into_raw();
        let half  = 1u128 << 63;
        let quart = 1u128 << 62;

        let a = Q64x64::new(one + half);
        let b = Q64x64::new(quart);
        let c = a * b;

        let expected = ((U256::from(a.into_raw()) * U256::from(b.into_raw())) >> 64).try_into().unwrap();
        assert_eq!(c.into_raw(), expected);
        assert_eq!(c.into_raw(), 3u128 << 61); // 0.375 * 2^64

        // divide identity
        let q = a / Q64x64::new(one);
        assert_eq!(q.into_raw(), a.into_raw());

        // square within range: 1.25^2 = 1.5625
        let a125 = Q64x64::new(one + (1u128 << 62));
        let s = a125.square();
        let exp = ((U256::from(a125.into_raw()) * U256::from(a125.into_raw())) >> 64).try_into().unwrap();
        assert_eq!(s.into_raw(), exp);
    }

    #[test]
    #[should_panic(expected = "square overflow")]
    fn q64x64_square_overflow_panics() {
        // (2 - ε)^2 overflows range after >>FRAC_BITS when narrowing to u128
        let near_two = Q64x64::MAX;
        let _ = near_two.square();
    }

    #[test]
    fn q64x64_bit_and_shift_ops_smoke() {
        let x = Q64x64::new(0xF0u128 << 64);
        let y = Q64x64::new(0x0Fu128 << 64);
        assert_eq!((x & y).into_raw(), 0u128);
        assert_eq!((x | y).into_raw(), (0xFFu128 << 64));
        assert_eq!((x ^ y).into_raw(), (0xFFu128 << 64));
        assert_eq!((!Q64x64::ZERO).into_raw(), u128::MAX);

        let mut z = Q64x64::ONE;
        z <<= 1u32;
        assert_eq!(z.into_raw(), 1u128 << 65);
        z >>= 1usize;
        assert_eq!(z.into_raw(), 1u128 << 64);
    }

    // ---------- SQ63x64 (signed) ----------

    #[test]
    fn sq63x64_consts_and_from_ints() {
        assert_eq!(SQ63x64::FRAC_BITS, 64);
        assert_eq!(SQ63x64::ONE.into_raw(), 1i128 << 64);
        assert_eq!(SQ63x64::MIN.into_raw(), i128::MIN);
        assert_eq!(SQ63x64::MAX.into_raw(), i128::MAX);

        // From ints (scales by << 64)
        assert_eq!(SQ63x64::try_from(0i32).unwrap().into_raw(), 0);
        assert_eq!(SQ63x64::try_from(1i32).unwrap().into_raw(), 1i128 << 64);
        assert_eq!(SQ63x64::try_from(-1i32).unwrap().into_raw(), -(1i128 << 64));
        assert_eq!(SQ63x64::try_from(5i64).unwrap().into_raw(), 5i128 << 64);
    }

    #[test]
    fn sq63x64_roundtrip_scaled() {
        // 1.500000000 -> Q64.64 -> back to 9dp
        let a = SQ63x64::from_scaled_u64(1_500_000_000);
        let back = a.try_to_scaled_u64().unwrap();
        assert_eq!(back, 1_500_000_000);

        // -0.25 → scaled_u64 is an error (negative to u64)
        let neg_quarter = SQ63x64::new(-(1i128 << 62));
        assert!(neg_quarter.try_to_scaled_u64().is_err());
    }

    #[test]
    fn sq63x64_arith_and_sign_helpers() {
        // 1.5 * 0.25 = 0.375
        let one   = SQ63x64::ONE.into_raw();
        let half  = 1i128 << 63;
        let quart = 1i128 << 62;

        let a = SQ63x64::new(one + half);
        let b = SQ63x64::new(quart);
        let c = a * b;
        let expected = ((I256::from(a.into_raw()) * I256::from(b.into_raw())) >> 64usize).try_into().unwrap();
        assert_eq!(c.into_raw(), expected);
        assert_eq!(c.into_raw(), 3i128 << 61);

        // division identity
        let q = a / SQ63x64::new(one);
        assert_eq!(q.into_raw(), a.into_raw());

        // signed helpers
        let neg = SQ63x64::new(-(1i128 << 60));
        assert!(neg.is_negative());
        assert_eq!(neg.abs().into_raw(), 1i128 << 60);
        assert_eq!((-neg).into_raw(), 1i128 << 60);
    }

    #[test]
    #[should_panic(expected = "unary negation overflow")]
    fn sq63x64_neg_overflow_panics() {
        // -MIN overflows
        let _ = -SQ63x64::MIN;
    }

    #[test]
    fn sq63x64_div_exact_two_ok() {
        // 1.5 / 0.75 = 2.0 (representable for i128 storage)
        let a = SQ63x64::new((1i128 << 64) + (1i128 << 63)); // 1.5
        let b = SQ63x64::new(3i128 << 62);                   // 0.75
        let q = a / b;
        assert_eq!(q.into_raw(), 2i128 << 64);
    }

    // ---------- Display ----------

    #[test]
    fn display_uses_scaled_u64() {
        // For unsigned type, Display should print the 9dp scaled integer
        let a = Q64x64::from_scaled_u64(42);
        assert_eq!(format!("{}", a), "42");

        // Signed positive prints as positive
        let b = SQ1x62::from_scaled_u64(123_456_789);
        assert_eq!(format!("{}", b), "123456789");
    }
}