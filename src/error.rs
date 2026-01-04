// Copyright (c) 2025, Arcane Labs
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Clone, Copy)]
pub enum FixedPointError {
    IntegerConversionError,
    RangeError,
}