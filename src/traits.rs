//! Numeric traits and shortcuts.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// Implement a trait that accepts `Self` as the right-hand side argument and returns `Self`.
///
/// TODO: split into a separate module.
macro_rules! impl_self_trait {
    ($target_trait:ident, $source_trait:ident) => {
        pub trait $target_trait<Rhs = Self, Output = Self>:
            $source_trait<Rhs, Output = Output>
        {
        }

        impl<T, Rhs, Output> $target_trait<Rhs, Output> for T where
            T: $source_trait<Rhs, Output = Output>
        {
        }
    };
}

impl_self_trait!(SelfAdd, Add);
impl_self_trait!(SelfSub, Sub);
impl_self_trait!(SelfMul, Mul);
impl_self_trait!(SelfDiv, Div);

pub trait SelfNeg<Output = Self>: Neg<Output = Output> {}

impl<T, Output> SelfNeg<Output> for T where T: Neg<Output = Output> {}

/// Combine the traits into a single trait (work around unavailable [trait aliases][1]).
///
/// [1]: https://github.com/rust-lang/rust/issues/41517
///
/// TODO: split into a separate module.
macro_rules! impl_derived_trait {
    ($target_trait:ident, $source_trait:ident $(+$next_trait:ident)*) => {
        pub trait $target_trait<Rhs = Self, Output = Self>:
            $source_trait<Rhs, Output> $(+$next_trait<Rhs, Output>)*
        {}

        impl<T, Rhs, Output> $target_trait<Rhs, Output> for T
        where T:
            $source_trait<Rhs, Output> $(+$next_trait<Rhs, Output>)*
        {}
    };
}

impl_derived_trait!(Additive, SelfAdd + SelfSub);
impl_derived_trait!(Multiplicative, SelfMul + SelfDiv);

// TODO: `Constants` trait.
