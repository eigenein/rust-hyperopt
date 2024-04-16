use crate::traits::loopback::{SelfAdd, SelfDiv, SelfMul, SelfSub};
/// Combine the traits into a single trait (work around unavailable [trait aliases][1]).
///
/// [1]: https://github.com/rust-lang/rust/issues/41517
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
