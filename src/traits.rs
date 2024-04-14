use core::ops::{Add, Div, Mul, Neg, Sub};

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

macro_rules! impl_derived_trait {
    ($target_trait:ident, $source_trait:ident $(+$source_traits:ident)*) => {
        pub trait $target_trait<Rhs = Self, Output = Self>:
            $source_trait<Rhs, Output> $(+$source_traits<Rhs, Output>)* {}

        impl<T, Rhs, Output> $target_trait<Rhs, Output> for T
        where T:
            $source_trait<Rhs, Output> $(+$source_traits<Rhs, Output>)*
        {}
    };
}

impl_derived_trait!(Additive, SelfAdd + SelfSub);
impl_derived_trait!(Multiplicative, SelfMul + SelfDiv);
