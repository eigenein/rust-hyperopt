use core::ops::{Add, Div, Mul, Neg, Sub};

pub trait SelfNeg<Output = Self>: Neg<Output = Output> {}

impl<T, Output> SelfNeg<Output> for T where T: Neg<Output = Output> {}

/// Implement a trait that accepts `Self` as the right-hand side argument and returns `Self`.
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
