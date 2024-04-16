use core::ops::{Add, Div, Mul, Neg, Sub};

/// Self-negation
pub trait SelfNeg<Output = Self>: Neg<Output = Output> {}

impl<T, Output> SelfNeg<Output> for T where T: Neg<Output = Output> {}

pub trait SelfMulAdd {
    /// Fused `(self * a) + b`.
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl<T> SelfMulAdd for T
where
    T: num_traits::MulAdd<T, T, Output = T>,
{
    fn mul_add(self, a: Self, b: Self) -> Self {
        num_traits::MulAdd::mul_add(self, a, b)
    }
}

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

pub trait SelfExp {
    fn exp(self) -> Self;
}

impl<T> SelfExp for T
where
    T: num_traits::Float,
{
    fn exp(self) -> Self {
        num_traits::Float::exp(self)
    }
}
