/// Self-negation
pub trait Neg: core::ops::Neg<Output = Self> {}

impl<T> Neg for T where T: core::ops::Neg<Output = Self> {}

/// Addition and subtraction.
pub trait Additive:
    core::ops::Add<Self, Output = Self> + core::ops::Sub<Self, Output = Self> + Sized
{
}

impl<T> Additive for T where
    T: core::ops::Add<Self, Output = Self> + core::ops::Sub<Self, Output = Self>
{
}

/// Multiplication and division.
pub trait Multiplicative:
    core::ops::Mul<Self, Output = Self> + core::ops::Div<Self, Output = Self> + Sized
{
}

impl<T> Multiplicative for T where
    T: core::ops::Mul<Self, Output = Self> + core::ops::Div<Self, Output = Self>
{
}

pub trait MulAdd {
    /// Perform fused `(self * a) + b`.
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl<T> MulAdd for T
where
    T: num_traits::MulAdd<T, T, Output = T>,
{
    fn mul_add(self, a: Self, b: Self) -> Self {
        num_traits::MulAdd::mul_add(self, a, b)
    }
}

/// Arithmetic operations without assuming any identities.
pub trait Arithmetic: Additive + Multiplicative {}

impl<T> Arithmetic for T where T: Additive + Multiplicative {}

pub trait Exp {
    /// Raise `e` to the power of `self`.
    fn exp(self) -> Self;
}

impl<T> Exp for T
where
    T: num_traits::Float,
{
    fn exp(self) -> Self {
        num_traits::Float::exp(self)
    }
}
