/// Self-negation
pub trait Neg<Output = Self>: core::ops::Neg<Output = Output> {}

impl<T, Output> Neg<Output> for T where T: core::ops::Neg<Output = Output> {}

/// Addition and subtraction.
pub trait Additive<Rhs = Self, Output = Self>:
    core::ops::Add<Rhs, Output = Output> + core::ops::Sub<Rhs, Output = Output>
{
}

impl<T, Rhs, Output> Additive<Rhs, Output> for T where
    T: core::ops::Add<Rhs, Output = Output> + core::ops::Sub<Rhs, Output = Output>
{
}

/// Multiplication and division.
pub trait Multiplicative<Rhs = Self, Output = Self>:
    core::ops::Mul<Rhs, Output = Output> + core::ops::Div<Rhs, Output = Output>
{
}

impl<T, Rhs, Output> Multiplicative<Rhs, Output> for T where
    T: core::ops::Mul<Rhs, Output = Output> + core::ops::Div<Rhs, Output = Output>
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
pub trait Arithmetic<Rhs = Self, Output = Self>:
    Additive<Rhs, Output> + Multiplicative<Rhs, Output> + MulAdd
{
}

impl<T, Rhs, Output> Arithmetic<Rhs, Output> for T where
    T: Additive<Rhs, Output> + Multiplicative<Rhs, Output> + MulAdd
{
}

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
