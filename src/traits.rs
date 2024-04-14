use core::ops::{Add, Div, Mul, Neg, Sub};

/// Shortcut for additive operations.
pub trait Additive<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

impl<T, Rhs, Output> Additive<Rhs, Output> for T where
    T: Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

/// Shortcut for multiplicative operations.
pub trait Multiplicative<Rhs = Self, Output = Self>:
    Mul<Rhs, Output = Output> + Div<Rhs, Output = Output>
{
}

impl<T, Rhs, Output> Multiplicative<Rhs, Output> for T where
    T: Mul<Rhs, Output = Output> + Div<Rhs, Output = Output>
{
}

/// Shortcut for a [rng][1] of numbers.
///
/// Do not confuse it with a random number generator.
///
/// [1]: https://en.wikipedia.org/wiki/Ring_(mathematics)#Variations_on_the_definition
#[deprecated = "some types do not implement `Neg`"]
pub trait NumRng<Rhs = Self, Output = Self>:
    Additive<Rhs, Output> + Multiplicative<Rhs, Output> + Neg<Output = Output>
{
}

impl<T, Rhs, Output> NumRng<Rhs, Output> for T where
    T: Additive<Rhs, Output> + Multiplicative<Rhs, Output> + Neg<Output = Output>
{
}

/// Shortcut for [ring][1] of numbers.
///
/// [1]: https://en.wikipedia.org/wiki/Ring_(mathematics)
pub trait NumRing<Rhs = Self, Output = Self>:
    NumRng<Rhs, Output> + num_traits::Zero + num_traits::One
{
}

impl<T, Rhs, Output> NumRing<Rhs, Output> for T where
    T: NumRng<Rhs, Output> + num_traits::Zero + num_traits::One
{
}
