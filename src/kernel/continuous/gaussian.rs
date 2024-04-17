use std::{f64::consts::TAU, fmt::Debug};

use fastrand::Rng;
use num_traits::FromPrimitive;

use crate::{
    constants::{ConstFrac1SqrtTau, ConstOneHalf},
    kernel::{Density, Kernel, Sample},
    traits::ops::{Arithmetic, Exp, Neg},
};

/// [Gaussian][1] kernel.
///
/// [1]: https://en.wikipedia.org/wiki/Normal_distribution
#[derive(Copy, Clone, Debug)]
pub struct Gaussian<T> {
    location: T,
    std: T,
}

impl<T> Density for Gaussian<T>
where
    T: Copy + ConstFrac1SqrtTau + Arithmetic + ConstOneHalf + Exp + Neg,
{
    type Param = T;
    type Output = T;

    fn density(&self, at: Self::Param) -> Self::Output {
        let normalized = (at - self.location) / self.std;
        T::FRAC_1_SQRT_TAU * (-T::ONE_HALF * normalized * normalized).exp() / self.std
    }
}

impl<T> Sample for Gaussian<T>
where
    T: Copy + Arithmetic + FromPrimitive,
{
    type Param = T;

    /// [Generate a sample][1] from the Gaussian kernel.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Box–Muller_transform
    fn sample(&self, rng: &mut Rng) -> Self::Param {
        let u1 = rng.f64();
        let u2 = rng.f64();
        let normalized = T::from_f64((-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()).unwrap();
        self.location + self.std * normalized
    }
}

impl<T> Kernel for Gaussian<T>
where
    Self: Density<Param = T, Output = T> + Sample<Param = T>,
    T: PartialOrd + num_traits::Zero,
{
    type Param = T;

    fn new(location: T, std: T) -> Self {
        assert!(std > T::zero());
        Self { location, std }
    }
}

impl<T> Default for Gaussian<T>
where
    T: num_traits::Zero + num_traits::One,
{
    /// Zero-centered standard normal distribution.
    fn default() -> Self {
        Self {
            location: T::zero(),
            std: T::one(),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn density_ok() {
        let kernel = Gaussian::default();
        assert_abs_diff_eq!(kernel.density(0.0), 0.398_942_280_401_432_7,);
        assert_abs_diff_eq!(kernel.density(1.0), 0.241_970_724_519_143_37,);
        assert_abs_diff_eq!(kernel.density(-1.0), 0.241_970_724_519_143_37,);
    }
}
