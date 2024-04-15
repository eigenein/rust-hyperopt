use std::{f64::consts::TAU, fmt::Debug};

use fastrand::Rng;

use crate::{
    consts::f64::FRAC_1_SQRT_TAU,
    kernel::{Density, Kernel, Sample},
    traits::{SelfAdd, SelfMul},
};

/// [Gaussian][1] kernel.
///
/// [1]: https://en.wikipedia.org/wiki/Normal_distribution
#[derive(Copy, Clone, Debug)]
pub struct Gaussian<T> {
    location: T,
    bandwidth: T,
}

impl<T> Density for Gaussian<T>
where
    T: num_traits::Float + num_traits::FromPrimitive,
{
    type Param = T;
    type Output = T;

    fn density(&self, at: Self::Param) -> Self::Output {
        let normalized = (at - self.location) / self.bandwidth;
        T::from_f64(FRAC_1_SQRT_TAU).unwrap()
            * (T::from_f64(-0.5).unwrap() * normalized * normalized).exp()
            / self.bandwidth
    }
}

impl<T> Sample for Gaussian<T>
where
    T: Copy + SelfAdd + SelfMul + num_traits::FromPrimitive,
{
    type Param = T;

    /// [Generate a sample][1] from the Gaussian kernel.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Box–Muller_transform
    fn sample(&self, rng: &mut Rng) -> Self::Param {
        let u1 = rng.f64();
        let u2 = rng.f64();
        let normalized = T::from_f64((-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()).unwrap();
        self.location + self.bandwidth * normalized
    }
}

impl<T> Kernel for Gaussian<T>
where
    Self: Density<Param = T, Output = T> + Sample<Param = T>,
    T: PartialOrd + num_traits::Zero,
{
    type Param = T;

    fn new(location: T, bandwidth: T) -> Self {
        debug_assert!(bandwidth > T::zero());
        Self {
            location,
            bandwidth,
        }
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
            bandwidth: T::one(),
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
