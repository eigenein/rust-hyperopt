use std::{f64::consts::TAU, fmt::Debug};

use fastrand::Rng;

use crate::{
    consts::f64::FRAC_1_SQRT_TAU,
    kernel::{Density, Sample},
};

/// [Gaussian][1] kernel.
///
/// [1]: https://en.wikipedia.org/wiki/Normal_distribution
#[derive(Copy, Clone, Debug)]
pub struct Gaussian;

impl<T> Density<T, T> for Gaussian
where
    T: Debug + num_traits::Float + num_traits::FromPrimitive,
{
    fn density(&self, at: T) -> T {
        T::from_f64(FRAC_1_SQRT_TAU).unwrap() * (T::from_f64(-0.5).unwrap() * at * at).exp()
    }
}

impl<P> Sample<P> for Gaussian
where
    P: num_traits::FromPrimitive,
{
    /// [Generate a sample][1] from the Gaussian kernel.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Box–Muller_transform
    fn sample(&self, rng: &mut Rng) -> P {
        let u1 = rng.f64();
        let u2 = rng.f64();
        P::from_f64((-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn density_ok() {
        assert_abs_diff_eq!(
            Density::<f64, f64>::density(&Gaussian, 0.0),
            0.398_942_280_401_432_7,
        );
        assert_abs_diff_eq!(
            Density::<f64, f64>::density(&Gaussian, 1.0),
            0.241_970_724_519_143_37,
        );
        assert_abs_diff_eq!(
            Density::<f64, f64>::density(&Gaussian, -1.0),
            0.241_970_724_519_143_37,
        );
    }
}
