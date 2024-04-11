use std::f64::consts::TAU;

use crate::{
    consts::f64::FRAC_1_SQRT_TAU,
    kernel::{Density, Sample},
    rand::Rand,
};

/// [Gaussian][1] kernel.
///
/// [1]: https://en.wikipedia.org/wiki/Normal_distribution
#[derive(Copy, Clone, Debug)]
pub struct Gaussian;

impl<T: num_traits::Float> Density<T> for Gaussian {
    fn density(&self, at: T) -> T {
        T::from(FRAC_1_SQRT_TAU).unwrap() * (T::from(-0.5).unwrap() * at * at).exp()
    }
}

impl<T, RNG> Sample<T, RNG> for Gaussian
where
    T: num_traits::Float,
    RNG: Rand<f64>,
{
    /// [Generate a sample][1] from the Gaussian kernel.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Box–Muller_transform
    fn sample(&self, rng: &mut RNG) -> T {
        let u1 = T::from(Rand::uniform(rng)).unwrap();
        let u2 = T::from(Rand::uniform(rng)).unwrap();
        (T::from(-2.0).unwrap() * u1.ln()).sqrt() * (T::from(TAU).unwrap() * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn density_ok() {
        assert_abs_diff_eq!(
            Density::<f64>::density(&Gaussian, 0.0),
            0.398_942_280_401_432_7,
        );
        assert_abs_diff_eq!(
            Density::<f64>::density(&Gaussian, 1.0),
            0.241_970_724_519_143_37,
        );
        assert_abs_diff_eq!(
            Density::<f64>::density(&Gaussian, -1.0),
            0.241_970_724_519_143_37,
        );
    }
}
