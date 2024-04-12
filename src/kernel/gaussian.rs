use std::{f64::consts::TAU, fmt::Debug};

use fastrand::Rng;

use crate::{
    consts::f64::FRAC_1_SQRT_TAU,
    convert::UnsafeInto,
    kernel::{Density, Sample},
};

/// [Gaussian][1] kernel.
///
/// [1]: https://en.wikipedia.org/wiki/Normal_distribution
#[derive(Copy, Clone, Debug)]
pub struct Gaussian;

impl<P, D> Density<P, D> for Gaussian
where
    P: Debug + UnsafeInto<D> + num_traits::Float,
    f64: UnsafeInto<P>,
{
    fn density(&self, at: P) -> D {
        (UnsafeInto::<P>::unsafe_into(FRAC_1_SQRT_TAU)
            * (UnsafeInto::<P>::unsafe_into(-0.5) * at * at).exp())
        .unsafe_into()
    }
}

impl<P> Sample<P> for Gaussian
where
    f64: UnsafeInto<P>,
{
    /// [Generate a sample][1] from the Gaussian kernel.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Box–Muller_transform
    fn sample(&self, rng: &mut Rng) -> P {
        let u1 = rng.f64();
        let u2 = rng.f64();
        ((-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()).unsafe_into()
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
