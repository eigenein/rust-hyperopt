use crate::kernel::{Density, Sample};

/// [Gaussian][1] kernel.
///
/// [1]: https://en.wikipedia.org/wiki/Normal_distribution
#[derive(Copy, Clone, Debug)]
pub struct Gaussian;

macro_rules! impl_kernel {
    ($type:ty, $tau:expr, $frac_1_sqrt_tau:expr) => {
        impl Density<$type> for Gaussian {
            fn density(&self, at: $type) -> $type {
                $frac_1_sqrt_tau * (-0.5 * at * at).exp()
            }
        }

        impl<RNG> Sample<$type, RNG> for Gaussian
        where
            RNG: crate::rand::Uniform<$type>,
        {
            /// [Generate a sample][1] from the Gaussian kernel.
            ///
            /// [1]: https://en.wikipedia.org/wiki/Box–Muller_transform
            fn sample(&self, rng: &mut RNG) -> $type {
                let u1: $type = crate::rand::Uniform::uniform(rng);
                let u2: $type = crate::rand::Uniform::uniform(rng);
                (-2.0 * u1.ln()).sqrt() * ($tau * u2).cos()
            }
        }
    };
}

impl_kernel!(
    f32,
    std::f32::consts::TAU,
    crate::consts::f32::FRAC_1_SQRT_TAU
);
impl_kernel!(
    f64,
    std::f64::consts::TAU,
    crate::consts::f64::FRAC_1_SQRT_TAU
);

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
