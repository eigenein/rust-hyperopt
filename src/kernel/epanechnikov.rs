use crate::kernel::{Density, Sample};

/// [Standardized][1] Epanechnikov (parabolic) kernel.
///
/// [1]: https://stats.stackexchange.com/questions/187678/different-definitions-of-epanechnikov-kernel
#[derive(Copy, Clone, Debug)]
pub struct Epanechnikov;

macro_rules! impl_kernel {
    ($type:ty, $normalization:expr) => {
        impl Density<$type> for Epanechnikov {
            fn density(&self, at: $type) -> $type {
                // Scale to `-1..1`:
                let at = at / $normalization;

                // Calculate the density and normalize:
                0.75 / $normalization * at.mul_add(-at, 1.0)
            }
        }

        impl<RNG> Sample<$type, RNG> for Epanechnikov
        where
            RNG: crate::rand::Uniform<$type> + crate::rand::Uniform<bool>,
        {
            /// [Generate a sample][1] from the Epanechnikov kernel.
            ///
            /// [1]: https://stats.stackexchange.com/questions/173637/generating-a-sample-from-epanechnikovs-kernel
            fn sample(&self, rng: &mut RNG) -> $type {
                // Select the two smallest numbers of the three iid uniform samples:
                let (x1, x2) = min_2(
                    crate::rand::Uniform::uniform(rng),
                    crate::rand::Uniform::uniform(rng),
                    crate::rand::Uniform::uniform(rng),
                );

                // Randomly select one of the two smallest numbers:
                let x = if crate::rand::Uniform::<bool>::uniform(rng) {
                    x1
                } else {
                    x2
                };

                // Randomly invert it:
                let x: $type = if crate::rand::Uniform::<bool>::uniform(rng) {
                    x
                } else {
                    -x
                };

                // Scale to have a standard deviation of 1:
                x * $normalization
            }
        }
    };
}

impl_kernel!(f32, crate::consts::f32::SQRT5);
impl_kernel!(f64, crate::consts::f64::SQRT5);

/// Pick and return the two smallest numbers.
fn min_2<T: PartialOrd>(mut x1: T, mut x2: T, x3: T) -> (T, T) {
    // Ensure x1 <= x2:
    if x1 > x2 {
        (x1, x2) = (x2, x1);
    }

    // x1 is now one of the two smallest numbers. Pick the smallest among the other two:
    (x1, if x2 > x3 { x3 } else { x2 })
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn density_ok() {
        assert_abs_diff_eq!(
            Density::<f64>::density(&Epanechnikov, 0.0),
            0.335_410_196_624_968_46
        );
        assert_abs_diff_eq!(Density::<f64>::density(&Epanechnikov, 5.0_f64.sqrt()), 0.0);
        assert_abs_diff_eq!(
            Density::<f64>::density(&Epanechnikov, -(5.0_f64.sqrt())),
            0.0
        );
    }

    #[test]
    fn min_2_ok() {
        assert_eq!(min_2(1, 2, 3), (1, 2));
        assert_eq!(min_2(1, 3, 2), (1, 2));
        assert_eq!(min_2(2, 1, 3), (1, 2));
        assert_eq!(min_2(2, 3, 1), (2, 1));
        assert_eq!(min_2(3, 1, 2), (1, 2));
        assert_eq!(min_2(3, 2, 1), (2, 1));
    }
}
