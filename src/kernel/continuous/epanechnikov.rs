use std::fmt::Debug;

use fastrand::Rng;

use crate::{
    consts::f64::SQRT_5,
    kernel::{Density, Kernel, Sample},
    traits::{NumRing, NumRng},
};

/// [Standardized][1] Epanechnikov (parabolic) kernel, over (-√5, +√5) range.
///
/// [1]: https://stats.stackexchange.com/questions/187678/different-definitions-of-epanechnikov-kernel
#[derive(Copy, Clone, Debug)]
pub struct Epanechnikov<T> {
    location: T,
    bandwidth: T,
}

impl<T> Density<T, T> for Epanechnikov<T>
where
    T: Copy + Debug + PartialOrd + NumRing + num_traits::FromPrimitive,
{
    fn density(&self, at: T) -> T {
        // Scale to `-1..1`:
        let normalized = (at - self.location) / self.bandwidth / T::from_f64(SQRT_5).unwrap();
        if (-T::one()..=T::one()).contains(&normalized) {
            // Calculate the density and normalize:
            T::from_f64(0.75 / SQRT_5).unwrap() * (T::one() - normalized * normalized)
                / self.bandwidth
        } else {
            // Zero outside the valid interval:
            T::zero()
        }
    }
}

impl<T> Sample<T> for Epanechnikov<T>
where
    T: Copy + NumRng + num_traits::FromPrimitive,
{
    /// [Generate a sample][1] from the Epanechnikov kernel.
    ///
    /// [1]: https://stats.stackexchange.com/questions/173637/generating-a-sample-from-epanechnikovs-kernel
    fn sample(&self, rng: &mut Rng) -> T {
        // Select the two smallest numbers of the three iid uniform samples:
        let (x1, x2) = min_2(rng.f64(), rng.f64(), rng.f64());

        // Randomly select one of the two smallest numbers:
        let abs_normalized = if rng.bool() { x1 } else { x2 };

        // Randomly invert it:
        let normalized = if rng.bool() {
            abs_normalized
        } else {
            -abs_normalized
        };

        // Scale to have a standard deviation of 1:
        self.location + self.bandwidth * T::from_f64(normalized * SQRT_5).unwrap()
    }
}

impl<T> Kernel<T, T> for Epanechnikov<T>
where
    Self: Density<T, T> + Sample<T>,
    T: PartialOrd + num_traits::Zero,
{
    fn new(location: T, bandwidth: T) -> Self {
        debug_assert!(bandwidth > T::zero());
        Self {
            location,
            bandwidth,
        }
    }
}

impl<T> Default for Epanechnikov<T>
where
    T: num_traits::Zero + num_traits::One,
{
    fn default() -> Self {
        Self {
            location: T::zero(),
            bandwidth: T::one(),
        }
    }
}

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
    fn density_inside_ok() {
        let kernel = Epanechnikov::<f64>::default();
        assert_abs_diff_eq!(kernel.density(0.0), 0.335_410_196_624_968_46);
        assert_abs_diff_eq!(kernel.density(SQRT_5), 0.0);
        assert_abs_diff_eq!(kernel.density(-SQRT_5), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn density_outside_ok() {
        let kernel = Epanechnikov::<f64>::default();
        assert_eq!(kernel.density(-10.0), 0.0);
        assert_eq!(kernel.density(10.0), 0.0);
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
