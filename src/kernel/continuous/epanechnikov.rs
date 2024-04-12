use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use fastrand::Rng;

use crate::{
    consts::f64::SQRT_5,
    convert::UnsafeFromPrimitive,
    kernel::{Density, Sample},
};

/// [Standardized][1] Epanechnikov (parabolic) kernel.
///
/// [1]: https://stats.stackexchange.com/questions/187678/different-definitions-of-epanechnikov-kernel
#[derive(Copy, Clone, Debug)]
pub struct Epanechnikov;

impl<T> Density<T, T> for Epanechnikov
where
    T: Copy
        + Add<Output = T>
        + Debug
        + Div<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + PartialOrd
        + Sub<Output = T>
        + UnsafeFromPrimitive<f64>
        + num_traits::One
        + num_traits::Zero,
{
    fn density(&self, at: T) -> T {
        // Scale to `-1..1`:
        let at = at / T::unsafe_from_primitive(SQRT_5);
        if !(-T::one()..=T::one()).contains(&at) {
            // Return zero outside the valid interval.
            return T::zero();
        }

        // Calculate the density and normalize:
        T::unsafe_from_primitive(0.75 / SQRT_5) * (T::one() - at * at)
    }
}

impl<P> Sample<P> for Epanechnikov
where
    P: Mul<Output = P> + UnsafeFromPrimitive<f64>,
{
    /// [Generate a sample][1] from the Epanechnikov kernel.
    ///
    /// [1]: https://stats.stackexchange.com/questions/173637/generating-a-sample-from-epanechnikovs-kernel
    fn sample(&self, rng: &mut Rng) -> P {
        // Select the two smallest numbers of the three iid uniform samples:
        let (x1, x2) = min_2(rng.f64(), rng.f64(), rng.f64());

        // Randomly select one of the two smallest numbers:
        let x = if rng.bool() { x1 } else { x2 };

        // Randomly invert it:
        let x = if rng.bool() { x } else { -x };

        // Scale to have a standard deviation of 1:
        P::unsafe_from_primitive(x * SQRT_5)
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
        assert_abs_diff_eq!(
            Density::<f64, f64>::density(&Epanechnikov, 0.0),
            0.335_410_196_624_968_46
        );
        assert_abs_diff_eq!(
            Density::<f64, f64>::density(&Epanechnikov, 5.0_f64.sqrt()),
            0.0
        );
        assert_abs_diff_eq!(
            Density::<f64, f64>::density(&Epanechnikov, -(5.0_f64.sqrt())),
            0.0
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn density_outside_ok() {
        assert_eq!(Density::<f64, f64>::density(&Epanechnikov, -10.0), 0.0);
        assert_eq!(Density::<f64, f64>::density(&Epanechnikov, 10.0), 0.0);
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