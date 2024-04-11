use crate::{
    consts::f64::SQRT_5,
    kernel::{Density, Sample},
    rand::Rand,
};

/// [Standardized][1] Epanechnikov (parabolic) kernel.
///
/// [1]: https://stats.stackexchange.com/questions/187678/different-definitions-of-epanechnikov-kernel
#[derive(Copy, Clone, Debug)]
pub struct Epanechnikov;

impl<T: num_traits::Float> Density<T> for Epanechnikov {
    fn density(&self, at: T) -> T {
        // Scale to `-1..1`:
        let at = at / T::from(SQRT_5).unwrap();

        // Calculate the density and normalize:
        T::from(0.75).unwrap() / T::from(SQRT_5).unwrap() * at.mul_add(-at, T::one())
    }
}

impl<T, RNG> Sample<T, RNG> for Epanechnikov
where
    T: num_traits::Float,
    RNG: Rand<f64> + Rand<bool>,
{
    /// [Generate a sample][1] from the Epanechnikov kernel.
    ///
    /// [1]: https://stats.stackexchange.com/questions/173637/generating-a-sample-from-epanechnikovs-kernel
    fn sample(&self, rng: &mut RNG) -> T {
        // Select the two smallest numbers of the three iid uniform samples:
        let (x1, x2) = min_2(
            T::from(Rand::<f64>::uniform(rng)).unwrap(),
            T::from(Rand::<f64>::uniform(rng)).unwrap(),
            T::from(Rand::<f64>::uniform(rng)).unwrap(),
        );

        // Randomly select one of the two smallest numbers:
        let x: T = if Rand::<bool>::uniform(rng) { x1 } else { x2 };

        // Randomly invert it:
        let x = if Rand::<bool>::uniform(rng) { x } else { -x };

        // Scale to have a standard deviation of 1:
        x * T::from(SQRT_5).unwrap()
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
