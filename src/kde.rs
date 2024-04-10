mod component;

use std::ops::{Add, Div, Mul, RangeInclusive, Sub};

pub use self::component::Component;
use crate::{rand::UniformRange, Density, Sample};

/// [Kernel density estimator][1].
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_density_estimation
#[derive(Copy, Clone, Debug)]
pub struct KernelDensityEstimator<I>(pub I);

impl<T, I, K> Density<T> for KernelDensityEstimator<I>
where
    T: Copy + Div<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + num_traits::Zero,
    I: IntoIterator<Item = Component<K, T>> + Copy,
    K: Density<T>,
    usize: Into<T>,
{
    /// Calculate the KDE's density at the specified point.
    fn density(&self, at: T) -> T {
        let (n_points, sum) = self
            .0
            .into_iter()
            .fold((0_usize, T::zero()), |(n, sum), component| {
                (n + 1, sum + component.density(at))
            });
        sum / n_points.into()
    }
}

impl<T, I, K, RNG> Sample<T, RNG> for KernelDensityEstimator<I>
where
    T: Add<T, Output = T> + Mul<T, Output = T>,
    I: IntoIterator<Item = Component<K, T>> + Copy,
    RNG: UniformRange<RangeInclusive<usize>, usize>,
    K: Sample<T, RNG>,
{
    /// Sample a random point from the KDE.
    ///
    /// The algorithm uses «[reservoir sampling][1]» to pick a random component,
    /// and then samples a point from that component.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Reservoir_sampling
    ///
    /// # Panics
    ///
    /// This method panics if called on an empty estimator.
    fn sample(&self, rng: &mut RNG) -> T {
        let component = self
            .0
            .into_iter()
            .enumerate()
            .filter(|(i, _)| rng.uniform_range(0..=*i) == 0)
            .last()
            .expect("KDE must have at least one component")
            .1;

        component.kernel.sample(rng) * component.bandwidth + component.location
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{consts::f32::SQRT_3, kernel::Uniform};

    #[test]
    fn sample_single_component_ok() {
        let component = Component {
            kernel: Uniform,
            location: 0.0,
            bandwidth: 1.0,
        };
        let sample = KernelDensityEstimator([component]).sample(&mut fastrand::Rng::new());
        assert!((-SQRT_3..=SQRT_3).contains(&sample));
    }
}
