use std::ops::{Add, Div, Mul, RangeInclusive, Sub};

pub use self::component::Component;
use crate::{rand::UniformRange, Density, Sample};

mod component;

/// [Kernel density estimator][1].
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_density_estimation
#[derive(Copy, Clone, Debug)]
pub struct KernelDensityEstimator<C>(pub C);

impl<T, C, K> Density<T> for KernelDensityEstimator<C>
where
    T: Copy + Div<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + num_traits::Zero,
    C: Iterator<Item = Component<K, T>> + Clone,
    K: Density<T>,
    usize: Into<T>,
{
    /// Calculate the KDE's density at the specified point.
    fn density(&self, at: T) -> T {
        let (n_points, sum) = self
            .0
            .clone()
            .fold((0_usize, T::zero()), |(n, sum), component| {
                (n + 1, sum + component.density(at))
            });
        sum / n_points.into()
    }
}

impl<T, C, K, RNG> Sample<T, RNG> for KernelDensityEstimator<C>
where
    T: Add<T, Output = T> + Mul<T, Output = T>,
    C: Iterator<Item = Component<K, T>> + Clone,
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
            .clone()
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
    use std::iter;

    use super::*;
    use crate::{consts::f32::SQRT_3, kernel::Uniform};

    #[test]
    fn sample_single_component_ok() {
        let component = Component {
            kernel: Uniform,
            location: 0.0,
            bandwidth: 1.0,
        };
        let kde = KernelDensityEstimator(iter::once(component));
        let mut rng = fastrand::Rng::new();

        let sample = kde.sample(&mut rng);
        assert!((-SQRT_3..=SQRT_3).contains(&sample));

        // Ensure that the iterator can be reused.
        let _ = kde.sample(&mut rng);
    }
}
