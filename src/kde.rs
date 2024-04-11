use std::ops::RangeInclusive;

pub use self::component::Component;
use crate::{rand::UniformRange, Density, Sample};

mod component;

/// [Kernel density estimator][1].
///
/// # Type parameters
///
/// - [`C`]: iterator of KDE's components that are [`Density`] and [`Sample`].
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_density_estimation
#[derive(Copy, Clone, Debug)]
pub struct KernelDensityEstimator<C>(pub C);

impl<T, C> Density<T> for KernelDensityEstimator<C>
where
    T: Copy + num_traits::FromPrimitive + num_traits::Num,
    C: Iterator + Clone,
    C::Item: Density<T>,
{
    /// Calculate the KDE's density at the specified point.
    ///
    /// The method returns [`T::zero()`], if there are no components.
    fn density(&self, at: T) -> T {
        let (n_points, sum) = self
            .0
            .clone()
            .fold((0_usize, T::zero()), |(n, sum), component| {
                (n + 1, sum + component.density(at))
            });
        if n_points == 0 {
            T::zero()
        } else {
            sum / T::from_usize(n_points).unwrap()
        }
    }
}

impl<T, C, Rng> Sample<Option<T>, Rng> for KernelDensityEstimator<C>
where
    T: num_traits::Num,
    C: Iterator + Clone,
    C::Item: Sample<T, Rng>,
    Rng: UniformRange<RangeInclusive<usize>, usize>,
{
    /// Sample a random point from the KDE.
    ///
    /// The algorithm uses «[reservoir sampling][1]» to pick a random component,
    /// and then samples a point from that component.
    ///
    /// The method returns [`None`], if the estimator has no components.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Reservoir_sampling
    fn sample(&self, rng: &mut Rng) -> Option<T> {
        let sample = self
            .0
            .clone()
            .enumerate()
            .filter(|(i, _)| rng.uniform_range(0..=*i) == 0)
            .last()?
            .1
            .sample(rng);
        Some(sample)
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::{consts::f64::SQRT_3, kernel::Uniform};

    #[test]
    fn sample_single_component_ok() {
        let component = Component {
            kernel: Uniform,
            location: 0.0,
            bandwidth: 1.0,
        };
        let kde = KernelDensityEstimator(iter::once(component));
        let mut rng = fastrand::Rng::new();

        let sample = kde.sample(&mut rng).unwrap();
        assert!((-SQRT_3..=SQRT_3).contains(&sample));

        // Ensure that the iterator can be reused.
        let _ = kde.sample(&mut rng).unwrap();
    }
}
