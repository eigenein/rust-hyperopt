use std::{
    fmt::Debug,
    ops::{Add, Div},
};

use fastrand::Rng;

pub use self::component::Component;
use crate::{convert::UnsafeInto, Density, Sample};

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

impl<P, D, C> Density<P, D> for KernelDensityEstimator<C>
where
    C: Iterator + Clone,
    C::Item: Density<P, D>,
    P: Copy,
    D: Add<Output = D> + Div<Output = D> + num_traits::Zero,
    f64: UnsafeInto<D>,
{
    /// Calculate the KDE's density at the specified point.
    ///
    /// The method returns [`P::zero()`], if there are no components.
    fn density(&self, at: P) -> D {
        let (n_points, sum) = self
            .0
            .clone()
            .fold((0_usize, D::zero()), |(n, sum), component| {
                (n + 1, sum + component.density(at))
            });
        if n_points == 0 {
            D::zero()
        } else {
            sum / (n_points as f64).unsafe_into()
        }
    }
}

impl<T, C> Sample<Option<T>> for KernelDensityEstimator<C>
where
    C: Iterator + Clone,
    C::Item: Sample<T>,
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
            .filter(|(i, _)| rng.usize(0..=*i) == 0)
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
        let mut rng = Rng::new();

        let sample = kde.sample(&mut rng).unwrap();
        assert!((-SQRT_3..=SQRT_3).contains(&sample));

        // Ensure that the iterator can be reused.
        let _ = kde.sample(&mut rng).unwrap();
    }
}
