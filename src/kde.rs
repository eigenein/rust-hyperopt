//! Kernel density estimator implementation.

use std::fmt::Debug;

use fastrand::Rng;
use num_traits::{FromPrimitive, Zero};

use crate::{traits::ops::Arithmetic, Density, Sample};

/// [Kernel density estimator][1].
///
/// It is used to model «good» and «bad» parameter distributions, but can also be used standalone.
///
/// # Type parameters
///
/// - [`C`]: iterator of KDE's components that are [`Density`] and [`Sample`].
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_density_estimation
#[derive(Copy, Clone, Debug)]
pub struct KernelDensityEstimator<C>(pub C);

impl<Ks> Density for KernelDensityEstimator<Ks>
where
    Ks: Iterator + Clone,
    Ks::Item: Density,
    <<Ks as Iterator>::Item as Density>::Param: Copy,
    <<Ks as Iterator>::Item as Density>::Output: Arithmetic + FromPrimitive + Zero,
{
    type Param = <<Ks as Iterator>::Item as Density>::Param;
    type Output = <<Ks as Iterator>::Item as Density>::Output;

    /// Calculate the KDE's density at the specified point.
    ///
    /// The method returns [`P::zero()`], if there are no components.
    #[allow(clippy::cast_precision_loss)]
    fn density(&self, at: Self::Param) -> Self::Output {
        let (n_points, sum) = self
            .0
            .clone()
            .fold((0_usize, Self::Output::zero()), |(n, sum), component| {
                (n + 1, sum + component.density(at))
            });
        if n_points == 0 {
            Self::Output::zero()
        } else {
            sum / Self::Output::from_usize(n_points).unwrap()
        }
    }
}

impl<Ks> Sample for KernelDensityEstimator<Ks>
where
    Ks: Iterator + Clone,
    Ks::Item: Sample,
{
    type Param = Option<<<Ks as Iterator>::Item as Sample>::Param>;

    /// Sample a random point from the KDE.
    ///
    /// The algorithm uses «[reservoir sampling][1]» to pick a random component,
    /// and then samples a point from that component.
    ///
    /// The method returns [`None`], if the estimator has no components.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Reservoir_sampling
    fn sample(&self, rng: &mut Rng) -> Self::Param {
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
    use crate::kernel::universal::Uniform;

    #[test]
    fn sample_single_component_ok() {
        let kernel: Uniform<_, ()> = Uniform::with_bounds(-1.0..=1.0);
        let kde = KernelDensityEstimator(iter::once(kernel));
        let mut rng = Rng::new();

        let sample = kde.sample(&mut rng).unwrap();
        assert!((-1.0..=1.0).contains(&sample));

        // Ensure that the iterator can be reused.
        let _ = kde.sample(&mut rng).unwrap();
    }
}
