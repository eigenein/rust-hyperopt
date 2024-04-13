//! Different [kernels][1] for [`crate::kde::KernelDensityEstimator`].
//!
//! These are used to model the «good» and «bad» parameter distributions.
//! One can always them separately, as well.
//!
//! [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)

pub mod continuous;
pub mod discrete;

use fastrand::Rng;

use crate::{iter::Triple, traits::Additive};

/// Density function.
///
/// # Type parameters
///
/// - [`P`]: parameter type
/// - [`D`]: density type
pub trait Density<P, D> {
    /// Calculate density at the given point.
    #[must_use]
    fn density(&self, at: P) -> D;
}

/// Sample function.
///
/// # Type parameters
///
/// - [`P`]: parameter type
pub trait Sample<P> {
    /// Generate a random sample.
    #[must_use]
    fn sample(&self, rng: &mut Rng) -> P;
}

pub trait Kernel<P, D>: Density<P, D> + Sample<P> {
    /// Construct a kernel with the given location and threshold.
    #[must_use]
    fn new(location: P, bandwidth: P) -> Self;

    fn from_triple(triple: Triple<P>) -> Option<Self>
    where
        Self: Sized,
        P: Copy + Ord + Additive,
    {
        match triple {
            // For the middle point we take the maximum of the two distances:
            Triple::Full(left, location, right) => Some(Kernel::new(
                location,
                (right - location).max(location - left),
            )),

            Triple::LeftMiddle(left, location) => Some(Self::new(location, location - left)),

            Triple::MiddleRight(location, right) => Some(Self::new(location, right - location)),

            _ => None,
        }
    }
}
