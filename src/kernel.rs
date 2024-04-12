//! Different [kernels][1] for [`crate::kde::KernelDensityEstimator`].
//!
//! These are used to model the «good» and «bad» parameter distributions.
//! One can always them separately, as well.
//!
//! [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)

pub mod continuous;
pub mod discrete;

use fastrand::Rng;

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
