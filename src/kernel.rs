pub mod continuous;
pub mod discrete;

use fastrand::Rng;

/// [Kernel][1] density function.
///
/// # Type parameters
///
/// - [`P`]: parameter type
/// - [`D`]: density type
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)
pub trait Density<P, D> {
    /// Calculate density at the given point.
    #[must_use]
    fn density(&self, at: P) -> D;
}

/// Sampler from [kernel][1] function.
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)
pub trait Sample<T> {
    /// Generate a random sample.
    #[must_use]
    fn sample(&self, rng: &mut Rng) -> T;
}
