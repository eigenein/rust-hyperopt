mod epanechnikov;
mod gaussian;
mod uniform;

pub use self::{epanechnikov::Epanechnikov, gaussian::Gaussian, uniform::Uniform};

/// [Kernel][1] density function.
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)
pub trait Density<T> {
    /// Calculate density at the given point.
    #[must_use]
    fn density(&self, at: T) -> T;
}

/// Sampler from [kernel][1] function.
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)
pub trait Sample<T, RNG> {
    /// Generate a random sample.
    #[must_use]
    fn sample(&self, rng: &mut RNG) -> T;
}
