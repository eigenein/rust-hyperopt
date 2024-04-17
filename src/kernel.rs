//! Different [kernels][1] for [`crate::kde::KernelDensityEstimator`].
//!
//! These are used to model the «good» and «bad» parameter distributions.
//! One can always them separately, as well.
//!
//! [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)

use fastrand::Rng;

pub mod continuous;
pub mod discrete;
pub mod universal;

/// Density function.
pub trait Density {
    /// Parameter type.
    type Param;

    /// Output density value type.
    type Output;

    /// Calculate the density at the given point.
    #[must_use]
    fn density(&self, at: Self::Param) -> Self::Output;
}

/// Parameter sampler.
pub trait Sample {
    /// Sampled value type.
    ///
    /// It is called that because parameters are sampled for evaluation.
    type Param;

    /// Generate a random sample from the kernel.
    #[must_use]
    fn sample(&self, rng: &mut Rng) -> Self::Param;
}

/// Kernel builder for [`crate::kde::KernelDensityEstimator`].
///
/// Note that it does not directly correspond to the [mathematical definition][1],
/// as for example, it is responsible for its own shift and scaling.
/// This is useful for discrete kernels which do not normally have a bandwidth parameter `h`.
///
/// [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)
pub trait Kernel {
    type Param;

    /// Construct a kernel with the given location and bandwidth.
    #[must_use]
    fn new(location: Self::Param, std: Self::Param) -> Self;
}
