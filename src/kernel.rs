//! Different [kernels][1] for [`crate::kde::KernelDensityEstimator`].
//!
//! These are used to model the «good» and «bad» parameter distributions.
//! One can always them separately, as well.
//!
//! [1]: https://en.wikipedia.org/wiki/Kernel_(statistics)

pub mod continuous;
pub mod discrete;
pub mod universal;

use fastrand::Rng;

use crate::{iter::Triple, traits::Additive};

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

/// A single kernel of a kernel density estimator.
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
    fn new(location: Self::Param, bandwidth: Self::Param) -> Self;

    /// Construct the kernel for the triple of adjacent trials.
    fn from_triple(triple: Triple<Self::Param>) -> Option<Self>
    where
        Self: Sized,
        Self::Param: Copy + Ord + Additive,
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
