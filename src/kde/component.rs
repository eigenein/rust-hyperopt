use std::ops::Sub;

use fastrand::Rng;

use crate::{consts::f64::DOUBLE_SQRT_3, iter::Triple, kernel::Uniform, Density, Sample};

/// Single component of a [`crate::kde::KernelDensityEstimator`].
#[derive(Copy, Clone, Debug)]
#[must_use]
pub struct Component<K, T> {
    /// Kernel function.
    pub kernel: K,

    /// Center of the corresponding kernel.
    pub location: T,

    /// Bandwidth of the corresponding kernel.
    pub bandwidth: T,
}

impl<K, T> Component<K, T> {
    /// Construct a [`Component`] from a [`Triple`] of adjacent points.
    ///
    /// Kernel should be standardized because distances to the neighbors are used as bandwidths.
    pub fn from_triple(kernel: K, triple: Triple<T>) -> Option<Self>
    where
        T: Copy + Ord + Sub<T, Output = T>,
    {
        match triple {
            // For the middle point we take the maximum of the two distances:
            Triple::Full(left, location, right) => Some(Self {
                kernel,
                location,
                bandwidth: (right - location).max(location - left),
            }),

            Triple::LeftMiddle(left, location) => Some(Self {
                kernel,
                location,
                bandwidth: location - left,
            }),

            Triple::MiddleRight(location, right) => Some(Self {
                kernel,
                location,
                bandwidth: right - location,
            }),

            _ => None,
        }
    }
}

impl<K, T> Density<T> for Component<K, T>
where
    K: Density<T>,
    T: Copy + num_traits::Num,
{
    fn density(&self, at: T) -> T {
        self.kernel.density((at - self.location) / self.bandwidth) / self.bandwidth
    }
}

impl<K, T> Sample<T> for Component<K, T>
where
    K: Sample<T>,
    T: Copy + num_traits::Num,
{
    fn sample(&self, rng: &mut Rng) -> T {
        self.kernel.sample(rng) * self.bandwidth + self.location
    }
}

impl<T: num_traits::Float> Component<Uniform, T> {
    /// Create a new component with the uniform kernel function.
    ///
    /// The kernel will be scaled and moved so that the «box» spans the specified range.
    #[allow(clippy::missing_panics_doc)]
    pub fn new(min: T, max: T) -> Self {
        Self {
            kernel: Uniform,
            location: (min + max) / T::from(2.0).unwrap(),
            bandwidth: (max - min) / T::from(DOUBLE_SQRT_3).unwrap(),
        }
    }
}
