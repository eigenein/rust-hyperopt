use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

use fastrand::Rng;

use crate::{
    consts::f64::DOUBLE_SQRT_3,
    convert::{UnsafeFromPrimitive, UnsafeInto},
    iter::Triple,
    kernel::Uniform,
    Density,
    Sample,
};

/// Single component of a [`crate::kde::KernelDensityEstimator`].
#[derive(Copy, Clone, Debug)]
#[must_use]
pub struct Component<K, P> {
    /// Kernel function.
    pub kernel: K,

    /// Center of the corresponding kernel.
    pub location: P,

    /// Bandwidth of the corresponding kernel.
    pub bandwidth: P,
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

impl<K, P, D> Density<P, D> for Component<K, P>
where
    K: Density<P, D>,
    P: Copy + Debug + Div<Output = P> + Sub<Output = P> + UnsafeInto<D>,
    D: Div<Output = D>,
{
    fn density(&self, at: P) -> D {
        self.kernel.density((at - self.location) / self.bandwidth) / self.bandwidth.unsafe_into()
    }
}

impl<K, P> Sample<P> for Component<K, P>
where
    K: Sample<P>,
    P: Add<Output = P> + Copy + Mul<Output = P>,
{
    fn sample(&self, rng: &mut Rng) -> P {
        self.kernel.sample(rng) * self.bandwidth + self.location
    }
}

impl<P> Component<Uniform, P> {
    /// Create a new component with the uniform kernel function.
    ///
    /// The kernel will be scaled and moved so that the «box» spans the specified range.
    #[allow(clippy::missing_panics_doc)]
    pub fn new(min: P, max: P) -> Self
    where
        P: Add<Output = P> + Copy + Div<Output = P> + Sub<Output = P> + UnsafeFromPrimitive<f64>,
    {
        Self {
            kernel: Uniform,
            location: (min + max) / P::unsafe_from_primitive(2.0),
            bandwidth: (max - min) / P::unsafe_from_primitive(DOUBLE_SQRT_3),
        }
    }
}
