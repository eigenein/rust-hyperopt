use fastrand::Rng;

use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    kernel::{Density, Kernel, Sample},
    traits::{NumRing, NumRng},
};

/// Normalized uniform kernel, also known as «boxcar function», over (-√3, +√3) range.
#[derive(Copy, Clone, Debug)]
pub struct Uniform<T> {
    location: T,
    bandwidth: T,
}

impl<T> Uniform<T> {
    /// Create a new uniform kernel.
    ///
    /// The kernel will be scaled and moved so that the «box» spans the specified range.
    ///
    /// # Panics
    ///
    /// This function panics if location or bandwidth cannot be represented with the parameter type.
    pub fn new(min: T, max: T) -> Self
    where
        T: Copy + NumRng + num_traits::FromPrimitive,
    {
        Self {
            location: (min + max) / T::from_u8(2).unwrap(),
            bandwidth: (max - min) / T::from_f64(DOUBLE_SQRT_3).unwrap(),
        }
    }
}

impl<T> Density<T, T> for Uniform<T>
where
    T: Copy + PartialOrd + NumRing + num_traits::FromPrimitive,
{
    fn density(&self, at: T) -> T {
        let normalized = (at - self.location) / self.bandwidth / T::from_f64(SQRT_3).unwrap();
        if (-T::one()..=T::one()).contains(&normalized) {
            T::from_f64(1.0 / SQRT_3).unwrap() / self.bandwidth
        } else {
            T::zero()
        }
    }
}

impl<T> Sample<T> for Uniform<T>
where
    T: Copy + NumRng + num_traits::FromPrimitive,
{
    /// Generate a sample from the uniform kernel.
    fn sample(&self, rng: &mut Rng) -> T {
        let normalized = rng.f64().mul_add(2.0, -1.0);
        self.location + self.bandwidth * T::from_f64(normalized * SQRT_3).unwrap()
    }
}

impl<T> Kernel<T, T> for Uniform<T>
where
    T: Copy + PartialOrd + NumRing + num_traits::FromPrimitive,
{
    fn new(location: T, bandwidth: T) -> Self {
        debug_assert!(bandwidth > T::zero());
        Self {
            location,
            bandwidth,
        }
    }
}
