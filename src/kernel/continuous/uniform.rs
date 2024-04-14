use fastrand::Rng;

use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    kernel::{Density, Kernel, Sample},
    traits::{Additive, SelfAdd, SelfDiv, SelfMul, SelfNeg, SelfSub},
};

/// Normalized uniform kernel, also known as «[boxcar function][1]», over (-σ√3, +σ√3) range.
///
/// [1]: https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Probability_density_function
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
        T: Copy + Additive + SelfDiv + num_traits::FromPrimitive,
    {
        Self {
            location: (min + max) / T::from_u8(2).unwrap(),
            bandwidth: (max - min) / T::from_f64(DOUBLE_SQRT_3).unwrap(),
        }
    }
}

impl<T> Density<T, T> for Uniform<T>
where
    T: Copy
        + PartialOrd
        + SelfSub
        + SelfDiv
        + SelfNeg
        + num_traits::One
        + num_traits::Zero
        + num_traits::FromPrimitive,
{
    fn density(&self, at: T) -> T {
        let normalized = (at - self.location) / self.bandwidth / T::from_f64(SQRT_3).unwrap();
        if (-T::one()..=T::one()).contains(&normalized) {
            T::from_f64(0.5 / SQRT_3).unwrap() / self.bandwidth
        } else {
            T::zero()
        }
    }
}

impl<T> Sample<T> for Uniform<T>
where
    T: Copy + SelfAdd + SelfMul + num_traits::FromPrimitive,
{
    /// Generate a sample from the uniform kernel.
    fn sample(&self, rng: &mut Rng) -> T {
        let normalized = rng.f64().mul_add(DOUBLE_SQRT_3, -SQRT_3);
        self.location + self.bandwidth * T::from_f64(normalized).unwrap()
    }
}

impl<T> Kernel<T, T> for Uniform<T>
where
    Self: Sample<T> + Density<T, T>,
    T: PartialOrd + num_traits::Zero,
{
    fn new(location: T, bandwidth: T) -> Self {
        debug_assert!(bandwidth > T::zero());
        Self {
            location,
            bandwidth,
        }
    }
}
