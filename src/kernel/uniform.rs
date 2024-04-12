use fastrand::Rng;

use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    kernel::{Density, Sample},
};

/// Normalized uniform kernel, also known as «boxcar function».
#[derive(Copy, Clone, Debug)]
pub struct Uniform;

impl<T: num_traits::Float> Density<T> for Uniform {
    fn density(&self, at: T) -> T {
        if (T::from(-SQRT_3).unwrap()..=T::from(SQRT_3).unwrap()).contains(&at) {
            T::one()
        } else {
            T::zero()
        }
    }
}

impl<T> Sample<T> for Uniform
where
    T: num_traits::Float,
{
    /// Generate a sample from the uniform kernel.
    fn sample(&self, rng: &mut Rng) -> T {
        T::from(rng.f64().mul_add(DOUBLE_SQRT_3, -SQRT_3)).unwrap()
    }
}
