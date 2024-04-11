use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    kernel::{Density, Sample},
    rand::Rand,
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

impl<T, RNG> Sample<T, RNG> for Uniform
where
    T: num_traits::Float,
    RNG: Rand<f64>,
{
    /// Generate a sample from the uniform kernel.
    fn sample(&self, rng: &mut RNG) -> T {
        T::from(rng.uniform())
            .unwrap()
            .mul_add(T::from(DOUBLE_SQRT_3).unwrap(), T::from(-SQRT_3).unwrap())
    }
}
