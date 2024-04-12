use fastrand::Rng;

use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    kernel::{Density, Sample},
};

/// Normalized uniform kernel, also known as «boxcar function».
#[derive(Copy, Clone, Debug)]
pub struct Uniform;

impl<P, D> Density<P, D> for Uniform
where
    P: num_traits::Float,
    D: num_traits::Num,
{
    fn density(&self, at: P) -> D {
        if (P::from(-SQRT_3).unwrap()..=P::from(SQRT_3).unwrap()).contains(&at) {
            D::one()
        } else {
            D::zero()
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
