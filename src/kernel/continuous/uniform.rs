use std::ops::{Add, Mul};

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
    P: PartialOrd + num_traits::FromPrimitive,
    D: num_traits::Zero + num_traits::One,
{
    fn density(&self, at: P) -> D {
        if (P::from_f64(-SQRT_3).unwrap()..=P::from_f64(SQRT_3).unwrap()).contains(&at) {
            D::one()
        } else {
            D::zero()
        }
    }
}

impl<P> Sample<P> for Uniform
where
    P: Add<Output = P> + Mul<Output = P> + num_traits::FromPrimitive,
{
    /// Generate a sample from the uniform kernel.
    fn sample(&self, rng: &mut Rng) -> P {
        P::from_f64(rng.f64().mul_add(DOUBLE_SQRT_3, -SQRT_3)).unwrap()
    }
}
