use std::ops::{Add, Mul};

use fastrand::Rng;

use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    convert::UnsafeFromPrimitive,
    kernel::{Density, Sample},
};

/// Normalized uniform kernel, also known as «boxcar function».
#[derive(Copy, Clone, Debug)]
pub struct Uniform;

impl<P, D> Density<P, D> for Uniform
where
    P: PartialOrd + UnsafeFromPrimitive<f64>,
    D: num_traits::Zero + num_traits::One,
{
    fn density(&self, at: P) -> D {
        if (P::unsafe_from_primitive(-SQRT_3)..=P::unsafe_from_primitive(SQRT_3)).contains(&at) {
            D::one()
        } else {
            D::zero()
        }
    }
}

impl<P> Sample<P> for Uniform
where
    P: Add<Output = P> + Mul<Output = P> + UnsafeFromPrimitive<f64>,
{
    /// Generate a sample from the uniform kernel.
    fn sample(&self, rng: &mut Rng) -> P {
        P::unsafe_from_primitive(rng.f64().mul_add(DOUBLE_SQRT_3, -SQRT_3))
    }
}
