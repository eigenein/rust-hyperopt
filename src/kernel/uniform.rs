use std::ops::{Add, Mul};
use fastrand::Rng;

use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    convert::UnsafeInto,
    kernel::{Density, Sample},
};

/// Normalized uniform kernel, also known as «boxcar function».
#[derive(Copy, Clone, Debug)]
pub struct Uniform;

impl<P, D> Density<P, D> for Uniform
where
    P: PartialOrd,
    D: num_traits::Zero + num_traits::One,
    f64: UnsafeInto<P>,
{
    fn density(&self, at: P) -> D {
        if ((-SQRT_3).unsafe_into()..=SQRT_3.unsafe_into()).contains(&at) {
            D::one()
        } else {
            D::zero()
        }
    }
}

impl<P> Sample<P> for Uniform
where
    P: Add<Output = P> + Mul<Output = P>,
    f64: UnsafeInto<P>,
{
    /// Generate a sample from the uniform kernel.
    fn sample(&self, rng: &mut Rng) -> P {
        (rng.f64() * DOUBLE_SQRT_3 - SQRT_3).unsafe_into()
    }
}
