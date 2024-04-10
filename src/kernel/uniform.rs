use crate::kernel::{Density, Sample};

/// Uniform kernel, also known as «boxcar function».
#[derive(Copy, Clone, Debug)]
pub struct Uniform;

macro_rules! impl_kernel {
    ($type:ty) => {
        impl Density<$type> for Uniform {
            fn density(&self, at: $type) -> $type {
                if (-1.0..=1.0).contains(&at) { 0.5 } else { 0.0 }
            }
        }

        impl<RNG> Sample<$type, RNG> for Uniform
        where
            RNG: crate::rand::Uniform<$type>,
        {
            /// Generate a sample from the uniform kernel.
            fn sample(&self, rng: &mut RNG) -> $type {
                crate::rand::Uniform::uniform(rng).mul_add(2.0, -1.0)
            }
        }
    };
}

impl_kernel!(f32);
impl_kernel!(f64);
