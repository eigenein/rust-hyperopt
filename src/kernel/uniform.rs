use crate::kernel::{Density, Sample};

/// Normalized uniform kernel, also known as «boxcar function».
#[derive(Copy, Clone, Debug)]
pub struct Uniform;

macro_rules! impl_kernel {
    ($type:ty,$sqrt_3:expr) => {
        impl Density<$type> for Uniform {
            fn density(&self, at: $type) -> $type {
                if (-$sqrt_3..=$sqrt_3).contains(&at) { 1.0 } else { 0.0 }
            }
        }

        impl<RNG> Sample<$type, RNG> for Uniform
        where
            RNG: crate::rand::Uniform<$type>,
        {
            /// Generate a sample from the uniform kernel.
            fn sample(&self, rng: &mut RNG) -> $type {
                const DOUBLE_SQRT_3: $type = 2.0 * $sqrt_3;
                crate::rand::Uniform::uniform(rng).mul_add(DOUBLE_SQRT_3, -$sqrt_3)
            }
        }
    };
}

impl_kernel!(f32, crate::consts::f32::SQRT_3);
impl_kernel!(f64, crate::consts::f64::SQRT_3);
