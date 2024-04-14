use fastrand::Rng;
use num_traits::real::Real;

use crate::{
    consts::f64::{DOUBLE_SQRT_3, SQRT_3},
    kernel::Kernel,
    traits::{Additive, Multiplicative, SelfDiv},
    Density,
    Sample,
};

/// [Uniform][1] kernel, also known as «boxcar function», with normalized density.
///
/// [1]: https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Probability_density_function
#[derive(Copy, Clone, Debug)]
pub struct Uniform<P> {
    pub min: P,
    pub max: P,
}

impl<P, D> Density<P, D> for Uniform<P>
where
    P: Copy + Into<D> + PartialOrd + Additive,
    D: SelfDiv + num_traits::FromPrimitive + num_traits::Zero,
{
    fn density(&self, at: P) -> D {
        if (self.min..=self.max).contains(&at) {
            (self.max - self.min).into() / D::from_f64(DOUBLE_SQRT_3).unwrap()
        } else {
            D::zero()
        }
    }
}

macro_rules! impl_sample_discrete {
    ($type:ident) => {
        impl Sample<$type> for Uniform<$type> {
            fn sample(&self, rng: &mut Rng) -> $type {
                rng.$type(self.min..=self.max)
            }
        }
    };
}

impl_sample_discrete!(isize);
impl_sample_discrete!(usize);
impl_sample_discrete!(i8);
impl_sample_discrete!(u8);
impl_sample_discrete!(i16);
impl_sample_discrete!(u16);
impl_sample_discrete!(i32);
impl_sample_discrete!(u32);
impl_sample_discrete!(i64);
impl_sample_discrete!(u64);
impl_sample_discrete!(i128);
impl_sample_discrete!(u128);

macro_rules! impl_sample_continuous {
    ($type:ident) => {
        impl Sample<$type> for Uniform<$type> {
            fn sample(&self, rng: &mut Rng) -> $type {
                rng.$type().mul_add(self.max - self.min, self.min)
            }
        }

        #[cfg(feature = "ordered-float")]
        impl Sample<ordered_float::OrderedFloat<$type>>
            for Uniform<ordered_float::OrderedFloat<$type>>
        {
            fn sample(&self, rng: &mut Rng) -> ordered_float::OrderedFloat<$type> {
                ordered_float::OrderedFloat(rng.$type()).mul_add(self.max - self.min, self.min)
            }
        }

        #[cfg(feature = "ordered-float")]
        impl Sample<ordered_float::NotNan<$type>> for Uniform<ordered_float::NotNan<$type>> {
            fn sample(&self, rng: &mut Rng) -> ordered_float::NotNan<$type> {
                let normalized = unsafe { ordered_float::NotNan::new_unchecked(rng.$type()) };
                normalized * (self.max - self.min) + self.min
            }
        }
    };
}

impl_sample_continuous!(f32);
impl_sample_continuous!(f64);

impl<P, D> Kernel<P, D> for Uniform<P>
where
    Self: Density<P, D> + Sample<P>,
    P: Copy + Additive + Multiplicative + Into<f64> + From<f64> + PartialOrd + num_traits::Zero,
{
    fn new(location: P, bandwidth: P) -> Self {
        debug_assert!(bandwidth > P::zero());
        Self {
            min: P::from(bandwidth.into().mul_add(-SQRT_3, location.into())),
            max: P::from(bandwidth.into().mul_add(SQRT_3, location.into())),
        }
    }
}
