use std::{marker::PhantomData, ops::RangeInclusive};

use fastrand::Rng;

use crate::{
    constants::{ConstDoubleSqrt3, ConstSqrt3},
    kernel::Kernel,
    traits::ops::{Additive, Arithmetic, Multiplicative, Neg},
    Density,
    Sample,
};

/// [Uniform][1] kernel, also known as «boxcar function», with normalized density.
///
/// [1]: https://en.wikipedia.org/wiki/Continuous_uniform_distribution#Probability_density_function
#[derive(Copy, Clone, Debug)]
pub struct Uniform<P, D> {
    min: P,
    max: P,
    _density: PhantomData<D>,
}

impl<P, D> Uniform<P, D>
where
    P: PartialOrd,
{
    /// Construct the kernel with specified **inclusive** bounds.
    pub fn with_bounds(range: RangeInclusive<P>) -> Self {
        let (min, max) = range.into_inner();
        Self {
            min,
            max,
            _density: PhantomData,
        }
    }
}

impl<P, D> Density for Uniform<P, D>
where
    P: Copy + Into<D> + PartialOrd + Additive,
    D: Multiplicative + num_traits::Zero + ConstDoubleSqrt3,
{
    type Param = P;
    type Output = D;

    fn density(&self, at: Self::Param) -> Self::Output {
        if (self.min..=self.max).contains(&at) {
            (self.max - self.min).into() / D::DOUBLE_SQRT_3
        } else {
            D::zero()
        }
    }
}

macro_rules! impl_sample_discrete {
    ($type:ident) => {
        impl<D> Sample for Uniform<$type, D> {
            type Param = $type;

            fn sample(&self, rng: &mut Rng) -> Self::Param {
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
        impl<D> Sample for Uniform<$type, D> {
            type Param = $type;

            fn sample(&self, rng: &mut Rng) -> Self::Param {
                rng.$type().mul_add(self.max - self.min, self.min)
            }
        }

        #[cfg(feature = "ordered-float")]
        impl<D> Sample for Uniform<ordered_float::OrderedFloat<$type>, D> {
            type Param = ordered_float::OrderedFloat<$type>;

            fn sample(&self, rng: &mut Rng) -> ordered_float::OrderedFloat<$type> {
                use num_traits::Float;
                ordered_float::OrderedFloat(rng.$type()).mul_add(self.max - self.min, self.min)
            }
        }

        #[cfg(feature = "ordered-float")]
        impl<D> Sample for Uniform<ordered_float::NotNan<$type>, D> {
            type Param = ordered_float::NotNan<$type>;

            fn sample(&self, rng: &mut Rng) -> ordered_float::NotNan<$type> {
                let normalized = unsafe { ordered_float::NotNan::new_unchecked(rng.$type()) };
                normalized * (self.max - self.min) + self.min
            }
        }
    };
}

impl_sample_continuous!(f32);
impl_sample_continuous!(f64);

impl<P, D> Kernel for Uniform<P, D>
where
    Self: Density<Param = P, Output = D> + Sample<Param = P>,
    P: Copy + Arithmetic + PartialOrd + num_traits::Zero + ConstSqrt3 + Neg,
{
    type Param = P;

    fn new(location: P, std: P) -> Self {
        assert!(std > P::zero());
        Self {
            min: std.mul_add(-P::SQRT_3, location),
            max: std.mul_add(P::SQRT_3, location),
            _density: PhantomData,
        }
    }
}
