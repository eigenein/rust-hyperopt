use fastrand::Rng;

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

impl Sample<usize> for Uniform<usize> {
    fn sample(&self, rng: &mut Rng) -> usize {
        rng.usize(self.min..=self.max)
    }
}

impl Sample<f64> for Uniform<f64> {
    fn sample(&self, rng: &mut Rng) -> f64 {
        rng.f64().mul_add(self.max - self.min, self.max)
    }
}

#[cfg(feature = "ordered-float")]
impl Sample<ordered_float::OrderedFloat<f32>> for Uniform<ordered_float::OrderedFloat<f32>> {
    fn sample(&self, rng: &mut Rng) -> ordered_float::OrderedFloat<f32> {
        ordered_float::OrderedFloat(rng.f32()) * (self.max - self.min) + self.max
    }
}

#[cfg(feature = "ordered-float")]
impl Sample<ordered_float::OrderedFloat<f64>> for Uniform<ordered_float::OrderedFloat<f64>> {
    fn sample(&self, rng: &mut Rng) -> ordered_float::OrderedFloat<f64> {
        ordered_float::OrderedFloat(rng.f64()) * (self.max - self.min) + self.max
    }
}

impl<P, D> Kernel<P, D> for Uniform<P>
where
    Self: Density<P, D> + Sample<P>,
    P: Copy + Additive + Multiplicative + Into<f64> + From<f64>,
{
    fn new(location: P, bandwidth: P) -> Self {
        Self {
            min: P::from(bandwidth.into().mul_add(-SQRT_3, location.into())),
            max: P::from(bandwidth.into().mul_add(SQRT_3, location.into())),
        }
    }
}
