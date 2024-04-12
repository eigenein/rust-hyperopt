use std::fmt::Debug;

use fastrand::Rng;
use num_integer::binomial;
use num_iter::range_step_from;

use crate::{
    convert::{UnsafeFromPrimitive, UnsafeInto},
    Density,
    Sample,
};

/// Discrete kernel function based on the [binomial distribution][1].
///
/// The probability mass function is normalized by dividing on the standard deviation.
///
/// [1]: https://en.wikipedia.org/wiki/Binomial_distribution
#[derive(Copy, Clone, Debug)]
pub struct Binomial<P, D> {
    /// Number of independent experiments (distribution parameter).
    pub n: P,

    /// Experiment success rate (distribution parameter).
    pub p: D,
}

impl<P, D> Binomial<P, D> {
    /// Probability mass function.
    fn pmf(&self, at: P) -> D
    where
        P: Copy + num_integer::Integer + UnsafeInto<D>,
        D: num_traits::Float,
    {
        binomial(self.n, at).unsafe_into()
            * self.p.powf(at.unsafe_into())
            * (D::one() - self.p).powf((self.n - at).unsafe_into())
    }

    /// Standard deviation.
    fn std(&self) -> D
    where
        P: Copy + UnsafeInto<D>,
        D: num_traits::Float,
    {
        (self.n.unsafe_into() * self.p * (D::one() - self.p)).sqrt()
    }

    fn inverse_cdf(&self, cdf: D) -> P
    where
        P: Copy + Debug + UnsafeInto<D> + num_integer::Integer,
        D: Copy + num_traits::Float,
    {
        range_step_from(P::zero(), P::one())
            .scan(D::zero(), |acc, at| {
                *acc = *acc + self.pmf(at);
                Some((at, *acc))
            })
            .find(|(_, acc)| *acc >= cdf)
            .expect("there should be a next sample")
            .0
    }
}

impl<P, D> Density<P, D> for Binomial<P, D>
where
    P: Copy + num_integer::Integer + UnsafeInto<D>,
    D: num_traits::Float,
{
    fn density(&self, at: P) -> D {
        self.pmf(at) / self.std()
    }
}

impl<P, D> Sample<P> for Binomial<P, D>
where
    P: Copy + Debug + UnsafeInto<D> + num_integer::Integer,
    D: UnsafeFromPrimitive<f64> + num_traits::Float,
{
    fn sample(&self, rng: &mut Rng) -> P {
        self.inverse_cdf(D::unsafe_from_primitive(rng.f64()))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn pmf_ok() {
        assert_abs_diff_eq!(
            Binomial { n: 20, p: 0.5 }.pmf(10),
            0.176_197,
            epsilon = 0.000_001
        );
        assert_abs_diff_eq!(
            Binomial { n: 20, p: 0.5 }.pmf(5),
            0.014_786,
            epsilon = 0.000_001
        );
    }

    #[test]
    fn cdf_ok() {
        assert_eq!(Binomial { n: 20, p: 0.5 }.inverse_cdf(0.588), 10);
        assert_eq!(Binomial { n: 20, p: 0.5 }.inverse_cdf(0.020_694), 5);
    }

    #[test]
    fn std_ok() {
        assert_abs_diff_eq!(Binomial { n: 20, p: 0.5 }.std(), 2.23607, epsilon = 0.00001);
    }
}
