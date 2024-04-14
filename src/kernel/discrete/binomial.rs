use std::fmt::Debug;

use fastrand::Rng;
use num_integer::binomial;
use num_iter::range_step_from;

use crate::{
    kernel::Kernel,
    traits::{Additive, Multiplicative},
    Density,
    Sample,
};

/// Discrete kernel function based on the [binomial distribution][1].
///
/// The probability mass function is normalized by dividing on the standard deviation.
///
/// [1]: https://en.wikipedia.org/wiki/Binomial_distribution
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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
        P: Copy + num_integer::Integer + Into<D>,
        D: num_traits::Float,
    {
        binomial(self.n, at).into()
            * self.p.powf(at.into())
            * (D::one() - self.p).powf((self.n - at).into())
    }

    /// Standard deviation: √(p * (1 - p) / n).
    fn std(&self) -> D
    where
        P: Copy + Into<D>,
        D: num_traits::Float,
    {
        (self.n.into() * self.p * (D::one() - self.p)).sqrt()
    }

    fn inverse_cdf(&self, cdf: D) -> P
    where
        P: Copy + Into<D> + num_integer::Integer,
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
    P: Copy + num_integer::Integer + Into<D>,
    D: num_traits::Float,
{
    fn density(&self, at: P) -> D {
        self.pmf(at) / self.std()
    }
}

impl<P, D> Sample<P> for Binomial<P, D>
where
    P: Copy + Into<D> + num_integer::Integer,
    D: num_traits::Float + num_traits::FromPrimitive,
{
    fn sample(&self, rng: &mut Rng) -> P {
        self.inverse_cdf(D::from_f64(rng.f64()).unwrap())
    }
}

impl<P, D> Kernel<P, D> for Binomial<P, D>
where
    Self: Density<P, D> + Sample<P>,
    P: Copy + Ord + MaxN + Additive + Multiplicative + Into<D> + num_traits::One,
    D: Multiplicative,
{
    fn new(location: P, bandwidth: P) -> Self {
        // Solving these for `p` and `n`:
        // Bandwidth: σ = √(p(1-p)/n)
        // Location: l = pn

        // Getting:
        // σ² = pn(1-p) = l(1-p)
        // 1-p = σ²/l
        // p = 1-(σ²/l)
        // n = l/p = l/(1-(σ²/l)) = l/((l-σ²)/l) = l²/(l-σ²)

        // Restrict bandwidth to avoid infinite `n` and/or negative `p`:
        let sigma_squared = (bandwidth * bandwidth).min(location - P::one());

        #[allow(clippy::suspicious_operation_groupings)]
        let n = (location * location / (location - sigma_squared)).clamp(P::one(), P::MAX_N);
        Self {
            n,
            p: Into::<D>::into(location) / Into::<D>::into(n),
        }
    }
}

pub trait MaxN {
    /// Maximum `n` such that there will be no overflow in [`binomial`].
    const MAX_N: Self;
}

impl MaxN for u8 {
    const MAX_N: Self = 10;
}

impl MaxN for i8 {
    const MAX_N: Self = 9;
}

impl MaxN for u16 {
    const MAX_N: Self = 18;
}

impl MaxN for i16 {
    const MAX_N: Self = 17;
}

impl MaxN for u32 {
    const MAX_N: Self = 34;
}

impl MaxN for i32 {
    const MAX_N: Self = 33;
}

impl MaxN for u64 {
    const MAX_N: Self = 67;
}

impl MaxN for i64 {
    const MAX_N: Self = 66;
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
        assert_abs_diff_eq!(Binomial { n: 1, p: 0.0 }.pmf(0), 1.0);
    }

    #[test]
    fn cdf_ok() {
        assert_eq!(Binomial { n: 20, p: 0.5 }.inverse_cdf(0.588), 10);
        assert_eq!(Binomial { n: 20, p: 0.5 }.inverse_cdf(0.020_694), 5);
        assert_eq!(Binomial { n: 1, p: 0.0 }.inverse_cdf(1.0), 0);
    }

    #[test]
    fn std_ok() {
        assert_abs_diff_eq!(Binomial { n: 20, p: 0.5 }.std(), 2.23607, epsilon = 0.00001);
    }

    #[test]
    fn new_ok() {
        let kernel = Binomial::<_, f64>::new(5, 2);
        assert_eq!(kernel.n, 25);
        assert_abs_diff_eq!(kernel.p, 0.2);
    }

    #[test]
    fn new_bandwidth_overflow() {
        let kernel = Binomial::<_, f64>::new(2, 100);
        assert_eq!(kernel.n, 4);
        assert_abs_diff_eq!(kernel.p, 0.5);
    }

    #[test]
    fn new_type_overflow() {
        let kernel = Binomial::<u32, f64>::new(20, 3);
        assert_eq!(kernel.n, u32::MAX_N);
    }
}
