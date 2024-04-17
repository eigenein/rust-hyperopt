use std::{fmt::Debug, iter::Sum};

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, One, Zero};

use crate::{
    iter::{range_inclusive, range_step_from},
    kernel::Kernel,
    traits::ops::{Additive, Arithmetic, Multiplicative},
    Density,
    Sample,
};

/// Discrete kernel function based on the [binomial distribution][1].
///
/// The probability mass function is normalized by dividing on the standard deviation.
///
/// Note that [`Binomial::density`] is a `O(at)` operation, so it's pretty slow.
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
        P: Copy + Into<D> + PartialOrd + Zero + One + Additive,
        D: Float + Sum,
    {
        if self.p == D::one() {
            // The only possible outcome is `at == n`:
            if at == self.n { D::one() } else { D::zero() }
        } else if self.p == D::zero() {
            // The only possible outcome is `at == 0`:
            if at == P::zero() { D::one() } else { D::zero() }
        } else if at <= self.n {
            // lg(n choose k) = Σ ln(n + 1 - i) - ln(i)
            let log_binomial: D = range_inclusive(P::one(), at)
                .map(|i| (self.n - at + i).into().ln() - i.into().ln())
                .sum();
            let log_pmf = log_binomial
                + at.into() * self.p.ln()
                + (self.n - at).into() * (D::one() - self.p).ln();
            log_pmf.exp()
        } else {
            // It is impossible to have more successes than experiments, hence the zero.
            D::zero()
        }
    }

    /// Standard deviation: √(p * (1 - p) / n).
    fn std(&self) -> D
    where
        P: Copy + Into<D>,
        D: Float,
    {
        (self.n.into() * self.p * (D::one() - self.p)).sqrt()
    }

    fn inverse_cdf(&self, cdf: D) -> P
    where
        P: Copy + Into<D> + One + Zero + PartialOrd + Additive,
        D: Copy + Float + Sum,
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

impl<P, D> Density for Binomial<P, D>
where
    P: Copy + Into<D> + Zero + PartialOrd + One + Additive,
    D: Float + Sum,
{
    type Param = P;
    type Output = D;

    fn density(&self, at: Self::Param) -> Self::Output {
        self.pmf(at) / self.std()
    }
}

impl<P, D> Sample for Binomial<P, D>
where
    P: Copy + Into<D> + One + Zero + PartialOrd + Additive,
    D: Float + FromPrimitive + Sum,
{
    type Param = P;

    fn sample(&self, rng: &mut Rng) -> Self::Param {
        self.inverse_cdf(D::from_f64(rng.f64()).unwrap())
    }
}

impl<P, D> Kernel for Binomial<P, D>
where
    Self: Density<Param = P, Output = D> + Sample<Param = P>,
    P: Copy + Ord + Arithmetic + Into<D> + One,
    D: Multiplicative,
{
    type Param = P;

    fn new(location: P, std: P) -> Self {
        // Solving these for `p` and `n`:
        // Bandwidth: σ = √(p(1-p)/n)
        // Location: l = pn

        // Getting:
        // σ² = pn(1-p) = l(1-p)
        // 1-p = σ²/l
        // p = 1-(σ²/l)
        // n = l/p = l/(1-(σ²/l)) = l/((l-σ²)/l) = l²/(l-σ²)

        // Restrict bandwidth to avoid infinite `n` and/or negative `p`:
        let sigma_squared = (std * std).min(location - P::one());

        #[allow(clippy::suspicious_operation_groupings)]
        let n = (location * location / (location - sigma_squared)).max(P::one());
        Self {
            n,
            p: Into::<D>::into(location) / Into::<D>::into(n),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn pmf_ok() {
        assert_abs_diff_eq!(Binomial { n: 5, p: 0.5 }.pmf(2), 0.3125);
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
        assert_abs_diff_eq!(Binomial { n: 20_u32, p: 0.5 }.pmf(21_u32), 0.0);
    }

    #[test]
    fn pmf_corner_cases() {
        assert_abs_diff_eq!(Binomial { n: 1, p: 0.0 }.pmf(0), 1.0);
        assert_abs_diff_eq!(Binomial { n: 1, p: 0.0 }.pmf(1), 0.0);
        assert_abs_diff_eq!(Binomial { n: 1, p: 1.0 }.pmf(0), 0.0);
        assert_abs_diff_eq!(Binomial { n: 1, p: 1.0 }.pmf(1), 1.0);
    }

    #[test]
    fn inverse_cdf_ok() {
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
}
