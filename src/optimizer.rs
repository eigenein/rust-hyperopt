use std::{fmt::Debug, iter, ops::RangeInclusive};

use fastrand::Rng;
use num_traits::{FromPrimitive, One, Zero};

use crate::{
    iter::{Triple, Triples},
    kde::KernelDensityEstimator,
    kernel::Kernel,
    optimizer::trial::{Trial, Trials},
    range::CopyRange,
    traits::{
        loopback::{SelfMul, SelfSub},
        shortcuts::Multiplicative,
    },
    Density,
    Sample,
};

mod trial;

/// ✨ Hyperparameter optimizer.
///
/// # Generic parameters
///
/// - [`KInit`]: kernel type of the initial (prior) estimator component
/// - [`P`]: type of parameter that is optimized
/// - [`M`]: value of the target function, the less – the better
#[derive(Debug)]
pub struct Optimizer<KInit, P, M> {
    pub cutoff: f64,
    pub n_candidates: usize,
    pub bandwidth: P,

    range: RangeInclusive<P>,
    init_kernel: KInit,
    rng: Rng,
    good_trials: Trials<P, M>,
    bad_trials: Trials<P, M>,
}

impl<KInit, P, M> Optimizer<KInit, P, M> {
    /// Construct the new optimizer.
    ///
    /// Here begins your adventure!
    ///
    /// # Parameters
    ///
    /// - `range`: parameter search range, [`Optimizer`] will clamp random samples to this range
    /// - `init_kernel`: your prior belief about which values of the searched parameter is more optimal
    /// - `kernel`: kernel for the trial components
    pub fn new(range: RangeInclusive<P>, init_kernel: KInit, rng: Rng) -> Self
    where
        P: One,
    {
        Self {
            range,
            init_kernel,
            rng,
            cutoff: 0.1,
            n_candidates: 25,
            bandwidth: P::one(),
            good_trials: Trials::new(),
            bad_trials: Trials::new(),
        }
    }

    /// Set the ratio of «good» trials.
    #[must_use]
    pub fn cutoff(mut self, cutoff: impl Into<f64>) -> Self {
        self.cutoff = cutoff.into();
        self
    }

    /// Set the number of candidates to choose the next trial from the acquisition function[^1].
    ///
    /// Sampling from the acquisition function is cheaper than evaluating the target cost function,
    /// so the more candidates – the more effective is the optimization step.
    ///
    /// However, the acquisition function is an approximation of a potential gain,
    /// so the fewer candidates – the more precise is the optimization step.
    ///
    /// The number of candidates is therefore a tradeoff.
    ///
    /// [^1]: Acquisition function is basically a ratio
    ///       between the «good» KDE and «bad» KDE at the same point.
    #[must_use]
    pub fn n_candidates(mut self, n_candidates: impl Into<usize>) -> Self {
        self.n_candidates = n_candidates.into();
        self
    }

    /// Set the bandwidth multiplier for the estimator kernels.
    ///
    /// Standard deviation of the kernel is the distance from the point to its furthest neighbour,
    /// multiplied by this coefficient.
    ///
    /// The default multiplier is [`P::one`]. Lower bandwidth approximates the density better,
    /// however, is also prone to over-fitting. Higher bandwidth avoid over-fitting better,
    /// but is also smoother and less precise.
    #[must_use]
    pub fn bandwidth(mut self, bandwidth: impl Into<P>) -> Self {
        self.bandwidth = bandwidth.into();
        self
    }

    /// Provide the information about the trial, or in other words, «fit» the optimizer on the sample.
    ///
    /// Normally, you'll call your target function on parameters supplied by [`Optimizer::new_trial`],
    /// and feed back the results. But you also can feed it with any arbitrary parameters.
    ///
    /// # Parameters
    ///
    /// - `parameter`: the target function parameter
    /// - `metric`: the target function metric
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::missing_panics_doc
    )]
    pub fn feed_back(&mut self, parameter: P, metric: M)
    where
        P: Copy + Debug + Ord,
        M: Debug + Ord,
    {
        let trial = Trial { metric, parameter };
        let n_expected_good_trials = {
            // `+ 1` is for this new trial.
            let n_total_trials = self.good_trials.len() + self.bad_trials.len() + 1;
            (self.cutoff * n_total_trials as f64).round() as usize
        };

        // This uses an algorithm similar to the median tracking using two heaps,
        // only I'm tracking the `cutoff` quantile, and I have the B-tree sets instead of heaps.
        if self
            .good_trials
            .worst()
            .is_some_and(|worst_good_trial| &trial <= worst_good_trial)
        {
            // New trial is not worse than the worst good trial, so it belongs to the good trials:
            if self.good_trials.insert(trial) {
                // Re-balance:
                while self.good_trials.len() > n_expected_good_trials {
                    self.bad_trials
                        .insert(self.good_trials.pop_worst().unwrap());
                }
            }
        }
        // Otherwise, it belongs to the bad trials:
        else if self.bad_trials.insert(trial) {
            // Re-balance:
            while self.good_trials.len() < n_expected_good_trials {
                self.good_trials.insert(self.bad_trials.pop_best().unwrap());
            }
        }

        // Verify the invariant:
        #[cfg(debug_assertions)]
        {
            if let (Some(worst_good), Some(best_bad)) =
                (self.good_trials.worst(), self.bad_trials.best())
            {
                assert!(
                    worst_good <= best_bad,
                    "the worst good trial `{worst_good:?}` must not be worse than the best bad trial `{best_bad:?}`",
                );
            }
        }
    }

    /// Construct the kernel for the triple of adjacent trials.
    fn construct_kernel<K>(triple: Triple<P>, bounds: RangeInclusive<P>, bandwidth: P) -> K
    where
        K: Kernel<Param = P>,
        P: Copy + Ord + SelfSub + SelfMul,
    {
        match triple {
            Triple::Full(left, location, right) => {
                // For the middle point we take the maximum of the distances to the left and right neighbors:
                Kernel::new(
                    location,
                    bandwidth * (right - location).max(location - left),
                )
            }

            Triple::LeftMiddle(left, location) => {
                // For the left-middle pair: the maximum between them and to the right bound:
                K::new(
                    location,
                    bandwidth * (location - left).max(*bounds.end() - location),
                )
            }

            Triple::MiddleRight(location, right) => {
                // Similar, but to the left bound:
                K::new(
                    location,
                    bandwidth * (right - location).max(location - *bounds.start()),
                )
            }

            Triple::Left(location) | Triple::Middle(location) | Triple::Right(location) => {
                // Maximum between the distances to the bounds:
                K::new(
                    location,
                    bandwidth * (*bounds.end() - location).max(location - *bounds.start()),
                )
            }
        }
    }

    /// Construct a [`KernelDensityEstimator`] from the trials.
    fn construct_kde<K>(
        parameters: impl Iterator<Item = P> + Clone,
        bounds: RangeInclusive<P>,
        bandwidth: P,
    ) -> KernelDensityEstimator<impl Iterator<Item = K> + Clone>
    where
        P: Copy + Ord + SelfSub + SelfMul,
        K: Copy + Kernel<Param = P>,
    {
        KernelDensityEstimator(
            Triples::new(parameters)
                .map(move |triple| Self::construct_kernel(triple, bounds.copy(), bandwidth)),
        )
    }

    /// Generate a parameter value for a new trial.
    ///
    /// After evaluating the target function with this parameter,
    /// you'd better feed the metric back with [`Optimizer::feed_back`].
    ///
    /// # Type parameters
    ///
    /// - [`K`]: kernel type
    /// - [`D`]: kernel density type
    ///
    /// # Panics
    ///
    /// This method may panic if a random or calculated number cannot be converted to
    /// the parameter or density type.
    #[allow(clippy::cast_precision_loss)]
    pub fn new_trial<K>(&mut self) -> P
    where
        KInit: Copy + Density<Param = P> + Sample<Param = P>,
        KInit::Output: Copy + Debug + Ord + Multiplicative + FromPrimitive + Zero,
        K: Copy
            + Kernel<Param = P>
            + Sample<Param = P>
            + Density<Param = P, Output = <KInit as Density>::Output>,
        P: Copy + Ord + SelfSub + SelfMul,
    {
        // Abandon hope, all ye who enter here!
        // Okay… Slow breath in… and out…

        // First, construct the KDEs:
        let good_kde = Self::construct_kde::<K>(
            self.good_trials.iter_parameters(),
            self.range.copy(),
            self.bandwidth,
        );
        let bad_kde = Self::construct_kde::<K>(
            self.bad_trials.iter_parameters(),
            self.range.copy(),
            self.bandwidth,
        );

        // Now, sample candidates:
        let candidates = iter::from_fn(|| {
            if self.good_trials.len() < 2 || self.rng.usize(0..=self.good_trials.len()) == 0 {
                // Select from the first component, if the good KDE is empty or with probability `1 / (n + 1)`.
                Some(self.init_kernel.sample(&mut self.rng))
            } else {
                // Select normally from the good KDE:
                Some(
                    good_kde
                        .sample(&mut self.rng)
                        .expect("KDE should return a sample"),
                )
            }
        });

        // Clamp them to the bounds:
        let valid_candidates = candidates
            .map(|parameter| num_traits::clamp(parameter, *self.range.start(), *self.range.end()));

        // Filter out tried ones:
        let new_candidates = valid_candidates
            .filter(|parameter| !self.good_trials.contains(parameter))
            .filter(|parameter| !self.bad_trials.contains(parameter));

        // Calculate the acquisition function:
        let evaluated_candidates = new_candidates
            .map(|parameter| {
                // Use weighted average of the initial component and KDE:
                let init_density = self.init_kernel.density(parameter);
                let l = (init_density
                    + good_kde.density(parameter)
                        * K::Output::from_usize(self.good_trials.len()).unwrap())
                    / K::Output::from_usize(self.good_trials.len() + 1).unwrap();
                assert!(
                    l >= K::Output::zero(),
                    "«good» density should not be negative: {l:?}"
                );
                let g = (init_density
                    + bad_kde.density(parameter)
                        * K::Output::from_usize(self.bad_trials.len()).unwrap())
                    / K::Output::from_usize(self.bad_trials.len() + 1).unwrap();
                assert!(
                    g > K::Output::zero(),
                    "«bad» density should be positive: {g:?}"
                );
                (parameter, l / g)
            })
            .take(self.n_candidates);

        // Take the best one by the acquisition function value:
        evaluated_candidates
            .max_by_key(|(_, acquisition)| *acquisition)
            .expect("there should be at least one candidate")
            .0
    }

    /// Get the best trial.
    pub fn best_trial(&self) -> Option<&Trial<P, M>>
    where
        P: Ord,
        M: Ord,
    {
        self.good_trials.best().or_else(|| self.bad_trials.best())
    }
}
