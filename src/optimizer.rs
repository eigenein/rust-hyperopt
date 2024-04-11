use std::ops::{Add, Mul, RangeInclusive, Sub};

use crate::{
    iter::Triples,
    kde::{Component, KernelDensityEstimator},
    optimizer::trial::{Trial, Trials},
    rand::UniformRange,
    Density,
    Sample,
};

mod trial;

/// ✨ Hyperparameter optimizer.
///
/// # Generic parameters
///
/// - [`KFirst`]: kernel type of the first (prior) estimator component
/// - [`K`]: kernel type of the trials, it may (and will likely) be different from the prior component
/// - [`P`]: type of parameter that is optimized
/// - [`M`]: value of the target function, the less – the better
pub struct Optimizer<R, KFirst, K, P, M> {
    range: R,
    first_component: Component<KFirst, P>,
    kernel: K,
    cutoff: f64,
    n_candidates: usize,
    good_trials: Trials<P, M>,
    bad_trials: Trials<P, M>,
}

impl<R, KFirst, K, P, M> Optimizer<R, KFirst, K, P, M> {
    /// Construct the new optimizer.
    ///
    /// Here begins your adventure!
    ///
    /// # Parameters
    ///
    /// - `range`: parameter range
    /// - `first_component`: your prior belief about which values of the searched parameter is more optimal
    /// - `kernel`: kernel for the trial components
    pub const fn new(range: R, first_component: Component<KFirst, P>, kernel: K) -> Self {
        Self {
            range,
            first_component,
            kernel,
            cutoff: 0.1,
            n_candidates: 25,
            good_trials: Trials::new(),
            bad_trials: Trials::new(),
        }
    }

    /// Set the ratio of «good» trials.
    pub const fn cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Set the number of candidates to choose the next trial from the acquisition function[^1].
    ///
    /// [^1]: Acquisition function is basically a ratio
    ///       between the «good» KDE and «bad» KDE at the same point.
    pub const fn n_candidates(mut self, n_candidates: usize) -> Self {
        self.n_candidates = n_candidates;
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
        clippy::cast_sign_loss
    )]
    pub fn feed_back(&mut self, parameter: P, metric: M)
    where
        P: Copy + Ord,
        M: Ord,
    {
        self.good_trials.insert(Trial { metric, parameter });

        // Balance the classes:
        let n_expected_good_trials = {
            let n_total_trials = self.good_trials.len() + self.bad_trials.len();
            (self.cutoff * n_total_trials as f64).round() as usize
        };
        while self.good_trials.len() > n_expected_good_trials {
            let worst_good_trial = self.good_trials.pop_worst().unwrap();
            self.bad_trials.insert(worst_good_trial);
        }
    }

    /// Generate a parameter value for a new trial.
    ///
    /// After evaluating the target function with this parameter,
    /// you'd better feed the metric back with [`Optimizer::feed_back`].
    ///
    /// Abandon hope, all ye who enter here!
    pub fn new_trial<Rng>(&self, mut rng: Rng) -> P
    where
        KFirst: Copy + Density<P> + Sample<P, Rng>,
        K: Copy + Density<P> + Sample<P, Rng>,
        P: Copy + Ord + Add<P, Output = P> + Mul<P, Output = P> + Sub<P, Output = P>,
        Rng: UniformRange<RangeInclusive<usize>, usize>,
    {
        // Okay… Slow breath in… and out…
        // First, construct the KDEs:
        let good_kde = self.good_trials.to_kde(self.kernel);
        let bad_kde = self.bad_trials.to_kde(self.kernel);
        todo!()
    }
}
