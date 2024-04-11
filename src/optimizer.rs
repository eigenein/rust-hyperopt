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
/// - [`K1`]: kernel type of the first (prior) estimator component
/// - [`KS`]: kernel type of the trials, it may (and will likely) be different from the prior component
/// - [`P`]: type of parameter that is optimized
/// - [`M`]: value of the target function, the less – the better
pub struct Optimizer<R, K1, KS, P, M> {
    range: R,
    first_component: Component<K1, P>,
    kernel: KS,
    cutoff: f64,
    n_candidates: usize,
    good_trials: Trials<P, M>,
    bad_trials: Trials<P, M>,
}

impl<R, K1, KS, P, M> Optimizer<R, K1, KS, P, M> {
    /// Construct the new optimizer.
    ///
    /// Here begins your adventure!
    ///
    /// # Parameters
    ///
    /// - `range`: parameter range
    /// - `first_component`: your prior belief about which values of the searched parameter is more optimal
    /// - `kernel`: kernel for the trial components
    pub const fn new(range: R, first_component: Component<K1, P>, kernel: KS) -> Self {
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
    pub fn new_trial<RNG>(&self, mut rng: RNG) -> P
    where
        K1: Copy + Density<P> + Sample<P, RNG>,
        KS: Copy + Density<P> + Sample<P, RNG>,
        P: Copy + Ord + Add<P, Output = P> + Mul<P, Output = P> + Sub<P, Output = P>,
        RNG: UniformRange<RangeInclusive<usize>, usize>,
    {
        // Okay… Slow breath in… and out…
        // First, construct the KDEs:
        // TODO: I'd be happy to de-duplicate these, but the closure made me struggle to define the output type.
        let good_kde = KernelDensityEstimator(
            Triples::new(self.good_trials.iter_parameters())
                .filter_map(|triple| Component::from_triple(self.kernel, triple)),
        );
        let bad_kde = KernelDensityEstimator(
            Triples::new(self.bad_trials.iter_parameters())
                .filter_map(|triple| Component::from_triple(self.kernel, triple)),
        );
        todo!()
    }
}
