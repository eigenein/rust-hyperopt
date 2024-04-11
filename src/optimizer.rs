use std::{iter, ops::RangeInclusive};

use crate::{
    kde::Component,
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
/// - [`KInit`]: kernel type of the initial (prior) estimator component
/// - [`K`]: kernel type of the trials, it may (and will likely) be different from the prior component
/// - [`P`]: type of parameter that is optimized
/// - [`M`]: value of the target function, the less – the better
pub struct Optimizer<R, KInit, K, P, M> {
    range: R,
    init_component: Component<KInit, P>,
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
    /// - `init_component`: your prior belief about which values of the searched parameter is more optimal
    /// - `kernel`: kernel for the trial components
    pub const fn new(range: R, init_component: Component<KFirst, P>, kernel: K) -> Self {
        Self {
            range,
            init_component,
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
        P: Copy + Ord + From<usize> + num_traits::Num,
        Rng: UniformRange<RangeInclusive<usize>, usize>,
    {
        // Okay… Slow breath in… and out…
        // First, construct the KDEs:
        let good_kde = self.good_trials.to_kde(self.kernel);
        let bad_kde = self.bad_trials.to_kde(self.kernel);

        // Now, sample candidates:
        let candidates = iter::from_fn(|| {
            if self.good_trials.is_empty() || rng.uniform_range(0..=self.good_trials.len()) == 0 {
                // Select from the first component, if the good KDE is empty or with probability `1 / (n + 1)`.
                Some(self.init_component.sample(&mut rng))
            } else {
                // Select normally from the good KDE:
                Some(
                    good_kde
                        .sample(&mut rng)
                        .expect("non-empty KDE should return a sample"),
                )
            }
        });
        // Filter out tried ones:
        let new_candidates = candidates
            .filter(|parameter| !self.good_trials.contains(parameter))
            .filter(|parameter| !self.bad_trials.contains(parameter));
        // Calculate the acquisition function:
        let evaluated_candidates = new_candidates
            .map(|parameter| {
                // Use weighted average of the initial component and KDE:
                let init_density = self.init_component.density(parameter);
                let l = (init_density
                    + good_kde.density(parameter) * P::from(self.good_trials.len()))
                    / P::from(self.good_trials.len() + 1);
                let g = (init_density
                    + bad_kde.density(parameter) * P::from(self.bad_trials.len()))
                    / P::from(self.bad_trials.len() + 1);
                (parameter, l / g)
            })
            .take(self.n_candidates);
        // Take the best one by the acquisition function value:
        evaluated_candidates
            .max_by_key(|(_, acquisition)| *acquisition)
            .expect("there should be at least one candidate")
            .0
    }
}
