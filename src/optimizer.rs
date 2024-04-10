use std::collections::BTreeSet;
use crate::kde::Component;

use crate::optimizer::trial::Trial;

mod trial;

/// ✨ Hyperparameter optimizer.
///
/// # Generic parameters
///
/// - [`K1`]: kernel type of the first (prior) estimator component
/// - [`KS`]: kernel type of the trials, it may (and will likely) be different from the prior component
/// - [`P`]: type of parameter that is optimized
/// - [`M`]: value of the target function, the less – the better
pub struct Optimizer<K1, KS, P, M> {
    first_component: Component<K1, P>,
    cutoff: f64,
    n_candidates: usize,
    n_trials: usize,
    trials: BTreeSet<Trial<P, M>>,
}

impl<K1, KS, P, M> Optimizer<K1, KS, P, M> {
    /// Construct the new optimizer.
    ///
    /// Here begins your adventure!
    ///
    /// # Parameters
    ///
    /// - `first_component`: your prior belief about which values of the searched parameter is more optimal
    pub const fn new(first_component: Component<K1, P>) -> Self {
        Self {
            first_component,
            trials: BTreeSet::new(),
            cutoff: 0.1,
            n_candidates: 25,
            n_trials: 0,
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
}

impl<K1, KS, P, M: Ord> Optimizer<K1, KS, P, M> {
    /// Provide the information about the trial, or in other words, «fit» the optimizer on the sample.
    ///
    /// # Parameters
    ///
    /// - `parameter`: the target function parameter
    /// - `metric`: the target function metric
    pub fn feed_back(&mut self, parameter: P, metric: M) {
        self.trials.insert(Trial { parameter, metric, tag: self.n_trials });
        self.n_trials += 1;
    }
}
