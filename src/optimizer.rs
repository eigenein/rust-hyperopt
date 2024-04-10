use std::collections::BTreeSet;

use crate::{kde::Component, optimizer::trial::Trial};

mod trial;

/// ✨ Hyperparameter optimizer.
///
/// # Generic parameters
///
/// - [`K1`]: kernel type of the first (prior) estimator component
/// - [`KS`]: kernel type of the trials, it may (and will likely) be different from the prior component
/// - [`P`]: type of parameter that is optimized
/// - [`M`]: value of the target function, the less – the better
pub struct Optimizer<R, K1, P, M> {
    range: R,
    first_component: Component<K1, P>,
    cutoff: f64,
    n_candidates: usize,
    trials: BTreeSet<Trial<P, M>>,
}

impl<R, K1, P, M> Optimizer<R, K1, P, M> {
    /// Construct the new optimizer.
    ///
    /// Here begins your adventure!
    ///
    /// # Parameters
    ///
    /// - `range`: parameter range
    /// - `first_component`: your prior belief about which values of the searched parameter is more optimal
    pub const fn new(range: R, first_component: Component<K1, P>) -> Self {
        Self {
            range,
            first_component,
            trials: BTreeSet::new(),
            cutoff: 0.1,
            n_candidates: 25,
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

impl<R, K1, P: Ord, M: Ord> Optimizer<R, K1, P, M> {
    /// Provide the information about the trial, or in other words, «fit» the optimizer on the sample.
    ///
    /// Normally, you'll call your target function on parameters supplied by [`Optimizer::new_trial`],
    /// and feed back the results. But you also can feed it with any arbitrary parameters.
    ///
    /// # Parameters
    ///
    /// - `parameter`: the target function parameter
    /// - `metric`: the target function metric
    pub fn feed_back(&mut self, parameter: P, metric: M) {
        self.trials.insert(Trial { parameter, metric });
    }
}

impl<R, K1, P, M: Ord> Optimizer<R, K1, P, M> {
    /// Generate a parameter value for a new trial.
    ///
    /// After evaluating the target function with this parameter,
    /// you'd better feed the metric back with [`Optimizer::feed_back`].
    pub fn new_trial(&self) -> P {
        // Abandon hope, all ye who enter here!
        todo!()
    }
}
