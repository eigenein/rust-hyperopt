use std::{
    fmt::Debug,
    iter,
    ops::{Add, Div, Mul, Sub},
};

use fastrand::Rng;

use crate::{
    convert::{UnsafeFromPrimitive, UnsafeInto},
    kde::Component,
    optimizer::trial::{Trial, Trials},
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
#[derive(Debug)]
pub struct Optimizer<KInit, K, P, M> {
    min: P,
    max: P,
    init_component: Component<KInit, P>,
    kernel: K,
    cutoff: f64,
    n_candidates: usize,
    good_trials: Trials<P, M>,
    bad_trials: Trials<P, M>,
}

impl<KFirst, K, P, M> Optimizer<KFirst, K, P, M> {
    /// Construct the new optimizer.
    ///
    /// Here begins your adventure!
    ///
    /// # Parameters
    ///
    /// - `min` and `max`: parameter range, bounds are **included**
    /// - `init_component`: your prior belief about which values of the searched parameter is more optimal
    /// - `kernel`: kernel for the trial components
    pub const fn new(min: P, max: P, init_component: Component<KFirst, P>, kernel: K) -> Self {
        Self {
            min,
            max,
            init_component,
            kernel,
            cutoff: 0.1,
            n_candidates: 25,
            good_trials: Trials::new(),
            bad_trials: Trials::new(),
        }
    }

    /// Set the ratio of «good» trials.
    #[must_use]
    pub const fn cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Set the number of candidates to choose the next trial from the acquisition function[^1].
    ///
    /// [^1]: Acquisition function is basically a ratio
    ///       between the «good» KDE and «bad» KDE at the same point.
    #[must_use]
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
        clippy::cast_sign_loss,
        clippy::missing_panics_doc
    )]
    pub fn feed_back(&mut self, parameter: P, metric: M)
    where
        P: Copy + Debug + Ord,
        M: Debug + Ord,
    {
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
            .is_some_and(|worst_good_trial| metric <= worst_good_trial.metric)
        {
            // New trial is not worse than the worst good trial, so it belongs to the good trials:
            if self.good_trials.insert(Trial { metric, parameter }) {
                // Re-balance:
                while self.good_trials.len() > n_expected_good_trials {
                    self.bad_trials
                        .insert(self.good_trials.pop_worst().unwrap());
                }
            }
        }
        // Otherwise, it belongs to the bad trials:
        else if self.bad_trials.insert(Trial { metric, parameter }) {
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

    /// Generate a parameter value for a new trial.
    ///
    /// After evaluating the target function with this parameter,
    /// you'd better feed the metric back with [`Optimizer::feed_back`].
    ///
    /// # Type parameters
    ///
    /// - [`D`]: kernel density type
    #[allow(clippy::cast_precision_loss, clippy::missing_panics_doc)]
    pub fn new_trial<D>(&self, rng: &mut Rng) -> P
    where
        KFirst: Copy + Density<P, D> + Sample<P>,
        K: Copy + Density<P, D> + Sample<P>,
        P: Add<Output = P>
            + Copy
            + Debug
            + Div<Output = P>
            + Mul<Output = P>
            + Ord
            + Sub<Output = P>
            + UnsafeInto<D>,
        D: Add<Output = D>
            + Copy
            + Debug
            + Div<Output = D>
            + Mul<Output = D>
            + Ord
            + UnsafeFromPrimitive<usize>
            + num_traits::Zero,
    {
        // Abandon hope, all ye who enter here!
        // Okay… Slow breath in… and out…

        // First, construct the KDEs:
        let good_kde = self.good_trials.to_kde(self.kernel);
        let bad_kde = self.bad_trials.to_kde(self.kernel);

        // Now, sample candidates:
        let candidates = iter::from_fn(|| {
            if self.good_trials.len() < 2 || rng.usize(0..=self.good_trials.len()) == 0 {
                // Select from the first component, if the good KDE is empty or with probability `1 / (n + 1)`.
                Some(self.init_component.sample(rng))
            } else {
                // Select normally from the good KDE:
                Some(good_kde.sample(rng).expect("KDE should return a sample"))
            }
        });

        // Clamp them to the bounds:
        let valid_candidates =
            candidates.map(|parameter| num_traits::clamp(parameter, self.min, self.max));

        // Filter out tried ones:
        let new_candidates = valid_candidates
            .filter(|parameter| !self.good_trials.contains(parameter))
            .filter(|parameter| !self.bad_trials.contains(parameter));

        // Calculate the acquisition function:
        let evaluated_candidates = new_candidates
            .map(|parameter| {
                // Use weighted average of the initial component and KDE:
                let init_density = self.init_component.density(parameter);
                let l = (init_density
                    + good_kde.density(parameter)
                        * D::unsafe_from_primitive(self.good_trials.len()))
                    / D::unsafe_from_primitive(self.good_trials.len() + 1);
                debug_assert!(
                    l >= D::zero(),
                    "«good» density should not be negative: {l:?}"
                );
                let g = (init_density
                    + bad_kde.density(parameter) * D::unsafe_from_primitive(self.bad_trials.len()))
                    / D::unsafe_from_primitive(self.bad_trials.len() + 1);
                debug_assert!(
                    g >= D::zero(),
                    "«bad» density should not be negative: {g:?}"
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
        self.good_trials.best()
    }
}
