use std::{
    collections::{btree_set::Iter, BTreeSet},
    fmt::Debug,
    iter,
    iter::Copied,
    ops::Sub,
};

use crate::{
    iter::Triples,
    kde::{Component, KernelDensityEstimator},
};

/// Single trial in the optimizer.
///
/// Note that trials are ordered first by metric, and then by tag.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Trial<P, M> {
    pub metric: M,
    pub parameter: P,
}

/// Ordered collection of trials.
///
/// Here be dragons! 🐉 It basically maintains two inner collections:
///
/// - Set of trials (a pair of parameter and metric, ordered by metric): that allows tracking of
///   the best (worst) trials
/// - Set of parameters, ordered by parameter itself: that allows to estimate bandwidth for each trial
///
/// All this is for the sake of insertion and removal in `O(log n)` time.
///
/// The optimizer **should not** try the same parameter twice.
pub struct Trials<P, M> {
    by_metric: BTreeSet<Trial<P, M>>,
    by_parameter: BTreeSet<P>,
}

impl<P, M> Trials<P, M> {
    /// Instantiate a new empty trial collection.
    pub const fn new() -> Self {
        Self {
            by_metric: BTreeSet::new(),
            by_parameter: BTreeSet::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.by_parameter.len()
    }
}

impl<P, M> Trials<P, M> {
    /// Iterate parameters of the trials in ascending order.
    pub fn iter_parameters(&self) -> Copied<Iter<P>>
    where
        P: Copy,
    {
        self.by_parameter.iter().copied()
    }

    /// Push the trial to the collection.
    ///
    /// **Repetitive parameters will be ignored.**
    pub fn insert(&mut self, trial: Trial<P, M>)
    where
        P: Copy + Ord,
        M: Ord,
    {
        if self.by_parameter.insert(trial.parameter) {
            assert!(self.by_metric.insert(trial));
            assert_eq!(self.by_parameter.len(), self.by_metric.len());
        }
    }

    /// Pop the worst trial.
    pub fn pop_worst(&mut self) -> Option<Trial<P, M>>
    where
        P: Ord,
        M: Ord,
    {
        let worst_trial = self.by_metric.pop_last()?;
        assert!(self.by_parameter.remove(&worst_trial.parameter));
        Some(worst_trial)
    }

    /// Construct a [`KernelDensityEstimator`] from the trials.
    pub fn to_kde<'a, K>(
        &'a self,
        kernel: K,
    ) -> KernelDensityEstimator<impl Iterator<Item = Component<K, P>> + 'a>
    where
        P: Copy + Ord + Sub<P, Output = P>,
        K: Copy + 'a,
    {
        KernelDensityEstimator(
            iter::repeat(kernel)
                .zip(Triples::new(self.iter_parameters()))
                .filter_map(|(kernel, triple)| Component::from_triple(kernel, triple)),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizer::trial::{Trial, Trials};

    #[test]
    fn ordering_ok() {
        assert!(
            Trial {
                metric: 42,
                parameter: 1,
            } < Trial {
                metric: 43,
                parameter: 0,
            }
        );
    }

    #[test]
    fn trials_ok() {
        let mut trials = Trials::new();

        trials.insert(Trial {
            metric: 42,
            parameter: 1,
        });
        assert_eq!(trials.len(), 1);
        assert_eq!(trials.iter_parameters().collect::<Vec<_>>(), [1]);

        trials.insert(Trial {
            metric: 41,
            parameter: 2,
        });
        assert_eq!(trials.len(), 2);
        assert_eq!(trials.iter_parameters().collect::<Vec<_>>(), [1, 2]);

        assert_eq!(
            trials.pop_worst(),
            Some(Trial {
                metric: 42,
                parameter: 1
            })
        );
        assert_eq!(trials.len(), 1);
        assert_eq!(trials.iter_parameters().collect::<Vec<_>>(), [2]);
    }
}
