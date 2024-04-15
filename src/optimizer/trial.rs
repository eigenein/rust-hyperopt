use std::{
    collections::{btree_set::Iter, BTreeSet},
    fmt::Debug,
    iter::Copied,
    ops::RangeInclusive,
};

use crate::{iter::Triples, kde::KernelDensityEstimator, kernel::Kernel, traits::Additive};

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
#[derive(Debug)]
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

    pub fn contains(&self, parameter: &P) -> bool
    where
        P: Ord,
    {
        self.by_parameter.contains(parameter)
    }

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
    ///
    /// # Returns
    ///
    /// [`true`], if the trial was inserted, and [`false`] if it was ignored as repetitive.
    pub fn insert(&mut self, trial: Trial<P, M>) -> bool
    where
        P: Copy + Ord,
        M: Ord,
    {
        if self.by_parameter.insert(trial.parameter) {
            assert!(self.by_metric.insert(trial));
            assert_eq!(self.by_parameter.len(), self.by_metric.len());
            true
        } else {
            false
        }
    }

    /// Retrieve the best trial.
    pub fn best(&self) -> Option<&Trial<P, M>>
    where
        P: Ord,
        M: Ord,
    {
        self.by_metric.first()
    }

    /// Retrieve the worst trial.
    pub fn worst(&self) -> Option<&Trial<P, M>>
    where
        P: Ord,
        M: Ord,
    {
        self.by_metric.last()
    }

    /// Pop the best trial.
    pub fn pop_best(&mut self) -> Option<Trial<P, M>>
    where
        P: Ord,
        M: Ord,
    {
        let best_trial = self.by_metric.pop_first()?;
        self.remove_parameter(&best_trial.parameter);
        Some(best_trial)
    }

    /// Pop the worst trial.
    pub fn pop_worst(&mut self) -> Option<Trial<P, M>>
    where
        P: Ord,
        M: Ord,
    {
        let worst_trial = self.by_metric.pop_last()?;
        self.remove_parameter(&worst_trial.parameter);
        Some(worst_trial)
    }

    /// Remove the parameter and ensure the variants.
    fn remove_parameter(&mut self, parameter: &P)
    where
        P: Ord,
        M: Ord,
    {
        assert!(self.by_parameter.remove(parameter));
        assert_eq!(self.by_parameter.len(), self.by_metric.len());
    }

    /// Construct a [`KernelDensityEstimator`] from the trials.
    pub fn to_kde<'a, K>(
        &'a self,
        bounds: RangeInclusive<P>,
    ) -> KernelDensityEstimator<impl Iterator<Item = K> + Clone + 'a>
    where
        P: Copy + Ord + Additive,
        K: Copy + Kernel<Param = P> + 'a,
    {
        KernelDensityEstimator(
            Triples::new(self.iter_parameters())
                .map(move |triple| K::from_triple(triple, *bounds.start()..=*bounds.end())),
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

        assert!(trials.insert(Trial {
            metric: 42,
            parameter: 1,
        }));
        assert_eq!(trials.len(), 1);
        assert_eq!(trials.iter_parameters().collect::<Vec<_>>(), [1]);

        assert!(trials.insert(Trial {
            metric: 41,
            parameter: 2,
        }));
        assert_eq!(trials.len(), 2);
        assert_eq!(trials.iter_parameters().collect::<Vec<_>>(), [1, 2]);

        assert!(!trials.insert(Trial {
            metric: 41,
            parameter: 2,
        }));

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
