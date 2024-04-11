use std::{collections::BTreeSet, fmt::Debug};

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

impl<P: Copy + Ord, M: Ord> Trials<P, M> {
    /// Push the trial to the collection.
    ///
    /// **Repetitive parameters will be ignored.**
    pub fn insert(&mut self, trial: Trial<P, M>) {
        if self.by_parameter.insert(trial.parameter) {
            assert!(self.by_metric.insert(trial));
            assert_eq!(self.by_parameter.len(), self.by_metric.len());
        }
    }

    /// Pop the worst trial.
    pub fn pop_worst(&mut self) -> Option<Trial<P, M>> {
        let worst_trial = self.by_metric.pop_last()?;
        assert!(self.by_parameter.remove(&worst_trial.parameter));
        Some(worst_trial)
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizer::trial::Trial;

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
}
