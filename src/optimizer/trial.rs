use std::cmp::Ordering;

/// Single trial in the optimizer.
///
/// Note that trials are ordered first by metric, and then by tag.
#[derive(Debug)]
pub struct Trial<P, M> {
    pub metric: M,
    pub parameter: P,

    /// Workaround to store trials in a [`std::collections::BTreeSet`]:
    /// it is used to distinguish trials which yielded the same metric.
    pub tag: usize,
}

impl<P, M: Ord> Eq for Trial<P, M> {}

impl<P, M: Ord> PartialEq<Self> for Trial<P, M> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<P, M: Ord> PartialOrd<Self> for Trial<P, M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P, M: Ord> Ord for Trial<P, M> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.metric.cmp(&other.metric) {
            ordering @ (Ordering::Less | Ordering::Greater) => ordering,
            Ordering::Equal => self.tag.cmp(&other.tag),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizer::trial::Trial;

    #[test]
    fn ordering_ok() {
        assert!(Trial { metric: 42, parameter: 0, tag: 1 } < Trial { metric: 43, parameter: 0, tag: 0 });
    }

    #[test]
    fn equality_ok() {
        assert_ne!(Trial { metric: 42, parameter: 0, tag: 1 }, Trial { metric: 42, parameter: 0, tag: 2 });
        assert_eq!(Trial { metric: 42, parameter: 0, tag: 1 }, Trial { metric: 42, parameter: 1, tag: 1 });
    }
}
