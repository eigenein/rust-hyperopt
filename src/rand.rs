//! This crate, by its nature, uses a lot of random sampling.
//! While only [`fastrand`] is supported at the moment, the traits here make it possible
//! to add other random generators, for example, [`rand`].

use std::ops::RangeBounds;

/// Parameterless uniform distribution.
pub trait Rand<T> {
    fn uniform(&mut self) -> T;
}

impl Rand<bool> for fastrand::Rng {
    /// Generate a random boolean value with 50% probability of each class.
    #[inline]
    fn uniform(&mut self) -> bool {
        self.bool()
    }
}

impl Rand<f64> for fastrand::Rng {
    /// Generate a uniformly random [`f64`] in range `0..1`.
    #[inline]
    fn uniform(&mut self) -> f64 {
        self.f64()
    }
}

/// Uniform distribution over a range.
pub trait UniformRange<R, T> {
    fn uniform_range(&mut self, range: R) -> T;
}

impl<R: RangeBounds<usize>> UniformRange<R, usize> for fastrand::Rng {
    /// Generate a random [`usize`] in the specified range.
    #[inline]
    fn uniform_range(&mut self, range: R) -> usize {
        self.usize(range)
    }
}
