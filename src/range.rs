use std::ops::RangeInclusive;

/// [`Copy`] implementation for ranges, since they do not want to include it into the
/// standard library 😕
pub trait CopyRange<T> {
    fn copy(&self) -> Self;
}

impl<T: Copy> CopyRange<T> for RangeInclusive<T> {
    fn copy(&self) -> Self {
        *self.start()..=*self.end()
    }
}
