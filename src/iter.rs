use std::{fmt::Debug, iter};

use crate::traits::loopback::SelfAdd;

/// Iterator over 3-tuple windows, including partial ones.
///
/// It's the need for partial windows that prompted me to implement the custom iterator.
#[derive(Copy, Clone, Debug)]
pub struct Triples<I, T>(I, Option<T>, Option<T>, Option<T>);

impl<I, T> Triples<I, T> {
    pub const fn new(inner: I) -> Self {
        Self(inner, None, None, None)
    }
}

impl<I, T> Iterator for Triples<I, T>
where
    I: Iterator<Item = T>,
    T: Copy,
{
    type Item = Triple<T>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.1, self.2, self.3) = (self.2, self.3, self.0.next());

        match (self.1, self.2, self.3) {
            (Some(left), None, None) => Some(Triple::Left(left)),
            (None, Some(middle), None) => Some(Triple::Middle(middle)),
            (Some(left), Some(middle), None) => Some(Triple::LeftMiddle(left, middle)),
            (Some(left), Some(middle), Some(right)) => Some(Triple::Full(left, middle, right)),
            (None, Some(middle), Some(right)) => Some(Triple::MiddleRight(middle, right)),
            (None, None, Some(right)) => Some(Triple::Right(right)),
            (None, None, None) => None,
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Triple<T> {
    Right(T),
    MiddleRight(T, T),
    Full(T, T, T),
    LeftMiddle(T, T),
    Left(T),
    Middle(T),
}

pub fn range_step_from<T>(start: T, step: T) -> impl Iterator<Item = T>
where
    T: Copy + SelfAdd,
{
    let mut item = start;
    iter::from_fn(move || {
        let yield_ = Some(item);
        item = item + step;
        yield_
    })
}

pub fn range_inclusive<T>(start: T, end: T) -> impl Iterator<Item = T>
where
    T: Copy + SelfAdd + num_traits::One + PartialOrd,
{
    range_step_from(start, T::one()).take_while(move |item| *item <= end)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::iter::Triple::*;

    #[test]
    fn range_step_from_ok() {
        assert_eq!(
            range_step_from(1, 2).take(5).collect::<Vec<_>>(),
            [1, 3, 5, 7, 9]
        );
    }

    #[test]
    fn range_inclusive_ok() {
        assert_eq!(range_inclusive(1, 5).collect::<Vec<_>>(), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn triples_from_empty_ok() {
        assert_eq!(Triples::new(iter::empty::<()>()).collect::<Vec<_>>(), []);
    }

    #[test]
    fn triples_from_one_ok() {
        assert_eq!(
            Triples::new(iter::once(1)).collect::<Vec<_>>(),
            [Right(1), Middle(1), Left(1)]
        );
    }

    #[test]
    fn triples_from_two_ok() {
        assert_eq!(
            Triples::new([1, 2].into_iter()).collect::<Vec<_>>(),
            [Right(1), MiddleRight(1, 2), LeftMiddle(1, 2), Left(2)]
        );
    }

    #[test]
    fn triples_from_three_ok() {
        assert_eq!(
            Triples::new([1, 2, 3].into_iter()).collect::<Vec<_>>(),
            [
                Right(1),
                MiddleRight(1, 2),
                Full(1, 2, 3),
                LeftMiddle(2, 3),
                Left(3)
            ]
        );
    }
}
