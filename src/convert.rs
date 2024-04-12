use std::{any::type_name, fmt::Debug};

use num_traits::FromPrimitive;

pub trait UnsafeInto<T> {
    /// Convert the value to [`T`].
    ///
    /// # Panics
    ///
    /// The value is not convertable.
    fn unsafe_into(self) -> T;
}

impl<S, T> UnsafeInto<T> for S
where
    S: Copy + Debug + TryInto<T>,
    <S as TryInto<T>>::Error: Debug,
{
    fn unsafe_into(self) -> T {
        self.try_into().unwrap_or_else(|error| {
            panic!(
                "`{self:?}` should be convertable from `{}` to `{}`: {error:?}",
                type_name::<Self>(),
                type_name::<T>(),
            )
        })
    }
}

pub trait UnsafeFromPrimitive<T> {
    /// Convert the primitive value.
    ///
    /// # Panics
    ///
    /// The value is not convertable.
    fn unsafe_from_primitive(primitive: T) -> Self;
}

impl<T> UnsafeFromPrimitive<usize> for T
where
    T: FromPrimitive,
{
    fn unsafe_from_primitive(primitive: usize) -> Self {
        Self::from_usize(primitive).unwrap_or_else(|| {
            panic!(
                "usize `{primitive:?}` should be convertable to `{}`",
                type_name::<T>(),
            )
        })
    }
}

impl<T> UnsafeFromPrimitive<f64> for T
where
    T: FromPrimitive,
{
    fn unsafe_from_primitive(primitive: f64) -> Self {
        Self::from_f64(primitive).unwrap_or_else(|| {
            panic!(
                "f64 `{primitive:?}` should be convertable to `{}`",
                type_name::<T>(),
            )
        })
    }
}
