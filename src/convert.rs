use std::{any::type_name, fmt::Debug};

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
