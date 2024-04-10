use std::{collections::Bound, ops::RangeBounds};
use std::ops::{Div, Sub};

use crate::{kernel::Uniform, Density};

/// Single Kernel Density Estimator component.
#[derive(Copy, Clone, Debug)]
pub struct Component<K, T> {
    /// Kernel function.
    pub kernel: K,

    /// Center of the corresponding kernel.
    pub location: T,

    /// Bandwidth of the corresponding kernel.
    pub bandwidth: T,
}

impl<K, T> Density<T> for Component<K, T>
where
    K: Density<T>,
    T: Copy + Div<T, Output = T> + Sub<T, Output = T>,
{
    fn density(&self, at: T) -> T {
        self.kernel.density((at - self.location) / self.bandwidth) / self.bandwidth
    }
}

macro_rules! new_uniform_impl {
    ($type:ty, $double_sqrt_3:expr) => {
        impl Component<Uniform, $type> {
            /// Create a new component with the uniform kernel function.
            ///
            /// The kernel will be scaled and moved so that the «box» spans the specified range.
            ///
            /// # Panics
            ///
            /// Unbounded ranges are not supported by definition.
            pub fn new(range: impl RangeBounds<$type>) -> Self {
                let min = Self::from_bound(range.start_bound());
                let max = Self::from_bound(range.end_bound());
                Self {
                    kernel: Uniform,
                    location: (min + max) / 2.0,
                    bandwidth: (max - min) / $double_sqrt_3,
                }
            }

            fn from_bound(bound: Bound<&$type>) -> $type {
                match bound {
                    Bound::Included(value) | Bound::Excluded(value) => *value,
                    Bound::Unbounded => panic!("uniform kernel cannot be unbounded"),
                }
            }
        }
    };
}

new_uniform_impl!(f32, crate::consts::f32::DOUBLE_SQRT_3);
new_uniform_impl!(f64, crate::consts::f64::DOUBLE_SQRT_3);
