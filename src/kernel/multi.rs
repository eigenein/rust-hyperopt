//! Multivariate kernels.
//!
//! This module implements the traits for a number of tuples.
//! Use [`impl_multivariate`] macro if you need longer tuples.
//!
//! # Note on variable independence
//!
//! Note that they do **not** explicitly incorporate any relationships between the inner parameters.
//! However, they are still quite useful since the kernels will only be placed in «good» («bad»)
//! combinations of the parameters, as opposed to **any** combinations of «good» («bad») inner parameters.
//!
//! For example, assume that `f(2, 0)` and `f(0, 3)` are «good» trials. Using multiple optimizers
//! will virtually lead to `(0, 0)` and `(2, 3)` being considered as «good» too, which is not
//! necessarily the case.
//!
//! # Note on [`crate::kernel::Density`]
//!
//! Note that the calculated density is **not** a valid probability density as it lacks
//! normalization. However, for the purpose of sampling and ordering, normalization is irrelevant.
//! Do **not** use these kernels for statistical calculations.
//!
//! # Note on runtime performance
//!
//! Using a multivariate kernel yields better performance since [`crate::Optimizer`] will
//! need to store each trial just once as opposed to using a separate [`crate::Optimizer`]
//! for each variable.

use fastrand::Rng;

use crate::{kernel::Kernel, traits::SelfMul, Density, Sample};

/// Implement [`Density`], [`Sample`], and [`Kernel`] for a generic tuple of kernels.
///
/// Due to the macro syntax limitation, one also has to specify indices manually like so:
///
/// ```ignore
/// impl_multivariate!(K0, 1 K1, 2 K2, 3 K3);
/// ```
#[macro_export]
macro_rules! impl_multivariate {
    ($first_type:ident $(,$next_index:tt $next_type:ident)*) => {
        impl<Output, $first_type $(, $next_type)*> Density for ($first_type, $($next_type, )*)
        where
            Output: SelfMul,
            $first_type: Density<Output = Output>,
            $($next_type: Density<Output = Output>,)*
        {
            // FIXME: implement `Param` new-type.
            type Param = (<$first_type as Density>::Param, $(<$next_type as Density>::Param, )*);

            type Output = Output;

            fn density(&self, at: Self::Param) -> Self::Output {
                self.0.density(at.0) $(* self.$next_index.density(at.$next_index))*
            }
        }

        impl<$first_type $(, $next_type)*> Sample for ($first_type, $($next_type,)*)
        where
            $first_type: Sample,
            $($next_type: Sample,)*
        {
            // FIXME: implement `Param` new-type.
            type Param = (<$first_type as Sample>::Param, $(<$next_type as Sample>::Param, )*);

            fn sample(&self, rng: &mut Rng) -> Self::Param {
                (self.0.sample(rng), $(self.$next_index.sample(rng), )*)
            }
        }

        impl<$first_type $(, $next_type)*> Kernel for ($first_type, $($next_type,)*)
        where
            $first_type: Kernel,
            $($next_type: Kernel,)*
        {
            // FIXME: implement `Param` new-type.
            type Param = (<$first_type as Kernel>::Param, $(<$next_type as Kernel>::Param, )*);

            fn new(location: Self::Param, std: Self::Param) -> Self {
                ($first_type::new(location.0, std.0), $($next_type::new(location.$next_index, std.$next_index), )*)
            }
        }
    };
}

impl_multivariate!(K0);
impl_multivariate!(K0, 1 K1);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::universal::Uniform;

    #[test]
    fn multi_sample_ok() {
        let kernel = (
            Uniform::<_, ()>::with_bounds(1..=1),
            Uniform::<_, ()>::with_bounds(2..=2),
        );
        assert_eq!(kernel.sample(&mut Rng::new()), (1, 2));
    }
}
