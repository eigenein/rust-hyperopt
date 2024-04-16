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

use crate::{kernel::Kernel, Density, Sample};

/// Implement [`Density`], [`Sample`], and [`Kernel`] for a generic tuple of kernels.
///
/// Due to the macro syntax limitation, one also has to specify indices manually like so:
///
/// ```ignore
/// impl_multivariate!(0 K1, 1 K2, 2 K3,);
/// ```
#[macro_export]
macro_rules! impl_multivariate {
    ($($index:tt $type_:ident,)+) => {
        impl<$($type_,)+> Density for ($($type_,)+)
        where
            $($type_: Density,)+
        {
            type Param = ($(<$type_ as Density>::Param,)+);

            type Output = ($(<$type_ as Density>::Output,)+);

            #[inline]
            fn density(&self, at: Self::Param) -> Self::Output {
                ($(self.$index.density(at.$index),)+)
            }
        }

        impl<$($type_,)+> Sample for ($($type_,)+)
        where
            $($type_: Sample,)+
        {
            type Param = ($(<$type_ as Sample>::Param,)+);

            #[inline]
            fn sample(&self, rng: &mut Rng) -> Self::Param {
                ($(self.$index.sample(rng),)+)
            }
        }

        impl<$($type_,)+> Kernel for ($($type_,)+)
        where
            $($type_: Kernel,)+
        {
            type Param = ($(<$type_ as Kernel>::Param,)+);

            #[inline]
            fn new(location: Self::Param, std: Self::Param) -> Self {
                ($($type_::new(location.$index, std.$index),)+)
            }
        }
    };
}

impl_multivariate!(0 K1, );
impl_multivariate!(0 K1, 1 K2, );
impl_multivariate!(0 K1, 1 K2, 2 K3, );
impl_multivariate!(0 K1, 1 K2, 2 K3, 3 K4, );
impl_multivariate!(0 K1, 1 K2, 2 K3, 3 K4, 4 K5, );
impl_multivariate!(0 K1, 1 K2, 2 K3, 3 K4, 4 K5, 5 K6, );
impl_multivariate!(0 K1, 1 K2, 2 K3, 3 K4, 4 K5, 5 K6, 6 K7, );
impl_multivariate!(0 K1, 1 K2, 2 K3, 3 K4, 4 K5, 5 K6, 6 K7, 7 K8, );

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
