//! Constants used in the algorithms and kernels.
//!
//! Some constants are missing in [`num_traits`] and/or the standard library.
//! Some _are_ available, but only through [`num_traits::Float`], which is not implemented
//! for certain types. All in all, it was easier to go this way.

#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

macro_rules! define_trait {
    ($trait_:ident, $ident:ident, $value:literal, $comment:literal) => {
        #[doc = $comment]
        pub trait $trait_ {
            const $ident: Self;
        }

        impl $trait_ for f32 {
            const $ident: Self = $value;
        }

        impl $trait_ for f64 {
            const $ident: Self = $value;
        }

        #[cfg(feature = "ordered-float")]
        impl_for_ordered_float!($trait_, $ident);
    };
}

macro_rules! impl_for_ordered_float {
    ($trait_:ident, $ident:ident) => {
        impl $trait_ for ordered_float::NotNan<f32> {
            const $ident: Self = unsafe { Self::new_unchecked(f32::$ident) };
        }

        impl $trait_ for ordered_float::NotNan<f64> {
            const $ident: Self = unsafe { Self::new_unchecked(f64::$ident) };
        }

        impl $trait_ for ordered_float::OrderedFloat<f32> {
            const $ident: Self = Self(f32::$ident);
        }

        impl $trait_ for ordered_float::OrderedFloat<f64> {
            const $ident: Self = Self(f64::$ident);
        }
    };
}

define_trait!(
    ConstSqrt3,
    SQRT_3,
    1.7320508075688772935274463415058723669428052538103806280558069794,
    "√3"
);
define_trait!(
    ConstSqrt5,
    SQRT_5,
    2.2360679774997896964091736687312762354406183596115257242708972454,
    "√5"
);
define_trait!(
    ConstDoubleSqrt3,
    DOUBLE_SQRT_3,
    3.4641016151377545870548926830117447338856105076207612561116139589,
    "2√3"
);
define_trait!(
    ConstFrac1SqrtTau,
    FRAC_1_SQRT_TAU,
    0.3989422804014326779399460599343818684758586311649346576659258296,
    "1 / √(2π)"
);
define_trait!(ConstThreeQuarters, THREE_QUARTERS, 0.75, "`0.75`");
define_trait!(ConstOneHalf, ONE_HALF, 0.5, "`0.5`");
