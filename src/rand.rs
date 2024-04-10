//! Random number generator abstractions.
//!
//! One does not need to use them normally.
//! However, they may be useful as building blocks to extend the package functionality.

mod uniform;

pub use self::uniform::Uniform;
