#![doc = include_str!("../README.md")]

mod iter;
pub mod kde;
pub mod kernel;
mod optimizer;
mod range;
mod traits;

pub use self::{
    kernel::{Density, Sample},
    optimizer::Optimizer,
    traits::constants,
};
