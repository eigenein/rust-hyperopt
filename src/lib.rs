#![doc = include_str!("../README.md")]

pub mod consts;
mod iter;
pub mod kde;
pub mod kernel;
mod optimizer;
mod traits;

pub use self::{
    kernel::{Density, Sample},
    optimizer::Optimizer,
};
