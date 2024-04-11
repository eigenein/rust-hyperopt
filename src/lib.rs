#![doc = include_str!("../README.md")]

pub mod consts;
mod iter;
pub mod kde;
pub mod kernel;
mod optimizer;
pub mod rand;

pub use self::{
    kernel::{Density, Sample},
    optimizer::Optimizer,
};
