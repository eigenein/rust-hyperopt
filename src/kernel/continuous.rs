//! Continuous kernels.

mod epanechnikov;
mod gaussian;
mod uniform;

pub use self::{epanechnikov::Epanechnikov, gaussian::Gaussian, uniform::Uniform};
