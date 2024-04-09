#[allow(unused)]
pub use tracing::{debug, error, info, instrument, warn};

pub type Result<T = ()> = anyhow::Result<T>;
