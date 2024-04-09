mod cli;
mod prelude;
mod tracing;

use clap::Parser;
use dotenvy::dotenv;

use crate::{cli::Cli, prelude::*};

fn main() -> Result {
    let _ = dotenv();
    let cli = Cli::parse();
    let _tracing_guards = tracing::init(cli.sentry_dsn.as_deref())?;
    Ok(())
}
