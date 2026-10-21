//! Set the `RUST_LOG` environment variable and run this example to see the output change.
//!
//! Valid values are `OFF`, `ERROR`, `WARN`, `INFO`, `DEBUG`, and `TRACE`.
//!
//! ```shell
//! RUST_LOG=WARN cargo run --example init_with_env
//! ```
//!
//! It should also work if the environment variable is not set:
//!
//! ```shell
//! cargo run --example init_with_env
//! ```
use simple_logger;

fn main() {
    simple_logger::init_with_env().unwrap();

    log::trace!("This is an example message.");
    log::debug!("This is an example message.");
    log::info!("This is an example message.");
    log::warn!("This is an example message.");
    log::error!("This is an example message.");
}
