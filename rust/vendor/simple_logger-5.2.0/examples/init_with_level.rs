use log::LevelFilter;
use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().with_level(LevelFilter::Warn).init().unwrap();

    log::warn!("This will be logged.");
    log::info!("This will NOT be logged.");
}
