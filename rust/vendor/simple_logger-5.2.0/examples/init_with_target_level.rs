use log::LevelFilter;
use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new()
        .with_level(LevelFilter::Info)
        .with_module_level("init_with_target_level", LevelFilter::Off)
        .init()
        .unwrap();

    log::info!("This will NOT be logged. (Target disabled)");
}
