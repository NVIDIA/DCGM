use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().without_timestamps().init().unwrap();

    log::warn!("This is an example message.");
}
