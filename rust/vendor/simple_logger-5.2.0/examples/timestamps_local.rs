use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().with_local_timestamps().init().unwrap();

    log::warn!("This is an example message.");
}
