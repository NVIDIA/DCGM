use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().with_threads(true).init().unwrap();

    log::info!("Main thread logs here.");

    // If the "nightly" feature is enabled, the output will include thread ids.
    for _ in 1..=5 {
        std::thread::spawn(|| {
            log::info!("Unnamed thread logs here.");
        })
        .join()
        .unwrap();
    }

    std::thread::Builder::new()
        .name("named_thread".to_string())
        .spawn(|| {
            log::info!("Named thread logs here.");
        })
        .unwrap()
        .join()
        .unwrap();
}
