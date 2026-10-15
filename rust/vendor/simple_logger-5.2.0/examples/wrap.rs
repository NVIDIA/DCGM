use log::{Log, Metadata, Record};
use simple_logger::SimpleLogger;

struct WrapperLogger {
    simple_logger: SimpleLogger,
}

impl Log for WrapperLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        self.simple_logger.enabled(metadata)
    }

    fn log(&self, record: &Record) {
        self.simple_logger.log(record)
    }

    fn flush(&self) {
        self.simple_logger.flush()
    }
}

fn main() {
    let simple_logger = SimpleLogger::new();
    log::set_max_level(simple_logger.max_level());

    let wrapper_logger = WrapperLogger { simple_logger };
    log::set_boxed_logger(Box::new(wrapper_logger)).unwrap();

    log::warn!("This is an example message.");
}
