use simple_logger::SimpleLogger;
use time::UtcOffset;

fn main() {
    SimpleLogger::new()
        .with_utc_offset(UtcOffset::from_hms(14, 0, 0).unwrap())
        .init()
        .unwrap();

    log::warn!("This is an example message using a static UTC offset.");
    log::info!("Daylight savings or other timezone changes will not be respected.");
}
