#[cfg(feature = "std")]
use fslock::LockFile;
#[cfg(feature = "std")]
use std::{env, process};

#[cfg(feature = "std")]
fn main() -> Result<(), fslock::Error> {
    let mut args = env::args();
    args.next();

    let path = match args.next() {
        Some(arg) if args.next().is_none() => arg,
        _ => {
            eprintln!("Expected one argument");
            process::exit(1);
        },
    };

    let mut lockfile = LockFile::open(&path)?;

    if lockfile.try_lock()? {
        println!("SUCCESS");
    } else {
        println!("FAILURE");
    }

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {}
