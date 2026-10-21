#[cfg(feature = "std")]
use fslock::LockFile;
#[cfg(feature = "std")]
use std::{env, fs::read_to_string, process};

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

    if lockfile.try_lock_with_pid()? {
        let content_a = read_to_string(&path)?;
        let content_b = read_to_string(&path)?;
        assert!(content_a.trim().len() > 0);
        assert!(content_a.trim().chars().all(|ch| ch.is_ascii_digit()));
        assert_eq!(content_a, content_b);

        println!("{}", content_a);
    } else {
        println!("FAILURE");
    }

    Ok(())
}

#[cfg(not(feature = "std"))]
fn main() {}
