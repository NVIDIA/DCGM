use crate::{Error, LockFile};
use core::str;

#[cfg(feature = "std")]
#[test]
fn read_pid() -> Result<(), Error> {
    use std::fs::read_to_string;

    let path = "testfiles/read_pid.lock";
    let mut file = LockFile::open(path)?;
    file.lock_with_pid()?;

    let content_a = read_to_string(path)?;
    let content_b = read_to_string(path)?;

    assert!(content_a.trim().len() > 0);
    assert!(content_a.trim().chars().all(|ch| ch.is_ascii_digit()));

    assert_eq!(content_a, content_b);
    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn try_read_pid() -> Result<(), Error> {
    use std::fs::read_to_string;

    let path = "testfiles/try_read_pid.lock";
    let mut file = LockFile::open(path)?;
    assert!(file.try_lock_with_pid()?);

    let content_a = read_to_string(path)?;
    let content_b = read_to_string(path)?;

    assert!(content_a.trim().len() > 0);
    assert!(content_a.trim().chars().all(|ch| ch.is_ascii_digit()));

    assert_eq!(content_a, content_b);
    Ok(())
}

#[cfg(feature = "std")]
fn check_try_lock_example(
    lockpath: &str,
    expected: &[u8],
) -> Result<(), Error> {
    use std::process::{Command, Stdio};

    let child = Command::new("cargo")
        .arg("run")
        .arg("-q")
        .arg("--example")
        .arg("try_lock")
        .arg("--")
        .arg(lockpath)
        .stdout(Stdio::piped())
        .spawn()?;
    let output = child.wait_with_output()?;

    assert!(output.status.success());
    assert_eq!(output.stderr, b"");
    assert_eq!(output.stdout, expected);

    Ok(())
}

#[derive(Debug, Clone)]
enum TryPidExpectedRes<'pid> {
    Success { pid_to_differ: &'pid str },
    Failure,
}

#[cfg(feature = "std")]
fn check_try_lock_with_pid_example(
    lockpath: &str,
    expected: TryPidExpectedRes,
) -> Result<(), Error> {
    use std::process::{Command, Stdio};

    let child = Command::new("cargo")
        .arg("run")
        .arg("-q")
        .arg("--example")
        .arg("try_lock_with_pid")
        .arg("--")
        .arg(lockpath)
        .stdout(Stdio::piped())
        .spawn()?;
    let output = child.wait_with_output()?;

    assert!(output.status.success());
    assert_eq!(output.stderr, b"");
    match expected {
        TryPidExpectedRes::Success { pid_to_differ: pid } => {
            let output = str::from_utf8(&output.stdout).unwrap();
            assert!(output.trim().len() > 0);
            assert!(output.trim().chars().all(|ch| ch.is_ascii_digit()));
            assert_ne!(output.trim(), pid);
        },

        TryPidExpectedRes::Failure => assert_eq!(output.stdout, b"FAILURE\n"),
    }

    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn other_process() -> Result<(), Error> {
    let path = "testfiles/other_process.lock";
    let mut file = LockFile::open(path)?;
    file.lock()?;
    check_try_lock_example(path, b"FAILURE\n")?;
    file.unlock()?;
    check_try_lock_example(path, b"SUCCESS\n")?;
    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn other_process_pid() -> Result<(), Error> {
    use std::fs::read_to_string;

    let path = "testfiles/other_process_pid.lock";
    let mut file = LockFile::open(path)?;
    assert!(file.try_lock_with_pid()?);

    let content = read_to_string(path)?;
    assert!(content.trim().len() > 0);
    assert!(content.trim().chars().all(|ch| ch.is_ascii_digit()));

    check_try_lock_example(path, b"FAILURE\n")?;
    let content_again = read_to_string(path)?;
    assert!(content_again.trim().len() > 0);
    assert!(content_again.trim().chars().all(|ch| ch.is_ascii_digit()));
    file.unlock()?;
    check_try_lock_example(path, b"SUCCESS\n")?;

    let child_content = read_to_string(path)?;
    assert!(child_content.trim().len() == 0);

    assert!(file.try_lock_with_pid()?);

    let content_again = read_to_string(path)?;
    assert_eq!(content_again, content);

    check_try_lock_with_pid_example(path, TryPidExpectedRes::Failure)?;
    let content_again = read_to_string(path)?;
    assert!(content_again.trim().len() > 0);
    assert!(content_again.trim().chars().all(|ch| ch.is_ascii_digit()));
    file.unlock()?;
    check_try_lock_with_pid_example(
        path,
        TryPidExpectedRes::Success { pid_to_differ: &content },
    )?;

    let child_content = read_to_string(path)?;
    assert!(child_content.trim().len() == 0);

    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn other_process_but_curr_reads() -> Result<(), Error> {
    use std::fs::read_to_string;

    let path = "testfiles/other_process_but_curr_reads.lock";
    let mut file = LockFile::open(path)?;
    file.lock()?;

    check_try_lock_example(path, b"FAILURE\n")?;
    let mut _content = read_to_string(path)?;
    check_try_lock_example(path, b"FAILURE\n")?;

    file.unlock()?;
    check_try_lock_example(path, b"SUCCESS\n")?;
    Ok(())
}
