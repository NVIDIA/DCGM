# fslock

API to use files as a lock. Supports non-std crates by disabling feature
`std`.

# Types
Currently, only one type is provided: [`LockFile`]. It does not destroy the
file after closed and behaviour on locking different file handles owned by
the same process is different between Unix and Windows, unless you activate the
`multilock` feature, which enables the `open_excl` method that locks files per
file descriptor/handle on all platforms.

# Example
```rust
use fslock::LockFile;
fn main() -> Result<(), fslock::Error> {

    let mut file = LockFile::open("mylock")?;
    file.lock()?;
    do_stuff();
    file.unlock()?;

    Ok(())
}
```

# Docs on Master

https://brunoczim.github.io/fslock/fslock
