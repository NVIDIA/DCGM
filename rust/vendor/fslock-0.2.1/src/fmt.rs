//! This module implements formatting functions for writing into lock files.

use crate::sys;
use core::{
    fmt::{self, Write},
    mem,
};

/// I/O buffer size, chosen targeting possible PID's digits (I belive 11 would
/// be enough tho).
const BUF_SIZE: usize = 16;

/// A fmt Writer that writes data into the given open file.
#[derive(Debug, Clone, Copy)]
pub struct Writer(
    /// The open file to which data will be written.
    pub sys::FileDesc,
);

impl Writer {
    /// Writes formatting arguments into the file.
    pub fn write_fmt(
        &self,
        arguments: fmt::Arguments,
    ) -> Result<(), sys::Error> {
        let mut adapter = Adapter::new(self.0);
        let _ = adapter.write_fmt(arguments);
        adapter.finish()
    }
}

/// Fmt <-> IO adapter.
///
/// Buffer is flushed on drop.
#[derive(Debug)]
struct Adapter {
    /// File being written to.
    desc: sys::FileDesc,
    /// Temporary buffer of bytes being written.
    buffer: [u8; BUF_SIZE],
    /// Cursor tracking where new bytes should be written at the buffer.
    cursor: usize,
    /// Partial result for writes.
    result: Result<(), sys::Error>,
}

impl Adapter {
    /// Creates a zeroed adapter from an open file.
    fn new(desc: sys::FileDesc) -> Self {
        Self { desc, buffer: [0; BUF_SIZE], cursor: 0, result: Ok(()) }
    }

    /// Flushes the buffer into the open file.
    fn flush(&mut self) -> Result<(), sys::Error> {
        sys::write(self.desc, &self.buffer[.. self.cursor])?;
        self.buffer = [0; BUF_SIZE];
        self.cursor = 0;
        Ok(())
    }

    /// Finishes the adapter, returning the I/O Result
    fn finish(mut self) -> Result<(), sys::Error> {
        mem::replace(&mut self.result, Ok(()))
    }
}

impl Write for Adapter {
    fn write_str(&mut self, data: &str) -> fmt::Result {
        let mut bytes = data.as_bytes();

        while bytes.len() > 0 && self.result.is_ok() {
            let start = self.cursor;
            let size = (BUF_SIZE - self.cursor).min(bytes.len());
            let end = start + size;

            self.buffer[start .. end].copy_from_slice(&bytes[.. size]);
            self.cursor = end;
            bytes = &bytes[size ..];

            if bytes.len() > 0 {
                self.result = self.flush();
            }
        }

        match self.result {
            Ok(_) => Ok(()),
            Err(_) => Err(fmt::Error),
        }
    }
}

impl Drop for Adapter {
    fn drop(&mut self) {
        let _ = self.flush();
        let _ = sys::fsync(self.desc);
    }
}
