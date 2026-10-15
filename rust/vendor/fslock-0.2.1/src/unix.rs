use crate::{EitherOsStr, IntoOsString, ToOsStr};
use core::{fmt, mem::transmute, ptr::NonNull, slice, str};

#[cfg(feature = "std")]
use std::{ffi, os::unix::ffi::OsStrExt};

#[cfg(not(feature = "std"))]
extern "C" {
    /// Yeah, I had to copy this from std
    #[cfg(not(target_os = "dragonfly"))]
    #[cfg_attr(
        any(
            target_os = "linux",
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "l4re"
        ),
        link_name = "__errno_location"
    )]
    #[cfg_attr(
        any(
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "android",
            target_os = "redox",
            target_env = "newlib"
        ),
        link_name = "__errno"
    )]
    #[cfg_attr(target_os = "solaris", link_name = "___errno")]
    #[cfg_attr(
        any(target_os = "macos", target_os = "ios", target_os = "freebsd"),
        link_name = "__error"
    )]
    #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
    fn errno_location() -> *mut libc::c_int;
}

#[cfg(not(feature = "std"))]
fn errno() -> libc::c_int {
    unsafe { *errno_location() }
}

#[cfg(feature = "std")]
fn errno() -> libc::c_int {
    Error::last_os_error().raw_os_error().unwrap_or(0) as libc::c_int
}

/// A type representing file descriptor on Unix.
pub type FileDesc = libc::c_int;

/// A type representing Process ID on Unix.
pub type Pid = libc::pid_t;

#[cfg(feature = "std")]
/// An IO error.
pub type Error = std::io::Error;

#[cfg(not(feature = "std"))]
#[derive(Debug)]
/// An IO error. Without std, you can only get a message or an OS error code.
pub struct Error {
    code: i32,
}

#[cfg(not(feature = "std"))]
impl Error {
    /// Creates an error from a raw OS error code.
    pub fn from_raw_os_error(code: i32) -> Self {
        Self { code }
    }

    /// Creates an error from the last OS error code.
    pub fn last_os_error() -> Error {
        Self::from_raw_os_error(errno() as i32)
    }

    /// Raw OS error code. Returns option for compatibility with std.
    pub fn raw_os_error(&self) -> Option<i32> {
        Some(self.code)
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let msg_ptr = unsafe { libc::strerror(self.code as libc::c_int) };
        let len = unsafe { libc::strlen(msg_ptr) };
        let slice = unsafe { slice::from_raw_parts(msg_ptr, len) };
        write!(fmt, "{}", unsafe { OsStr::from_slice(slice) })?;
        Ok(())
    }
}

/// Owned allocation of an OS-native string.
pub struct OsString {
    alloc: NonNull<libc::c_char>,
    /// Length without the nul-byte.
    len: usize,
}

impl Drop for OsString {
    fn drop(&mut self) {
        let ptr = self.alloc.as_ptr() as *mut libc::c_void;
        unsafe { libc::free(ptr) }
    }
}

impl AsRef<OsStr> for OsString {
    fn as_ref(&self) -> &OsStr {
        unsafe {
            OsStr::from_slice(slice::from_raw_parts(
                self.alloc.as_ptr(),
                self.len,
            ))
        }
    }
}

/// Borrowed allocation of an OS-native string.
#[repr(transparent)]
pub struct OsStr {
    bytes: [libc::c_char],
}

impl OsStr {
    /// Unsafe cause sequence needs to end with 0.
    unsafe fn from_slice(slice: &[libc::c_char]) -> &Self {
        transmute(slice)
    }
}

impl fmt::Debug for OsStr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut first = false;
        write!(fmt, "[")?;

        for &signed in &self.bytes {
            let byte = signed as u8;
            if first {
                first = false;
            } else {
                write!(fmt, ", ")?;
            }
            if byte.is_ascii() {
                write!(fmt, "{:?}", char::from(byte))?;
            } else {
                write!(fmt, "'\\x{:x}'", byte)?;
            }
        }

        write!(fmt, "]")?;
        Ok(())
    }
}

impl fmt::Display for OsStr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let ptr = self.bytes.as_ptr();
        let len = self.bytes.len();
        let slice = unsafe { slice::from_raw_parts(ptr as _, len) };

        let mut sub = slice;

        while sub.len() > 0 {
            match str::from_utf8(sub) {
                Ok(string) => {
                    write!(fmt, "{}", string)?;
                    sub = &[];
                },
                Err(err) => {
                    let string = str::from_utf8(&sub[.. err.valid_up_to()])
                        .expect("Inconsistent utf8 error");
                    write!(fmt, "{}�", string,)?;

                    sub = &sub[err.valid_up_to() + 1 ..];
                },
            }
        }

        Ok(())
    }
}

impl<'str> IntoOsString for &'str OsStr {
    fn into_os_string(self) -> Result<OsString, Error> {
        let len = self.bytes.len();
        let alloc = unsafe { libc::malloc(len + 1) };
        let alloc = match NonNull::new(alloc as *mut libc::c_char) {
            Some(alloc) => alloc,
            None => {
                return Err(Error::last_os_error());
            },
        };
        unsafe {
            libc::memcpy(
                alloc.as_ptr() as *mut libc::c_void,
                self.bytes.as_ptr() as *const libc::c_void,
                len + 1,
            );
        }

        Ok(OsString { alloc, len })
    }
}

impl ToOsStr for str {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        make_os_str(self.as_bytes())
    }
}

#[cfg(feature = "std")]
impl ToOsStr for ffi::OsStr {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        make_os_str(self.as_bytes())
    }
}

/// Path must not contain a nul-byte in the middle, but a nul-byte in the end
/// (and only in the end) is allowed, which in this case no extra allocation
/// will be made. Otherwise, an extra allocation is made.
fn make_os_str(slice: &[u8]) -> Result<EitherOsStr, Error> {
    if let Some((&last, init)) = slice.split_last() {
        if init.contains(&0) {
            panic!("Path to file cannot contain nul-byte in the middle");
        }
        if last == 0 {
            let str = unsafe { OsStr::from_slice(transmute(slice)) };
            return Ok(EitherOsStr::Borrowed(str));
        }
    }

    let alloc = unsafe { libc::malloc(slice.len() + 1) };
    let alloc = match NonNull::new(alloc as *mut libc::c_char) {
        Some(alloc) => alloc,
        None => {
            return Err(Error::last_os_error());
        },
    };
    unsafe {
        libc::memcpy(
            alloc.as_ptr() as *mut libc::c_void,
            slice.as_ptr() as *const libc::c_void,
            slice.len(),
        );
        *alloc.as_ptr().add(slice.len()) = 0;
    }

    Ok(EitherOsStr::Owned(OsString { alloc, len: slice.len() }))
}

/// Returns the ID of the current process.
pub fn pid() -> Pid {
    unsafe { libc::getpid() }
}

/// Opens a file with only purpose of locking it. Creates it if it does not
/// exist. Path must not contain a nul-byte in the middle, but a nul-byte in the
/// end (and only in the end) is allowed, which in this case no extra allocation
/// will be made. Otherwise, an extra allocation is made.
pub fn open(path: &OsStr) -> Result<FileDesc, Error> {
    let fd = unsafe {
        libc::open(
            path.bytes.as_ptr(),
            libc::O_RDWR | libc::O_CLOEXEC | libc::O_CREAT,
            (libc::S_IRUSR | libc::S_IWUSR | libc::S_IRGRP | libc::S_IROTH)
                as libc::c_int,
        )
    };

    if fd >= 0 {
        Ok(fd)
    } else {
        Err(Error::last_os_error())
    }
}

/// Writes data into the given open file.
pub fn write(fd: FileDesc, mut bytes: &[u8]) -> Result<(), Error> {
    while bytes.len() > 0 {
        let written = unsafe {
            libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len())
        };
        if written < 0 && errno() != libc::EAGAIN {
            return Err(Error::last_os_error());
        }
        bytes = &bytes[written as usize ..];
    }

    Ok(())
}

pub fn fsync(fd: FileDesc) -> Result<(), Error> {
    let result = unsafe { libc::fsync(fd) };

    if result >= 0 {
        Ok(())
    } else {
        Err(Error::last_os_error())
    }
}

/// Truncates the file referenced by the given file descriptor and seeks it to
/// the start.
pub fn truncate(fd: FileDesc) -> Result<(), Error> {
    let res = unsafe { libc::lseek(fd, 0, libc::SEEK_SET) };
    if res < 0 {
        return Err(Error::last_os_error());
    }

    let res = unsafe { libc::ftruncate(fd, 0) };
    if res < 0 {
        Err(Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Tries to lock a file and blocks until it is possible to lock.
pub fn lock(fd: FileDesc) -> Result<(), Error> {
    let res = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if res >= 0 {
        Ok(())
    } else {
        Err(Error::last_os_error())
    }
}

/// Tries to lock a file but returns as soon as possible if already locked.
pub fn try_lock(fd: FileDesc) -> Result<bool, Error> {
    let res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if res >= 0 {
        Ok(true)
    } else {
        let err = errno();
        if err == libc::EWOULDBLOCK || err == libc::EINTR {
            Ok(false)
        } else {
            Err(Error::from_raw_os_error(err as i32))
        }
    }
}

/// Unlocks the file.
pub fn unlock(fd: FileDesc) -> Result<(), Error> {
    let res = unsafe { libc::flock(fd, libc::LOCK_UN) };
    if res >= 0 {
        Ok(())
    } else {
        Err(Error::last_os_error())
    }
}

/// Closes the file.
pub fn close(fd: FileDesc) {
    unsafe { libc::close(fd) };
}
