#[cfg(not(feature = "std"))]
use winapi::um::{
    winbase::{
        FormatMessageW,
        FORMAT_MESSAGE_ALLOCATE_BUFFER,
        FORMAT_MESSAGE_FROM_SYSTEM,
        FORMAT_MESSAGE_IGNORE_INSERTS,
    },
    winnt::{LANG_USER_DEFAULT, LPWSTR},
};

#[cfg(feature = "std")]
use std::{ffi, os::windows::ffi::OsStrExt};

use crate::{EitherOsStr, IntoOsString, ToOsStr};
use core::{
    convert::TryFrom,
    fmt,
    mem::{transmute, MaybeUninit},
    ptr::{self, NonNull},
    slice,
};
use winapi::{
    shared::{
        minwindef::{DWORD, FALSE, LPCVOID, LPVOID, TRUE},
        winerror::{ERROR_INVALID_DATA, ERROR_LOCK_VIOLATION},
    },
    um::{
        errhandlingapi::GetLastError,
        fileapi::{
            CreateFileW,
            FlushFileBuffers,
            LockFileEx,
            SetEndOfFile,
            SetFilePointer,
            UnlockFileEx,
            WriteFile,
            INVALID_SET_FILE_POINTER,
            OPEN_ALWAYS,
        },
        handleapi::{CloseHandle, INVALID_HANDLE_VALUE},
        minwinbase::{
            OVERLAPPED_u,
            LMEM_FIXED,
            LOCKFILE_EXCLUSIVE_LOCK,
            LOCKFILE_FAIL_IMMEDIATELY,
            LPOVERLAPPED,
            LPSECURITY_ATTRIBUTES,
            OVERLAPPED,
            SECURITY_ATTRIBUTES,
        },
        processthreadsapi::GetCurrentProcessId,
        synchapi::{CreateEventW, WaitForSingleObject},
        winbase::{LocalAlloc, LocalFree, FILE_BEGIN, WAIT_FAILED},
        winnt::{
            RtlCopyMemory,
            FILE_SHARE_DELETE,
            FILE_SHARE_READ,
            FILE_SHARE_WRITE,
            GENERIC_WRITE,
            HANDLE,
            WCHAR,
        },
    },
};

/// A type representing file descriptor on Unix.
pub type FileDesc = HANDLE;

/// A type representing Process ID on Windows.
pub type Pid = DWORD;

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
        Self::from_raw_os_error(unsafe { GetLastError() } as i32)
    }

    /// Raw OS error code. Returns option for compatibility with std.
    pub fn raw_os_error(&self) -> Option<i32> {
        Some(self.code)
    }
}

#[cfg(not(feature = "std"))]
impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut buf: LPWSTR = ptr::null_mut();
        let res = unsafe {
            FormatMessageW(
                FORMAT_MESSAGE_ALLOCATE_BUFFER
                    | FORMAT_MESSAGE_FROM_SYSTEM
                    | FORMAT_MESSAGE_IGNORE_INSERTS,
                ptr::null_mut(),
                self.code as DWORD,
                LANG_USER_DEFAULT as DWORD,
                &mut buf as *mut LPWSTR as LPWSTR,
                0,
                ptr::null_mut(),
            )
        };

        if res == 0 {
            write!(fmt, "error getting error message")?;
        } else {
            {
                let slice = unsafe {
                    slice::from_raw_parts(buf as *const WCHAR, res as usize)
                };
                let str = unsafe { OsStr::from_slice(slice) };
                write!(fmt, "{}", str)?;
            }
            unsafe {
                LocalFree(buf as LPVOID);
            }
        }

        Ok(())
    }
}

/// Owned allocation of an OS-native string.
pub struct OsString {
    alloc: NonNull<WCHAR>,
    /// Length without the nul-byte.
    len: usize,
}

impl Drop for OsString {
    fn drop(&mut self) {
        let ptr = self.alloc.as_ptr() as LPVOID;
        unsafe {
            LocalFree(ptr);
        }
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
pub struct OsStr {
    chars: [WCHAR],
}

impl OsStr {
    /// Unsafe cause sequence needs to end with 0.
    unsafe fn from_slice(slice: &[WCHAR]) -> &Self {
        transmute(slice)
    }

    fn chars(&self) -> Chars {
        Chars { inner: self.chars.iter() }
    }
}

impl fmt::Debug for OsStr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut first = false;
        write!(fmt, "[")?;

        for ch in self.chars() {
            if first {
                first = false;
            } else {
                write!(fmt, ", ")?;
            }
            write!(fmt, "{:?}", ch)?;
        }

        write!(fmt, "]")?;
        Ok(())
    }
}

impl fmt::Display for OsStr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        for ch in self.chars() {
            write!(fmt, "{}", ch)?;
        }

        Ok(())
    }
}

impl<'str> IntoOsString for &'str OsStr {
    fn into_os_string(self) -> Result<OsString, Error> {
        let len = self.chars.len();
        let alloc = unsafe { LocalAlloc(LMEM_FIXED, len * 2 + 2) };
        let alloc = match NonNull::new(alloc as *mut WCHAR) {
            Some(alloc) => alloc,
            None => {
                return Err(Error::last_os_error());
            },
        };
        unsafe {
            RtlCopyMemory(
                alloc.as_ptr() as LPVOID,
                self.chars.as_ptr() as _,
                len * 2 + 2,
            );
        }

        Ok(OsString { alloc, len })
    }
}

impl ToOsStr for str {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        let res = unsafe { make_os_string(|| self.encode_utf16()) };
        res.map(EitherOsStr::Owned)
    }
}

#[cfg(feature = "std")]
impl ToOsStr for ffi::OsStr {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        let res = unsafe { make_os_string(|| self.encode_wide()) };
        res.map(EitherOsStr::Owned)
    }
}

/// Unsafe because the returned iterator must be exactly the same.
unsafe fn make_os_string<F, I>(mut make_iter: F) -> Result<OsString, Error>
where
    F: FnMut() -> I,
    I: Iterator<Item = u16>,
{
    let mut len = 0;
    let mut prev_zero = false;
    for ch in make_iter() {
        if prev_zero {
            Err(Error::from_raw_os_error(ERROR_INVALID_DATA as i32))?;
        }
        if ch == 0 {
            prev_zero = true;
        } else {
            len += 1;
        }
    }

    let alloc = LocalAlloc(LMEM_FIXED, len * 2 + 2);
    let alloc = match NonNull::new(alloc as *mut WCHAR) {
        Some(alloc) => alloc,
        None => {
            return Err(Error::last_os_error());
        },
    };

    let mut iter = make_iter();
    for i in 0 .. len {
        let ch = iter.next().expect("Inconsistent .encode_utf16()");
        *alloc.as_ptr().add(i) = ch;
    }
    *alloc.as_ptr().add(len) = 0;
    Ok(OsString { alloc, len })
}

#[derive(Debug)]
struct Chars<'str> {
    inner: slice::Iter<'str, WCHAR>,
}

impl<'str> Iterator for Chars<'str> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = *self.inner.next()?;
        if curr <= 0xD7FF || curr >= 0xE000 {
            let ch = char::try_from(curr as u32)
                .expect("Inconsistent char implementation");
            Some(ch)
        } else {
            let next = *self.inner.next()?;
            let high = curr as u32 - 0xD800;
            let low = next as u32 - 0xDC00;
            let ch = char::try_from((high << 10 | low) + 0x10000)
                .expect("Inconsistent char implementation");
            Some(ch)
        }
    }
}

/// Helper to auto-drop a HANDLE.
#[derive(Debug)]
struct DropHandle {
    /// HANDLE being dropped.
    handle: HANDLE,
}

impl Drop for DropHandle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

/// Creates an event to be used by this implementation.
fn make_event() -> Result<HANDLE, Error> {
    let mut security = make_security_attributes();
    let res = unsafe {
        CreateEventW(
            &mut security as LPSECURITY_ATTRIBUTES,
            FALSE,
            FALSE,
            ptr::null_mut(),
        )
    };

    if res != INVALID_HANDLE_VALUE {
        Ok(res)
    } else {
        Err(Error::last_os_error())
    }
}

/// Creates security attributes to be used with this implementation.
fn make_security_attributes() -> SECURITY_ATTRIBUTES {
    SECURITY_ATTRIBUTES {
        nLength: 0,
        lpSecurityDescriptor: ptr::null_mut(),
        bInheritHandle: FALSE,
    }
}

/// Creates an overlapped struct to be used with this implementation.
fn make_overlapped() -> Result<OVERLAPPED, Error> {
    Ok(OVERLAPPED {
        Internal: 0,
        InternalHigh: 0,
        u: {
            let mut uninit = MaybeUninit::<OVERLAPPED_u>::uninit();
            unsafe {
                let mut refer = (&mut *uninit.as_mut_ptr()).s_mut();
                refer.Offset = DWORD::max_value() - 1;
                refer.OffsetHigh = DWORD::max_value() - 1;
                uninit.assume_init()
            }
        },
        hEvent: make_event()?,
    })
}

/// Returns the ID of the current process.
pub fn pid() -> Pid {
    unsafe { GetCurrentProcessId() }
}

/// Opens a file with only purpose of locking it. Creates it if it does not
/// exist. Path must not contain a nul-byte in the middle, but a nul-byte in the
/// end (and only in the end) is allowed, which in this case no extra allocation
/// will be made. Otherwise, an extra allocation is made.
pub fn open(path: &OsStr) -> Result<FileDesc, Error> {
    let mut security = make_security_attributes();
    let handle = unsafe {
        CreateFileW(
            path.chars.as_ptr(),
            GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            &mut security as LPSECURITY_ATTRIBUTES,
            OPEN_ALWAYS,
            0,
            ptr::null_mut(),
        )
    };

    if handle != INVALID_HANDLE_VALUE {
        Ok(handle)
    } else {
        Err(Error::last_os_error())
    }
}

/// Writes data into the given open file.
pub fn write(handle: FileDesc, bytes: &[u8]) -> Result<(), Error> {
    let result = unsafe {
        WriteFile(
            handle,
            bytes.as_ptr() as LPCVOID,
            bytes.len() as DWORD,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    if result == 0 {
        Err(Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn fsync(handle: FileDesc) -> Result<(), Error> {
    let result = unsafe { FlushFileBuffers(handle) };
    if result == 0 {
        Err(Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Truncates the file referenced by the given HANDLE and seeks it to the start.
pub fn truncate(handle: FileDesc) -> Result<(), Error> {
    let res = unsafe { SetFilePointer(handle, 0, ptr::null_mut(), FILE_BEGIN) };
    if res == INVALID_SET_FILE_POINTER {
        return Err(Error::last_os_error());
    }

    let res = unsafe { SetEndOfFile(handle) };
    if res == 0 {
        Err(Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Tries to lock a file and blocks until it is possible to lock.
pub fn lock(handle: FileDesc) -> Result<(), Error> {
    let mut overlapped = make_overlapped()?;
    let drop_handle = DropHandle { handle: overlapped.hEvent };
    let res = unsafe {
        LockFileEx(
            handle,
            LOCKFILE_EXCLUSIVE_LOCK,
            0,
            1,
            1,
            &mut overlapped as LPOVERLAPPED,
        )
    };

    let ret = if res == TRUE {
        let res = unsafe { WaitForSingleObject(overlapped.hEvent, 0) };
        if res != WAIT_FAILED {
            Ok(())
        } else {
            Err(Error::last_os_error())
        }
    } else {
        Err(Error::last_os_error())
    };

    drop(drop_handle);
    ret
}

/// Tries to lock a file but returns as soon as possible if already locked.
pub fn try_lock(handle: FileDesc) -> Result<bool, Error> {
    let mut overlapped = make_overlapped()?;
    let drop_handle = DropHandle { handle: overlapped.hEvent };
    let res = unsafe {
        LockFileEx(
            handle,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            1,
            &mut overlapped as LPOVERLAPPED,
        )
    };

    let ret = if res == TRUE {
        let res = unsafe { WaitForSingleObject(overlapped.hEvent, 0) };
        if res != WAIT_FAILED {
            Ok(true)
        } else {
            Err(Error::last_os_error())
        }
    } else {
        let err = unsafe { GetLastError() };
        if err == ERROR_LOCK_VIOLATION {
            Ok(false)
        } else {
            Err(Error::from_raw_os_error(err as i32))
        }
    };

    drop(drop_handle);
    ret
}

/// Unlocks the file.
pub fn unlock(handle: FileDesc) -> Result<(), Error> {
    let mut overlapped = make_overlapped()?;
    let drop_handle = DropHandle { handle: overlapped.hEvent };
    let res = unsafe {
        UnlockFileEx(handle, 0, 1, 1, &mut overlapped as LPOVERLAPPED)
    };

    let ret = if res == TRUE {
        let res = unsafe { WaitForSingleObject(overlapped.hEvent, 0) };
        if res != WAIT_FAILED {
            Ok(())
        } else {
            Err(Error::last_os_error())
        }
    } else {
        Err(Error::last_os_error())
    };

    drop(drop_handle);
    ret
}

/// Closes the file.
pub fn close(handle: FileDesc) {
    unsafe {
        CloseHandle(handle);
    }
}
