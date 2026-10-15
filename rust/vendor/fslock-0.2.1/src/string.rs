//! This module implements common functionalities for OS's strings.

use crate::sys::{Error, OsStr, OsString};
use core::{fmt, ops::Deref};

#[cfg(feature = "std")]
use std::{
    ffi,
    path::{Path, PathBuf},
};

impl Clone for OsString {
    fn clone(&self) -> Self {
        self.to_os_str()
            .and_then(|str| str.into_os_string())
            .expect("Allocation error")
    }
}

impl fmt::Debug for OsString {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", self.as_ref())
    }
}

impl fmt::Display for OsString {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.as_ref())
    }
}

impl Deref for OsString {
    type Target = OsStr;

    fn deref(&self) -> &OsStr {
        self.as_ref()
    }
}

/// Either borrowed or owned allocation of an OS-native string.
#[derive(Debug)]
pub enum EitherOsStr<'str> {
    /// Borrowed allocation.
    Borrowed(&'str OsStr),
    /// Owned allocation.
    Owned(OsString),
}

impl<'str> AsRef<OsStr> for EitherOsStr<'str> {
    fn as_ref(&self) -> &OsStr {
        match self {
            Self::Borrowed(str) => str,
            Self::Owned(string) => string.as_ref(),
        }
    }
}

impl<'str> Deref for EitherOsStr<'str> {
    type Target = OsStr;

    fn deref(&self) -> &OsStr {
        self.as_ref()
    }
}

/// Conversion of anything into an owned OS-native string. If allocation fails,
/// an error shall be returned.
pub trait IntoOsString {
    /// Converts with possible allocation error.
    fn into_os_string(self) -> Result<OsString, Error>;
}

impl IntoOsString for OsString {
    fn into_os_string(self) -> Result<OsString, Error> {
        Ok(self)
    }
}

impl<'str> IntoOsString for EitherOsStr<'str> {
    fn into_os_string(self) -> Result<OsString, Error> {
        match self {
            Self::Borrowed(str) => str.into_os_string(),
            Self::Owned(string) => Ok(string),
        }
    }
}

#[cfg(feature = "std")]
impl<'str> IntoOsString for &'str ffi::OsStr {
    fn into_os_string(self) -> Result<OsString, Error> {
        self.to_os_str()?.into_os_string()
    }
}

#[cfg(feature = "std")]
impl IntoOsString for PathBuf {
    fn into_os_string(self) -> Result<OsString, Error> {
        (*self).into_os_string()
    }
}

#[cfg(feature = "std")]
impl<'str> IntoOsString for &'str Path {
    fn into_os_string(self) -> Result<OsString, Error> {
        AsRef::<ffi::OsStr>::as_ref(self).to_os_str()?.into_os_string()
    }
}

#[cfg(feature = "std")]
impl IntoOsString for ffi::OsString {
    fn into_os_string(self) -> Result<OsString, Error> {
        (*self).into_os_string()
    }
}

impl<'str> IntoOsString for &'str str {
    fn into_os_string(self) -> Result<OsString, Error> {
        self.to_os_str()?.into_os_string()
    }
}

#[cfg(feature = "std")]
impl IntoOsString for String {
    fn into_os_string(self) -> Result<OsString, Error> {
        self.to_os_str()?.into_os_string()
    }
}

#[cfg(feature = "std")]
impl ToOsStr for String {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        (**self).to_os_str()
    }
}

/// Conversion of anything to an either borrowed or owned OS-native string. If
/// allocation fails, an error shall be returned.
pub trait ToOsStr {
    /// Converts with possible allocation error.
    fn to_os_str(&self) -> Result<EitherOsStr, Error>;
}

impl<'str> ToOsStr for EitherOsStr<'str> {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        Ok(match self {
            EitherOsStr::Owned(string) => {
                EitherOsStr::Owned(string.to_os_str()?.into_os_string()?)
            },
            EitherOsStr::Borrowed(str) => EitherOsStr::Borrowed(str),
        })
    }
}

impl ToOsStr for OsStr {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        Ok(EitherOsStr::Borrowed(self))
    }
}

impl ToOsStr for OsString {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        Ok(EitherOsStr::Borrowed(self.as_ref()))
    }
}

#[cfg(feature = "std")]
impl ToOsStr for ffi::OsString {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        (**self).to_os_str()
    }
}

#[cfg(feature = "std")]
impl ToOsStr for PathBuf {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        (**self).to_os_str()
    }
}

#[cfg(feature = "std")]
impl ToOsStr for Path {
    fn to_os_str(&self) -> Result<EitherOsStr, Error> {
        AsRef::<ffi::OsStr>::as_ref(self).to_os_str()
    }
}
