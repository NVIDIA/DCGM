/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use core::ffi::c_char;

/// Errors returned while converting a fixed-size C character buffer to [`String`]
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum CBufToStringError {
    /// The buffer did not contain a NUL byte, but the full buffer was valid UTF-8.
    #[error("C character buffer is not NUL-terminated: {0:?}")]
    NotNulTerminated(String),
    /// The bytes before the first NUL, or the whole unterminated buffer, were not valid UTF-8.
    #[error("C character buffer contains invalid UTF-8")]
    InvalidUtf8,
}

/// Converts a fixed-size C character buffer to a [`String`].
///
/// Decoding stops at the first NUL byte. If no NUL byte is present, the entire
/// buffer is decoded and returned in [`CBufToStringError::NotNulTerminated`].
///
/// # Errors
///
/// - Returns [`CBufToStringError::InvalidUtf8`] when the bytes before the first NUL,
///   or the full unterminated buffer, is not valid UTF-8.
/// - Returns [`CBufToStringError::NotNulTerminated`] when the buffer has no NUL byte
///   but is otherwise valid UTF-8.
pub fn c_char_buf_to_string(buf: &[c_char]) -> Result<String, CBufToStringError> {
    // SAFETY: `c_char` is a one-byte C ABI character type. Reinterpreting the
    // caller-owned fixed buffer as bytes preserves the raw contents regardless
    // of whether `c_char` is signed on the target architecture.
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len()) };

    bytes.iter().position(|byte| *byte == b'\0').map_or_else(
        || {
            str::from_utf8(bytes).map_or(Err(CBufToStringError::InvalidUtf8), |value| {
                Err(CBufToStringError::NotNulTerminated(value.to_owned()))
            })
        },
        |nul_index| {
            str::from_utf8(&bytes[..nul_index])
                .map(ToOwned::to_owned)
                .map_err(|_| CBufToStringError::InvalidUtf8)
        },
    )
}

/// Writes a NUL-terminated Rust string into a fixed-size C character buffer.
///
/// The buffer is cleared first. If `value` does not fit, it is truncated at
/// the largest UTF-8 character boundary that still leaves room for a trailing
/// NUL byte, so the written prefix is always valid UTF-8 and round-trips
/// through [`c_char_buf_to_string`].
pub fn write_c_char_buf(buf: &mut [c_char], value: &str) {
    if buf.is_empty() {
        return;
    }

    // SAFETY: `c_char` is a one-byte C ABI character type. Reinterpreting the
    // caller-owned mutable buffer as bytes preserves the raw representation
    // regardless of whether `c_char` is signed on the target architecture.
    let bytes = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), buf.len()) };

    bytes.fill(0);
    let max_copy = bytes.len().saturating_sub(1);
    let copy_len = value.floor_char_boundary(max_copy);
    bytes[..copy_len].copy_from_slice(&value.as_bytes()[..copy_len]);
}

#[cfg(test)]
mod tests {
    use super::{CBufToStringError, c_char_buf_to_string, write_c_char_buf};
    use core::ffi::c_char;

    fn byte_to_c_char(byte: u8) -> c_char {
        c_char::from_ne_bytes([byte])
    }

    fn c_char_buffer(bytes: &[u8]) -> Vec<c_char> {
        bytes.iter().copied().map(byte_to_c_char).collect()
    }

    #[test]
    fn c_char_buf_to_string_reads_until_first_nul() {
        let buffer = c_char_buffer(b"true\0ignored");

        assert_eq!(c_char_buf_to_string(&buffer), Ok(String::from("true")));
    }

    #[test]
    fn c_char_buf_to_string_returns_partial_string_without_nul() {
        let buffer = c_char_buffer(b"passthrough");

        assert_eq!(
            c_char_buf_to_string(&buffer),
            Err(CBufToStringError::NotNulTerminated(String::from("passthrough")))
        );
    }

    #[test]
    fn c_char_buf_to_string_rejects_invalid_utf8_before_nul() {
        let buffer = c_char_buffer(b"valid-\xFF\0");

        assert_eq!(c_char_buf_to_string(&buffer), Err(CBufToStringError::InvalidUtf8));
    }

    #[test]
    fn c_char_buf_to_string_rejects_incomplete_utf8_before_nul() {
        let buffer = c_char_buffer(b"valid-\xE2\0");

        assert_eq!(c_char_buf_to_string(&buffer), Err(CBufToStringError::InvalidUtf8));
    }

    #[test]
    fn c_char_buf_to_string_rejects_invalid_utf8_without_nul() {
        let buffer = c_char_buffer(b"invalid-\xFF");

        assert_eq!(c_char_buf_to_string(&buffer), Err(CBufToStringError::InvalidUtf8));
    }

    #[test]
    fn write_c_char_buf_writes_and_nul_terminates() {
        let mut buffer = vec![byte_to_c_char(b'x'); 8];

        write_c_char_buf(&mut buffer, "abc");

        assert_eq!(c_char_buf_to_string(&buffer), Ok(String::from("abc")));
        assert_eq!(buffer[3], byte_to_c_char(0));
    }

    #[test]
    fn write_c_char_buf_truncates_to_keep_nul() {
        let mut buffer = vec![byte_to_c_char(b'x'); 4];

        write_c_char_buf(&mut buffer, "abcdef");

        assert_eq!(c_char_buf_to_string(&buffer), Ok(String::from("abc")));
        assert_eq!(buffer[3], byte_to_c_char(0));
    }

    #[test]
    fn write_c_char_buf_truncates_at_utf8_char_boundary() {
        // "h\u{e9}llo": 'h' (1) + '\u{e9}' (2) + 'l' (1) + 'l' (1) + 'o' (1) = 6 bytes.
        // A 3-byte buffer reserves 2 bytes for content; a naive byte truncation
        // would write 'h' + 0xC3 (the lead byte of '\u{e9}'), producing invalid UTF-8.
        // floor_char_boundary must back off to keep only "h".
        let mut buffer = vec![byte_to_c_char(b'x'); 3];

        write_c_char_buf(&mut buffer, "h\u{e9}llo");

        assert_eq!(c_char_buf_to_string(&buffer), Ok(String::from("h")));
        assert_eq!(buffer[1], byte_to_c_char(0));
    }

    #[test]
    fn write_c_char_buf_drops_all_when_first_char_does_not_fit() {
        // '\u{03A9}' (Ω) is 2 bytes; a 2-byte buffer reserves 1 byte for content, which is
        // not enough for the first char. Result is an empty NUL-terminated string.
        let mut buffer = vec![byte_to_c_char(b'x'); 2];

        write_c_char_buf(&mut buffer, "\u{03A9}xx");

        assert_eq!(c_char_buf_to_string(&buffer), Ok(String::new()));
        assert_eq!(buffer[0], byte_to_c_char(0));
    }

    #[test]
    fn write_c_char_buf_accepts_empty_buffers() {
        let mut buffer = Vec::new();

        write_c_char_buf(&mut buffer, "ignored");

        assert!(buffer.is_empty());
    }
}
