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

use tokio::io::AsyncRead;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

const MAX_FRAME_LENGTH: usize = 64 * 1024;

/// Builds a length-delimited codec using a 4-byte little-endian length prefix.
#[must_use]
pub fn codec() -> LengthDelimitedCodec {
    LengthDelimitedCodec::builder()
        .length_field_type::<u32>()
        .little_endian()
        .max_frame_length(MAX_FRAME_LENGTH)
        .new_codec()
}

/// Wraps an [`AsyncRead`] in a length-delimited framed reader.
#[must_use]
pub fn framed_reader<R: AsyncRead>(reader: R) -> FramedRead<R, LengthDelimitedCodec> {
    FramedRead::new(reader, codec())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use bytes::BytesMut;
    use tokio_util::codec::Decoder;

    fn encode_frame(payload: &[u8]) -> BytesMut {
        let len = u32::try_from(payload.len()).unwrap();
        let mut buf = BytesMut::with_capacity(4 + payload.len());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn valid_frame_decodes_payload() {
        let mut c = codec();
        let mut buf = encode_frame(b"hello");
        let frame = c.decode(&mut buf).unwrap().unwrap();
        assert_eq!(&frame[..], b"hello");
    }

    #[test]
    fn oversize_frame_produces_error() {
        let mut c = codec();
        let payload = vec![0u8; MAX_FRAME_LENGTH + 1];
        let mut buf = encode_frame(&payload);
        let result = c.decode(&mut buf);
        assert!(result.is_err(), "frame exceeding 64 KiB should error");
    }

    #[test]
    fn empty_frame_decodes_to_empty() {
        let mut c = codec();
        let mut buf = encode_frame(b"");
        let frame = c.decode(&mut buf).unwrap().unwrap();
        assert!(frame.is_empty());
    }

    #[test]
    fn truncated_frame_returns_none() {
        let mut c = codec();
        let mut buf = encode_frame(b"hello world");
        buf.truncate(8);
        let result = c.decode(&mut buf).unwrap();
        assert!(result.is_none(), "incomplete frame should return None");
    }
}
