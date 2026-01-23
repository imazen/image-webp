# zenwebp

[![crates.io](https://img.shields.io/crates/v/zenwebp.svg)](https://crates.io/crates/zenwebp)
[![Documentation](https://docs.rs/zenwebp/badge.svg)](https://docs.rs/zenwebp)
[![Build Status](https://github.com/imazen/zenwebp/workflows/Rust%20CI/badge.svg)](https://github.com/imazen/zenwebp/actions)

High-performance WebP encoding and decoding in pure Rust.

**Forked from [`image-webp`](https://github.com/image-rs/image-webp)** with significant improvements:
- **~1.8x faster decoding** (from ~2.5x slower than libwebp to ~1.4x slower)
- **Lossy encoding** (the original only supported lossless)
- **Complete API rewrite** with webpx-compatible interface

## Current Status

* **Decoder:** Supports all WebP format features including both lossless and
  lossy compression, alpha channel, and animation. Both the "simple" and
  "extended" formats are handled, and it exposes methods to extract ICC, EXIF,
  and XMP chunks. Decoding speed is approximately **70%** of libwebp.

* **Encoder:** Supports both **lossy and lossless** encoding. The lossy encoder
  includes RD-optimized mode selection, trellis quantization, and SIMD
  acceleration. Encoding speed is approximately **40%** of libwebp with
  comparable quality.

## Features

- Pure Rust implementation (no C dependencies)
- `#![forbid(unsafe_code)]` - completely safe Rust
- SIMD acceleration via `archmage` (SSE2/SSE4.1/AVX2)
- Lossy encoding with full mode search (I16, I4, UV modes)
- Lossless encoding
- Animation support (decode)
- Alpha channel support
- ICC, EXIF, XMP metadata extraction
- no_std compatible (error types only; encoding/decoding requires std)

## Usage

```rust
use zenwebp::{WebPDecoder, WebPEncoder, EncoderParams};

// Decode
let decoder = WebPDecoder::new(reader)?;
let image = decoder.decode()?;

// Encode lossy
let encoder = WebPEncoder::new_with_params(writer, EncoderParams::lossy(75));
encoder.encode(width, height, color_type, &data)?;

// Encode lossless
let encoder = WebPEncoder::new_with_params(writer, EncoderParams::lossless());
encoder.encode(width, height, color_type, &data)?;
```

## Performance

Benchmarks on 768x512 Kodak image at Q75:

| Encoder | Time | Throughput |
|---------|------|------------|
| zenwebp | 66ms | 5.9 MPix/s |
| libwebp | 25ms | 15.6 MPix/s |

| Decoder | Time | Throughput |
|---------|------|------------|
| zenwebp | 4.2ms | 93 MPix/s |
| libwebp | 3.0ms | 129 MPix/s |

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
