# zenwebp

[![crates.io](https://img.shields.io/crates/v/zenwebp.svg)](https://crates.io/crates/zenwebp)
[![Documentation](https://docs.rs/zenwebp/badge.svg)](https://docs.rs/zenwebp)
[![Build Status](https://github.com/imazen/zenwebp/workflows/Rust%20CI/badge.svg)](https://github.com/imazen/zenwebp/actions)

High-performance WebP encoding and decoding in pure Rust.

**Forked from [`image-webp`](https://github.com/image-rs/image-webp)** with significant improvements:
- **~2x faster decoding** (from ~3x slower than libwebp to ~1.4x slower)
- **Lossy encoding** (the original only supported lossless)
- **Full no_std support** with alloc (both encoder and decoder)
- **WASM SIMD128 support** for WebAssembly targets

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
- SIMD acceleration via `archmage` (x86: SSE2/SSE4.1/AVX2, WASM: SIMD128)
- **no_std + alloc** - full encoding/decoding without std
- Lossy encoding with full mode search (I16, I4, UV modes)
- Lossless encoding
- Animation support (decode)
- Alpha channel support
- ICC, EXIF, XMP metadata extraction

## Quick Start

### Decoding

```rust
use zenwebp::WebPDecoder;

// From a slice (no_std compatible)
let webp_data: &[u8] = &read_webp_file();
let mut decoder = WebPDecoder::new(webp_data)?;
let (width, height) = decoder.dimensions();

// Allocate output buffer (RGBA)
let mut output = vec![0u8; decoder.output_buffer_size()?];
decoder.read_image(&mut output)?;
```

### Encoding

```rust
use zenwebp::{WebPEncoder, EncoderParams, ColorType};

// Lossy encoding (quality 0-100)
let mut output = Vec::new();
let mut encoder = WebPEncoder::new(&mut output);
encoder.set_params(EncoderParams::lossy(75));
encoder.encode(&rgb_data, width, height, ColorType::Rgb8)?;

// Lossless encoding
let mut output = Vec::new();
let mut encoder = WebPEncoder::new(&mut output);
encoder.set_params(EncoderParams::lossless());
encoder.encode(&rgb_data, width, height, ColorType::Rgb8)?;
```

### no_std Usage

```toml
[dependencies]
zenwebp = { version = "0.1", default-features = false }
```

Both encoder and decoder work with `no_std + alloc`. The decoder takes `&[u8]` slices,
and the encoder writes to `Vec<u8>`. Only `encode_to_writer()` requires the `std` feature.

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
