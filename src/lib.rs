//! Decoding and Encoding of WebP Images
//!
//! This crate provides both encoding and decoding of WebP images with a
//! webpx-compatible API.
//!
//! # Encoding
//!
//! Use the [`Encoder`] builder for a fluent API:
//!
//! ```rust
//! use zenwebp::{Encoder, Preset};
//!
//! let rgba_data = vec![255u8; 4 * 4 * 4]; // 4x4 RGBA image
//! let webp = Encoder::new_rgba(&rgba_data, 4, 4)
//!     .preset(Preset::Photo)
//!     .quality(85.0)
//!     .encode()?;
//! # Ok::<(), zenwebp::EncodingError>(())
//! ```
//!
//! Or use [`EncoderConfig`] for reusable configuration:
//!
//! ```rust
//! use zenwebp::{EncoderConfig, Preset};
//!
//! let config = EncoderConfig::new().quality(85.0).preset(Preset::Photo);
//! let rgba_data = vec![255u8; 4 * 4 * 4];
//! let webp = config.encode_rgba(&rgba_data, 4, 4)?;
//! # Ok::<(), zenwebp::EncodingError>(())
//! ```
//!
//! # Decoding
//!
//! Use the convenience functions:
//!
//! ```rust,no_run
//! let webp_data: &[u8] = &[]; // your WebP data
//! let (pixels, width, height) = zenwebp::decode_rgba(webp_data)?;
//! # Ok::<(), zenwebp::DecodingError>(())
//! ```
//!
//! Or the [`WebPDecoder`] for more control:
//!
//! ```rust,no_run
//! use zenwebp::WebPDecoder;
//! use std::io::Cursor;
//!
//! let webp_data: &[u8] = &[]; // your WebP data
//! let mut decoder = WebPDecoder::new(Cursor::new(webp_data))?;
//! let (width, height) = decoder.dimensions();
//! let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];
//! decoder.read_image(&mut output)?;
//! # Ok::<(), zenwebp::DecodingError>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Increase recursion limit for the `quick_error!` macro.
#![recursion_limit = "256"]
// Enable nightly benchmark functionality if "_benchmarks" feature is enabled.
#![cfg_attr(all(test, feature = "_benchmarks"), feature(test))]
#[cfg(all(test, feature = "_benchmarks"))]
extern crate test;

// Decoder exports
pub use self::decoder::{
    decode_rgb, decode_rgb_into, decode_rgba, decode_rgba_into, DecodingError, ImageInfo,
    LoopCount, UpsamplingMethod, WebPDecodeOptions, WebPDecoder,
};

// Encoder exports
pub use self::encoder::{
    ColorType, Encoder, EncoderConfig, EncoderParams, EncodingError, Preset, WebPEncoder,
};

mod alpha_blending;
mod decoder;
mod encoder;
mod extended;
mod huffman;
mod loop_filter;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod loop_filter_avx2;
mod lossless;
mod lossless_transform;
#[cfg(feature = "simd")]
mod simd_sse;
mod transform;
#[cfg(feature = "simd")]
mod transform_simd_intrinsics;
mod vp8_arithmetic_decoder;
mod vp8_arithmetic_encoder;
mod vp8_bit_reader;
mod vp8_common;
mod vp8_cost;
mod vp8_encoder;
mod vp8_prediction;
mod yuv;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod yuv_simd;

pub mod vp8;
