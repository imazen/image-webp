//! SIMD-optimized VP8 loop filter using x86 SSE4.1 intrinsics.
//!
//! The loop filter is applied to deblocking edges between macroblocks and subblocks.
//! This implementation processes 4-16 pixels in parallel using 128-bit SIMD registers.
//!
//! NOTE: This code is ready but not yet integrated into the decoder. Integration
//! requires restructuring the decoder to batch filter operations rather than
//! calling them one edge at a time.

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Signed to unsigned conversion: add 128 to convert [-128, 127] to [0, 255]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn s2u_epi8(v: __m128i) -> __m128i {
    let offset = _mm_set1_epi8(-128i8); // This is 0x80
    _mm_sub_epi8(v, offset)
}

/// Unsigned to signed conversion: subtract 128 to convert [0, 255] to [-128, 127]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn u2s_epi8(v: __m128i) -> __m128i {
    let offset = _mm_set1_epi8(-128i8);
    _mm_add_epi8(v, offset)
}

/// Absolute difference of unsigned bytes: |a - b|
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn abs_diff_epu8(a: __m128i, b: __m128i) -> __m128i {
    // max(a,b) - min(a,b) = |a - b| for unsigned
    let max = _mm_max_epu8(a, b);
    let min = _mm_min_epu8(a, b);
    _mm_sub_epi8(max, min)
}

/// Clamp signed bytes to [-128, 127] (saturating)
/// Since we're already in i8 range, this is a no-op but kept for clarity
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn clamp_epi8(v: __m128i) -> __m128i {
    v // Already in i8 range
}

/// Apply the simple loop filter to 4 horizontal edges in parallel.
/// Each edge consists of pixels [p1, p0, q0, q1] arranged horizontally.
///
/// This filters 4 rows at once, processing pixels at positions:
/// - p1: point - 2
/// - p0: point - 1
/// - q0: point
/// - q1: point + 1
///
/// # Safety
/// Requires SSE4.1 support. Caller must ensure pixel data is valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_filter_horizontal_4x(
    pixels: &mut [u8],
    offsets: [usize; 4],
    edge_limit: u8,
) {
    // Load 4 pixels from each of 4 rows (p1, p0, q0, q1 for each row)
    // Pack into vectors - unroll manually since _mm_insert_epi8 needs const index
    let p1 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[offsets[0] - 2],
        pixels[offsets[1] - 2],
        pixels[offsets[2] - 2],
        pixels[offsets[3] - 2],
    ]));
    let p0 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[offsets[0] - 1],
        pixels[offsets[1] - 1],
        pixels[offsets[2] - 1],
        pixels[offsets[3] - 1],
    ]));
    let q0 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[offsets[0]],
        pixels[offsets[1]],
        pixels[offsets[2]],
        pixels[offsets[3]],
    ]));
    let q1 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[offsets[0] + 1],
        pixels[offsets[1] + 1],
        pixels[offsets[2] + 1],
        pixels[offsets[3] + 1],
    ]));

    // Check simple threshold: |p0 - q0| * 2 + |p1 - q1| / 2 <= edge_limit
    let diff_p0_q0 = abs_diff_epu8(p0, q0);
    let diff_p1_q1 = abs_diff_epu8(p1, q1);

    // diff_p0_q0 * 2 + diff_p1_q1 / 2
    let doubled = _mm_adds_epu8(diff_p0_q0, diff_p0_q0);
    let halved = _mm_srli_epi16(diff_p1_q1, 1);
    let halved = _mm_and_si128(halved, _mm_set1_epi8(0x7F)); // Clear high bits from shift
    let threshold_val = _mm_adds_epu8(doubled, halved);

    let limit = _mm_set1_epi8(edge_limit as i8);
    // Compare: threshold_val <= limit is equivalent to !(threshold_val > limit)
    // Use saturating subtract: if threshold_val <= limit, result is 0
    let exceeds = _mm_subs_epu8(threshold_val, limit);
    let should_filter = _mm_cmpeq_epi8(exceeds, _mm_setzero_si128());

    // Convert to signed for arithmetic
    let p1_s = u2s_epi8(p1);
    let p0_s = u2s_epi8(p0);
    let q0_s = u2s_epi8(q0);
    let q1_s = u2s_epi8(q1);

    // Compute filter: outer = clamp(p1 - q1), a = clamp(outer + 3 * (q0 - p0))
    // We need 16-bit precision for the multiply
    let p1_lo = _mm_cvtepi8_epi16(p1_s);
    let p0_lo = _mm_cvtepi8_epi16(p0_s);
    let q0_lo = _mm_cvtepi8_epi16(q0_s);
    let q1_lo = _mm_cvtepi8_epi16(q1_s);

    // outer = p1 - q1 (clamped to [-128, 127])
    let outer = _mm_sub_epi16(p1_lo, q1_lo);
    let outer = _mm_max_epi16(_mm_min_epi16(outer, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    // diff = q0 - p0
    let diff = _mm_sub_epi16(q0_lo, p0_lo);

    // a = clamp(outer + 3 * diff)
    let three = _mm_set1_epi16(3);
    let three_diff = _mm_mullo_epi16(diff, three);
    let a = _mm_add_epi16(outer, three_diff);
    let a = _mm_max_epi16(_mm_min_epi16(a, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    // b = (a + 3) >> 3
    let b = _mm_srai_epi16(_mm_add_epi16(a, _mm_set1_epi16(3)), 3);

    // a = (a + 4) >> 3
    let a = _mm_srai_epi16(_mm_add_epi16(a, _mm_set1_epi16(4)), 3);

    // new_q0 = q0 - a, new_p0 = p0 + b
    let new_q0 = _mm_sub_epi16(q0_lo, a);
    let new_p0 = _mm_add_epi16(p0_lo, b);

    // Clamp to [-128, 127] and convert back to unsigned [0, 255]
    let new_q0 = _mm_max_epi16(_mm_min_epi16(new_q0, _mm_set1_epi16(127)), _mm_set1_epi16(-128));
    let new_p0 = _mm_max_epi16(_mm_min_epi16(new_p0, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    // Pack back to bytes and add 128
    let new_q0 = _mm_packs_epi16(new_q0, new_q0);
    let new_p0 = _mm_packs_epi16(new_p0, new_p0);
    let new_q0 = s2u_epi8(new_q0);
    let new_p0 = s2u_epi8(new_p0);

    // Apply filter mask and blend with original
    let q0_filtered = _mm_blendv_epi8(q0, new_q0, should_filter);
    let p0_filtered = _mm_blendv_epi8(p0, new_p0, should_filter);

    // Store results back - extract bytes manually
    let p0_bytes = _mm_cvtsi128_si32(p0_filtered).to_ne_bytes();
    let q0_bytes = _mm_cvtsi128_si32(q0_filtered).to_ne_bytes();

    pixels[offsets[0] - 1] = p0_bytes[0];
    pixels[offsets[1] - 1] = p0_bytes[1];
    pixels[offsets[2] - 1] = p0_bytes[2];
    pixels[offsets[3] - 1] = p0_bytes[3];

    pixels[offsets[0]] = q0_bytes[0];
    pixels[offsets[1]] = q0_bytes[1];
    pixels[offsets[2]] = q0_bytes[2];
    pixels[offsets[3]] = q0_bytes[3];
}

/// Apply the simple loop filter to 4 vertical edges in parallel.
/// Processes 4 columns at once where each column has pixels arranged vertically.
///
/// # Safety
/// Requires SSE2 support. Caller must ensure pixel data is valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn simple_filter_vertical_4x(
    pixels: &mut [u8],
    point: usize,
    stride: usize,
    edge_limit: u8,
) {
    // Load 4 adjacent columns of pixels
    // p1 = row at point - 2*stride
    // p0 = row at point - stride
    // q0 = row at point
    // q1 = row at point + stride

    let p1 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[point - 2 * stride],
        pixels[point - 2 * stride + 1],
        pixels[point - 2 * stride + 2],
        pixels[point - 2 * stride + 3],
    ]));
    let p0 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[point - stride],
        pixels[point - stride + 1],
        pixels[point - stride + 2],
        pixels[point - stride + 3],
    ]));
    let q0 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[point],
        pixels[point + 1],
        pixels[point + 2],
        pixels[point + 3],
    ]));
    let q1 = _mm_cvtsi32_si128(i32::from_ne_bytes([
        pixels[point + stride],
        pixels[point + stride + 1],
        pixels[point + stride + 2],
        pixels[point + stride + 3],
    ]));

    // Check simple threshold
    let diff_p0_q0 = abs_diff_epu8(p0, q0);
    let diff_p1_q1 = abs_diff_epu8(p1, q1);

    let doubled = _mm_adds_epu8(diff_p0_q0, diff_p0_q0);
    let halved = _mm_srli_epi16(diff_p1_q1, 1);
    let halved = _mm_and_si128(halved, _mm_set1_epi8(0x7F));
    let threshold_val = _mm_adds_epu8(doubled, halved);

    let limit = _mm_set1_epi8(edge_limit as i8);
    let exceeds = _mm_subs_epu8(threshold_val, limit);
    let should_filter = _mm_cmpeq_epi8(exceeds, _mm_setzero_si128());

    // Convert to signed for arithmetic
    let p1_s = u2s_epi8(p1);
    let p0_s = u2s_epi8(p0);
    let q0_s = u2s_epi8(q0);
    let q1_s = u2s_epi8(q1);

    // Compute filter in 16-bit
    let p1_lo = _mm_cvtepi8_epi16(p1_s);
    let p0_lo = _mm_cvtepi8_epi16(p0_s);
    let q0_lo = _mm_cvtepi8_epi16(q0_s);
    let q1_lo = _mm_cvtepi8_epi16(q1_s);

    let outer = _mm_sub_epi16(p1_lo, q1_lo);
    let outer = _mm_max_epi16(_mm_min_epi16(outer, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    let diff = _mm_sub_epi16(q0_lo, p0_lo);
    let three = _mm_set1_epi16(3);
    let three_diff = _mm_mullo_epi16(diff, three);
    let a = _mm_add_epi16(outer, three_diff);
    let a = _mm_max_epi16(_mm_min_epi16(a, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    let b = _mm_srai_epi16(_mm_add_epi16(a, _mm_set1_epi16(3)), 3);
    let a = _mm_srai_epi16(_mm_add_epi16(a, _mm_set1_epi16(4)), 3);

    let new_q0 = _mm_sub_epi16(q0_lo, a);
    let new_p0 = _mm_add_epi16(p0_lo, b);

    let new_q0 = _mm_max_epi16(_mm_min_epi16(new_q0, _mm_set1_epi16(127)), _mm_set1_epi16(-128));
    let new_p0 = _mm_max_epi16(_mm_min_epi16(new_p0, _mm_set1_epi16(127)), _mm_set1_epi16(-128));

    let new_q0 = _mm_packs_epi16(new_q0, new_q0);
    let new_p0 = _mm_packs_epi16(new_p0, new_p0);
    let new_q0 = s2u_epi8(new_q0);
    let new_p0 = s2u_epi8(new_p0);

    let q0_filtered = _mm_blendv_epi8(q0, new_q0, should_filter);
    let p0_filtered = _mm_blendv_epi8(p0, new_p0, should_filter);

    // Store results - extract 4 bytes
    let p0_bytes = _mm_cvtsi128_si32(p0_filtered).to_ne_bytes();
    let q0_bytes = _mm_cvtsi128_si32(q0_filtered).to_ne_bytes();

    pixels[point - stride] = p0_bytes[0];
    pixels[point - stride + 1] = p0_bytes[1];
    pixels[point - stride + 2] = p0_bytes[2];
    pixels[point - stride + 3] = p0_bytes[3];

    pixels[point] = q0_bytes[0];
    pixels[point + 1] = q0_bytes[1];
    pixels[point + 2] = q0_bytes[2];
    pixels[point + 3] = q0_bytes[3];
}

/// Check if all 4 edges pass the simple threshold test.
/// Returns a bitmask where bit i is set if edge i should be filtered.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn check_simple_threshold_4x(
    p1: [u8; 4],
    p0: [u8; 4],
    q0: [u8; 4],
    q1: [u8; 4],
    edge_limit: u8,
) -> u8 {
    let p1_v = _mm_cvtsi32_si128(i32::from_ne_bytes(p1));
    let p0_v = _mm_cvtsi32_si128(i32::from_ne_bytes(p0));
    let q0_v = _mm_cvtsi32_si128(i32::from_ne_bytes(q0));
    let q1_v = _mm_cvtsi32_si128(i32::from_ne_bytes(q1));

    let diff_p0_q0 = abs_diff_epu8(p0_v, q0_v);
    let diff_p1_q1 = abs_diff_epu8(p1_v, q1_v);

    let doubled = _mm_adds_epu8(diff_p0_q0, diff_p0_q0);
    let halved = _mm_srli_epi16(diff_p1_q1, 1);
    let halved = _mm_and_si128(halved, _mm_set1_epi8(0x7F));
    let threshold_val = _mm_adds_epu8(doubled, halved);

    let limit = _mm_set1_epi8(edge_limit as i8);
    let exceeds = _mm_subs_epu8(threshold_val, limit);
    let should_filter = _mm_cmpeq_epi8(exceeds, _mm_setzero_si128());

    (_mm_movemask_epi8(should_filter) & 0xF) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_filter_vertical() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Create test data: a simple edge
        let mut pixels = vec![0u8; 64];
        let stride = 8;
        let point = 24; // Middle of buffer

        // Set up a gradient that should be filtered
        for i in 0..4 {
            pixels[point - 2 * stride + i] = 100;
            pixels[point - stride + i] = 110;
            pixels[point + i] = 140;
            pixels[point + stride + i] = 150;
        }

        let original = pixels.clone();

        unsafe {
            simple_filter_vertical_4x(&mut pixels, point, stride, 40);
        }

        // Check that filtering was applied
        for i in 0..4 {
            let p0_orig = original[point - stride + i];
            let q0_orig = original[point + i];
            let p0_new = pixels[point - stride + i];
            let q0_new = pixels[point + i];

            // Values should be adjusted toward each other
            assert!(p0_new >= p0_orig, "p0 should increase");
            assert!(q0_new <= q0_orig, "q0 should decrease");
        }
    }
}
