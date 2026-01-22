//! SIMD-optimized YUV to RGB conversion using x86 SSE4.1 intrinsics.
//!
//! Processes 8 pixels at a time for significant speedup over scalar code.
//! Uses the same formula as libwebp's yuv.h to ensure bit-exact output.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// YUV to RGB conversion constants matching libwebp's yuv.h.
///
/// The scalar code uses:
///   R = clip((mulhi(y, 19077) + mulhi(v, 26149) - 14234) >> 6)
///   G = clip((mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708) >> 6)
///   B = clip((mulhi(y, 19077) + mulhi(u, 33050) - 17685) >> 6)
/// where mulhi(x, c) = (x * c) >> 8
///
/// We combine to: result = (y * Y_COEFF + u * U_COEFF + v * V_COEFF + OFFSET) >> 14
/// So coefficients are shifted left by 6 (multiply by 64).
const Y_COEFF: i32 = 19077;
const V_R_COEFF: i32 = 26149;
const U_G_COEFF: i32 = 6419;
const V_G_COEFF: i32 = 13320;
const U_B_COEFF: i32 = 33050;
const R_OFFSET: i32 = -14234;
const G_OFFSET: i32 = 8708;
const B_OFFSET: i32 = -17685;

// For SIMD: we need to do mulhi then the final shift, so just use these directly
// Final: ((y * 19077) >> 8) + ((v * 26149) >> 8) - 14234, then >> 6

/// Process 8 pixels of YUV to RGB conversion (1:1 Y:U:V mapping).
///
/// Uses the same formula as the scalar code for bit-exact results.
///
/// # Safety
/// Requires SSE4.1. Input slices must have at least 8 elements.
/// Output slice must have at least 24 bytes (8 RGB pixels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
pub unsafe fn yuv_to_rgb_8x(y: &[u8], u: &[u8], v: &[u8], rgb: &mut [u8]) {
    debug_assert!(y.len() >= 8);
    debug_assert!(u.len() >= 8);
    debug_assert!(v.len() >= 8);
    debug_assert!(rgb.len() >= 24);

    // Load 8 Y values and zero-extend to 32-bit
    let y8 = _mm_loadl_epi64(y.as_ptr() as *const __m128i);
    let y16 = _mm_unpacklo_epi8(y8, _mm_setzero_si128());
    let y_lo = _mm_unpacklo_epi16(y16, _mm_setzero_si128());
    let y_hi = _mm_unpackhi_epi16(y16, _mm_setzero_si128());

    // Load 8 U values and zero-extend to 32-bit
    let u8_vec = _mm_loadl_epi64(u.as_ptr() as *const __m128i);
    let u16 = _mm_unpacklo_epi8(u8_vec, _mm_setzero_si128());
    let u_lo = _mm_unpacklo_epi16(u16, _mm_setzero_si128());
    let u_hi = _mm_unpackhi_epi16(u16, _mm_setzero_si128());

    // Load 8 V values and zero-extend to 32-bit
    let v8 = _mm_loadl_epi64(v.as_ptr() as *const __m128i);
    let v16 = _mm_unpacklo_epi8(v8, _mm_setzero_si128());
    let v_lo = _mm_unpacklo_epi16(v16, _mm_setzero_si128());
    let v_hi = _mm_unpackhi_epi16(v16, _mm_setzero_si128());

    // Coefficients
    let c_y = _mm_set1_epi32(Y_COEFF);
    let c_vr = _mm_set1_epi32(V_R_COEFF);
    let c_ug = _mm_set1_epi32(U_G_COEFF);
    let c_vg = _mm_set1_epi32(V_G_COEFF);
    let c_ub = _mm_set1_epi32(U_B_COEFF);
    let r_off = _mm_set1_epi32(R_OFFSET);
    let g_off = _mm_set1_epi32(G_OFFSET);
    let b_off = _mm_set1_epi32(B_OFFSET);

    // Low 4 pixels - compute mulhi equivalent: (x * c) >> 8
    let y_mul_lo = _mm_srai_epi32(_mm_mullo_epi32(y_lo, c_y), 8);
    let v_mul_r_lo = _mm_srai_epi32(_mm_mullo_epi32(v_lo, c_vr), 8);
    let u_mul_g_lo = _mm_srai_epi32(_mm_mullo_epi32(u_lo, c_ug), 8);
    let v_mul_g_lo = _mm_srai_epi32(_mm_mullo_epi32(v_lo, c_vg), 8);
    let u_mul_b_lo = _mm_srai_epi32(_mm_mullo_epi32(u_lo, c_ub), 8);

    let r_lo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_lo, v_mul_r_lo), r_off),
        6,
    );
    let g_lo = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_lo, u_mul_g_lo), v_mul_g_lo),
            g_off,
        ),
        6,
    );
    let b_lo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_lo, u_mul_b_lo), b_off),
        6,
    );

    // High 4 pixels
    let y_mul_hi = _mm_srai_epi32(_mm_mullo_epi32(y_hi, c_y), 8);
    let v_mul_r_hi = _mm_srai_epi32(_mm_mullo_epi32(v_hi, c_vr), 8);
    let u_mul_g_hi = _mm_srai_epi32(_mm_mullo_epi32(u_hi, c_ug), 8);
    let v_mul_g_hi = _mm_srai_epi32(_mm_mullo_epi32(v_hi, c_vg), 8);
    let u_mul_b_hi = _mm_srai_epi32(_mm_mullo_epi32(u_hi, c_ub), 8);

    let r_hi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_hi, v_mul_r_hi), r_off),
        6,
    );
    let g_hi = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_hi, u_mul_g_hi), v_mul_g_hi),
            g_off,
        ),
        6,
    );
    let b_hi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_hi, u_mul_b_hi), b_off),
        6,
    );

    // Pack to bytes with saturation
    let r16 = _mm_packs_epi32(r_lo, r_hi);
    let g16 = _mm_packs_epi32(g_lo, g_hi);
    let b16 = _mm_packs_epi32(b_lo, b_hi);

    let r8 = _mm_packus_epi16(r16, r16);
    let g8 = _mm_packus_epi16(g16, g16);
    let b8 = _mm_packus_epi16(b16, b16);

    // Write RGB interleaved
    let r_bytes: [u8; 16] = core::mem::transmute(r8);
    let g_bytes: [u8; 16] = core::mem::transmute(g8);
    let b_bytes: [u8; 16] = core::mem::transmute(b8);

    for i in 0..8 {
        rgb[i * 3] = r_bytes[i];
        rgb[i * 3 + 1] = g_bytes[i];
        rgb[i * 3 + 2] = b_bytes[i];
    }
}

/// Process 8 pixels of YUV to RGBA conversion (1:1 Y:U:V mapping).
/// Alpha is set to 255.
///
/// Uses the same formula as the scalar code for bit-exact results.
///
/// # Safety
/// Requires SSE4.1. Input slices must have at least 8 elements.
/// Output slice must have at least 32 bytes (8 RGBA pixels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
pub unsafe fn yuv_to_rgba_8x(y: &[u8], u: &[u8], v: &[u8], rgba: &mut [u8]) {
    debug_assert!(y.len() >= 8);
    debug_assert!(u.len() >= 8);
    debug_assert!(v.len() >= 8);
    debug_assert!(rgba.len() >= 32);

    // Load and zero-extend to 32-bit
    let y8 = _mm_loadl_epi64(y.as_ptr() as *const __m128i);
    let y16 = _mm_unpacklo_epi8(y8, _mm_setzero_si128());
    let y_lo = _mm_unpacklo_epi16(y16, _mm_setzero_si128());
    let y_hi = _mm_unpackhi_epi16(y16, _mm_setzero_si128());

    let u8_vec = _mm_loadl_epi64(u.as_ptr() as *const __m128i);
    let u16 = _mm_unpacklo_epi8(u8_vec, _mm_setzero_si128());
    let u_lo = _mm_unpacklo_epi16(u16, _mm_setzero_si128());
    let u_hi = _mm_unpackhi_epi16(u16, _mm_setzero_si128());

    let v8 = _mm_loadl_epi64(v.as_ptr() as *const __m128i);
    let v16 = _mm_unpacklo_epi8(v8, _mm_setzero_si128());
    let v_lo = _mm_unpacklo_epi16(v16, _mm_setzero_si128());
    let v_hi = _mm_unpackhi_epi16(v16, _mm_setzero_si128());

    // Coefficients
    let c_y = _mm_set1_epi32(Y_COEFF);
    let c_vr = _mm_set1_epi32(V_R_COEFF);
    let c_ug = _mm_set1_epi32(U_G_COEFF);
    let c_vg = _mm_set1_epi32(V_G_COEFF);
    let c_ub = _mm_set1_epi32(U_B_COEFF);
    let r_off = _mm_set1_epi32(R_OFFSET);
    let g_off = _mm_set1_epi32(G_OFFSET);
    let b_off = _mm_set1_epi32(B_OFFSET);

    // Low 4 pixels
    let y_mul_lo = _mm_srai_epi32(_mm_mullo_epi32(y_lo, c_y), 8);
    let v_mul_r_lo = _mm_srai_epi32(_mm_mullo_epi32(v_lo, c_vr), 8);
    let u_mul_g_lo = _mm_srai_epi32(_mm_mullo_epi32(u_lo, c_ug), 8);
    let v_mul_g_lo = _mm_srai_epi32(_mm_mullo_epi32(v_lo, c_vg), 8);
    let u_mul_b_lo = _mm_srai_epi32(_mm_mullo_epi32(u_lo, c_ub), 8);

    let r_lo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_lo, v_mul_r_lo), r_off),
        6,
    );
    let g_lo = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_lo, u_mul_g_lo), v_mul_g_lo),
            g_off,
        ),
        6,
    );
    let b_lo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_lo, u_mul_b_lo), b_off),
        6,
    );

    // High 4 pixels
    let y_mul_hi = _mm_srai_epi32(_mm_mullo_epi32(y_hi, c_y), 8);
    let v_mul_r_hi = _mm_srai_epi32(_mm_mullo_epi32(v_hi, c_vr), 8);
    let u_mul_g_hi = _mm_srai_epi32(_mm_mullo_epi32(u_hi, c_ug), 8);
    let v_mul_g_hi = _mm_srai_epi32(_mm_mullo_epi32(v_hi, c_vg), 8);
    let u_mul_b_hi = _mm_srai_epi32(_mm_mullo_epi32(u_hi, c_ub), 8);

    let r_hi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_hi, v_mul_r_hi), r_off),
        6,
    );
    let g_hi = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_hi, u_mul_g_hi), v_mul_g_hi),
            g_off,
        ),
        6,
    );
    let b_hi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_hi, u_mul_b_hi), b_off),
        6,
    );

    // Pack to bytes with saturation
    let r16 = _mm_packs_epi32(r_lo, r_hi);
    let g16 = _mm_packs_epi32(g_lo, g_hi);
    let b16 = _mm_packs_epi32(b_lo, b_hi);

    let r8 = _mm_packus_epi16(r16, r16);
    let g8 = _mm_packus_epi16(g16, g16);
    let b8 = _mm_packus_epi16(b16, b16);

    // Write RGBA interleaved
    let r_bytes: [u8; 16] = core::mem::transmute(r8);
    let g_bytes: [u8; 16] = core::mem::transmute(g8);
    let b_bytes: [u8; 16] = core::mem::transmute(b8);

    for i in 0..8 {
        rgba[i * 4] = r_bytes[i];
        rgba[i * 4 + 1] = g_bytes[i];
        rgba[i * 4 + 2] = b_bytes[i];
        rgba[i * 4 + 3] = 255;
    }
}

/// Process 8 pixels with 4:2:0 chroma subsampling (4 U/V pairs for 8 Y values).
/// Each U/V pair is shared by 2 adjacent Y pixels.
///
/// This matches the layout used by fill_rgba_row_simple and uses the exact
/// same formula as the scalar code:
///   R = clip((mulhi(y, 19077) + mulhi(v, 26149) - 14234) >> 6)
///   G = clip((mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708) >> 6)
///   B = clip((mulhi(y, 19077) + mulhi(u, 33050) - 17685) >> 6)
/// where mulhi(x, c) = (x * c) >> 8
///
/// # Safety
/// Requires SSE4.1. y must have 8 elements, u/v must have 4 elements.
/// rgb output must have 24 bytes (8 RGB pixels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
pub unsafe fn yuv420_to_rgb_8x(
    y: &[u8],
    u: &[u8], // 4 values, each shared by 2 Y pixels
    v: &[u8], // 4 values, each shared by 2 Y pixels
    rgb: &mut [u8],
) {
    debug_assert!(y.len() >= 8);
    debug_assert!(u.len() >= 4);
    debug_assert!(v.len() >= 4);
    debug_assert!(rgb.len() >= 24);

    // Load 8 Y values and zero-extend to 32-bit
    let y8 = _mm_loadl_epi64(y.as_ptr() as *const __m128i);
    let y16 = _mm_unpacklo_epi8(y8, _mm_setzero_si128());
    let y_lo = _mm_unpacklo_epi16(y16, _mm_setzero_si128());
    let y_hi = _mm_unpackhi_epi16(y16, _mm_setzero_si128());

    // Load 4 U values and duplicate each to get 8 values: [u0,u0,u1,u1,u2,u2,u3,u3]
    let u4 = _mm_cvtsi32_si128(i32::from_ne_bytes([u[0], u[1], u[2], u[3]]));
    let u8_dup = _mm_unpacklo_epi8(u4, u4);
    let u16 = _mm_unpacklo_epi8(u8_dup, _mm_setzero_si128());
    let u_lo = _mm_unpacklo_epi16(u16, _mm_setzero_si128());
    let u_hi = _mm_unpackhi_epi16(u16, _mm_setzero_si128());

    // Same for V
    let v4 = _mm_cvtsi32_si128(i32::from_ne_bytes([v[0], v[1], v[2], v[3]]));
    let v8_dup = _mm_unpacklo_epi8(v4, v4);
    let v16 = _mm_unpacklo_epi8(v8_dup, _mm_setzero_si128());
    let v_lo = _mm_unpacklo_epi16(v16, _mm_setzero_si128());
    let v_hi = _mm_unpackhi_epi16(v16, _mm_setzero_si128());

    // Coefficients (matching scalar mulhi which does >> 8)
    let c_y = _mm_set1_epi32(Y_COEFF);
    let c_vr = _mm_set1_epi32(V_R_COEFF);
    let c_ug = _mm_set1_epi32(U_G_COEFF);
    let c_vg = _mm_set1_epi32(V_G_COEFF);
    let c_ub = _mm_set1_epi32(U_B_COEFF);
    let r_off = _mm_set1_epi32(R_OFFSET);
    let g_off = _mm_set1_epi32(G_OFFSET);
    let b_off = _mm_set1_epi32(B_OFFSET);

    // Compute Y * 19077 >> 8 for all pixels (mulhi equivalent)
    // Actually compute: (y * coeff) >> 8 + (v * coeff) >> 8 + offset, then >> 6
    // We can combine: ((y * coeff + v * coeff) >> 8 + offset) >> 6
    // But to match scalar exactly: mulhi(y,c) = (y*c)>>8, then add, then >>6
    //
    // For exact match, we compute each mulhi separately then combine

    // Low 4 pixels
    let y_mul_lo = _mm_srai_epi32(_mm_mullo_epi32(y_lo, c_y), 8);
    let v_mul_r_lo = _mm_srai_epi32(_mm_mullo_epi32(v_lo, c_vr), 8);
    let u_mul_g_lo = _mm_srai_epi32(_mm_mullo_epi32(u_lo, c_ug), 8);
    let v_mul_g_lo = _mm_srai_epi32(_mm_mullo_epi32(v_lo, c_vg), 8);
    let u_mul_b_lo = _mm_srai_epi32(_mm_mullo_epi32(u_lo, c_ub), 8);

    // R = (y_mul + v_mul_r + r_off) >> 6
    let r_lo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_lo, v_mul_r_lo), r_off),
        6,
    );
    // G = (y_mul - u_mul_g - v_mul_g + g_off) >> 6
    let g_lo = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_lo, u_mul_g_lo), v_mul_g_lo),
            g_off,
        ),
        6,
    );
    // B = (y_mul + u_mul_b + b_off) >> 6
    let b_lo = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_lo, u_mul_b_lo), b_off),
        6,
    );

    // High 4 pixels
    let y_mul_hi = _mm_srai_epi32(_mm_mullo_epi32(y_hi, c_y), 8);
    let v_mul_r_hi = _mm_srai_epi32(_mm_mullo_epi32(v_hi, c_vr), 8);
    let u_mul_g_hi = _mm_srai_epi32(_mm_mullo_epi32(u_hi, c_ug), 8);
    let v_mul_g_hi = _mm_srai_epi32(_mm_mullo_epi32(v_hi, c_vg), 8);
    let u_mul_b_hi = _mm_srai_epi32(_mm_mullo_epi32(u_hi, c_ub), 8);

    let r_hi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_hi, v_mul_r_hi), r_off),
        6,
    );
    let g_hi = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_hi, u_mul_g_hi), v_mul_g_hi),
            g_off,
        ),
        6,
    );
    let b_hi = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_hi, u_mul_b_hi), b_off),
        6,
    );

    // Pack to 16-bit with signed saturation, then to 8-bit with unsigned saturation
    let r16 = _mm_packs_epi32(r_lo, r_hi);
    let g16 = _mm_packs_epi32(g_lo, g_hi);
    let b16 = _mm_packs_epi32(b_lo, b_hi);

    let r8 = _mm_packus_epi16(r16, r16);
    let g8 = _mm_packus_epi16(g16, g16);
    let b8 = _mm_packus_epi16(b16, b16);

    // Write RGB interleaved
    let r_bytes: [u8; 16] = core::mem::transmute(r8);
    let g_bytes: [u8; 16] = core::mem::transmute(g8);
    let b_bytes: [u8; 16] = core::mem::transmute(b8);

    for i in 0..8 {
        rgb[i * 3] = r_bytes[i];
        rgb[i * 3 + 1] = g_bytes[i];
        rgb[i * 3 + 2] = b_bytes[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference scalar implementation matching yuv.rs
    fn scalar_yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
        fn mulhi(val: u8, coeff: u16) -> i32 {
            ((u32::from(val) * u32::from(coeff)) >> 8) as i32
        }
        fn clip(v: i32) -> u8 {
            (v >> 6).clamp(0, 255) as u8
        }
        let r = clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234);
        let g = clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708);
        let b = clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685);
        (r, g, b)
    }

    #[test]
    fn test_yuv_to_rgb_simd_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        // Test with various YUV values
        let test_cases: [(u8, u8, u8); 8] = [
            (128, 128, 128), // Gray
            (255, 128, 128), // Bright gray
            (0, 128, 128),   // Dark
            (203, 40, 42),   // Test values from yuv.rs
            (77, 34, 97),    // From test_fancy_grid
            (162, 101, 167), // From test_fancy_grid
            (202, 84, 150),  // From test_fancy_grid
            (185, 101, 167), // From test_fancy_grid
        ];

        let y: Vec<u8> = test_cases.iter().map(|(y, _, _)| *y).collect();
        let u: Vec<u8> = test_cases.iter().map(|(_, u, _)| *u).collect();
        let v: Vec<u8> = test_cases.iter().map(|(_, _, v)| *v).collect();
        let mut rgb_simd = [0u8; 24];

        unsafe {
            yuv_to_rgb_8x(&y, &u, &v, &mut rgb_simd);
        }

        for (i, &(y_val, u_val, v_val)) in test_cases.iter().enumerate() {
            let (r_scalar, g_scalar, b_scalar) = scalar_yuv_to_rgb(y_val, u_val, v_val);
            let r_simd = rgb_simd[i * 3];
            let g_simd = rgb_simd[i * 3 + 1];
            let b_simd = rgb_simd[i * 3 + 2];

            assert_eq!(
                r_simd, r_scalar,
                "R mismatch at {}: SIMD={}, scalar={} (Y={}, U={}, V={})",
                i, r_simd, r_scalar, y_val, u_val, v_val
            );
            assert_eq!(
                g_simd, g_scalar,
                "G mismatch at {}: SIMD={}, scalar={} (Y={}, U={}, V={})",
                i, g_simd, g_scalar, y_val, u_val, v_val
            );
            assert_eq!(
                b_simd, b_scalar,
                "B mismatch at {}: SIMD={}, scalar={} (Y={}, U={}, V={})",
                i, b_simd, b_scalar, y_val, u_val, v_val
            );
        }
    }

    #[test]
    fn test_yuv420_to_rgb_simd_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        // Test with 4:2:0 subsampling (4 U/V pairs for 8 Y values)
        let y = [77u8, 162, 202, 185, 28, 13, 199, 182];
        let u = [34u8, 101, 123, 163]; // Each shared by 2 Y pixels
        let v = [97u8, 167, 149, 23];
        let mut rgb_simd = [0u8; 24];

        unsafe {
            yuv420_to_rgb_8x(&y, &u, &v, &mut rgb_simd);
        }

        // Compare with scalar - each U/V pair is used for 2 adjacent Y pixels
        for i in 0..8 {
            let y_val = y[i];
            let u_val = u[i / 2];
            let v_val = v[i / 2];
            let (r_scalar, g_scalar, b_scalar) = scalar_yuv_to_rgb(y_val, u_val, v_val);

            let r_simd = rgb_simd[i * 3];
            let g_simd = rgb_simd[i * 3 + 1];
            let b_simd = rgb_simd[i * 3 + 2];

            assert_eq!(
                r_simd, r_scalar,
                "R mismatch at {}: SIMD={}, scalar={} (Y={}, U={}, V={})",
                i, r_simd, r_scalar, y_val, u_val, v_val
            );
            assert_eq!(
                g_simd, g_scalar,
                "G mismatch at {}: SIMD={}, scalar={} (Y={}, U={}, V={})",
                i, g_simd, g_scalar, y_val, u_val, v_val
            );
            assert_eq!(
                b_simd, b_scalar,
                "B mismatch at {}: SIMD={}, scalar={} (Y={}, U={}, V={})",
                i, b_simd, b_scalar, y_val, u_val, v_val
            );
        }
    }

    #[test]
    fn test_yuv_conversions_match_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        // Test the exact values from yuv.rs tests
        let (y, u, v) = (203, 40, 42);
        let (r, g, b) = scalar_yuv_to_rgb(y, u, v);
        assert_eq!(r, 80, "R mismatch");
        assert_eq!(g, 255, "G mismatch");
        assert_eq!(b, 40, "B mismatch");
    }
}
