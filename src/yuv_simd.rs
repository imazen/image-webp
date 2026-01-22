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

/// Compute fancy chroma interpolation: (9*a + 3*b + 3*c + d + 8) / 16
/// Using libwebp's efficient approach with _mm_avg_epu8.
///
/// The formula is computed as:
///   result = (a + m + 1) / 2
///   where m = (k + t + 1) / 2 - correction
///   where k = (a + b + c + d) / 4
///   where s = (a + d + 1) / 2, t = (b + c + 1) / 2
///
/// # Safety
/// Requires SSE2. All input registers must be valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn fancy_upsample_16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i) {
    let one = _mm_set1_epi8(1);

    // s = (a + d + 1) / 2
    let s = _mm_avg_epu8(a, d);
    // t = (b + c + 1) / 2
    let t = _mm_avg_epu8(b, c);
    // st = s ^ t
    let st = _mm_xor_si128(s, t);

    // ad = a ^ d
    let ad = _mm_xor_si128(a, d);
    // bc = b ^ c
    let bc = _mm_xor_si128(b, c);

    // k = (s + t + 1) / 2 - ((a^d) | (b^c) | (s^t)) & 1
    // This computes (a + b + c + d) / 4 with proper rounding
    let t1 = _mm_or_si128(ad, bc);
    let t2 = _mm_or_si128(t1, st);
    let t3 = _mm_and_si128(t2, one);
    let t4 = _mm_avg_epu8(s, t);
    let k = _mm_sub_epi8(t4, t3);

    // m1 = (k + t + 1) / 2 - (((b^c) & (s^t)) | (k^t)) & 1
    // This computes (a + 3*b + 3*c + d) / 8
    let tmp1 = _mm_avg_epu8(k, t);
    let tmp2 = _mm_and_si128(bc, st);
    let tmp3 = _mm_xor_si128(k, t);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let m1 = _mm_sub_epi8(tmp1, tmp5);

    // m2 = (k + s + 1) / 2 - (((a^d) & (s^t)) | (k^s)) & 1
    // This computes (3*a + b + c + 3*d) / 8
    let tmp1 = _mm_avg_epu8(k, s);
    let tmp2 = _mm_and_si128(ad, st);
    let tmp3 = _mm_xor_si128(k, s);
    let tmp4 = _mm_or_si128(tmp2, tmp3);
    let tmp5 = _mm_and_si128(tmp4, one);
    let m2 = _mm_sub_epi8(tmp1, tmp5);

    // diag1 = (9*a + 3*b + 3*c + d + 8) / 16 = (a + m1 + 1) / 2
    let diag1 = _mm_avg_epu8(a, m1);
    // diag2 = (3*a + 9*b + c + 3*d + 8) / 16 = (b + m2 + 1) / 2
    let diag2 = _mm_avg_epu8(b, m2);

    (diag1, diag2)
}

/// Process a row with fancy upsampling and YUV to RGB conversion.
/// Takes 16 Y pixels and 9 U/V pixels from two adjacent rows,
/// producing 16 RGB pixels.
///
/// This handles the "top" row output where diag1 goes with a and diag2 goes with b.
///
/// # Safety
/// Requires SSE4.1. y must have 16 elements, u_row1/u_row2/v_row1/v_row2 must
/// have 9 elements each. rgb output must have 48 bytes (16 RGB pixels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
pub unsafe fn fancy_upsample_row_16(
    y: &[u8],        // 16 Y pixels
    u_row1: &[u8],   // 9 U pixels from top chroma row
    u_row2: &[u8],   // 9 U pixels from bottom chroma row
    v_row1: &[u8],   // 9 V pixels from top chroma row
    v_row2: &[u8],   // 9 V pixels from bottom chroma row
    rgb: &mut [u8],  // 48 bytes output (16 RGB pixels)
    is_top_row: bool, // true for top output row, false for bottom
) {
    debug_assert!(y.len() >= 16);
    debug_assert!(u_row1.len() >= 9);
    debug_assert!(u_row2.len() >= 9);
    debug_assert!(v_row1.len() >= 9);
    debug_assert!(v_row2.len() >= 9);
    debug_assert!(rgb.len() >= 48);

    // Load U values from both rows
    // We need 9 values: positions 0-8 for interpolating 16 output pixels
    // a[i] = row1[i], b[i] = row1[i+1], c[i] = row2[i], d[i] = row2[i+1]
    let u_a = _mm_loadu_si128(u_row1.as_ptr() as *const __m128i);
    let u_b = _mm_loadu_si128(u_row1.as_ptr().add(1) as *const __m128i);
    let u_c = _mm_loadu_si128(u_row2.as_ptr() as *const __m128i);
    let u_d = _mm_loadu_si128(u_row2.as_ptr().add(1) as *const __m128i);

    let v_a = _mm_loadu_si128(v_row1.as_ptr() as *const __m128i);
    let v_b = _mm_loadu_si128(v_row1.as_ptr().add(1) as *const __m128i);
    let v_c = _mm_loadu_si128(v_row2.as_ptr() as *const __m128i);
    let v_d = _mm_loadu_si128(v_row2.as_ptr().add(1) as *const __m128i);

    // Compute upsampled U/V
    let (u_diag1, u_diag2) = fancy_upsample_16(u_a, u_b, u_c, u_d);
    let (v_diag1, v_diag2) = fancy_upsample_16(v_a, v_b, v_c, v_d);

    // For the top row: output[2i] uses diag1, output[2i+1] uses diag2
    // For the bottom row: output[2i] uses diag2, output[2i+1] uses diag1
    let (u_even, u_odd, v_even, v_odd) = if is_top_row {
        (u_diag1, u_diag2, v_diag1, v_diag2)
    } else {
        (u_diag2, u_diag1, v_diag2, v_diag1)
    };

    // Interleave to get per-pixel U/V values:
    // even/odd -> pixel 0, 1, 2, 3, ...
    // u_interleaved[i] = if i%2==0 { u_even[i/2] } else { u_odd[i/2] }
    let u_lo = _mm_unpacklo_epi8(u_even, u_odd); // u0 u1 u2 u3 u4 u5 u6 u7 ...
    let v_lo = _mm_unpacklo_epi8(v_even, v_odd);

    // Load Y and convert first 8 pixels
    let y8_0 = _mm_loadl_epi64(y.as_ptr() as *const __m128i);
    let y16_0 = _mm_unpacklo_epi8(y8_0, _mm_setzero_si128());
    let y_lo_0 = _mm_unpacklo_epi16(y16_0, _mm_setzero_si128());
    let y_hi_0 = _mm_unpackhi_epi16(y16_0, _mm_setzero_si128());

    // Extract U/V for first 8 pixels
    let u16_0 = _mm_unpacklo_epi8(u_lo, _mm_setzero_si128());
    let u_lo_0 = _mm_unpacklo_epi16(u16_0, _mm_setzero_si128());
    let u_hi_0 = _mm_unpackhi_epi16(u16_0, _mm_setzero_si128());

    let v16_0 = _mm_unpacklo_epi8(v_lo, _mm_setzero_si128());
    let v_lo_0 = _mm_unpacklo_epi16(v16_0, _mm_setzero_si128());
    let v_hi_0 = _mm_unpackhi_epi16(v16_0, _mm_setzero_si128());

    // Convert first 8 pixels
    let c_y = _mm_set1_epi32(Y_COEFF);
    let c_vr = _mm_set1_epi32(V_R_COEFF);
    let c_ug = _mm_set1_epi32(U_G_COEFF);
    let c_vg = _mm_set1_epi32(V_G_COEFF);
    let c_ub = _mm_set1_epi32(U_B_COEFF);
    let r_off = _mm_set1_epi32(R_OFFSET);
    let g_off = _mm_set1_epi32(G_OFFSET);
    let b_off = _mm_set1_epi32(B_OFFSET);

    // First 4 pixels
    let y_mul_0 = _mm_srai_epi32(_mm_mullo_epi32(y_lo_0, c_y), 8);
    let v_mul_r_0 = _mm_srai_epi32(_mm_mullo_epi32(v_lo_0, c_vr), 8);
    let u_mul_g_0 = _mm_srai_epi32(_mm_mullo_epi32(u_lo_0, c_ug), 8);
    let v_mul_g_0 = _mm_srai_epi32(_mm_mullo_epi32(v_lo_0, c_vg), 8);
    let u_mul_b_0 = _mm_srai_epi32(_mm_mullo_epi32(u_lo_0, c_ub), 8);

    let r_0 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_0, v_mul_r_0), r_off),
        6,
    );
    let g_0 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_0, u_mul_g_0), v_mul_g_0),
            g_off,
        ),
        6,
    );
    let b_0 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_0, u_mul_b_0), b_off),
        6,
    );

    // Next 4 pixels
    let y_mul_1 = _mm_srai_epi32(_mm_mullo_epi32(y_hi_0, c_y), 8);
    let v_mul_r_1 = _mm_srai_epi32(_mm_mullo_epi32(v_hi_0, c_vr), 8);
    let u_mul_g_1 = _mm_srai_epi32(_mm_mullo_epi32(u_hi_0, c_ug), 8);
    let v_mul_g_1 = _mm_srai_epi32(_mm_mullo_epi32(v_hi_0, c_vg), 8);
    let u_mul_b_1 = _mm_srai_epi32(_mm_mullo_epi32(u_hi_0, c_ub), 8);

    let r_1 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_1, v_mul_r_1), r_off),
        6,
    );
    let g_1 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_1, u_mul_g_1), v_mul_g_1),
            g_off,
        ),
        6,
    );
    let b_1 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_1, u_mul_b_1), b_off),
        6,
    );

    // Pack first 8 pixels
    let r16_0 = _mm_packs_epi32(r_0, r_1);
    let g16_0 = _mm_packs_epi32(g_0, g_1);
    let b16_0 = _mm_packs_epi32(b_0, b_1);

    let r8_0 = _mm_packus_epi16(r16_0, r16_0);
    let g8_0 = _mm_packus_epi16(g16_0, g16_0);
    let b8_0 = _mm_packus_epi16(b16_0, b16_0);

    // Write first 8 pixels
    let r_bytes: [u8; 16] = core::mem::transmute(r8_0);
    let g_bytes: [u8; 16] = core::mem::transmute(g8_0);
    let b_bytes: [u8; 16] = core::mem::transmute(b8_0);

    for i in 0..8 {
        rgb[i * 3] = r_bytes[i];
        rgb[i * 3 + 1] = g_bytes[i];
        rgb[i * 3 + 2] = b_bytes[i];
    }

    // Now process pixels 8-15
    let u_hi = _mm_unpackhi_epi8(u_even, u_odd);
    let v_hi = _mm_unpackhi_epi8(v_even, v_odd);

    let y8_1 = _mm_loadl_epi64(y.as_ptr().add(8) as *const __m128i);
    let y16_1 = _mm_unpacklo_epi8(y8_1, _mm_setzero_si128());
    let y_lo_1 = _mm_unpacklo_epi16(y16_1, _mm_setzero_si128());
    let y_hi_1 = _mm_unpackhi_epi16(y16_1, _mm_setzero_si128());

    let u16_1 = _mm_unpacklo_epi8(u_hi, _mm_setzero_si128());
    let u_lo_1 = _mm_unpacklo_epi16(u16_1, _mm_setzero_si128());
    let u_hi_1 = _mm_unpackhi_epi16(u16_1, _mm_setzero_si128());

    let v16_1 = _mm_unpacklo_epi8(v_hi, _mm_setzero_si128());
    let v_lo_1 = _mm_unpacklo_epi16(v16_1, _mm_setzero_si128());
    let v_hi_1 = _mm_unpackhi_epi16(v16_1, _mm_setzero_si128());

    // Convert pixels 8-11
    let y_mul_2 = _mm_srai_epi32(_mm_mullo_epi32(y_lo_1, c_y), 8);
    let v_mul_r_2 = _mm_srai_epi32(_mm_mullo_epi32(v_lo_1, c_vr), 8);
    let u_mul_g_2 = _mm_srai_epi32(_mm_mullo_epi32(u_lo_1, c_ug), 8);
    let v_mul_g_2 = _mm_srai_epi32(_mm_mullo_epi32(v_lo_1, c_vg), 8);
    let u_mul_b_2 = _mm_srai_epi32(_mm_mullo_epi32(u_lo_1, c_ub), 8);

    let r_2 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_2, v_mul_r_2), r_off),
        6,
    );
    let g_2 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_2, u_mul_g_2), v_mul_g_2),
            g_off,
        ),
        6,
    );
    let b_2 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_2, u_mul_b_2), b_off),
        6,
    );

    // Convert pixels 12-15
    let y_mul_3 = _mm_srai_epi32(_mm_mullo_epi32(y_hi_1, c_y), 8);
    let v_mul_r_3 = _mm_srai_epi32(_mm_mullo_epi32(v_hi_1, c_vr), 8);
    let u_mul_g_3 = _mm_srai_epi32(_mm_mullo_epi32(u_hi_1, c_ug), 8);
    let v_mul_g_3 = _mm_srai_epi32(_mm_mullo_epi32(v_hi_1, c_vg), 8);
    let u_mul_b_3 = _mm_srai_epi32(_mm_mullo_epi32(u_hi_1, c_ub), 8);

    let r_3 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_3, v_mul_r_3), r_off),
        6,
    );
    let g_3 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_3, u_mul_g_3), v_mul_g_3),
            g_off,
        ),
        6,
    );
    let b_3 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_3, u_mul_b_3), b_off),
        6,
    );

    // Pack pixels 8-15
    let r16_1 = _mm_packs_epi32(r_2, r_3);
    let g16_1 = _mm_packs_epi32(g_2, g_3);
    let b16_1 = _mm_packs_epi32(b_2, b_3);

    let r8_1 = _mm_packus_epi16(r16_1, r16_1);
    let g8_1 = _mm_packus_epi16(g16_1, g16_1);
    let b8_1 = _mm_packus_epi16(b16_1, b16_1);

    // Write pixels 8-15
    let r_bytes: [u8; 16] = core::mem::transmute(r8_1);
    let g_bytes: [u8; 16] = core::mem::transmute(g8_1);
    let b_bytes: [u8; 16] = core::mem::transmute(b8_1);

    for i in 0..8 {
        rgb[24 + i * 3] = r_bytes[i];
        rgb[24 + i * 3 + 1] = g_bytes[i];
        rgb[24 + i * 3 + 2] = b_bytes[i];
    }
}

/// Process 8 pixel pairs (16 Y pixels) with fancy upsampling and YUV->RGB conversion.
///
/// This matches the structure of `fill_row_fancy_with_2_uv_rows` but processes
/// 16 pixels at a time using SIMD.
///
/// # Arguments
/// * `y_row` - 16 Y values (positions 1..17 of the row, skipping the first edge pixel)
/// * `u_row_1` - 9 U values from top chroma row (positions 0..9)
/// * `u_row_2` - 9 U values from bottom chroma row (positions 0..9)
/// * `v_row_1` - 9 V values from top chroma row
/// * `v_row_2` - 9 V values from bottom chroma row
/// * `rgb` - Output buffer for 16 RGB pixels (48 bytes)
///
/// # Safety
/// Requires SSE4.1. All input slices must have the required minimum lengths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
pub unsafe fn fancy_upsample_8_pairs(
    y_row: &[u8],     // 16 Y values
    u_row_1: &[u8],   // 9 U values from top chroma row
    u_row_2: &[u8],   // 9 U values from bottom chroma row
    v_row_1: &[u8],   // 9 V values from top chroma row
    v_row_2: &[u8],   // 9 V values from bottom chroma row
    rgb: &mut [u8],   // 48 bytes output
) {
    debug_assert!(y_row.len() >= 16);
    debug_assert!(u_row_1.len() >= 9);
    debug_assert!(u_row_2.len() >= 9);
    debug_assert!(v_row_1.len() >= 9);
    debug_assert!(v_row_2.len() >= 9);
    debug_assert!(rgb.len() >= 48);

    // Load 8 chroma values from each row (we'll use windows of 2)
    // For pixel pair i, we need u_row[i], u_row[i+1] from both rows
    let u_a = _mm_loadl_epi64(u_row_1.as_ptr() as *const __m128i); // u[0..7]
    let u_b = _mm_loadl_epi64(u_row_1.as_ptr().add(1) as *const __m128i); // u[1..8]
    let u_c = _mm_loadl_epi64(u_row_2.as_ptr() as *const __m128i); // u[0..7]
    let u_d = _mm_loadl_epi64(u_row_2.as_ptr().add(1) as *const __m128i); // u[1..8]

    let v_a = _mm_loadl_epi64(v_row_1.as_ptr() as *const __m128i);
    let v_b = _mm_loadl_epi64(v_row_1.as_ptr().add(1) as *const __m128i);
    let v_c = _mm_loadl_epi64(v_row_2.as_ptr() as *const __m128i);
    let v_d = _mm_loadl_epi64(v_row_2.as_ptr().add(1) as *const __m128i);

    // Compute fancy upsampled U/V for 8 pairs
    // diag1 = (9*a + 3*b + 3*c + d + 8) / 16 - used for first pixel of pair
    // diag2 = (3*a + 9*b + c + 3*d + 8) / 16 - used for second pixel of pair
    let (u_diag1, u_diag2) = fancy_upsample_16(u_a, u_b, u_c, u_d);
    let (v_diag1, v_diag2) = fancy_upsample_16(v_a, v_b, v_c, v_d);

    // Interleave to get per-pixel U/V: [diag1[0], diag2[0], diag1[1], diag2[1], ...]
    let u_interleaved = _mm_unpacklo_epi8(u_diag1, u_diag2);
    let v_interleaved = _mm_unpacklo_epi8(v_diag1, v_diag2);

    // Load Y values
    let y_vec = _mm_loadu_si128(y_row.as_ptr() as *const __m128i);

    // Process first 8 pixels (low half of Y and interleaved U/V)
    let y_lo = _mm_unpacklo_epi8(y_vec, _mm_setzero_si128());
    let u_lo = _mm_unpacklo_epi8(u_interleaved, _mm_setzero_si128());
    let v_lo = _mm_unpacklo_epi8(v_interleaved, _mm_setzero_si128());

    // Extend to 32-bit for first 4 pixels
    let y_0 = _mm_unpacklo_epi16(y_lo, _mm_setzero_si128());
    let u_0 = _mm_unpacklo_epi16(u_lo, _mm_setzero_si128());
    let v_0 = _mm_unpacklo_epi16(v_lo, _mm_setzero_si128());

    let y_1 = _mm_unpackhi_epi16(y_lo, _mm_setzero_si128());
    let u_1 = _mm_unpackhi_epi16(u_lo, _mm_setzero_si128());
    let v_1 = _mm_unpackhi_epi16(v_lo, _mm_setzero_si128());

    // Coefficients
    let c_y = _mm_set1_epi32(Y_COEFF);
    let c_vr = _mm_set1_epi32(V_R_COEFF);
    let c_ug = _mm_set1_epi32(U_G_COEFF);
    let c_vg = _mm_set1_epi32(V_G_COEFF);
    let c_ub = _mm_set1_epi32(U_B_COEFF);
    let r_off = _mm_set1_epi32(R_OFFSET);
    let g_off = _mm_set1_epi32(G_OFFSET);
    let b_off = _mm_set1_epi32(B_OFFSET);

    // Convert pixels 0-3
    let y_mul_0 = _mm_srai_epi32(_mm_mullo_epi32(y_0, c_y), 8);
    let v_mul_r_0 = _mm_srai_epi32(_mm_mullo_epi32(v_0, c_vr), 8);
    let u_mul_g_0 = _mm_srai_epi32(_mm_mullo_epi32(u_0, c_ug), 8);
    let v_mul_g_0 = _mm_srai_epi32(_mm_mullo_epi32(v_0, c_vg), 8);
    let u_mul_b_0 = _mm_srai_epi32(_mm_mullo_epi32(u_0, c_ub), 8);

    let r_0 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_0, v_mul_r_0), r_off),
        6,
    );
    let g_0 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_0, u_mul_g_0), v_mul_g_0),
            g_off,
        ),
        6,
    );
    let b_0 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_0, u_mul_b_0), b_off),
        6,
    );

    // Convert pixels 4-7
    let y_mul_1 = _mm_srai_epi32(_mm_mullo_epi32(y_1, c_y), 8);
    let v_mul_r_1 = _mm_srai_epi32(_mm_mullo_epi32(v_1, c_vr), 8);
    let u_mul_g_1 = _mm_srai_epi32(_mm_mullo_epi32(u_1, c_ug), 8);
    let v_mul_g_1 = _mm_srai_epi32(_mm_mullo_epi32(v_1, c_vg), 8);
    let u_mul_b_1 = _mm_srai_epi32(_mm_mullo_epi32(u_1, c_ub), 8);

    let r_1 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_1, v_mul_r_1), r_off),
        6,
    );
    let g_1 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_1, u_mul_g_1), v_mul_g_1),
            g_off,
        ),
        6,
    );
    let b_1 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_1, u_mul_b_1), b_off),
        6,
    );

    // Pack first 8 pixels
    let r16_lo = _mm_packs_epi32(r_0, r_1);
    let g16_lo = _mm_packs_epi32(g_0, g_1);
    let b16_lo = _mm_packs_epi32(b_0, b_1);

    // Process second 8 pixels (high half)
    let y_hi = _mm_unpackhi_epi8(y_vec, _mm_setzero_si128());
    let u_hi = _mm_unpackhi_epi8(u_interleaved, _mm_setzero_si128());
    let v_hi = _mm_unpackhi_epi8(v_interleaved, _mm_setzero_si128());

    let y_2 = _mm_unpacklo_epi16(y_hi, _mm_setzero_si128());
    let u_2 = _mm_unpacklo_epi16(u_hi, _mm_setzero_si128());
    let v_2 = _mm_unpacklo_epi16(v_hi, _mm_setzero_si128());

    let y_3 = _mm_unpackhi_epi16(y_hi, _mm_setzero_si128());
    let u_3 = _mm_unpackhi_epi16(u_hi, _mm_setzero_si128());
    let v_3 = _mm_unpackhi_epi16(v_hi, _mm_setzero_si128());

    // Convert pixels 8-11
    let y_mul_2 = _mm_srai_epi32(_mm_mullo_epi32(y_2, c_y), 8);
    let v_mul_r_2 = _mm_srai_epi32(_mm_mullo_epi32(v_2, c_vr), 8);
    let u_mul_g_2 = _mm_srai_epi32(_mm_mullo_epi32(u_2, c_ug), 8);
    let v_mul_g_2 = _mm_srai_epi32(_mm_mullo_epi32(v_2, c_vg), 8);
    let u_mul_b_2 = _mm_srai_epi32(_mm_mullo_epi32(u_2, c_ub), 8);

    let r_2 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_2, v_mul_r_2), r_off),
        6,
    );
    let g_2 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_2, u_mul_g_2), v_mul_g_2),
            g_off,
        ),
        6,
    );
    let b_2 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_2, u_mul_b_2), b_off),
        6,
    );

    // Convert pixels 12-15
    let y_mul_3 = _mm_srai_epi32(_mm_mullo_epi32(y_3, c_y), 8);
    let v_mul_r_3 = _mm_srai_epi32(_mm_mullo_epi32(v_3, c_vr), 8);
    let u_mul_g_3 = _mm_srai_epi32(_mm_mullo_epi32(u_3, c_ug), 8);
    let v_mul_g_3 = _mm_srai_epi32(_mm_mullo_epi32(v_3, c_vg), 8);
    let u_mul_b_3 = _mm_srai_epi32(_mm_mullo_epi32(u_3, c_ub), 8);

    let r_3 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_3, v_mul_r_3), r_off),
        6,
    );
    let g_3 = _mm_srai_epi32(
        _mm_add_epi32(
            _mm_sub_epi32(_mm_sub_epi32(y_mul_3, u_mul_g_3), v_mul_g_3),
            g_off,
        ),
        6,
    );
    let b_3 = _mm_srai_epi32(
        _mm_add_epi32(_mm_add_epi32(y_mul_3, u_mul_b_3), b_off),
        6,
    );

    // Pack second 8 pixels
    let r16_hi = _mm_packs_epi32(r_2, r_3);
    let g16_hi = _mm_packs_epi32(g_2, g_3);
    let b16_hi = _mm_packs_epi32(b_2, b_3);

    // Combine and pack to bytes
    let r8 = _mm_packus_epi16(r16_lo, r16_hi);
    let g8 = _mm_packus_epi16(g16_lo, g16_hi);
    let b8 = _mm_packus_epi16(b16_lo, b16_hi);

    // Write interleaved RGB
    let r_bytes: [u8; 16] = core::mem::transmute(r8);
    let g_bytes: [u8; 16] = core::mem::transmute(g8);
    let b_bytes: [u8; 16] = core::mem::transmute(b8);

    for i in 0..16 {
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

    /// Reference scalar fancy chroma interpolation matching yuv.rs
    fn get_fancy_chroma_value(main: u8, secondary1: u8, secondary2: u8, tertiary: u8) -> u8 {
        let val0 = u16::from(main);
        let val1 = u16::from(secondary1);
        let val2 = u16::from(secondary2);
        let val3 = u16::from(tertiary);
        ((9 * val0 + 3 * val1 + 3 * val2 + val3 + 8) / 16) as u8
    }

    #[test]
    fn test_fancy_upsample_8_pairs_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        // Use test data from yuv.rs test_fancy_grid but expanded for 16 pixels
        // Y values (16 values starting from position 1 in the row)
        let y_row: [u8; 16] = [
            77, 162, 202, 185, 28, 13, 199, 182, 135, 147, 164, 135, 66, 27, 171, 130,
        ];

        // U/V values (9 values from each row for 8 pixel pairs)
        let u_row_1: [u8; 9] = [34, 101, 84, 123, 163, 90, 110, 140, 120];
        let u_row_2: [u8; 9] = [123, 163, 133, 150, 100, 80, 95, 105, 115];
        let v_row_1: [u8; 9] = [97, 167, 150, 149, 23, 45, 67, 89, 100];
        let v_row_2: [u8; 9] = [149, 23, 86, 100, 120, 55, 75, 95, 110];

        let mut rgb_simd = [0u8; 48];
        unsafe {
            fancy_upsample_8_pairs(&y_row, &u_row_1, &u_row_2, &v_row_1, &v_row_2, &mut rgb_simd);
        }

        // Compare with scalar implementation
        let mut rgb_scalar = [0u8; 48];
        for i in 0..8 {
            // For pixel pair i, we need u[i], u[i+1] from both rows
            let u_a = u_row_1[i];
            let u_b = u_row_1[i + 1];
            let u_c = u_row_2[i];
            let u_d = u_row_2[i + 1];
            let v_a = v_row_1[i];
            let v_b = v_row_1[i + 1];
            let v_c = v_row_2[i];
            let v_d = v_row_2[i + 1];

            // First pixel of pair uses diag1 = (9a + 3b + 3c + d) / 16
            let u_diag1 = get_fancy_chroma_value(u_a, u_b, u_c, u_d);
            let v_diag1 = get_fancy_chroma_value(v_a, v_b, v_c, v_d);

            // Second pixel of pair uses diag2 = (3a + 9b + c + 3d) / 16
            // which is get_fancy_chroma_value(b, a, d, c)
            let u_diag2 = get_fancy_chroma_value(u_b, u_a, u_d, u_c);
            let v_diag2 = get_fancy_chroma_value(v_b, v_a, v_d, v_c);

            let y1 = y_row[i * 2];
            let y2 = y_row[i * 2 + 1];

            let (r1, g1, b1) = scalar_yuv_to_rgb(y1, u_diag1, v_diag1);
            let (r2, g2, b2) = scalar_yuv_to_rgb(y2, u_diag2, v_diag2);

            rgb_scalar[i * 6] = r1;
            rgb_scalar[i * 6 + 1] = g1;
            rgb_scalar[i * 6 + 2] = b1;
            rgb_scalar[i * 6 + 3] = r2;
            rgb_scalar[i * 6 + 4] = g2;
            rgb_scalar[i * 6 + 5] = b2;
        }

        for i in 0..16 {
            let r_simd = rgb_simd[i * 3];
            let g_simd = rgb_simd[i * 3 + 1];
            let b_simd = rgb_simd[i * 3 + 2];
            let r_scalar = rgb_scalar[i * 3];
            let g_scalar = rgb_scalar[i * 3 + 1];
            let b_scalar = rgb_scalar[i * 3 + 2];

            assert_eq!(
                r_simd, r_scalar,
                "R mismatch at pixel {}: SIMD={}, scalar={}",
                i, r_simd, r_scalar
            );
            assert_eq!(
                g_simd, g_scalar,
                "G mismatch at pixel {}: SIMD={}, scalar={}",
                i, g_simd, g_scalar
            );
            assert_eq!(
                b_simd, b_scalar,
                "B mismatch at pixel {}: SIMD={}, scalar={}",
                i, b_simd, b_scalar
            );
        }
    }
}
