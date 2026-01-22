//! SIMD-optimized YUV to RGB conversion using x86 SSE2/SSSE3 intrinsics.
//!
//! Processes 8 pixels at a time for significant speedup over scalar code.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// YUV to RGB conversion constants (fixed-point, 8-bit fractional)
/// Based on libwebp's YUV conversion: R = Y + 1.402*(V-128), etc.
const YUV_FIX: i32 = 16; // Fixed-point precision
const YUV_HALF: i32 = 1 << (YUV_FIX - 1);

// Coefficients scaled by 2^16
const C_Y: i32 = 76309; // 1.164 * 65536
const C_VR: i32 = 104597; // 1.596 * 65536
const C_UG: i32 = 25675; // 0.392 * 65536
const C_VG: i32 = 53279; // 0.813 * 65536
const C_UB: i32 = 132201; // 2.017 * 65536

/// Process 8 pixels of YUV to RGB conversion using SSE2.
///
/// # Safety
/// Requires SSE2. Input slices must have at least 8 elements.
/// Output slice must have at least 24 bytes (8 RGB pixels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn yuv_to_rgb_8x(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    rgb: &mut [u8],
) {
    debug_assert!(y.len() >= 8);
    debug_assert!(u.len() >= 8);
    debug_assert!(v.len() >= 8);
    debug_assert!(rgb.len() >= 24);

    // Load 8 Y values and zero-extend to 16-bit
    let y8 = _mm_loadl_epi64(y.as_ptr() as *const __m128i);
    let y16 = _mm_unpacklo_epi8(y8, _mm_setzero_si128());

    // Load 8 U values and zero-extend
    let u8_vec = _mm_loadl_epi64(u.as_ptr() as *const __m128i);
    let u16 = _mm_unpacklo_epi8(u8_vec, _mm_setzero_si128());

    // Load 8 V values and zero-extend
    let v8 = _mm_loadl_epi64(v.as_ptr() as *const __m128i);
    let v16 = _mm_unpacklo_epi8(v8, _mm_setzero_si128());

    // Subtract 128 from U and V (convert to signed)
    let offset_128 = _mm_set1_epi16(128);
    let u_centered = _mm_sub_epi16(u16, offset_128);
    let v_centered = _mm_sub_epi16(v16, offset_128);

    // Subtract 16 from Y (studio swing)
    let offset_16 = _mm_set1_epi16(16);
    let y_adj = _mm_sub_epi16(y16, offset_16);

    // Convert to 32-bit for precision
    // Process low 4 pixels
    let y_lo = _mm_unpacklo_epi16(y_adj, _mm_setzero_si128());
    let u_lo = _mm_unpacklo_epi16(u_centered, _mm_cmplt_epi16(u_centered, _mm_setzero_si128()));
    let v_lo = _mm_unpacklo_epi16(v_centered, _mm_cmplt_epi16(v_centered, _mm_setzero_si128()));

    // High 4 pixels
    let y_hi = _mm_unpackhi_epi16(y_adj, _mm_setzero_si128());
    let u_hi = _mm_unpackhi_epi16(u_centered, _mm_cmplt_epi16(u_centered, _mm_setzero_si128()));
    let v_hi = _mm_unpackhi_epi16(v_centered, _mm_cmplt_epi16(v_centered, _mm_setzero_si128()));

    // Compute RGB using fixed-point arithmetic
    // R = clip((C_Y * Y + C_VR * V + YUV_HALF) >> YUV_FIX)
    // G = clip((C_Y * Y - C_UG * U - C_VG * V + YUV_HALF) >> YUV_FIX)
    // B = clip((C_Y * Y + C_UB * U + YUV_HALF) >> YUV_FIX)

    let c_y = _mm_set1_epi32(C_Y);
    let c_vr = _mm_set1_epi32(C_VR);
    let c_ug = _mm_set1_epi32(C_UG);
    let c_vg = _mm_set1_epi32(C_VG);
    let c_ub = _mm_set1_epi32(C_UB);
    let half = _mm_set1_epi32(YUV_HALF);

    // Low 4 pixels
    let y_scaled_lo = _mm_mullo_epi32(y_lo, c_y);

    // R = Y * C_Y + V * C_VR
    let r_lo = _mm_add_epi32(y_scaled_lo, _mm_mullo_epi32(v_lo, c_vr));
    let r_lo = _mm_add_epi32(r_lo, half);
    let r_lo = _mm_srai_epi32(r_lo, YUV_FIX);

    // G = Y * C_Y - U * C_UG - V * C_VG
    let g_lo = _mm_sub_epi32(y_scaled_lo, _mm_mullo_epi32(u_lo, c_ug));
    let g_lo = _mm_sub_epi32(g_lo, _mm_mullo_epi32(v_lo, c_vg));
    let g_lo = _mm_add_epi32(g_lo, half);
    let g_lo = _mm_srai_epi32(g_lo, YUV_FIX);

    // B = Y * C_Y + U * C_UB
    let b_lo = _mm_add_epi32(y_scaled_lo, _mm_mullo_epi32(u_lo, c_ub));
    let b_lo = _mm_add_epi32(b_lo, half);
    let b_lo = _mm_srai_epi32(b_lo, YUV_FIX);

    // High 4 pixels
    let y_scaled_hi = _mm_mullo_epi32(y_hi, c_y);

    let r_hi = _mm_add_epi32(y_scaled_hi, _mm_mullo_epi32(v_hi, c_vr));
    let r_hi = _mm_add_epi32(r_hi, half);
    let r_hi = _mm_srai_epi32(r_hi, YUV_FIX);

    let g_hi = _mm_sub_epi32(y_scaled_hi, _mm_mullo_epi32(u_hi, c_ug));
    let g_hi = _mm_sub_epi32(g_hi, _mm_mullo_epi32(v_hi, c_vg));
    let g_hi = _mm_add_epi32(g_hi, half);
    let g_hi = _mm_srai_epi32(g_hi, YUV_FIX);

    let b_hi = _mm_add_epi32(y_scaled_hi, _mm_mullo_epi32(u_hi, c_ub));
    let b_hi = _mm_add_epi32(b_hi, half);
    let b_hi = _mm_srai_epi32(b_hi, YUV_FIX);

    // Pack back to 16-bit with signed saturation
    let r16 = _mm_packs_epi32(r_lo, r_hi);
    let g16 = _mm_packs_epi32(g_lo, g_hi);
    let b16 = _mm_packs_epi32(b_lo, b_hi);

    // Pack to 8-bit with unsigned saturation (clamps to 0-255)
    let r8 = _mm_packus_epi16(r16, r16);
    let g8 = _mm_packus_epi16(g16, g16);
    let b8 = _mm_packus_epi16(b16, b16);

    // Interleave R, G, B to RGB format
    // We have: R0 R1 R2 R3 R4 R5 R6 R7 ...
    //          G0 G1 G2 G3 G4 G5 G6 G7 ...
    //          B0 B1 B2 B3 B4 B5 B6 B7 ...
    // Want:    R0 G0 B0 R1 G1 B1 R2 G2 B2 ...

    // Interleave R and G: RG0 RG1 RG2 RG3 RG4 RG5 RG6 RG7
    let rg_lo = _mm_unpacklo_epi8(r8, g8);

    // For RGB interleaving, we need to manually construct the output
    // since SSE2 doesn't have a direct 3-channel interleave

    // Extract individual bytes and write
    let r_bytes: [u8; 16] = core::mem::transmute(r8);
    let g_bytes: [u8; 16] = core::mem::transmute(g8);
    let b_bytes: [u8; 16] = core::mem::transmute(b8);

    for i in 0..8 {
        rgb[i * 3] = r_bytes[i];
        rgb[i * 3 + 1] = g_bytes[i];
        rgb[i * 3 + 2] = b_bytes[i];
    }
}

/// Process 8 pixels of YUV to RGBA conversion using SSE2.
/// Alpha is set to 255.
///
/// # Safety
/// Requires SSE2. Input slices must have at least 8 elements.
/// Output slice must have at least 32 bytes (8 RGBA pixels).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn yuv_to_rgba_8x(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    rgba: &mut [u8],
) {
    debug_assert!(y.len() >= 8);
    debug_assert!(u.len() >= 8);
    debug_assert!(v.len() >= 8);
    debug_assert!(rgba.len() >= 32);

    // Same computation as yuv_to_rgb_8x
    let y8 = _mm_loadl_epi64(y.as_ptr() as *const __m128i);
    let y16 = _mm_unpacklo_epi8(y8, _mm_setzero_si128());
    let u8_vec = _mm_loadl_epi64(u.as_ptr() as *const __m128i);
    let u16 = _mm_unpacklo_epi8(u8_vec, _mm_setzero_si128());
    let v8 = _mm_loadl_epi64(v.as_ptr() as *const __m128i);
    let v16 = _mm_unpacklo_epi8(v8, _mm_setzero_si128());

    let offset_128 = _mm_set1_epi16(128);
    let u_centered = _mm_sub_epi16(u16, offset_128);
    let v_centered = _mm_sub_epi16(v16, offset_128);
    let offset_16 = _mm_set1_epi16(16);
    let y_adj = _mm_sub_epi16(y16, offset_16);

    let y_lo = _mm_unpacklo_epi16(y_adj, _mm_setzero_si128());
    let u_lo = _mm_unpacklo_epi16(u_centered, _mm_cmplt_epi16(u_centered, _mm_setzero_si128()));
    let v_lo = _mm_unpacklo_epi16(v_centered, _mm_cmplt_epi16(v_centered, _mm_setzero_si128()));
    let y_hi = _mm_unpackhi_epi16(y_adj, _mm_setzero_si128());
    let u_hi = _mm_unpackhi_epi16(u_centered, _mm_cmplt_epi16(u_centered, _mm_setzero_si128()));
    let v_hi = _mm_unpackhi_epi16(v_centered, _mm_cmplt_epi16(v_centered, _mm_setzero_si128()));

    let c_y = _mm_set1_epi32(C_Y);
    let c_vr = _mm_set1_epi32(C_VR);
    let c_ug = _mm_set1_epi32(C_UG);
    let c_vg = _mm_set1_epi32(C_VG);
    let c_ub = _mm_set1_epi32(C_UB);
    let half = _mm_set1_epi32(YUV_HALF);

    let y_scaled_lo = _mm_mullo_epi32(y_lo, c_y);
    let r_lo = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(y_scaled_lo, _mm_mullo_epi32(v_lo, c_vr)), half), YUV_FIX);
    let g_lo = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(_mm_sub_epi32(y_scaled_lo, _mm_mullo_epi32(u_lo, c_ug)), _mm_mullo_epi32(v_lo, c_vg)), half), YUV_FIX);
    let b_lo = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(y_scaled_lo, _mm_mullo_epi32(u_lo, c_ub)), half), YUV_FIX);

    let y_scaled_hi = _mm_mullo_epi32(y_hi, c_y);
    let r_hi = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(y_scaled_hi, _mm_mullo_epi32(v_hi, c_vr)), half), YUV_FIX);
    let g_hi = _mm_srai_epi32(_mm_add_epi32(_mm_sub_epi32(_mm_sub_epi32(y_scaled_hi, _mm_mullo_epi32(u_hi, c_ug)), _mm_mullo_epi32(v_hi, c_vg)), half), YUV_FIX);
    let b_hi = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(y_scaled_hi, _mm_mullo_epi32(u_hi, c_ub)), half), YUV_FIX);

    let r16 = _mm_packs_epi32(r_lo, r_hi);
    let g16 = _mm_packs_epi32(g_lo, g_hi);
    let b16 = _mm_packs_epi32(b_lo, b_hi);

    let r8 = _mm_packus_epi16(r16, r16);
    let g8 = _mm_packus_epi16(g16, g16);
    let b8 = _mm_packus_epi16(b16, b16);

    // Interleave to RGBA format
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuv_to_rgb_simd() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Test with known values
        // Pure white in YUV (Y=235, U=128, V=128) should give RGB(255,255,255)
        let y = [235u8; 8];
        let u = [128u8; 8];
        let v = [128u8; 8];
        let mut rgb = [0u8; 24];

        unsafe {
            yuv_to_rgb_8x(&y, &u, &v, &mut rgb);
        }

        // Check that all pixels are close to white
        for i in 0..8 {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
            // Allow some tolerance due to fixed-point arithmetic
            assert!(r > 250, "R should be close to 255, got {}", r);
            assert!(g > 250, "G should be close to 255, got {}", g);
            assert!(b > 250, "B should be close to 255, got {}", b);
        }
    }

    #[test]
    fn test_yuv_to_rgb_black() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Black in YUV (Y=16, U=128, V=128) should give RGB(0,0,0)
        let y = [16u8; 8];
        let u = [128u8; 8];
        let v = [128u8; 8];
        let mut rgb = [0u8; 24];

        unsafe {
            yuv_to_rgb_8x(&y, &u, &v, &mut rgb);
        }

        for i in 0..8 {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
            assert!(r < 5, "R should be close to 0, got {}", r);
            assert!(g < 5, "G should be close to 0, got {}", g);
            assert!(b < 5, "B should be close to 0, got {}", b);
        }
    }

    #[test]
    fn test_yuv_to_rgb_red() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // Pure red in YUV (Y=81, U=90, V=240) should give approximately RGB(255,0,0)
        let y = [81u8; 8];
        let u = [90u8; 8];
        let v = [240u8; 8];
        let mut rgb = [0u8; 24];

        unsafe {
            yuv_to_rgb_8x(&y, &u, &v, &mut rgb);
        }

        for i in 0..8 {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
            assert!(r > 200, "R should be high, got {}", r);
            assert!(g < 50, "G should be low, got {}", g);
            assert!(b < 50, "B should be low, got {}", b);
        }
    }
}
