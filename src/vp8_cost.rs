//! Cost estimation for VP8 encoding
//!
//! Provides RD (rate-distortion) cost calculation for mode selection.
//! Based on libwebp's cost estimation approach.

/// Distortion multiplier - scales distortion to match bit cost units
pub const RD_DISTO_MULT: u32 = 256;

/// Fixed mode costs for Intra16 modes (DC, V, H, TM)
/// These are pre-calculated bit costs from the VP8 coding tree.
/// Values from libwebp's VP8FixedCostsI16.
/// Note: these include the fixed VP8BitCost(1, 145) mode selection cost.
pub const FIXED_COSTS_I16: [u16; 4] = [663, 919, 872, 919];

/// Fixed mode costs for chroma modes (DC, V, H, TM)
/// Values from libwebp's VP8FixedCostsUV.
pub const FIXED_COSTS_UV: [u16; 4] = [302, 984, 439, 642];

/// Fixed mode costs for Intra4 modes (B_DC, B_TM, B_VE, B_HE, B_LD, B_RD, B_VR, B_VL, B_HD, B_HU)
/// These are approximate average costs since actual costs are context-dependent.
/// Order matches IntraMode enum: DC, TM, VE, HE, LD, RD, VR, VL, HD, HU
/// Values derived from libwebp's probability trees (roughly: -log2(prob) * BIT_COST_SCALE)
pub const FIXED_COSTS_I4: [u16; 10] = [
    300,  // DC - most common, cheapest
    800,  // TM - less common
    600,  // VE - vertical edge
    650,  // HE - horizontal edge
    1000, // LD - left-down diagonal
    1000, // RD - right-down diagonal
    1000, // VR - vertical-right
    1000, // VL - vertical-left
    1000, // HD - horizontal-down
    1000, // HU - horizontal-up
];

/// Lambda values for mode selection distortion trade-off.
/// These empirical constants from libwebp's RefineUsingDistortion function
/// represent the bits-per-distortion trade-off for each block type.
///
/// Higher lambda = prefer lower distortion over smaller bit cost
pub const LAMBDA_I16: u32 = 106;
pub const LAMBDA_I4: u32 = 11;
pub const LAMBDA_UV: u32 = 120;

/// Penalty for choosing Intra4 over Intra16.
/// Intra4 requires encoding 16 subblock modes plus doesn't benefit from
/// Y2 WHT DC coefficient collection.
///
/// This value represents the approximate extra bit cost in RD score units.
/// The penalty accounts for:
/// - 16 subblock modes vs 1 macroblock mode (~4-5 bits each = ~64-80 extra bits)
/// - No Y2 WHT DC coefficient collection benefit
/// - Coefficient encoding overhead not captured by SSE-only comparison
///
/// This is a conservative penalty that prefers Intra16 unless Intra4 has
/// significantly better prediction quality.
#[allow(dead_code)] // Kept for future Intra4 vs Intra16 comparison
pub const INTRA4_PENALTY: u64 = 50_000_000;

/// Calculate RD score for mode selection.
///
/// Formula: score = SSE * RD_DISTO_MULT + mode_cost * lambda
///
/// Lower score = better trade-off between quality and bits.
#[inline]
pub fn rd_score(sse: u32, mode_cost: u16, lambda: u32) -> u64 {
    let distortion = u64::from(sse) * u64::from(RD_DISTO_MULT);
    let rate = u64::from(mode_cost) * u64::from(lambda);
    distortion + rate
}

/// Calculate RD score using i64 for signed values (useful for diffs)
#[inline]
#[allow(dead_code)] // Will be used when Intra4 is re-enabled
pub fn rd_score_i64(sse: u32, mode_cost: u16, lambda: u32) -> i64 {
    let distortion = i64::from(sse) * i64::from(RD_DISTO_MULT);
    let rate = i64::from(mode_cost) * i64::from(lambda);
    distortion + rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rd_score_basic() {
        // Zero SSE and zero cost = zero score
        assert_eq!(rd_score(0, 0, LAMBDA_I16), 0);

        // Only distortion
        assert_eq!(rd_score(100, 0, LAMBDA_I16), 100 * 256);

        // Only rate
        assert_eq!(rd_score(0, 663, LAMBDA_I16), 663 * 106);

        // Combined
        let sse = 1000u32;
        let mode_cost = FIXED_COSTS_I16[0]; // DC mode
        let expected = 1000 * 256 + u64::from(mode_cost) * 106;
        assert_eq!(rd_score(sse, mode_cost, LAMBDA_I16), expected);
    }

    #[test]
    fn test_mode_cost_ordering() {
        // DC should be cheapest for Intra16
        assert!(FIXED_COSTS_I16[0] < FIXED_COSTS_I16[1]); // DC < V
        assert!(FIXED_COSTS_I16[0] < FIXED_COSTS_I16[2]); // DC < H
        assert!(FIXED_COSTS_I16[0] < FIXED_COSTS_I16[3]); // DC < TM

        // DC should be cheapest for UV too
        assert!(FIXED_COSTS_UV[0] < FIXED_COSTS_UV[1]); // DC < V
        assert!(FIXED_COSTS_UV[0] < FIXED_COSTS_UV[2]); // DC < H
        assert!(FIXED_COSTS_UV[0] < FIXED_COSTS_UV[3]); // DC < TM
    }

    #[test]
    fn test_rd_score_prefers_lower_sse() {
        let mode_cost = FIXED_COSTS_I16[0];
        let lambda = LAMBDA_I16;

        let score_low_sse = rd_score(100, mode_cost, lambda);
        let score_high_sse = rd_score(1000, mode_cost, lambda);

        assert!(score_low_sse < score_high_sse);
    }

    #[test]
    fn test_rd_score_prefers_lower_mode_cost() {
        let sse = 500u32;
        let lambda = LAMBDA_I16;

        let score_cheap_mode = rd_score(sse, FIXED_COSTS_I16[0], lambda); // DC
        let score_expensive_mode = rd_score(sse, FIXED_COSTS_I16[1], lambda); // V

        assert!(score_cheap_mode < score_expensive_mode);
    }

    #[test]
    fn test_rd_trade_off_sse_vs_mode_cost() {
        // Test that a mode with higher SSE but lower cost can win when appropriate
        let lambda = LAMBDA_I16;

        // DC mode has lower cost (663) but let's say higher SSE
        let dc_sse = 1000u32;
        let dc_cost = FIXED_COSTS_I16[0]; // 663

        // V mode has higher cost (919) but lower SSE
        let v_sse = 500u32;
        let v_cost = FIXED_COSTS_I16[1]; // 919

        let dc_score = rd_score(dc_sse, dc_cost, lambda);
        let v_score = rd_score(v_sse, v_cost, lambda);

        // In this case, V should win because:
        // DC: 1000 * 256 + 663 * 106 = 256,000 + 70,278 = 326,278
        // V:  500 * 256 + 919 * 106 = 128,000 + 97,414 = 225,414
        assert!(
            v_score < dc_score,
            "V should win: V={} < DC={}",
            v_score,
            dc_score
        );
    }

    #[test]
    fn test_rd_trade_off_prefers_dc_for_small_sse_diff() {
        // When SSE difference is small, cheaper mode should win
        let lambda = LAMBDA_I16;

        // DC mode with slightly higher SSE
        let dc_sse = 550u32;
        let dc_cost = FIXED_COSTS_I16[0]; // 663

        // V mode with slightly lower SSE
        let v_sse = 500u32;
        let v_cost = FIXED_COSTS_I16[1]; // 919

        let dc_score = rd_score(dc_sse, dc_cost, lambda);
        let v_score = rd_score(v_sse, v_cost, lambda);

        // DC should win because:
        // DC: 550 * 256 + 663 * 106 = 140,800 + 70,278 = 211,078
        // V:  500 * 256 + 919 * 106 = 128,000 + 97,414 = 225,414
        assert!(
            dc_score < v_score,
            "DC should win: DC={} < V={}",
            dc_score,
            v_score
        );
    }

    #[test]
    fn test_intra4_costs_reasonable() {
        // Verify I4 mode costs are in reasonable range
        for &cost in &FIXED_COSTS_I4 {
            assert!(cost >= 200, "Mode cost {} too low", cost);
            assert!(cost <= 1500, "Mode cost {} too high", cost);
        }

        // DC should be cheapest
        assert!(FIXED_COSTS_I4[0] <= FIXED_COSTS_I4[1]); // DC <= TM
        assert!(FIXED_COSTS_I4[0] <= FIXED_COSTS_I4[4]); // DC <= LD
    }

    #[test]
    fn test_lambda_values_from_libwebp() {
        // Verify our lambda values match libwebp's RefineUsingDistortion
        assert_eq!(LAMBDA_I16, 106, "I16 lambda should be 106");
        assert_eq!(LAMBDA_I4, 11, "I4 lambda should be 11");
        assert_eq!(LAMBDA_UV, 120, "UV lambda should be 120");
    }

    #[test]
    fn test_uv_rd_selection() {
        // Test UV mode selection with RD cost
        let lambda = LAMBDA_UV;

        // For chroma, DC is typically best for smooth areas
        let sse = 200u32;
        let dc_score = rd_score(sse, FIXED_COSTS_UV[0], lambda);
        let v_score = rd_score(sse, FIXED_COSTS_UV[1], lambda);
        let h_score = rd_score(sse, FIXED_COSTS_UV[2], lambda);
        let tm_score = rd_score(sse, FIXED_COSTS_UV[3], lambda);

        // DC should be cheapest when SSE is same
        assert!(dc_score < v_score);
        assert!(dc_score < h_score);
        assert!(dc_score < tm_score);
    }
}
