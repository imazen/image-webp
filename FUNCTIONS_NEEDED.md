# Detailed Function List for VP8 Lossy Encoder

## Component 1: SSE/Distortion Functions

**Source:** `libwebp/src/dsp/enc.c:557-669`

### Core Functions (MUST HAVE)

```rust
/// Sum of squared errors for 16x16 block
/// Used by: PickBestIntra16, StoreSSE
fn sse_16x16(a: &[u8], b: &[u8], stride: usize) -> u32 {
    let mut count = 0u32;
    for y in 0..16 {
        for x in 0..16 {
            let diff = a[y * stride + x] as i32 - b[y * stride + x] as i32;
            count += (diff * diff) as u32;
        }
    }
    count
}

/// SSE for 16x8 (UV plane)
/// Used by: PickBestUV
fn sse_16x8(a: &[u8], b: &[u8], stride: usize) -> u32;

/// SSE for 8x8
/// Used by: StoreSSE (chroma)
fn sse_8x8(a: &[u8], b: &[u8], stride: usize) -> u32;

/// SSE for 4x4
/// Used by: PickBestIntra4
fn sse_4x4(a: &[u8], b: &[u8], stride: usize) -> u32;
```

### Spectral Distortion (SHOULD HAVE)

```rust
/// Hadamard transform for distortion weighting
/// Returns weighted sum of absolute transformed coefficients
fn t_transform(input: &[u8], stride: usize, weights: &[u16; 16]) -> i32 {
    let mut tmp = [0i32; 16];
    // horizontal pass
    for i in 0..4 {
        let a0 = input[i*stride + 0] as i32 + input[i*stride + 2] as i32;
        let a1 = input[i*stride + 1] as i32 + input[i*stride + 3] as i32;
        let a2 = input[i*stride + 1] as i32 - input[i*stride + 3] as i32;
        let a3 = input[i*stride + 0] as i32 - input[i*stride + 2] as i32;
        tmp[i*4 + 0] = a0 + a1;
        tmp[i*4 + 1] = a3 + a2;
        tmp[i*4 + 2] = a3 - a2;
        tmp[i*4 + 3] = a0 - a1;
    }
    // vertical pass + weighting
    let mut sum = 0i32;
    for i in 0..4 {
        let a0 = tmp[0 + i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[0 + i] - tmp[8 + i];
        sum += weights[0 + i] as i32 * (a0 + a1).abs();
        sum += weights[4 + i] as i32 * (a3 + a2).abs();
        sum += weights[8 + i] as i32 * (a3 - a2).abs();
        sum += weights[12 + i] as i32 * (a0 - a1).abs();
    }
    sum
}

/// 4x4 spectral distortion
/// Used by: PickBestIntra4, PickBestIntra16 (when tlambda > 0)
fn tdisto_4x4(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    let sum1 = t_transform(a, stride, w);
    let sum2 = t_transform(b, stride, w);
    (sum2 - sum1).abs() >> 5
}

/// 16x16 spectral distortion (calls tdisto_4x4 16 times)
fn tdisto_16x16(a: &[u8], b: &[u8], stride: usize, w: &[u16; 16]) -> i32;
```

### Constants

```rust
/// Distortion weights for luma (kWeightY in C)
const WEIGHT_Y: [u16; 16] = [
    38, 32, 20, 9,
    32, 28, 17, 7,
    20, 17, 10, 4,
    9,  7,  4,  2,
];

/// Distortion weights for chroma
const WEIGHT_UV: [u16; 16] = [/* similar pattern */];
```

---

## Component 2: Mode Score Structure

**Source:** `libwebp/src/enc/vp8i_enc.h:139-170`

```rust
/// Rate-distortion score accumulator
#[derive(Clone, Default)]
pub struct VP8ModeScore {
    pub D: i64,                      // distortion (SSE)
    pub SD: i64,                     // spectral distortion
    pub H: i64,                      // header cost (mode bits)
    pub R: i64,                      // rate (coefficient bits)
    pub score: i64,                  // D + lambda * (R + H)

    pub y_dc_levels: [i16; 16],      // I16 DC coefficients (after WHT)
    pub y_ac_levels: [[i16; 16]; 16], // Y AC coefficients per 4x4 block
    pub uv_levels: [[i16; 16]; 8],   // UV coefficients (4 U + 4 V blocks)

    pub mode_i16: i32,               // selected I16 mode [0..3]
    pub modes_i4: [u8; 16],          // selected I4 modes per sub-block
    pub mode_uv: i32,                // selected UV mode [0..3]
    pub nz: u32,                     // non-zero coefficient flags
    pub derr: [[i8; 3]; 2],          // DC diffusion errors (optional)
}

// Helper functions
fn init_score(rd: &mut VP8ModeScore);
fn copy_score(dst: &mut VP8ModeScore, src: &VP8ModeScore);
fn add_score(dst: &mut VP8ModeScore, src: &VP8ModeScore);
fn set_rd_score(lambda: i32, rd: &mut VP8ModeScore) {
    rd.score = rd.D + rd.SD + (rd.R + rd.H) * lambda as i64;
}
```

---

## Component 3: Quantization Matrix

**Source:** `libwebp/src/enc/vp8i_enc.h:184-198`

```rust
/// Quantization parameters for one coefficient type
#[derive(Clone)]
pub struct VP8Matrix {
    pub q: [u16; 16],        // quantizer values
    pub iq: [u16; 16],       // reciprocals for division (fixed point)
    pub bias: [u32; 16],     // rounding bias
    pub zthresh: [u32; 16],  // zero threshold
    pub sharpen: [i16; 16],  // sharpening (negative = soften)
}

// QFIX = 17 for fixed-point reciprocal
const QFIX: u32 = 17;

/// Quantize single coefficient
fn quantdiv(coeff: u32, iq: u16, bias: u32) -> i32 {
    ((coeff as u64 * iq as u64 + bias as u64) >> QFIX) as i32
}
```

---

## Component 4: Segment Info (Lambda values)

**Source:** `libwebp/src/enc/vp8i_enc.h:200-230`

```rust
/// Per-segment quantization and RD parameters
pub struct VP8SegmentInfo {
    pub y1: VP8Matrix,      // luma AC
    pub y2: VP8Matrix,      // luma DC (I16 mode)
    pub uv: VP8Matrix,      // chroma

    pub alpha: i32,         // segment complexity [-127..127]
    pub beta: i32,          // filter susceptibility
    pub quant: i32,         // base quantizer [0..127]
    pub fstrength: i32,     // loop filter strength

    // Lambda values for RD
    pub lambda_i16: i32,    // lambda for I16 mode
    pub lambda_i4: i32,     // lambda for I4 mode
    pub lambda_uv: i32,     // lambda for UV mode
    pub lambda_mode: i32,   // lambda for mode decision
    pub lambda_trellis_i16: i32,
    pub lambda_trellis_i4: i32,
    pub lambda_trellis_uv: i32,
    pub tlambda: i32,       // spectral distortion lambda

    pub min_disto: i32,     // minimum distortion threshold
    pub max_edge: i32,      // max edge delta (for filter)
    pub i4_penalty: i64,    // penalty for choosing I4 over I16
}
```

---

## Component 5: Mode Selection Functions

**Source:** `libwebp/src/enc/quant_enc.c:982-1189`

### PickBestIntra16

```rust
/// Select best 16x16 luma prediction mode
/// Tries all 4 modes (DC, TM, V, H), picks lowest RD score
fn pick_best_intra16(it: &mut VP8EncIterator, rd: &mut VP8ModeScore) {
    let dqm = &it.enc.dqm[it.mb.segment as usize];
    let lambda = dqm.lambda_i16;
    let tlambda = dqm.tlambda;
    let src = &it.yuv_in[Y_OFF..];

    rd.mode_i16 = -1;
    let mut best_score = i64::MAX;

    for mode in 0..4 {  // DC, TM, V, H
        let tmp_dst = &mut it.yuv_out2[Y_OFF..];

        // 1. Generate prediction + transform + quantize + reconstruct
        let nz = reconstruct_intra16(it, rd_cur, tmp_dst, mode);

        // 2. Compute distortion
        rd_cur.D = sse_16x16(src, tmp_dst, BPS) as i64;
        rd_cur.SD = if tlambda != 0 {
            mult_8b(tlambda, tdisto_16x16(src, tmp_dst, BPS, &WEIGHT_Y)) as i64
        } else { 0 };

        // 3. Compute rate
        rd_cur.H = VP8_FIXED_COSTS_I16[mode] as i64;
        rd_cur.R = get_cost_luma16(it, rd_cur);

        // 4. RD score
        set_rd_score(lambda, &mut rd_cur);

        if mode == 0 || rd_cur.score < best_score {
            best_score = rd_cur.score;
            std::mem::swap(rd, rd_cur);
            swap_out(it);
        }
    }
}
```

### PickBestIntra4

```rust
/// Select best 4x4 luma prediction modes
/// Tries all 10 modes for each of 16 sub-blocks
/// Returns true if I4 is better than current I16
fn pick_best_intra4(it: &mut VP8EncIterator, rd: &mut VP8ModeScore) -> bool {
    let dqm = &it.enc.dqm[it.mb.segment as usize];
    let lambda = dqm.lambda_i4;

    let mut rd_best = VP8ModeScore::default();
    init_score(&mut rd_best);
    rd_best.H = 211;  // VP8BitCost(0, 145)
    set_rd_score(dqm.lambda_mode, &mut rd_best);

    // Iterate over 16 sub-blocks
    vp8_iterator_start_i4(it);
    loop {
        let src = &it.yuv_in[Y_OFF + VP8_SCAN[it.i4]..];
        let mode_costs = get_cost_mode_i4(it, &rd.modes_i4);

        let mut best_mode = -1i32;
        let mut rd_i4 = VP8ModeScore::default();

        make_intra4_preds(it);  // generate all 10 predictions

        for mode in 0..10 {  // B_DC..B_HU
            let tmp_dst = &mut it.yuv_p[I4TMP..];

            // Reconstruct
            let nz = reconstruct_intra4(it, &mut tmp_levels, src, tmp_dst, mode);

            // Distortion
            rd_tmp.D = sse_4x4(src, tmp_dst, BPS) as i64;
            rd_tmp.SD = if tlambda != 0 {
                mult_8b(tlambda, tdisto_4x4(src, tmp_dst, BPS, &WEIGHT_Y)) as i64
            } else { 0 };

            // Rate
            rd_tmp.H = mode_costs[mode] as i64;
            rd_tmp.R = get_cost_luma4(it, &tmp_levels);

            set_rd_score(lambda, &mut rd_tmp);

            if best_mode < 0 || rd_tmp.score < rd_i4.score {
                copy_score(&mut rd_i4, &rd_tmp);
                best_mode = mode as i32;
                // save levels, swap buffers...
            }
        }

        add_score(&mut rd_best, &rd_i4);

        // Early exit if I4 worse than I16
        if rd_best.score >= rd.score {
            return false;
        }

        rd.modes_i4[it.i4] = best_mode as u8;

        if !vp8_iterator_rotate_i4(it) { break; }
    }

    copy_score(rd, &rd_best);
    true  // I4 is better
}
```

### PickBestUV

```rust
/// Select best 8x8 chroma prediction mode
fn pick_best_uv(it: &mut VP8EncIterator, rd: &mut VP8ModeScore) {
    let dqm = &it.enc.dqm[it.mb.segment as usize];
    let lambda = dqm.lambda_uv;
    let src = &it.yuv_in[U_OFF..];

    rd.mode_uv = -1;
    let mut rd_best = VP8ModeScore::default();

    for mode in 0..4 {  // DC, TM, V, H
        let tmp_dst = &mut it.yuv_out2[U_OFF..];

        // Reconstruct (both U and V planes)
        let nz = reconstruct_uv(it, &mut rd_uv, tmp_dst, mode);

        // Distortion (16x8 covers both U and V)
        rd_uv.D = sse_16x8(src, tmp_dst, BPS) as i64;
        rd_uv.SD = 0;  // No spectral disto for UV

        // Rate
        rd_uv.H = VP8_FIXED_COSTS_UV[mode] as i64;
        rd_uv.R = get_cost_uv(it, &rd_uv);

        set_rd_score(lambda, &mut rd_uv);

        if mode == 0 || rd_uv.score < rd_best.score {
            copy_score(&mut rd_best, &rd_uv);
            rd.mode_uv = mode as i32;
            // copy levels, swap buffers...
        }
    }

    add_score(rd, &rd_best);
}
```

---

## Component 6: Reconstruction Functions

**Source:** `libwebp/src/enc/quant_enc.c:700-950`

```rust
/// Reconstruct 16x16 luma with given mode
/// Returns non-zero coefficient flags
fn reconstruct_intra16(
    it: &mut VP8EncIterator,
    rd: &mut VP8ModeScore,
    yuv_out: &mut [u8],
    mode: i32
) -> u32 {
    let dqm = &it.enc.dqm[it.mb.segment as usize];
    let src = &it.yuv_in[Y_OFF..];
    let ref_block = &it.yuv_p[VP8_I16_MODE_OFFSETS[mode as usize]..];

    let mut nz = 0u32;
    let mut dc = [0i16; 16];
    let mut tmp = [[0i16; 16]; 16];

    // Forward transform all 16 4x4 blocks
    for n in 0..16 {
        vp8_f_transform(&src[VP8_SCAN[n]..], &ref_block[VP8_SCAN[n]..], &mut tmp[n]);
        dc[n] = tmp[n][0];  // extract DC
        tmp[n][0] = 0;       // zero it for AC quantization
    }

    // WHT on DC coefficients
    vp8_f_transform_wht(&dc);

    // Quantize DC
    nz |= (quantize_block(&mut dc, &mut rd.y_dc_levels, &dqm.y2) as u32) << 24;

    // Quantize AC
    for n in 0..16 {
        nz |= (quantize_block(&mut tmp[n], &mut rd.y_ac_levels[n], &dqm.y1) as u32) << n;
    }

    // Inverse WHT
    vp8_i_transform_wht(&rd.y_dc_levels, &mut dc);

    // Put DC back and inverse transform
    for n in 0..16 {
        tmp[n][0] = dc[n];
        vp8_i_transform(&ref_block[VP8_SCAN[n]..], &tmp[n], &mut yuv_out[VP8_SCAN[n]..], false);
    }

    nz
}

/// Reconstruct single 4x4 block
fn reconstruct_intra4(...) -> u32;

/// Reconstruct 8x8 UV (8 blocks total)
fn reconstruct_uv(...) -> u32;
```

---

## Component 7: Cost Estimation

**Source:** `libwebp/src/enc/cost_enc.c`

### Fixed Mode Costs (Constants)

```rust
/// Fixed bit costs for I16 modes (includes VP8BitCost(1, 145))
const VP8_FIXED_COSTS_I16: [u16; 4] = [663, 919, 872, 919];

/// Fixed bit costs for UV modes
const VP8_FIXED_COSTS_UV: [u16; 4] = [302, 984, 439, 642];

/// Fixed bit costs for I4 modes (indexed by [top][left][mode])
const VP8_FIXED_COSTS_I4: [[[u16; 10]; 10]; 10] = [/* large table */];
```

### Rate Estimation Functions

```rust
/// Estimate bits for I16 coefficients
fn get_cost_luma16(it: &VP8EncIterator, rd: &VP8ModeScore) -> i64 {
    let proba = &it.enc.proba;
    let mut cost = 0i64;

    // DC coefficients (WHT)
    cost += get_cost_block(&rd.y_dc_levels, TYPE_I16_DC, proba);

    // AC coefficients (skip DC position)
    for n in 0..16 {
        let ctx = it.top_nz[n & 3] + it.left_nz[n >> 2];
        cost += get_cost_block_ac(&rd.y_ac_levels[n], TYPE_I16_AC, ctx, proba);
    }

    cost
}

/// Estimate bits for single I4 block
fn get_cost_luma4(it: &VP8EncIterator, levels: &[i16; 16]) -> i64;

/// Estimate bits for UV coefficients
fn get_cost_uv(it: &VP8EncIterator, rd: &VP8ModeScore) -> i64;

/// Cost of single block (internal)
fn get_cost_block(levels: &[i16; 16], typ: i32, proba: &VP8EncProba) -> i64;
```

---

## Component 8: Quantization Functions

**Source:** `libwebp/src/dsp/enc.c:676-713`

```rust
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const MAX_LEVEL: i32 = 2047;

/// Quantize 4x4 block in-place, output levels in zigzag order
/// Returns true if any non-zero coefficients
fn quantize_block(
    coeffs: &mut [i16; 16],    // in/out: dequantized coeffs
    levels: &mut [i16; 16],     // out: quantized levels (zigzag)
    mtx: &VP8Matrix
) -> bool {
    let mut last = -1i32;

    for n in 0..16 {
        let j = ZIGZAG[n];
        let sign = coeffs[j] < 0;
        let mut coeff = if sign { -coeffs[j] } else { coeffs[j] } as u32;
        coeff = coeff.saturating_add(mtx.sharpen[j] as u32);

        if coeff > mtx.zthresh[j] {
            let mut level = quantdiv(coeff, mtx.iq[j], mtx.bias[j]);
            level = level.min(MAX_LEVEL);
            if sign { level = -level; }

            coeffs[j] = (level * mtx.q[j] as i32) as i16;  // dequantize
            levels[n] = level as i16;
            if level != 0 { last = n as i32; }
        } else {
            levels[n] = 0;
            coeffs[j] = 0;
        }
    }

    last >= 0
}

/// Quantize two adjacent 4x4 blocks
fn quantize_2blocks(
    coeffs: &mut [i16; 32],
    levels: &mut [i16; 32],
    mtx: &VP8Matrix
) -> u32 {
    let nz0 = quantize_block(&mut coeffs[0..16], &mut levels[0..16], mtx) as u32;
    let nz1 = quantize_block(&mut coeffs[16..32], &mut levels[16..32], mtx) as u32;
    nz0 | (nz1 << 1)
}
```

---

## Summary: Minimum Viable Implementation

### Must Have (for any quality improvement):
1. `sse_4x4`, `sse_8x8`, `sse_16x8`, `sse_16x16` - **4 functions, ~40 lines**
2. `VP8ModeScore` struct - **1 struct, ~20 lines**
3. `set_rd_score`, `init_score`, `copy_score` - **3 helpers, ~15 lines**
4. `pick_best_intra16` - **1 function, ~60 lines**
5. `pick_best_uv` - **1 function, ~50 lines**

**Total MVP: ~185 lines**

### Should Have (for good quality):
6. `tdisto_4x4`, `tdisto_16x16` - **2 functions, ~60 lines**
7. `pick_best_intra4` - **1 function, ~100 lines**
8. `VP8Matrix`, `quantize_block` - **improved quant, ~80 lines**

**Total Good: ~425 lines**

### Nice to Have (for great quality):
9. Cost estimation tables and functions - **~200 lines**
10. Trellis quantization - **~300 lines**

**Total Great: ~925 lines**
