# VP8 Lossy Encoder Component Inventory

## Summary

| Category | Total Functions | In libwebp-rs | In image-webp | Need to Port |
|----------|-----------------|---------------|---------------|--------------|
| **Transforms** | 4 | 2 (inverse only) | 4 (all!) | 0 |
| **Predictions** | 3 dispatchers | 3 (SSE2/AVX2) | 3 (all modes) | 0 |
| **SSE/Distortion** | 6 | 0 | 0 | **6** |
| **Quantization** | 3 | 0 | 1 (basic) | **2** |
| **Histogram** | 1 | 0 | 0 | **1** |
| **Bit Writer** | 1 | 0 | 1 (complete) | 0 |
| **Mode Selection** | 3 | 0 | 0 | **3** |

---

## Detailed Inventory

### 1. Transforms

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `VP8FTransform` | `enc.c:161` | - | `transform.rs:dct4x4` | Forward DCT 4x4 |
| `VP8FTransform2` | `enc.c:193` | - | Can compose | Two 4x4 DCTs |
| `VP8FTransformWHT` | `enc.c:201` | - | `transform.rs:wht4x4` | Forward WHT |
| `VP8ITransform` | `enc.c:114` | `vp8_dsp.rs:idct_add` | `transform.rs:idct4x4` | Inverse DCT |
| `transform_wht` (inv) | `enc.c:~215` | `vp8_dsp.rs:transform_wht` | `transform.rs:iwht4x4` | Inverse WHT |

**Status: COMPLETE in image-webp**

### 2. Intra Predictions

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `VP8EncPredLuma16` | `enc.c:349` | `vp8_dsp.rs:pred_luma_16` (SSE2/AVX2) | `vp8_prediction.rs` | 4 modes: DC, V, H, TM |
| `VP8EncPredLuma4` | `enc.c:538` | `vp8_dsp.rs:pred_luma_4` (SSE2) | `vp8_prediction.rs` | 10 modes: B_DC..B_HU |
| `VP8EncPredChroma8` | `enc.c:327` | `vp8_dsp.rs:pred_chroma_8` (SSE2) | `vp8_prediction.rs` | 4 modes: DC, V, H, TM |

**Status: COMPLETE in both** - libwebp-rs has SIMD, image-webp has scalar

### 3. SSE/Distortion Metrics (Sum of Squared Errors)

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `VP8SSE16x16` | `enc.c:573` | - | - | **NEEDED** |
| `VP8SSE16x8` | `enc.c:577` | - | - | **NEEDED** |
| `VP8SSE8x8` | `enc.c:581` | - | - | **NEEDED** |
| `VP8SSE4x4` | `enc.c:585` | - | - | **NEEDED** |
| `VP8TDisto4x4` | `enc.c:650` | - | - | **NEEDED** - spectral distortion |
| `VP8TDisto16x16` | `enc.c:658` | - | - | **NEEDED** - spectral distortion |
| `VP8Mean16x4` | `enc.c:591` | - | - | Optional - DC mean |

**Status: NEED TO PORT** - Critical for mode selection

### 4. Quantization

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `VP8EncQuantizeBlock` | `enc.c:681` | - | Basic in `vp8_encoder.rs` | **NEED PROPER** |
| `VP8EncQuantize2Blocks` | `enc.c:707` | - | - | **NEED** |
| `VP8Matrix` struct | `vp8i_enc.h:184` | - | - | **NEED** - with iq, bias, zthresh |

**Status: NEED PROPER IMPLEMENTATION** - Current is simple division

### 5. Histogram Collection

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `VP8CollectHistogram` | `enc.c:64` | - | - | **NEEDED** for analysis |
| `VP8SetHistogramData` | `enc.c:48` | - | - | Helper |

**Status: NEED TO PORT** - For segment analysis

### 6. Bit Writer (Arithmetic Encoder)

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `VP8BitWriter` | `bit_writer_enc.c` | - | `vp8_arithmetic_encoder.rs` | Complete |
| `VP8PutBit` | same | - | `write_bool` | Complete |
| `VP8PutBitUniform` | same | - | `write_flag` | Complete |

**Status: COMPLETE in image-webp**

### 7. Mode Selection (The Critical Gap)

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `PickBestIntra16` | `quant_enc.c:951` | - | - | **NEEDED** |
| `PickBestIntra4` | `quant_enc.c:1033` | - | - | **NEEDED** |
| `PickBestUV` | `quant_enc.c:1139` | - | - | **NEEDED** |
| `VP8Decimate` | `quant_enc.c:1343` | - | - | Main dispatcher |
| `VP8ModeScore` struct | `vp8i_enc.h:139` | - | - | **NEEDED** |

**Status: NEED TO PORT** - This is the main quality blocker

### 8. Cost Estimation

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| `VP8CalculateLevelCosts` | `cost_enc.c:38` | - | - | Pre-computed tables |
| `VP8GetCostLuma16` | `cost_enc.c:126` | - | - | Rate estimation |
| `VP8GetCostLuma4` | `cost_enc.c:143` | - | - | Rate estimation |
| `VP8GetCostUV` | `cost_enc.c:152` | - | - | Rate estimation |

**Status: NEED TO PORT** - For accurate RD optimization

### 9. Loop Filter (Encoder Side)

| Function | libwebp C | libwebp-rs | image-webp | Notes |
|----------|-----------|------------|------------|-------|
| Simple V/H filters | `enc.c:636+` | `vp8_dsp.rs` (complete) | - | Reuse decoder |
| Normal V/H filters | same | `vp8_dsp.rs` (complete) | - | Reuse decoder |

**Status: COMPLETE in libwebp-rs** - Can reuse for encoder

---

## Functions to Port (Priority Order)

### Phase 1: SSE/Distortion (enables mode selection)
```rust
// In new file: src/enc/sse.rs
fn sse_16x16(a: &[u8], b: &[u8], stride: usize) -> u32;
fn sse_16x8(a: &[u8], b: &[u8], stride: usize) -> u32;
fn sse_8x8(a: &[u8], b: &[u8], stride: usize) -> u32;
fn sse_4x4(a: &[u8], b: &[u8], stride: usize) -> u32;
fn tdisto_4x4(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32;
fn tdisto_16x16(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32;
```

**C Source:** `libwebp/src/dsp/enc.c:557-669`

### Phase 2: Mode Selection Infrastructure
```rust
// In new file: src/enc/mode_score.rs
struct VP8ModeScore {
    D: i64,           // distortion
    SD: i64,          // spectral distortion
    H: i64,           // header bits
    R: i64,           // rate
    score: i64,       // D + lambda*R

    y_dc_levels: [i16; 16],
    y_ac_levels: [[i16; 16]; 16],
    uv_levels: [[i16; 16]; 8],

    mode_i16: i32,
    modes_i4: [u8; 16],
    mode_uv: i32,
    nz: u32,
}
```

### Phase 3: Mode Selection Functions
```rust
// In new file: src/enc/mode_selection.rs
fn pick_best_intra16(it: &mut Iterator, rd: &mut ModeScore, segs: &SegmentInfo);
fn pick_best_intra4(it: &mut Iterator, rd: &mut ModeScore, segs: &SegmentInfo);
fn pick_best_uv(it: &mut Iterator, rd: &mut ModeScore, segs: &SegmentInfo);
fn vp8_decimate(it: &mut Iterator, segs: &SegmentInfo, rd_opt: RdLevel) -> bool;
```

**C Source:** `libwebp/src/enc/quant_enc.c:951-1343`

### Phase 4: Improved Quantization
```rust
// In new file: src/enc/quant.rs
struct VP8Matrix {
    q: [u16; 16],
    iq: [u16; 16],
    bias: [u32; 16],
    zthresh: [u32; 16],
    sharpen: [i16; 16],
}

fn quantize_block(coeffs: &mut [i16; 16], out: &mut [i16; 16], mtx: &VP8Matrix) -> bool;
fn quantize_2blocks(coeffs: &mut [i16; 32], out: &mut [i16; 32], mtx: &VP8Matrix) -> u32;
```

**C Source:** `libwebp/src/dsp/enc.c:681-713`

### Phase 5: Cost Estimation (Optional for v1)
```rust
// In new file: src/enc/cost.rs
struct LevelCosts {
    // Pre-computed cost tables
}

fn calculate_level_costs(proba: &VP8EncProba) -> LevelCosts;
fn get_cost_luma16(levels: &[[i16; 16]; 17], costs: &LevelCosts) -> i64;
```

**C Source:** `libwebp/src/enc/cost_enc.c`

---

## Reusable Components from libwebp-rs

These are already implemented with SSE2/AVX2 optimization:

1. **Predictions** - `pred_luma_16`, `pred_luma_4`, `pred_chroma_8`
2. **Loop Filter** - All simple and normal filter variants
3. **IDCT** - Can verify encoder reconstructions match decoder

---

## What image-webp Already Has

1. **Complete arithmetic encoder** - `vp8_arithmetic_encoder.rs`
2. **All transform functions** - `transform.rs` (DCT, WHT, inverse)
3. **All prediction modes** - `vp8_prediction.rs`
4. **Basic encoder loop** - `vp8_encoder.rs`
5. **Coefficient coding** - `encode_coefficients`
6. **YUV conversion** - `yuv.rs`

---

## Test Infrastructure

Quality comparison tests are available in `~/work/image-webp/tests/lossy_encoder_quality.rs`.

### Running Quality Tests

```bash
cd ~/work/image-webp
cargo test --test lossy_encoder_quality -- --nocapture
```

### Quality Metrics

The tests compare our encoder against libwebp using three metrics:

| Metric | What it Measures | Interpretation |
|--------|------------------|----------------|
| **PSNR** | Pixel-level difference | Higher = better (dB) |
| **DSSIM** | Structural similarity | Lower = better (0 = identical) |
| **SSIMULACRA2** | Perceptual quality | Higher = better (90+ excellent) |

### Current Baseline (DC-only mode)

At quality 75 on synthetic test images:

| Metric | Our Encoder | libwebp | Comparison |
|--------|-------------|---------|------------|
| Size | 7148 bytes | 7124 bytes | ~same |
| PSNR | 16.43 dB | 16.91 dB | 97% |
| DSSIM | 0.0283 | 0.0330 | **14% better** |
| SSIMULACRA2 | 10.45 | 17.36 | 60% |

**Key insight**: SSIMULACRA2 shows the largest gap (60%), indicating mode selection improvements will primarily benefit perceptual quality.

### Size Comparison

```
Quality 50: ours = 756 bytes, libwebp = 658 bytes (1.15x)
Quality 75: ours = 1114 bytes, libwebp = 752 bytes (1.48x)
Quality 90: ours = 1512 bytes, libwebp = 1106 bytes (1.37x)
```

### Dev Dependencies

The tests use:
- `webp` - libwebp Rust wrapper for decoding verification
- `dssim-core` - DSSIM calculation
- `fast-ssim2` - SSIMULACRA2 calculation (SIMD-accelerated)

---

## Recommended Porting Strategy

1. **Port SSE functions first** (~100 lines of trivial code)
   - Can test immediately with unit tests

2. **Add VP8ModeScore struct** (~50 lines)
   - Data structure for RD optimization

3. **Port pick_best_intra16** (~150 lines)
   - Immediate quality improvement
   - Only 4 modes to test

4. **Port pick_best_uv** (~100 lines)
   - Similar to intra16

5. **Port pick_best_intra4** (~200 lines)
   - Most complex but biggest quality gain
   - 10 modes × 16 blocks

6. **Improve quantization** (~100 lines)
   - Better rounding/bias

7. **Add cost estimation** (optional, ~200 lines)
   - More accurate RD decisions

Total new code: ~700-900 lines for significant quality improvement.
