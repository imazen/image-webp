# WebP Lossy Encoder Analysis for Porting to Rust

This document provides a comprehensive analysis of libwebp's lossy (VP8-based) encoder for porting to the `image-webp` Rust crate.

## Table of Contents
1. [Quick Start](#quick-start)
2. [High-Level Overview](#high-level-overview)
3. [Encoding Pipeline Flowchart](#encoding-pipeline-flowchart)
4. [Key Data Structures](#key-data-structures)
5. [Algorithm & Math Pseudocode](#algorithm--math-pseudocode)
6. [DSP Functions](#dsp-functions)
7. [Porting Plan](#porting-plan)

---

## Quick Start

### Running Quality Tests

```bash
cd ~/work/image-webp
cargo test --test lossy_encoder_quality -- --nocapture
```

This compares our encoder against libwebp using PSNR, DSSIM, and SSIMULACRA2 metrics.

### Current Status

The encoder is functional but uses **DC-only mode selection** (no RD optimization). Quality metrics at Q75:

| Metric | vs libwebp | Notes |
|--------|------------|-------|
| PSNR | 97% | Nearly identical |
| DSSIM | 14% better | Structural similarity |
| SSIMULACRA2 | 60% | **Main improvement target** |

### Related Documents

- **[ENCODER_COMPONENT_INVENTORY.md](ENCODER_COMPONENT_INVENTORY.md)** - What's implemented vs needed
- **[FUNCTIONS_NEEDED.md](FUNCTIONS_NEEDED.md)** - Detailed function signatures to port

---

## High-Level Overview

WebP lossy encoding is based on VP8 intra-frame encoding. The key concepts:

- **Macroblock-based**: Image divided into 16x16 macroblocks
- **Intra prediction**: Predict pixels from neighbors (no inter-frame)
- **DCT transform**: 4x4 discrete cosine transform on residuals
- **Quantization**: Lossy step - reduces precision of DCT coefficients
- **Entropy coding**: Boolean arithmetic coder with adaptive probabilities

### Encoding Modes
1. **Intra16x16 (I16)**: Entire 16x16 luma block predicted with one mode
2. **Intra4x4 (I4)**: Each 4x4 sub-block predicted independently
3. **Chroma**: 8x8 U/V blocks always predicted as a unit

### Prediction Modes

**16x16 Luma (4 modes):**
- DC_PRED (0): Fill with average of top + left
- TM_PRED (1): TrueMotion - top + left - top_left
- V_PRED (2): Copy top row vertically
- H_PRED (3): Copy left column horizontally

**4x4 Luma (10 modes):**
- B_DC_PRED (0), B_TM_PRED (1), B_VE_PRED (2), B_HE_PRED (3)
- B_RD_PRED (4), B_VR_PRED (5), B_LD_PRED (6), B_VL_PRED (7)
- B_HD_PRED (8), B_HU_PRED (9)

**8x8 Chroma (4 modes):** Same as 16x16 luma

---

## Encoding Pipeline Flowchart

```
                    ┌─────────────────────────┐
                    │    WebPEncode()         │
                    │  (webp_enc.c:335)       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  Input Validation       │
                    │  - Config validation    │
                    │  - Picture validation   │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │ lossless?                         │
              │                                   │
    ┌─────────▼─────────┐           ┌─────────────▼─────────────┐
    │ VP8LEncodeImage() │           │ LOSSY ENCODING PATH       │
    │ (lossless)        │           │                           │
    └───────────────────┘           └─────────────┬─────────────┘
                                                  │
                                    ┌─────────────▼─────────────┐
                                    │ RGB→YUV420 Conversion     │
                                    │ WebPPictureARGBToYUVA()   │
                                    │ or SharpYUV               │
                                    └─────────────┬─────────────┘
                                                  │
                                    ┌─────────────▼─────────────┐
                                    │ InitVP8Encoder()          │
                                    │ - Allocate memory         │
                                    │ - Setup quantization      │
                                    │ - Initialize probabilities│
                                    └─────────────┬─────────────┘
                                                  │
                                    ┌─────────────▼─────────────┐
                                    │ VP8EncAnalyze()           │
                                    │ (analysis_enc.c)          │
                                    │ - Segment analysis        │
                                    │ - Complexity estimation   │
                                    │ - Initial mode decisions  │
                                    └─────────────┬─────────────┘
                                                  │
                              ┌────────────────────┴────────────────────┐
                              │ use_tokens?                             │
                              │                                         │
                    ┌─────────▼─────────┐               ┌───────────────▼───────────────┐
                    │ VP8EncLoop()      │               │ VP8EncTokenLoop()             │
                    │ (Basic path)      │               │ (Multi-pass with token buffer)│
                    └─────────┬─────────┘               └───────────────┬───────────────┘
                              │                                         │
                              └─────────────────┬───────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │  MAIN ENCODING LOOP   │
                                    │  For each macroblock: │
                                    └───────────┬───────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
          ┌─────────▼─────────┐     ┌───────────▼───────────┐   ┌───────────▼───────────┐
          │ VP8MakeLuma16Preds│     │ VP8MakeChroma8Preds   │   │ MakeIntra4Preds       │
          │ (4 predictions)   │     │ (4 predictions)       │   │ (10 predictions)      │
          └─────────┬─────────┘     └───────────┬───────────┘   └───────────┬───────────┘
                    │                           │                           │
                    └───────────────────────────┼───────────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ VP8Decimate()         │
                                    │ (quant_enc.c:1343)    │
                                    │ Mode selection &      │
                                    │ quantization          │
                                    └───────────┬───────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
          ┌─────────▼─────────┐     ┌───────────▼───────────┐   ┌───────────▼───────────┐
          │ PickBestIntra16() │     │ PickBestIntra4()      │   │ PickBestUV()          │
          │ or RefineUsing    │     │ (if better score)     │   │                       │
          │ Distortion()      │     │                       │   │                       │
          └─────────┬─────────┘     └───────────┬───────────┘   └───────────┬───────────┘
                    │                           │                           │
                    └───────────────────────────┼───────────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ For each block:       │
                                    │ 1. Compute residual   │
                                    │    (src - prediction) │
                                    │ 2. Forward DCT        │
                                    │ 3. Quantize           │
                                    │ 4. Inverse DCT        │
                                    │ 5. Reconstruct        │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ CodeResiduals() or    │
                                    │ RecordTokens()        │
                                    │ - Entropy code coeffs │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ VP8EncWrite()         │
                                    │ (syntax_enc.c:320)    │
                                    │ - Write RIFF header   │
                                    │ - Write VP8 bitstream │
                                    └───────────────────────┘
```

---

## Key Data Structures

### VP8Encoder (Main Encoder State)
```rust
/// Main encoder state - one per encoding operation
struct VP8Encoder {
    // Configuration
    config: WebPConfig,
    pic: WebPPicture,

    // Dimensions in macroblocks (16x16 blocks)
    mb_w: i32,           // width in macroblocks
    mb_h: i32,           // height in macroblocks
    preds_w: i32,        // stride of prediction plane (4*mb_w + 1)

    // Partitions (VP8 supports 1, 2, 4, or 8 token partitions)
    num_parts: i32,
    bw: VP8BitWriter,              // partition 0 (headers + modes)
    parts: [VP8BitWriter; 8],      // token partitions
    tokens: VP8TBuffer,            // optional token buffer for multi-pass

    // Quantization info (4 segments max)
    dqm: [VP8SegmentInfo; 4],      // per-segment quantization
    base_quant: i32,               // nominal quantizer [0..127]

    // Quantizer deltas
    dq_y1_dc: i32,    // luma DC delta
    dq_y2_dc: i32,    // luma DC (for I16) delta
    dq_y2_ac: i32,    // luma AC (for I16) delta
    dq_uv_dc: i32,    // chroma DC delta
    dq_uv_ac: i32,    // chroma AC delta

    // Probability tables
    proba: VP8EncProba,

    // Memory buffers
    mb_info: Vec<VP8MBInfo>,    // per-macroblock info
    preds: Vec<u8>,             // prediction modes (4x4 grid)
    nz: Vec<u32>,               // non-zero coefficient flags
    y_top: Vec<u8>,             // top luma samples for prediction
    uv_top: Vec<u8>,            // top chroma samples

    // Quality settings
    method: i32,                // 0=fastest, 6=slowest
    rd_opt_level: VP8RDLevel,   // rate-distortion optimization level
}
```

### VP8MBInfo (Per-Macroblock Info)
```rust
/// Information stored per macroblock
struct VP8MBInfo {
    type_: u8,      // 0 = I4x4, 1 = I16x16
    uv_mode: u8,    // chroma prediction mode [0..3]
    skip: bool,     // true if all coefficients are zero
    segment: u8,    // segment index [0..3]
    alpha: u8,      // quantization susceptibility (complexity)
}
```

### VP8Matrix (Quantization Parameters)
```rust
/// Quantization matrix for a coefficient type
struct VP8Matrix {
    q: [u16; 16],        // quantizer steps
    iq: [u16; 16],       // reciprocals (fixed point) for division
    bias: [u32; 16],     // rounding bias
    zthresh: [u32; 16],  // threshold below which coeff becomes zero
    sharpen: [u16; 16],  // sharpening boost for mid-range quality
}
```

### VP8SegmentInfo (Per-Segment Quantization)
```rust
/// Quantization parameters for one segment
struct VP8SegmentInfo {
    y1: VP8Matrix,      // luma AC quantization
    y2: VP8Matrix,      // luma DC (I16 mode) quantization
    uv: VP8Matrix,      // chroma quantization

    alpha: i32,         // complexity indicator [-127..127]
    beta: i32,          // filter susceptibility [0..255]
    quant: i32,         // quantizer index [0..127]
    fstrength: i32,     // loop filter strength [0..63]

    // Lambda values for RD optimization
    lambda_i16: i32,
    lambda_i4: i32,
    lambda_uv: i32,
    lambda_mode: i32,
    lambda_trellis: i32,

    i4_penalty: i64,    // penalty for choosing I4 over I16
}
```

### VP8ModeScore (RD Scoring)
```rust
/// Accumulator for rate-distortion optimization
struct VP8ModeScore {
    D: i64,                      // distortion (SSE)
    SD: i64,                     // spectral distortion
    H: i64,                      // header bits
    R: i64,                      // rate (total bits)
    score: i64,                  // R + lambda * D

    y_dc_levels: [i16; 16],      // I16 DC coefficients
    y_ac_levels: [[i16; 16]; 16], // Y AC coefficients (16 blocks)
    uv_levels: [[i16; 16]; 8],   // UV coefficients (8 blocks)

    mode_i16: i32,               // I16 prediction mode
    modes_i4: [u8; 16],          // I4 prediction modes
    mode_uv: i32,                // UV prediction mode
    nz: u32,                     // non-zero coefficient flags
}
```

### VP8EncIterator (Macroblock Iterator)
```rust
/// Iterator for traversing macroblocks
struct VP8EncIterator {
    x: i32,              // current MB x position
    y: i32,              // current MB y position

    yuv_in: [u8; YUV_SIZE],    // input samples
    yuv_out: [u8; YUV_SIZE],   // reconstructed samples
    yuv_out2: [u8; YUV_SIZE],  // scratch buffer
    yuv_p: [u8; PRED_SIZE],    // predictions cache

    mb: *mut VP8MBInfo,        // current macroblock
    preds: *mut u8,            // prediction modes
    nz: *mut u32,              // non-zero flags

    i4_boundary: [u8; 37],     // boundary samples for I4
    i4_top: *mut u8,           // pointer to top boundary
    i4: i32,                   // current I4 sub-block [0..15]

    top_nz: [i32; 9],          // top non-zero context
    left_nz: [i32; 9],         // left non-zero context

    y_left: *mut u8,           // left Y samples
    u_left: *mut u8,           // left U samples
    v_left: *mut u8,           // left V samples
    y_top: *mut u8,            // top Y samples
    uv_top: *mut u8,           // top UV samples
}
```

### VP8EncProba (Probability Tables)
```rust
/// Frame-persistent probability tables
struct VP8EncProba {
    segments: [u8; 3],                               // segment tree probabilities
    skip_proba: u8,                                  // skip probability
    use_skip_proba: bool,

    coeffs: [[[[u8; 11]; 3]; 8]; 4],                // coefficient probabilities
                                                     // [type][band][ctx][proba]
    stats: [[[[u32; 11]; 3]; 8]; 4],                // statistics for updating
    level_cost: [[[[u16; 68]; 3]; 8]; 4],           // precomputed costs

    dirty: bool,                                     // need to recalculate costs
    nb_skip: i32,                                    // number of skipped MBs
}
```

---

## Algorithm & Math Pseudocode

### 1. Main Encoding Loop

```
function VP8EncLoop(encoder):
    // Statistics collection pass
    StatLoop(encoder)

    // Initialize bit writers
    for each macroblock (x, y):
        // Import source samples
        import_macroblock(x, y) → yuv_in

        // Generate all predictions
        make_luma16_predictions(yuv_p)
        make_chroma8_predictions(yuv_p)

        // Find best modes via RD optimization
        rd_info = VP8Decimate(iterator, rd_opt_level)

        if rd_info.nz == 0:  // all zero
            mark_as_skip()
        else:
            // Encode coefficients
            code_residuals(bitwriter, rd_info)

        // Save reconstructed boundary for next MB
        save_boundary()

    // Finalize bitstream
    VP8EncWrite()
```

### 2. Mode Selection (VP8Decimate)

```
function VP8Decimate(iterator, rd_opt):
    init_score(rd)

    // Make prediction candidates
    VP8MakeLuma16Preds()
    VP8MakeChroma8Preds()

    if rd_opt > RD_OPT_NONE:
        // Full RD optimization
        PickBestIntra16(iterator, rd)

        if method >= 2:
            PickBestIntra4(iterator, rd)  // may override I16

        PickBestUV(iterator, rd)

        if rd_opt == RD_OPT_TRELLIS:
            SimpleQuantize(iterator, rd)  // final trellis pass
    else:
        // Fast mode: distortion-based selection only
        RefineUsingDistortion(iterator, rd)

    return rd.nz == 0  // is_skipped
```

### 3. Intra16 Mode Selection

```
function PickBestIntra16(iterator, rd):
    best_score = MAX_COST

    for mode in [DC_PRED, TM_PRED, V_PRED, H_PRED]:
        // Reconstruct with this mode
        nz = ReconstructIntra16(yuv_out, mode)

        // Compute distortion
        D = SSE16x16(yuv_in, yuv_out)
        SD = tlambda * TDisto16x16(yuv_in, yuv_out)  // spectral

        // Compute rate
        H = FixedCostsI16[mode]  // header bits
        R = GetCostLuma16(levels)  // coefficient bits

        // RD score
        score = (R + H) * lambda + (D + SD) * RD_DISTO_MULT

        if score < best_score:
            best_score = score
            best_mode = mode
            save_levels()

    rd.mode_i16 = best_mode
```

### 4. Forward DCT (4x4)

```
function FTransform(src[4][4], ref[4][4]) → out[16]:
    // Compute residual and transform
    tmp[16]

    // Horizontal pass (rows)
    for i in 0..4:
        d0 = src[i][0] - ref[i][0]  // residual, 9-bit
        d1 = src[i][1] - ref[i][1]
        d2 = src[i][2] - ref[i][2]
        d3 = src[i][3] - ref[i][3]

        a0 = d0 + d3    // 10-bit
        a1 = d1 + d2
        a2 = d1 - d2
        a3 = d0 - d3

        tmp[0 + i*4] = (a0 + a1) * 8           // 14-bit
        tmp[1 + i*4] = (a2*2217 + a3*5352 + 1812) >> 9
        tmp[2 + i*4] = (a0 - a1) * 8
        tmp[3 + i*4] = (a3*2217 - a2*5352 + 937) >> 9

    // Vertical pass (columns)
    for i in 0..4:
        a0 = tmp[0 + i] + tmp[12 + i]  // 15-bit
        a1 = tmp[4 + i] + tmp[8 + i]
        a2 = tmp[4 + i] - tmp[8 + i]
        a3 = tmp[0 + i] - tmp[12 + i]

        out[0 + i] = (a0 + a1 + 7) >> 4        // 12-bit output
        out[4 + i] = ((a2*2217 + a3*5352 + 12000) >> 16) + (a3 != 0)
        out[8 + i] = (a0 - a1 + 7) >> 4
        out[12 + i] = (a3*2217 - a2*5352 + 51000) >> 16

    return out
```

### 5. Quantization

```
function QuantizeBlock(coeffs[16], mtx: VP8Matrix) → (levels[16], nz):
    // Zigzag order for coefficient scanning
    ZIGZAG = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]

    last = -1  // last non-zero position

    for n in 0..16:
        j = ZIGZAG[n]
        sign = coeffs[j] < 0
        coeff = abs(coeffs[j]) + mtx.sharpen[j]  // sharpening boost

        if coeff > mtx.zthresh[j]:  // above zero threshold?
            // QUANTDIV: (coeff * iq + bias) >> QFIX
            // where QFIX = 17
            level = (coeff * mtx.iq[j] + mtx.bias[j]) >> 17
            level = min(level, MAX_LEVEL)  // MAX_LEVEL = 2047

            if sign: level = -level

            coeffs[j] = level * mtx.q[j]  // dequantize for reconstruction
            levels[n] = level
            if level != 0: last = n
        else:
            levels[n] = 0
            coeffs[j] = 0

    return (levels, last >= 0)
```

### 6. Inverse DCT (4x4)

```
function ITransform(ref[4][4], coeffs[16]) → dst[4][4]:
    C[16]

    // Constants for approximate DCT
    // WEBP_TRANSFORM_AC3_MUL1(x) = x * 20091 >> 16
    // WEBP_TRANSFORM_AC3_MUL2(x) = x * 35468 >> 16

    // Vertical pass (process columns)
    for i in 0..4:
        a = coeffs[0 + i] + coeffs[8 + i]
        b = coeffs[0 + i] - coeffs[8 + i]
        c = MUL2(coeffs[4 + i]) - MUL1(coeffs[12 + i])
        d = MUL1(coeffs[4 + i]) + MUL2(coeffs[12 + i])

        C[0 + i*4] = a + d
        C[1 + i*4] = b + c
        C[2 + i*4] = b - c
        C[3 + i*4] = a - d

    // Horizontal pass (process rows)
    for i in 0..4:
        dc = C[i*4 + 0] + 4  // rounding
        a = dc + C[i*4 + 2]
        b = dc - C[i*4 + 2]
        c = MUL2(C[i*4 + 1]) - MUL1(C[i*4 + 3])
        d = MUL1(C[i*4 + 1]) + MUL2(C[i*4 + 3])

        // Add to reference and clip to [0, 255]
        dst[i][0] = clip(ref[i][0] + ((a + d) >> 3))
        dst[i][1] = clip(ref[i][1] + ((b + c) >> 3))
        dst[i][2] = clip(ref[i][2] + ((b - c) >> 3))
        dst[i][3] = clip(ref[i][3] + ((a - d) >> 3))
```

### 7. Walsh-Hadamard Transform (for I16 DC)

```
function FTransformWHT(dc_in[16]) → dc_out[16]:
    tmp[16]

    // Input is DC coefficients from 16 4x4 blocks (arranged 4x4)
    // Process in 4x4 groups

    // Horizontal pass
    for i in 0..4:
        a0 = dc_in[i*4 + 0] + dc_in[i*4 + 2]
        a1 = dc_in[i*4 + 1] + dc_in[i*4 + 3]
        a2 = dc_in[i*4 + 1] - dc_in[i*4 + 3]
        a3 = dc_in[i*4 + 0] - dc_in[i*4 + 2]

        tmp[i*4 + 0] = a0 + a1
        tmp[i*4 + 1] = a3 + a2
        tmp[i*4 + 2] = a3 - a2
        tmp[i*4 + 3] = a0 - a1

    // Vertical pass
    for i in 0..4:
        a0 = tmp[0 + i] + tmp[8 + i]
        a1 = tmp[4 + i] + tmp[12 + i]
        a2 = tmp[4 + i] - tmp[12 + i]
        a3 = tmp[0 + i] - tmp[8 + i]

        b0 = a0 + a1
        b1 = a3 + a2
        b2 = a3 - a2
        b3 = a0 - a1

        dc_out[0 + i] = b0 >> 1
        dc_out[4 + i] = b1 >> 1
        dc_out[8 + i] = b2 >> 1
        dc_out[12 + i] = b3 >> 1
```

### 8. Boolean Arithmetic Coding

```
function VP8PutBit(bw: BitWriter, bit: bool, prob: u8):
    // prob is P(bit=0) in [1..255], where 255 means ~100% zero

    split = 1 + (((bw.range - 1) * prob) >> 8)

    if bit:
        bw.value += split
        bw.range -= split
    else:
        bw.range = split

    // Normalize: ensure range >= 128
    while bw.range < 128:
        if bw.value >= 256:  // carry
            propagate_carry()

        bw.range <<= 1
        bw.value <<= 1
        bw.bits += 1

        if bw.bits >= 8:
            emit_byte()
```

### 9. Coefficient Coding

```
function PutCoeffs(bw: BitWriter, ctx: i32, res: VP8Residual):
    // Probability tables indexed by [type][band][context][proba]
    p = res.prob[band][ctx]

    // First: code whether block has any non-zero
    if not PutBit(bw, res.last >= 0, p[0]):
        return 0  // all zeros

    for n in res.first..16:
        c = res.coeffs[n]
        sign = c < 0
        v = abs(c)

        // Code "is non-zero?"
        if not PutBit(bw, v != 0, p[1]):
            p = res.prob[band(n+1)][0]  // context = 0 after zero
            continue

        // Code magnitude
        if not PutBit(bw, v > 1, p[2]):
            p = res.prob[band(n+1)][1]  // context = 1 after ±1
        else:
            // Value > 1, use tree to code magnitude
            if not PutBit(bw, v > 4, p[3]):
                // 2, 3, or 4
                ...
            else:
                // 5..10 or 11+
                // Use category codes (VP8Cat3..VP8Cat6)
                ...
            p = res.prob[band(n+1)][2]  // context = 2 after |v|>1

        // Code sign
        PutBitUniform(bw, sign)

        // Code "more coefficients?"
        if n == 16 or not PutBit(bw, n <= res.last, p[0]):
            return 1  // EOB
```

---

## DSP Functions

### Core Functions to Implement

| Function | Purpose | Location |
|----------|---------|----------|
| `VP8FTransform` | 4x4 forward DCT | enc.c:161 |
| `VP8FTransform2` | Two 4x4 forward DCTs | enc.c:193 |
| `VP8FTransformWHT` | 4x4 WHT for I16 DC | enc.c:201 |
| `VP8ITransform` | 4x4 inverse DCT | enc.c:114 |
| `VP8EncPredLuma16` | 16x16 intra predictions | enc.c:349 |
| `VP8EncPredLuma4` | 4x4 intra predictions | enc.c:538 |
| `VP8EncPredChroma8` | 8x8 chroma predictions | enc.c:327 |
| `VP8EncQuantizeBlock` | Quantize 4x4 block | enc.c:681 |
| `VP8EncQuantize2Blocks` | Quantize two 4x4 blocks | enc.c:707 |
| `VP8EncQuantizeBlockWHT` | Quantize WHT block | same as QuantizeBlock |
| `VP8SSE16x16` | SSE for 16x16 block | enc.c:573 |
| `VP8SSE16x8` | SSE for 16x8 block | enc.c:577 |
| `VP8SSE8x8` | SSE for 8x8 block | enc.c:581 |
| `VP8SSE4x4` | SSE for 4x4 block | enc.c:585 |
| `VP8TDisto4x4` | Spectral distortion 4x4 | enc.c:650 |
| `VP8TDisto16x16` | Spectral distortion 16x16 | enc.c:658 |
| `VP8Mean16x4` | Mean of 16x4 block | enc.c:591 |
| `VP8CollectHistogram` | DCT histogram | enc.c:64 |

### SIMD Considerations

The C implementations can be autovectorized, but libwebp has optimized versions:
- SSE2: Most functions (x86)
- SSE4.1: Some functions (x86)
- NEON: ARM platforms

For Rust, consider:
1. Start with scalar C-equivalent code
2. Use `std::simd` (nightly) or `packed_simd` crate
3. Use `safe_arch` or raw intrinsics for platform-specific

---

## Current State: image-webp Crate

The `image-webp` crate already has a **working VP8 lossy encoder**! Here's what exists:

### Existing Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Arithmetic Encoder | `vp8_arithmetic_encoder.rs` | **Complete** | Bool coding, tree coding |
| DCT 4x4 | `transform.rs:dct4x4` | **Complete** | Forward transform |
| IDCT 4x4 | `transform.rs:idct4x4` | **Complete** | Inverse transform |
| WHT 4x4 | `transform.rs:wht4x4` | **Complete** | Forward Walsh-Hadamard |
| IWHT 4x4 | `transform.rs:iwht4x4` | **Complete** | Inverse Walsh-Hadamard |
| 16x16 Prediction | `vp8_prediction.rs` | **Complete** | DC, V, H, TM modes |
| 8x8 Chroma Pred | `vp8_prediction.rs` | **Complete** | DC, V, H, TM modes |
| 4x4 Prediction | `vp8_prediction.rs` | **Complete** | All 10 intra modes |
| VP8 Encoder | `vp8_encoder.rs` | **Basic** | Works but limited |
| YUV Conversion | `yuv.rs` | **Complete** | RGB→YUV420 |
| Coefficient Coding | `vp8_encoder.rs:encode_coefficients` | **Complete** | Token coding |
| Quantization | `vp8_encoder.rs` | **Basic** | Simple division only |

### Key Gaps in image-webp

1. **Mode Selection** - Hardcoded to DC mode for all macroblocks
   ```rust
   fn choose_macroblock_info(&self, _mbx: usize, _mby: usize) -> MacroblockInfo {
       let (luma_mode, luma_bpred) = (LumaMode::DC, None);  // Always DC!
       let chroma_mode = ChromaMode::DC;
       // ...
   }
   ```

2. **No Rate-Distortion Optimization** - Can't evaluate which mode is best
3. **No SSE/Distortion Functions** - Missing `VP8SSE16x16`, `VP8SSE4x4`, etc.
4. **No Spectral Distortion** - Missing `VP8TDisto4x4/16x16`
5. **No Multi-Pass** - Single pass only, no probability refinement
6. **No Trellis Quantization** - Simple floor division
7. **No Segment Analysis** - No adaptive quantization per region
8. **No Quality Levels** - No `method` parameter for speed/quality trade-off

---

## Porting Plan (Updated)

Given the existing foundation, the plan focuses on adding the **missing functionality**.

### Phase 1: Distortion Metrics (Foundation)
**Priority: CRITICAL - enables all subsequent work**

These functions are needed to compare prediction modes:

```rust
// Sum of squared errors (distortion measurement)
fn sse_16x16(a: &[u8], b: &[u8], stride: usize) -> u32;
fn sse_8x8(a: &[u8], b: &[u8], stride: usize) -> u32;
fn sse_4x4(a: &[u8], b: &[u8], stride: usize) -> u32;

// Spectral distortion (frequency-weighted)
fn tdisto_4x4(src: &[u8], pred: &[u8], stride: usize, weights: &[u16]) -> u32;
fn tdisto_16x16(src: &[u8], pred: &[u8], stride: usize, weights: &[u16]) -> u32;
```

**Source:** `libwebp/src/dsp/enc.c:573-658`

### Phase 2: Mode Selection Infrastructure
**Priority: HIGH**

1. Add `VP8ModeScore` struct for RD scoring:
   ```rust
   struct VP8ModeScore {
       distortion: i64,      // SSE
       rate: i64,            // bits used
       score: i64,           // D + lambda * R
       mode_i16: u8,
       modes_i4: [u8; 16],
       mode_uv: u8,
       nz: u32,              // non-zero flags
       // coefficient storage...
   }
   ```

2. Add lambda calculation (quality → lambda mapping)

### Phase 3: Best Mode Selection Functions
**Priority: HIGH**

Port the core mode selection:

```rust
// Try all 16x16 modes, return best
fn pick_best_intra16(it: &mut Iterator, rd: &mut ModeScore);

// Try all 4x4 modes for each sub-block
fn pick_best_intra4(it: &mut Iterator, rd: &mut ModeScore);

// Try all chroma modes
fn pick_best_uv(it: &mut Iterator, rd: &mut ModeScore);
```

**Source:** `libwebp/src/enc/quant_enc.c:700-1100`

### Phase 4: Cost Estimation
**Priority: MEDIUM**

Pre-computed rate tables for fast mode decisions:

```rust
// Precompute coefficient coding costs
fn init_level_costs(proba: &VP8EncProba) -> LevelCosts;

// Cost of coding a single coefficient
fn get_level_cost(level: i32, costs: &LevelCosts, ctx: usize) -> u32;
```

**Source:** `libwebp/src/enc/cost_enc.c`

### Phase 5: Improved Quantization
**Priority: MEDIUM**

1. Add `VP8Matrix` with bias and sharpen:
   ```rust
   struct VP8Matrix {
       q: [u16; 16],       // quantizer steps
       iq: [u16; 16],      // reciprocals
       bias: [u32; 16],    // rounding bias
       zthresh: [u32; 16], // zero threshold
       sharpen: [i16; 16], // sharpening (negative = boost)
   }
   ```

2. Proper quantization with rounding:
   ```rust
   fn quantize_block(coeffs: &mut [i16; 16], mtx: &VP8Matrix) -> bool;
   ```

**Source:** `libwebp/src/enc/quant_enc.c:200-400`

### Phase 6: Segment Analysis (Optional)
**Priority: LOW - significant quality improvement**

1. Histogram collection during first pass
2. K-means clustering for segment assignment
3. Per-segment quantization parameters

**Source:** `libwebp/src/enc/analysis_enc.c`

### Phase 7: Advanced Features (Optional)
**Priority: LOW**

1. **Trellis quantization** - optimal coefficient selection
2. **Multi-pass encoding** - probability refinement
3. **Skip optimization** - detect all-zero blocks

---

## Implementation Strategy

### Recommended Approach: Incremental

1. **Start with SSE functions** - Can test immediately
2. **Add best mode selection one mode at a time**:
   - First: `pick_best_intra16` (4 modes to test)
   - Then: `pick_best_uv` (4 modes)
   - Finally: `pick_best_intra4` (10 modes × 16 blocks)
3. **Add cost estimation** after basic RD works
4. **Improve quantization** once modes are being selected
5. **Segment analysis** as a final polish

### Testing Strategy

1. **Unit tests for each SSE function** - compare with libwebp
2. **Visual comparison** - encode same image, compare quality
3. **SSIMULACRA2/Butteraugli** - objective quality metrics
4. **Round-trip test** - encode → decode with libwebp → compare
5. **Bitstream compatibility** - ensure libwebp can decode output

### Key Files to Create/Modify

| File | Purpose |
|------|---------|
| `src/enc/sse.rs` (new) | SSE/distortion functions |
| `src/enc/cost.rs` (new) | Rate estimation |
| `src/enc/mode_selection.rs` (new) | Mode decision logic |
| `src/vp8_encoder.rs` | Integrate new mode selection |
| `src/enc/quant.rs` (new) | Improved quantization |

---

## Detailed Algorithm: Mode Selection

Here's pseudocode for the key `pick_best_intra16` function:

```rust
fn pick_best_intra16(it: &Iterator, rd: &mut ModeScore, lambda: i64) {
    let mut best_score = i64::MAX;

    for mode in [DC_PRED, TM_PRED, V_PRED, H_PRED] {
        // Skip modes that aren't available (e.g., V_PRED on top row)
        if !mode_available(it, mode) { continue; }

        // 1. Generate prediction
        let pred = predict_16x16(it.yuv_in, mode, it.y_left, it.y_top);

        // 2. Compute distortion
        let sse = sse_16x16(it.yuv_in, &pred);
        let tdisto = tdisto_16x16(it.yuv_in, &pred, WEIGHTS);
        let D = sse + (lambda_trellis * tdisto) >> DISTO_SHIFT;

        // 3. Compute residual and transform
        let mut residual = compute_residual(&it.yuv_in, &pred);
        transform_16x16(&mut residual);  // DCT each 4x4, WHT on DC

        // 4. Quantize
        let mut levels = [[0i16; 16]; 17];  // 16 AC + 1 DC
        quantize_all(&residual, &levels, &dqm.y1, &dqm.y2);

        // 5. Compute rate (bits to code coefficients)
        let R = compute_rate(&levels, proba);
        let H = MODE_COST[mode];  // header bits for mode

        // 6. RD score
        let score = (R + H) * lambda + D;

        if score < best_score {
            best_score = score;
            rd.mode_i16 = mode;
            rd.y_dc_levels = levels[16];
            rd.y_ac_levels = levels[0..16];
            rd.distortion = D;
            rd.rate = R + H;
        }
    }
}
```

This pattern repeats for `pick_best_intra4` and `pick_best_uv` with appropriate modifications.

---

## Comparison: Simple vs Full Mode Selection

| Aspect | Current (DC only) | With RD Optimization |
|--------|-------------------|----------------------|
| Quality | Poor for edges/gradients | Adapts to content |
| Speed | Very fast | 5-20x slower |
| File size | Larger (wastes bits) | More efficient |
| Visual | Blocky artifacts | Much cleaner |

The quality improvement from proper mode selection is **dramatic** - it's the most important missing feature.

---

## File-by-File Mapping

| C File | Purpose | Rust Module |
|--------|---------|-------------|
| `webp_enc.c` | Entry point | `encoder/mod.rs` |
| `analysis_enc.c` | Macroblock analysis | `encoder/analysis.rs` |
| `frame_enc.c` | Main loop, residual coding | `encoder/frame.rs` |
| `quant_enc.c` | Quantization, mode selection | `encoder/quant.rs` |
| `syntax_enc.c` | Bitstream writing | `encoder/syntax.rs` |
| `tree_enc.c` | Probability tables, mode coding | `encoder/tree.rs` |
| `token_enc.c` | Token buffer | `encoder/token.rs` |
| `iterator_enc.c` | Macroblock iteration | `encoder/iterator.rs` |
| `filter_enc.c` | Loop filter | `encoder/filter.rs` |
| `cost_enc.c` | Rate estimation | `encoder/cost.rs` |
| `dsp/enc.c` | DSP functions | `dsp/enc.rs` |

---

## Testing Strategy

1. **Unit tests per DSP function**: Compare output with libwebp
2. **Round-trip test**: Encode → Decode → Compare
3. **Bitstream compatibility**: Ensure output decodes with libwebp
4. **Quality metrics**: SSIMULACRA2/Butteraugli comparison
5. **Fuzz testing**: Random inputs for robustness

---

## Risk Areas

1. **Numeric precision**: Fixed-point math must match exactly
2. **Probability tables**: Large constant arrays to verify
3. **Bit-exact output**: Boolean coder must produce identical bits
4. **SIMD**: Platform differences in rounding
5. **Memory layout**: BPS (stride) assumptions

---

## Recommended Order of Implementation

1. `dsp/enc.rs` - DSP functions (testable in isolation)
2. `encoder/cost.rs` - Rate estimation tables
3. `encoder/quant.rs` - Quantization logic
4. `encoder/iterator.rs` - Macroblock iteration
5. `encoder/analysis.rs` - Segment analysis
6. `encoder/frame.rs` - Main encoding loop
7. `encoder/syntax.rs` - Bitstream assembly
8. Integration and testing
9. Multi-pass and advanced features
10. SIMD optimization
