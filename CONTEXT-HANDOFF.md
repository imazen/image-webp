# Context Handoff: Lossy Encoder Parity with libwebp

## Current State (2026-01-21)

The lossy VP8 encoder has been significantly improved but still has a 2-11% size overhead compared to libwebp. The goal is to reach 100% parity.

### Recent Commits
```
d796260 fix: compute optimized segment tree probabilities
85be0ee fix: always use two-pass encoding for better compression
a25cd9e fix: use lambda_mode and BMODE_COST for I4 scoring
```

### Current Benchmark Results (Kodak corpus)
```
Q20: 103.0%    Q60: 104.9%    Q85: 108.3%
Q30: 102.3%    Q70: 105.8%    Q90: 107.3%
Q40: 103.3%    Q75: 105.1%    Q95: 111.4%
Q50: 103.2%    Q80: 105.6%
```

Average PSNR at equal BPP: -1.07 dB (was -1.22 dB before fixes)

## Key Finding: Residual Encoding Overhead

Analysis of VP8 bitstream structure shows:
- **Mode partition**: Now efficient (same or smaller than libwebp)
- **Residual partition**: 2-13% larger than libwebp, increasing at high quality

At Q95 (single image kodak/1.png):
- Mode partition: ours=9277 bytes, libwebp=10319 bytes (-10% smaller!)
- Residual: ours=197016 bytes, libwebp=174251 bytes (+13% larger)

The residual overhead is the bottleneck.

## Areas to Investigate

### 1. Trellis Quantization Efficiency
File: `src/vp8_cost.rs` function `trellis_quantize_block` (line ~1069)

At high quality (Q95), trellis lambda is very low (56 for I4), which should favor distortion reduction. However, we're producing more bits than libwebp.

Possible issues:
- Level cost calculation differences
- Skip/EOB cost estimation
- Context tracking for coefficient costs

Reference: libwebp's `TrellisQuantizeBlock` in `/home/lilith/work/webp-porting/libwebp/src/enc/quant_enc.c` line 569

### 2. Coefficient Cost Estimation
File: `src/vp8_cost.rs` struct `LevelCosts`

The level costs are used for both trellis decisions and mode selection. If these costs don't match libwebp exactly, RD decisions will differ.

Key functions:
- `LevelCosts::calculate()` - computes cost tables from probabilities
- `get_cost_table()`, `get_eob_cost()`, `get_init_cost()`, `get_skip_eob_cost()`

### 3. Mode Selection at High Quality
At Q95, lambda_mode = 1 (very low), so distortion dominates the I4 selection. But lambda_i16 = 432, meaning rate still matters for I16.

The I4 vs I16 decision might be subtly different from libwebp, leading to different coefficient distributions.

File: `src/vp8_encoder.rs` functions:
- `pick_best_intra16()` (line ~1480)
- `pick_best_intra4()` (line ~1766)

## Test Commands

```bash
# Run single-image comparison
cd /home/lilith/work/image-webp
cargo test debug_encode --release -- --ignored --nocapture

# Run full Kodak benchmark
cargo test codec_benchmark --release -- --ignored --nocapture

# Analyze VP8 structure
python3 /tmp/analyze_vp8.py
```

## Reference Files

- libwebp source: `/home/lilith/work/webp-porting/libwebp/src/enc/`
- c2rust reference: `/home/lilith/work/webp-porting/c2rust-reference/`
- Key libwebp files:
  - `quant_enc.c` - quantization, trellis, mode selection
  - `cost_enc.c` - level cost calculation
  - `frame_enc.c` - frame encoding, probability updates

## Lambda Values at Different Quality Levels

```
Q50: quant_index=39, lambda_trellis_i4=1617, lambda_trellis_i16=1089
Q75: quant_index=26, lambda_trellis_i4=787, lambda_trellis_i16=529
Q90: quant_index=9, lambda_trellis_i4=147, lambda_trellis_i16=100
Q95: quant_index=4, lambda_trellis_i4=56, lambda_trellis_i16=36
```

## Hypothesis

At high quality, our trellis quantization is producing more non-zero coefficients than libwebp. The lambda values are correct, but the actual RD decisions during trellis traversal may differ due to:
1. Different rounding/precision in cost calculations
2. Different handling of the distortion term (weight matrix)
3. Context state differences affecting cost lookups

## Next Steps

1. Add debug logging to trellis_quantize_block to compare decisions with libwebp
2. Verify LevelCosts match libwebp's level_cost tables exactly
3. Check if VP8_WEIGHT_TRELLIS weights match libwebp's kWeightTrellis
4. Consider comparing coefficient-by-coefficient output between encoders

## Delete This File

Delete this file after loading it into a new session.
