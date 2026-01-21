# Context Handoff: Lossy Encoder Parity with libwebp

## Current State (2026-01-21)

The lossy VP8 encoder has been significantly improved. At low-to-medium quality settings (Q50, Q75), we now produce SMALLER files than libwebp! At high quality (Q90+) we're about 3-4% larger.

### Recent Commits
```
c919521 fix: align record_coeffs with encoder's skip_eob pattern
d796260 fix: compute optimized segment tree probabilities
85be0ee fix: always use two-pass encoding for better compression
a25cd9e fix: use lambda_mode and BMODE_COST for I4 scoring
```

### Current Benchmark Results (kodak/1.png)
```
Q50: 95.6%  (smaller than libwebp!)
Q75: 98.8%  (smaller than libwebp!)
Q90: 103.6%
Q95: 103.4%
```

Average PSNR at equal BPP: -0.85 dB (was -1.07 dB before record_coeffs fix)

## Key Finding: Statistics Collection Fix

The main issue was in `record_coeffs` - the statistics collection function didn't match
the encoder's coefficient encoding pattern. Specifically:

1. **OLD BUG**: `record_coeffs` looped through ALL 16 positions, even trailing zeros
2. **OLD BUG**: It recorded at node 0 only ONCE at the start
3. **FIX**: Now matches encoder's `skip_eob` pattern exactly:
   - Process only up to end_of_block (last non-zero + 1)
   - Record at node 0 for each coefficient where skip_eob=false
   - Set skip_eob=true after zeros (like encoder does)
   - Do NOT reset skip_eob after non-zeros (encoder leaves it unchanged)

This fix improved compression significantly, especially at lower quality settings.

## Remaining Areas to Investigate (for high quality Q90+)

### 1. Trellis Quantization at High Quality
File: `src/vp8_cost.rs` function `trellis_quantize_block` (line ~1069)

Trellis is verified to match libwebp exactly via `test_trellis_vs_libwebp`.
However, at high quality (Q95), we're still ~3.4% larger. The remaining overhead
may be in mode selection or quantization decisions.

### 2. Mode Selection Differences
At Q95, lambda_mode = 1 (very low), so distortion dominates the I4 selection.
But lambda_i16 = 432, meaning rate still matters for I16.

The I4 vs I16 decision might be subtly different from libwebp, leading to
different coefficient distributions.

File: `src/vp8_encoder.rs` functions:
- `pick_best_intra16()` (line ~1480)
- `pick_best_intra4()` (line ~1766)

### 3. Potential Further Optimization
- The remaining ~3% overhead at Q90+ might be acceptable
- Consider investigating if there are quantization table differences
- Check if AC/DC prediction differences affect high quality encoding

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

## Summary

The main bottleneck was the `record_coeffs` function not matching the encoder's
bitstream encoding pattern. After fixing this:

- **Low/Medium Quality (Q50-Q75)**: We're now SMALLER than libwebp!
- **High Quality (Q90-Q95)**: Still ~3-4% larger, likely due to mode selection or
  quantization differences that have less impact at these settings.

The fix ensures probability estimates are accurate, leading to better entropy coding.

## Delete This File

Delete this file after loading it into a new session.
