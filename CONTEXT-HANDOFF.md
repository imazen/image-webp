# Context Handoff: Lossy Encoder Parity with libwebp

## Current State (2026-01-21)

**STATUS: QUALITY GAP DETECTED** - The encoder is functional but has significant quality and size gaps vs libwebp on the CLIC 2025 validation set. Kodak results were more favorable, suggesting image-dependent issues.

### CLIC 2025 Validation Benchmark (32 high-res images, ~2048px)

| Quality | Size Δ | SSIM2 Δ | Speed |
|---------|--------|---------|-------|
| q50 | +1.2% | **-4.14 dB** | 0.16x (6× slower) |
| q75 | +2.5% | **-1.93 dB** | 0.17x |
| q90 | +4.5% | **-1.37 dB** | 0.19x |

**Per-image variance is HIGH**: Some images +2 dB better, others -10 dB worse.
- `1e2f9d41`: +2.3 dB better at q50
- `28d24b9c`: -10.5 dB worse at q50
- `b51d5fb5`: -10.6 dB worse at q50

**Encoding failures**: FIXED (see "Arithmetic Encoder Carry Bug" section below)

### Previous Kodak Results (24 smaller images, ~512px)

**Size comparison at same quality setting:**
```
Q50: 99.2%  (smaller than libwebp)
Q75: 100.6%
Q90: 101.7%
Q95: 104.0%
```

**SSIMULACRA2 at equal BPP:**
```
BPP 0.50: -0.02 SSIM2 (essentially equivalent)
BPP 1.00: -0.19 SSIM2 (essentially equivalent)
BPP 2.00: -1.16 SSIM2 (slightly worse)
```

Average SSIMULACRA2 difference at equal BPP: -0.69

**CONCLUSION**: Performance degrades significantly on larger, more complex images.

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

## Key Finding: Edge Handling Bug (FIXED)

The RGB to YUV conversion function `convert_image_yuv` had a critical bug with non-16-aligned image dimensions:

1. **BUG**: Used `chunks_exact` which ignored remainder pixels when width/height wasn't a multiple of 2
2. **EFFECT**: Edge pixels were left as 0 (uninitialized) in the YUV buffer
3. **SYMPTOM**: Catastrophic SSIMULACRA2 scores (-100 to -700) for 1-pixel edges

**Fix** (commit e227645):
- Process odd width column separately, duplicating horizontally for chroma averaging
- Process odd height row separately, duplicating vertically for chroma averaging
- Handle corner case where both width and height are odd
- Replicate edge pixels to fill macroblock padding (width to luma_width, height to luma_height)

**Before fix**: 257x256 image edge had Blue channel = 0 for all rows
**After fix**: Blue channel correctly follows source pattern

**Edge tile test results at Q90:**
- Before: Average -9.68 vs libwebp 85.54 (diff **-95.22**)
- After: Average 84.72 vs libwebp 85.54 (diff **-0.82**)

New test added: `tests/edge_tile_ssim2_comparison.rs`

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

---

## C → Rust Reference Mapping

### libwebp Source Locations
Base path: `/home/lilith/work/libwebp/src/enc/`

| C File | Purpose | Rust Equivalent |
|--------|---------|-----------------|
| `vp8_enc.c` | VP8 encoder main | `src/vp8_encoder.rs` |
| `frame_enc.c` | Frame encoding, mode selection | `src/vp8_encoder.rs` |
| `quant_enc.c` | Quantization, RD optimization, trellis | `src/vp8_quant.rs`, `src/vp8_cost.rs` |
| `analysis_enc.c` | Macroblock analysis, segmentation | `src/vp8_encoder.rs` |
| `cost_enc.c` | Cost/rate estimation | `src/vp8_cost.rs` |
| `iterator_enc.c` | Macroblock iteration | (inlined in encoder) |
| `syntax_enc.c` | Bitstream syntax | `src/encoder.rs` |

### Key C Functions to Compare

**Mode Selection** (`quant_enc.c`):
```c
VP8MakeLuma16Preds()    // Generate all I16 predictions
VP8MakeChroma8Preds()   // Generate all chroma predictions
VP8MakeIntra4Preds()    // Generate all I4 predictions
PickBestIntra16()       // RD-based I16 mode selection
PickBestIntra4()        // RD-based I4 mode selection
ReconstructIntra16()    // Reconstruct after I16 decision
ReconstructIntra4()     // Reconstruct after I4 decision
```

**Quantization** (`quant_enc.c`):
```c
VP8SetQuantizer()       // Setup quantization params
VP8SetSegmentParams()   // Segment-based quantization
kAcTable[], kDcTable[]  // Quantization lookup tables
TrellisQuantizeBlock()  // Trellis quantization (VERIFIED MATCHING)
```

**Cost Estimation** (`cost_enc.c`):
```c
VP8CalculateLevelCosts()  // Coefficient cost tables
VP8GetCostLuma16()        // I16 mode cost
VP8GetCostLuma4()         // I4 mode cost
```

**YUV Conversion** (`dsp/yuv.h`):
```c
VP8RGBToY()   // Coefficients: 16839, 33059, 6420 (VERIFIED MATCHING)
VP8RGBToU()   // Coefficients: -9719, -19081, 28800 (VERIFIED MATCHING)
VP8RGBToV()   // Coefficients: 28800, -24116, -4684 (VERIFIED MATCHING)
VP8ClipUV()   // Rounding: YUV_HALF << 2 (VERIFIED MATCHING)
```

### c2rust Reference
Path: `/home/lilith/work/webp-porting/c2rust-reference/`

This contains the mechanically-translated Rust code from libwebp. Useful for:
- Understanding exact algorithm behavior
- Finding subtle differences in loop structures
- Checking integer overflow handling

---

## Investigation Strategy for Quality Gap

### Phase 1: Reproduce on Single Image
1. Pick worst-performing image: `28d24b9c` or `b51d5fb5`
2. Extract both encoders' intermediate data:
   - Chosen prediction modes per macroblock
   - Quantized coefficients
   - Reconstructed pixels
3. Find first divergence point

### Phase 2: Mode Selection Analysis
Add logging to compare mode distributions:
```rust
// In vp8_encoder.rs, add counters:
static mut MODE_I16_COUNTS: [u32; 4] = [0; 4];  // DC, V, H, TM
static mut MODE_I4_COUNTS: [u32; 10] = [0; 10]; // 10 I4 modes
static mut I16_VS_I4: [u32; 2] = [0; 2];        // [I16 chosen, I4 chosen]
```

Check:
- Are we choosing I4 vs I16 differently?
- Within I16, are mode preferences different?
- Is the distortion metric (SSD) calculated the same?

### Phase 3: Rate-Distortion Lambda Check
libwebp lambda calculation (`quant_enc.c:SetupMatrices`):
```c
// Check if our lambda values match at each quality level
// Current known values (from previous session):
// Q50: lambda_trellis_i4=1617, lambda_trellis_i16=1089
// Q90: lambda_trellis_i4=147, lambda_trellis_i16=100
```

### Phase 4: Cost Table Verification
Compare `VP8LevelCost` arrays:
- Are the token probability costs identical?
- Is the base cost calculation the same?

---

## SIMD Optimization Notes (from codec-design)

Current state: Pure Rust, no SIMD. 5-6× slower than libwebp.

### Quick Wins (Autovectorization)
The `wide` crate + `multiversion` can help:
```rust
use multiversion::multiversion;

#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.1", "aarch64+neon"))]
fn transform_row(data: &mut [i16]) {
    for x in data.iter_mut() {
        *x = (*x * 2217 + 2048) >> 12;  // Compiler will vectorize
    }
}
```

### Hot Paths to Optimize
1. **DCT/IDCT** (`vp8_encoder.rs`): 8×8 butterfly operations
2. **Prediction modes** (`vp8_prediction.rs`): Many pixel operations
3. **Quantization** (`vp8_quant.rs`): Integer multiply/shift
4. **Cost estimation** (`vp8_cost.rs`): Table lookups

### When Autovectorization Fails → Use `archmage`
For complex shuffles, DCT butterflies, or precise FMA ordering:
```rust
use archmage::{Desktop64, HasAvx2, arcane};

#[arcane]
fn dct_8x8_avx2(token: impl HasAvx2, block: &mut [i16; 64]) {
    // Explicit SIMD with AVX2 intrinsics
}
```

---

## Test Commands

```bash
cd /home/lilith/work/image-webp

# CLIC benchmark (32 high-res images)
cargo test --release clic_benchmark -- --nocapture --ignored

# Detailed quality sweep (5 images, all quality levels)
cargo test --release clic_detailed -- --nocapture --ignored

# Edge handling test
cargo test --release edge_tile -- --nocapture --ignored

# YUV conversion benchmark
cargo test --release yuv_benchmark -- --nocapture --ignored

# All tests
cargo test --release
```

## Test Corpus Locations

- **CLIC 2025**: `/home/lilith/work/codec-corpus/clic2025/validation/` (32 PNG, ~2048px)
- **Kodak**: Typically in `/home/lilith/work/codec-corpus/kodak/` (24 PNG, ~512px)

---

## Key Finding: Arithmetic Encoder Carry Bug (FIXED)

**Bug**: The `add_one_to_output()` function in `vp8_arithmetic_encoder.rs` had a critical
bug in carry propagation. When a chain of 0xFF bytes needed to carry (add 1 and overflow
to 0x00), the bytes were being discarded instead of being converted to 0x00.

**Symptom**: Certain images at specific quality levels would produce valid-looking files
that libwebp's decoder rejected with "NOT_ENOUGH_DATA" error. The failures were
content-dependent and appeared random (Q50 fails, Q52 works, Q55 fails, Q60 works, etc.)

**Root Cause**: When `bottom` overflowed (bit 31 set), `add_one_to_output()` would
pop 0xFF bytes without pushing 0x00 back, causing bytes to be lost from the output.
Example: `[0x10, 0xFF, 0xFF]` + carry → `[0x11]` instead of `[0x11, 0x00, 0x00]`.

**Fix** (commit 7acd4b5):
```rust
fn add_one_to_output(&mut self) {
    let mut i = self.writer.len();
    while i > 0 {
        i -= 1;
        if self.writer[i] < 255 {
            self.writer[i] += 1;
            return;
        }
        self.writer[i] = 0; // 0xFF + 1 = 0x00 with carry
    }
    // All bytes were 0xFF - prepend a 0x01
    self.writer.insert(0, 1);
}
```

**Before fix**: `d79d465a` at q50 failed to decode
**After fix**: All 32 CLIC images encode successfully at all quality levels

---

## Recent Session Fixes

1. **Edge handling** (v0.2.5): Fixed `convert_image_yuv` for non-16-aligned dimensions
2. **U/V rounding** (v0.2.6): Added `YUV_HALF << 2` rounding to match libwebp
3. **SIMD YUV** (optional): Added `fast-yuv` feature using `yuv` crate (inconsistent speedup)
4. **Arithmetic encoder carry** (commit 7acd4b5): Fixed carry propagation that lost 0xFF bytes

## Delete This File

Delete this file after loading it into a new session.
