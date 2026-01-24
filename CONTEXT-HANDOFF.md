# Context Handoff: Investigate File Size Bloat

## Problem Statement

Our encoder produces larger files than libwebp at equivalent quality settings:

| Encoder | Method | Time | File Size | Ratio |
|---------|--------|------|-----------|-------|
| Ours | 4 (trellis) | 65ms | 78KB | 1.07x |
| libwebp | 6 (trellis) | 75ms | 73KB | baseline |
| Ours | 2 (no trellis) | 56ms | 87KB | 1.13x |
| libwebp | 4 (no trellis) | ~30ms | 77KB | baseline |

We're faster but produce 7-13% larger files. This suggests suboptimal rate-distortion decisions.

## Current State

All code committed. Working tree clean. Branch: main (14 commits ahead of origin).

### Recent Optimizations Completed
- SIMD quantization using `wide::i64x4` (29% speedup for methods 0-3)
- GetResidualCost SIMD (30% speedup)
- Trellis loop unrolling (minimal impact)

### Key Files
- `src/encoder/cost.rs` - RD cost estimation, trellis quantization, level costs
- `src/encoder/vp8.rs` - Mode selection (`choose_macroblock_info`), lambda calculations
- `src/encoder/analysis.rs` - Initial mode analysis, histogram collection
- `src/encoder/tables.rs` - Cost tables, probability tables

## Investigation Areas

### 1. Lambda/RD Tradeoff Parameters

Location: `src/encoder/vp8.rs` around line 300-400 (segment setup)

libwebp calculates lambda based on:
- `lambda_i16`, `lambda_i4`, `lambda_uv` for distortion weighting
- `lambda_trellis_i16`, `lambda_trellis_i4` for trellis RD
- `tlambda` scaling factors

Check if our lambda calculations match libwebp's `SetupMatrices()` and `SetupFilterStrength()` in `quant_enc.c`.

### 2. Trellis Quantization Decisions

Location: `src/encoder/cost.rs:777` - `trellis_quantize_block()`

Our trellis produces correct output but may make different level choices due to:
- Cost table differences (`LevelCosts` calculation)
- EOB cost estimation (`get_eob_cost`, `get_skip_eob_cost`)
- Distortion weights (`VP8_WEIGHT_TRELLIS`)

Compare against libwebp's `TrellisQuantizeBlock()` in `quant_enc.c:575`.

### 3. Mode Selection Bias

Location: `src/encoder/vp8.rs` - `choose_macroblock_info()`

Mode selection uses RD scores to pick I16 vs I4. If our costs are miscalibrated:
- We might pick I4 when I16 is better (I4 has higher signaling cost)
- Or vice versa

Check `get_cost_luma16()` and `get_cost_luma4()` against libwebp's equivalents.

### 4. Probability Table Updates

Location: `src/encoder/vp8.rs` - two-pass encoding, `proba_stats`

libwebp updates coefficient probabilities based on statistics from pass 1. Verify:
- Our probability updates match libwebp's algorithm
- Token statistics collection is accurate
- Updated probabilities are applied correctly in pass 2

### 5. Coefficient Cost Tables

Location: `src/encoder/cost.rs` - `LevelCosts`, `VP8_LEVEL_FIXED_COSTS`

The `LevelCosts` structure precomputes probability-dependent costs. Verify:
- `calculate()` method matches libwebp's `VP8CalculateLevelCosts()`
- Cost table indexing is correct (ctype, band, ctx)

## Suggested Investigation Approach

1. **Add debug logging** to compare decisions:
   ```rust
   // In trellis_quantize_block, log:
   // - Input coefficients
   // - Quantized output levels
   // - Final RD score
   // Compare against libwebp with TRELLIS_DEBUG enabled
   ```

2. **Isolate the difference**:
   - Test with identical lambda values (hardcode libwebp's values)
   - Test with identical probability tables
   - Test single macroblock to trace exact decision differences

3. **Compare specific metrics**:
   ```bash
   # Enable libwebp debug output
   cd ~/work/libwebp
   # Add TRELLIS_DEBUG to quant_enc.c and rebuild
   # Compare trellis decisions block-by-block
   ```

4. **Check quality metrics**:
   - If our files are larger but same visual quality, it's pure inefficiency
   - If our files are larger with better quality, lambda is too low
   - Use SSIMULACRA2 to compare at equal file sizes

## libwebp Reference Code

Key functions to compare:
- `quant_enc.c:TrellisQuantizeBlock()` - trellis quantization
- `quant_enc.c:SetupMatrices()` - lambda and quantizer setup
- `cost_enc.c:VP8CalculateLevelCosts()` - level cost tables
- `frame_enc.c:VP8EncLoop()` - main encoding loop
- `iterator_enc.c` - macroblock iteration and context

## Commands

```bash
# Build and benchmark
cargo build --release --features _profiling
./target/release/profile_encode ~/work/codec-corpus/kodak/1.png 75 10 4

# Compare with libwebp
./target/release/profile_libwebp ~/work/codec-corpus/kodak/1.png 75 10 6

# Run callgrind for detailed analysis
valgrind --tool=callgrind ./target/release/profile_encode ~/work/codec-corpus/kodak/1.png 75 1 4

# View callgrind results
callgrind_annotate --auto=yes callgrind.out.* | head -50
```

## Expected Outcome

Identify the root cause of 7-13% file size bloat and either:
1. Fix the miscalibration to match libwebp's compression ratio
2. Document intentional tradeoffs if the difference is by design
