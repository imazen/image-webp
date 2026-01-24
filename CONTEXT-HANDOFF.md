# Context Handoff: Code Refactoring

## Current Task
Refactor large files to be under 1800 lines each:
- `src/vp8_cost.rs`: 3922 lines → <1800
- `src/vp8_encoder.rs`: 3148 lines → <1800
- `src/vp8.rs`: 2016 lines → <1800

## Completed Work
1. **vp8_tables.rs created** (~500 lines) - Contains all const tables
2. **pub use re-export added** to vp8_cost.rs line 19

## vp8_cost.rs Structure Analysis

| Section | Lines | Size | Extract To |
|---------|-------|------|------------|
| Doc comment + allows | 1-20 | 20 | Keep |
| **Tables (DUPLICATES)** | 22-363 | ~340 | **REMOVE** (in vp8_tables.rs) |
| Distortion functions | 369-538 | ~170 | Keep |
| Quantization/Lambda | 544-1006 | ~460 | Keep |
| **Trellis quant** | 1017-1310 | ~290 | vp8_trellis.rs |
| Cost estimation | 1311-1445 | ~135 | Keep |
| Token/Proba system | 1448-1878 | ~430 | Keep |
| Residual cost | 1879-2078 | ~200 | Keep |
| **Analysis/Segment** | 2080-3230 | ~1150 | vp8_analysis.rs |
| **Tests** | 3230-3922 | ~690 | Move to tests/ |

### Extraction Plan for vp8_cost.rs

**Step 1: Remove duplicate tables (~340 lines saved)**
Tables to remove (already in vp8_tables.rs):
- `VP8_ENTROPY_COST` (lines 26-40)
- `VP8_LEVEL_FIXED_COSTS` (lines 69-243) - This is the BIG one
- `VP8_ENC_BANDS` (line 244)
- `VP8_LEVEL_CODES` (lines 253-276)
- `VP8_DC_TABLE` (lines 278-289)
- `VP8_AC_TABLE` (lines 291-302)
- `VP8_AC_TABLE2` (lines 304-315)
- `VP8_ZIGZAG` (line 316)
- `VP8_FREQ_SHARPENING` (lines 320-325)
- `VP8_WEIGHT_TRELLIS` (lines 327-340)
- `VP8_WEIGHT_Y` (lines 342-349)
- `VP8_WEIGHT_UV` (lines 351-363)
- `FIXED_COSTS_I16`, `FIXED_COSTS_UV` (lines 662-666)
- `NUM_BMODES`, `VP8_FIXED_COSTS_I4` (lines 669-784)
- `LEVELS_FROM_DELTA` (lines 572-606)

**Step 2: Create vp8_analysis.rs (~1150 lines)**
Extract from line 2080 to 3230:
- Constants: MAX_ALPHA, ALPHA_SCALE, MAX_COEFF_THRESH, MAX_ITERS_K_MEANS, NUM_SEGMENTS, etc.
- DctHistogram struct + impl
- forward_dct_4x4
- collect_histogram_bps
- final_alpha_value
- pred_luma16_dc, fill_block, vertical_pred, horizontal_pred, pred_luma16_tm
- make_luma16_preds, pred_chroma8_dc, pred_chroma8_tm, make_chroma8_preds
- import_block, import_line
- **AnalysisIterator struct + impl** (this is the largest piece)
- collect_histogram_with_offset
- analyze_macroblock, analyze_image
- collect_dct_histogram, compute_mb_alpha
- assign_segments_kmeans, compute_segment_quant

**Step 3: Move tests to tests/vp8_cost_tests.rs (~690 lines)**
Tests start at line 3230.

**Result:**
3922 - 340 (tables) - 1150 (analysis) - 690 (tests) = **1742 lines** ✓

### vp8_encoder.rs Structure (3148 lines)

| Section | Lines | Size | Notes |
|---------|-------|------|-------|
| Imports | 1-36 | 36 | Keep |
| Helper functions | 37-128 | ~90 | quality_to_*, sse_* |
| Structs | 130-175 | ~45 | Complexity, QuantizationIndices, MacroblockInfo |
| **Vp8Encoder struct** | 176-235 | ~60 | Main encoder struct |
| **Vp8Encoder impl** | 236-3118 | ~2880 | **TOO BIG** |
| get_coeffs0_from_block | 3119-3148 | ~30 | Helper |

**Extraction candidates:**
- Frame header writing → vp8_header.rs (~300 lines)
- Partition encoding → vp8_partitions.rs (~400 lines)
- Macroblock encoding loop → could stay but needs analysis

### vp8.rs Structure (2016 lines)

| Section | Lines | Size | Extract To |
|---------|-------|------|------------|
| Imports | 1-37 | 37 | Keep |
| **Filter functions** | 38-457 | ~420 | loop_filter.rs |
| TreeNode impl | 458-548 | ~90 | Keep or move to vp8_common |
| Structs | 549-683 | ~135 | MacroBlock, PreviousMacroBlock, Frame |
| **Vp8Decoder struct** | 684-744 | ~60 | Keep |
| **Vp8Decoder impl** | 745-1994 | ~1250 | Keep (main decoder) |
| set_chroma_border | 1995-2016 | ~20 | Keep |

**Extraction plan:**
- Filter functions (lines 38-457) → already have loop_filter.rs and loop_filter_avx2.rs
- These scalar filter functions may be fallbacks - check if they're still used
- TreeNode could move to vp8_common.rs if shared

**Result after filter extraction:**
2016 - 420 = **1596 lines** ✓

## Verification Strategy

After each extraction:
1. `cargo build --all-features` - Must compile
2. `cargo test` - All tests must pass
3. `diff` original vs new to verify no value changes
4. Check line counts: `wc -l src/vp8_*.rs`

## Files to Create

1. `src/vp8_analysis.rs` - Analysis/segmentation code from vp8_cost.rs
2. `tests/vp8_cost_tests.rs` - Tests from vp8_cost.rs
3. Possibly: `src/vp8_trellis.rs` - If more extraction needed

## Files to Modify

1. `src/vp8_cost.rs` - Remove duplicates, add re-exports
2. `src/lib.rs` - Add new modules
3. `src/vp8_encoder.rs` - TBD after analysis

## Key Dependencies Between Modules

vp8_analysis.rs will need:
- `use crate::vp8_tables::*;`
- `use crate::transform::{dct4x4, WHT_DCT4X4};`
- Various types from vp8_encoder.rs or vp8_cost.rs

vp8_cost.rs will need:
- `pub use crate::vp8_tables::*;` (already there)
- `pub use crate::vp8_analysis::*;` (after extraction)

## Safe Extraction Process

1. Create new file with extracted content
2. Add `mod` and `pub use` to lib.rs
3. Remove from original file
4. Run `cargo build` immediately
5. Fix any import errors
6. Run `cargo test`
7. Verify with `diff` that values unchanged

## Line Ranges for Extraction

### From vp8_cost.rs to vp8_analysis.rs:
```
Lines 2080-3229 (analysis section)
```

### Tables to REMOVE from vp8_cost.rs:
```
Lines 22-40: RD_DISTO_MULT, VP8_ENTROPY_COST (keep RD_DISTO_MULT, remove table)
Lines 61-243: MAX_LEVEL, MAX_VARIABLE_LEVEL, VP8_LEVEL_FIXED_COSTS
Lines 244-363: VP8_ENC_BANDS through VP8_WEIGHT_UV
Lines 565-606: LEVELS_FROM_DELTA
Lines 662-784: FIXED_COSTS_I16 through VP8_FIXED_COSTS_I4
```

### Tests to MOVE:
```
Lines 3230-3922: mod tests { ... }
```

## Notes

- NEVER use regex on table values - manually verify or use line-based extraction
- The `pub use crate::vp8_tables::*;` at line 19 means downstream code already works
- Tables are large const arrays - removing them is safe since re-export provides them
- VP8_LEVEL_FIXED_COSTS is 175 lines (lines 69-243) - the biggest single table
