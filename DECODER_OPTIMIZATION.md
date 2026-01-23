# Decoder Optimization Plan

## Decoder-Related Commits

These commits can be cherry-picked or extracted to separate decoder optimization work from encoder work.

### Commit Range
**First decoder commit:** `f8982ab` (feat: add decoder benchmark and document optimization status)
**Latest decoder commit:** `7d0c19e` (docs: update CLAUDE.md with current decoder performance)

### Full List (chronological order, oldest first)
```
f8982ab feat: add decoder benchmark and document optimization status
4546ee8 feat: add SIMD modules for decoder optimization (not yet integrated)
b9dcc1d fix: correct SIMD YUV conversion to match scalar formula exactly
56fc261 docs: update CLAUDE.md with decoder SIMD status
a01e138 feat: add SIMD optimizations for fancy upsampling and loop filter
9837681 feat: add 16-pixel-at-once SIMD loop filter with transpose technique
52bd9c0 perf: add lookup table for arithmetic decoder range normalization
7d0c19e docs: update CLAUDE.md with current decoder performance
```

### Files Modified (decoder-specific)
- `src/vp8_arithmetic_decoder.rs` - Lookup table optimization
- `src/yuv_simd.rs` - SIMD YUV conversion and fancy upsampling
- `src/yuv.rs` - SIMD dispatch integration
- `src/loop_filter_simd.rs` - SSE4.1 loop filter (4 edges)
- `src/loop_filter_avx2.rs` - AVX2 loop filter (16 pixels)
- `src/vp8.rs` - SIMD integration points
- `benches/profile_decode.rs` - Decoder profiling

---

## Current Performance

| Metric | Value |
|--------|-------|
| Our decoder | ~55-57 MPix/s |
| libwebp | ~140 MPix/s |
| Speed ratio | 2.5x slower |

### Profiler Hot Spots
| Function | % Time | Notes |
|----------|--------|-------|
| `read_with_tree_with_first_node` | ~24% | Arithmetic decoder core |
| Loop filter functions | ~15% | Multiple functions |
| YUV conversion | ~4% | Now SIMD-accelerated |
| IDCT | ~4% | Already SIMD |

---

## Architectural Changes Plan

### Phase 1: Batch Coefficient Reading (High Impact)

**Problem:** Current implementation creates a new `FastDecoder` instance for each bit read, with overhead from:
1. Copying state to `uncommitted_state` on entry
2. Copying state back to `self.state` on commit
3. Bounds check for EOF on every operation

**libwebp Approach:** They inline bit reading directly in coefficient parsing loops without intermediate abstractions.

**Proposed Solution:** Add a `BatchReader` API that:
1. Acquires exclusive access to decoder state for a batch of operations
2. Performs multiple reads without state copying overhead
3. Commits once at the end of the batch

```rust
// Proposed API sketch
impl ArithmeticDecoder {
    /// Start a batch of reads. Returns a BatchReader that must be
    /// committed or dropped before the decoder can be used again.
    fn start_batch(&mut self) -> BatchReader<'_>;
}

impl BatchReader<'_> {
    /// Read a bit with given probability. No intermediate commits.
    fn read_bit(&mut self, prob: u8) -> bool;

    /// Read from a tree. Returns immediately without commit.
    fn read_tree(&mut self, tree: &[TreeNode]) -> i8;

    /// Commit all reads and check for EOF.
    fn commit(self) -> Result<(), DecodingError>;
}
```

**Key Changes:**
- `src/vp8_arithmetic_decoder.rs`: Add `BatchReader` type
- `src/vp8.rs`: Refactor `read_coefficients` to use `BatchReader`

**Expected Impact:** 10-20% speedup in coefficient decoding

---

### Phase 2: Integrate Normal Loop Filter SIMD (Medium Impact)

**Problem:** Current SIMD loop filter only handles "simple" filter mode. Most images use "normal" filter (DoFilter4 for subblocks, DoFilter6 for macroblocks).

**libwebp Approach:**
- `DoFilter2`: Simple filter (what we have)
- `DoFilter4`: Subblock filter with 4-tap adjustment
- `DoFilter6`: Macroblock filter with 6-tap weighted adjustment

**Proposed Solution:**
1. Implement `normal_v_filter16` and `normal_h_filter16` in `loop_filter_avx2.rs`
2. Add SIMD versions of `macroblock_filter_*` and `subblock_filter_*`
3. Integrate into decoder loop in `vp8.rs`

**Key Changes:**
- `src/loop_filter_avx2.rs`: Add DoFilter4/DoFilter6 SIMD implementations
- `src/vp8.rs`: Add dispatch to SIMD filter paths

**Expected Impact:** 5-10% speedup (loop filter is ~15% of decode time)

---

### Phase 3: Larger Buffer Reads (Low-Medium Impact)

**Problem:** We read 32 bits (4 bytes) at a time into the bit buffer. libwebp uses 56 bits on x86_64.

**libwebp Approach:**
```c
#if defined(__x86_64__) || defined(_M_X64)
#define BITS 56  // Read 7 bytes at a time
#else
#define BITS 24  // Read 3 bytes at a time
#endif
```

**Proposed Solution:**
1. Change `State::value` from `u64` to allow 56 bits of buffered data
2. Update buffer loading to read 7 bytes instead of 4
3. Adjust bit counting accordingly

**Considerations:**
- Requires careful handling of endianness
- May need different code paths for different architectures
- Benchmark to verify improvement (may be minimal)

**Expected Impact:** 2-5% speedup

---

### Phase 4: Branchless Bit Reading (Low Impact)

**Problem:** Current bit reading uses conditional branches for the split comparison.

**libwebp Approach:** `VP8GetSigned` uses arithmetic shift to create branchless mask:
```c
const int32_t mask = (int32_t)(split - value) >> 31;  // -1 or 0
```

**Challenges:**
- libwebp stores `range-1` internally, our implementation stores `range`
- Direct port of their branchless code doesn't work with our state representation
- Would require changing fundamental state representation

**Decision:** Defer this. The complexity vs. benefit ratio is unfavorable. Modern branch predictors handle the comparison well.

---

### Phase 5: Tree Reading Optimization (Low Impact)

**Problem:** Tree traversal does bounds checks on every node access.

**Options:**
1. Use `get_unchecked()` (requires unsafe, minimal benefit)
2. Restructure tree encoding to eliminate bounds checks
3. Inline common tree patterns (like libwebp's hardcoded mode trees)

**Decision:** Defer. The bounds check is well-predicted and the real overhead is in bit reading, not tree traversal.

---

## Implementation Priority

1. **Phase 1 (Batch Reading)** - Highest impact, moderate complexity
2. **Phase 2 (Normal Filter SIMD)** - Medium impact, moderate complexity
3. **Phase 3 (Larger Buffers)** - Optional, low complexity
4. **Phase 4-5** - Defer unless profiling shows clear need

---

## Benchmarking Commands

```bash
# Quick decode benchmark
cargo run --release --features "simd,_profiling" --bin profile_decode

# Full test suite
cargo test --release --features simd

# Profiling with perf
perf record cargo run --release --features "simd,_profiling" --bin profile_decode
perf report
```

---

## Reference: libwebp Key Files

- `src/utils/bit_reader_inl_utils.h` - VP8GetBit, VP8GetSigned, VP8GetBitAlt
- `src/utils/bit_reader_utils.c` - kVP8Log2Range, kVP8NewRange tables
- `src/dec/vp8_dec.c` - GetCoeffsFast, GetCoeffsAlt, coefficient parsing
- `src/dsp/dec_sse2.c` - DoFilter2, DoFilter4, DoFilter6 SIMD
- `src/dsp/upsampling_sse2.c` - UPSAMPLE_32PIXELS macro

## Reference: webp-porting Rust Port

- `~/work/webp-porting/libwebp-rs/` - Mechanical C2Rust port
- Useful for understanding exact libwebp behavior
- Not a direct source for clean Rust implementations
