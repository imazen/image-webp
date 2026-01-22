# Context Handoff - Decoder Optimization

## Recent Work Completed
Successfully integrated libwebp-rs style bit reader for coefficient decoding:
- **16% speedup**: 55 → 62 MPix/s
- **Gap reduced**: 2.5x → 2.15x slower than libwebp

### Key Files
- `src/vp8_bit_reader.rs` - New VP8BitReader, VP8Partitions, PartitionReader
- `src/vp8.rs` - Uses VP8Partitions for coefficient reading
- `src/vp8_arithmetic_decoder.rs` - Still used for header/mode parsing (self.b)

### Commits
- `5588e44` - perf: use libwebp-rs style bit reader for coefficient decoding
- `3a050a3` - docs: update decoder performance documentation

## Current Profile (after optimization)
| Function | % Time |
|----------|--------|
| read_coefficients | 18.85% |
| idct4x4_avx512 | 6.05% |
| should_filter_vertical | 5.56% |
| decode_frame_ | 5.20% |
| Loop filter total | ~12% |

## NEXT TASK: Branch/Cache Miss Analysis

User request: Check branch miss and cache miss counts comparing libwebp vs our decoder.

### How to do this:
```bash
# Branch misses
perf stat -e branches,branch-misses cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode

# Cache misses
perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode

# Combined
perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode
```

The profile_decode binary runs both our decoder and libwebp decoder on the same image,
so we need to separate the measurements. Options:
1. Modify profile_decode to run them separately with markers
2. Create separate binaries for each decoder
3. Use perf record with call graph to attribute misses to specific functions

### RESULTS: Branch/Cache Analysis (2026-01-22)

Per-decode stats (normalized from 100 iterations):

| Metric | Our Decoder | libwebp | Ratio | Notes |
|--------|------------|---------|-------|-------|
| Cycles | 229M | 40M | 5.7x | |
| Instructions | 449M | 99M | **4.5x** | Algorithmic overhead |
| Branches | 70M | 7.5M | 9.3x | Many more decisions |
| Branch misses | 2.66M (3.81%) | 613K (8.15%) | 4.3x | **Better rate!** |
| Cache misses | 1.63M (14.57%) | 23.5K (5.01%) | **69x** | **Main problem** |
| L1 misses | 4.65M | 256K | 18x | Cache thrashing |

**Key Findings:**
1. **Branch prediction is NOT the problem** - our miss rate is actually lower (3.81% vs 8.15%)
2. **Cache behavior is terrible** - 69x more cache misses, 18x more L1 misses
3. **We execute 4.5x more instructions** - algorithmic overhead
4. **9.3x more branches** - more conditional logic

**Root Cause Hypothesis:**
- Memory access patterns are inefficient
- Possible cache thrashing from:
  - Partition data layout (VP8Partitions stores all data + boundaries)
  - Token probability tables access pattern
  - Output buffer access pattern during coefficient writing
- libwebp likely has better cache-aware data layout

### Next Steps
- [ ] Profile cache misses by function (perf record with mem events)
- [ ] Compare data structure layouts between our code and libwebp
- [ ] Consider cache-line aligned allocations
- [ ] Consider prefetching for sequential access patterns

## Remaining Optimization Opportunities
- [ ] SIMD normal/macroblock filter (~12% opportunity)
- [ ] Use libwebp-rs bit reader for mode parsing (self.b field)
- [ ] Investigate branch/cache behavior (current task)

## Test Commands
```bash
cargo test --release --features "unsafe-simd"
cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode
```
