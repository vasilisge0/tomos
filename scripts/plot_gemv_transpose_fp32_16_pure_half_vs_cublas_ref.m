function plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref(csv_path, tag)
%PLOT_GEMV_TRANSPOSE_FP32_16_PURE_HALF_VS_CUBLAS_REF Plot the TRANSPOSED
%   (y = alpha*A^T*x + beta*y) GENUINE 16-BIT COMPUTE GEMV sweep from
%   scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh
%   (which itself builds and runs benchmarks/kernels/cuda/matrix/mv_
%   fp32_16_pure_half_transpose_vs_cublas_ref.cu). Distinct from plot_
%   gemv_transpose_fp32_recon_vs_cublas_ref.m: that one compares the
%   "full fp32 reconstruction" tier (packed 16-bit halves decoded back
%   to fp32 BEFORE computing); this one compares kernels where the
%   multiply-accumulate itself happens in __half/__half2 arithmetic --
%   see the .cu's own header comment for the full distinction and for
%   why its cuBLAS baseline (Hgemm n=1) is a DIFFERENT cuBLAS entry
%   point than the recon plot's own cuBLAS Sgemv column, so the two
%   plots' cuBLAS numbers are NOT directly comparable to each other even
%   though the shape axes line up ("exactly the same cases", by
%   request).
%
%   Seven series, all UNPROMOTED (this is the first real-hardware
%   measurement of any of them for TRANSPOSE specifically -- see the
%   .cu's own header comment on existing correctness coverage):
%
%   cuBLAS FP16 (Hgemm n=1, TRANSPOSE via CUBLAS_OP_T) -- the real fp16
%   GEMV baseline this codebase already established elsewhere (verify_mv_
%   fp32_16_gemv.cu, bench_mv_cuda_kernels.cu, compare_precision.m).
%
%   cuBLAS BF16 (GemmEx, CUDA_R_16BF, n=1, TRANSPOSE) -- the real bf16
%   GEMV baseline, same "no native cublasXgemv for this dtype" reasoning
%   as cuBLAS FP16 above (there is no cublasBgemv either) -- same
%   CUDA_R_16BF/CUBLAS_COMPUTE_32F/CUBLAS_GEMM_DEFAULT_TENSOR_OP call
%   convention this codebase's own benchmarks_gemm_fp16.cu
%   already established (there for a real N>1 GEMM; here N=1, same
%   "GEMV via GEMM" convention cuBLAS FP16 itself uses).
%
%   TOMOS FP32/FP16 (pure) = mv_fp32_16_pure_half -- this codebase's
%   existing pure-__half-compute GEMV family's baseline dispatch (wide/
%   tiled/narrow_vec by shape, same threshold structure as every other
%   tiled/narrow_vec/wide family here). Genuinely TRANSPOSE-capable
%   already (not newly built for this file).
%
%   TOMOS FP32/FP16 (pure, half2) = mv_fp32_16_pure_half_wide_ilp_
%   shared_half2 -- EXPERIMENTAL A/B sibling reusing the EXACT SAME
%   tiled/narrow_vec paths, only the wide (split-K) kernel differs:
%   packs the 4-way unrolled accumulators into __half2 and uses __hfma2
%   (2 scalar half FMAs/instruction) instead of 4 separate __hfma calls.
%   An existing comment elsewhere in fp32_16_cuda_kernels.cu calls this
%   "the current best design (0.84 of cuBLAS)" from an earlier, different
%   (NO_TRANSPOSE and/or cublasSgemv-baselined) measurement -- this plot
%   is the first real TRANSPOSE-vs-cublasHgemm measurement of it, not
%   assumed to reproduce that number.
%
%   TOMOS FP32/FP16 (pure, atomiconly) = mv_fp32_16_pure_half_transpose_
%   atomiconly_scaled_mblock_vec4 -- NEW standalone TRANSPOSE kernel, NOT
%   a variant of the wide_ilp_shared_half2 kernel above. Built after
%   real-hardware numbers from this exact plot showed the half2 series
%   far below cuBLAS Hgemm despite its own name: its wh load is never
%   actually vectorized (two independent scalar 16-bit reads, only the
%   RESULT gets packed for the FMA -- zero memory-bandwidth benefit in a
%   memory-bandwidth-bound kernel family). This kernel reuses the fp32-
%   reconstruction tier's own proven TRANSPOSE structure (warp-per-
%   output-column/split-M/atomicAdd/shared-x-staging) with a genuinely
%   vectorized ushort4 wh load packed into __half2 for __hfma2 -- see its
%   own comment in fp32_16_cuda_kernels.cu for the full derivation,
%   including the temp-dependencies/randls2.0 cross-check that confirmed
%   the same gap exists on that side too (except for one kernel there
%   that DOES vectorize, by exploiting a different memory adjacency this
%   kernel's own TRANSPOSE structure already exploits). MEASURED on real
%   A100 hardware: a dramatic win, 3.5x-4.5x faster than both series
%   above at every size tested, reaching 0.95x-1.06x of cuBLAS Hgemm at
%   M=1024-2048/24576-65536 -- with a real dip to 0.59x-0.79x at
%   M=3072-8192, which "(atomiconly, retuned_floor)" below targets.
%
%   TOMOS FP32/FP16 (pure, atomiconly, retuned_floor) = mv_fp32_16_pure_
%   half_transpose_atomiconly_scaled_mblock_vec4_retuned_floor --
%   EXPERIMENTAL A/B sibling of "(pure, atomiconly)" above, reusing the
%   EXACT SAME kernel unchanged, only a more aggressive M_BLOCK-scaling
%   target (2048 instead of 512) with a floor (256) -- both already
%   validated for the fp32_16 reconstruction tier's own TRANSPOSE
%   family, reused directly since the M_BLOCK-scaling formula is decode-
%   independent. Targets the M=3072-8192 dip noted above.
%
%   TOMOS FP32/FP16 (fast) = mv_fp32_16_pure_half_transpose_atomiconly_
%   scaled_mblock_vec4_fast -- UNSAFE-on-arbitrary-data sibling of
%   "(atomiconly, retuned_floor)": decode swapped from fp32_16_half_only
%   (the full, 3-path decode that checks control/offset/degraded) to
%   fp32_16_fast_half (an unconditional bit-reinterpret, only correct
%   when every wh value already has control=0 -- i.e. every source
%   magnitude is inside fp16's own normal representable range). Same
%   "fast" naming/contract convention as this codebase's own
%   RECONSTRUCTION-tier sibling (fp32_16_combine_fast_normal). Tested
%   here against its own dedicated safe-range data (mag in [0.5,1.0]),
%   not the plain data every other series here uses. Everything else is
%   IDENTICAL to "(atomiconly, retuned_floor)" (same M_BLOCK-scaling
%   target=2048/floor=256), isolating decode cost alone. See its own
%   comment in fp32_16_cuda_kernels.cu.
%
%   test = mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock -- a
%   hybrid TRANSPOSE kernel for the "TOMOS FP32/FP16" family: uses the
%   expensive decode (fp32_16_half_only's own control-bit check) per
%   element, but where every OTHER pure_half sibling accepts the
%   information loss of fp32_16_half_only's own control=1 fallback
%   (flushing an out-of-fp16-range value to +-Infinity/+-0), this one
%   instead decodes an approximate fp32 value from wh ALONE -- the
%   offset/direction bits already in wh give an extended exponent
%   (effectively 8-bit-exponent-capable, vs fp16's own 5-bit limit), and
%   mant_hi (also already in wh) the same top mantissa bits a control=0
%   value would use, missing lower bits zero-padded -- accumulating that
%   element's contribution in genuine float. Uses ONLY wh, like every
%   other pure_half sibling -- no wr, no extra memory traffic. Not hand-
%   vectorized (plain scalar loop) -- first-cut, not yet a tuned
%   candidate, plotted here as "test" per its own first real-hardware
%   measurement. MEASURED on real A100 hardware: falls off past
%   M=3072ish -- diagnosed as the scalar (unvectorized) wh load paying
%   full cost as M_BLOCK grows with M, unlike every OTHER kernel in this
%   family (all ushort4). See "test (vec4)" below, and this series' own
%   comment in fp32_16_cuda_kernels.cu.
%
%   test (vec4) = mv_fp32_16_hybrid_transpose_atomiconly_scaled_mblock_
%   vec4 -- vectorized-load fix for "test"'s own measured falloff: the
%   IDENTICAL per-element branch (cheap __half2 path vs wh-only
%   extended-exponent float path), only the wh load is now a ushort4
%   fetch (same scalar-prefix/vectorized-middle/scalar-suffix structure
%   every vec4 kernel in this file uses) instead of one scalar read at a
%   time. MEASURED on real A100 hardware: closes most of "test"'s own
%   falloff (roughly 1.8x-2x faster), but still trailed "(atomiconly,
%   retuned_floor)" at 0.75x-0.93x -- first diagnosed as a redundant
%   global x load (the per-element decode helper took x as an
%   unconditionally-evaluated argument, even for the ~90% of elements
%   taking the cheap control=0 path that never used it), then simplified
%   further: BOTH paths now multiply against the same shared-memory
%   __half-truncated x copy every other pure_half kernel already stages
%   (only the WEIGHT gets the extended-precision reconstruction for the
%   rare control=1 case) -- removing the separate global x read
%   entirely, not just deferring it. MEASURED on real A100 hardware
%   after that simplification: improved further to 0.81x-0.93x of
%   "(atomiconly, retuned_floor)" -- remaining gap diagnosed as an
%   arithmetic-throughput loss: this kernel processes each of the 4
%   loaded elements individually, each __hfma2 wasting one lane on a
%   zero pad, unlike "(atomiconly, retuned_floor)" itself, which packs
%   two REAL adjacent elements into every __hfma2 (true half2 SIMD).
%   See "test (vec4, paired)" below, and this series' own comment in
%   fp32_16_cuda_kernels.cu.
%
%   test (vec4, paired) = mv_fp32_16_hybrid_transpose_atomiconly_
%   scaled_mblock_vec4_paired -- fixes "test (vec4)"'s own arithmetic-
%   throughput gap: checks each adjacent pair's control bits together --
%   if BOTH are cheap (control=0, the common case), packs them into one
%   real __hfma2 exactly like "(atomiconly, retuned_floor)" does;
%   otherwise falls back to the existing per-element helper for that
%   pair. The extended-precision decode logic itself is completely
%   unchanged -- only the common path's own arithmetic packing improves.
%   See its own comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP16 (fast) = mv_fp32_16_pure_half_transpose_atomiconly_
%   scaled_mblock_vec4_split_fast -- the SAME fast-decode swap as
%   "TOMOS FP32/FP16 (fast)" above (fp32_16_half_only -> fp32_16_
%   fast_half, tested against the same kind of dedicated safe-range
%   data), but applied to "TOMOS FP16" (the split-combine kernel)
%   instead of "(atomiconly, retuned_floor)" (the atomicAdd one) --
%   isolates whether the decode win holds independent of combine
%   strategy, since the original "TOMOS FP16" vs "TOMOS FP32/FP16
%   (fast)" comparison confounded decode AND combine together. See its
%   own comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP32/BF16 = mv_fp32_bf16_pure_bf16_transpose_atomiconly_
%   scaled_mblock_vec4 -- bf16 counterpart of "TOMOS FP32/FP16 (fast)",
%   built on its exact structure (__half/__half2 swapped for __nv_
%   bfloat16/__nv_bfloat162, fp32_16_fast_half swapped for fp32_bf16_
%   bf16_only). UNLIKE its fp16 lineage, this has NO unsafe-on-
%   arbitrary-data caveat: bfloat16 shares fp32's own 8-bit exponent, so
%   its truncation decode is ALWAYS valid for any input -- tested and
%   benchmarked against plain (not safe-range) data. See its own
%   comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP32/FP16 (pure, wmma) = mv_fp32_16_pure_half_transpose_wmma --
%   standalone Tensor Core (mma.sync via nvcuda::wmma) TRANSPOSE kernel,
%   same 16x16x16-tile/x-replication trick this codebase's own
%   NO_TRANSPOSE mv_fp32_16_pure_half_wide_wmma already uses (that one
%   falls back to a scalar kernel for TRANSPOSE, so no direct TRANSPOSE
%   counterpart existed before this). CONFIRMED, SEVERE NEGATIVE RESULT
%   on real A100 hardware: 0.043x of cuBLAS (M=K=4096) -- WORSE than even
%   mv_fp32_16_pure_half_wide_wmma's own confirmed 0.14x for NO_TRANSPOSE,
%   and 15x slower than "(atomiconly, retuned_floor)" above at the same
%   shape -- the staging-load-latency bottleneck (ordinary scalar loads
%   before wmma::load_matrix_sync can run) dominates regardless of
%   TRANSPOSE vs NO_TRANSPOSE. Tensor Cores do not appear to help this
%   memory-bound streaming GEMV shape -- see its own comment in
%   fp32_16_cuda_kernels.cu for the full derivation.
%
%   TOMOS FP32/FP16 (pure, atomiconly, retuned_floor, half I/O) =
%   mv_fp32_16_pure_half_transpose_atomiconly_scaled_mblock_vec4_half_io_
%   retuned_floor -- IDENTICAL compute to "(atomiconly, retuned_floor)"
%   above (same decode, same vectorized wh load, same M_BLOCK-scaling),
%   only x and y are __half* instead of float* -- matching cuBLAS's own
%   Hgemm(n=1) semantics exactly (A, x, AND y all __half), closing the
%   float-vs-half I/O semantics mismatch every OTHER TOMOS series here
%   has against the cuBLAS baseline. Internal accumulation still happens
%   in float (only the final atomicAdd narrows to __half) -- isolates
%   the I/O type change from any accumulation-precision change. See its
%   own comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP16 = mv_fp32_16_pure_half_transpose_warp_owner -- ENTIRELY
%   different grid/reduction strategy from every other TOMOS series
%   above: ported from randls2.0's own gemv_fp16_cm_T_warp_owner (dense_
%   kernels.cu:7533), one warp owns a whole output column for its whole
%   lifetime (no M-splitting, no atomics, no M_BLOCK-scaling). This exact
%   structural design already regressed -18.7% to -25.7% for the fp32_16
%   RECONSTRUCTION tier (mv_fp32_16_transpose_warpdot_tiled_fast_normal)
%   -- pure_half's cheaper decode and genuinely vectorized wh load make
%   this a materially different case, not a rerun. CONFIRMED on real A100
%   hardware: the worst of every method plotted here, exactly the fixed-
%   grid under-occupancy failure this comment originally flagged as a
%   risk -- see "(warp owner, split)" below, built specifically to fix
%   it. See its own comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP16 (warp owner, split) = mv_fp32_16_pure_half_transpose_
%   warp_owner_split -- split-then-reduce fix for "TOMOS FP16"'s own
%   confirmed-worst result: that kernel's grid is fixed at (1, ceil(k/8))
%   and NEVER scales with M, so too few blocks launch to fill the GPU at
%   any shape where k isn't huge -- the same under-occupancy failure the
%   fp32_16 reconstruction tier's own first atomiconly kernel had before
%   ITS OWN M_BLOCK-scaling retune fixed it. Reuses that already-
%   validated target/floor (2048/256), applied here to split M across
%   blocks; each block's partial sum goes to a unique scratch slot (no
%   atomics, still true to the warp-owner design's own premise) combined
%   by a separate reduce pass. CONFIRMED on real A100 hardware (square
%   M=K, M=1024-65536): a real, consistent win over plain "TOMOS FP16"
%   at every size (1.10x-1.78x faster), landing close to parity with
%   "(atomiconly, retuned_floor)" (0.83x-1.10x of it, even beating it at
%   M=8192/12288) -- with the gap widening again at the largest M tested
%   (0.83x-0.91x at M=16384-65536). See "(warp owner, split, f32)" below,
%   built to test whether that's the scratch buffer's own double-vs-float
%   bandwidth, and this series' own comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP16 (warp owner, split, f32) = mv_fp32_16_pure_half_transpose_
%   warp_owner_split_f32 -- IDENTICAL compute to "(warp owner, split)"
%   above, only the scratch buffer is float instead of double. CONFIRMED
%   on real A100 hardware: NO improvement over "(warp owner, split)" for
%   M > 12k -- rules out buffer bandwidth as the cause of that series'
%   own trailing-off there. See "(atomiconly, vec4, split)" below, built
%   to isolate the real cause, and this series' own comment in fp32_16_
%   cuda_kernels.cu.
%
%   TOMOS FP16 (atomiconly, vec4, split) = mv_fp32_16_pure_half_
%   transpose_atomiconly_scaled_mblock_vec4_split -- the real fix: with
%   buffer bandwidth ruled out, the actual gap was "(warp owner, split)"'s
%   own compute body itself -- ported from randls2.0's warp-owner design,
%   it loads wh only 2-wide and accumulates in float via fmaf, while
%   "(atomiconly, retuned_floor)" (this file's own proven-fastest
%   TRANSPOSE kernel) loads wh 4-wide (ushort4) and accumulates in packed
%   __half2 via __hfma2 -- twice the elements per load AND per FMA. This
%   series reuses that proven compute body VERBATIM, changing ONLY the
%   final combine step to split+reduce instead of atomicAdd -- the first
%   truly apples-to-apples isolation of "atomics vs split+reduce" in this
%   file. CONFIRMED on real A100 hardware: fully closes the large-M
%   trailing-off "(warp owner, split)" had -- tracks "(atomiconly,
%   retuned_floor)" at 0.91x-1.0x across the ENTIRE M=1024-65536 range
%   (was 0.83x-0.91x at M>=16384). Remaining gap to cuBLAS is now
%   concentrated at M=3072-8192 (0.69x-0.82x) -- the same tensor-core-
%   instruction-throughput ceiling already diagnosed via ncu profiling
%   earlier this session (cuBLAS's ampere_h16816gemm does ~6x more work
%   per instruction than __hfma2); WMMA is a confirmed dead end for this
%   shape. See "(atomiconly, vec4, split, retarget)" below, a grid-math
%   retune targeting that specific dip, and this series' own comment in
%   fp32_16_cuda_kernels.cu.
%
%   TOMOS FP16 (atomiconly, vec4, split, retarget) = mv_fp32_16_pure_
%   half_transpose_atomiconly_scaled_mblock_vec4_split_retarget -- reuses
%   the EXACT SAME __global__ kernel as "(atomiconly, vec4, split)"
%   above, only the M_BLOCK-scaling target differs (2176 vs 2048). A
%   grid-math search over the M=3072-8192 dip found target=2176 lands on
%   a near-single-variable change: M=3072/6144 and everything outside
%   3072-8192 land on the EXACT SAME grid as target=2048 -- only M=4096
%   (wave-quantization tail 37%->96% full) and M=8192 (37%->56% full)
%   actually move. CONFIRMED, NEGATIVE RESULT on real A100 hardware: no
%   improvement at M=4096/6144, a real -9.2% regression at M=8192 -- the
%   wave-quantization model this retune was based on ignored the
%   redundant x-restaging/scratch-buffer cost that grows with gridM, and
%   grid/launch retuning has now failed twice in a row for this kernel
%   (see "(warp owner, split, f32)" above for the first) -- consistent
%   with the ncu-profiled finding that the remaining gap is cuBLAS's own
%   Tensor Core instruction throughput, not occupancy. See its own
%   comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP16 (atomiconly, vec4, split, retarget, lo) = mv_fp32_16_pure_
%   half_transpose_atomiconly_scaled_mblock_vec4_split_retarget_lo --
%   tests the OPPOSITE hypothesis of "(retarget)" above: if that kernel's
%   regression came from the redundant restaging/scratch-buffer cost of
%   MORE blocks (not from occupancy), REDUCING gridM instead (target=
%   1280 vs the shared 2048) at the M=3072-24576 dip -- widened from the
%   original 3072-8192 framing per the user's own observation that the
%   gap actually extends to M=24576 -- should help. M=8192 and everything
%   above it land on the EXACT SAME grid as target=2048, so this cannot
%   repeat that regression. See its own comment in fp32_16_cuda_
%   kernels.cu.
%
%   The CSV is expected to have M/K columns (BLAS convention: A is MxK,
%   column-major -- for TRANSPOSE, x has length M and y has length K)
%   plus *_ms/*_gflops columns per method -- see benchmarks_gemv_
%   transpose_fp32_16_pure_half_vs_cublas_ref.sh's own embedded parser
%   for the exact schema. Rows are split into two groups by shape, same
%   convention as plot_gemv_transpose_fp32_recon_vs_cublas_ref.m:
%     - square:          M == K            -> x-axis is M (=K).
%     - tall-and-thin:   M ~= K (M >> K)    -> x-axis is M, K held fixed
%                                              at one of two values (512,
%                                              4096), both overlaid on the
%                                              same axes.
%   ("SKIPPED" cells, from a shape a method ran out of device memory for,
%   become NaN and simply leave a gap in that method's line.)
%
%   tag (optional) is appended (as "_<tag>") to both output basenames --
%   used by scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_
%   cublas_ref_advanced.sh to write "_avg"-suffixed plots (averaged over
%   a caller-controlled warmup/iteration count, see that script's own
%   header) alongside, without overwriting, this function's own default-
%   named plots. Omit it (or pass '') for the untagged filenames below.
%
%   Writes, under <repo>/build/benchmarks/output/plots/:
%     plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_square[_<tag>].png / .fig
%     plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_tall[_<tag>].png   / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        tag = '';
    end
    tag = char(tag);
    tag_suffix = '';
    if ~isempty(tag)
        tag_suffix = ['_' tag];
    end

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mv_fp32_16_pure_half_transpose_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref:noCsv', ...
                ['No mv_fp32_16_pure_half_transpose_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');
    K = to_double_col(T, 'K');

    % Dashed (vs the two TOMOS series' solid lines) since this is a
    % reference baseline (vendor library), not a TOMOS kernel -- same
    % convention plot_gemv_transpose_fp32_recon_vs_cublas_ref.m's own
    % cuBLAS series uses. Both TOMOS series plotted solid despite neither
    % being promoted yet -- this file's own purpose IS the first real
    % measurement that would inform a promotion decision, unlike the
    % dotted "(test)" convention used elsewhere for an already-promoted
    % series' unproven successor.
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_hgemm_gflops'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'cuBLAS BF16', 'values', to_double_col(T, 'cublas_gemmex_bf16_gflops'), 'style', '--s');
    % "TOMOS FP16 (atomiconly, vec4, split)" -- NOT promoted, isolates
    % atomics vs split+reduce using "(atomiconly, retuned_floor)"'s own
    % proven compute body verbatim. CONFIRMED on real A100 hardware: fully
    % closes the large-M trailing-off "(warp owner, split)" had, tracking
    % "(atomiconly, retuned_floor)" at 0.91x-1.0x across the ENTIRE shape
    % range -- solid style now, same "real, measured candidate" convention
    % as every promoted-adjacent series here, moved off dashed-dot. See
    % this series' own comment in fp32_16_cuda_kernels.cu.
    % "TOMOS FP16" now means mv_fp16_native_storage (mv_cuda_kernels.cu):
    % A, x, AND y all genuine native __half, no decode of any kind --
    % renamed from the FP32_16-packed-storage series below, which is now
    % "TOMOS FP16 (implicit)", once this native-storage kernel was added.
    % to_double_col_optional since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col_optional(T, 'native_storage_gflops', numel(M)), 'style', '->');
    % "TOMOS FP16 (implicit)" -- NOT promoted, isolates atomics vs
    % split+reduce using "(atomiconly, retuned_floor)"'s own proven
    % compute body verbatim. CONFIRMED on real A100 hardware: fully closes
    % the large-M trailing-off "(warp owner, split)" had, tracking
    % "(atomiconly, retuned_floor)" at 0.91x-1.0x across the ENTIRE shape
    % range -- solid style now, same "real, measured candidate" convention
    % as every promoted-adjacent series here, moved off dashed-dot. See
    % this series' own comment in fp32_16_cuda_kernels.cu. FP32_16-packed
    % storage, genuine fp16 COMPUTE via fp32_16_half_only decode -- this is
    % what plain "TOMOS FP16" meant before mv_fp16_native_storage above
    % existed. Kept plotted under a new name, not removed.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit)', 'values', to_double_col(T, 'pure_half_atomiconly_vec4_split_gflops'), 'style', '-<');
    % Fast-decode sibling of TOMOS FP16 (implicit) above -- calls
    % mv_fp32_16_pure_half_fast_transpose_atomiconly_scaled_mblock_vec4_
    % split, byte-for-byte identical except its decode calls use
    % fp32_16_half_only_fast instead of fp32_16_half_only (saturates
    % underflowed values to signed zero instead of recovering a genuine
    % fp16 subnormal -- see that function's own comment in fp32_16_pack.
    % hpp). NOT the same as the UNSAFE-on-arbitrary-data "TOMOS FP32/FP16
    % (implicit)" series below (fp32_16_fast_half, no control-bit check at
    % all -- an unrelated, pre-existing series that happens to share the
    % word "implicit" for a different reason). to_double_col_optional
    % since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast)', 'values', to_double_col_optional(T, 'pure_half_fast_atomiconly_vec4_split_gflops', numel(M)), 'style', ':>');
    % "TOMOS FP16 (implicit, fast2, vec4)" -- mv_fp32_16_pure_half_fast2_
    % transpose_atomiconly_scaled_mblock_vec4_split: BIT-EXACT-safe fast2
    % decode (fp32_16_half_only_fast2, not the lossy "fast" above; the
    % kernel actually runs the branchless no-overflow decode internally
    % now), no cp.async staging. NOT what production calls anymore (see
    % the vec2 series below) -- kept here as the register-blocking-width
    % sweep's own baseline/comparison point. to_double_col_optional since
    % older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast2, vec4)', 'values', to_double_col_optional(T, 'pure_half_fast2_atomiconly_vec4_split_gflops', numel(M)), 'style', '-s');
    % "TOMOS FP16 (implicit, fast2, cp.async)" -- mv_fp32_16_pure_half_
    % fast2_transpose_atomiconly_scaled_mblock_vec4_split_cpasync: SAME
    % decode as the vec4 series above, plus double-buffered cp.async
    % prefetch of wh. NOT what production calls -- kept here as a
    % comparison point since it measured consistently slower.
    % to_double_col_optional since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast2, cp.async)', 'values', to_double_col_optional(T, 'pure_half_fast2_atomiconly_vec4_split_cpasync_gflops', numel(M)), 'style', '-d');
    % Register-blocking width sweep of the vec4 kernel above (N=4) -- N=2
    % and N=8 siblings, same decode/data. Added after ncu profiling found
    % the vec4 kernel 86% Compute-SM-bound (ALU flagged as the over-
    % utilized pipeline) with ~50% of stall cycles waiting on L1TEX
    % scoreboard dependencies, to check whether a different per-lane
    % vectorization width overlaps that ALU work with the wh loads better
    % -- see these kernels' own comment in fp32_16_cuda_kernels.cu. Both
    % neighbors measured consistently faster than vec4 across every shape
    % tested, vec2 the best overall (~7.6% faster than vec4 at
    % M=1,000,000) -- mv_pure_half_impl's TRANSPOSE branch (fp32_16_
    % registries.cpp) now dispatches to vec2, hence "[production]" here
    % instead of on the vec4 series above. to_double_col_optional since
    % older CSVs predate these series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast2, vec2) [production]', 'values', to_double_col_optional(T, 'pure_half_fast2_vec2_split_gflops', numel(M)), 'style', ':^');
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast2, vec8)', 'values', to_double_col_optional(T, 'pure_half_fast2_vec8_split_gflops', numel(M)), 'style', ':v');
    % "mv_fp16_native_mixed" (explicit, production) -- the ACTUAL kernel
    % --explicit-duplicates dispatches to for its own TRANSPOSE matvec
    % (dense_registries.cpp's mv_fp16_native_impl). Never plotted here
    % before -- this file only ever measured mv_fp16_native_storage
    % (all-__half x/y, matching cuBLAS Hgemm's own convention), not this
    % mixed-precision (A=half, x/y=float) sibling LSQR actually calls.
    % to_double_col_optional since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'mv_fp16_native_mixed (explicit, production)', 'values', to_double_col_optional(T, 'native_mixed_gflops', numel(M)), 'style', '-o');
    % "isolated_mv_apply" pair -- the SAME two kernels as the fast2/
    % native_mixed series above, but called through mv_apply(Device,
    % TransposeType, Dense&, ...), the exact entry point lsqr.cpp itself
    % uses, instead of calling the raw kernel functions directly. Added
    % to check whether the two call paths agree -- this session's own
    % lsqr_gemv_isolated.cu harness printed TRANSPOSE numbers for these
    % same kernels that didn't cleanly reconcile with this file's own
    % raw-call numbers (explicit read LOWER through mv_apply, not higher
    % -- the opposite direction from every other benchmark-vs-harness gap
    % found this session), so this plots both call paths side by side
    % rather than comparing printf output by eye. to_double_col_optional
    % since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'isolated mv_apply (implicit)', 'values', to_double_col_optional(T, 'isolated_mv_apply_implicit_gflops', numel(M)), 'style', '--^');
    i = i + 1; series(i) = struct('label', 'isolated mv_apply (explicit)', 'values', to_double_col_optional(T, 'isolated_mv_apply_explicit_gflops', numel(M)), 'style', '--v');
    % "TOMOS FP32 / FP16" = mv_fp32_16_hybrid_transpose_atomiconly_
    % scaled_mblock_vec4_paired -- promoted from "test (vec4, paired)".
    % Unlike a plain fp32_16_half_only decode (which flushes any value
    % needing more than fp16's 5-bit exponent to +-Infinity/+-0), this
    % kernel reconstructs an approximate fp32 value from wh ALONE
    % (extended exponent + mant_hi) for that case, accumulating it in
    % genuine float -- and packs adjacent same-control pairs into one
    % real __hfma2 (matching a plain vec4 kernel's own arithmetic
    % throughput) for the common cheap case. See its own comment in
    % fp32_16_cuda_kernels.cu.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'hybrid_vec4_paired_gflops'), 'style', '-v');
    % "TOMOS FP32/FP16 (implicit)" -- UNSAFE-on-arbitrary-data sibling
    % (decode swapped to fp32_16_fast_half, tested against its own
    % dedicated safe-range data). Solid (continuous) style, same
    % convention as every other TOMOS series here -- dashed is reserved
    % for the two vendor cuBLAS baselines. Relabeled from "(fast)" (was
    % this file's own convention) so that label is free for a genuinely
    % different, SAFE decode-fast series (fp32_16_half_only_fast, see
    % below) -- this one keeps "(implicit)" for consistency with the
    % same rename made in the GEMM plots. See its own comment in
    % fp32_16_cuda_kernels.cu.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (implicit)', 'values', to_double_col(T, 'pure_half_atomiconly_vec4_fast_gflops'), 'style', '-x');
    % "TOMOS FP32/BF16" -- bf16 counterpart of "TOMOS FP32/FP16 (fast)"
    % -- safe on arbitrary data (unlike its fp16 lineage). Solid
    % (continuous) style, same convention as every other TOMOS series
    % here. See its own comment in fp32_16_cuda_kernels.cu.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'fp32_bf16_pure_bf16_gflops'), 'style', '-*');

    is_square = (M == K);
    is_tall = ~is_square;

    plot_series(M, series, is_square, 'Matrix dimensions (m = k)', ...
        'square', fullfile(out_dir, ['plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_square' tag_suffix]));

    plot_series(M, series, is_tall, 'Row count m (k held fixed -- see legend/title)', ...
        'tall', fullfile(out_dir, ['plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref_tall' tag_suffix]));
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref:missingColumn', ...
            'CSV is missing expected column "%s".', name);
    end
    col = T.(name);
    if isnumeric(col)
        v = double(col);
    else
        % Non-numeric cells (e.g. "SKIPPED") become NaN, which plot()
        % naturally renders as a gap in that method's line.
        v = str2double(string(col));
    end
end

% Same as to_double_col, but returns an all-NaN column (renders as an
% empty/absent line) instead of erroring when the CSV doesn't have this
% column at all -- for a series added after older CSVs were already
% generated (e.g. pure_half_fast_atomiconly_vec4_split_gflops, added
% alongside the fp32_16_half_only_fast decode kernel).
function v = to_double_col_optional(T, name, n_rows)
    if ~any(strcmp(name, T.Properties.VariableNames))
        v = NaN(n_rows, 1);
        return
    end
    v = to_double_col(T, name);
end

function csv_path = find_latest_csv(dir_path)
    csv_path = '';
    if ~exist(dir_path, 'dir')
        return
    end
    files = dir(fullfile(dir_path, 'mv_fp32_16_pure_half_transpose_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, group_name, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref:emptyGroup', ...
            'No rows for "%s" -- skipping.', group_name);
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % Same fixed qualitative palette as plot_gemv_transpose_fp32_recon_
    % vs_cublas_ref.m's own plot_series, sized with headroom past this
    % plot's own (smaller, 7-series) count.
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.90 0.55 0.10;  % orange
        0.10 0.75 0.80;  % cyan
        0.85 0.20 0.60;  % magenta
        0.55 0.35 0.15;  % brown
        0.50 0.50 0.50;  % gray
        0.65 0.65 0.10;  % olive
        0.30 0.30 0.90;  % indigo
        0.20 0.70 0.40;  % teal-green
    ];
    if numel(series) <= size(kPalette, 1)
        colors = kPalette(1:numel(series), :);
    else
        colors = [kPalette; lines(numel(series) - size(kPalette, 1))];
    end
    % Values are stored/read from the CSV in GFlop/s (matching the .cu
    % benchmark's own printed units) and plotted in GFlop/s directly here,
    % same convention as plot_gemv_transpose_fp32_recon_vs_cublas_ref.m --
    % GEMV's own arithmetic intensity is fixed regardless of size
    % (bandwidth-bound, not compute-bound).
    max_y = 0;
    for i = 1:numel(series)
        y = series(i).values(mask);
        y = y(order);
        if any(~isnan(y))
            max_y = max(max_y, max(y, [], 'omitnan'));
        end
        plot(x_sorted, y, series(i).style, 'LineWidth', 4.5, 'MarkerSize', 6, ...
            'Color', colors(i, :), 'DisplayName', series(i).label);
    end
    hold off;

    % Legend placed below the plot (southoutside, 3 columns) rather than
    % inside it, same convention/sizes as every other performance plot in
    % this codebase (plot_fp16_bf16_pure_vs_cutlass_ref.m,
    % plot_gemm_format_32_16_compute_32.m, plot_gemv_fp32_recon_vs_
    % cutlass_ref.m, plot_gemv_transpose_fp32_recon_vs_cublas_ref.m,
    % plot_gemv_fp32_16_pure_half_vs_cublas_ref.m).
    kAxisFontSize = 40;
    kLegendFontSize = 18;
    set(gca, 'XScale', 'log', 'XTick', unique(x_sorted), 'FontSize', kAxisFontSize, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('GFlop/s', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;
    if max_y > 0
        ylim([0, max_y * 1.1]);
    end

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
