function plot_gemv_fp32_16_pure_half_vs_cublas_ref(csv_path, tag)
%PLOT_GEMV_FP32_16_PURE_HALF_VS_CUBLAS_REF Plot the NO_TRANSPOSE (y =
%   alpha*A*x + beta*y) GENUINE 16-BIT COMPUTE GEMV sweep from scripts/
%   benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh (which itself builds
%   and runs benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_vs_cublas_
%   ref.cu) -- the NO_TRANSPOSE mirror of plot_gemv_transpose_fp32_16_
%   pure_half_vs_cublas_ref.m, covering the same final four-series "TOMOS"
%   set that plot settled on (not its full exploratory kernel history --
%   see that file's own header for the derivation of each design choice
%   below, most of which carries over unchanged since only the memory-
%   access pattern differs between TRANSPOSE and NO_TRANSPOSE, not the
%   decode logic).
%
%   Eight series (every "test" dotted series removed by request -- each
%   candidate they covered is still built/benchmarked/verified, just no
%   longer plotted here; see fp32_16_cuda_kernels.cu and this file's own
%   header history for the full exploratory writeups):
%
%   cuBLAS FP16 (Hgemm n=1, NO_TRANSPOSE via CUBLAS_OP_N) -- the real fp16
%   GEMV baseline (no native cublasHgemv exists, so cublasHgemm with n=1
%   is the standard workaround -- same reasoning as every other GEMV-via-
%   GEMM baseline in this codebase).
%
%   cuBLAS BF16 (GemmEx, CUDA_R_16BF, n=1, NO_TRANSPOSE) -- the real bf16
%   GEMV baseline, same "no native cublasXgemv for this dtype" reasoning.
%
%   TOMOS FP16 = mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_best --
%   PROMOTED dispatcher (no merged kernel, no interaction risk): below
%   4096 rows, calls mv_fp32_16_pure_half_wide_ilp_shared_bigtile_half2_
%   atomic (atomicAdd combine -- no reduce-kernel launch, no partial
%   buffer -- combined with a 4x-larger shared-memory x tile); at/above
%   it, calls mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_wide4x
%   (same atomicAdd combine, quadrupled col_splits instead). Both callees
%   reuse mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic's own genuine
%   __half compute and safe fp32_16_half_only decode unchanged. Built
%   after Nsight Compute profiling of the PRIOR "TOMOS FP16" (plain
%   half2_atomic, still what every other series' own ratio line in
%   fp32_16_cuda_kernels.cu compares against) found it only reaching 0.59
%   waves / 46.86% achieved occupancy at M=4096 -- more col_splits (via
%   wide4x) was the profiling-motivated fix for that gap. NOTE the 4096
%   threshold was picked to match the profiling shape, not derived from a
%   measured crossover between the two callees, and neither callee has
%   been benchmarked against the other or against the prior "TOMOS FP16"
%   yet -- UNTESTED as of this promotion. See its own comment in
%   fp32_16_cuda_kernels.cu.
%
%   TOMOS FP32 / FP16 = mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_
%   branchless_best -- PROMOTED dispatcher: below 4096 rows, calls
%   mv_fp32_16_hybrid_wide_ilp_shared_bigtile_half2_atomic_branchless;
%   at/above it, calls mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_
%   branchless_wide4x. Both callees reuse mv_fp32_16_hybrid_process_
%   element_branchless unchanged: per element, computes BOTH the cheap
%   __hfma2-packed path (control=0, fits fp16's own 5 exponent bits) and
%   the extended float-reconstruction path (control=1, needs the full 8
%   exponent bits fp32 itself uses) UNCONDITIONALLY, discarding the
%   unwanted one via arithmetic masking instead of a data-dependent
%   branch. Built after Nsight Compute profiling of the PRIOR "TOMOS
%   FP32/FP16" (the branchy mv_fp32_16_hybrid_wide_ilp_shared_half2_
%   atomic, still what every other series' own ratio line in fp32_16_
%   cuda_kernels.cu compares against) found Branch Efficiency only
%   54.09% at M=4096 -- this benchmark's own 10% control=1 rate (unlike
%   every OTHER decode kernel's real-data ~0.006% rate) means ~97% of
%   warps paid for both code paths serially on nearly every element, a
%   fundamentally different divergence regime from the ~0.006%-rate
%   branch mv_fp32_16_pure_half_wide_ilp_shared_half2_branchless's own
%   confirmed NET LOSS for. CONFIRMED on real A100 hardware: this
%   dispatcher beats the branchy one by +5.7% to +18.7% at every square
%   size measured (M=1024: 162.5 -> 171.8 GFlop/s; M=16384: 798.7 ->
%   926.3; M=65536: 795.1 -> 940.3). NOTE this promotion took two
%   attempts: mv_fp32_16_hybrid_wide_ilp_shared_half2_atomic_branchless_
%   fastx_best (same branchless masking PLUS a cached float-typed x_tile
%   meant to avoid a redundant __half2float call) was briefly promoted
%   here first and CONFIRMED NEGATIVE instead (5%-26% slower than plain
%   branchless -- the extra shared-memory READ traffic from a second,
%   float-typed tile cost more than the conversion it was meant to save)
%   before being reverted and replaced with THIS dispatcher, which adds
%   no such caching -- just the confirmed branchless combine wrapped in
%   the proven bigtile/wide4x/best structure, nothing else changed. NOTE
%   the 4096 threshold is carried over unchanged from "TOMOS FP16"'s own
%   dispatcher choice, not independently re-derived for the hybrid
%   decode. See all relevant kernels' own comments in fp32_16_cuda_
%   kernels.cu for the full derivation and measurements.
%
%   TOMOS FP32 / FP16 (implicit) = mv_fp32_16_pure_half_wide_ilp_shared_half2_
%   atomic_fasthalf_best -- relabeled from "(fast)" so that label is free
%   for a genuinely different, SAFE decode-fast series (fp32_16_half_only_
%   fast) added separately -- this one keeps "(implicit)" for consistency
%   with the same rename made in the GEMM plots. NEW dispatcher, ports
%   "TOMOS FP16"'s own
%   atomicAdd-combine promotion to the UNSAFE-on-arbitrary-data fasthalf
%   decode (same relationship the OLD mv_fp32_16_pure_half_wide_ilp_
%   shared_fasthalf had to the OLD plain half2 kernel, now applied to the
%   CURRENT dispatcher instead): below 4096 rows, calls mv_fp32_16_pure_
%   half_wide_ilp_shared_bigtile_half2_atomic_fasthalf; at/above it, calls
%   mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_fasthalf_wide4x.
%   Both callees reuse mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_
%   fasthalf's own __hfma2 packing and atomicAdd combine, decode swapped
%   from fp32_16_half_only to fp32_16_fast_half (an unconditional bit-
%   reinterpret, only correct when every wh value already has control=0).
%   SAME CORRECTNESS CAVEAT AS THE OLD KERNEL (still what this series'
%   own ratio line in fp32_16_cuda_kernels.cu compares against, kept
%   unchanged as the comparison baseline): tested against its own
%   dedicated safe-range data (mag in [0.5,1.0]), not the plain data
%   "TOMOS FP16" uses. NOTE the 4096 threshold is carried over unchanged
%   from "TOMOS FP16"'s own dispatcher choice, not independently re-
%   derived for the fasthalf decode. UNTESTED: no GEMV performance
%   measurement has been run yet for this whole family. See its own
%   comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP32 / FP16 (mixed) = mv_fp32_16_pure_half_wide_ilp_shared_half2_
%   atomic_mixed -- NEW, single kernel (no bigtile/wide4x/best dispatch
%   tree of its own; always calls mv_fp32_16_pure_half_wide_ilp_shared_
%   half2_atomic_mixed_kernel for the small-rows_op regime, falling
%   through to the existing unmodified pure_half tiled/narrow_vec kernels
%   for narrow-column/large-rows_op shapes, same as every other pure_half
%   wrapper). Per element: control=0 takes the same cheap __hfma2-packed
%   __half path as plain "TOMOS FP16"; control=1 (extended OR degraded)
%   decodes EXACTLY via fp32_16_combine(wh,wr) -- unlike "TOMOS FP32 /
%   FP16" above, which is wh-only and only reconstructs an APPROXIMATE
%   value for the extended branch (the degraded branch is unrecoverable
%   without wr) -- multiplies against the original float x (not the
%   half-narrowed shared tile) in float, then narrows the product to
%   __half before accumulating, so the accumulator itself never widens.
%   Scalar, no half2/ILP packing on that path (targets correctness on the
%   rare special-encoded entries, not throughput). Needs wr as a second
%   input array, unlike the rest of the wh-only pure_half family. Same
%   mixed normal/extended/degraded-range wh/wr this benchmark's own
%   "TOMOS FP32 / FP16" series above already uses (gen_fp32_16_wh_hybrid,
%   ~10% control=1), so both series are exercised on identical data.
%   UNTESTED: no GEMV performance measurement has been run yet. See its
%   own comment in fp32_16_cuda_kernels.cu.
%
%   TOMOS FP32 / BF16 = mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_
%   best_v2 -- NEW dispatcher, REQUESTED as a separate, distinctly-named
%   kernel family for this plot slot, explicitly confirmed IDENTICAL to
%   "TOMOS BF16" below on wh (same fp32_bf16_bf16_only decode, same
%   __nv_bfloat162/__hfma2 packing, same atomicAdd combine, same bigtile/
%   wide4x/best dispatch structure, same 4096-row threshold) -- a "_v2"
%   duplicate under its own symbols so this slot gets its own
%   independently-measured benchmark leg and verify check rather than
%   silently reusing "TOMOS BF16"'s own symbols. There is genuinely no
%   decode/compute axis left to differentiate these two series on for
%   bf16 (unlike fp16, which has a real reconstruction axis for "TOMOS
%   FP32 / FP16" and a real fast/safe decode axis for "TOMOS FP32 / FP16
%   (fast)") -- bf16 shares fp32's own 8-bit exponent, so fp32_bf16_
%   bf16_only is already exact across the full range, with no fast/safe
%   split or reconstruction trick possible. UNTESTED: no GEMV performance
%   measurement has been run yet for this whole family. See its own
%   comment in fp32_16_cuda_kernels.cu. (The OLD mv_fp32_bf16_pure_bf16_
%   wide_ilp_shared kernel -- the NO_TRANSPOSE port of the TRANSPOSE
%   plot's own original "TOMOS FP32 / BF16", split+reduce combine, no
%   bigtile/wide4x -- is kept unchanged as this series' own comparison
%   baseline, same role the plain half2 kernel plays for "TOMOS FP16".)
%
%   TOMOS BF16 = mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_best -- NEW
%   dispatcher, ports "TOMOS FP16"'s own atomicAdd-combine promotion to
%   genuine bf16 compute: below kMvBf16AtomicBestCrossoverThreshold(=4096)
%   rows, calls mv_fp32_bf16_pure_bf16_wide_ilp_shared_bigtile_atomic
%   (atomicAdd combine + 4x-larger shared-memory tile); at/above it, calls
%   mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic_wide4x (same atomicAdd
%   combine, quadrupled col_splits instead). Both callees reuse
%   mv_fp32_bf16_pure_bf16_wide_ilp_shared_atomic's own genuine
%   __nv_bfloat162/__hfma2 packing and fp32_bf16_bf16_only decode
%   unchanged (same "TOMOS FP32 / BF16" arithmetic above). UNLIKE the
%   fp16 family, this does NOT add a 3-way (wide/tiled/narrow_vec)
%   dispatch -- no bf16 tiled/narrow_vec kernels exist in this codebase,
%   so it stays always-split-K, matching the plain bf16 kernel's own
%   scope. NOTE the 4096 threshold is carried over unchanged from the
%   fp16 dispatcher's own choice, not independently re-derived for bf16.
%   UNTESTED: no GEMV performance measurement has been run yet for this
%   whole family (atomic/bigtile_atomic/atomic_wide4x/atomic_best all
%   built together, matching "TOMOS FP16"'s own final design in one
%   pass rather than replaying its full exploratory history). See its
%   own comment in fp32_16_cuda_kernels.cu.
%
%   The CSV is expected to have M/K columns (BLAS convention: A is MxK,
%   column-major -- for NO_TRANSPOSE, x has length K and y has length M)
%   plus *_ms/*_gflops columns per method -- see benchmarks_gemv_fp32_16_
%   pure_half_vs_cublas_ref.sh's own embedded parser for the exact schema.
%   Rows are split into two groups by shape, same convention as the
%   TRANSPOSE plot:
%     - square:          M == K            -> x-axis is M (=K).
%     - tall-and-thin:   M ~= K (M >> K)    -> x-axis is M, K held fixed
%                                              at one of two values (512,
%                                              4096), both overlaid on the
%                                              same axes.
%   ("SKIPPED" cells, from a shape a method ran out of device memory for,
%   become NaN and simply leave a gap in that method's line.)
%
%   tag (optional) is appended (as "_<tag>") to both output basenames --
%   used by scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref_
%   advanced.sh to write "_avg"-suffixed plots (averaged over a caller-
%   controlled warmup/iteration count, see that script's own header)
%   alongside, without overwriting, this function's own default-named
%   plots. Omit it (or pass '') for the untagged filenames below.
%
%   Writes, under <repo>/build/benchmarks/output/plots/:
%     plot_gemv_fp32_16_pure_half_vs_cublas_ref_square[_<tag>].png       / .fig
%     plot_gemv_fp32_16_pure_half_vs_cublas_ref_tall[_<tag>].png         / .fig
%     plot_gemv_fp32_16_pure_half_vs_cublas_ref_square_final[_<tag>].png / .fig
%     plot_gemv_fp32_16_pure_half_vs_cublas_ref_tall_final[_<tag>].png   / .fig
%   The "_final" pair is the same square/tall data reduced to six headline
%   series (cuBLAS FP16/BF16, TOMOS FP16/BF16, TOMOS FP32 / FP16/BF16),
%   same style/fontsize as the full plots above (see plot_series's own
%   comment).

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
        default_dir = fullfile(repo_root, 'build', 'mv_fp32_16_pure_half_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_fp32_16_pure_half_vs_cublas_ref:noCsv', ...
                ['No mv_fp32_16_pure_half_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_fp32_16_pure_half_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_fp32_16_pure_half_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');
    K = to_double_col(T, 'K');

    % Dashed for the two vendor cuBLAS baselines, solid for every TOMOS
    % series -- same convention plot_gemv_transpose_fp32_16_pure_half_vs_
    % cublas_ref.m's own final series set uses.
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_hgemm_gflops'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'cuBLAS BF16', 'values', to_double_col(T, 'cublas_gemmex_bf16_gflops'), 'style', '--s');
    % "TOMOS FP16" = mv_fp16_native_storage (mv_cuda_kernels.cu): A, x, AND
    % y all genuine native __half, NO decode of any kind. This is where
    % "TOMOS FP16" belongs on principle -- plain FP16 storage never needs a
    % decode step at all, you just read the bits and compute; a decode
    % step only exists because of FP32_16's custom packed encoding (extra
    % control/offset/direction bits trading mantissa for a wider exponent
    % range than fp16 can natively hold, needed for LSQR's ill-conditioned
    % matrices). Briefly swapped to pure_half_half2_gflops (below,
    % commented out) after tracing "TOMOS FP16 used to match cuBLAS" back
    % through git history to this plot's very first commit (f3ce63d) --
    % reverted once that principle was reaffirmed: pure_half_half2 is
    % STILL FP32_16-packed/decoded, so it was never really "plain FP16"
    % either, just a cheaper decode path than "TOMOS FP16 (implicit)"
    % below. to_double_col_optional since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col_optional(T, 'native_storage_gflops', numel(M)), 'style', '->');
    % FP32_16-packed storage decoded through fp32_16_half_only -- this is
    % what "TOMOS FP16" itself meant at this plot's very first commit
    % (f3ce63d, "Adds 16-bit GEMV"), traced via git history after a report
    % that "TOMOS FP16" used to match cuBLAS closely (measured 78-90% of
    % cuBLAS in CSVs from 2026-08-31). Not genuine native fp16 storage
    % despite the historical label (see "TOMOS FP16" above for why that
    % matters) -- its performance moves with fp32_16_half_only like every
    % other packed-storage series here: it dropped to ~30-33% of cuBLAS
    % after 2026-09-15's "Updates decode functions" commit made that
    % function do real shift-based subnormal reconstruction instead of
    % flushing underflow to zero (a necessary accuracy fix, not a
    % regression). Commented out (not deleted) rather than plotted under
    % yet another label, since mv_fp32_16_pure_half_wide_ilp_shared_half2_
    % atomic_best directly below is the SAME idea (packed storage, safe
    % decode) with real col_splits/atomic tuning on top -- this plain,
    % untuned predecessor would be redundant clutter next to it. Uncomment
    % to bring it back.
    % i = i + 1; series(i) = struct('label', 'TOMOS FP16 (packed, untuned)', 'values', to_double_col(T, 'pure_half_half2_gflops'), 'style', '-.o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit)', 'values', to_double_col(T, 'half2_atomic_best_gflops'), 'style', '-<');
    % REMOVED from the plot (commented out, not deleted -- its CSV column
    % is still populated, so re-enabling is a one-line uncomment).
    % Fast-decode sibling of TOMOS FP16 (implicit) above -- calls
    % mv_fp32_16_pure_half_fast_wide_ilp_shared_half2_atomic_best, byte-
    % for-byte identical except its decode calls use fp32_16_half_only_
    % fast instead of fp32_16_half_only (saturates underflowed values to
    % signed zero instead of recovering a genuine fp16 subnormal -- see
    % that function's own comment in fp32_16_pack.hpp). Labeled against
    % "TOMOS FP16 (implicit)", not "TOMOS FP32 / FP16", since that's the
    % series it's actually a decode-sibling of. to_double_col_optional
    % since older CSVs predate this series.
    % i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast)', 'values', to_double_col_optional(T, 'half2_atomic_best_fast_gflops', numel(M)), 'style', ':>');
    % Fast2-decode sibling of TOMOS FP16 (implicit) -- calls
    % mv_fp32_16_pure_half_fast2_wide_ilp_shared_half2_atomic_best.
    % Collapses fp32_16_half_only's two sequential branches (control, then
    % direction) into ONE ("is this genuine underflow?"), but -- unlike
    % TOMOS FP16 (implicit, fast) above -- still does the real shift-based
    % subnormal reconstruction when that one check is true, so it doesn't
    % give up fast decode's accuracy loss to get (some of) its speed back.
    % See fp32_16_half_only_fast2's own comment in fp32_16_pack.hpp.
    % to_double_col_optional since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast2)', 'values', to_double_col_optional(T, 'half2_atomic_best_fast2_gflops', numel(M)), 'style', '-.h');
    % SAME kernel as the leg directly above (mv_fp32_16_pure_half_fast2_
    % wide_ilp_shared_half2_atomic_best -- the ACTUAL kernel mv_pure_
    % half_impl's NO_TRANSPOSE safe-decode branch dispatches to at this
    % shape, fp32_16_registries.cpp), only the wh DATA differs:
    % gen_fp32_16_wh_realistic's own ~18% underflow rate (matched to a
    % direct sample of the real LSQR test matrix, tall_thin_1000.mtx --
    % see that generator's own comment in mv_fp32_16_pure_half_vs_
    % cublas_ref.cu) instead of plain gen_fp32_16_wh's ~0% rate.
    % fp32_16_half_only_fast2's is_underflow branch is genuinely data-
    % dependent (fp32_16_pack.hpp) -- this series exists to show that
    % this benchmark's own long-standing synthetic uniform-[-1,1] data
    % was never representative of a real ill-conditioned LSQR matrix's
    % actual decode cost, for ANY series in this plot that reads wh (not
    % just this one). to_double_col_optional since older CSVs predate
    % this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast2, realistic ~18% underflow)', 'values', to_double_col_optional(T, 'half2_atomic_best_fast2_realistic_underflow_gflops', numel(M)), 'style', ':h');
    % "test-new": mv_fp32_16_pure_half_branchless_wide_ilp_shared_half2_
    % atomic_best -- genuinely branchless decode (fp32_16_half_only_
    % branchless, fp32_16_pack.hpp), same realistic ~18%-underflow data
    % and same shape-dispatch tree as the fast2/realistic-underflow
    % series directly above -- checks whether removing the data-
    % dependent branch entirely (compute both paths unconditionally,
    % blend via bitmask) closes the gap fast2's own single-branch decode
    % still showed on real-rate data. to_double_col_optional since older
    % CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'test-new', 'values', to_double_col_optional(T, 'test_new_gflops', numel(M)), 'style', '--p');
    % "test-new2": SAME kernel/dispatcher as "test-new" directly above
    % (mv_fp32_16_pure_half_test_new_dispatch, its own current decode --
    % see decode.md at the repo root for the full history of what that
    % is as of any given run), but fed the plain gen_fp32_16_wh buffer
    % (~0% underflow, uniform [-1,1]) instead of wh_realistic's ~18%.
    % Isolates how much of test-new's own remaining gap to native_mixed
    % is attributable to real data's underflow rate specifically, for
    % whatever the current most-optimized kernel is -- not a fixed
    % historical comparison point like the fast2-vs-fast2_realistic pair
    % above, since test-new itself keeps changing as this investigation
    % continues. to_double_col_optional since older CSVs predate this
    % series.
    i = i + 1; series(i) = struct('label', 'test-new2 (uniform ~0% underflow)', 'values', to_double_col_optional(T, 'test_new2_gflops', numel(M)), 'style', ':p');
    % "test-new3": SAME dispatcher/data as test-new2 (uniform ~0%
    % underflow), but the large-M branch routes to a kMvRowsPerThread=2
    % sibling kernel instead of test-new's own proven kMvRowsPerThread=4
    % -- isolates register-blocking (fewer, larger-work threads) vs.
    % occupancy (more, smaller-work threads) as the only variable, given
    % the base kernel's own low achieved occupancy (65% vs. --explicit-
    % duplicates' 95%, both ncu-measured -- see decode.md). to_double_col_
    % optional since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'test-new3 (rows/thread=2)', 'values', to_double_col_optional(T, 'test_new3_gflops', numel(M)), 'style', ':s');
    % "test-new4": SAME dispatcher/data as test-new2/test-new3, but the
    % large-M branch routes to a kMvRowsPerThread=8 sibling kernel --
    % extends the register-blocking-vs-occupancy trend in the opposite
    % direction from test-new3 (test-new3 measured worse than the base
    % rows/thread=4 kernel, confirming this kernel is ALU-bound rather
    % than occupancy-limited; this tests whether MORE register-blocking
    % continues to help). to_double_col_optional since older CSVs predate
    % this series.
    i = i + 1; series(i) = struct('label', 'test-new4 (rows/thread=8)', 'values', to_double_col_optional(T, 'test_new4_gflops', numel(M)), 'style', ':d');
    % "test" (cp.async) and "test2" (cp.async, warp-uniform issuer) --
    % full history preserved below, commented out (not deleted): both were
    % promoted OUT of the "test" slot into "TOMOS FP32 / FP16" itself once
    % test2 was confirmed the winner (uncomment either line to bring it
    % back as a live "test"/"test2" series for a future comparison).
    %
    % history: "test" was plain half2_atomic (checked directly, did NOT
    % reach the ~90-95%-of-cuBLAS an in-file comment claimed -- that data
    % is still in the CSV as half2_atomic_gflops, just not plotted here);
    % then half2_atomic_lowsplit (col_splits*1 instead of wide4x's own *4
    % -- also checked directly, landed in the same ~420-450 GFlop/s
    % plateau, because at M=32768 kMvTargetTotalThreads*1 and the plain
    % kernel's own target collapse to the SAME col_splits=4, so this
    % wasn't actually testing a different configuration at all -- still in
    % the CSV as half2_atomic_lowsplit_gflops).
    %
    % then mv_fp32_16_pure_half_wide_ilp_shared_half2_split(_workspace) --
    % direct Nsight-Compute-profiling-motivated candidate: at wide4x/
    % "_best"'s own col_splits=16, warp occupancy is a healthy 85%, but
    % NEITHER DRAM throughput (33%) NOR SM throughput (20%) come close to
    % saturating -- looked like the signature of a contended atomic
    % (every col_splits block atomicAdd-ing into the SAME y[i]). This
    % candidate was byte-for-byte identical to wide4x except the combine
    % step (partial[] buffer + separate reduce kernel instead of
    % atomicAdd) -- MEASURED NO FASTER (checked directly, first with a
    % per-call cudaMalloc/cudaFree confound, then again with a
    % pre-allocated workspace buffer to rule that confound out -- still no
    % faster either way), which actually DISPROVES the atomic-contention
    % hypothesis rather than confirming it (still in the CSV as
    % half2_split_gflops).
    %
    % then mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch --
    % re-profiled wide4x/"_best" itself (Warp State Statistics + Memory
    % Workload Analysis sections) instead of guessing again: 88.7% of
    % stall cycles are L1TEX scoreboard stalls (warps waiting on the wh
    % global loads) while L1 hit rate is ~0.24% (expected -- A is streamed
    % once) but Mem Busy is only ~12% and Max Bandwidth ~25% -- i.e.
    % genuinely latency-bound, not bandwidth- or atomic-bound. This
    % candidate software-pipelined the inner 4-wide unrolled loop 1 batch
    % ahead (issue NEXT batch's wh loads before consuming CURRENT batch)
    % -- MEASURED NO FASTER (checked directly), and re-profiling it found
    % the L1TEX-stall share barely moved (88.7% -> 85.2%), consistent with
    % ptxas already scheduling that single independent load about as well
    % on its own regardless of source order (still in the CSV as
    % half2_atomic_prefetch_gflops).
    %
    % then mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_prefetch2 --
    % same idea, deeper: kept 2 batches (8 elements) of wh loads in flight
    % per thread instead of 1, directly doubling the number of independent
    % outstanding memory requests instead of relying on the compiler to
    % find that overlap from a 1-deep hint. SAME col_splits formula, SAME
    % decode, SAME atomicAdd combine as wide4x otherwise -- MEASURED NO
    % FASTER either (checked directly). Register-based lookahead appears
    % exhausted as a lever: depth-2 costs 48 registers (vs 30 for depth-1,
    % 32 for the non-pipelined kernel), plausibly trading occupancy for
    % per-thread ILP in a way that cancels out, since warp-level overlap
    % (more concurrent warps to switch to) is usually the GPU's primary
    % latency-hiding mechanism, more so than per-thread lookahead (still in
    % the CSV as half2_atomic_prefetch2_gflops).
    %
    % then mv_fp32_16_pure_half_wide_ilp_shared_half2_combine_best --
    % prompted by fixing mv_fp16_native_storage's own severe small/medium-M
    % underperformance: that fix traced to atomicAdd contention (choose_
    % col_splits_* targets a FIXED total thread count regardless of row
    % count, so col_splits -- and the number of blocks atomicAdd-
    % contending on the same output row -- balloons at small M), fixed
    % there by dispatching to a reduce-kernel combine below a crossover
    % row count instead of atomicAdd throughout. This candidate tested
    % whether the SAME mechanism affects THIS safe-decode packed-storage
    % family too (reduce-kernel combine below M=8192, atomicAdd at/above
    % it, crossover reused as-is from native storage's own measurement) --
    % MEASURED NO FASTER (checked directly): this family's own col_splits
    % formula (kMvTargetTotalThreads*4 via wide4x, vs native storage's
    % choose_col_splits_fp32 *1) evidently doesn't create the same
    % contention severity at small M, so there was nothing for the reduce-
    % kernel swap to win back here (still in the CSV as half2_combine_
    % best_gflops).
    %
    % then mv_fp32_16_pure_half_wide_ilp_shared_half2_atomic_cpasync --
    % switched mechanism entirely: Ampere's cp.async stages wh into shared
    % memory via the async-copy engine, never occupying a register for the
    % pending value and never blocking the issuing warp until an explicit
    % wait -- unlike every register-mediated load above (ordinary or
    % reordered), which register renaming/scheduling limits apply to
    % regardless of source order. Reused mv_fp32_16_pure_half_wide_ilp_
    % shared_cpasync2_kernel's own proven double-buffered staging (already
    % in this file from an earlier investigation, which found its single-
    % buffered sibling measured 0.62x cuBLAS, below the 0.83x baseline --
    % but that kernel never actually overlapped anything, staging THEN
    % waiting for the whole chunk before computing), adapted to match
    % wide4x/"_best" on every other axis (same col_splits formula, __hfma2
    % accumulation, atomicAdd combine) for a clean single-variable
    % comparison -- the decode itself (fp32_16_half_only) still runs, but
    % reads from the shared-memory-staged wh bits instead of straight from
    % global, so the expensive part (waiting on the L1TEX-bound global
    % load) is off the critical path; only the cheap ALU decode work is
    % left in the hot loop. MEASURED a real, substantial win: 1.45x-1.74x
    % over "TOMOS FP16 (implicit)" at M=4096-65536 (still in the CSV as
    % half2_atomic_cpasync_gflops).
    % i = i + 1; series(i) = struct('label', 'test', 'values', to_double_col_optional(T, 'half2_atomic_cpasync_gflops', numel(M)), 'style', '-.p');
    %
    % then "test2" -- warp-uniform-issuer sibling of "test" (cp.async)
    % above, kept as its OWN separate series rather than replacing "test".
    % Re-profiling "test" (Warp State Statistics + Occupancy) found its
    % own biggest flagged opportunity (Est. Local Speedup 25.51%) is warp
    % divergence during staging, not the memory latency cp.async already
    % fixed (warp cycles/instruction dropped 59.5->11.8 there): only 24.67
    % of 32 threads active per warp on average, from "test"'s own issuer =
    % (threadIdx.x % kMvCpAsyncRowChunk == 0) alternating even/odd WITHIN
    % every 32-thread warp (16 issue, 16 idle, every warp, for the whole
    % staging loop) -- real intra-warp predication cost. This candidate
    % changed ONLY which 128 of 256 threads issue (same total transfer
    % count, same shared-memory layout the compute phase reads from):
    % issuer = (threadIdx.x < kMvTileSize/2) -- the first 4 warps, warp-
    % uniform (every thread in warps 0-3 takes the same branch, every
    % thread in warps 4-7 takes the other), eliminating intra-warp
    % divergence during staging entirely. Same col_splits/decode/combine/
    % occupancy footprint as "test" (56 vs 55 registers, same 33280 bytes
    % shared mem). MEASURED faster than "test" -- promoted below to "TOMOS
    % FP32 / FP16" (still in the CSV as half2_atomic_cpasync_warpuniform_
    % gflops). See its own comment in fp32_16_cuda_kernels.cu.
    % i = i + 1; series(i) = struct('label', 'test2', 'values', to_double_col_optional(T, 'half2_atomic_cpasync_warpuniform_gflops', numel(M)), 'style', ':d');
    %
    % "TOMOS FP32 / FP16" plugged to mv_fp32_16_pure_half_wide_ilp_shared_
    % half2_atomic_cpasync_warpuniform ("test2" above) by explicit request,
    % replacing the previous hybrid_branchless_best_gflops series (a
    % DIFFERENT kernel family, fp32_16_hybrid_*, unrelated to test/test2's
    % own fp32_16_pure_half/cp.async lineage -- flagged at the time, chosen
    % anyway). Old hybrid data is still measured every run and still in
    % the CSV as hybrid_branchless_best_gflops, just not plotted under this
    % label anymore -- to_double_col_optional so a plot against a CSV that
    % predates half2_atomic_cpasync_warpuniform_gflops still renders (NaN
    % gap) instead of erroring.
    % i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (hybrid, pre-cpasync)', 'values', to_double_col(T, 'hybrid_branchless_best_gflops'), 'style', '-v');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col_optional(T, 'half2_atomic_cpasync_warpuniform_gflops', numel(M)), 'style', '-v');
    % SWAR-decode sibling of the series directly above -- see mv_fp32_16_
    % pure_half_wide_ilp_shared_half2_atomic_cpasync_warpuniform_swar's
    % own comment in fp32_16_cuda_kernels.cu for the full derivation
    % (decodes two wh values per 32-bit register instead of four scalar
    % decodes, same cp.async staging/tiling/accumulation, same accuracy
    % otherwise -- bit-exact per fp32_16_half2_decode_swar's own
    % verification). Labeled "(SIMD decode)" here rather than "(SWAR
    % decode)" (the sibling family's own naming, e.g. "TOMOS FP16
    % (implicit, flush 256, SWAR decode)" above) -- same kernel, just a
    % more accessible label; the underlying technique is still SWAR
    % (bit-packed two-lanes-per-register tricks on scalar hardware), not
    % dedicated vector/SIMD instructions. Measured once (real hardware
    % run confirmed it works, +8-10% vs the exact decode at N>=4096);
    % to_double_col_optional since older CSVs predate this column.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (SIMD decode)', 'values', to_double_col_optional(T, 'half2_atomic_cpasync_warpuniform_swar_gflops', numel(M)), 'style', '-p');
    % SAME kernel as the leg directly above (mv_fp32_16_pure_half_wide_
    % ilp_shared_half2_atomic_cpasync_warpuniform -- SAFE decode,
    % fp32_16_half_only, TWO sequential data-dependent branches: control,
    % then direction), only the wh DATA differs: gen_fp32_16_wh_
    % realistic's own ~18% underflow rate (matched to the real LSQR test
    % matrix's own measured rate) instead of plain gen_fp32_16_wh's ~0%
    % rate. Companion to "TOMOS FP16 (implicit, fast2, realistic ~18%
    % underflow)" below -- checks whether "TOMOS FP32 / FP16"'s own good-
    % looking ~0%-underflow numbers are similarly unrepresentative of real
    % matrix data. to_double_col_optional since older CSVs predate this
    % series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (realistic ~18% underflow)', 'values', to_double_col_optional(T, 'half2_atomic_cpasync_warpuniform_realistic_underflow_gflops', numel(M)), 'style', ':v');
    % "native mixed LSQR" (was "test") -- a reused free slot for mv_fp16_native_mixed (mv_cuda_
    % kernels.cu), a DIFFERENT lineage entirely from every prior "test"
    % candidate above (those were all fp32_16_pure_half-packed/decode
    % kernels; this is a true mechanical duplicate of "TOMOS FP16"
    % (mv_fp16_native_storage)'s own dispatch tree -- same __hfma2 packing,
    % same col_splits formulas/crossover, same reduce-vs-atomic "_best"
    % split -- with x/y kept at float instead of __half, matching
    % mv_fp16_native's own existing mixed-precision contract). This is now
    % the actual kernel lsqr_precision_runner's own --explicit-duplicates
    % dispatches to (the (CUDA, FP16, FP32, FP32, compute=FP16) slot in
    % dense_registries.cpp) -- plotted here purely as a sanity check on its
    % real-hardware performance now that it is live in that production
    % path, not because it belongs in the FP32_16-packed "test" lineage
    % documented above. to_double_col_optional since older CSVs predate
    % this series.
    % Relabeled from the reused-slot name "test" now that this is the
    % kernel LSQR's --explicit-duplicates actually runs.
    i = i + 1; series(i) = struct('label', 'native mixed LSQR', 'values', to_double_col_optional(T, 'native_mixed_gflops', numel(M)), 'style', '-.p');
    % "TOMOS FP16 (implicit, native accumulation)" -- mv_fp32_16_pure_half_
    % nativeaccum: packed wh-only storage with the SAME branchless no-
    % overflow decode as "TOMOS FP16 (implicit)", but mv_fp16_native_mixed's
    % accumulation ("native mixed LSQR" above): __half chains capped at 256
    % columns, flushed into a double per tile (three-tier dispatch, same
    % thresholds). Added to see whether the accuracy that accumulation gives
    % LSQR can be had from the packed kernels without losing their
    % throughput. to_double_col_optional since older CSVs predate it.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, native accumulation)', 'values', to_double_col_optional(T, 'nativeaccum_gflops', numel(M)), 'style', '-.s');
    % LSQR's actual implicit NO_TRANSPOSE dispatcher (mv_fp32_16_pure_half_
    % no_transpose_fast2) and its periodic-flush drop-ins: identical routing,
    % but at M >= 131072 (LSQR's M=1e6) the narrow_vec kernel adds its four
    % __half accumulators into float every 64 / 256 columns, bounding every
    % fp16 chain like native mixed LSQR does (which caps them at 256) -- the
    % accuracy fix without nativeaccum's full-tree throughput loss. Below
    % 131072 rows all three run the same kernel, so they only separate in
    % the tall shapes. to_double_col_optional since older CSVs predate them.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, LSQR)', 'values', to_double_col_optional(T, 'noT_fast2_gflops', numel(M)), 'style', '-.>');
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, flush 64)', 'values', to_double_col_optional(T, 'noT_flush64_gflops', numel(M)), 'style', '-.<');
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, flush 256)', 'values', to_double_col_optional(T, 'noT_flush256_gflops', numel(M)), 'style', '-.^');
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, flush 256, SWAR decode)', 'values', to_double_col_optional(T, 'noT_flush256_swar_gflops', numel(M)), 'style', '-.v');
    % Lossy fast-decode counterpart of the SWAR flush-256 kernel above (underflow
    % band flushed to 0, same trade as "TOMOS FP16 (implicit, fast)"). Not exact
    % on matrices with many underflowing entries -- for throughput comparison only.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, fast, flush 256, SWAR decode)', 'values', to_double_col_optional(T, 'noT_flush256_fast_swar_gflops', numel(M)), 'style', ':d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (implicit)', 'values', to_double_col(T, 'fasthalf_atomic_best_gflops'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (mixed)', 'values', to_double_col(T, 'half2_atomic_mixed_gflops'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'bf16_atomic_best_v2_gflops'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS BF16', 'values', to_double_col(T, 'bf16_atomic_best_gflops'), 'style', '-^');
    % NOT a tuned kernel -- mv_strided<__half,float,float> (mv_cuda_
    % kernels.cu's gemv_generic_kernel), the generic mixed-precision
    % fallback this codebase's Dense::lookup_mv dispatch registry
    % silently uses whenever no dedicated kernel is registered for a
    % requested (A_type, x_type, y_type) triple -- e.g. this is exactly
    % what lsqr_precision_runner's own --mv-type fp16 (no --value-type
    % fp16, no --implicit-duplicates) actually runs. A stored plain
    % native __half, x/y float, but every multiply-accumulate upcasts to
    % double first -- still measured by mv_fp32_16_pure_half_vs_cublas_
    % ref.cu (double_acc_gflops column) and left in the CSV, just
    % commented out of the plot itself by request -- uncomment to bring
    % it back.
    % i = i + 1; series(i) = struct('label', 'FP16 (double acc)', 'values', to_double_col_optional(T, 'double_acc_gflops', numel(M)), 'style', '-.o');

    is_square = (M == K);
    is_tall = ~is_square;

    plot_series(M, series, is_square, 'Matrix dimensions (m = k)', ...
        'square', fullfile(out_dir, ['plot_gemv_fp32_16_pure_half_vs_cublas_ref_square' tag_suffix]));

    plot_series(M, series, is_tall, 'Row count m (k held fixed -- see legend/title)', ...
        'tall', fullfile(out_dir, ['plot_gemv_fp32_16_pure_half_vs_cublas_ref_tall' tag_suffix]));

    % "_final" companion plots: the same square/tall data, reduced to just
    % the six headline series (the two cuBLAS baselines and the best
    % kernel from each of the four TOMOS families), pulled straight out of
    % the `series` struct array above so their own values/style strings
    % (and therefore each line's dash pattern/marker) are byte-identical
    % to the full plot's -- plot_series itself is reused unchanged, so
    % fontsize/legend placement/line width match too, the only difference
    % is the smaller series count (which reflows the same fixed qualitative
    % palette from its own start, same as any other plot_series call).
    final_labels = {'cuBLAS FP16', 'cuBLAS BF16', 'TOMOS FP16', 'TOMOS BF16', ...
        'TOMOS FP32 / FP16', 'TOMOS FP32 / FP16 (SIMD decode)', ...
        'TOMOS FP16 (implicit, fast, flush 256, SWAR decode)', 'TOMOS FP32 / BF16'};
    all_labels = {series.label};
    final_idx = zeros(1, numel(final_labels));
    for li = 1:numel(final_labels)
        found = find(strcmp(all_labels, final_labels{li}), 1);
        if isempty(found)
            error('plot_gemv_fp32_16_pure_half_vs_cublas_ref:missingFinalSeries', ...
                'Series "%s" not found -- expected among this function''s own series definitions above.', ...
                final_labels{li});
        end
        final_idx(li) = found;
    end
    series_final = series(final_idx);
    % Renamed to a short, "_final"-plot-only label -- SAME values/style
    % (dash pattern/marker) as its full-plot definition above, only the
    % legend text differs here. This is the LOSSY fast decode (underflow
    % saturates to signed zero, see that series' own comment above), not
    % the same kernel family as "TOMOS FP32 / FP16" just above it despite
    % the matching short name -- kept this short, and placed directly
    % after "TOMOS FP32 / FP16" in final_labels above (so it lands in the
    % same legend column, one row below it, under legend()'s column-major
    % fill), at the user's own request.
    fast_swar_idx = strcmp({series_final.label}, 'TOMOS FP16 (implicit, fast, flush 256, SWAR decode)');
    series_final(fast_swar_idx).label = 'TOMOS FP32 / FP16 (fast)';

    plot_series(M, series_final, is_square, 'Matrix dimensions (m = k)', ...
        'square_final', fullfile(out_dir, ['plot_gemv_fp32_16_pure_half_vs_cublas_ref_square_final' tag_suffix]));

    plot_series(M, series_final, is_tall, 'Row count m (k held fixed -- see legend/title)', ...
        'tall_final', fullfile(out_dir, ['plot_gemv_fp32_16_pure_half_vs_cublas_ref_tall_final' tag_suffix]));
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_fp32_16_pure_half_vs_cublas_ref:missingColumn', ...
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
% generated (e.g. half2_atomic_best_fast_gflops, added alongside the
% fp32_16_half_only_fast decode kernel).
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
    files = dir(fullfile(dir_path, 'mv_fp32_16_pure_half_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, group_name, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_fp32_16_pure_half_vs_cublas_ref:emptyGroup', ...
            'No rows for "%s" -- skipping.', group_name);
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % Same fixed qualitative palette as plot_gemv_transpose_fp32_16_pure_
    % half_vs_cublas_ref.m's own plot_series.
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
    % same convention as the TRANSPOSE plot -- GEMV's own arithmetic
    % intensity is fixed regardless of size (bandwidth-bound, not
    % compute-bound).
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
    % cutlass_ref.m, plot_gemv_transpose_fp32_recon_vs_cublas_ref.m).
    kAxisFontSize = 40;
    % Full plots carry 20+ series (this file's own no-tag square/tall
    % pair), whose 3-column legend at the old fixed 18pt was wide enough
    % to get clipped by exportgraphics's own tight bounding box (visually
    % confirmed: leftmost column's labels were cut off mid-word in the
    % exported PNG). The "_final" pair's 6-series legend has plenty of
    % room at 18pt, so this only shrinks the font once there are enough
    % series for it to matter.
    if numel(series) > 10
        kLegendFontSize = 11;
    else
        kLegendFontSize = 18;
    end
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
