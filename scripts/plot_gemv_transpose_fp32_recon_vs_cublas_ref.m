function plot_gemv_transpose_fp32_recon_vs_cublas_ref(csv_path, tag)
%PLOT_GEMV_TRANSPOSE_FP32_RECON_VS_CUBLAS_REF Plot the TRANSPOSED (y =
%   alpha*A^T*x + beta*y) fp32-compute GEMV sweep from scripts/
%   benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh (which itself
%   builds and runs benchmarks/kernels/cuda/matrix/mv_fp32_transpose_
%   recon_vs_cublas_ref.cu). Same SIX promoted series as plot_gemv_fp32_
%   recon_vs_cutlass_ref.m compares for NO_TRANSPOSE (minus CUTLASS, see
%   below):
%
%   cuBLAS (Sgemv, CUBLAS_OP_T), TOMOS FP32 (native, mv_fp32_transpose_
%   atomiconly_scaled_mblock_vec4_stagevec), TOMOS FP32/FP16 (mv_fp32_16_
%   transpose_atomiconly_scaled_mblock_vec4_wrapborrow_dispatch), TOMOS
%   FP32/FP16 (fast) (mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_
%   normal_wrapborrow_vec4_dispatch),
%   TOMOS FP32/BF16 (mv_fp32_bf16_transpose_atomiconly_scaled_mblock_
%   vec4_rounded_dispatch), TOMOS FP32/BF16 (truncated) (mv_fp32_bf16_
%   transpose_atomiconly_scaled_mblock_vec4).
%
%   An experimental "(truncated, test)" large-M dispatcher series
%   (mv_fp32_bf16_transpose_atomiconly_scaled_mblock_vec4_dispatch) was
%   plotted here previously -- removed from this plot (still compiled,
%   checked, and printed by the benchmark .cu/.sh as fp32_bf16_truncated_
%   test_gflops, just no longer a series here); see Experiments.md's own
%   Row 2 section for what it is and why it isn't plotted.
%
%   TOMOS FP32's own two predecessor kernels -- mv_fp32_transpose_
%   atomiconly_scaled_mblock_vec4 ("(vec4)": float4-vectorized loads of A
%   and the shared x tile, but scalar x-staging) and mv_fp32_tiled
%   ("(unvectorized)": scalar everywhere, this codebase's original
%   TRANSPOSE fast kernel) -- are still both compiled, correctness-
%   checked, and printed by the benchmark .cu/.sh (the CSV still carries
%   their native_vec4_gflops/native_unvectorized_gflops columns), just
%   not plotted here; see the .sh's own METHOD_MARKERS comment and each
%   kernel's own comment in mv_cuda_kernels.cu for the full promotion
%   history and real-hardware numbers behind each step. TOMOS FP32/FP16's
%   own predecessor, mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock
%   ("(scalar)": no vectorization at all), is likewise still compiled,
%   checked, and printed (fp32_16_scalar_gflops in the CSV), just not
%   plotted -- same convention. TOMOS FP32/FP16 now has FOUR demoted
%   predecessors -- "(scalar)" above, mv_fp32_16_transpose_atomiconly_
%   scaled_mblock_vec4 ("(vec4)": float4-vectorized loads, but the
%   original, less aggressive M_BLOCK-scaling target), mv_fp32_16_
%   transpose_atomiconly_scaled_mblock_vec4_retuned_floor ("(retuned_
%   floor)": same M_BLOCK-scaling as "(wrapborrow)" below, but the OLDER
%   branchless decode formula), and mv_fp32_16_transpose_atomiconly_
%   scaled_mblock_vec4_wrapborrow ("(wrapborrow)": the wraparound-
%   subtraction decode reformulation, but the ORIGINAL, less aggressive
%   M_BLOCK-scaling target -- the single best TUNED kernel in this
%   family, superseded here only by a ROUTER over it, not a better
%   kernel, see below) -- all four still compiled, checked, and printed
%   (fp32_16_scalar_gflops, fp32_16_vec4_gflops, fp32_16_retuned_floor_
%   gflops, and fp32_16_wrapborrow_gflops in the CSV), none plotted.
%   TOMOS FP32/FP16 (fast) likewise now has FIVE demoted predecessors -- "(fast,
%   scalar)" (mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_
%   normal, no vectorization at all), mv_fp32_16_transpose_atomiconly_
%   scaled_mblock_fast_normal_vec4 ("(fast, vec4)": float4/ushort4-
%   vectorized loads, but the original, less conservative M_BLOCK-
%   scaling target), mv_fp32_16_transpose_atomiconly_scaled_mblock_
%   fast_normal_vec4_retuned ("(fast, retuned)": same kernel as "(fast,
%   vec4)" with the conservative M_BLOCK-scaling target, but the OLDER
%   is_carry-branch decode formula), mv_fp32_16_transpose_
%   atomiconly_scaled_mblock_fast_normal_wrapborrow_vec4 ("(fast,
%   wrapborrow)": the wraparound-subtraction decode reformulation, but
%   the ORIGINAL, less aggressive M_BLOCK-scaling target), and
%   mv_fp32_16_transpose_atomiconly_scaled_mblock_fast_normal_
%   wrapborrow_vec4_retarget ("(fast, retarget)": same kernel as "(fast,
%   wrapborrow)" with a more aggressive M_BLOCK-scaling target -- the
%   single best TUNED kernel in this family, superseded here only by a
%   ROUTER over it, not a better kernel, see below) -- all five
%   still compiled, checked, and printed (fp32_16_fast_scalar_gflops,
%   fp32_16_fast_vec4_gflops, fp32_16_fast_vec4_retuned_gflops,
%   fp32_16_fast_wrapborrow_vec4_gflops, and fp32_16_fast_retarget_gflops
%   in the CSV), none plotted.
%
%   Every TOMOS series here is sourced from a dedicated TRANSPOSE fast
%   path (trans==TRANSPOSE && ld_a==size[0]): before this codebase's own
%   TRANSPOSE work (this file's own history), none of these row-
%   vectorized "fast"/"ready"/"atomiconly" kernels had any TRANSPOSE-
%   specialized code path at all -- TRANSPOSE calls fell back to a
%   generic double-accumulator dispatch whose own vectorized-load fast
%   path was ALSO NO_TRANSPOSE-only, so there was no transpose-
%   specialized GEMV kernel of any kind in this codebase before then.
%   Every one of these kernels is warp-per-output-column with the
%   reduction dimension (M) split into chunks across blockIdx.x and
%   atomicAdd combining partial sums, x staged into shared memory once
%   per block and reused by every warp in that block -- ported from
%   randls2.0's own sgemv_cm_transpose_splitM_atomic_xshared (src/
%   kernels/cuda/matrix/dense_kernels.cu, that sibling project -- same
%   author, no separate license/attribution needed). See mv_fp32_
%   transpose_atomiconly_scaled_mblock_kernel's own comment (src/kernels/
%   cuda/matrix/mv_cuda_kernels.cu) for the full algorithmic derivation
%   shared by all these TRANSPOSE kernels.
%
%   TOMOS FP32 = mv_fp32_transpose_atomiconly_scaled_mblock_vec4_
%   stagevec: float4-vectorized loads of A, the shared x tile, AND the
%   x-staging loop itself (global x -> shared sx) -- runtime 16-byte
%   alignment check before committing to each vectorized path, scalar
%   fallback otherwise (always correct, vectorization is a pure
%   throughput bonus when it's hit) -- see its own comment in mv_cuda_
%   kernels.cu. PROMOTED through two prior rounds, both on real A100
%   hardware: first over "(unvectorized)" (mv_fp32_tiled, this codebase's
%   original TRANSPOSE fast kernel, scalar everywhere) via vectorizing
%   the main loop alone (+3%-+6% from M=K=3072 upward, one dip at
%   M=2048), then over that same "(vec4)" intermediate by ALSO
%   vectorizing the staging loop, which fixed that dip and added +8%-
%   +12% at the sizes vec4 itself was weakest -- reaching parity with or
%   beating cuBLAS from M=12288 upward (0.999x-1.081x).
%
%   TOMOS FP32/FP16 = mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
%   wrapborrow_dispatch: NOT a new tuned __global__ kernel -- a thin
%   host-side router over two already-independently-tested siblings,
%   calling "(wrapborrow)" (the demoted predecessor described below)
%   directly for M<8192, and its own retarget2 sibling (target=8192,
%   same shared floor=256, not its own visible CSV/plot series) for
%   M>=8192 -- a direct, high-confidence port of "(fast)"'s own dispatch
%   (below), since the M_BLOCK-scaling grid math is decode-independent --
%   see both kernels' own comments in fp32_16_cuda_kernels.cu. PROMOTED:
%   on real A100 hardware, tracks "(wrapborrow)" almost exactly below
%   M=8192 (0.996x-1.006x, negligible routing overhead) and its own
%   retarget2 almost exactly at/above it (1.006x-1.077x), capturing
%   retarget2's own real M=8192-12288 win (+7.8% at M=12288) while
%   completely avoiding its severe M=3072-6144 regression (down to
%   -20.4% for retarget2 alone, 0.999x-1.001x for this dispatcher at
%   those same sizes) -- no regression found anywhere tested (M=1024-
%   12288 at the time of this promotion). CAVEAT: not yet confirmed at
%   M>=16384, and (mirroring "(fast)"'s own dispatcher) this M-only
%   threshold is unvalidated for tall-and-thin/large-K shapes.
%
%   "(wrapborrow)" = mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_
%   wrapborrow: reuses the EXACT SAME __global__ kernel STRUCTURE and
%   M_BLOCK-scaling as "(retuned_floor)" (below) unchanged, only decoding
%   via fp32_16_combine_branchless_wrapborrow instead of fp32_16_combine_
%   branchless -- the SAME wraparound-subtraction technique the UNSAFE
%   fast_normal_wrapborrow decode already proved out (see "(fast)"
%   below), applied here to the SAFE decode's own identical is_carry/
%   round_flag/mant_hi pattern -- see its own comment in fp32_16_cuda_
%   kernels.cu. Confirmed via cuobjdump: eliminates all 26 SEL
%   instructions and reduces ISETP 46->33 (864->832 total instructions),
%   though register usage goes the WRONG way (REG:32 vs REG:31, one MORE,
%   unlike fast_normal_wrapborrow's own reduction). Its own promotion: on
%   real A100 hardware the fewer-instructions side won decisively
%   regardless -- a remarkably clean, MONOTONIC win at every square size
%   tested (M=1024-65536), from +4.0% at the smallest to +12.1% at the
%   largest, no regressions. DEMOTED now that "TOMOS FP32/FP16" itself
%   (above) sources that slot instead -- kept as its own visible series
%   (fp32_16_wrapborrow_gflops in the CSV), same "keep both, promote the
%   winner" convention.
%
%   "(retuned_floor)" = mv_fp32_16_transpose_atomiconly_scaled_mblock_
%   vec4_retuned_floor: reuses the EXACT SAME __global__ kernel as
%   "(vec4)" unchanged (float4-vectorized x-staging/sx reads, ushort4-
%   vectorized wh/wr loads -- decode itself, fp32_16_combine_branchless,
%   stays scalar), just a more aggressive M_BLOCK-scaling grid-
%   oversubscription target (2048 vs 512) with a floor (256) capping how
%   far M_BLOCK can shrink -- see its own comment in fp32_16_cuda_
%   kernels.cu. Its own promotion, through two rounds over "(scalar)":
%   first "(vec4)" itself (a real caveat -- essentially no benefit below
%   M=K=8192, a stable +22% from M=16384 up), then this retuning over
%   "(vec4)" -- CLEAN this time, a real win at every size tested on real
%   A100 hardware, never worse by more than measurement noise (-0.3% at
%   M=1024): +5.5%-+14.2% at M=3072-12288, ~parity elsewhere. An earlier
%   attempt at this same retuning idea WITHOUT the floor regressed
%   severely at small M (down to 0.662x vs "(vec4)" at M=1024, from
%   excessive atomicAdd contention once M_BLOCK shrank too far) -- kept
%   as mv_fp32_16_transpose_atomiconly_scaled_mblock_vec4_retuned, a
%   documented negative result, not plotted or promoted. DEMOTED now that
%   "(wrapborrow)" above sources that slot instead -- kept as
%   its own visible series (fp32_16_retuned_floor_gflops in the CSV), not
%   plotted, same "keep both, promote the winner" convention.
%
%   TOMOS FP32/FP16 (fast) = mv_fp32_16_transpose_atomiconly_scaled_
%   mblock_fast_normal_wrapborrow_vec4_dispatch: NOT a new tuned
%   __global__ kernel -- a thin host-side router over two already-
%   independently-validated siblings, calling "(fast, retarget)" (the
%   demoted predecessor described below) directly for M<8192, and its
%   own retarget2 sibling (target=8192, same shared floor=256, not its
%   own visible CSV/plot series) for M>=8192 -- see both kernels' own
%   comments in fp32_16_cuda_kernels.cu. PROMOTED: validated on real
%   A100 hardware across the FULL square sweep (M=1024-65536), not just
%   the two shapes checked when the M=8192 threshold was first chosen --
%   tracks "(fast, retarget)" almost exactly below the threshold
%   (0.998x-1.003x) and its retarget2 sibling almost exactly at and
%   above it (matching to 3 decimal places at M=24576/32768/49152/65536,
%   e.g. 1.820ms vs 1.819ms at M=24576), with dispatch overhead itself
%   never exceeding ~0.3% anywhere -- at least as good as "(fast,
%   retarget)" everywhere, and a real win from M=8192 up (+2.8% to
%   +10.7% vs "(fast, retarget)" on this run) by riding retarget2's own
%   genuinely smaller M_BLOCK there. CAVEAT: the M=8192 threshold was
%   only validated on SQUARE shapes -- the tall-and-thin sweep below
%   only covers M>=16384 (already past the threshold either way), so a
%   small-M/large-K combination (where gridK could shift where retarget2
%   starts winning) remains untested. retarget2 itself gets no CSV
%   column of its own here -- it exists purely as dispatch's own
%   internal building block, same as retarget2_floor (a documented
%   negative result: fixes retarget2's own M=3072-6144 regression, but
%   introduces a worse one at M=1024-1536 from its higher floor exceeding
%   m itself there, so it is not, and will not be, folded into this
%   dispatcher).
%
%   "(fast, retarget)" = mv_fp32_16_transpose_atomiconly_scaled_mblock_
%   fast_normal_wrapborrow_vec4_retarget: reuses the EXACT SAME
%   __global__ kernel as "(fast, wrapborrow)" unchanged (decoding via
%   fp32_16_combine_fast_normal_wrapborrow -- a bit-identical
%   reformulation of fp32_16_combine_fast_normal's own is_carry ternary/
%   SEL logic, exhaustively verified over all 1024 possible (mant_hi,
%   round_flag) combinations, see its own comment in fp32_16_pack.hpp),
%   just a more aggressive M_BLOCK-scaling grid-oversubscription target
%   (2048 vs 1024) with the same shared floor (256) -- see its own
%   comment in fp32_16_cuda_kernels.cu. Re-derived specifically for
%   wrapborrow's OWN measured register footprint (REG:32, confirmed via
%   cuobjdump on the compiled kernel body -- eliminating the is_carry
%   branch's 38 SEL/19 ISETP.EQ.U32.AND instructions apparently also
%   freed up register pressure elsewhere, dropping from the pre-
%   wrapborrow sibling's own REG:39), which now matches the SAFE decode
%   kernel's own REG:31/2048-target residency profile (8 blocks/SM
%   either way, 864-block full-residency capacity) -- not a blind copy
%   of that number, the same math this file's own convention already
%   applies whenever a kernel's measured register count changes. Its own
%   promotion, through two rounds over "(fast, retuned)": first "(fast,
%   wrapborrow)" itself (a real, decode-formula-only win, +1.4% at
%   M=1024 up to +4.1% at M=2048), then this retuning over "(fast,
%   wrapborrow)" -- a clean win at every size where the grid config
%   actually differs between the two targets (re-verified by recomputing
%   M_BLOCK for both, not just trusting the measured ratio): +7.5% to
%   +14.3% across M=3072-8192 square, and +15.2% at the one tall-and-
%   thin shape where target=2048 actually shrinks M_BLOCK below the
%   8192 ceiling (K=4096, M=16384). At M=1024-1536, M_BLOCK is already
%   floor-bound (256) under BOTH targets, so ~1.00x there is the
%   CORRECT expected outcome, not noise. CAVEAT, verified the same way:
%   at the two LARGEST tall-and-thin shapes tested (K=4096, M=32768 and
%   M=65536), M_BLOCK/grid are BYTE-FOR-BYTE IDENTICAL between "(fast,
%   wrapborrow)" and this kernel (neither target's own scaled_block
%   drops below the ceiling there) -- the +15.8%/+5.6% originally
%   measured at those two shapes is therefore almost certainly
%   measurement noise, not a real effect, since there is no code-level
%   mechanism by which two byte-identical launches would differ.
%   Whether an even higher target could push M_BLOCK below the ceiling
%   at those two shapes too, and capture a genuine win there, is
%   untested. See the .sh's own METHOD_MARKERS comment for the full
%   promotion history (including "(fast, vec4)"'s own earlier +6.2%-
%   +21.8% win over "(fast, scalar)"). Reaches 0.87x-0.96x of cuBLAS
%   from M=3072 upward (absolute measurements, not comparisons -- these
%   hold regardless of the noise caveat above), the best-measured
%   fp32_16 TRANSPOSE kernel in this file at every size M>=3072, now
%   matched or exceeded (never worse) by "(fast)" itself above.
%
%   TOMOS FP32/BF16 (truncated) = mv_fp32_bf16_transpose_atomiconly_
%   scaled_mblock_vec4: byte-for-byte the SAME warp-per-output-column/
%   split-M/atomicAdd/M_BLOCK-scaling structure as SAFE fp32_16's own
%   "(vec4)" TRANSPOSE kernel, ported directly per this codebase's own
%   "base the design on FP32/FP16 (fast)" convention -- float4-vectorized
%   x-staging AND ushort4-vectorized wh/wr loads, decode via
%   fp32_bf16_combine UNCHANGED (a single shift+OR, no branch -- unlike
%   fp16's decode, there is nothing here for a wrapborrow-style
%   reformulation to remove, see fp32_16_cuda_kernels.cu's own comment).
%   Same M_BLOCK-scaling (target=512, floor=1) as "(truncated, workspace)"
%   below, so this isolates ONLY the vectorization change. PROMOTED: a
%   dramatic, MONOTONICALLY GROWING win on real A100 hardware over
%   "(truncated, workspace)" at every square size tested (M=1024-6144) --
%   +0.6% at the smallest up to +34.6% at M=6144, no regressions,
%   already beating cuBLAS outright from M=4096 up. bf16's decode being
%   essentially free (no compute to hide memory latency behind, unlike
%   fp16) makes this kernel MORE bandwidth-sensitive, not less, than its
%   fp32_16 counterpart -- wider loads matter more here. A further
%   M_BLOCK retuning sibling, mv_fp32_bf16_transpose_atomiconly_scaled_
%   mblock_vec4_retuned_floor (reusing fp32_16's own target=2048/
%   floor=256), measured a real further win at M=3072-8192 (+2.2%-+5.3%)
%   but a real regression at M=2048 specifically in two independent runs
%   (-7.6%/-9.4% vs plain vec4) -- CROSS-VALIDATED as likely NOT noise:
%   the rounded-encoding sibling (see "TOMOS FP32/BF16" below) shows the
%   SAME dip at the SAME M=2048 with a completely different decode
%   function. Grid-math root-cause candidate: at M=K=2048, gridK,
%   M_BLOCK, and gridM all land on clean powers of 2 simultaneously, a
%   triple alignment no neighboring size shares -- kept as a documented
%   open question, not promoted.
%
%   "(truncated, workspace)" = mv_fp32_bf16_accurate_ready_atomiconly_
%   scaled_kblock_workspace's own TRANSPOSE fast path, decoding via plain
%   fp32_bf16_combine against plain-truncating-split data, scalar-only
%   loads, the ORIGINAL untuned M_BLOCK-scaling target. DEMOTED now that
%   "(truncated)" above sources that slot instead -- kept as its own
%   visible series (fp32_bf16_truncated_workspace_gflops in the CSV), not
%   plotted, same "keep both, promote the winner" convention.
%
%   TOMOS FP32/BF16 = mv_fp32_bf16_transpose_atomiconly_scaled_mblock_
%   vec4_rounded: rounded-encoding counterpart of "(truncated)" above
%   (byte-for-byte identical except decoding via fp32_bf16_combine_
%   rounded), same "base the design on FP32/FP16 (fast)" port. Same
%   M_BLOCK-scaling (target=512, floor=1) as "(rounded, workspace)"
%   below, so this isolates ONLY the vectorization change. PROMOTED: on
%   real A100 hardware, a dramatic, growing win over "(rounded,
%   workspace)" mirroring "(truncated)"'s own curve closely -- roughly
%   flat at M=1024/1536 (0.972x-1.000x), then +4.3% at M=2048 climbing to
%   +21.6% at M=8192, no regressions anywhere tested. A further M_BLOCK
%   retuning sibling, mv_fp32_bf16_transpose_atomiconly_scaled_mblock_
%   vec4_rounded_retuned_floor, measured a further win at every size
%   except the SAME M=2048 anomaly "(truncated)"'s own retuned_floor
%   showed (-4.7% here) -- cross-validating that anomaly as decode-
%   independent (see above) -- kept as a documented open question, not
%   promoted.
%
%   "(rounded, workspace)" = mv_fp32_bf16_accurate_ready_atomiconly_
%   scaled_kblock_rounded's own TRANSPOSE fast path, decoding via
%   fp32_bf16_combine_rounded against rounded-split data, scalar-only
%   loads, the ORIGINAL untuned M_BLOCK-scaling target. DEMOTED now that
%   "TOMOS FP32/BF16" above sources that slot instead -- kept as its own
%   visible series (fp32_bf16_rounded_workspace_gflops in the CSV), not
%   plotted, same "keep both, promote the winner" convention.
%
%   fp32_16 (plain) vs fp32_16 (fast), and fp32_bf16 (plain) vs fp32_bf16
%   (truncated), are the SAME safety/speed and encoding tradeoffs the
%   NO_TRANSPOSE plot documents in full -- see plot_gemv_fp32_recon_vs_
%   cutlass_ref.m's own header comment for that derivation, unchanged
%   here (only the shape dispatch's TRANSPOSE branch is new, not the
%   underlying decode functions or their own safety contracts).
%
%   mv_fp32_transpose_recon_vs_cublas_ref()          uses the most
%       recently written mv_fp32_transpose_recon_vs_cublas_ref_*.csv
%       under <repo>/build/mv_fp32_transpose_recon_vs_cublas_ref_output/
%       (that directory is shared with scripts/benchmarks_gemv_
%       transpose_fp32_recon_vs_cublas_ref.sh's own runs, so this picks
%       up whichever run happened last).
%   plot_gemv_transpose_fp32_recon_vs_cublas_ref(csvPath)   reads that
%       CSV directly.
%
%   The CSV is expected to have M/K columns (BLAS convention: A is MxK,
%   column-major -- for TRANSPOSE, x has length M and y has length K, the
%   OPPOSITE of the NO_TRANSPOSE plot's own x-length-K/y-length-M
%   convention) plus *_ms/*_gflops columns per method -- see
%   benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh's own embedded
%   parser for the exact schema. Rows are split into two groups by shape,
%   same convention as the NO_TRANSPOSE plot:
%     - square:          M == K            -> x-axis is M (=K).
%     - tall-and-thin:   M ~= K (M >> K)    -> x-axis is M, K held fixed
%                                              at one of two values (512,
%                                              4096), both overlaid on the
%                                              same axes -- see
%                                              plot_series's own tall-
%                                              and-thin handling.
%   ("SKIPPED" cells, from a shape a method ran out of device memory for,
%   become NaN and simply leave a gap in that method's line.)
%
%   tag (optional) is appended (as "_<tag>") to both output basenames --
%   used by scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref_
%   advanced.sh to write "_avg"-suffixed plots (averaged over a caller-
%   controlled warmup/iteration count, see that script's own header)
%   alongside, without overwriting, this function's own default-named
%   plots. Omit it (or pass '') for the untagged filenames below.
%
%   Writes, under <repo>/build/benchmarks/output/plots/:
%     plot_gemv_transpose_fp32_recon_vs_cublas_ref_square[_<tag>].png / .fig
%     plot_gemv_transpose_fp32_recon_vs_cublas_ref_tall[_<tag>].png   / .fig

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
        default_dir = fullfile(repo_root, 'build', 'mv_fp32_transpose_recon_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_transpose_fp32_recon_vs_cublas_ref:noCsv', ...
                ['No mv_fp32_transpose_recon_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_transpose_fp32_recon_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_transpose_fp32_recon_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');
    K = to_double_col(T, 'K');

    % Dashed (vs every TOMOS series' solid line) since this is a reference
    % baseline (vendor library), not a TOMOS kernel -- same convention
    % plot_gemv_fp32_recon_vs_cutlass_ref.m's own series use. No CUTLASS
    % series here at all (unlike that file) -- this CSV has no cutlass_ms/
    % cutlass_gflops columns to begin with, see this file's own header
    % comment.
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'cuBLAS', 'values', to_double_col(T, 'cublas_gflops'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'native_gflops'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'fp32_16_gflops'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (implicit)', 'values', to_double_col(T, 'fp32_16_fast_gflops'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'fp32_bf16_gflops'), 'style', '-+');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (truncated)', 'values', to_double_col(T, 'fp32_bf16_truncated_gflops'), 'style', '-v');

    is_square = (M == K);
    is_tall = ~is_square;

    plot_series(M, series, is_square, 'Matrix dimensions (m = k)', ...
        'square', fullfile(out_dir, ['plot_gemv_transpose_fp32_recon_vs_cublas_ref_square' tag_suffix]));

    plot_series(M, series, is_tall, 'Row count m (k held fixed -- see legend/title)', ...
        'tall', fullfile(out_dir, ['plot_gemv_transpose_fp32_recon_vs_cublas_ref_tall' tag_suffix]));
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_transpose_fp32_recon_vs_cublas_ref:missingColumn', ...
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

function csv_path = find_latest_csv(dir_path)
    csv_path = '';
    if ~exist(dir_path, 'dir')
        return
    end
    files = dir(fullfile(dir_path, 'mv_fp32_transpose_recon_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, group_name, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_transpose_fp32_recon_vs_cublas_ref:emptyGroup', ...
            'No rows for "%s" -- skipping.', group_name);
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % Same fixed qualitative palette as plot_gemv_fp32_recon_vs_cutlass_
    % ref.m's own plot_series, sized with headroom past this plot's own
    % (6-series) count.
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
    % same convention as plot_gemv_fp32_recon_vs_cutlass_ref.m -- GEMV's
    % own arithmetic intensity is fixed regardless of size (bandwidth-
    % bound, not compute-bound).
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
    % inside it -- with 6 series, an inside legend covers curves. Same
    % column count/font size as plot_gemv_fp32_recon_vs_cutlass_ref.m's
    % own NO_TRANSPOSE plot, now that both plots have 6 series.
    kAxisFontSize = 40;
    kLegendFontSize = 28;
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
