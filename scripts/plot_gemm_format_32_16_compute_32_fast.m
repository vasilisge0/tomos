function plot_gemm_format_32_16_compute_32_fast(csv_path, tag)
%PLOT_GEMM_FORMAT_32_16_COMPUTE_32_FAST Same fp32-compute GEMM sweep as
%   plot_gemm_format_32_16_compute_32 (its own header below has the full
%   series writeup, not repeated here), PLUS two UNCONDITIONAL fast_normal
%   decode series -- "TOMOS FP32 / FP16 (fast)" (mm_fp32_16_tiled_
%   cpasync_dualbuf_fast_normal) and "TOMOS FP32 / FP16 (workspace,
%   fast)" (mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal) --
%   that the original function deliberately omits, to keep its own output
%   unchanged for any existing consumer. Both read the SAME CSV (this
%   file adds no new build/run step of its own); the two extra columns
%   are read via to_double_col_optional so a CSV from before these
%   kernels existed still plots cleanly (those two lines simply don't
%   appear). See fp32_16_cuda_kernels.cu's own comments on those two
%   kernels for the UNSAFE-ON-ARBITRARY-DATA contract they share.
%
%   scripts/benchmarks_gemm_format_32_16_compute_32.sh (which itself builds
%   and runs benchmarks/kernels/cuda/matrix/benchmarks_gemm_fp32.cu):
%   cuBLAS (Sgemm), real CUTLASS (SIMT fp32), this codebase's own native
%   fp32 GEMM (plain, and its "regpipe" sibling -- gemm_fp32_tiled_
%   cpasync_dualbuf_regpipe adds a register-level software-pipeline
%   stage to the inner kk-loop, prefetching next-kk operands before this
%   kk's FFMAs run, motivated by an ncu/SASS comparison showing CUTLASS's
%   compiled loop interleaves loads with FFMAs far more tightly despite
%   both kernels being equally register/occupancy-bound), and the
%   fp32-compute tier of both packed-precision formats (fp32_16 and
%   fp32_bf16, both via their tiled_cpasync_dualbuf kernel).
%   fp32_16 appears six times, comparing distinct strategies for its
%   decode cost. "TOMOS FP32 / FP16" is mm_fp32_16_tiled_cpasync_dualbuf
%   -- the OLDER fused reference kernel (decodes each tile's wh/wr
%   straight into shared memory via fp32_16_combine, in transform_tile,
%   with the older BK=16/2-stage/plain-row-major tiling). A single prior
%   N=8192 data point once suggested the newer, genuinely-fused
%   midpoint_exact-based kernel (mm_fp32_16_syncpipe4_bk8_warptile_
%   nostride_snake_midpoint_exact -- ports this codebase's current best
%   native fp32 kernel's tiling into the fp32_16 file) beat dualbuf by
%   ~3%, but that did NOT hold up: a controlled, same-run, full-sweep
%   head-to-head (scripts/fp32_16_plotted_vs_cutlass_ref.sh) showed
%   dualbuf actually winning at EVERY measured size instead, by a
%   modest but consistent 0.8-4.7% (mostly ~1-2% at large N) -- the
%   single-point measurement was apparently just run-to-run noise, not
%   reproducible. dualbuf is switched IN for this series as a result
%   (the fused midpoint_exact kernel switched OUT, commented not
%   deleted); its own decode strategy (its "truncated" sibling --
%   fp32_16_combine_truncated, no rounding, no round_flag correction;
%   "adaptive" -- mm_fp32_16_tiled_cpasync_dualbuf_adaptive, skips the
%   exponent-mode dispatch per-tile via a block-wide __syncthreads_or
%   reduction when a whole tile is control=0; "adaptive, warp" -- same
%   idea, voted per-warp via __any_sync instead of per-block; "multin"
%   -- mm_fp32_16_tiled_cpasync_dualbuf_multin, each block covers 2
%   N-tiles, decoding A into shared memory once per K-step and reusing
%   it for both) still backs the other four fp32_16 series here. "TOMOS
%   FP32 / FP16 (materialize)" is mm_fp32_16_gemm_materialize_midpoint_
%   exact -- decodes A to plain fp32 in global memory up front, then
%   runs midpoint_exact as a black box, eliminating the redundant
%   per-N-tile decode entirely rather than reducing it; this is the
%   BEST fp32_16 kernel of any kind. All six decode the SAME (wh,wr)
%   data to bit-identical results -- only their runtime cost differs.
%
%   Fused (decode into shared memory, part of the same kernel as the
%   tiled compute) vs materialize (decode to a scratch buffer, then call
%   the native GEMM as a black box) are genuinely different
%   architectures, not parameter tweaks of each other -- see fp32_16_
%   cuda_kernels.cu's own comments on mm_fp32_16_tiled_cpasync_dualbuf_
%   kernel, mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_
%   exact_kernel, and mm_fp32_16_gemm_materialize_midpoint_exact.
%   Measured best-to-worst at N=8192 (scripts/fp32_new_kernels_vs_
%   cutlass_ref.sh 8192): materialize_midpoint_exact (18450.2 GFlop/s,
%   0.961x vs CUTLASS, "TOMOS FP32 / FP16 (materialize)" below) > the
%   plain gemm_materialize this series used before being switched to
%   materialize_midpoint_exact (17586.9 GFlop/s) > dualbuf (16164.2
%   GFlop/s at N=8192 in the full-sweep run above, "TOMOS FP32 / FP16"
%   above) > the genuinely-fused midpoint_exact-based kernel (15975.5
%   GFlop/s at N=8192 in that same run, switched out of "TOMOS FP32 /
%   FP16" above).
%
%   "TOMOS FP32 / FP16 (truncated)" is mm_fp32_16_tiled_cpasync_dualbuf_
%   truncated -- the plain, 0-spill truncated-decode reference kernel
%   (decodes via fp32_16_combine_truncated: never-rounded, no round_flag
%   correction). This is the OPPOSITE direction from the plain fp32_16
%   swap above: a full-sweep head-to-head (scripts/fp32_16_plotted_vs_
%   cutlass_ref.sh) showed the fused midpoint_exact-based truncated
%   kernel (mm_fp32_16_syncpipe4_bk8_warptile_nostride_snake_midpoint_
%   exact_truncated) LOSES to plain dualbuf_truncated at every measured
%   size, ~3% at N=1024 widening to ~8-10% for N>=8192, because
%   fp32_16_combine_truncated apparently has a meaningfully larger
%   register footprint than fp32_16_combine -- the fused kernel compiles
%   with a real spill (26 STL + 17 LDL at 128 registers) here, unlike
%   its 0-spill plain-decode sibling above. dualbuf_truncated is
%   switched IN for this series (the fused kernel switched OUT,
%   commented not deleted) as a result.
%
%   "TOMOS FP32 / BF16" is mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_
%   snake_midpoint_exact -- the fp32_bf16 counterpart to the fp32_16
%   fused midpoint_exact kernel above, decoding via fp32_bf16_combine
%   instead of fp32_16_combine. Unlike fp32_16, where materialize wins
%   because fp32_16_combine's rounding logic is register-heavy,
%   fp32_bf16_combine is unconditional/branchless (bf16's 8-bit exponent
%   matches fp32's exactly, so there's no degraded-range correction
%   branch) and compiles with 0 spill at 128 registers -- so here the
%   fused kernel wins outright over mm_fp32_bf16_tiled_cpasync_dualbuf,
%   by ~7-14% across every measured size (scripts/fp32_16_plotted_vs_
%   cutlass_ref.sh). dualbuf itself is switched out of this series below
%   (commented, not deleted) in favor of the fused kernel.
%
%   plot_gemm_format_32_16_compute_32_fast()          uses the most recently
%       written fp32_recon_vs_cutlass_ref_*.csv under
%       <repo>/build/output/benchmarks/gemm/fp32_compute/data/ (that directory is
%       shared with fp32_recon_vs_cutlass_ref.sh's own runs, so this picks
%       up whichever script ran last).
%   plot_gemm_format_32_16_compute_32_fast(csvPath)   reads that CSV directly.
%
%   The CSV is expected to have M/K/N columns (BLAS convention: A is MxK,
%   B is KxN, C is MxN) plus *_ms/*_gflops columns per method -- see
%   benchmarks_gemm_format_32_16_compute_32.sh's own embedded parser for
%   the exact schema. Rows are split into two groups by shape:
%     - square:      M == K == N        -> x-axis is the matrix size N.
%     - rectangular: M == N, K varies   -> x-axis is the inner dimension K.
%   ("SKIPPED" cells, from a shape a method ran out of device memory for,
%   become NaN and simply leave a gap in that method's line.)
%
%   tag (optional) is appended (as "_<tag>") to both output basenames --
%   used by scripts/benchmarks_gemm_format_32_16_compute_32_advanced.sh
%   to write "_avg"-suffixed plots (averaged over a caller-controlled
%   warmup/iteration count, see that script's own header) alongside,
%   without overwriting, this function's own default-named plots. Omit
%   it (or pass '') for the untagged filenames below.
%
%   Writes, under <repo>/build/output/benchmarks/gemm/fp32_compute/plots/:
%     plot_gemm_format_32_16_compute_32_fast_square[_<tag>].png / .fig
%     plot_gemm_format_32_16_compute_32_fast_rect[_<tag>].png   / .fig

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
        default_dir = fullfile(repo_root, 'build', 'output', 'benchmarks', 'gemm', 'fp32_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_format_32_16_compute_32_fast:noCsv', ...
                ['No fp32_recon_vs_cutlass_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_format_32_16_compute_32.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_format_32_16_compute_32_fast:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'benchmarks', 'gemm', 'fp32_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');
    K = to_double_col(T, 'K');
    N = to_double_col(T, 'N');

    % Built as individually-indexed struct-array elements (rather than
    % one struct() call over three parallel cell arrays) specifically so
    % a single series can be swapped out and commented -- not deleted --
    % without disturbing the others; see the materialize/materialize_
    % midpoint_exact swap below.
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    % Dashed (vs every other series' solid line) since these two are
    % reference baselines (vendor libraries), not TOMOS kernels -- makes
    % them visually distinct from the kernels actually being compared.
    i = i + 1; series(i) = struct('label', 'cuBLAS', 'values', to_double_col(T, 'cublas_gflops'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'CUTLASS', 'values', to_double_col(T, 'cutlass_gflops'), 'style', '--s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'native_gflops'), 'style', '-^');
    % Commented out (not deleted) to declutter the plot -- data still
    % captured in the CSV as native_regpipe_ms/gflops either way:
    %     i = i + 1; series(i) = struct('label', 'TOMOS Native (regpipe)', 'values', to_double_col(T, 'native_regpipe_gflops'), 'style', '-v');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'fp32_16_dualbuf_gflops'), 'style', '-d');
    % UNCONDITIONAL fast_normal decode sibling of TOMOS FP32 / FP16 above
    % (mm_fp32_16_tiled_cpasync_dualbuf_fast_normal) -- byte-for-byte
    % identical kernel except A's decode skips the exponent-mode dispatch
    % entirely, unconditionally (no per-tile validity check). UNSAFE ON
    % ARBITRARY DATA -- this is a pure throughput ceiling measurement
    % (what decode COULD cost if the dispatch were free), not a
    % recommended drop-in kernel; pair with mm_fp32_compute32_l2_error_
    % vs_cublas_dgemm_ref.cu's own "TOMOS FP32/FP16 (fast)" series for the
    % accuracy cost this speed (if any) comes with. Relabeled "(implicit)"
    % here (was "(fast)") to free that label for a genuinely different
    % decode-fast concept (fp32_16_half_only_fast's saturate-only
    % underflow handling) used in the FP16-compute-tier GEMM plots.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (implicit)', 'values', to_double_col_optional(T, 'fp32_16_dualbuf_fast_gflops', numel(M)), 'style', ':d');
    % Switched back to plain dualbuf (mm_fp32_16_tiled_cpasync_dualbuf)
    % in favor over the genuinely-fused midpoint_exact-based kernel
    % below, confirmed by a full-sweep head-to-head (scripts/fp32_16_
    % plotted_vs_cutlass_ref.sh, all square + rectangular sizes):
    % dualbuf wins at EVERY measured size, 0.8-4.7% (mostly ~1-2% at
    % large N, e.g. 16164.2 vs 15975.5 GFlop/s at N=8192). This
    % contradicts an earlier single N=8192 data point that had shown the
    % fused kernel winning by ~3% -- that measurement did not hold up
    % under a controlled, same-run comparison and was apparently just
    % noise. Fused kernel switched out below -- commented out, not
    % deleted:
    %     i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'fp32_16_fused_midpoint_exact_gflops'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (truncated)', 'values', to_double_col(T, 'fp32_16_dualbuf_trunc_gflops'), 'style', '-x');
    % Switched back to plain dualbuf_trunc (mm_fp32_16_tiled_cpasync_
    % dualbuf_truncated -- 0-spill) in favor over the fused midpoint_
    % exact-based truncated kernel below, confirmed by a full-sweep
    % head-to-head (scripts/fp32_16_plotted_vs_cutlass_ref.sh, all square
    % + rectangular sizes): dualbuf_trunc wins at EVERY measured size,
    % ~3% at N=1024 widening to ~8-10% for N>=8192. This is the OPPOSITE
    % direction from the plain (non-truncated) fp32_16 case just above,
    % where the fused kernel currently wins (though that comparison is
    % itself only a single N=8192 data point, not yet confirmed by a
    % full sweep) -- here the fused kernel's real compiled spill (26 STL
    % + 17 LDL at 128 registers, vs 0 for dualbuf_trunc's fp32_16_
    % combine_truncated decode) costs it the win outright. Fused kernel
    % switched out below -- commented out, not deleted:
    %     i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (truncated)', 'values', to_double_col(T, 'fp32_16_fused_midpoint_exact_trunc_gflops'), 'style', '-x');
    % Commented out (not deleted) to declutter the plot -- data still
    % captured in the CSV as fp32_16_dualbuf_adaptive_ms/gflops,
    % fp32_16_dualbuf_adaptive_warp_ms/gflops, and fp32_16_dualbuf_
    % multin_ms/gflops either way:
    %     i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (adaptive)', 'values', to_double_col(T, 'fp32_16_dualbuf_adaptive_gflops'), 'style', '-p');
    %     i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (adaptive, warp)', 'values', to_double_col(T, 'fp32_16_dualbuf_adaptive_warp_gflops'), 'style', '-h');
    %     i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (multin)', 'values', to_double_col(T, 'fp32_16_dualbuf_multin_gflops'), 'style', '->');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (workspace)', 'values', to_double_col(T, 'fp32_16_gemm_materialize_midpoint_exact_gflops'), 'style', '-<');
    % UNCONDITIONAL fast_normal decode sibling of TOMOS FP32 / FP16
    % (workspace) above (mm_fp32_16_gemm_materialize_midpoint_exact_
    % fast_normal) -- same materialize-then-native-GEMM structure, but
    % the chunk-decode pass skips the exponent-mode dispatch entirely.
    % Same UNSAFE-ON-ARBITRARY-DATA caveat as TOMOS FP32 / FP16
    % (implicit) above; relabeled for the same reason.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (workspace, implicit)', 'values', to_double_col_optional(T, 'fp32_16_gemm_materialize_midpoint_exact_fast_gflops', numel(M)), 'style', ':<');
    % Switched out in favor of the materialize_midpoint_exact series
    % above (measured 4.9% faster at N=8192, same materialize strategy
    % -- decode to a scratch buffer, then call the native GEMM as a
    % black box -- just pointed at the newer, faster native kernel) --
    % commented out, not deleted:
    %     i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (materialize)', 'values', to_double_col(T, 'fp32_16_gemm_materialize_gflops'), 'style', '-<');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (truncated)', 'values', to_double_col(T, 'fp32_bf16_fused_midpoint_exact_gflops'), 'style', '-+');
    % Switched out in favor of the fused midpoint_exact-based bf16 kernel
    % above (mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_
    % midpoint_exact -- same tiling as this codebase's best native fp32
    % kernel, decode via fp32_bf16_combine straight into the tile
    % pipeline's shared-memory A buffer). Unlike fp32_16, where the
    % materialize strategy wins because fp32_16_combine's rounding logic
    % is register-heavy, fp32_bf16_combine is unconditional/branchless
    % (bf16's 8-bit exponent matches fp32's exactly, so no degraded-
    % range correction is ever needed) and compiles with 0 spill at 128
    % registers -- so here the fused kernel wins outright, beating
    % dualbuf below by ~7-14% at every measured size (14.2% at N=1024,
    % narrowing to ~7.4% for N>=16384 and holding there through the
    % largest sizes and the full rectangular K sweep) -- commented out,
    % not deleted:
    %     i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'fp32_bf16_dualbuf_gflops'), 'style', '-+');
    % mm_fp32_bf16_gemm_materialize_rounded -- the ROUNDED-encoding bf16
    % format (wh formed by round-to-nearest-even instead of truncation,
    % with the round decision stored in wr's own LSB -- see fp32_bf16_
    % pack.hpp's fp32_bf16_split_rounded/fp32_bf16_combine_rounded)
    % decoded via the SAME materialize strategy as "TOMOS FP32 / FP16
    % (workspace)" above: decode A to a scratch fp32 buffer up front,
    % then run the native GEMM as a black box. Built specifically to
    % sidestep the fused rounded kernel's own real spill (REG:128
    % STACK:56, 23 STL + 11 LDL -- investigated at length and not
    % resolved, see mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_snake_
    % midpoint_exact_rounded_kernel's own comment in fp32_16_cuda_
    % kernels.cu), since materialize's decode pass is a simple per-
    % element kernel with no GEMM tiling register pressure at all.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (workspace)', 'values', to_double_col(T, 'fp32_bf16_materialize_rounded_gflops'), 'style', '-*');

    is_square = (M == K) & (K == N);
    is_rect = ~is_square;

    plot_series(N, series, is_square, 'Matrix dimensions (m = k = n)', ...
        'square', fullfile(out_dir, ['plot_gemm_format_32_16_compute_32_fast_square' tag_suffix]));

    plot_series(K, series, is_rect, 'Inner dimension k', ...
        'rectangular', fullfile(out_dir, ['plot_gemm_format_32_16_compute_32_fast_rect' tag_suffix]));
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_format_32_16_compute_32_fast:missingColumn', ...
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
% empty/absent line, same as an all-SKIPPED column would) instead of
% erroring when the CSV doesn't have this column at all -- for series
% added after older CSVs were already generated by a sibling driver
% script (e.g. benchmarks_gemm_format_32_16_compute_32_advanced.sh's own
% fp32_recon_vs_cutlass_ref_advanced.cu, which doesn't yet know about the
% fast_normal kernels) that still calls this same plotting function.
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
    files = dir(fullfile(dir_path, 'fp32_recon_vs_cutlass_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, group_name, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_format_32_16_compute_32_fast:emptyGroup', ...
            'No rows for "%s" -- skipping.', group_name);
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % lines(N) only has 7 truly distinct colors and CYCLES for N>7 --
    % with 10 series that silently reuses colors (e.g. series 8 gets
    % series 1's color), which is impossible to tell apart at a glance
    % since only the marker shape then differs. This is a fixed
    % qualitative palette instead, sized with headroom past the current
    % series count so it doesn't need revisiting every time one more
    % method is added.
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
    % benchmarks' own printed units) but plotted in TFlop/s -- easier to
    % read at the scale these kernels actually run at (order 10-20
    % TFlop/s rather than 10000-20000 GFlop/s).
    kGflopsToTflops = 1e-3;
    max_y = 0;
    for i = 1:numel(series)
        y = series(i).values(mask) * kGflopsToTflops;
        y = y(order);
        if any(~isnan(y))
            max_y = max(max_y, max(y, [], 'omitnan'));
        end
        plot(x_sorted, y, series(i).style, 'LineWidth', 4.5, 'MarkerSize', 6, ...
            'Color', colors(i, :), 'DisplayName', series(i).label);
    end
    hold off;

    % Default axis/legend font sizes (~8-10pt) read as too small once a
    % plot is scaled down for a slide or paper -- bumped up here for
    % readability at a glance. Legend placed below the plot (southoutside,
    % 3 columns) rather than inside it -- with 8 series, an inside legend
    % covers curves. 3 columns (not 4) to match plot_fp16_bf16_pure_vs_
    % cutlass_ref.m's own layout -- fewer, wider columns read more easily
    % than more, narrower ones for labels this long.
    kAxisFontSize = 40;
    kLegendFontSize = 28;
    set(gca, 'XScale', 'log', 'XTick', unique(x_sorted), 'FontSize', kAxisFontSize, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('TFlop/s', 'FontSize', kAxisFontSize);
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
