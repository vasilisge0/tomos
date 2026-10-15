function plot_fp16_bf16_pure_vs_cutlass_ref(csv_path, tag)
%PLOT_FP16_BF16_PURE_VS_CUTLASS_REF Plot the "16-bit-compute" GEMM sweep
%   from scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh (which builds
%   and runs benchmarks/kernels/cuda/matrix/fp16_bf16_pure_vs_cutlass_
%   ref.cu, UNMODIFIED, across the SAME square-N sweep as scripts/
%   fp32_recon_vs_cutlass_ref.sh's own 32-bit-compute tier -- see that
%   script's own header comment for why it invokes the binary once per
%   size instead of one continuous sweep -- so results here are directly
%   size-for-size comparable to plot_gemm_format_32_16_compute_32.m's
%   own fp32-compute-tier numbers at every N both sweeps share):
%   cuBLAS (Hgemm for fp16, GemmEx/CUDA_R_16BF for bf16), real CUTLASS
%   (fp16/bf16 tensor-op device::Gemm -- OpClassTensorOp, NOT the fp32-
%   compute tier's own OpClassSimt config), this codebase's own best
%   native (unpacked) fp16/bf16 GEMM (gemm_fp16_mma_ptx_wide_bigreg_sq_
%   deep4 / gemm_bf16_mma_ptx_wide_bigreg_sq_deep4), and the packed-
%   precision "16-bit-compute" tier of both formats:
%
%   "TOMOS FP32 / FP16 (packed)" is mm_fp32_16_pure_half_wide_bigreg_sq_
%   inline_dirty -- decodes fp32_16's wh via fp32_16_half_only straight
%   to a __half value (never reconstructing the real fp32 the way the
%   fp32-compute tier's fp32_16_combine does), so the actual FFMA/MMA
%   compute happens in half precision throughout, not just storage.
%   "TOMOS FP32 / BF16 (packed)" is mm_fp32_bf16_pure_bf16_gemm_direct_
%   fastk_precomp2_serp -- structurally NOT a fused/inline-decode kernel
%   the way its fp16 sibling above is: bfloat16's exponent field is bit-
%   for-bit the same width and bias as float32's own, so fp32_bf16's wh
%   (float32's own top 16 bits, truncated) is ALREADY a valid bf16 bit
%   pattern for every possible input, unconditionally, with no exception
%   -- unlike fp16's control-bit-gated fp32_16_half_only. So this kernel
%   just reinterprets wh directly as __nv_bfloat16* (no decode kernel,
%   no scratch buffer, no branching) and calls this codebase's own best
%   native bf16 kernel on it unmodified -- promoted here (2026-08-28)
%   over the prior fused mm_fp32_bf16_pure_bf16_wide_bigreg_sq_deep4; see
%   that kernel's own comment in mm_cuda_kernels.cu for the full
%   derivation.
%   Both fp16 and bf16 "packed" kernels are genuinely different kernels
%   from their fp32-compute-tier namesakes plotted elsewhere, not the
%   same kernel relabeled -- they trade reconstruction accuracy (now
%   only as good as half/bf16 itself) for the possibility of real
%   tensor-core throughput.
%
%   plot_fp16_bf16_pure_vs_cutlass_ref()          uses the most recently
%       written fp16_bf16_pure_vs_cutlass_ref_matched_*.csv under
%       <repo>/build/output/benchmarks/gemm/fp16_compute/data/.
%   plot_fp16_bf16_pure_vs_cutlass_ref(csvPath)   reads that CSV directly.
%
%   The CSV is expected to have M/K/N columns (BLAS convention: A is MxK,
%   B is KxN, C is MxN) plus *_ms/*_gflops columns per method -- see
%   scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh's own embedded
%   parser for the exact schema. Split square (M==K==N, x-axis is N) vs
%   rectangular (M==N held fixed at 65536, x-axis is the inner dimension
%   K) rows -- same is_square = (M==K)&(K==N) convention as
%   plot_gemm_format_32_16_compute_32.m's own square+rect split.
%   ("SKIPPED" cells, from a shape a method ran out of device memory for,
%   become NaN and simply leave a gap in that method's line.)
%
%   tag (optional) is appended (as "_<tag>") to both output basenames --
%   used by scripts/fp16_bf16_pure_vs_cutlass_ref_matched_advanced.sh to
%   write "_avg"-suffixed plots (averaged over a caller-controlled
%   warmup/iteration count, see that script's own header) alongside,
%   without overwriting, this function's own default-named plots. Omit
%   it (or pass '') for the untagged filenames below -- fully backward
%   compatible with scripts/plot_fp16_bf16_pure_vs_cutlass_ref.sh's own
%   existing 0-or-1-arg calls.
%
%   Writes, under <repo>/build/output/benchmarks/gemm/fp16_compute/plots/:
%     plot_gemm_format_32_16_compute_16_square[_<tag>].png / .fig
%     plot_gemm_format_32_16_compute_16_rect[_<tag>].png / .fig

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
        default_dir = fullfile(repo_root, 'build', 'output', 'benchmarks', 'gemm', 'fp16_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_fp16_bf16_pure_vs_cutlass_ref:noCsv', ...
                ['No fp16_bf16_pure_vs_cutlass_ref_matched_*.csv found under %s.\n' ...
                 'Run scripts/fp16_bf16_pure_vs_cutlass_ref_matched.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_fp16_bf16_pure_vs_cutlass_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'benchmarks', 'gemm', 'fp16_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');
    K = to_double_col(T, 'K');
    N = to_double_col(T, 'N');

    % Vendor-library baselines (cuBLAS, CUTLASS) use non-solid styles,
    % solid for every TOMOS kernel -- same base convention as
    % plot_gemm_format_32_16_compute_32.m's own series-construction
    % block. Within the baselines, FP16 uses a plain dash ('--') and
    % BF16 uses a dash-dot ('-.') so the two precisions stay visually
    % distinct from each other, not just from TOMOS.
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_fp16_gflops'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'cuBLAS BF16', 'values', to_double_col(T, 'cublas_bf16_gflops'), 'style', '-.s');
    i = i + 1; series(i) = struct('label', 'CUTLASS FP16', 'values', to_double_col(T, 'cutlass_fp16_gflops'), 'style', '--^');
    i = i + 1; series(i) = struct('label', 'CUTLASS BF16', 'values', to_double_col(T, 'cutlass_bf16_gflops'), 'style', '-.d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col(T, 'native_fp16_gflops'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS BF16', 'values', to_double_col(T, 'native_bf16_gflops'), 'style', '-<');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'pure_half_gflops'), 'style', '-+');
    % Fast-decode sibling of TOMOS FP32 / FP16 above -- calls
    % mm_fp32_16_pure_half_wide_bigreg_sq_inline_dirty_fast_acongruous_
    % fastk_precomp2_serp, byte-for-byte identical to the kernel above
    % except its one decode call uses fp32_16_half_only_fast instead of
    % fp32_16_half_only (saturates underflowed values to signed zero
    % instead of recovering a genuine fp16 subnormal -- see that
    % function's own comment in fp32_16_pack.hpp). to_double_col_optional
    % since older CSVs predate this series.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (fast)', 'values', to_double_col_optional(T, 'pure_half_fast_gflops', numel(M)), 'style', ':+');
    % "(workspace)" -- same label convention plot_gemm_format_32_16_
    % compute_32.m already uses for its own materialize series: decodes
    % wh into a scratch buffer ahead of time (chunk-bounded, cached
    % across calls -- see mm_fp32_16_pure_half_gemm_materialize_cached_
    % vec_fastk_precomp2_serp's own comment in mm_cuda_kernels.cu), then
    % calls the native fp16 kernel completely unmodified -- loses only
    % at N=1024 (two kernel launches vs one, even with the allocator
    % warm) but reaches ~100.3% of native fp16 by N=32768, the best
    % result this whole family has produced. Uses the CACHED+VEC variant
    % specifically -- caching removes the plain kernel's repeated
    % cudaMalloc/cudaFree overhead (dominant at small N), and the
    % vectorized (ushort4/uint2) decode kernel closes most of the
    % remaining gap at N=2048-8192 over the scalar-decode cached kernel.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (workspace)', 'values', to_double_col(T, 'pure_half_workspace_gflops'), 'style', '->');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'pure_bf16_gflops'), 'style', '-*');

    is_square = (M == K) & (K == N);
    is_rect = ~is_square;

    plot_series(N, series, is_square, 'Matrix dimensions (m = k = n)', ...
        'square', fullfile(out_dir, ['plot_gemm_format_32_16_compute_16_square' tag_suffix]));

    plot_series(K, series, is_rect, 'Inner dimension k', ...
        'rectangular', fullfile(out_dir, ['plot_gemm_format_32_16_compute_16_rect' tag_suffix]));
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_fp16_bf16_pure_vs_cutlass_ref:missingColumn', ...
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
% generated (e.g. pure_half_fast_gflops, added alongside the
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
    files = dir(fullfile(dir_path, 'fp16_bf16_pure_vs_cutlass_ref_matched_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, group_name, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_fp16_bf16_pure_vs_cutlass_ref:emptyGroup', ...
            'No rows for "%s" -- skipping.', group_name);
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % lines(N) only has 7 truly distinct colors and CYCLES for N>7 --
    % with 8 series that silently reuses colors, which is impossible to
    % tell apart at a glance since only the marker shape then differs.
    % This is a fixed qualitative palette instead, sized with headroom
    % past the current series count so it doesn't need revisiting every
    % time one more method is added -- same palette as
    % plot_gemm_format_32_16_compute_32.m's own plot_series.
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
    % benchmark's own printed units) but plotted in TFlop/s -- easier to
    % read at the scale these kernels actually run at.
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
    % readability at a glance (axis size matches plot_gemm_format_32_16_
    % compute_32.m's own plot_series). Legend placed below the plot
    % (southoutside, 3 columns) rather than inside it -- this plot has 9
    % series, too many to fit inside the axes without covering curves.
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
