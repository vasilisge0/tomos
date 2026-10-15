function plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref(csv_path, scale_tag)
%PLOT_GEMV_FP32_16_PURE_HALF_L2_ERROR_VS_CUBLAS_REF Plot the NO_TRANSPOSE
%   (y = alpha*A*x) L2-relative-error sweep from scripts/benchmarks_gemv_
%   fp32_16_pure_half_l2_error_vs_cublas_ref.sh (which itself builds and
%   runs benchmarks/kernels/cuda/matrix/mv_fp32_16_pure_half_l2_error_vs_
%   cublas_ref.cu) -- the accuracy-vs-M companion to plot_gemv_fp32_16_
%   pure_half_vs_cublas_ref.m (which plots GFlop/s for the same shapes).
%   Six "TOMOS" genuine-16-bit-compute NO_TRANSPOSE GEMV series, same
%   shapes, but the y-axis here is how far each method's result drifts
%   from a genuine FP32-precision reference (cublasSgemv(CUBLAS_OP_N) on
%   an uncompressed float32 copy of that method's own data), not
%   throughput:
%
%   TOMOS FP16              = mv_fp32_16_pure_half_wide_ilp_shared_half2_
%                              atomic_best
%   TOMOS FP16 (fast)       = mv_fp32_16_pure_half_fast_wide_ilp_shared_
%                              half2_atomic_best
%   TOMOS FP32 / FP16       = mv_fp32_16_hybrid_wide_ilp_shared_half2_
%                              atomic_branchless_best
%   TOMOS FP32 / FP16 (implicit) = mv_fp32_16_pure_half_wide_ilp_shared_
%                              half2_atomic_fasthalf_best
%   TOMOS FP32 / BF16       = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
%                              atomic_best_v2
%   TOMOS BF16              = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
%                              atomic_best
%
%   Six series (not four, unlike this tier's TRANSPOSE sibling
%   plot_gemv_transpose_fp32_16_pure_half_l2_error_vs_cublas_ref.m) --
%   verified directly against plot_gemv_fp32_16_pure_half_vs_cublas_ref.m's
%   own current `series` struct: this NO_TRANSPOSE tier plots a separate
%   plain "TOMOS BF16" series in addition to "TOMOS FP32/BF16" (the two
%   are genuinely distinct dispatchers, not just a naming difference --
%   see the .cu file's own header). See that plot's own header for the
%   full derivation/promotion history of each dispatcher above.
%
%   Plus two external, non-TOMOS GEMV baselines (dashed lines), measuring
%   THEIR OWN error against the SAME plain-tier float32 reference/ground
%   truth as TOMOS FP16/FP32-BF16 above (independent __half/__nv_bfloat16-
%   typed A/x, index-matched via the SAME pseudo_rand(idx) call):
%
%   cuBLAS FP16              = cublasHgemm (n=1, CUBLAS_OP_N)
%   cuBLAS BF16               = cublasGemmEx (CUDA_R_16BF, n=1, CUBLAS_OP_N)
%
%   cuBLAS itself (cublasSgemv(CUBLAS_OP_N), real fp32) remains this
%   file's own reference throughout.
%
%   Square sweep ONLY (M=K). Y-AXIS IS LOG-SCALE: L2 relative error spans
%   orders of magnitude across these methods/shapes, unlike GFlop/s.
%
%   The CSV is expected to have M/K columns plus *_l2_rel_err columns per
%   method -- see benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_
%   ref.sh's own embedded parser for the exact schema. ("SKIPPED" cells
%   become NaN and leave a gap in that method's line.)
%
%   scale_tag (optional) identifies which "normal_scale" tier this CSV's
%   own data was generated at -- appended to the output filename and
%   shown in the plot's own title. Omit it (or pass '') for a single ad
%   hoc run with no tier to distinguish.
%
%   Writes, under <repo>/build/benchmarks/output/plots/error/gemv16/:
%     plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref[_scale<scale_tag>].png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        scale_tag = '';
    end
    scale_tag = char(scale_tag);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mv_fp32_16_pure_half_l2_error_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref:noCsv', ...
                ['No mv_fp32_16_pure_half_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref.sh first.'], ...
                default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots', 'error', 'gemv16');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col(T, 'tomos_fp16_l2_rel_err'), 'style', '->');
    % Fast-decode sibling of TOMOS FP16 above (fp32_16_half_only_fast) --
    % safe on arbitrary data, unlike TOMOS FP32/FP16 (implicit) below.
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (fast)', 'values', to_double_col(T, 'tomos_fp16_fast_l2_rel_err'), 'style', ':>');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_l2_rel_err'), 'style', '-v');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (implicit)', 'values', to_double_col(T, 'tomos_fp32_fp16_implicit_l2_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'tomos_fp32_bf16_l2_rel_err'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS BF16', 'values', to_double_col(T, 'tomos_bf16_l2_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_fp16_l2_rel_err'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'cuBLAS BF16', 'values', to_double_col(T, 'cublas_bf16_l2_rel_err'), 'style', '--s');

    if isempty(scale_tag)
        out_base = fullfile(out_dir, 'plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref');
    else
        out_base = fullfile(out_dir, ['plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref_scale' scale_tag]);
    end

    plot_series(M, series, true(size(M)), 'Matrix dimensions (m = k)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref:missingColumn', ...
            'CSV is missing expected column "%s".', name);
    end
    col = T.(name);
    if isnumeric(col)
        v = double(col);
    else
        v = str2double(string(col));
    end
end

function csv_path = find_latest_csv(dir_path)
    csv_path = '';
    if ~exist(dir_path, 'dir')
        return
    end
    files = dir(fullfile(dir_path, 'mv_fp32_16_pure_half_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_fp32_16_pure_half_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.10 0.75 0.80;  % cyan (TOMOS FP16 (fast))
        0.85 0.10 0.10;  % red
        0.90 0.55 0.10;  % orange (TOMOS FP32/FP16 (implicit))
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple (TOMOS BF16)
        0.30 0.30 0.30;  % dark grey (cuBLAS FP16)
        0.55 0.35 0.15;  % brown (cuBLAS BF16)
    ];
    if numel(series) <= size(kPalette, 1)
        colors = kPalette(1:numel(series), :);
    else
        colors = [kPalette; lines(numel(series) - size(kPalette, 1))];
    end
    for i = 1:numel(series)
        y = series(i).values(mask);
        y = y(order);
        plot(x_sorted, y, series(i).style, 'LineWidth', 4.5, 'MarkerSize', 6, ...
            'Color', colors(i, :), 'DisplayName', series(i).label);
    end
    hold off;

    kAxisFontSize = 40;
    kTickFontSize = 32;
    kLegendFontSize = 28;
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XTick', unique(x_sorted), ...
        'FontSize', kTickFontSize, 'XTickLabelRotation', 40, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('L2 relative error vs FP32', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
