function plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref(csv_path)
%PLOT_GEMM_FP16_BF16_PURE_SCALE_SWEEP_L2_ERROR_VS_CUBLAS_REF Plot the
%   pure/composite-tier (tensor-core) GEMM Frobenius-relative-error sweep
%   from scripts/benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_
%   cublas_ref.sh: the SAME binary and SAME 9 series as plot_gemm_fp16_
%   bf16_pure_l2_error_vs_cublas_ref.m (see that file's own header for the
%   full series derivation, not repeated here), but a SINGLE FIXED SHAPE
%   (M=K=N=65536) swept across normal_scale instead of across N -- the
%   complementary axis to that sibling plot's own N-sweep-at-one-scale-
%   per-run.
%
%   Why this axis: fp32_16_pack.hpp's/fp32_bf16_pack.hpp's own encoding
%   has NORMAL/EXTENDED/degraded branches, and only large enough |A[i,j]|
%   values ever cross out of "normal" -- the N-sweep sibling plot never
%   varies normal_scale at a given run, so it can only show rounding-
%   precision loss WITHIN the normal branch at whatever one scale that
%   run used. This plot's x-axis instead pushes more of A's (and B's,
%   this tier's own reference captures both operands' rounding) own
%   elements toward and across that boundary at one large, fixed shape.
%
%   Data was generated with --normalized (see the driver .sh's own header
%   for why: without it, this tier's own three fp16-group series overflow
%   their __half-typed C epilogue at scale=4096+; --normalized removes
%   that K-dependent ceiling, making the full SCALES range safe).
%
%   Square shape ONLY (M=K=N=65536, fixed, not swept). X-AXIS
%   (normal_scale) IS LOG-SCALE. Y-AXIS IS LOG-SCALE too -- same reasoning
%   as the N-sweep sibling plot's own header.
%
%   The CSV is expected to have a `scale` column (the swept normal_scale
%   value) plus M/K/N columns (constant across every row here) plus
%   *_frobenius_rel_err columns per series -- see benchmarks_gemm_fp16_
%   bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh's own embedded parser
%   for the exact schema. ("SKIPPED" cells become NaN and leave a gap in
%   that series' line.)
%
%   Writes, under <repo>/build/output/error_magnitude_scaling/gemm/fp16_compute/plots/:
%     plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_magnitude_scaling', 'gemm', 'fp16_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref.sh first.'], ...
                default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_magnitude_scaling', 'gemm', 'fp16_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    scale = to_double_col(T, 'scale');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col(T, 'tomos_fp16_frobenius_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS BF16', 'values', to_double_col(T, 'tomos_bf16_frobenius_rel_err'), 'style', '-<');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_frobenius_rel_err'), 'style', '-+');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_fp16_workspace_frobenius_rel_err'), 'style', '->');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'tomos_fp32_bf16_frobenius_rel_err'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_fp16_frobenius_rel_err'), 'style', '--d');
    i = i + 1; series(i) = struct('label', 'CUTLASS FP16', 'values', to_double_col(T, 'cutlass_fp16_frobenius_rel_err'), 'style', '--^');
    i = i + 1; series(i) = struct('label', 'cuBLAS BF16', 'values', to_double_col(T, 'cublas_bf16_frobenius_rel_err'), 'style', '--v');
    i = i + 1; series(i) = struct('label', 'CUTLASS BF16', 'values', to_double_col(T, 'cutlass_bf16_frobenius_rel_err'), 'style', '--p');

    out_base = fullfile(out_dir, 'plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref');

    plot_series(scale, series, true(size(scale)), 'Magnitude scale M', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp16_bf16_pure_scale_sweep_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.90 0.55 0.10;  % orange
        0.30 0.30 0.30;  % dark grey (cuBLAS FP16)
        0.00 0.80 0.80;  % cyan (CUTLASS FP16)
        0.60 0.40 0.20;  % brown (cuBLAS BF16)
        1.00 0.75 0.80;  % pink (CUTLASS BF16)
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
    ylabel('Frobenius relative error', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);

    yl = ylim;
    y_lo_dec = floor(log10(yl(1)));
    y_hi_dec = ceil(log10(yl(2)));
    if y_hi_dec <= y_lo_dec
        y_hi_dec = y_lo_dec + 1;
    end
    ylim(10.^[y_lo_dec, y_hi_dec]);
    set(gca, 'YTick', 10.^(y_lo_dec:y_hi_dec), 'YMinorTick', 'on');
    grid on;
    ax = gca;
    ax.YMinorGrid = 'on';

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
