function plot_gemm_fp32_c32_scale_sweep_raw_vs_dgemm_ref(csv_path)
%PLOT_GEMM_FP32_C32_SCALE_SWEEP_RAW_VS_DGEMM_REF
%   Plot the fp32-compute-tier GEMM Frobenius-relative-error sweep from
%   scripts/benchmarks_gemm_fp32_compute32_scale_sweep_raw_l2_error_vs_
%   cublas_dgemm_ref.sh: the SAME fixed shape (M=K=N=24576) and the SAME
%   normal_scale axis and SCALES range as plot_gemm_fp32_c32_scale_sweep_
%   vs_dgemm_ref.m's own data, but generated WITHOUT --normalized (B
%   stays at plain scale=1.0, not rescaled by 1/sqrt(K)).
%
%   Why this plot exists, alongside the --normalized one: Frobenius
%   relative error is scale-invariant in exact arithmetic -- rescaling B
%   rescales numerator and denominator by the same factor, so the ratio
%   itself shouldn't change, PROVIDED nothing overflows along the way.
%   Every series here accumulates AND stores C in genuine float (see the
%   .cu's own header), so unlike mm_fp32_16_variable_rounding_l2_error_
%   vs_cublas_ref.cu's own raw/normalized pair (whose raw sweep needed a
%   much tighter SCALES cap to avoid overflowing its own __half-typed C),
%   this file uses the IDENTICAL SCALES range as its --normalized
%   sibling -- comparing the two directly at every scale they share (not
%   just a partial overlap) is the actual point: it's a full-range check
%   of whether normalizing B distorts the rounding-method comparison
%   itself (see the .cu's own run_shape comment for the full derivation
%   of what --normalized does).
%
%   Same four series as the --normalized sibling plot, all measuring
%   float-accumulating kernels against a TRUE FP64 (cublasDgemm)
%   reference -- see plot_gemm_fp32_c32_vs_dgemm_ref.m's own header for
%   the full series description.
%
%   Square shape ONLY (M=K=N=24576, fixed, not swept). X-AXIS
%   (normal_scale) IS LOG-SCALE. Y-AXIS IS LOG-SCALE too -- see the
%   N-sweep sibling plot's own header for why.
%
%   The CSV is expected to have a `scale` column (the swept normal_scale
%   value) plus M/K/N columns (constant across every row here) plus
%   *_frobenius_rel_err columns per series -- see benchmarks_gemm_fp32_
%   compute32_scale_sweep_raw_l2_error_vs_cublas_dgemm_ref.sh's own
%   embedded parser for the exact schema. ("SKIPPED" cells become NaN and
%   leave a gap in that series' line.)
%
%   Writes, under <repo>/build/output/error_magnitude_scaling/gemm/fp32_compute/plots/:
%     plot_gemm_fp32_c32_scale_sweep_raw_vs_dgemm_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_magnitude_scaling', 'gemm', 'fp32_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp32_c32_scale_sweep_raw_vs_dgemm_ref:noCsv', ...
                ['No mm_fp32_compute32_scale_sweep_raw_l2_error_vs_cublas_dgemm_ref_*.csv found under ' ...
                 '%s.\nRun scripts/benchmarks_gemm_fp32_compute32_scale_sweep_raw_l2_error_vs_cublas_' ...
                 'dgemm_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp32_c32_scale_sweep_raw_vs_dgemm_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_magnitude_scaling', 'gemm', 'fp32_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    scale = to_double_col(T, 'scale');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'cuBLAS FP32', 'values', to_double_col(T, 'cublas_fp32_frobenius_rel_err'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'tomos_fp32_frobenius_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_frobenius_rel_err'), 'style', '-s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16 (truncated)', 'values', to_double_col(T, 'tomos_fp32_fp16_truncated_frobenius_rel_err'), 'style', '--s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_fp16_workspace_frobenius_rel_err'), 'style', '-.s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/BF16 (truncated)', 'values', to_double_col(T, 'tomos_fp32_bf16_truncated_frobenius_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/BF16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_bf16_workspace_frobenius_rel_err'), 'style', '--^');

    out_base = fullfile(out_dir, 'plot_gemm_fp32_c32_scale_sweep_raw_vs_dgemm_ref');

    plot_series(scale, series, true(size(scale)), 'Magnitude scale M', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp32_c32_scale_sweep_raw_vs_dgemm_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp32_compute32_scale_sweep_raw_l2_error_vs_cublas_dgemm_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp32_c32_scale_sweep_raw_vs_dgemm_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.30 0.30 0.30;  % dark grey (cuBLAS FP32 -- industry-standard baseline)
        0.55 0.55 0.55;  % grey (TOMOS FP32 -- no 16-bit encoding)
        0.00 0.45 0.74;  % blue (TOMOS FP32/FP16)
        0.00 0.70 0.90;  % light blue (TOMOS FP32/FP16 truncated)
        0.00 0.30 0.55;  % dark blue (TOMOS FP32/FP16 workspace)
        0.90 0.55 0.10;  % orange (TOMOS FP32/BF16 truncated)
        0.70 0.35 0.05;  % dark orange (TOMOS FP32/BF16 workspace)
    ];
    colors = kPalette(1:numel(series), :);
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
    grid on;

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
