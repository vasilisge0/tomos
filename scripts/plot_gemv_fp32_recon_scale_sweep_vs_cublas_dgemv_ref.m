function plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref(csv_path)
%PLOT_GEMV_FP32_RECON_SCALE_SWEEP_VS_CUBLAS_DGEMV_REF Plot the fp32-
%   recon-tier GEMV L2-relative-error sweep from scripts/benchmarks_gemv_
%   fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh: the SAME
%   binary and SAME six series (five TOMOS plus cuBLAS FP32) as plot_
%   gemv_fp32_recon_l2_error_vs_cublas_dgemv_ref.m (see that file's own
%   header for the full series derivation and for why a TRUE FP64
%   reference is used here instead of FP32, not repeated below), but a
%   SINGLE FIXED SHAPE (M=K=24576) swept across normal_scale instead of
%   across M -- the complementary axis to that sibling plot's own
%   M-sweep-at-one-scale-per-run. TRUE FP64 reference counterpart of
%   plot_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_ref.m (this file's
%   own FP32-reference sibling) -- same axis, only the reference precision
%   (and the added cuBLAS FP32 series) differs.
%
%   Square shape ONLY (M=K=24576, fixed, not swept) -- smaller than the
%   FP32-reference sibling's own M=K=65536, since each series here needs
%   four concurrently-alive M*K-sized buffers at peak instead of that
%   sibling's own two (see the .cu's own header for the memory
%   derivation). X-AXIS (normal_scale) IS LOG-SCALE. Y-AXIS IS LOG-SCALE
%   too.
%
%   The CSV is expected to have a `scale` column (the swept normal_scale
%   value) plus M/K columns (constant across every row here) plus
%   *_l2_rel_err columns per series -- see benchmarks_gemv_fp32_recon_
%   scale_sweep_l2_error_vs_cublas_dgemv_ref.sh's own embedded parser for
%   the exact schema. ("SKIPPED" cells become NaN and leave a gap in that
%   series' line.)
%
%   Writes, under <repo>/build/output/error_magnitude_scaling/gemv/fp32_compute/plots/:
%     plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_magnitude_scaling', 'gemv', 'fp32_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref:noCsv', ...
                ['No mv_fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref.sh ' ...
                 'first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_magnitude_scaling', 'gemv', 'fp32_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    scale = to_double_col(T, 'scale');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'tomos_fp32_l2_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_l2_rel_err'), 'style', '-v');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (fast)', 'values', to_double_col(T, 'tomos_fp32_fp16_fast_l2_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'tomos_fp32_bf16_l2_rel_err'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_bf16_workspace_l2_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP32', 'values', to_double_col(T, 'cublas_fp32_l2_rel_err'), 'style', '--d');

    out_base = fullfile(out_dir, 'plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref');

    plot_series(scale, series, true(size(scale)), 'Magnitude scale M', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mv_fp32_recon_scale_sweep_l2_error_vs_cublas_dgemv_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_fp32_recon_scale_sweep_vs_cublas_dgemv_ref:emptyGroup', 'No rows to plot -- skipping.');
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
        0.30 0.30 0.30;  % dark grey (cuBLAS FP32)
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
    ylabel('L2 relative error vs FP64', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
