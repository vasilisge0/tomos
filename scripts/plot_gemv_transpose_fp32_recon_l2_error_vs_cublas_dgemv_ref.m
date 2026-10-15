function plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref(csv_path, scale_tag)
%PLOT_GEMV_TRANSPOSE_FP32_RECON_L2_ERROR_VS_CUBLAS_DGEMV_REF Plot the
%   TRANSPOSED (y = alpha*A^T*x) L2-relative-error sweep from scripts/
%   benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh
%   (which itself builds and runs benchmarks/kernels/cuda/matrix/mv_
%   fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref.cu) -- the SAME five
%   "TOMOS" fp32-reconstruction-tier TRANSPOSE GEMV series as plot_gemv_
%   transpose_fp32_recon_l2_error_vs_cublas_ref.m (that file's own FP32-
%   reference sibling), but measured against a TRUE FP64 (cublasDgemv(
%   CUBLAS_OP_T)) reference instead of FP32 (cublasSgemv(CUBLAS_OP_T)).
%   TRANSPOSE counterpart of plot_gemv_fp32_recon_l2_error_vs_cublas_
%   dgemv_ref.m -- see that file's own header for the full derivation of
%   why FP64 matters here, not repeated below (identical reasoning, only
%   the direction differs):
%
%   TOMOS FP32                  = mv_fp32_transpose_atomiconly_scaled_
%                              mblock_vec4_stagevec
%   TOMOS FP32 / FP16           = mv_fp32_16_transpose_atomiconly_
%                              scaled_mblock_vec4_wrapborrow_dispatch
%   TOMOS FP32 / FP16 (fast)    = mv_fp32_16_transpose_atomiconly_
%                              scaled_mblock_fast_normal_wrapborrow_
%                              vec4_dispatch
%   TOMOS FP32 / BF16           = mv_fp32_bf16_transpose_atomiconly_
%                              scaled_mblock_vec4_rounded_dispatch
%   TOMOS FP32 / BF16 (truncated) = mv_fp32_bf16_transpose_atomiconly_
%                              scaled_mblock_vec4
%
%   Plus cuBLAS FP32 (dashed) -- cublasSgemv(CUBLAS_OP_T) on the SAME
%   A_native/x the TOMOS FP32 series above shares, measuring cuBLAS's OWN
%   FP32 implementation against the same TRUE FP64 reference.
%
%   Square sweep ONLY (M=K), CAPPED AT 24576 (not the FP32-reference
%   sibling's own 131072) -- see the .cu's own header for the memory
%   derivation. Y-AXIS IS LOG-SCALE.
%
%   The CSV is expected to have M/K columns plus *_l2_rel_err columns per
%   method -- see benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_
%   dgemv_ref.sh's own embedded parser for the exact schema. ("SKIPPED"
%   cells become NaN and leave a gap in that method's line.)
%
%   scale_tag (optional) identifies which "normal_scale" tier this CSV's
%   own data was generated at -- appended to the output filename. Omit it
%   (or pass '') for a single ad hoc run with no tier to distinguish.
%
%   Writes, under <repo>/build/output/error_dimension_scaling/gemv/fp32_compute_transpose/plots/:
%     plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref[_scale<scale_tag>].png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        scale_tag = '';
    end
    scale_tag = char(scale_tag);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemv', 'fp32_compute_transpose', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref:noCsv', ...
                ['No mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref.sh ' ...
                 'first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref:missingCsv', 'CSV not found: %s', ...
            csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemv', 'fp32_compute_transpose', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'tomos_fp32_l2_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_l2_rel_err'), 'style', '-v');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (fast)', 'values', to_double_col(T, 'tomos_fp32_fp16_fast_l2_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'tomos_fp32_bf16_l2_rel_err'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (truncated)', 'values', to_double_col(T, 'tomos_fp32_bf16_truncated_l2_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP32', 'values', to_double_col(T, 'cublas_fp32_l2_rel_err'), 'style', '--d');

    if isempty(scale_tag)
        out_base = fullfile(out_dir, 'plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref');
    else
        out_base = fullfile(out_dir, ['plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref_scale' scale_tag]);
    end

    plot_series(M, series, true(size(M)), 'Matrix dimensions (m = k)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mv_fp32_transpose_recon_l2_error_vs_cublas_dgemv_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_transpose_fp32_recon_l2_error_vs_cublas_dgemv_ref:emptyGroup', ...
            'No rows to plot -- skipping.');
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
