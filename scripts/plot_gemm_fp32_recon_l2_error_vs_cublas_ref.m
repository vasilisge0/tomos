function plot_gemm_fp32_recon_l2_error_vs_cublas_ref(csv_path, scale_tag)
%PLOT_GEMM_FP32_RECON_L2_ERROR_VS_CUBLAS_REF Plot the full-fp32-
%   reconstruction-tier GEMM Frobenius-relative-error sweep from
%   scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh (which
%   itself builds and runs benchmarks/kernels/cuda/matrix/mm_fp32_recon_
%   l2_error_vs_cublas_ref.cu) -- the accuracy-vs-N companion to
%   plot_gemm_format_32_16_compute_32.m (which plots GFlop/s for the same
%   six series over the same square sweep). Named "l2_error" for
%   consistency with every sibling *_l2_error_vs_cublas_ref.m file, even
%   though the y-axis here is the FROBENIUS relative error (per explicit
%   user confirmation: GEMM error uses Frobenius, not L2, since the
%   output is a matrix) -- not throughput:
%
%   TOMOS FP32                  = gemm_fp32_tiled_syncpipe4_bk8_warptile_
%                                  nostride_snake_midpoint_exact
%   TOMOS FP32 / FP16            = mm_fp32_16_tiled_cpasync_dualbuf
%   TOMOS FP32 / FP16 (truncated) = mm_fp32_16_tiled_cpasync_dualbuf_truncated
%   TOMOS FP32 / FP16 (workspace) = mm_fp32_16_gemm_materialize_midpoint_exact
%   TOMOS FP32 / BF16 (truncated) = mm_fp32_bf16_syncpipe4_bk8_warptile_
%                                  nostride_snake_midpoint_exact
%   TOMOS FP32 / BF16 (workspace) = mm_fp32_bf16_gemm_materialize_rounded
%
%   See plot_gemm_format_32_16_compute_32.m's own header/inline comments
%   for the full derivation/promotion history of each of these six
%   dispatchers -- not repeated here, since it's identical (this plot
%   analyzes the SAME promoted kernels, just a different metric).
%
%   TOMOS FP32's own error is a sanity check on this file's methodology
%   (plain float32 throughout, cublasSgemm reference on the SAME
%   underlying float32 A array) -- expected near machine epsilon
%   regardless of normal_scale.
%
%   No cuBLAS/CUTLASS series here (unlike the GFlop/s plot): cuBLAS/
%   cublasSgemm IS this file's own reference.
%
%   Square sweep ONLY (M=K=N) -- scripts/benchmarks_gemm_fp32_recon_l2_
%   error_vs_cublas_ref.sh has no rectangular sweep, matching its own
%   GFlop/s sibling driver. Y-AXIS IS LOG-SCALE: Frobenius relative error
%   spans orders of magnitude across these methods/shapes, unlike GFlop/s.
%
%   The CSV is expected to have M/K/N columns plus *_frobenius_rel_err
%   columns per method -- see benchmarks_gemm_fp32_recon_l2_error_vs_
%   cublas_ref.sh's own embedded parser for the exact schema. ("SKIPPED"
%   cells, from a shape a method ran out of device memory for, become NaN
%   and simply leave a gap in that method's line.)
%
%   scale_tag (optional) identifies which "normal_scale" tier this CSV's
%   own data was generated at -- appended to the output filename and
%   shown in the plot's own title. Omit it (or pass '') for a single ad
%   hoc run with no tier to distinguish.
%
%   Writes, under <repo>/build/benchmarks/output/plots/error/gemm32/:
%     plot_gemm_fp32_recon_l2_error_vs_cublas_ref[_scale<scale_tag>].png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        scale_tag = '';
    end
    scale_tag = char(scale_tag);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mm_fp32_recon_l2_error_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp32_recon_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp32_recon_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp32_recon_l2_error_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp32_recon_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots', 'error', 'gemm32');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    N = to_double_col(T, 'N');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'tomos_fp32_frobenius_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_frobenius_rel_err'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (truncated)', 'values', to_double_col(T, 'tomos_fp32_fp16_truncated_frobenius_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_fp16_workspace_frobenius_rel_err'), 'style', '-<');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (truncated)', 'values', to_double_col(T, 'tomos_fp32_bf16_truncated_frobenius_rel_err'), 'style', '-+');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_bf16_workspace_frobenius_rel_err'), 'style', '-*');

    if isempty(scale_tag)
        out_base = fullfile(out_dir, 'plot_gemm_fp32_recon_l2_error_vs_cublas_ref');
        plot_title = '';
    else
        out_base = fullfile(out_dir, ['plot_gemm_fp32_recon_l2_error_vs_cublas_ref_scale' scale_tag]);
        plot_title = sprintf('Normal-range magnitude scale = %s (data in [-%s, %s])', ...
            scale_tag, scale_tag, scale_tag);
    end

    plot_series(N, series, true(size(N)), 'Matrix dimensions (M = K = N)', out_base, plot_title);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp32_recon_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp32_recon_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base, plot_title)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp32_recon_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
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
        0.30 0.30 0.30;  % grey
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
    ylabel('Frobenius relative error vs FP32', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    if nargin >= 6 && ~isempty(plot_title)
        title(plot_title, 'Interpreter', 'none', 'FontSize', 20);
    end
    grid on;

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
