function plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref(csv_path)
%PLOT_GEMM_FP16_KERNELS_SCALE_SWEEP_L2_ERROR_VS_CUBLAS_REF
%   Plot the FP16 GEMM kernel-implementation comparison sweep from
%   scripts/benchmarks_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_
%   ref.sh (same binary as plot_gemm_fp16_kernels_l2_error_vs_cublas_
%   ref.m's own data, run a different way): a SINGLE FIXED SHAPE
%   (M=K=N=49152), swept across normal_scale instead of across matrix
%   dimensions -- the complementary axis to that sibling plot's own
%   N-sweep-at-fixed-scale=1.0.
%
%   Same three series as the N-sweep sibling plot (TOMOS FP16, cuBLAS
%   FP16, CUTLASS FP16), all run on the IDENTICAL native-rounded A/B data
%   -- see that .m file's own header for the full series description.
%
%   Data was generated with --normalized (B scaled by 1/sqrt(K)) -- keeps
%   C's own row-sum magnitude ~normal_scale regardless of K, avoiding the
%   K-dependent output-storage-overflow "inf" this __half-typed C tier is
%   prone to at a shape this large without it.
%
%   SCALES range is IDENTICAL to plot_gemm_fp32_16_round_scale_sweep_l2_
%   error_vs_cublas_ref.m's own (the variable-rounding sibling plot
%   family's own scale-sweep), so the two plot families are directly
%   comparable at the same scales.
%
%   Square shape ONLY (M=K=N=49152, fixed, not swept). X-AXIS
%   (normal_scale) IS LOG-SCALE. Y-AXIS IS LINEAR.
%
%   The CSV is expected to have a `scale` column (the swept normal_scale
%   value) plus M/K/N columns (constant across every row here) plus
%   *_frobenius_rel_err columns per series -- see benchmarks_gemm_fp16_
%   kernels_scale_sweep_l2_error_vs_cublas_ref.sh's own embedded parser
%   for the exact schema. ("SKIPPED" cells become NaN and leave a gap in
%   that series' line.)
%
%   Writes, under <repo>/build/output/error_rounding/gemm/fp16_compute/plots/:
%     plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_rounding', 'gemm', 'fp16_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref.sh ' ...
                 'first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_rounding', 'gemm', 'fp16_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    scale = to_double_col(T, 'scale');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col(T, 'tomos_fp16_frobenius_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_fp16_frobenius_rel_err'), 'style', '--d');
    i = i + 1; series(i) = struct('label', 'CUTLASS FP16', 'values', to_double_col(T, 'cutlass_fp16_frobenius_rel_err'), 'style', '--^');

    out_base = fullfile(out_dir, 'plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref');

    plot_series(scale, series, true(size(scale)), 'Magnitude scale M', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp16_kernels_scale_sweep_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.00 0.45 0.74;  % blue (TOMOS FP16)
        0.85 0.10 0.10;  % red (cuBLAS FP16)
        0.20 0.80 0.20;  % bright green (CUTLASS FP16)
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
    set(gca, 'XScale', 'log', 'YScale', 'linear', 'XTick', unique(x_sorted), ...
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
