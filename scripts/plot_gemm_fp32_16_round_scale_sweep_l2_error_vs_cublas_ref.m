function plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref(csv_path)
%PLOT_GEMM_FP32_16_ROUND_SCALE_SWEEP_L2_ERROR_VS_CUBLAS_REF
%   Plot the fp32_16 encode-side rounding-precision sweep from scripts/
%   benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_
%   cublas_ref.sh (same binary as plot_gemm_fp32_16_variable_rounding_l2_
%   error_vs_cublas_ref.m's own data, run a different way): a SINGLE
%   FIXED SHAPE (M=K=N=49152), swept across normal_scale instead of
%   across matrix dimensions -- the complementary axis to that sibling
%   plot's own N-sweep-at-fixed-scale=1.0.
%
%   NOTE ON THIS FILE'S OWN NAME: the fully-consistent name (matching
%   this codebase's own *_variable_rounding_scale_sweep_* convention for
%   the .cu/.sh siblings) is 70 characters, which exceeds MATLAB's own
%   63-character function/file name limit -- MATLAB truncates the call
%   silently at the command line but the file on disk keeps its full
%   name, so the truncated call resolves to nothing ("Unrecognized
%   function or variable"). This file is deliberately named shorter
%   ("round" in place of "variable_rounding") purely to fit under that
%   limit; the .cu/.sh files keep their own full, unabbreviated names,
%   since bash/C++ have no such constraint.
%
%   Why this axis: fp32_16's own encoding has three branches (normal /
%   extended / degraded -- see fp32_16_pack.hpp's own header), and only
%   large enough |A[i,j]| values ever cross out of "normal" into
%   "extended"/"degraded". The N-sweep sibling plot never varies
%   normal_scale, so it can only show rounding-precision loss WITHIN the
%   normal branch; this plot's x-axis instead pushes more of A's own
%   elements across those branch boundaries as normal_scale grows.
%
%   Same six series as the N-sweep sibling plot, all measuring the SAME
%   real, unmodified native tensor-core kernel (gemm_fp16_mma_ptx_wide_
%   bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp, "TOMOS
%   FP16") -- see that .m file's own header for the full series
%   description and the mantissa-width-vs-rounding-method 2x2 design
%   (TOMOS FP16 (native), TOMOS FP16 (trunc10), 13/9/5/1 sticky bits,
%   truncation).
%
%   Data was generated with --normalized (B scaled by 1/sqrt(K)) --
%   see the .sh's own header for why: it keeps C's own row-sum magnitude
%   ~normal_scale regardless of K, avoiding the K-dependent output-
%   storage-ceiling "inf" bug this tier is prone to at large, fixed K,
%   without perturbing which encoding branch A's own elements fall into
%   (normalization only ever rescales B).
%
%   Square shape ONLY (M=K=N=49152, fixed, not swept). X-AXIS
%   (normal_scale) IS LOG-SCALE, same convention as this codebase's
%   other scale-tagged plots' own x-axes. Y-AXIS IS LINEAR -- the
%   observed Frobenius relative error across these rounding configs/
%   scales stays within roughly one decade, not the multi-decade range a
%   log y-axis is meant for, so linear reads the actual magnitude of the
%   differences more directly than a log axis would.
%
%   The CSV is expected to have a `scale` column (the swept normal_scale
%   value) plus M/K/N columns (constant across every row here) plus
%   *_frobenius_rel_err columns per series -- see benchmarks_gemm_
%   fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref.sh's own
%   embedded parser for the exact schema. ("SKIPPED" cells become NaN and
%   leave a gap in that series' line.)
%
%   Writes, under <repo>/build/output/error_rounding/gemm/fp16_compute/plots/:
%     plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_rounding', 'gemm', 'fp16_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref.sh ' ...
                 'first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref:missingCsv', ...
            'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_rounding', 'gemm', 'fp16_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    scale = to_double_col(T, 'scale');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 10, Sticky bits: 12 (FP32 to FP16)', 'values', to_double_col(T, 'tomos_fp16_native_frobenius_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 10, Sticky bits: 0 (truncation)', 'values', to_double_col(T, 'tomos_fp16_trunc10_frobenius_rel_err'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, Sticky bits: 13 (TOMOS FP32 / FP16)', 'values', to_double_col(T, 'tomos_fp32_fp16_round_13_frobenius_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, sticky bits: 5', 'values', to_double_col(T, 'tomos_fp32_fp16_round_05_frobenius_rel_err'), 'style', '--s');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, sticky bits: 1', 'values', to_double_col(T, 'tomos_fp32_fp16_round_01_frobenius_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, sticky bits: 0', 'values', to_double_col(T, 'tomos_fp32_fp16_trunc_frobenius_rel_err'), 'style', '-*');

    out_base = fullfile(out_dir, 'plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref');

    plot_series(scale, series, true(size(scale)), 'Magnitude scale M', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp32_16_variable_rounding_scale_sweep_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp32_16_round_scale_sweep_l2_error_vs_cublas_ref:emptyGroup', ...
            'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.55 0.55 0.55;  % grey (native fp16 reference, 10-bit rounded)
        0.30 0.30 0.30;  % dark grey (10-bit truncated reference)
        0.00 0.45 0.74;  % blue (13 -- exact match to real encoder)
        0.20 0.80 0.20;  % bright green
        0.90 0.55 0.10;  % orange
        0.85 0.10 0.10;  % red (truncation, 9-bit)
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
    set(gca, 'XScale', 'log', 'YScale', 'linear', 'XTick', unique(x_sorted), ...
        'FontSize', kTickFontSize, 'XTickLabelRotation', 40, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('Frobenius relative error', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 2, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
