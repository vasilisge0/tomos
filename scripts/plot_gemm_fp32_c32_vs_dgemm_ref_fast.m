function plot_gemm_fp32_c32_vs_dgemm_ref_fast(csv_path)
%PLOT_GEMM_FP32_C32_VS_DGEMM_REF_FAST Same CSV/sweep as plot_gemm_fp32_
%   c32_vs_dgemm_ref (its own header below has the full series writeup,
%   not repeated here), PLUS two UNCONDITIONAL fast_normal decode series
%   -- "TOMOS FP32/FP16 (fast)" and "TOMOS FP32/FP16 (workspace, fast)"
%   -- that the original function deliberately omits, to keep its own
%   output unchanged for any existing consumer. See fp32_16_cuda_
%   kernels.cu's own comments on mm_fp32_16_tiled_cpasync_dualbuf_fast_
%   normal / mm_fp32_16_gemm_materialize_midpoint_exact_fast_normal for
%   the UNSAFE-ON-ARBITRARY-DATA contract these two series share.
%
%   Frobenius-relative-error sweep from scripts/benchmarks_gemm_fp32_
%   compute32_l2_error_vs_cublas_dgemm_ref.sh (which builds and runs
%   benchmarks/kernels/cuda/matrix/mm_fp32_compute32_l2_error_vs_cublas_
%   dgemm_ref.cu). Y-axis is the FROBENIUS relative error against a
%   genuine FP64 (cublasDgemm, on A/B widened to double) reference -- NOT
%   an FP32 reference, unlike every *_recon_l2_error_vs_cublas_ref.cu
%   sibling in this codebase. See the .cu's own header for why: measured
%   against FP32, the "TOMOS FP32" series' own error stays near machine
%   epsilon (both share the same float rounding model), never showing
%   the real accumulated rounding error a large float GEMM reduction
%   produces. Measured against FP64, that error becomes a genuine,
%   non-trivial baseline that the FP16/BF16-encoded series' own
%   additional error can be compared against.
%
%   Four series, all measuring float-accumulating kernels (see the .cu's
%   own header for why none of these has an fp16/bf16-accumulator-
%   overflow or output-storage-overflow concern):
%
%   cuBLAS FP32      = cublasSgemm, on the SAME A/B every other series
%                       shares -- the industry-standard FP32 GEMM
%                       baseline, to check whether TOMOS FP32's own error
%                       is comparable to, better, or worse than cuBLAS's.
%   TOMOS FP32       = gemm_fp32_tiled_syncpipe4_bk8_warptile_nostride_
%                       snake_midpoint_exact -- A consumed directly, no
%                       16-bit encoding. Its own error here is the real
%                       float-GEMM accumulation error, not a sanity-check
%                       near-zero number.
%   TOMOS FP32/FP16  = mm_fp32_16_tiled_cpasync_dualbuf -- A packed via
%                       fp32_16_split (plain, round-to-nearest).
%   TOMOS FP32/BF16  = mm_fp32_bf16_syncpipe4_bk8_warptile_nostride_
%                       snake_midpoint_exact -- A packed via
%                       fp32_bf16_split (plain).
%
%   Square sweep ONLY (M=K=N). X-AXIS IS LOG-SCALE (matrix dimension).
%   Y-AXIS IS LOG-SCALE too -- unlike the fp32_16-vs-FP16 rounding-method
%   sweeps elsewhere in this codebase (whose 9-vs-10-bit-mantissa series
%   all stay within roughly one decade of each other), TOMOS FP32's own
%   error here can sit several decades below the FP16/BF16-encoded
%   series' own error, so a linear axis would flatten the native series
%   into an indistinguishable line at the bottom.
%
%   The CSV is expected to have M/K/N columns plus *_frobenius_rel_err
%   columns per series -- see benchmarks_gemm_fp32_compute32_l2_error_vs_
%   cublas_dgemm_ref.sh's own embedded parser for the exact schema.
%   ("SKIPPED" cells become NaN and leave a gap in that series' line --
%   expected at the largest shapes, where the FP64 reference's own extra
%   double-typed buffers can exceed device memory; see the .cu's own
%   header.)
%
%   Writes, under <repo>/build/output/error_dimension_scaling/gemm/fp32_compute/plots/:
%     plot_gemm_fp32_c32_vs_dgemm_ref_fast.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemm', 'fp32_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp32_c32_vs_dgemm_ref_fast:noCsv', ...
                ['No mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp32_compute32_l2_error_vs_cublas_dgemm_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp32_c32_vs_dgemm_ref_fast:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemm', 'fp32_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    N = to_double_col(T, 'N');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'cuBLAS FP32', 'values', to_double_col(T, 'cublas_fp32_frobenius_rel_err'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'tomos_fp32_frobenius_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_frobenius_rel_err'), 'style', '-s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16 (fast)', 'values', to_double_col(T, 'tomos_fp32_fp16_fast_frobenius_rel_err'), 'style', ':s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16 (truncated)', 'values', to_double_col(T, 'tomos_fp32_fp16_truncated_frobenius_rel_err'), 'style', '--s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_fp16_workspace_frobenius_rel_err'), 'style', '-.s');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/FP16 (workspace, fast)', 'values', to_double_col(T, 'tomos_fp32_fp16_workspace_fast_frobenius_rel_err'), 'style', ':d');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/BF16 (truncated)', 'values', to_double_col(T, 'tomos_fp32_bf16_truncated_frobenius_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32/BF16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_bf16_workspace_frobenius_rel_err'), 'style', '--^');

    out_base = fullfile(out_dir, 'plot_gemm_fp32_c32_vs_dgemm_ref_fast');

    plot_series(N, series, true(size(N)), 'Matrix dimension, N (M = K = N)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp32_c32_vs_dgemm_ref_fast:missingColumn', 'CSV is missing expected column "%s".', name);
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
    files = dir(fullfile(dir_path, 'mm_fp32_compute32_l2_error_vs_cublas_dgemm_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp32_c32_vs_dgemm_ref_fast:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % ORIGINAL per-series fixed palette -- disabled below in favor of a
    % scanned, group-by-identical-curve palette (see kGroupPalette): this
    % benchmark's own data currently has several series producing
    % bit-identical Frobenius relative error at every N (a real, unresolved
    % correctness question, not a plotting choice -- see the .cu's own
    % header), and coloring them identically makes that visible instead of
    % hiding it behind 7 distinct, misleadingly-independent-looking colors.
    % kPalette = [
    %     0.30 0.30 0.30;  % dark grey (cuBLAS FP32 -- industry-standard baseline)
    %     0.55 0.55 0.55;  % grey (TOMOS FP32 -- no 16-bit encoding)
    %     0.00 0.45 0.74;  % blue (TOMOS FP32/FP16)
    %     0.00 0.70 0.90;  % light blue (TOMOS FP32/FP16 truncated)
    %     0.00 0.30 0.55;  % dark blue (TOMOS FP32/FP16 workspace)
    %     0.90 0.55 0.10;  % orange (TOMOS FP32/BF16 truncated)
    %     0.70 0.35 0.05;  % dark orange (TOMOS FP32/BF16 workspace)
    % ];
    % colors = kPalette(1:numel(series), :);

    % Scan every series' own (masked, x-sorted) curve and group series
    % whose values match at every point (within a tight relative
    % tolerance, to allow for text-roundtrip rounding in the CSV) into the
    % same color group -- distinct series with distinct data still get
    % distinct colors, but curves that are actually identical now render
    % as visually identical (same color) instead of merely overlapping.
    n_series = numel(series);
    y_all = NaN(n_series, numel(x_sorted));
    for i = 1:n_series
        y = series(i).values(mask);
        y_all(i, :) = y(order);
    end
    group_id = zeros(1, n_series);
    next_group = 0;
    for i = 1:n_series
        if group_id(i) ~= 0
            continue
        end
        next_group = next_group + 1;
        group_id(i) = next_group;
        for j = (i + 1):n_series
            if group_id(j) == 0 && curves_equal(y_all(i, :), y_all(j, :))
                group_id(j) = next_group;
            end
        end
    end
    num_groups = next_group;

    fprintf('Curve scan: %d series collapse into %d unique-color group(s):\n', n_series, num_groups);
    for g = 1:num_groups
        members = {series(group_id == g).label};
        fprintf('  group %d: %s\n', g, strjoin(members, ', '));
    end

    kGroupPalette = [
        0.30 0.30 0.30;  % dark grey
        0.00 0.45 0.74;  % blue
        0.90 0.55 0.10;  % orange
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.85 0.10 0.10;  % red
        0.55 0.55 0.55;  % light grey
    ];
    if num_groups <= size(kGroupPalette, 1)
        group_colors = kGroupPalette(1:num_groups, :);
    else
        group_colors = [kGroupPalette; lines(num_groups - size(kGroupPalette, 1))];
    end
    colors = group_colors(group_id, :);

    for i = 1:n_series
        plot(x_sorted, y_all(i, :), series(i).style, 'LineWidth', 4.5, 'MarkerSize', 6, ...
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

function eq = curves_equal(a, b)
%CURVES_EQUAL True if two same-length curves match at every point, within
%   a tight relative tolerance (to allow for CSV text-roundtrip rounding),
%   treating NaN ("SKIPPED") as equal only when both curves are NaN at
%   that same point.
    nan_a = isnan(a);
    nan_b = isnan(b);
    if ~isequal(nan_a, nan_b)
        eq = false;
        return
    end
    valid = ~nan_a;
    if ~any(valid)
        eq = true;
        return
    end
    av = a(valid);
    bv = b(valid);
    tol = 1e-9 * max(abs(av), abs(bv));
    tol(tol == 0) = 1e-12;
    eq = all(abs(av - bv) <= tol);
end
