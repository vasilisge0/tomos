function plot_lsqr_performance(csv_path, out_dir)
%PLOT_LSQR_PERFORMANCE Grouped bar chart of LSQR solve performance per
%   precision/preconditioning configuration, one bar per test matrix.
%   Driven by scripts/run_lsqr_precision_performance_compare.sh, which runs
%   the same five configurations as scripts/run_lsqr_precision_compare.sh
%   (the convergence-curve counterpart of this plot) and writes one CSV row
%   per (matrix, configuration): median/min/max solve time over repeated
%   runs, iteration count, and time per iteration.
%
%   Two stacked panels, same x axis (configurations, grouped; bars within a
%   group = matrices):
%     top    -- total solve time (median, error bars = min..max over the
%               repeats), each bar annotated with its iteration count. A
%               config that needs more iterations to reach the same
%               tolerance looks slower here even if each matvec is fast.
%     bottom -- time per iteration (median solve time / iterations), which
%               isolates per-matvec cost from iteration-count differences.
%
%   csv_path: lsqr_performance.csv written by the driver script.
%   out_dir:  where to write the plot (default: <repo>/build/output/lsqr/
%             performance/plots -- deliberately NOT the convergence-plot
%             directory, so this never overwrites lsqr_residual_history.*).
%
%   A (matrix, configuration) pair missing from the CSV (e.g. every run of
%   it failed) is drawn as an empty slot rather than aborting the plot.
%
%   Writes: <out_dir>/lsqr_performance_barplot.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);
    if nargin < 2 || isempty(out_dir)
        out_dir = fullfile(repo_root, 'build', 'output', 'lsqr', 'performance', 'plots');
    end
    % Accept MATLAB string or char paths alike ([out_base '.png'] below
    % would otherwise build a string ARRAY from a string out_dir).
    csv_path = char(csv_path);
    out_dir = char(out_dir);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    if ~isfile(csv_path)
        error('plot_lsqr_performance:missingFile', 'CSV not found: %s', csv_path);
    end

    T = readtable(csv_path, 'TextType', 'string');
    if isempty(T)
        error('plot_lsqr_performance:noData', 'CSV %s has no rows.', csv_path);
    end

    % Fixed configuration order/labels, matching plot_lsqr_residual_history's
    % legend labels from run_lsqr_precision_compare.sh (line breaks only so
    % the longer names fit under their bars).
    keys = ["fp32_no_precond"; "fp32_precond"; "fp16_explicit"; "fp16_implicit"; "fp16_implicit_fast"];
    % Each label is a cell array of lines, drawn with text() below the
    % bottom axes -- neither an embedded newline nor a nested cell is
    % accepted by XTickLabel, which would split/reject them.
    tick_labels = { ...
        {'FP32', '(no precond)'}, ...
        {'FP32', '(precond)'}, ...
        {'TOMOS FP16', '(native, explicit refine)'}, ...
        {'TOMOS FP16', '(implicit, safe-decode)'}, ...
        {'TOMOS FP16', '(implicit, fast-decode)'}};

    matrices = unique(T.matrix_label, 'stable');
    nC = numel(keys);
    nM = numel(matrices);

    med = nan(nC, nM);
    lo = nan(nC, nM);
    hi = nan(nC, nM);
    iters = nan(nC, nM);
    per_iter = nan(nC, nM);
    for j = 1:nM
        for i = 1:nC
            idx = T.matrix_label == matrices(j) & T.config_key == keys(i);
            if any(idx)
                r = find(idx, 1, 'first');
                med(i, j) = T.solve_ms_median(r);
                lo(i, j) = T.solve_ms_min(r);
                hi(i, j) = T.solve_ms_max(r);
                iters(i, j) = T.iterations(r);
                per_iter(i, j) = T.ms_per_iter(r);
            end
        end
    end

    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
    ];
    if nM <= size(kPalette, 1)
        colors = kPalette(1:nM, :);
    else
        colors = [kPalette; lines(nM - size(kPalette, 1))];
    end

    kAxisFontSize = 22;
    kLegendFontSize = 20;
    kAnnotFontSize = 16;
    kTickLabelFontSize = 18;

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 1100]);
    % Explicit axes positions (not tiledlayout) so there is guaranteed room
    % under the bottom axes for the two-line configuration labels.

    % ---- Top: total solve time, annotated with iteration counts ----
    ax1 = axes('Parent', fig, 'Position', [0.09 0.58 0.89 0.38]);
    b1 = bar(ax1, med, 'grouped');
    hold(ax1, 'on');
    for j = 1:nM
        b1(j).FaceColor = colors(j, :);
        x = b1(j).XEndPoints;
        errorbar(ax1, x, med(:, j), med(:, j) - lo(:, j), hi(:, j) - med(:, j), ...
            'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);
        for i = 1:nC
            if ~isnan(med(i, j))
                text(ax1, x(i), hi(i, j), sprintf(' %d it', iters(i, j)), ...
                    'Rotation', 90, 'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'middle', 'FontSize', kAnnotFontSize, ...
                    'Interpreter', 'none');
            end
        end
    end
    hold(ax1, 'off');
    set(ax1, 'FontSize', kAxisFontSize, 'LineWidth', 2, 'XTick', 1:nC, 'XTickLabel', {}, ...
        'YGrid', 'on', 'Box', 'off');
    ylabel(ax1, 'Solve time (ms)', 'FontSize', kAxisFontSize);
    ylim(ax1, [0, max(hi(:), [], 'omitnan') * 1.25]);
    legend(ax1, cellstr(matrices), 'Location', 'northeast', 'Interpreter', 'none', ...
        'FontSize', kLegendFontSize);

    % ---- Bottom: time per iteration ----
    ax2 = axes('Parent', fig, 'Position', [0.09 0.17 0.89 0.33]);
    b2 = bar(ax2, per_iter, 'grouped');
    for j = 1:nM
        b2(j).FaceColor = colors(j, :);
    end
    set(ax2, 'FontSize', kAxisFontSize, 'LineWidth', 2, 'XTick', 1:nC, ...
        'XTickLabel', {}, 'YGrid', 'on', 'Box', 'off');
    ylabel(ax2, 'Time per iteration (ms)', 'FontSize', kAxisFontSize);
    yl = ylim(ax2);
    for i = 1:nC
        text(ax2, i, yl(1), tick_labels{i}, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', 'FontSize', kTickLabelFontSize, ...
            'Interpreter', 'none', 'Clipping', 'off');
    end

    out_base = fullfile(out_dir, 'lsqr_performance_barplot');
    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d matrices x %d configurations)\n', ...
        out_base, out_base, nM, nC);
end
