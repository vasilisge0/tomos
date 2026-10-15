function plot_lsqr_speedup(csv_path, out_dir)
%PLOT_LSQR_SPEEDUP Grouped bar chart of the speedup of each mixed-precision
%   LSQR configuration over the preconditioned FP32 baseline, one bar per
%   test matrix.
%   Reads the lsqr_performance.csv written by scripts/
%   run_lsqr_precision_performance_compare.sh (one row per matrix x
%   configuration, median solve time over repeated runs) and, for each
%   matrix, plots
%
%       speedup = solve_ms_median(FP32 precond) / solve_ms_median(method)
%
%   for the three mixed-precision configurations (all of which use the
%   preconditioner):
%     TOMOS FP16 (native, explicit refine)   -- fp16_explicit
%     TOMOS FP16 (implicit, safe-decode)     -- fp16_implicit
%     TOMOS FP16 (implicit, fast-decode)     -- fp16_implicit_fast
%   The non-preconditioned FP32 configuration (fp32_no_precond) is
%   deliberately NOT plotted: the baseline here is FP32 WITH the
%   preconditioner, so every bar isolates the effect of the mixed-precision
%   storage/compute strategy alone. A bar above the dashed 1x line is
%   faster than the baseline; below it, slower.
%
%   The ratio is of TOTAL solve time, so it includes any difference in
%   iteration count (a lower-precision configuration that needs more
%   iterations to reach the same tolerance can land below 1x even when each
%   matvec is faster). The per-iteration comparison is the bottom panel of
%   the performance bar chart (plot_lsqr_performance.m).
%
%   csv_path: lsqr_performance.csv written by the driver script.
%   out_dir:  where to write the plot (default: <repo>/build/output/lsqr/
%             performance/plots, next to lsqr_performance_barplot.* --
%             different file name, so neither overwrites the other, and the
%             convergence plots in build/output/lsqr/error*/ are untouched).
%
%   A matrix whose FP32-precond baseline row is missing is skipped with a
%   warning (its bars are left empty); a missing method row is likewise
%   left empty.
%
%   Writes: <out_dir>/lsqr_speedup_barplot.png / .fig

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
        error('plot_lsqr_speedup:missingFile', 'CSV not found: %s', csv_path);
    end

    T = readtable(csv_path, 'TextType', 'string');
    if isempty(T)
        error('plot_lsqr_speedup:noData', 'CSV %s has no rows.', csv_path);
    end

    base_key = "fp32_precond";
    keys = ["fp16_explicit"; "fp16_implicit"; "fp16_implicit_fast"];
    % Each label is a cell array of lines, drawn with text() below the axes
    % -- neither an embedded newline nor a nested cell is accepted by
    % XTickLabel, which would split/reject them.
    tick_labels = { ...
        {'TOMOS FP16', '(native, explicit refine)'}, ...
        {'TOMOS FP16', '(implicit, safe-decode)'}, ...
        {'TOMOS FP16', '(implicit, fast-decode)'}};

    matrices = unique(T.matrix_label, 'stable');
    nC = numel(keys);
    nM = numel(matrices);

    speedup = nan(nC, nM);
    for j = 1:nM
        rb = find(T.matrix_label == matrices(j) & T.config_key == base_key, 1, 'first');
        if isempty(rb)
            warning('plot_lsqr_speedup:noBaseline', ...
                'No %s row for matrix "%s"; its bars are left empty.', base_key, matrices(j));
            continue
        end
        base_ms = T.solve_ms_median(rb);
        for i = 1:nC
            r = find(T.matrix_label == matrices(j) & T.config_key == keys(i), 1, 'first');
            if ~isempty(r)
                speedup(i, j) = base_ms / T.solve_ms_median(r);
            end
        end
    end
    if all(isnan(speedup(:)))
        error('plot_lsqr_speedup:noData', ...
            'No (matrix, method) pair had both a method row and an %s baseline row.', base_key);
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
    kAnnotFontSize = 18;
    kTickLabelFontSize = 18;

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 800]);
    % Explicit axes position so there is guaranteed room under the axes for
    % the two-line method labels.
    ax = axes('Parent', fig, 'Position', [0.09 0.22 0.89 0.72]);
    b = bar(ax, speedup, 'grouped');
    hold(ax, 'on');
    for j = 1:nM
        b(j).FaceColor = colors(j, :);
        x = b(j).XEndPoints;
        for i = 1:nC
            if ~isnan(speedup(i, j))
                text(ax, x(i), speedup(i, j), sprintf('%.2fx', speedup(i, j)), ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', kAnnotFontSize, 'Interpreter', 'none');
            end
        end
    end
    % Parity with the FP32-precond baseline.
    yline(ax, 1, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 2.5);
    hold(ax, 'off');

    set(ax, 'FontSize', kAxisFontSize, 'LineWidth', 2, 'XTick', 1:nC, ...
        'XTickLabel', {}, 'YGrid', 'on', 'Box', 'off');
    ylabel(ax, 'Speedup over FP32 (precond)', 'FontSize', kAxisFontSize);
    ylim(ax, [0, max(max(speedup(:), [], 'omitnan'), 1) * 1.2]);
    legend(ax, b, cellstr(matrices), 'Location', 'northeast', 'Interpreter', 'none', ...
        'FontSize', kLegendFontSize);

    yl = ylim(ax);
    for i = 1:nC
        text(ax, i, yl(1), tick_labels{i}, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', 'FontSize', kTickLabelFontSize, ...
            'Interpreter', 'none', 'Clipping', 'off');
    end

    out_base = fullfile(out_dir, 'lsqr_speedup_barplot');
    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d matrices x %d methods)\n', ...
        out_base, out_base, nM, nC);
end
