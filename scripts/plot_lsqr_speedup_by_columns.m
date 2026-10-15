function plot_lsqr_speedup_by_columns(csv_path, out_dir, out_name, cond_labels)
%PLOT_LSQR_SPEEDUP_BY_COLUMNS One grouped bar chart of the speedup of each
%   mixed-precision LSQR configuration over preconditioned FP32, with the
%   bars GROUPED PER NUMBER OF COLUMNS.
%   Reads the lsqr_speedup_by_columns.csv written by scripts/
%   run_lsqr_precision_speedup_by_columns.sh (one row per columns x
%   condition number x configuration, median solve time over repeated runs)
%   and, for every (columns, condition number), plots
%
%       speedup = solve_ms_median(FP32 precond) / solve_ms_median(method)
%
%   for the three mixed-precision configurations:
%     TOMOS FP32 / FP16 (explicit)             -- fp16_explicit
%     TOMOS FP32 / FP16 (implicit)             -- fp16_implicit
%     TOMOS FP32 / FP16 (implicit, fast-decode) -- fp16_implicit_fast
%   Layout: one group per column count (x axis); inside a group, one bar
%   per (method, condition number). Colour identifies the method; the
%   lighter shade of a method's colour is the higher-index (later in the CSV)
%   condition number -- with the driver script's order, the solid bar is
%   cond=1000 and the lighter one cond=500. A dashed line marks 1x (parity
%   with FP32 + preconditioner): above it is faster, below it slower.
%
%   The ratio is of TOTAL solve time, so it includes iteration-count
%   differences between configurations (see plot_lsqr_speedup.m's header).
%
%   csv_path: lsqr_speedup_by_columns.csv written by the driver script.
%   out_dir:  where to write the plot (default: <repo>/build/output/lsqr/
%             speedup_by_columns).
%   out_name: output file base name (default 'lsqr_speedup_by_columns_
%             barplot'); lets several figures from the same CSV coexist.
%   cond_labels: cell/string array of cond_label values to draw, in order
%             (default: every label in the CSV, in file order). If given,
%             EVERY listed label must have rows in the CSV, otherwise this
%             errors -- so a figure is never silently drawn with a
%             condition missing. The driver script uses it to draw
%             {'cond=1000','cond=500'} and {'cond=1000','cond=1000-normalized'}
%             as two separate figures from one CSV. A label ending in
%             '-normalized' is shown in the legend as "<base> normalized";
%             the plain one is shown as just "<base>" (the two are told
%             apart by presence/absence of that suffix alone).
%
%   A (columns, condition number) whose FP32-precond baseline row is missing
%   is skipped with a warning, and a missing method row is left as an empty
%   slot -- a partial (or interrupted) run still plots what completed.
%
%   Writes: <out_dir>/<out_name>.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);
    if nargin < 2 || isempty(out_dir)
        out_dir = fullfile(repo_root, 'build', 'output', 'lsqr', 'speedup_by_columns');
    end
    % Accept MATLAB string or char paths alike ([out_base '.png'] below
    % would otherwise build a string ARRAY from a string out_dir).
    if nargin < 3 || isempty(out_name)
        out_name = 'lsqr_speedup_by_columns_barplot';
    end
    if nargin < 4
        cond_labels = [];
    end
    csv_path = char(csv_path);
    out_dir = char(out_dir);
    out_name = char(out_name);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    if ~isfile(csv_path)
        error('plot_lsqr_speedup_by_columns:missingFile', 'CSV not found: %s', csv_path);
    end

    T = readtable(csv_path, 'TextType', 'string');
    if isempty(T)
        error('plot_lsqr_speedup_by_columns:noData', 'CSV %s has no rows.', csv_path);
    end

    base_key = "fp32_precond";
    keys = ["fp16_explicit"; "fp16_implicit"; "fp16_implicit_fast"];
    method_labels = {'TOMOS FP32 / FP16 (explicit)', ...
                     'TOMOS FP32 / FP16 (implicit)', ...
                     'TOMOS FP32 / FP16 (implicit, fast-decode)'};

    cols = unique(T.columns, 'sorted');
    if isempty(cond_labels)
        conds = unique(T.cond_label, 'stable');
    else
        conds = string(cond_labels(:));
        missing = conds(~ismember(conds, T.cond_label));
        if ~isempty(missing)
            error('plot_lsqr_speedup_by_columns:missingCondition', ...
                'Condition label(s) not in %s: %s', csv_path, strjoin(missing, ', '));
        end
    end
    nG = numel(cols);
    nK = numel(keys);
    nCond = numel(conds);
    nB = nK * nCond;

    % speedup(g, b): group g = column count, bar b = (method k, cond c) with
    % b = (k-1)*nCond + c, so a method's conditions sit side by side.
    % iters(g, b) holds that same (columns, cond, method) row's own
    % iteration count, purely for the annotation below -- it plays no part
    % in the speedup ratio itself (that stays solve-time-based, unchanged).
    speedup = nan(nG, nB);
    iters = nan(nG, nB);
    for g = 1:nG
        for c = 1:nCond
            sel = T.columns == cols(g) & T.cond_label == conds(c);
            rb = find(sel & T.config_key == base_key, 1, 'first');
            if isempty(rb)
                warning('plot_lsqr_speedup_by_columns:noBaseline', ...
                    'No %s row for %d columns, %s; those bars are left empty.', ...
                    base_key, cols(g), conds(c));
                continue
            end
            base_ms = T.solve_ms_median(rb);
            for k = 1:nK
                r = find(sel & T.config_key == keys(k), 1, 'first');
                if ~isempty(r)
                    speedup(g, (k - 1) * nCond + c) = base_ms / T.solve_ms_median(r);
                    iters(g, (k - 1) * nCond + c) = T.iterations(r);
                end
            end
        end
    end
    if all(isnan(speedup(:)))
        error('plot_lsqr_speedup_by_columns:noData', ...
            'No (columns, condition number, method) had both a method row and an %s baseline row.', base_key);
    end

    % One base colour per METHOD; later condition numbers get progressively
    % lighter shades of it (mixed toward white).
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
    ];
    % Legend text for each condition (see cond_labels above). The two
    % cond=1000 variants are told apart by presence/absence of the
    % "normalized" suffix alone -- no "unnormalized" counterpart (that
    % word made the leftmost legend column's combined text long enough
    % to clip against the figure's left edge in lsqr_speedup_by_columns_
    % normalized_barplot.png; removing it is the fix, not a cosmetic
    % choice).
    cond_display = cell(1, nCond);
    for c = 1:nCond
        if endsWith(conds(c), "-normalized")
            cond_display{c} = char(strrep(conds(c), "-normalized", " normalized"));
        else
            cond_display{c} = char(conds(c));
        end
    end

    bar_colors = zeros(nB, 3);
    legend_labels = cell(1, nB);
    for k = 1:nK
        for c = 1:nCond
            b = (k - 1) * nCond + c;
            lighten = 0.6 * (c - 1) / max(nCond - 1, 1);
            bar_colors(b, :) = kPalette(k, :) + (1 - kPalette(k, :)) * lighten;
            legend_labels{b} = sprintf('%s, %s', method_labels{k}, cond_display{c});
        end
    end

    kAxisFontSize = 28;
    % Longer legend entries (e.g. "..., cond=1000 unnormalized") need a smaller font to fit 3 columns.
    kLegendFontSize = 24;
    if max(cellfun(@numel, legend_labels)) > 50
        kLegendFontSize = 20;
    end
    kAnnotFontSize = 22;

    % Width 2200, not 1800 -- at 3 columns x 2 rows, the longest legend
    % entries (e.g. "TOMOS FP32 / FP16 (implicit, fast-decode), cond=1000
    % normalized") need more total width than 1800px even at the smaller
    % (20pt) font, so the centered legend row overflowed the figure on
    % both sides -- invisibly on the right (white background), visibly
    % clipping the leftmost column's own left edge ("TOMOS " truncated).
    fig = figure('Visible', 'off', 'Position', [100, 100, 2200, 950]);
    % Explicit axes position so the legend (outside, below -- same convention
    % as plot_gemv_fp32_16_pure_half_vs_cublas_ref.m and the other
    % performance plots in this codebase) has guaranteed room. Legend
    % entries are ordered method-major, so NumColumns = #methods puts each
    % method in its own legend column (one row per condition number).
    ax = axes('Parent', fig, 'Position', [0.07 0.34 0.91 0.61]);
    hb = bar(ax, speedup, 'grouped');
    hold(ax, 'on');
    for b = 1:nB
        hb(b).FaceColor = bar_colors(b, :);
        x = hb(b).XEndPoints;
        for g = 1:nG
            if ~isnan(speedup(g, b))
                % Two-line label: speedup (as before) on top, that same
                % run's own iteration count below it -- lets a reader tell
                % apart a bar that's fast because each matvec is cheaper
                % from one that's fast (or slow) because it also took a
                % different number of iterations than the FP32 baseline.
                % Actual newline character, not a TeX \newline macro (MATLAB's
                % tex interpreter doesn't support that LaTeX command) -- a
                % literal "\n" inside the string breaks text() onto two lines
                % regardless of Interpreter, so this stays 'none' like every
                % other label in this file.
                if ~isnan(iters(g, b))
                    label = sprintf('%.2fx\n%d it', speedup(g, b), iters(g, b));
                else
                    label = sprintf('%.2fx', speedup(g, b));
                end
                text(ax, x(g), speedup(g, b), label, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                    'FontSize', kAnnotFontSize, 'Interpreter', 'none');
            end
        end
    end
    % Parity with the FP32-precond baseline.
    yline(ax, 1, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 2.5);
    hold(ax, 'off');

    tick_labels = arrayfun(@(n) sprintf('%d columns', n), cols, 'UniformOutput', false);
    set(ax, 'FontSize', kAxisFontSize, 'LineWidth', 2, 'XTick', 1:nG, ...
        'XTickLabel', tick_labels, 'TickLabelInterpreter', 'none', ...
        'YGrid', 'on', 'Box', 'off');
    ylabel(ax, 'Speedup over FP32 (precond)', 'FontSize', kAxisFontSize);
    % Extra headroom vs. the single-line-label 1.15x used elsewhere: each
    % annotation is now two lines (speedup + iteration count) at a larger
    % font size (kAnnotFontSize). No longer needs to also clear a top-side
    % legend now that the legend moved below the axes.
    ylim(ax, [0, max(max(speedup(:), [], 'omitnan'), 1) * 1.3]);
    legend(ax, hb, legend_labels, 'Location', 'southoutside', 'NumColumns', nK, ...
        'Interpreter', 'none', 'FontSize', kLegendFontSize);

    out_base = fullfile(out_dir, out_name);
    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d column counts x %d methods x %d condition numbers)\n', ...
        out_base, out_base, nG, nK, nCond);
end
