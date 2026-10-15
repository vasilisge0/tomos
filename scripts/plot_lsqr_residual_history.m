function plot_lsqr_residual_history(files, labels, out_dir, out_name)
%PLOT_LSQR_RESIDUAL_HISTORY Plot relative-residual-vs-iteration curves
%   from one or more LSQR --residual-history files (see
%   benchmarks/solvers/cuda/lsqr_precision_runner.cu / scripts/
%   run_lsqr_precision.sh's own doc comment for that file format: plain
%   text, "iteration value" per line, 1-based iteration index, value
%   already a relative residual -- no further scaling needed here).
%   Driven by scripts/run_lsqr_precision_compare.sh, which runs a fixed
%   set of LSQR precision/preconditioning configurations against the same
%   (A, x_true, b) problem and writes one history file per configuration;
%   this function overlays however many of those it's given on one plot,
%   semilogy (residuals span orders of magnitude as LSQR converges).
%
%   files:  cell array of residual-history file paths.
%   labels: cell array of legend labels, same length as files.
%   out_dir: where to write the plot (default: <repo>/build/output/lsqr/
%            error -- every matrix's own plot lands flat in this one
%            directory; the residual-history files themselves live one
%            level down, in out_dir/<matrix-name>/, written by
%            run_lsqr_precision_compare.sh).
%   out_name: output file base name (default 'lsqr_residual_history');
%            run_lsqr_precision_compare.sh passes
%            'lsqr_residual_history_<matrix-name>' so every matrix's plot
%            gets its own filename in the shared out_dir instead of
%            colliding on the default name.
%
%   A file that doesn't exist or can't be read is skipped with a warning
%   (not a hard error) -- so a partial run's plot still shows whatever
%   histories did get written.
%
%   Writes: <out_dir>/<out_name>.png / .fig

    if numel(files) ~= numel(labels)
        error('plot_lsqr_residual_history:sizeMismatch', ...
            'files (%d) and labels (%d) must have the same length.', ...
            numel(files), numel(labels));
    end

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);
    if nargin < 3 || isempty(out_dir)
        out_dir = fullfile(repo_root, 'build', 'output', 'lsqr', 'error');
    end
    if nargin < 4 || isempty(out_name)
        out_name = 'lsqr_residual_history';
    end
    out_name = char(out_name);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    series = struct('label', {}, 'iter', {}, 'resid', {});
    for i = 1:numel(files)
        f = files{i};
        if ~isfile(f)
            warning('plot_lsqr_residual_history:missingFile', ...
                'Skipping "%s" (label "%s"): file not found.', f, labels{i});
            continue
        end
        data = readmatrix(f, 'FileType', 'text');
        if isempty(data) || size(data, 2) < 2
            warning('plot_lsqr_residual_history:emptyFile', ...
                'Skipping "%s" (label "%s"): no (iteration, residual) rows.', f, labels{i});
            continue
        end
        k = numel(series) + 1;
        series(k).label = labels{i};
        series(k).iter = data(:, 1);
        series(k).resid = data(:, 2);
    end

    if isempty(series)
        error('plot_lsqr_residual_history:noData', ...
            'None of the given residual-history files could be read.');
    end

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % Same fixed qualitative palette as the GEMV/GEMM performance plots in
    % this directory (e.g. plot_gemv_fp32_16_pure_half_vs_cublas_ref.m),
    % kept for visual consistency across this codebase's plots.
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.90 0.55 0.10;  % orange
        0.10 0.75 0.80;  % cyan
        0.85 0.20 0.60;  % magenta
        0.55 0.35 0.15;  % brown
    ];
    n = numel(series);
    if n <= size(kPalette, 1)
        colors = kPalette(1:n, :);
    else
        colors = [kPalette; lines(n - size(kPalette, 1))];
    end

    for i = 1:n
        semilogy(series(i).iter, series(i).resid, '-', 'LineWidth', 3.5, ...
            'Color', colors(i, :), 'DisplayName', series(i).label);
    end
    hold off;

    kAxisFontSize = 34;
    kLegendFontSize = 30;
    set(gca, 'YScale', 'log', 'FontSize', kAxisFontSize, 'LineWidth', 2);
    xlabel('Iteration', 'FontSize', kAxisFontSize);
    ylabel('Relative residual', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 2, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;

    out_base = fullfile(out_dir, out_name);
    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d series)\n', out_base, out_base, n);
end
