function plot_gemv_fp32_recon_l2_error_vs_cublas_ref(csv_path, scale_tag)
%PLOT_GEMV_FP32_RECON_L2_ERROR_VS_CUBLAS_REF Plot the NO_TRANSPOSE (y =
%   alpha*A*x) L2-relative-error sweep from scripts/benchmarks_gemv_fp32_
%   recon_l2_error_vs_cublas_ref.sh (which itself builds and runs
%   benchmarks/kernels/cuda/matrix/mv_fp32_recon_l2_error_vs_cublas_ref.cu)
%   -- the accuracy-vs-M companion to plot_gemv_fp32_recon_vs_cutlass_
%   ref.m (which plots GFlop/s for the same five series over the same
%   square sweep). Same five "TOMOS" fp32-reconstruction-tier GEMV
%   series, same shapes, but the y-axis here is how far each method's
%   result drifts from a genuine FP32-precision reference (cublasSgemv on
%   an uncompressed float32 copy of that method's own data -- see the
%   .cu's own header for exactly how that reference is generated on-
%   device without ever needing the full M*K matrix on the host), not
%   throughput:
%
%   TOMOS FP32              = mv_fp32_tiled
%   TOMOS FP32 / FP16       = mv_fp32_16_accurate_ready4_atomiconly_
%                              scaled_kblock
%   TOMOS FP32 / FP16 (fast) = mv_fp32_16_accurate_ready_atomiconly_
%                              scaled_kblock_fast_normal
%   TOMOS FP32 / BF16       = mv_fp32_bf16_accurate_ready_atomiconly_
%                              scaled_kblock_rounded
%   TOMOS FP32 / BF16 (workspace) = mv_fp32_bf16_accurate_ready_
%                              atomiconly_scaled_kblock_workspace
%
%   See plot_gemv_fp32_recon_vs_cutlass_ref.m's own header for the full
%   derivation/promotion history of each of these five dispatchers -- not
%   repeated here, since it's identical (this plot analyzes the SAME
%   promoted kernels, just a different metric).
%
%   TOMOS FP32's own error is a sanity check on this file's methodology,
%   not a real compression-precision measurement -- it computes in plain
%   float32 throughout (no 16-bit storage at all) against a cublasSgemv
%   reference on the SAME underlying float32 array, so it should stay
%   near machine epsilon regardless of normal_scale; see the .cu's own
%   header for why.
%
%   No cuBLAS series here (unlike the GFlop/s plot, and unlike CUTLASS):
%   cuBLAS/cublasSgemv IS this file's own reference, so it has no "error
%   vs itself" to plot.
%
%   Square sweep ONLY (M=K) -- scripts/benchmarks_gemv_fp32_recon_l2_
%   error_vs_cublas_ref.sh has no tall-and-thin sweep yet, unlike the
%   GFlop/s benchmark. Y-AXIS IS LOG-SCALE: L2 relative error spans
%   orders of magnitude across these methods/shapes, unlike GFlop/s
%   (bandwidth-bound, roughly flat), so a linear axis would flatten every
%   series but the largest into an unreadable line near zero.
%
%   The CSV is expected to have M/K columns plus *_l2_rel_err columns per
%   method -- see benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh's
%   own embedded parser for the exact schema. ("SKIPPED" cells, from a
%   shape a method ran out of device memory for, become NaN and simply
%   leave a gap in that method's line.)
%
%   scale_tag (optional) identifies which "normal_scale" tier (see the
%   .cu's own header comment) this CSV's own data was generated at --
%   scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh sweeps
%   several tiers, one full run each, and passes its own scale value
%   straight through here so each tier's plot is independently readable:
%   it's appended to the output filename (so tiers don't overwrite each
%   other) and shown in the plot's own title. Omit it (or pass '') for a
%   single ad hoc run with no tier to distinguish -- title/filename both
%   fall back to their original, untagged form.
%
%   Writes, under <repo>/build/benchmarks/output/plots/error/gemv32/:
%     plot_gemv_fp32_recon_l2_error_vs_cublas_ref[_scale<scale_tag>].png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        scale_tag = '';
    end
    scale_tag = char(scale_tag);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mv_fp32_recon_l2_error_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_fp32_recon_l2_error_vs_cublas_ref:noCsv', ...
                ['No mv_fp32_recon_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_fp32_recon_l2_error_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_fp32_recon_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots', 'error', 'gemv32');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');

    % Solid lines for every series -- no dashed/vendor-baseline distinction
    % here since there's no cuBLAS series (see this file's own header).
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'tomos_fp32_l2_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_l2_rel_err'), 'style', '-v');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (fast)', 'values', to_double_col(T, 'tomos_fp32_fp16_fast_l2_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'tomos_fp32_bf16_l2_rel_err'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_bf16_workspace_l2_rel_err'), 'style', '-^');

    if isempty(scale_tag)
        out_base = fullfile(out_dir, 'plot_gemv_fp32_recon_l2_error_vs_cublas_ref');
    else
        out_base = fullfile(out_dir, ['plot_gemv_fp32_recon_l2_error_vs_cublas_ref_scale' scale_tag]);
    end

    plot_series(M, series, true(size(M)), 'Matrix dimensions (m = k)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_fp32_recon_l2_error_vs_cublas_ref:missingColumn', ...
            'CSV is missing expected column "%s".', name);
    end
    col = T.(name);
    if isnumeric(col)
        v = double(col);
    else
        % Non-numeric cells (e.g. "SKIPPED") become NaN, which plot()
        % naturally renders as a gap in that method's line.
        v = str2double(string(col));
    end
end

function csv_path = find_latest_csv(dir_path)
    csv_path = '';
    if ~exist(dir_path, 'dir')
        return
    end
    files = dir(fullfile(dir_path, 'mv_fp32_recon_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_fp32_recon_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    % Wider than the 1000x650 the GFlop/s sibling plot uses -- with 14
    % rotated numeric x-tick labels at kTickFontSize below, a narrower
    % figure runs the rightmost label off the saved PNG's own right edge
    % (confirmed once already for the sibling fp32_16 L2-error plot,
    % unrelated to any particular data -- a pure figure-sizing issue).
    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % Same fixed qualitative palette as plot_gemv_fp32_16_l2_error_vs_
    % cublas_ref.m's own plot_series, first five entries (one per series
    % here).
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.90 0.55 0.10;  % orange
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

    % kTickFontSize is deliberately SMALLER than kAxisFontSize (14 rotated
    % 5-6-digit numbers, not a handful of short category names) -- same
    % convention plot_gemv_fp32_16_l2_error_vs_cublas_ref.m's own
    % plot_series uses, see its own comment for why.
    kAxisFontSize = 40;
    kTickFontSize = 32;
    kLegendFontSize = 28;
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XTick', unique(x_sorted), ...
        'FontSize', kTickFontSize, 'XTickLabelRotation', 40, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('L2 relative error vs FP32', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;

    % exportgraphics (not saveas) for the PNG -- see plot_gemv_fp32_16_
    % l2_error_vs_cublas_ref.m's own comment for why.
    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
