function plot_gemv_fp32_16_l2_error_vs_cublas_ref(csv_path, scale_tag)
%PLOT_GEMV_FP32_16_L2_ERROR_VS_CUBLAS_REF Plot the NO_TRANSPOSE (y =
%   alpha*A*x) L2-relative-error sweep from scripts/benchmarks_gemv_fp32_
%   16_l2_error_vs_cublas_ref.sh (which itself builds and runs benchmarks/
%   kernels/cuda/matrix/mv_fp32_16_l2_error_vs_cublas_ref.cu) -- the
%   accuracy-vs-M companion to plot_gemv_fp32_16_pure_half_vs_cublas_ref.m
%   (which plots GFlop/s for the same five series over the same square
%   sweep). Same five "TOMOS" 16-bit-storage GEMV series, same shapes,
%   but the y-axis here is how far each method's result drifts from a
%   genuine FP32-precision reference (cublasSgemv on an uncompressed
%   float32 copy of that method's own data -- see the .cu's own header
%   for exactly how that reference is generated on-device without ever
%   needing the full M*K matrix on the host), not throughput:
%
%   TOMOS FP16              = mv_fp32_16_pure_half_wide_ilp_shared_half2_
%                              atomic_best
%   TOMOS FP32 / FP16       = mv_fp32_16_hybrid_wide_ilp_shared_half2_
%                              atomic_branchless_best
%   TOMOS FP32 / FP16 (fast) = mv_fp32_16_pure_half_wide_ilp_shared_half2_
%                              atomic_fasthalf_best
%   TOMOS FP32 / BF16       = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
%                              atomic_best_v2
%   TOMOS BF16              = mv_fp32_bf16_pure_bf16_wide_ilp_shared_
%                              atomic_best
%
%   See plot_gemv_fp32_16_pure_half_vs_cublas_ref.m's own header for the
%   full derivation/promotion history of each of these five dispatchers
%   -- not repeated here, since it's identical (this plot analyzes the
%   SAME promoted kernels, just a different metric).
%
%   Plus two external, non-TOMOS GEMV baselines (dashed lines), measuring
%   THEIR OWN error against the SAME plain-tier float32 reference/ground
%   truth as TOMOS FP16/BF16/FP32-BF16 above (independent __half/
%   __nv_bfloat16-typed A/x, index-matched via the SAME pseudo_rand(idx)
%   call, not a converted copy of any TOMOS series' own data -- see the
%   .cu's own gen_half/gen_bf16 comment):
%
%   cuBLAS FP16              = cublasHgemm (n=1, GEMV-via-GEMM, no native
%                              cublasHgemv exists)
%   cuBLAS BF16               = cublasGemmEx (CUDA_R_16BF, n=1, no native
%                              cublasXgemv for this dtype either)
%
%   cuBLAS itself (cublasSgemv, real fp32) remains this file's own
%   reference throughout -- these two are cuBLAS's OWN 16-bit compute
%   paths being measured against that reference, not the reference itself.
%
%   Square sweep ONLY (M=K) -- scripts/benchmarks_gemv_fp32_16_l2_error_
%   vs_cublas_ref.sh has no tall-and-thin sweep yet, unlike the GFlop/s
%   benchmark. Y-AXIS IS LOG-SCALE: L2 relative error spans orders of
%   magnitude across these methods/shapes, unlike GFlop/s (bandwidth-
%   bound, roughly flat), so a linear axis would flatten every series but
%   the largest into an unreadable line near zero.
%
%   The CSV is expected to have M/K columns plus *_l2_rel_err columns per
%   method -- see benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh's own
%   embedded parser for the exact schema. ("SKIPPED" cells, from a shape
%   a method ran out of device memory for, become NaN and simply leave a
%   gap in that method's line.)
%
%   scale_tag (optional) identifies which "normal_scale" tier (see the
%   .cu's own header comment) this CSV's own data was generated at --
%   scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh sweeps
%   several tiers, one full run each, and passes its own scale value
%   straight through here so each tier's plot is independently readable:
%   it's appended to the output filename (so tiers don't overwrite each
%   other) and shown in the plot's own title. Omit it (or pass '') for a
%   single ad hoc run with no tier to distinguish -- title/filename both
%   fall back to their original, untagged form.
%
%   Writes, under <repo>/build/output/error_dimension_scaling/gemv/fp16_compute/plots/:
%     plot_gemv_fp32_16_l2_error_vs_cublas_ref[_scale<scale_tag>].png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        scale_tag = '';
    end
    scale_tag = char(scale_tag);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemv', 'fp16_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_fp32_16_l2_error_vs_cublas_ref:noCsv', ...
                ['No mv_fp32_16_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_fp32_16_l2_error_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_fp32_16_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemv', 'fp16_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');

    % Solid lines for every series -- no dashed/vendor-baseline distinction
    % here since there's no cuBLAS series (see this file's own header).
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    % "TOMOS FP16" = native fp16 INPUTS (A a plain __half array), the same
    % kernel/meaning the GEMV performance plots give that label. Optional
    % since CSVs from before this series existed don't have the column.
    if any(strcmp('tomos_fp16_native_l2_rel_err', T.Properties.VariableNames))
        i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col(T, 'tomos_fp16_native_l2_rel_err'), 'style', '->');
    end
    % mv_fp16_native_mixed -- the kernel LSQR's --explicit-duplicates runs
    % (native __half A, float x/y). Optional, same reason as above.
    if any(strcmp('tomos_fp16_native_mixed_l2_rel_err', T.Properties.VariableNames))
        i = i + 1; series(i) = struct('label', 'TOMOS FP16 (native, mixed) [LSQR explicit]', 'values', to_double_col(T, 'tomos_fp16_native_mixed_l2_rel_err'), 'style', '-d');
    end
    % narrow_vec (the kernel LSQR's implicit NO_TRANSPOSE matvec runs at large
    % M) unflushed vs with its __half accumulators added into float every 64 /
    % 256 columns. Optional, same reason as above.
    if any(strcmp('tomos_fp16_narrowvec_l2_rel_err', T.Properties.VariableNames))
        i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, narrow_vec)', 'values', to_double_col(T, 'tomos_fp16_narrowvec_l2_rel_err'), 'style', '-s');
        i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, narrow_vec, flush 64)', 'values', to_double_col(T, 'tomos_fp16_narrowvec_flush64_l2_rel_err'), 'style', '-p');
        i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, narrow_vec, flush 256)', 'values', to_double_col(T, 'tomos_fp16_narrowvec_flush256_l2_rel_err'), 'style', '-h');
    end
    % Same kernel as flush 256 with the SWAR (two wh per register) decode --
    % bit-identical decode, so its error should equal flush 256's. Optional.
    if any(strcmp('tomos_fp16_narrowvec_flush256_swar_l2_rel_err', T.Properties.VariableNames))
        i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit, narrow_vec, flush 256, SWAR decode)', 'values', to_double_col(T, 'tomos_fp16_narrowvec_flush256_swar_l2_rel_err'), 'style', '-+');
    end
    % The packed FP32_16 wh-only kernel (formerly labeled just "TOMOS FP16"
    % here, which no longer matched the performance plots' convention).
    i = i + 1; series(i) = struct('label', 'TOMOS FP16 (implicit)', 'values', to_double_col(T, 'tomos_fp16_l2_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_l2_rel_err'), 'style', '-v');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (fast)', 'values', to_double_col(T, 'tomos_fp32_fp16_fast_l2_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'tomos_fp32_bf16_l2_rel_err'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS BF16', 'values', to_double_col(T, 'tomos_bf16_l2_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_fp16_l2_rel_err'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'cuBLAS BF16', 'values', to_double_col(T, 'cublas_bf16_l2_rel_err'), 'style', '--s');

    if isempty(scale_tag)
        out_base = fullfile(out_dir, 'plot_gemv_fp32_16_l2_error_vs_cublas_ref');
    else
        out_base = fullfile(out_dir, ['plot_gemv_fp32_16_l2_error_vs_cublas_ref_scale' scale_tag]);
    end

    plot_series(M, series, true(size(M)), 'Matrix dimensions (m = k)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_fp32_16_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mv_fp32_16_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_fp32_16_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    % Wider than plot_gemv_fp32_16_pure_half_vs_cublas_ref.m's own 1000px
    % (1000, 650) -- with 14 rotated numeric x-tick labels at kTickFontSize
    % below, the rightmost one's text was running off the saved PNG's own
    % right edge under the narrower figure (confirmed: happened even at
    % scale=1, unrelated to any particular data -- a pure figure-sizing
    % issue, not something the data caused).
    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % Same fixed qualitative palette as plot_gemv_fp32_16_pure_half_vs_
    % cublas_ref.m's own plot_series, first five entries for the TOMOS
    % series plus two more (dark grey/brown, matching the vendor-baseline
    % colors plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.m's own
    % palette uses) for the two cuBLAS series added below.
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.90 0.55 0.10;  % orange
        0.30 0.30 0.30;  % dark grey (cuBLAS FP16)
        0.55 0.35 0.15;  % brown (cuBLAS BF16)
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

    % kTickFontSize is deliberately SMALLER than kAxisFontSize (unlike
    % plot_gemv_fp32_16_pure_half_vs_cublas_ref.m's own convention, which
    % uses one size for both, set via gca's own FontSize): the tick labels
    % are 14 rotated 5-6-digit numbers, not a handful of short category
    % names, so gca's own FontSize (which controls tick labels, not the
    % xlabel/ylabel/title calls below -- each of those sets its own
    % FontSize explicitly) is kept modest specifically to leave the
    % rightmost rotated label room to render fully within the figure
    % (see this function's own figure-width comment above).
    kAxisFontSize = 40;
    kTickFontSize = 32;
    kLegendFontSize = 28;
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XTick', unique(x_sorted), ...
        'FontSize', kTickFontSize, 'XTickLabelRotation', 40, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('L2 relative error vs FP32', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;

    % exportgraphics (not saveas) for the PNG: saveas renders via the
    % figure's own PaperPosition/PaperSize, which in headless -batch mode
    % doesn't reliably grow to fit a large rotated tick label that extends
    % past the axes' own box -- exportgraphics captures the figure's
    % actual on-screen (here off-screen) content directly, avoiding that
    % legacy path entirely.
    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
