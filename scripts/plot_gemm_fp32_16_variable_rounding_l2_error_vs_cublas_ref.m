function plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref(csv_path)
%PLOT_GEMM_FP32_16_VARIABLE_ROUNDING_L2_ERROR_VS_CUBLAS_REF Plot the
%   fp32_16 encode-side rounding-precision sweep from scripts/benchmarks_
%   gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.sh (which
%   itself builds and runs benchmarks/kernels/cuda/matrix/mm_fp32_16_
%   variable_rounding_l2_error_vs_cublas_ref.cu). Y-axis is the FROBENIUS
%   relative error vs a genuine FP32 (cublasSgemm) reference -- see
%   plot_gemm_fp32_recon_l2_error_vs_cublas_ref.m's own header for why
%   that's textually identical to L2 on a flattened array.
%
%   Six series, ALL measuring the SAME real, unmodified native
%   tensor-core kernel (gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_
%   epivec_acongruous_fastk_precomp2_serp, "TOMOS FP16") with the SAME B
%   (native __half) every time -- only how A's own __half array was
%   produced differs, so the only thing that ever differs between any of
%   the 6 series is what values ended up in A's own __half array --
%   never which kernel computed the GEMM. Together these isolate TWO
%   separate axes: mantissa WIDTH (10 bits, real __half, vs 9 bits, the
%   fp32_16_half_only-decoded wh-alone format -- see the .cu's own header
%   for why wh-alone is structurally capped at 9 bits) and rounding
%   METHOD (round-to-nearest vs truncate):
%
%   TOMOS FP16 (native)   = 10-bit mantissa, ROUNDED: A/B converted to
%                            __half directly via __float2half (no
%                            packed-format encode/decode at all)
%   TOMOS FP16 (trunc10)   = 10-bit mantissa, TRUNCATED: A converted via
%                            this experiment's own float_to_half_
%                            truncate10 (see the .cu's own header) --
%                            bypasses the packed wh/wr format entirely,
%                            isolating truncation vs rounding at a FIXED
%                            10-bit mantissa
%   13 sticky bits         = 9-bit mantissa, ROUNDED: the real
%                            fp32_16_split's own rounding decision,
%                            reproduced bit-for-bit by this experiment's
%                            own fp32_16_split_variable_rounding
%                            (sticky_bits=13), decoded via
%                            fp32_16_half_only
%   9 sticky bits          = 9-bit mantissa, same encoder, sticky_bits=9
%   5 sticky bits          = 9-bit mantissa, same encoder, sticky_bits=5
%   1 sticky bit           = 9-bit mantissa, same encoder, sticky_bits=1
%   Truncation (0 bits)    = 9-bit mantissa, TRUNCATED: same encoder,
%                            sticky_bits=0 -- no rounding at all,
%                            matching fp32_16_split_truncated's own
%                            output bit-for-bit
%
%   See the .cu's own header for the full pipeline, the mantissa-width-
%   vs-rounding-method 2x2 design table, and for why both new functions
%   (fp32_16_split_variable_rounding, float_to_half_truncate10) are new,
%   separate, additive functions, not changes to fp32_16_split/
%   fp32_16_pack.hpp.
%
%   Square sweep ONLY (M=K=N). X-AXIS IS LOG-SCALE (matrix dimension);
%   Y-AXIS IS LINEAR -- the observed Frobenius relative error across
%   these rounding configs/shapes stays within roughly one decade
%   (~1e-4 to ~1e-3), not the multi-decade range a log y-axis is meant
%   for, so linear reads the actual magnitude of the differences more
%   directly than a log axis would.
%
%   The CSV is expected to have M/K/N columns plus *_frobenius_rel_err
%   columns per series -- see benchmarks_gemm_fp32_16_variable_rounding_
%   l2_error_vs_cublas_ref.sh's own embedded parser for the exact schema.
%   ("SKIPPED" cells become NaN and leave a gap in that series' line.)
%
%   Writes, under <repo>/build/benchmarks/output/plots/error/gemm16/:
%     plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref.sh first.'], ...
                default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots', 'error', 'gemm16');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    N = to_double_col(T, 'N');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 10, Sticky bits: 12 (FP32 to FP16)', 'values', to_double_col(T, 'tomos_fp16_native_frobenius_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 10, Sticky bits: 0 (truncation)', 'values', to_double_col(T, 'tomos_fp16_trunc10_frobenius_rel_err'), 'style', '-d');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, Sticky bits: 13 (TOMOS FP32 / FP16)', 'values', to_double_col(T, 'tomos_fp32_fp16_round_13_frobenius_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, sticky bits: 5', 'values', to_double_col(T, 'tomos_fp32_fp16_round_05_frobenius_rel_err'), 'style', '--s');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, sticky bits: 1', 'values', to_double_col(T, 'tomos_fp32_fp16_round_01_frobenius_rel_err'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'Mantissa bits: 9, sticky bits: 0', 'values', to_double_col(T, 'tomos_fp32_fp16_trunc_frobenius_rel_err'), 'style', '-*');

    out_base = fullfile(out_dir, 'plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref');

    plot_series(N, series, true(size(N)), 'Matrix dimension, N (M = K = N)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp32_16_variable_rounding_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.55 0.55 0.55;  % grey (native fp16 reference, 10-bit rounded)
        0.30 0.30 0.30;  % dark grey (10-bit truncated reference)
        0.00 0.45 0.74;  % blue (13 -- exact match to real encoder)
        0.00 0.60 0.30;  % green
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
