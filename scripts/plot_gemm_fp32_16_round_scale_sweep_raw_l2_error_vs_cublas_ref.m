function plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref(csv_path)
%PLOT_GEMM_FP32_16_ROUND_SCALE_SWEEP_RAW_L2_ERROR_VS_CUBLAS_REF
%   Plot the fp32_16 encode-side rounding-precision sweep from scripts/
%   benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_
%   vs_cublas_ref.sh: the SAME fixed shape (M=K=N=49152) and the SAME
%   normal_scale axis as plot_gemm_fp32_16_round_scale_sweep_l2_error_
%   vs_cublas_ref.m's own data, but generated WITHOUT --normalized (B
%   stays at plain scale=1.0, not rescaled by 1/sqrt(K)).
%
%   Why this plot exists, alongside the --normalized one: Frobenius
%   relative error (||C_ref-C_method||_F / ||C_ref||_F) is scale-
%   invariant in exact arithmetic -- rescaling B rescales numerator and
%   denominator by the same factor, so the ratio itself shouldn't change,
%   PROVIDED nothing overflows or underflows along the way. --normalized
%   exists purely to keep C representable at large normal_scale/K, not to
%   change the comparison being measured (normalization only ever
%   rescales B, never A, so it doesn't touch which fp32_16 encoding
%   branch -- normal/extended/degraded -- A's own elements fall into).
%   This plot is the direct check of that expectation: at every
%   normal_scale value BOTH sweeps cover, this plot's values should
%   closely match the --normalized sibling's own values at the same
%   normal_scale, since here B is left at scale=1.0 instead. Only the
%   SCALES range differs (see the .sh's own header for the derivation) --
%   this sweep is deliberately capped tighter (1 through 64, not the
%   --normalized sibling's 1 through 16384), since without normalization
%   C's own row-sum magnitude grows with normal_scale*sqrt(K) instead of
%   staying ~normal_scale, and would overflow fp16 well before reaching
%   the wider range's own largest values at this fixed, large K.
%
%   NOTE ON THIS FILE'S OWN NAME: see plot_gemm_fp32_16_round_scale_
%   sweep_l2_error_vs_cublas_ref.m's own header for why "round_scale_
%   sweep" replaces "variable_rounding_scale_sweep" here -- MATLAB caps
%   function/file names at 63 characters, and the fully-consistent name
%   (matching this codebase's own .cu/.sh naming) exceeds that.
%
%   Same six series as the --normalized sibling plot, all measuring the
%   SAME real, unmodified native tensor-core kernel (gemm_fp16_mma_ptx_
%   wide_bigreg_sq_deep4_iter_epivec_acongruous_fastk_precomp2_serp,
%   "TOMOS FP16") -- see that .m file's own header for the full series
%   description and the mantissa-width-vs-rounding-method 2x2 design.
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
%   fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref.sh's
%   own embedded parser for the exact schema. ("SKIPPED" cells become
%   NaN and leave a gap in that series' line.)
%
%   Writes, under <repo>/build/benchmarks/output/plots/error/gemm16/:
%     plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mm_fp32_16_variable_rounding_l2_error_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref_*.csv found under ' ...
                 '%s.\nRun scripts/benchmarks_gemm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_' ...
                 'cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref:missingCsv', ...
            'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots', 'error', 'gemm16');
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

    out_base = fullfile(out_dir, 'plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref');

    plot_series(scale, series, true(size(scale)), 'Magnitude scale M', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp32_16_variable_rounding_scale_sweep_raw_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp32_16_round_scale_sweep_raw_l2_error_vs_cublas_ref:emptyGroup', ...
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
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
