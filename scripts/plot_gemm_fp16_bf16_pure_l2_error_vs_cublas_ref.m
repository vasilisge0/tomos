function plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref(csv_path, scale_tag)
%PLOT_GEMM_FP16_BF16_PURE_L2_ERROR_VS_CUBLAS_REF Plot the pure/
%   composite-tier (tensor-core) GEMM Frobenius-relative-error sweep from
%   scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh
%   (which itself builds and runs benchmarks/kernels/cuda/matrix/mm_
%   fp16_bf16_pure_l2_error_vs_cublas_ref.cu) -- the accuracy-vs-N
%   companion to plot_fp16_bf16_pure_vs_cutlass_ref.m (which plots
%   GFlop/s for the same square sweep). Named "l2_error" for consistency
%   with every sibling *_l2_error_vs_cublas_ref.m file, even though the
%   y-axis here is the FROBENIUS relative error (see plot_gemm_fp32_
%   recon_l2_error_vs_cublas_ref.m's own header for why that's textually
%   identical to L2 on a flattened array) -- not throughput:
%
%   TOMOS FP16              = gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_
%                              iter_epivec_acongruous_fastk_precomp2_serp
%   TOMOS BF16               = gemm_bf16_mma_ptx_wide_bigreg_sq_deep4_
%                              iter_epivec_acongruous_fastk_precomp2_serp
%   TOMOS FP32 / FP16        = mm_fp32_16_pure_half_wide_bigreg_sq_
%                              inline_dirty_acongruous_fastk_precomp2_serp
%   TOMOS FP32 / FP16 (workspace) = mm_fp32_16_pure_half_gemm_
%                              materialize_cached_vec_fastk_precomp2_serp
%   TOMOS FP32 / BF16        = mm_fp32_bf16_pure_bf16_gemm_direct_
%                              fastk_precomp2_serp
%
%   Plus four external, non-TOMOS GEMM baselines (dashed lines), each run
%   on the SAME native-rounded A/B as its own group's native TOMOS
%   series:
%
%   cuBLAS FP16              = cublasHgemm -- own accumulation precision
%                              not guaranteed to match TOMOS FP16's own.
%   CUTLASS FP16             = cutlass::gemm::device::Gemm, same config
%                              as scripts/cutlass_fp16_gemm_ref.cu's own
%                              binary.
%   cuBLAS BF16              = cublasGemmEx (CUDA_R_16BF I/O,
%                              CUBLAS_COMPUTE_32F).
%   CUTLASS BF16             = cutlass::gemm::device::Gemm, same
%                              tensorop config as CUTLASS FP16.
%
%   See plot_fp16_bf16_pure_vs_cutlass_ref.m's own header for the full
%   derivation/promotion history of each of these five dispatchers -- not
%   repeated here, since it's identical (this plot analyzes the SAME
%   promoted kernels, just a different metric).
%
%   Unlike this tier's GEMM-recon-tier sibling (plot_gemm_fp32_recon_l2_
%   error_vs_cublas_ref.m), there is no "TOMOS FP32" sanity-check series
%   here -- every one of these five kernels genuinely stores at least A
%   (and, for the two native kernels, both A and B) in fp16/bf16
%   precision by design (this tier IS the tensor-core compute tier), so
%   none of them can reproduce a near-machine-epsilon result the way a
%   plain float32 kernel would -- see the .cu's own header for why the
%   reference itself intentionally captures BOTH operands' storage
%   rounding, not just A's.
%
%   No cuBLAS/CUTLASS series here (unlike the GFlop/s plot): cuBLAS/
%   cublasSgemm IS this file's own reference.
%
%   Square sweep ONLY (M=K=N). Y-AXIS IS LOG-SCALE: Frobenius relative
%   error spans orders of magnitude across these methods/shapes, unlike
%   GFlop/s.
%
%   The CSV is expected to have M/K/N columns plus *_frobenius_rel_err
%   columns per method -- see benchmarks_gemm_fp16_bf16_pure_l2_error_
%   vs_cublas_ref.sh's own embedded parser for the exact schema.
%   ("SKIPPED" cells become NaN and leave a gap in that method's line.)
%
%   scale_tag (optional) identifies which "normal_scale" tier this CSV's
%   own data was generated at -- appended to the output filename. Omit it
%   (or pass '') for a single ad hoc run with no tier to distinguish.
%
%   Writes, under <repo>/build/output/error_dimension_scaling/gemm/fp16_compute/plots/:
%     plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref[_scale<scale_tag>].png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        scale_tag = '';
    end
    scale_tag = char(scale_tag);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemm', 'fp16_compute', 'data');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp16_bf16_pure_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'output', 'error_dimension_scaling', 'gemm', 'fp16_compute', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    N = to_double_col(T, 'N');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col(T, 'tomos_fp16_frobenius_rel_err'), 'style', '-x');
    i = i + 1; series(i) = struct('label', 'TOMOS BF16', 'values', to_double_col(T, 'tomos_bf16_frobenius_rel_err'), 'style', '-<');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'tomos_fp32_fp16_frobenius_rel_err'), 'style', '-+');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (workspace)', 'values', to_double_col(T, 'tomos_fp32_fp16_workspace_frobenius_rel_err'), 'style', '->');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'tomos_fp32_bf16_frobenius_rel_err'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_fp16_frobenius_rel_err'), 'style', '--d');
    i = i + 1; series(i) = struct('label', 'CUTLASS FP16', 'values', to_double_col(T, 'cutlass_fp16_frobenius_rel_err'), 'style', '--^');
    i = i + 1; series(i) = struct('label', 'cuBLAS BF16', 'values', to_double_col(T, 'cublas_bf16_frobenius_rel_err'), 'style', '--v');
    i = i + 1; series(i) = struct('label', 'CUTLASS BF16', 'values', to_double_col(T, 'cutlass_bf16_frobenius_rel_err'), 'style', '--p');

    if isempty(scale_tag)
        out_base = fullfile(out_dir, 'plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref');
    else
        out_base = fullfile(out_dir, ['plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref_scale' scale_tag]);
    end

    plot_series(N, series, true(size(N)), 'Matrix dimensions (m = k = n)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref:missingColumn', ...
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
    files = dir(fullfile(dir_path, 'mm_fp16_bf16_pure_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp16_bf16_pure_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.90 0.55 0.10;  % orange
        0.30 0.30 0.30;  % dark grey (cuBLAS FP16)
        0.00 0.80 0.80;  % cyan (CUTLASS FP16)
        0.60 0.40 0.20;  % brown (cuBLAS BF16)
        1.00 0.75 0.80;  % pink (CUTLASS BF16)
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
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XTick', unique(x_sorted), ...
        'FontSize', kTickFontSize, 'XTickLabelRotation', 40, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('Frobenius relative error', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);

    % Widen the y-axis to whole decades (rounding the auto-computed limits
    % OUTWARD only, so real data is never clipped) and label every decade
    % in between, plus minor (2-9x) ticks/gridlines within each decade --
    % MATLAB's own log-scale autoscale can otherwise leave only a single
    % "nice" decade labeled (e.g. just 10^-3) when the data happens to
    % cluster tightly within one decade, giving almost no numeric
    % reference on the y-axis.
    yl = ylim;
    y_lo_dec = floor(log10(yl(1)));
    y_hi_dec = ceil(log10(yl(2)));
    if y_hi_dec <= y_lo_dec
        y_hi_dec = y_lo_dec + 1;
    end
    ylim(10.^[y_lo_dec, y_hi_dec]);
    set(gca, 'YTick', 10.^(y_lo_dec:y_hi_dec), 'YMinorTick', 'on');
    grid on;
    ax = gca;
    ax.YMinorGrid = 'on';

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
