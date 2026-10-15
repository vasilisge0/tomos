function plot_gemm_fp16_kernels_l2_error_vs_cublas_ref(csv_path)
%PLOT_GEMM_FP16_KERNELS_L2_ERROR_VS_CUBLAS_REF Plot the FP16 GEMM
%   kernel-implementation comparison sweep from scripts/benchmarks_gemm_
%   fp16_kernels_l2_error_vs_cublas_ref.sh (which builds and runs
%   benchmarks/kernels/cuda/matrix/mm_fp16_kernels_l2_error_vs_cublas_
%   ref.cu). Y-axis is the FROBENIUS relative error vs a genuine FP32
%   (cublasSgemm) reference.
%
%   Three series, all run on the IDENTICAL native-rounded A/B data (no
%   per-series encoding variation, unlike plot_gemm_fp32_16_variable_
%   rounding_l2_error_vs_cublas_ref.m's own sibling plot, which instead
%   isolates A's own encoding/rounding method at a FIXED GEMM kernel) --
%   here the only thing that ever differs is which GEMM implementation
%   computed the result:
%
%   TOMOS FP16   = gemm_fp16_mma_ptx_wide_bigreg_sq_deep4_iter_epivec_
%                   acongruous_fastk_precomp2_serp -- tomos's own
%                   tensor-core kernel (accumulates in float).
%   cuBLAS FP16  = cublasHgemm -- cuBLAS's own simple FP16 GEMM API. Its
%                   own accumulation precision is NOT guaranteed to
%                   match TOMOS FP16's own (may accumulate in half
%                   rather than float internally) -- empirically this
%                   showed roughly a 30x larger error than TOMOS FP16 in
%                   informal testing, consistent with FP16-accumulate
%                   error compounding much faster than float-accumulate
%                   over a large K-reduction.
%   CUTLASS FP16 = cutlass::gemm::device::Gemm, the SAME "textbook"
%                   Ampere tensorop config (128x256x32/64x64x32/16x8x16,
%                   3-stage, FP32 accumulator) scripts/cutlass_fp16_
%                   gemm_ref.cu's own binary already uses as an external
%                   throughput comparison point -- reused here for
%                   accuracy instead of speed.
%
%   Uses the SAME square-sweep sizes as plot_gemm_fp32_16_variable_
%   rounding_l2_error_vs_cublas_ref.m's own N-sweep, so the two plot
%   families are directly comparable at the same shapes.
%
%   Square sweep ONLY (M=K=N). X-AXIS IS LOG-SCALE (matrix dimension).
%   Y-AXIS IS LOG-SCALE too -- given the ~30x gap already observed
%   between TOMOS FP16 and cuBLAS FP16, a linear axis would compress the
%   smaller series into an indistinguishable line at the bottom.
%
%   The CSV is expected to have M/K/N columns plus *_frobenius_rel_err
%   columns per series -- see benchmarks_gemm_fp16_kernels_l2_error_vs_
%   cublas_ref.sh's own embedded parser for the exact schema. ("SKIPPED"
%   cells become NaN and leave a gap in that series' line -- CUTLASS can
%   report its own SKIPPED if cutlass::Status::kSuccess isn't reached at
%   a given shape, see the .cu's own header.)
%
%   Writes, under <repo>/build/benchmarks/output/plots/error/gemm16kernels/:
%     plot_gemm_fp16_kernels_l2_error_vs_cublas_ref.png / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mm_fp16_kernels_l2_error_vs_cublas_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemm_fp16_kernels_l2_error_vs_cublas_ref:noCsv', ...
                ['No mm_fp16_kernels_l2_error_vs_cublas_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemm_fp16_kernels_l2_error_vs_cublas_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemm_fp16_kernels_l2_error_vs_cublas_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots', 'error', 'gemm16kernels');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    N = to_double_col(T, 'N');

    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'TOMOS FP16', 'values', to_double_col(T, 'tomos_fp16_frobenius_rel_err'), 'style', '-o');
    i = i + 1; series(i) = struct('label', 'cuBLAS FP16', 'values', to_double_col(T, 'cublas_fp16_frobenius_rel_err'), 'style', '--d');
    i = i + 1; series(i) = struct('label', 'CUTLASS FP16', 'values', to_double_col(T, 'cutlass_fp16_frobenius_rel_err'), 'style', '--^');

    out_base = fullfile(out_dir, 'plot_gemm_fp16_kernels_l2_error_vs_cublas_ref');

    plot_series(N, series, true(size(N)), 'Matrix dimension, N (M = K = N)', out_base);
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemm_fp16_kernels_l2_error_vs_cublas_ref:missingColumn', 'CSV is missing expected column "%s".', name);
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
    files = dir(fullfile(dir_path, 'mm_fp16_kernels_l2_error_vs_cublas_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemm_fp16_kernels_l2_error_vs_cublas_ref:emptyGroup', 'No rows to plot -- skipping.');
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    kPalette = [
        0.00 0.45 0.74;  % blue (TOMOS FP16)
        0.85 0.10 0.10;  % red (cuBLAS FP16)
        0.20 0.80 0.20;  % bright green (CUTLASS FP16)
    ];
    colors = kPalette(1:numel(series), :);
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
    grid on;

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
