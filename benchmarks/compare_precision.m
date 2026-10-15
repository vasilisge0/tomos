function compare_precision(build_root)
%COMPARE_PRECISION Bar-chart FP16-compute vs FP32-compute kernels, all tasks
%   compared at the SAME problem size, pulled from the CSVs
%   bench_latex_reporter.hpp writes next to each benchmark executable's
%   build output.
%
%   compare_precision()          searches <repo>/build recursively
%   compare_precision(buildRoot) searches buildRoot recursively instead
%
%   "FP16 compute" = kernels that actually perform arithmetic in __half:
%   fp32_16's pure_half tier (CPU+CUDA, mv+mm), the __half-typed
%   generic-kernel CUDA benchmarks (which store AND accumulate in __half),
%   BM_mm_cublas<__half> (cublasHgemm), and BM_mv_cublas<__half> (cuBLAS has
%   no native half-precision GEMV routine, so this goes through
%   cublasHgemm(n=1) instead — the standard matrix-vector-as-matrix-matrix
%   workaround — rather than a real cublasHgemv).
%   "FP32 compute" = every other dense mv/mm benchmark. The sparse
%   mm_csr/mm_csr_cuda benchmarks are excluded from both groups: they don't
%   run the square/tall/wide size ladder the other benchmarks share.
%
%   CPU benchmarks (mv_cpu, mm_cpu, and the fp32_16 CPU file) are excluded
%   entirely — the CPU-side naive/BLAS kernels take much longer to sweep
%   through than the CUDA ones, so this compares CUDA tasks only. Pass a
%   group including CPU specs yourself (see make_spec calls below) if you
%   want CPU data back in the comparison.
%
%   mv (2D: M,N) and mm (3D: M,K,N) tasks are different shapes and can't
%   share a literal size, so each group is split into an mv sub-plot and an
%   mm sub-plot. Within each sub-plot, every bar is measured at the SAME
%   problem size: the largest size that was actually tested by every task
%   in that sub-plot (the intersection of tested sizes, not each task's own
%   independent max) — otherwise a bar that happened to be tested at a much
%   bigger size would look faster for that reason alone, not because the
%   kernel itself is faster.
%
%   Writes fp16_compute_comparison_mv.png, fp16_compute_comparison_mm.png,
%   fp32_compute_comparison_mv.png, and fp32_compute_comparison_mm.png to
%   <build_root>/benchmarks/output/.

    if nargin < 1
        here = fileparts(mfilename('fullpath'));
        build_root = fullfile(fileparts(here), 'build');
    end
    out_dir = fullfile(build_root, 'benchmarks', 'output');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    half_tag = '<__half, __half, __half>';
    fp16_group = { ...
        make_spec('bench_fp32_16_cuda_kernels.csv', 'CUDA', {'BM_mv_fp32_16_pure_half', 'BM_mm_fp32_16_pure_half'}, {}), ...
        make_spec('bench_mv_cuda_kernels.csv',      'CUDA', {['BM_gemv_generic_kernel', half_tag]}, {}), ...
        make_spec('bench_mv_cuda_kernels.csv',      'CUDA', {['BM_mv_cublas', half_tag]}, {}), ...
        make_spec('bench_mm_cuda_kernels.csv',      'CUDA', {['BM_gemm_generic_kernel', half_tag]}, {}), ...
        make_spec('bench_mm_cuda_kernels.csv',      'CUDA', {['BM_mm_cublas', half_tag]}, {}) ...
    };

    fp32_group = { ...
        make_spec('bench_mv_cuda_kernels.csv',      'CUDA', {}, {['BM_gemv_generic_kernel', half_tag], ['BM_mv_cublas', half_tag]}), ...
        make_spec('bench_mm_cuda_kernels.csv',      'CUDA', {}, {['BM_gemm_generic_kernel', half_tag], ['BM_mm_cublas', half_tag], 'BM_mm_csr_cuda'}), ...
        make_spec('bench_fp32_16_cuda_kernels.csv', 'CUDA', {}, {'BM_mv_fp32_16_pure_half', 'BM_mm_fp32_16_pure_half'}) ...
    };

    fp16_tasks = gather_tasks(build_root, fp16_group);
    fp32_tasks = gather_tasks(build_root, fp32_group);

    plot_shape(fp16_tasks, 2, 'FP16-compute kernels (mv)', fullfile(out_dir, 'fp16_compute_comparison_mv.png'));
    plot_shape(fp16_tasks, 3, 'FP16-compute kernels (mm)', fullfile(out_dir, 'fp16_compute_comparison_mm.png'));
    plot_shape(fp32_tasks, 2, 'FP32-compute kernels (mv)', fullfile(out_dir, 'fp32_compute_comparison_mv.png'));
    plot_shape(fp32_tasks, 3, 'FP32-compute kernels (mm)', fullfile(out_dir, 'fp32_compute_comparison_mm.png'));
end

function s = make_spec(csv_name, tag, include, exclude)
    s = struct('csv', csv_name, 'tag', tag, 'include', {include}, 'exclude', {exclude});
end

% Returns a struct array, one entry per distinct benchmark task, with:
%   label     - "TAG | function_name" (unique across the whole group)
%   ndims     - 2 for mv-shaped tasks, 3 for mm-shaped tasks
%   sizes     - containers.Map from "M_N" / "M_K_N" key -> GFlop/s value,
%               covering every size that task was actually tested at
function tasks = gather_tasks(build_root, specs)
    tasks = struct('label', {}, 'ndims', {}, 'sizes', {});
    for i = 1:numel(specs)
        s = specs{i};
        csv_path = find_one_csv(build_root, s.csv);
        if isempty(csv_path)
            warning('compare_precision:missingCsv', ...
                'Skipping %s (%s) — CSV not found under %s. Run that benchmark first.', ...
                s.csv, s.tag, build_root);
            continue
        end

        T = readtable(csv_path, 'TextType', 'string');
        if ~any(strcmp('benchmark', T.Properties.VariableNames))
            warning('compare_precision:emptyCsv', ...
                'Skipping %s (%s) — %s has no "benchmark" column (likely empty/truncated, e.g. from a run that never reached Finalize()). Rerun that benchmark.', ...
                s.csv, s.tag, csv_path);
            continue
        end
        n = height(T);
        names = cellstr(T.benchmark);
        prefixes = strings(n, 1);
        keys = strings(n, 1);
        ndims_col = zeros(n, 1);
        valid = false(n, 1);
        for k = 1:n
            [pfx, dims] = parse_name(names{k});
            if isempty(pfx)
                continue
            end
            prefixes(k) = pfx;
            keys(k) = strjoin(string(dims), '_');
            ndims_col(k) = numel(dims);
            valid(k) = true;
        end

        keep = valid;
        if ~isempty(s.include)
            keep = keep & starts_with_any(prefixes, s.include);
        end
        if ~isempty(s.exclude)
            keep = keep & ~starts_with_any(prefixes, s.exclude);
        end

        uniq_prefixes = unique(prefixes(keep));
        for u = 1:numel(uniq_prefixes)
            idx = find(keep & prefixes == uniq_prefixes(u));
            m = containers.Map('KeyType', 'char', 'ValueType', 'double');
            for j = 1:numel(idx)
                r = idx(j);
                m(char(keys(r))) = get_numeric_column(T, r, 'GFlop_s');
            end
            task.label = sprintf('%s | %s', s.tag, char(uniq_prefixes(u)));
            task.ndims = ndims_col(idx(1));
            task.sizes = m;
            tasks(end + 1) = task; %#ok<AGROW>
        end
    end
end

function tf = starts_with_any(strs, prefix_list)
    tf = false(size(strs));
    for i = 1:numel(prefix_list)
        tf = tf | startsWith(strs, string(prefix_list{i}));
    end
end

function [prefix, dims] = parse_name(name)
    % Benchmark names look like:
    %   BM_mv_fp32_16_pure_half/65536/65536/real_time
    %   BM_mm_strided<double, double, double>/8192/8192/8192/real_time
    tok = regexp(name, '^(.*?)/(\d+(?:/\d+){1,2})/real_time$', 'tokens', 'once');
    if isempty(tok)
        prefix = "";
        dims = [];
        return
    end
    prefix = string(tok{1});
    dims = str2double(strsplit(tok{2}, '/'));
end

function val = get_numeric_column(T, row, col)
    if any(strcmp(col, T.Properties.VariableNames))
        val = T.(col)(row);
    else
        val = NaN;
    end
end

function csv_path = find_one_csv(root, name)
    matches = dir(fullfile(root, '**', name));
    if isempty(matches)
        csv_path = '';
    else
        csv_path = fullfile(matches(1).folder, matches(1).name);
    end
end

% Finds the largest size (by product of dims) tested by EVERY task of the
% given shape, then plots one bar per task at that exact common size.
function plot_shape(tasks, ndims_want, title_str, out_path)
    sel = tasks([tasks.ndims] == ndims_want);
    if isempty(sel)
        warning('compare_precision:emptyShape', 'No %dD tasks for "%s" — nothing to plot.', ndims_want, title_str);
        return
    end

    common_keys = keys(sel(1).sizes);
    for i = 2:numel(sel)
        common_keys = intersect(common_keys, keys(sel(i).sizes));
    end
    if isempty(common_keys)
        warning('compare_precision:noCommonSize', ...
            '"%s": no problem size is common to all %d tasks — nothing to plot. Tasks: %s', ...
            title_str, numel(sel), strjoin({sel.label}, '; '));
        return
    end

    % Prefer the largest total size; break ties toward the most "square"
    % shape (smallest max/min dim ratio) since that's the more intuitive
    % reference point, and equal-total shapes are otherwise interchangeable
    % for a fair comparison.
    best_total = -1;
    best_skew = Inf;
    best_key = '';
    for i = 1:numel(common_keys)
        dims = str2double(strsplit(common_keys{i}, '_'));
        total = prod(dims);
        skew = max(dims) / min(dims);
        if total > best_total || (total == best_total && skew < best_skew)
            best_total = total;
            best_skew = skew;
            best_key = common_keys{i};
        end
    end
    common_dims_str = strrep(best_key, '_', 'x');

    labels = {sel.label};
    gflops = zeros(1, numel(sel));
    for i = 1:numel(sel)
        gflops(i) = sel(i).sizes(best_key);
    end
    x = 1:numel(sel);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1500, 700]);
    bar(x, gflops);
    set(gca, 'XTick', x, 'XTickLabel', labels, 'XTickLabelRotation', 60, ...
        'TickLabelInterpreter', 'none');
    ylabel('GFlop/s');
    title(sprintf('%s — all tasks @ %s', title_str, common_dims_str), 'Interpreter', 'none');
    grid on;
    ylim([0, max(gflops) * 1.15]);

    saveas(fig, out_path);
    close(fig);
    fprintf('Wrote %s (common size %s, %d tasks)\n', out_path, common_dims_str, numel(sel));
end
