function m = max_rows_for_gpu(n, gpu_memory_gb, dtype, headroom_frac)
%MAX_ROWS_FOR_GPU How many rows an m x n dense matrix can have and still
%   fit on a GPU with gpu_memory_gb of memory, leaving headroom_frac of
%   that budget for the matrix itself (A) and the rest (1 - headroom_frac)
%   free for x/y vectors, solver workspace, CUDA context overhead, etc.
%   Meant to size the `m` argument of generate_tall_thin_mtx.m for a
%   known target GPU and column count n.
%
%   gpu_memory_gb is the RAW GPU memory in GB -- common values on this
%   project's hardware: 24 (A30), 40 or 80 (A100).
%
%   dtype is the precision the matrix will actually be loaded/computed
%   at on the GPU (NOT the .mtx file's own on-disk precision, which
%   generate_tall_thin_mtx.m always writes as double-precision text --
%   tomos::cpu::read_strided_storage<float> reads that text straight
%   into float/__half storage regardless): 'single'/'float'/'fp32'
%   (4 bytes/element, the default -- matches what every GEVM kernel in
%   this codebase actually computes at), 'double'/'fp64' (8 bytes), or
%   'half'/'fp16' (2 bytes).
%
%   headroom_frac (default 0.8) is the fraction of gpu_memory_gb budgeted
%   to A alone; the rest is left for everything else a real GEVM/LSQR
%   run also needs on the GPU at the same time.
%
%   Example (2000 columns, A100 40GB, float32, default 80% headroom,
%   condition number 1e6):
%     m = max_rows_for_gpu(2000, 40, 'single');
%     A = generate_tall_thin_mtx(m, 2000, 2000, 1, 1e6, 'geometric', ...
%                                 'test_matrices/tall_thin_a100_40gb.mtx', ...
%                                 'test_matrices/tall_thin_a100_40gb_x.mtx', ...
%                                 'test_matrices/tall_thin_a100_40gb_b.mtx');

    if nargin < 3 || isempty(dtype)
        dtype = 'single';
    end
    if nargin < 4 || isempty(headroom_frac)
        headroom_frac = 0.8;
    end
    if headroom_frac <= 0 || headroom_frac > 1
        error('max_rows_for_gpu:bad_headroom', ...
              'headroom_frac must be in (0, 1] (got %g).', headroom_frac);
    end

    switch dtype
        case {'single', 'float', 'fp32'}
            bytes_per_elem = 4;
        case {'double', 'fp64'}
            bytes_per_elem = 8;
        case {'half', 'fp16'}
            bytes_per_elem = 2;
        otherwise
            error('max_rows_for_gpu:bad_dtype', ...
                  ['dtype must be one of single/float/fp32, double/fp64, ' ...
                   'half/fp16 (got ''%s'').'], dtype);
    end

    budget_bytes = headroom_frac * gpu_memory_gb * 1e9;
    m = floor(budget_bytes / (n * bytes_per_elem));
end
