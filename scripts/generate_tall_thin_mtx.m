function [A, x_true, b] = generate_tall_thin_mtx(m, n, k, sv_lower, condition_number, distribution, output_path, x_output_path, b_output_path, normalize_spectrum)
%GENERATE_TALL_THIN_MTX Synthesize a tall-and-thin (m >> n) test matrix
%   with a PRESCRIBED singular value spectrum, a random Gaussian "true"
%   solution x_true, and the right-hand side b = A*x_true it induces --
%   a complete (A, b) least-squares test problem with a KNOWN exact
%   answer (x_true itself) to check a solver's output against, not just
%   A alone. All three are written to MatrixMarket (.mtx) files this
%   codebase's own reader can load directly (see cpu::read_strided_storage
%   in src/core/memory/dense_storage.cpp / src/io/mmio.c). Intended for
%   feeding LSQR (test/core/solver/test_lsqr.cpp) and other
%   least-squares/iterative-solver code real controlled-conditioning
%   inputs instead of the identity/diagonal/random systems built
%   directly in-code today -- this repo otherwise has no test-matrix
%   generator at all (no .mtx files, no MATLAB writer) prior to this
%   script.
%
%   Construction: A = U * diag(s) * V, a thin/compact-SVD-style build
%   run in reverse, so A's own singular values are EXACTLY s (up to
%   floating-point round-off), regardless of m, n, or how U/V happen to
%   be drawn:
%     U (m x k) -- orthonormal COLUMNS (U'*U = I_k): a random m x k
%       Gaussian matrix pushed through an economy QR factorization.
%     diag(s) (k x k) -- the k prescribed singular values, see
%       `distribution` below.
%     V (k x n) -- orthonormal ROWS (V*V' = I_k): a random n x k
%       Gaussian matrix pushed through an economy QR factorization, then
%       transposed.
%   Requires k <= min(m, n) (necessary for U/V to actually be
%   orthonormal). The common case for a full-column-rank tall/thin test
%   matrix is k == n, giving A full column rank n; k < n instead
%   produces a rank-k (rank-deficient) tall/thin matrix.
%
%   condition_number sets cond(A) = sv_upper/sv_lower DIRECTLY (require
%   condition_number >= 1) -- sv_upper = sv_lower * condition_number is
%   computed internally, so callers reason about the one number that
%   actually matters for how hard the system is to solve (and, e.g., for
%   whether ||b|| stays within a given --value-type's representable
%   range -- see the note on FP16 overflow below) instead of having to
%   pick sv_lower/sv_upper by hand and divide them themselves.
%
%   distribution selects how the k singular values are spread across
%   [sv_lower, sv_lower*condition_number]:
%     'linear'    -- s = linspace(sv_lower, sv_upper, k), i.e. equal
%                    ARITHMETIC spacing. Same cond(A) as 'geometric'
%                    below for the same bounds, but the spectrum
%                    clusters near sv_upper for any k > 2 -- the
%                    arithmetic mean of k linearly-spaced points skews
%                    toward the top of the range.
%     'geometric' -- s(i) = sv_lower * condition_number^((i-1)/(k-1)),
%                    i.e. equal spacing in LOG space (a true geometric
%                    sequence from sv_lower to sv_upper). Same condition
%                    number as 'linear', but the spectrum is spread
%                    evenly across ORDERS OF MAGNITUDE rather than
%                    clustering near sv_upper -- the more common choice
%                    for stress-testing a solver's behavior across a
%                    wide dynamic range (mirrors the classic randsvd-
%                    style geometric singular-value test-matrix
%                    convention).
%   k == 1 is degenerate for both: s = sv_lower (a single singular
%   value; condition_number is ignored, since there is no second point
%   to interpolate to).
%
%   normalize_spectrum (default false) rescales the ENTIRE matrix A by a
%   single scalar, 1/sv_upper, AFTER the spectrum-controlled construction
%   above -- i.e. A becomes A/sv_upper, so the largest singular value of
%   the written A is exactly 1 (up to round-off) and the smallest is
%   exactly 1/condition_number. Since U and V are orthonormal, this is
%   IDENTICAL to dividing every entry of s by sv_upper before forming
%   A = U*diag(s)*V -- a uniform rescale of the whole spectrum, not a
%   per-column one. b is then computed from THIS already-rescaled A
%   (b = A*x_true), so the written (A, b) pair is self-consistent end to
%   end, same convention as the option this replaces.
%   UNLIKE a former per-column-normalization option this replaces,
%   uniform spectrum scaling is EXACTLY singular-value-ratio-preserving:
%   cond(A) = sv_upper/sv_lower is a ratio, and dividing every singular
%   value by the same constant leaves that ratio, and therefore cond(A),
%   completely unchanged -- the condition_number/sv_lower/distribution
%   requested above are realized EXACTLY, not just approximately.
%   Motivation: bounding A's overall scale (‖A‖_2 = sv_upper, normalized
%   to 1) gives a principled, predictable ceiling on entry magnitude for
%   low-precision (e.g. FP16) matrix storage/matvecs -- a real
%   improvement over per-column normalization, which only bounds each
%   column's own norm and can still leave huge within-column magnitude
%   spread (one dominant entry, everything else tiny) even at unit
%   column norm.
%
%   x_true (n x 1) is a plain random Gaussian vector (randn(n,1)) -- the
%   "true" solution this test problem is built around. b (m x 1) is
%   A*x_true EXACTLY (computed in floating point, no added noise, using
%   the final A -- rescaled or not, per normalize_spectrum above): since
%   A is tall/thin (assume full column rank, k == n), this b is the
%   consistent/overdetermined case (residual ~0 at x_true, up to
%   round-off) -- the LSQR system LsqrSolveTest.DenseOverdeterminedConsistent
%   (test/core/solver/test_lsqr.cpp) already exercises in miniature, now
%   at a real, controllable scale/conditioning.
%
%   NOTE on choosing condition_number for low-precision solves: ||b|| is
%   NOT bounded by sv_upper alone -- for a random x_true it's typically
%   dominated by the largest singular direction, so ||b|| can itself run
%   up to roughly sv_upper (here, sv_lower*condition_number). A solver
%   storing internal norms in a narrow-range type (e.g. lsqr_solve's own
%   --value-type fp16, max representable magnitude ~65504) will silently
%   overflow to Inf/NaN the moment ||b|| exceeds that range -- keep
%   sv_lower*condition_number comfortably under whatever range limit
%   applies if you need a low-precision solve to actually run to
%   completion, rather than just to observe where it breaks.
%
%   output_path, x_output_path, and b_output_path each write A, x_true,
%   and b respectively as a MatrixMarket "array" (dense) file: banner
%   line "%%MatrixMarket matrix array real general", a single "M N"
%   size line, then M*N entries in COLUMN-MAJOR order, one per line,
%   each "%1.16e" scientific notation -- no comment lines, no
%   separators, byte-for-byte the same layout tomos's own
%   write_strided_storage (src/core/memory/dense_storage.cpp) produces.
%   x_output_path and b_output_path are REQUIRED (x_true/b are always
%   generated and always written -- this is the whole point of this
%   version of the script). output_path may be omitted (or passed as
%   '') to skip writing A itself and just return it in memory.
%
%   Example (tall/thin, full column rank, geometric spectrum spanning
%   6 orders of magnitude, condition number 1e6, writing A/x_true/b):
%     A = generate_tall_thin_mtx(100000, 50, 50, 1, 1e6, 'geometric', ...
%                                 'test_matrices/tall_thin_geom_1e6.mtx', ...
%                                 'test_matrices/tall_thin_geom_1e6_x.mtx', ...
%                                 'test_matrices/tall_thin_geom_1e6_b.mtx');
%
%   Example (rank-deficient: 200000 x 100 but only rank 60, linear
%   spectrum, condition number 100):
%     A = generate_tall_thin_mtx(200000, 100, 60, 1, 100, 'linear', ...
%                                 'test_matrices/tall_thin_rankdef.mtx', ...
%                                 'test_matrices/tall_thin_rankdef_x.mtx', ...
%                                 'test_matrices/tall_thin_rankdef_b.mtx');
%
%   Example (same shape/spectrum request as the first example, but with
%   the spectrum rescaled so the largest singular value is 1 -- b is
%   built from the rescaled A; cond(A) stays exactly 1e6):
%     A = generate_tall_thin_mtx(100000, 50, 50, 1, 1e6, 'geometric', ...
%                                 'test_matrices/tall_thin_normspec.mtx', ...
%                                 'test_matrices/tall_thin_normspec_x.mtx', ...
%                                 'test_matrices/tall_thin_normspec_b.mtx', ...
%                                 true);
%
%   To size m for a target GPU instead of picking it by hand, see
%   max_rows_for_gpu.m (same directory) -- e.g. for n = 2000 columns on
%   an A100 40GB, computed at float32:
%     m = max_rows_for_gpu(2000, 40, 'single');
%     A = generate_tall_thin_mtx(m, 2000, 2000, 1, 1e6, 'geometric', ...
%                                 'test_matrices/tall_thin_a100_40gb.mtx', ...
%                                 'test_matrices/tall_thin_a100_40gb_x.mtx', ...
%                                 'test_matrices/tall_thin_a100_40gb_b.mtx');

    if nargin < 7
        output_path = '';
    end
    if nargin < 10 || isempty(normalize_spectrum)
        normalize_spectrum = false;
    end
    if nargin < 9
        error('generate_tall_thin_mtx:missing_output_paths', ...
              ['x_output_path and b_output_path are required -- x_true ' ...
               'and b are always generated and must be written ' ...
               'somewhere.']);
    end
    if isempty(x_output_path) || isempty(b_output_path)
        error('generate_tall_thin_mtx:empty_output_paths', ...
              'x_output_path and b_output_path must be non-empty paths.');
    end

    if k > min(m, n)
        error('generate_tall_thin_mtx:k_too_large', ...
              'k (%d) must be <= min(m, n) (%d) for U/V to be orthonormal.', ...
              k, min(m, n));
    end
    if sv_lower <= 0 || condition_number < 1
        error('generate_tall_thin_mtx:bad_bounds', ...
              ['Require sv_lower > 0 and condition_number >= 1 (got ' ...
               'sv_lower=%g, condition_number=%g).'], ...
              sv_lower, condition_number);
    end
    sv_upper = sv_lower * condition_number;

    switch distribution
        case 'linear'
            if k == 1
                s = sv_lower;
            else
                s = linspace(sv_lower, sv_upper, k);
            end
        case 'geometric'
            if k == 1
                s = sv_lower;
            else
                s = sv_lower * condition_number .^ linspace(0, 1, k);
            end
        otherwise
            error('generate_tall_thin_mtx:bad_distribution', ...
                  'distribution must be ''linear'' or ''geometric'' (got ''%s'').', ...
                  distribution);
    end

    % Economy QR of a random Gaussian matrix gives genuinely orthonormal
    % columns (not just approximately, e.g. via Gram-Schmidt drift), so
    % A's singular values below come out EXACTLY s up to round-off.
    [U, ~] = qr(randn(m, k), 0);
    [Vt, ~] = qr(randn(n, k), 0);
    V = Vt';

    A = U * (s(:) .* V);  % U * diag(s) * V without forming diag(s)

    if normalize_spectrum
        % Uniform rescale of the whole spectrum (equivalently, of A itself,
        % since U/V are orthonormal): divides every singular value by
        % sv_upper, so the largest becomes exactly 1 -- cond(A) is a ratio,
        % so this leaves it exactly unchanged (see this function's own
        % header comment for the full derivation).
        A = A / sv_upper;
    end

    x_true = randn(n, 1);
    b = A * x_true;  % uses the (possibly normalized) A above -- b is self-consistent with whatever A ends up being written

    if ~isempty(output_path)
        write_mtx_array(A, output_path);
    end
    write_mtx_array(x_true, x_output_path);
    write_mtx_array(b, b_output_path);
end

function write_mtx_array(A, filepath)
    [m, n] = size(A);
    fid = fopen(filepath, 'w');
    if fid == -1
        error('generate_tall_thin_mtx:write_failed', ...
              'Could not open "%s" for writing.', filepath);
    end
    fprintf(fid, '%%%%MatrixMarket matrix array real general\n');
    fprintf(fid, '%d %d\n', m, n);
    % MATLAB/Octave iterate a matrix argument to fprintf in COLUMN-MAJOR
    % order -- exactly the layout tomos's own reader/writer expect, so no
    % explicit loop or transpose is needed here.
    fprintf(fid, '%1.16e\n', A);
    fclose(fid);
end
