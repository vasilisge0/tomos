function plot_gemv_fp32_recon_vs_cutlass_ref(csv_path, tag)
%PLOT_GEMV_FP32_RECON_VS_CUTLASS_REF Plot the fp32-compute GEMV sweep from
%   scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh (which itself
%   builds and runs benchmarks/kernels/cuda/matrix/mv_fp32_recon_vs_
%   cutlass_ref.cu): cuBLAS (Sgemv), real CUTLASS (fp32 SIMT device::
%   Gemv), this codebase's own tuned native (unpacked) fp32 GEMV kernel
%   (mv_fp32_tiled), and the fp32-compute tier of both packed-precision
%   formats (fp32_16 via mv_fp32_16_accurate, fp32_bf16 via mv_fp32_
%   bf16_accurate). Mirrors plot_gemm_format_32_16_compute_32.m's own
%   structure (that script's GEMM analog) -- see that file's own header
%   comment for the fuller derivation of the underlying kernel families
%   this reuses.
%
%   fp32_16 has THREE plotted variants (plain, "(workspace)", and
%   "(regular)"), fp32_bf16 has only two -- the underlying kernels each
%   series sources from differ between the two formats, not a naming
%   coincidence:
%
%   fp32_16 (plain) = mv_fp32_16_accurate_ready4_atomiconly_scaled_kblock:
%   4-rows/lane row-vectorized (packed uint64_t loads)/prescale+
%   atomicAdd/K-scaled-K_BLOCK structure, decoding via the FULL fp32_16_
%   combine_branchless (all three of its normal/extended/degraded
%   reconstruction paths, computed unconditionally) instead of the cheap
%   "normal"-only fast_normal shortcut used by "(regular)" below -- so
%   this one is safe on arbitrary caller data, the same correctness bar
%   as mv_fp32_16_accurate itself, and needs no external workspace/
%   scratch buffer of any kind. Won outright on real hardware over its
%   2-row predecessor (1.070x-1.115x, tall-and-thin M sweep at K=4096;
%   square M=K shapes not separately confirmed before promotion).
%
%   fp32_16 (regular) = mv_fp32_16_accurate_ready_atomiconly_scaled_
%   kblock_fast_normal: the SAME structure as plain above, but decoding
%   via fp32_16_combine_fast_normal (only the cheap "normal" path) --
%   UNSAFE on arbitrary data (the caller must independently guarantee no
%   extended/degraded-encoded values exist), but faster where that
%   contract holds. Deliberately kept as its own separate series, not
%   merged into plain, even though both trace an almost identical shape
%   -- the two differ specifically in the decode cost/safety tradeoff,
%   which is exactly the comparison this series pair exists to show.
%
%   fp32_bf16 has two plotted variants, but unlike fp32_16's plain/
%   "(implicit)" pair (relabeled from "(fast)" so that label is free for a
%   genuinely different, SAFE decode-fast series -- fp32_16_half_only_fast
%   -- ported into this format's pure_half-tier plots elsewhere, since
%   this recon/exact tier has no analogous shift-based decode work to
%   skip), the difference is the ENCODING each was split with, not
%   a safety/speed tradeoff on the same data -- bfloat16's own decode
%   needs no unsafe-on-arbitrary-data shortcut in the first place (see
%   fp32_bf16_pack.hpp's own header comment), so there's no "(implicit)"-style
%   series here at all:
%
%   fp32_bf16 (plain) = mv_fp32_bf16_accurate_ready_atomiconly_scaled_
%   kblock_rounded: the SAME row-vectorized/atomiconly/scaled-K_BLOCK fast
%   path as fp32_16's own plain series, but decoding via fp32_bf16_
%   combine_rounded against data split with fp32_bf16_split_rounded --
%   wh is a PROPERLY ROUNDED (not truncated) standalone bfloat16 value,
%   and wr's bit0 is a round_flag (recording whether that rounding
%   stepped wh up by 1) rather than a real mantissa bit, so decode has to
%   subtract it back out. See fp32_bf16_split_rounded/fp32_bf16_combine_
%   rounded's own comments in fp32_bf16_pack.hpp for the full derivation
%   and accuracy bound (negligible next to bf16's own ~2^-8 relative
%   precision).
%
%   fp32_bf16 (truncated) = mv_fp32_bf16_accurate_ready_atomiconly_
%   scaled_kblock_workspace, the ORIGINAL fp32_bf16 GEMV series (was
%   plain "TOMOS FP32/BF16" before mv_fp32_bf16_accurate_ready_
%   atomiconly_scaled_kblock_rounded was added and took over that slot):
%   decodes via plain fp32_bf16_combine against data split with plain
%   fp32_bf16_split (wh formed by TRUNCATION, wr the exact remaining
%   mantissa bits -- bit-exact reconstruction, no approximation at all,
%   see fp32_bf16_pack.hpp's own header comment). Generalized to every
%   TransposeType/ld_a via a caller-provided workspace buffer instead of
%   falling back to a malloc/free-per-call function, same as the
%   "(rounded)"/plain series above.
%
%   CUTLASS is deliberately NOT plotted (see the comment just above the
%   series list below), even though its cutlass_ms/cutlass_gflops columns
%   still exist in the CSV. mv_fp32_tiled is this codebase's first tuned
%   native fp32 GEMV kernel of any kind (previously only a naive gemv_
%   generic_kernel and cuBLAS's own Sgemv existed).
%
%   plot_gemv_fp32_recon_vs_cutlass_ref()          uses the most recently
%       written mv_fp32_recon_vs_cutlass_ref_*.csv under
%       <repo>/build/mv_fp32_recon_vs_cutlass_ref_output/ (that directory
%       is shared with scripts/benchmarks_gemv_fp32_recon_vs_cutlass_
%       ref.sh's own runs, so this picks up whichever run happened last).
%   plot_gemv_fp32_recon_vs_cutlass_ref(csvPath)   reads that CSV directly.
%
%   The CSV is expected to have M/K columns (BLAS convention: A is MxK,
%   x is length K, y is length M) plus *_ms/*_gflops columns per method
%   -- see benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh's own embedded
%   parser for the exact schema. Rows are split into two groups by shape:
%     - square:          M == K            -> x-axis is M (=K).
%     - tall-and-thin:   M ~= K (M >> K)    -> x-axis is M, one line per
%                                              fixed K value would need a
%                                              separate plot per K in
%                                              principle, but since this
%                                              sweep only holds K at two
%                                              values (512, 4096) and M
%                                              always dominates, both are
%                                              overlaid on the same axes
%                                              here -- see plot_series's
%                                              own tall-and-thin handling.
%   ("SKIPPED" cells, from a shape a method ran out of device memory for,
%   become NaN and simply leave a gap in that method's line.)
%
%   tag (optional) is appended (as "_<tag>") to both output basenames --
%   used by scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref_advanced.sh
%   to write "_avg"-suffixed plots (averaged over a caller-controlled
%   warmup/iteration count, see that script's own header) alongside,
%   without overwriting, this function's own default-named plots. Omit
%   it (or pass '') for the untagged filenames below.
%
%   Writes, under <repo>/build/benchmarks/output/plots/:
%     plot_gemv_fp32_recon_vs_cutlass_ref_square[_<tag>].png / .fig
%     plot_gemv_fp32_recon_vs_cutlass_ref_tall[_<tag>].png   / .fig

    here = fileparts(mfilename('fullpath'));
    repo_root = fileparts(here);

    if nargin < 2
        tag = '';
    end
    tag = char(tag);
    tag_suffix = '';
    if ~isempty(tag)
        tag_suffix = ['_' tag];
    end

    if nargin < 1 || isempty(csv_path)
        default_dir = fullfile(repo_root, 'build', 'mv_fp32_recon_vs_cutlass_ref_output');
        csv_path = find_latest_csv(default_dir);
        if isempty(csv_path)
            error('plot_gemv_fp32_recon_vs_cutlass_ref:noCsv', ...
                ['No mv_fp32_recon_vs_cutlass_ref_*.csv found under %s.\n' ...
                 'Run scripts/benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh first.'], default_dir);
        end
    end
    if ~isfile(csv_path)
        error('plot_gemv_fp32_recon_vs_cutlass_ref:missingCsv', 'CSV not found: %s', csv_path);
    end

    out_dir = fullfile(repo_root, 'build', 'benchmarks', 'output', 'plots');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_path, 'TextType', 'string');

    M = to_double_col(T, 'M');
    K = to_double_col(T, 'K');

    % Dashed (vs every TOMOS series' solid line) since this is a reference
    % baseline (vendor library), not a TOMOS kernel -- same convention
    % plot_gemm_format_32_16_compute_32.m's own series use. CUTLASS is
    % deliberately NOT plotted here (its own numbers are a documented,
    % non-tunable naive baseline, not a meaningful comparison point -- see
    % mv_fp32_recon_vs_cutlass_ref.cu's own header comment), even though
    % the CSV still carries its cutlass_ms/cutlass_gflops columns.
    series = struct('label', {}, 'values', {}, 'style', {});
    i = 0;
    i = i + 1; series(i) = struct('label', 'cuBLAS', 'values', to_double_col(T, 'cublas_gflops'), 'style', '--o');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32', 'values', to_double_col(T, 'native_gflops'), 'style', '-^');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16', 'values', to_double_col(T, 'fp32_16_gflops'), 'style', '-d');
    % REMOVED from the plot (commented out, not deleted -- its CSV column
    % is still populated by every run of this benchmark, so re-enabling
    % is a one-line uncomment): "TOMOS FP32 / FP16 (no subnormals)" --
    % mv_fp32_16_accurate_ready_atomiconly_scaled_kblock_fast_normal,
    % decoding via fp32_16_combine_fast_normal. UNSAFE on arbitrary data
    % -- computes ONLY the "normal" (control=0) reconstruction path, with
    % no underflow (extended/degraded) or overflow (saturate) handling at
    % all. NAMING CAVEAT: "(no subnormals)" is shorthand -- the actual
    % assumption is stricter than "no FP32 subnormals" (~1.18e-38): it's
    % silently WRONG for ANY element whose original FP32 magnitude falls
    % below fp16's own normal-range floor (~6.1e-5, 2^-14), a much easier
    % bar to hit than a true FP32 subnormal, so this assumes away a
    % realistic band of ordinary small (not FP32-subnormal) values too.
    % Stricter/riskier contract than "TOMOS FP32 / FP16 (async)" above
    % (which only assumes no overflow).
    % i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (no subnormals)', 'values', to_double_col(T, 'fp32_16_regular_gflops'), 'style', '-*');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16', 'values', to_double_col(T, 'fp32_bf16_gflops'), 'style', '-+');
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / BF16 (truncated)', 'values', to_double_col(T, 'fp32_bf16_truncated_gflops'), 'style', '-v');
    % "TOMOS FP32 / FP16 (async)" -- PROMOTED from "test4": mv_fp32_16_
    % ready8_atomiconly_scaled_kblock_cpasync_no_overflow. Combines
    % cp.async double-buffered prefetch of wh/wr with 8-rows/lane
    % register-blocking AND fp32_16_combine_branchless_no_overflow (skips
    % the OVERFLOW escape path entirely). Named "(async)" for its actual
    % performance driver -- cp.async double-buffered prefetch plus the
    % vectorized uint4 loads/8-row register blocking, NOT a SIMD/paired
    % decode trick (the decode itself, fp32_16_combine_branchless_no_
    % overflow, is called once per scalar (wh,wr) element, same as plain
    % "TOMOS FP32 / FP16"'s own decode) -- deliberately NOT named "(fast)"
    % (that name is already used, with a DIFFERENT meaning -- lossy
    % underflow-to-zero -- by "TOMOS FP16 (implicit, fast)" on the
    % pure_half plots; reusing it here for an unrelated safe/unsafe axis
    % would be misleading). Still UNSAFE on arbitrary data -- valid only
    % under the caller's guarantee this matrix's data never needs wh's
    % overflow encoding (only ever underflow, if anything). Measured
    % (square sweep) up to ~1.16x plain "TOMOS FP32 / FP16" at large M
    % (approaching native fp32 parity, e.g. 701.7 vs native's 700.1
    % GFlop/s at M=K=65536) but BELOW its own 4-row sibling ("test3", not
    % promoted) at small/mid M (<~12288, occupancy loss from halved
    % gridRows outweighs the extra ILP there) -- see experiments-gemv.md
    % for the full derivation and the complete measured table across all
    % five "test" candidates this was chosen from.
    i = i + 1; series(i) = struct('label', 'TOMOS FP32 / FP16 (async)', 'values', to_double_col_optional(T, 'fp32_16_test4_gflops', numel(M)), 'style', ':s');
    % "test"/"test2"/"test3"/"test5" -- the reusable, throwaway "test"
    % slots' own remaining candidates from the investigation that led to
    % "TOMOS FP32 / FP16 (async)" above (see experiments-gemv.md for the
    % full writeup, and benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh's own
    % METHOD_MARKERS comment for each one's own kernel/history). Commented
    % out, not deleted -- their CSV columns are still populated by every
    % run of this benchmark, so re-enabling any of these for a fresh
    % comparison is a one-line uncomment:
    %
    % i = i + 1; series(i) = struct('label', 'test', 'values', to_double_col_optional(T, 'fp32_16_test_gflops', numel(M)), 'style', '-.p');
    % i = i + 1; series(i) = struct('label', 'test2', 'values', to_double_col_optional(T, 'fp32_16_test2_gflops', numel(M)), 'style', ':d');
    % i = i + 1; series(i) = struct('label', 'test3', 'values', to_double_col_optional(T, 'fp32_16_test3_gflops', numel(M)), 'style', '-.x');
    % i = i + 1; series(i) = struct('label', 'test5', 'values', to_double_col_optional(T, 'fp32_16_test5_gflops', numel(M)), 'style', '--^');

    is_square = (M == K);
    is_tall = ~is_square;

    plot_series(M, series, is_square, 'Matrix dimensions (m = k)', ...
        'square', fullfile(out_dir, ['plot_gemv_fp32_recon_vs_cutlass_ref_square' tag_suffix]));

    plot_series(M, series, is_tall, 'Row count m (k held fixed -- see legend/title)', ...
        'tall', fullfile(out_dir, ['plot_gemv_fp32_recon_vs_cutlass_ref_tall' tag_suffix]));
end

function v = to_double_col(T, name)
    if ~any(strcmp(name, T.Properties.VariableNames))
        error('plot_gemv_fp32_recon_vs_cutlass_ref:missingColumn', ...
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

% Same as to_double_col, but returns an all-NaN column (renders as an
% empty/absent line) instead of erroring when the CSV doesn't have this
% column at all -- for the "test" series, which is a reusable, throwaway
% slot (see benchmarks_gemv_fp32_recon_vs_cutlass_ref.sh's own
% METHOD_MARKERS comment) that may be absent from an older CSV generated
% before this plot script itself started plotting it. Same convention
% plot_gemv_fp32_16_pure_half_vs_cublas_ref.m's own to_double_col_optional
% already uses for its own "test"/"test2" series.
function v = to_double_col_optional(T, name, n_rows)
    if ~any(strcmp(name, T.Properties.VariableNames))
        v = NaN(n_rows, 1);
        return
    end
    v = to_double_col(T, name);
end

function csv_path = find_latest_csv(dir_path)
    csv_path = '';
    if ~exist(dir_path, 'dir')
        return
    end
    files = dir(fullfile(dir_path, 'mv_fp32_recon_vs_cutlass_ref_*.csv'));
    if isempty(files)
        return
    end
    [~, idx] = max([files.datenum]);
    csv_path = fullfile(files(idx).folder, files(idx).name);
end

function plot_series(x_all, series, mask, x_label, group_name, out_base)
    x = x_all(mask);
    if isempty(x)
        warning('plot_gemv_fp32_recon_vs_cutlass_ref:emptyGroup', ...
            'No rows for "%s" -- skipping.', group_name);
        return
    end
    [x_sorted, order] = sort(x);

    fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 900]);
    hold on;
    % Same fixed qualitative palette as plot_gemm_format_32_16_compute_
    % 32.m's own plot_series, sized with headroom past this plot's own
    % (smaller, 5-series) count.
    kPalette = [
        0.00 0.45 0.74;  % blue
        0.85 0.10 0.10;  % red
        0.00 0.60 0.30;  % green
        0.49 0.18 0.56;  % purple
        0.90 0.55 0.10;  % orange
        0.10 0.75 0.80;  % cyan
        0.85 0.20 0.60;  % magenta
        0.55 0.35 0.15;  % brown
        0.50 0.50 0.50;  % gray
        0.65 0.65 0.10;  % olive
        0.30 0.30 0.90;  % indigo
        0.20 0.70 0.40;  % teal-green
    ];
    if numel(series) <= size(kPalette, 1)
        colors = kPalette(1:numel(series), :);
    else
        colors = [kPalette; lines(numel(series) - size(kPalette, 1))];
    end
    % Values are stored/read from the CSV in GFlop/s (matching the .cu
    % benchmark's own printed units) but plotted in GFlop/s directly here
    % (not converted to TFlop/s like the GEMM plot) -- GEMV's own
    % arithmetic intensity is fixed regardless of size (bandwidth-bound,
    % not compute-bound), so these numbers stay in the hundreds-to-low-
    % thousands GFlop/s range throughout the sweep rather than climbing
    % into the TFlop/s range GEMM reaches.
    max_y = 0;
    for i = 1:numel(series)
        y = series(i).values(mask);
        y = y(order);
        if any(~isnan(y))
            max_y = max(max_y, max(y, [], 'omitnan'));
        end
        plot(x_sorted, y, series(i).style, 'LineWidth', 4.5, 'MarkerSize', 6, ...
            'Color', colors(i, :), 'DisplayName', series(i).label);
    end
    hold off;

    % Legend placed below the plot (southoutside, 3 columns) rather than
    % inside it -- with 6 series, an inside legend covers curves.
    kAxisFontSize = 40;
    kLegendFontSize = 28;
    set(gca, 'XScale', 'log', 'XTick', unique(x_sorted), 'FontSize', kAxisFontSize, 'LineWidth', 2.5);
    xlabel(x_label, 'Interpreter', 'none', 'FontSize', kAxisFontSize);
    ylabel('GFlop/s', 'FontSize', kAxisFontSize);
    legend('Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'none', 'FontSize', kLegendFontSize);
    grid on;
    if max_y > 0
        ylim([0, max_y * 1.1]);
    end

    exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
    savefig(fig, [out_base '.fig']);
    close(fig);
    fprintf('Wrote %s.png and %s.fig (%d points)\n', out_base, out_base, numel(x_sorted));
end
