// One-shot LSQR runner: reads a matrix A, a "true" solution x_true, and a
// right-hand side b (all MatrixMarket .mtx, as written by
// scripts/generate_tall_thin_mtx.m) and solves A*x = b with LSQR at a
// caller-chosen combination of precisions:
//   --mv-type      A's own storage type (the "storage way" axis -- e.g.
//                  FP32 vs FP16).
//   --value-type   LSQR's internal Krylov-vector precision (u/v/w/temp1).
//   --refine-type  b/x's storage precision (the "residual"/solution axis).
// All three default to fp32, i.e. running with no flags at all gives you
// the "all float precision" case; pass --value-type fp16 with --mv-type
// fp32 or fp16 for the "float residual, half internal, FP32/FP16 storage
// sweep" case.
//
// NOTE: --mv-type fp32_16 selects NumericType::FP32_16, the packed wh/wr
// storage format used throughout this codebase's GEMV/GEMM kernel-tuning
// work (see fp32_16_pack.hpp). A stored this way, mv_apply(Dense&)
// (src/core/blas/blas.cpp) dispatches every per-iteration matvec straight
// to Fp32_16::lookup_mv_pure_half -- the tuned, cuBLAS-competitive kernel
// the GEVM plots label "TOMOS FP16" -- instead of cuBLAS or (for a mixed
// native-type triple like plain FP16 A against FP32 Krylov vectors) the
// slow, untiled gemv_generic_kernel fallback every other --mv-type still
// falls into whenever it differs from --value-type. fp32_16 is valid ONLY
// for --mv-type (A's own storage) -- --value-type/--refine-type/
// --precond-type stay restricted to fp64/fp32/fp16/bf16, since FP32_16
// isn't a single arithmetic type LSQR's Krylov-vector/refine/precond code
// paths can compute in directly.
//
// Reports: solve wall-clock time, LSQR iterations actually completed
// (via lsqr_solve's optional int* out-parameter -- see include/core/
// solver.hpp), final relative residual ||A*x-b||/||b||, and solution
// error ||x-x_true||/||x_true|| against the KNOWN exact answer embedded
// in the .mtx files by the generator.
//
// --precond adds a count-sketch + QR right-preconditioner (the classic
// sketch-and-precondition/Blendenpik-LSRN idiom, see src/core/precond/
// sketch_qr.hpp and test_lsqr.cpp's LsqrSketchPrecondTest for the
// reference pipeline this mirrors): sketch A down to k = 2*n rows with a
// CountSketch, QR-factor the k x n sketch, invert the n x n R factor, and
// hand that R^{-1} to LSQR as a right-preconditioner via add_precond().
// Built at --value-type precision (the same precision the preconditioner
// is applied to/from -- see LsqrWorkspace::precond_v/precond_t).
//
// IMPORTANT: apply_count_sketch's dense-source path (the S*A step) has no
// CUDA implementation anywhere in this codebase (dispatch::Sketch's
// apply_dense_ table is only ever populated for Device::CPU --
// src/dispatch/sketch_registries.cpp says so explicitly: "CUDA path
// omitted: mm_csr requires a cusparseHandle_t from Context"). So the
// preconditioner is ALWAYS built on a throwaway CPU Context from the
// already-in-memory host copy of A, regardless of --device -- then the
// resulting n x n R^{-1} is copied onto --device's Dense before
// add_precond(), mirroring exactly what LsqrSolveCudaTest.DensePreconditioned
// does. Only that small n x n matrix ever needs a device copy; the
// CPU-side sketch/QR/invert work is the same fixed cost either way.
//
// Usage:
//   lsqr_precision_runner --matrix A.mtx --x-true x.mtx --rhs b.mtx
//       [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T]
//       [--tol 1e-8] [--max-iters 200] [--max-relres 1e-8] [--restarts 1]
//       [--precond] [--normalize-columns] [--residual-history PATH]
//   T in {fp64,fp32,fp16,bf16} (case-insensitive; double/float/half/
//   bfloat16 also accepted). --mv-type additionally accepts fp32_16 (see
//   the NOTE above) -- not valid for --value-type/--refine-type/
//   --precond-type.

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas.hpp"
#include "context.hpp"
#include "countsketch.hpp"
#include "fp32_16_registries.hpp"
#include "matrix.hpp"
#include "mmio.h"
#include "precision.hpp"
#include "sketch_qr.hpp"
#include "solver.hpp"
#include "workspace.hpp"

using namespace tomos;

namespace {

// ---------------------------------------------------------------------------
// .mtx reading -- src/io/io.cpp is dead code (not compiled, has a column-
// major/row-major indexing bug -- see this file's own commit message /
// conversation for the trace), so this reads directly against the mmio.c
// API (which IS compiled, CMakeLists.txt links src/io/mmio.c) instead.
// Column-major, matching generate_tall_thin_mtx.m's writer and every
// StridedView<T> convention (values_[i + j*lead_dim_]) elsewhere in this
// codebase.
// ---------------------------------------------------------------------------
void read_mtx_host(const std::string& path, std::vector<double>& out, int& m,
                   int& n) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("could not open '" + path + "'");
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        std::fclose(f);
        throw std::runtime_error("'" + path + "': not a valid MatrixMarket file");
    }
    if (mm_read_mtx_array_size(f, &m, &n) != 0) {
        std::fclose(f);
        throw std::runtime_error("'" + path + "': could not read array size");
    }
    out.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (std::fscanf(f, "%lf", &out[static_cast<size_t>(i) +
                                            static_cast<size_t>(j) * m]) != 1) {
                std::fclose(f);
                throw std::runtime_error("'" + path + "': truncated/malformed data");
            }
        }
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// Runtime NumericType <-> compile-time type dispatch, for the 4 types LSQR's
// value/refine/precond axes actually support (FPTypes = double/float/
// __half/__nv_bfloat16 -- see src/dispatch/storage_registry_base.hpp).
// --mv-type additionally accepts fp32_16 (NumericType::FP32_16, see the file
// header comment) via parse_mv_precision below -- it's kept out of
// parse_precision/dispatch_precision since those two are also used for
// --value-type/--refine-type/--precond-type, none of which support it.
// ---------------------------------------------------------------------------
NumericType parse_precision(const std::string& raw) {
    std::string s;
    s.reserve(raw.size());
    for (char c : raw) s.push_back(static_cast<char>(std::tolower(c)));
    if (s == "fp64" || s == "double") return NumericType::FP64;
    if (s == "fp32" || s == "float" || s == "single") return NumericType::FP32;
    if (s == "fp16" || s == "half") return NumericType::FP16;
    if (s == "bf16" || s == "bfloat16") return NumericType::BF16;
    throw std::invalid_argument("unknown precision '" + raw +
                                "' (expected fp64/fp32/fp16/bf16)");
}

// --mv-type only: parse_precision plus fp32_16.
NumericType parse_mv_precision(const std::string& raw) {
    std::string s;
    s.reserve(raw.size());
    for (char c : raw) s.push_back(static_cast<char>(std::tolower(c)));
    if (s == "fp32_16" || s == "fp32-16") return NumericType::FP32_16;
    return parse_precision(raw);
}

const char* precision_name(NumericType t) {
    switch (t) {
        case NumericType::FP64: return "fp64";
        case NumericType::FP32: return "fp32";
        case NumericType::FP16: return "fp16";
        case NumericType::BF16: return "bf16";
        case NumericType::FP32_16: return "fp32_16";
        default: return "?";
    }
}

// Calls f(TagT{}) for the concrete C++ type matching `type`, where TagT is
// one of double/float/__half/__nv_bfloat16 -- f should be a generic lambda
// `[&](auto tag) { using T = decltype(tag); ... }`.
template <typename F>
void dispatch_precision(NumericType type, F&& f) {
    switch (type) {
        case NumericType::FP64: f(double{}); return;
        case NumericType::FP32: f(float{}); return;
        case NumericType::FP16: f(__half{}); return;
        case NumericType::BF16: f(__nv_bfloat16{}); return;
        default:
            throw std::invalid_argument(
                "dispatch_precision: unsupported NumericType (only fp64/"
                "fp32/fp16/bf16 are wired into LSQR)");
    }
}

// host double[] -> Dense of type T (CPU: direct cast/store; CUDA: cast on
// host into a typed staging buffer, then cudaMemcpy -- mirrors
// bench_lsqr_cuda.cu's own make_dense_consistent_system_cuda). Assumes
// dense was created via Dense::create_strided(..., {m,n}) with no custom
// lead_dim, so lead_dim_ == m (contiguous) -- true for every Dense this
// runner creates.
template <typename T>
void fill_dense_from_host(Device dev, Dense& dense, const std::vector<double>& src) {
    auto v = as_strided_view<T>(dense);
    size_t total = src.size();
    std::vector<T> typed(total);
    for (size_t k = 0; k < total; ++k) {
        if constexpr (std::is_same_v<T, double>) {
            typed[k] = src[k];
        } else if constexpr (std::is_same_v<T, float>) {
            typed[k] = static_cast<float>(src[k]);
        } else {
            typed[k] = static_cast<T>(static_cast<float>(src[k]));
        }
    }
    if (dev == Device::CPU) {
        std::memcpy(v.values_, typed.data(), total * sizeof(T));
    }
#if USE_CUDA
    else {
        cudaMemcpy(v.values_, typed.data(), total * sizeof(T), cudaMemcpyHostToDevice);
    }
#endif
}

// host double[] -> Dense of NumericType::FP32_16 (target must already be
// Dense::create_strided(..., NumericType::FP32_16, {m,n})): stages an FP32
// Dense from src, then packs it via dispatch::Fp32_16::lookup_split, the
// same split used throughout this codebase's GEVM/GEMM FP32_16 kernels
// (fp32_16_pack.hpp) -- there's no direct double[]/FP32_16 conversion since
// FP32_16 isn't a single arithmetic type fill_dense_from_host<T> can be
// instantiated with.
void fill_fp32_16_dense_from_host(Context& ctx, Device dev, Dense& target,
                                  const std::vector<double>& src, dim<2> size) {
    Dense staging = Dense::create_strided(ctx, Layout::COL_MAJOR,
                                          NumericType::FP32, size);
    fill_dense_from_host<float>(dev, staging, src);
    dispatch::Fp32_16::get().lookup_split(dev)(staging, target);
}

// Inverse of fill_dense_from_host: Dense of type T -> host double[].
template <typename T>
void read_dense_to_host(Device dev, Dense& dense, std::vector<double>& out) {
    auto v = as_strided_view<T>(dense);
    size_t total = v.size_[0] * v.size_[1];
    std::vector<T> typed(total);
    if (dev == Device::CPU) {
        std::memcpy(typed.data(), v.values_, total * sizeof(T));
    }
#if USE_CUDA
    else {
        cudaMemcpy(typed.data(), v.values_, total * sizeof(T), cudaMemcpyDeviceToHost);
    }
#endif
    out.resize(total);
    for (size_t k = 0; k < total; ++k) {
        if constexpr (std::is_same_v<T, double>) {
            out[k] = typed[k];
        } else {
            out[k] = static_cast<double>(static_cast<float>(typed[k]));
        }
    }
}

double rel_l2_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

struct Args {
    std::string matrix_path, x_true_path, rhs_path;
    std::string residual_history_path;  // empty = don't record
    Device device = Device::CUDA;
    NumericType mv_type = NumericType::FP32;
    NumericType value_type = NumericType::FP32;
    NumericType refine_type = NumericType::FP32;
    NumericType precond_type = NumericType::FP32;
    bool precond_type_set = false;  // if false, precond_type defaults to value_type once parsing finishes
    bool mv_type_set = false;       // if true, --mv-type was passed explicitly (see --implicit-duplicates)
    double tol = 1e-8;
    int max_iters = 200;
    double max_relres = 1e-8;
    int restarts = 1;
    bool precond = false;
    bool normalize_columns = false;
    bool explicit_duplicates = false;
    bool implicit_duplicates = false;
};

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --matrix A.mtx --x-true x.mtx --rhs b.mtx\n"
        "          [--device cpu|cuda] [--mv-type T] [--value-type T] [--refine-type T]\n"
        "          [--precond-type T]\n"
        "          [--tol TOL] [--max-iters N] [--max-relres R] [--restarts N]\n"
        "          [--precond] [--normalize-columns] [--explicit-duplicates]\n"
        "          [--implicit-duplicates] [--residual-history PATH]\n"
        "       T in {fp64,fp32,fp16,bf16}. All type flags default to fp32.\n"
        "       --mv-type additionally accepts fp32_16 (NumericType::\n"
        "       FP32_16, the packed wh/wr format -- see the file header\n"
        "       comment): routes every matvec to the tuned Fp32_16::\n"
        "       lookup_mv_pure_half kernel (the GEVM plots' \"TOMOS FP16\"),\n"
        "       its FP16 (\"pure_half\") compute tier. Not valid for\n"
        "       --value-type/--refine-type/--precond-type. See\n"
        "       --implicit-duplicates below to also use FP32_16's OTHER\n"
        "       compute tier (FP32, \"accurate\") for the residual checks.\n"
        "       --max-iters is iterations PER restart; --restarts (default 1)\n"
        "       is the number of restart cycles, so the true iteration\n"
        "       ceiling is restarts * max-iters. Each restart re-initializes\n"
        "       the Krylov basis from the current x -- convergence is\n"
        "       checked between restarts too, so extra restarts are\n"
        "       harmless once converged (the solve just stops early).\n"
        "       --precond adds a count-sketch + QR right-preconditioner\n"
        "       (k=2n sketch rows). Storage AND precond_apply() both run at\n"
        "       --precond-type -- independent of --mv-type (A's own storage/\n"
        "       matvec precision) and --value-type (the Krylov vectors\n"
        "       precond_apply's OUTPUT is added into). Defaults to whatever\n"
        "       --value-type is if not given explicitly (the sketch/QR/\n"
        "       invert pipeline itself always runs at FP32 or FP64\n"
        "       internally regardless -- LAPACK has no half-precision QR --\n"
        "       only the final R^-1 gets cast down to --precond-type).\n"
        "       --normalize-columns scales each column of A to unit norm\n"
        "       before solving (A' = A*D^-1, D = diag(column norms)) and\n"
        "       un-scales the solution afterward (x = D^-1*z) -- computed\n"
        "       in double precision from the .mtx data regardless of\n"
        "       --mv-type, to reduce A's own entry-magnitude spread before\n"
        "       it's cast down to a low-precision storage type. RHS b is\n"
        "       unaffected (A'*z = b exactly, same b, for z = D*x).\n"
        "       --explicit-duplicates: only meaningful when --mv-type !=\n"
        "       --refine-type. A is stored ONCE at --mv-type precision (for\n"
        "       fp16, PLAIN NATIVE __half -- no packing -- routed to a\n"
        "       genuinely-tuned real-fp16 GEMV kernel (mv_fp16_native), not\n"
        "       the generic mixed-precision fallback other unmatched type\n"
        "       combinations get), plus a SECOND copy built at --refine-type\n"
        "       precision, used for the residual recompute at every restart\n"
        "       boundary (u = b - A*x inside initialize()) -- so the\n"
        "       early-termination decision isn't capped by --mv-type's own\n"
        "       quantization noise, while the bulk of the per-iteration\n"
        "       matvecs still run at the cheaper --mv-type precision,\n"
        "       unchanged. Also used for this program's own final residual\n"
        "       report. Doubles A's memory footprint while enabled. Mutually\n"
        "       exclusive with --implicit-duplicates (see below).\n"
        "       --implicit-duplicates: an alternative to --explicit-\n"
        "       duplicates that costs no extra memory. Forces --mv-type to\n"
        "       fp32_16 (error if --mv-type was also given as something\n"
        "       else) and stores A ONCE, packed (NumericType::FP32_16 --\n"
        "       see the file header comment). Per-iteration matvecs read it\n"
        "       at the cheap FP16 (\"pure_half\", safe-decode variant --\n"
        "       fp32_16_half_only, \"TOMOS FP16 (implicit)\" in the GEMV\n"
        "       performance plots -- the fast-decode sibling was tried\n"
        "       here too, but on real ill-conditioned matrices its\n"
        "       saturate-the-underflow-band-to-zero behavior cost enough\n"
        "       accuracy to stall convergence, so this reverted back)\n"
        "       compute tier, but the restart-boundary residual recompute\n"
        "       (and this program's own final residual\n"
        "       report) instead reads that SAME buffer at its FP32\n"
        "       (\"accurate\"/hybrid) compute tier -- FP32_16's wr half\n"
        "       already holds enough information to reconstruct near-FP32\n"
        "       accuracy, so no second copy of A is needed. HARD mutually\n"
        "       exclusive with --explicit-duplicates (error if both are\n"
        "       given): the two now select different, incompatible storage\n"
        "       schemes for A -- FP32_16-packed vs. plain native fp16 --\n"
        "       not two independent options.\n"
        "       --residual-history writes one relative-residual value per\n"
        "       completed iteration (summed across all restarts) to PATH\n"
        "       as plain text, one per line (iteration index, 1-based,\n"
        "       then the value, space-separated).\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    bool have_matrix = false, have_x = false, have_b = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires a value\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--matrix") { a.matrix_path = next("--matrix"); have_matrix = true; }
        else if (arg == "--x-true") { a.x_true_path = next("--x-true"); have_x = true; }
        else if (arg == "--rhs") { a.rhs_path = next("--rhs"); have_b = true; }
        else if (arg == "--device") {
            std::string d = next("--device");
            if (d == "cpu") a.device = Device::CPU;
            else if (d == "cuda") a.device = Device::CUDA;
            else { std::fprintf(stderr, "error: --device must be cpu or cuda\n"); std::exit(1); }
        }
        else if (arg == "--mv-type") { a.mv_type = parse_mv_precision(next("--mv-type")); a.mv_type_set = true; }
        else if (arg == "--value-type") a.value_type = parse_precision(next("--value-type"));
        else if (arg == "--refine-type") a.refine_type = parse_precision(next("--refine-type"));
        else if (arg == "--precond-type") {
            a.precond_type = parse_precision(next("--precond-type"));
            a.precond_type_set = true;
        }
        else if (arg == "--tol") a.tol = std::stod(next("--tol"));
        else if (arg == "--max-iters") a.max_iters = std::stoi(next("--max-iters"));
        else if (arg == "--max-relres") a.max_relres = std::stod(next("--max-relres"));
        else if (arg == "--restarts") a.restarts = std::stoi(next("--restarts"));
        else if (arg == "--residual-history") a.residual_history_path = next("--residual-history");
        else if (arg == "--precond") a.precond = true;
        else if (arg == "--normalize-columns") a.normalize_columns = true;
        else if (arg == "--explicit-duplicates") a.explicit_duplicates = true;
        else if (arg == "--implicit-duplicates") a.implicit_duplicates = true;
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "error: unrecognized argument '%s'\n", arg.c_str()); usage(argv[0]); std::exit(1); }
    }
    if (!have_matrix || !have_x || !have_b) {
        std::fprintf(stderr, "error: --matrix, --x-true, and --rhs are all required\n");
        usage(argv[0]);
        std::exit(1);
    }
    // --precond-type defaults to --value-type when not given explicitly --
    // resolved here (not as a plain default member) so it correctly
    // tracks whatever --value-type ended up being, regardless of which
    // order the two flags were passed in.
    if (!a.precond_type_set) a.precond_type = a.value_type;
    // --implicit-duplicates and --explicit-duplicates are mutually
    // exclusive: they now select two different, incompatible STORAGE
    // schemes for A (FP32_16-packed vs. plain native fp16), not two
    // independent options that could ever both apply to the same buffer.
    if (a.implicit_duplicates && a.explicit_duplicates) {
        std::fprintf(stderr,
            "error: --implicit-duplicates and --explicit-duplicates are "
            "mutually exclusive (they select two different, incompatible "
            "storage schemes for A -- FP32_16-packed vs. plain native "
            "fp16 -- not two independent options).\n");
        std::exit(1);
    }
    // --implicit-duplicates: A is ALWAYS stored packed (NumericType::
    // FP32_16, see the file header comment) -- --mv-type stops meaning
    // "A's storage type" here (there'd be nothing left to choose: storage
    // is fixed) and instead selects which of FP32_16's two registered
    // compute tiers the per-iteration matvecs (step_1_orth_basis) use:
    // fp16 ("pure_half", the cheap tier -- safe-decode kernel variant,
    // fp32_16_half_only, "TOMOS FP16 (implicit)" in the GEMV performance
    // plots; the fast-decode sibling was tried here too but reverted --
    // see the usage banner above for why) or fp32 ("accurate"/hybrid).
    // Defaults to fp16 (the cheap tier -- the whole point of
    // --implicit-duplicates) when --mv-type isn't given. The
    // restart-boundary/final residual check always uses the fp32 tier
    // regardless of this choice -- see initialize()'s own comment.
    if (a.implicit_duplicates) {
        if (a.mv_type_set && a.mv_type != NumericType::FP32 &&
            a.mv_type != NumericType::FP16) {
            std::fprintf(stderr,
                "error: --implicit-duplicates only accepts --mv-type fp32 "
                "or fp16 here (it selects the per-iteration compute tier "
                "over A's fixed fp32_16 storage, not the storage type "
                "itself)\n");
            std::exit(1);
        }
        if (!a.mv_type_set) a.mv_type = NumericType::FP16;
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    // Force line-buffered stdout -- when stdout isn't an interactive
    // terminal (e.g. piped through srun --overlap, as scripts/
    // run_lsqr_precision.sh does), the C library defaults to FULLY
    // buffered instead, so printf output can sit invisible for a long
    // time even while real progress is happening. Without this, "one line
    // then nothing" is genuinely ambiguous between "still working" and
    // "hung" -- with it, every line below appears as soon as it's printed.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    Args args = parse_args(argc, argv);

    std::printf("==> Reading %s / %s / %s\n", args.matrix_path.c_str(),
               args.x_true_path.c_str(), args.rhs_path.c_str());
    std::vector<double> a_host, x_true_host, b_host;
    int m, n, x_m, x_n, b_m, b_n;
    read_mtx_host(args.matrix_path, a_host, m, n);
    read_mtx_host(args.x_true_path, x_true_host, x_m, x_n);
    read_mtx_host(args.rhs_path, b_host, b_m, b_n);
    if (x_m != n || x_n != 1)
        throw std::runtime_error("x_true size mismatch: expected " +
                                 std::to_string(n) + " x 1");
    if (b_m != m || b_n != 1)
        throw std::runtime_error("b size mismatch: expected " +
                                 std::to_string(m) + " x 1");

    // Column equilibration: A' = A*D^-1 (D = diag(column norms)), solved
    // for z (A'*z = b, same b); x = D^-1*z is recovered at the very end,
    // right before computing the solution-error metric. Computed here in
    // double precision from the just-read .mtx data, BEFORE A gets cast
    // down to --mv-type -- reduces A's own entry-magnitude spread ahead
    // of that cast, independent of (and composable with) --precond
    // (which builds its sketch from this same, already-scaled a_host).
    std::vector<double> col_norms;
    if (args.normalize_columns) {
        col_norms.assign(static_cast<size_t>(n), 0.0);
        for (int j = 0; j < n; ++j) {
            double ss = 0.0;
            for (int i = 0; i < m; ++i) {
                double v = a_host[static_cast<size_t>(i) + static_cast<size_t>(j) * m];
                ss += v * v;
            }
            col_norms[j] = std::sqrt(ss);
            if (col_norms[j] == 0.0) col_norms[j] = 1.0;  // all-zero column: leave as-is
        }
        for (int j = 0; j < n; ++j) {
            double inv = 1.0 / col_norms[j];
            for (int i = 0; i < m; ++i)
                a_host[static_cast<size_t>(i) + static_cast<size_t>(j) * m] *= inv;
        }
        std::printf("==> Column-normalized A (unit-norm columns; un-scaled after solving)\n");
    }

    // A's actual storage type: always FP32_16 under --implicit-duplicates
    // (args.mv_type there instead holds the per-iteration COMPUTE tier,
    // fp32 or fp16 -- see the post-parse block above), otherwise
    // args.mv_type itself, same as always.
    NumericType mv_storage_type =
        args.implicit_duplicates ? NumericType::FP32_16 : args.mv_type;

    std::printf("==> A: %d x %d, device=%s\n", m, n,
               args.device == Device::CPU ? "cpu" : "cuda");
    if (args.implicit_duplicates) {
        std::printf("==> mv_storage=fp32_16 mv_compute=%s value_type=%s refine_type=%s precond=%s precond_type=%s normalize_columns=%s explicit_duplicates=%s implicit_duplicates=yes\n",
                   precision_name(args.mv_type), precision_name(args.value_type),
                   precision_name(args.refine_type), args.precond ? "sketch+qr" : "none",
                   args.precond ? precision_name(args.precond_type) : "n/a",
                   args.normalize_columns ? "yes" : "no",
                   args.explicit_duplicates ? "yes" : "no");
    } else {
        std::printf("==> mv_type=%s value_type=%s refine_type=%s precond=%s precond_type=%s normalize_columns=%s explicit_duplicates=%s implicit_duplicates=no\n",
                   precision_name(args.mv_type), precision_name(args.value_type),
                   precision_name(args.refine_type), args.precond ? "sketch+qr" : "none",
                   args.precond ? precision_name(args.precond_type) : "n/a",
                   args.normalize_columns ? "yes" : "no",
                   args.explicit_duplicates ? "yes" : "no");
    }
    std::printf("==> max_iters=%d (per restart) restarts=%d (ceiling: %d iterations)\n",
               args.max_iters, args.restarts, args.max_iters * args.restarts);

    Context ctx{args.device};

    Dense A = Dense::create_strided(ctx, Layout::COL_MAJOR, mv_storage_type, {static_cast<size_t>(m), static_cast<size_t>(n)});
    Dense b = Dense::create_strided(ctx, Layout::COL_MAJOR, args.refine_type, {static_cast<size_t>(m), 1});
    Dense x = Dense::create_strided(ctx, Layout::COL_MAJOR, args.refine_type, {static_cast<size_t>(n), 1});

    if (mv_storage_type == NumericType::FP32_16) {
        fill_fp32_16_dense_from_host(ctx, args.device, A, a_host,
                                     dim<2>{static_cast<size_t>(m), static_cast<size_t>(n)});
    } else {
        dispatch_precision(mv_storage_type, [&](auto tag) {
            using T = decltype(tag);
            fill_dense_from_host<T>(args.device, A, a_host);
        });
    }
    dispatch_precision(args.refine_type, [&](auto tag) {
        using T = decltype(tag);
        fill_dense_from_host<T>(args.device, b, b_host);
    });
    // x is fully overwritten by lsqr_solve's own internal set_values(0, x)
    // before iterating -- no need to initialize it here.

    // Explicit refine_type-precision duplicate of A (--explicit-duplicates):
    // used by lsqr_solve for the restart-boundary residual recompute (see
    // include/core/solver.hpp's own comment) AND, below, for this
    // program's own final residual report -- both would otherwise be
    // computed through A's own mv_type precision, capping the achievable
    // accuracy at mv_type's own quantization noise regardless of how
    // precise everything else (x, the Krylov vectors) is.
    std::unique_ptr<Dense> A_refine;
    if (args.explicit_duplicates) {
        if (mv_storage_type == args.refine_type) {
            std::printf("==> --explicit-duplicates has no effect here (mv_type == refine_type already)\n");
        } else {
            std::printf("==> Building explicit %s-precision duplicate of A (%d x %d)...\n",
                       precision_name(args.refine_type), m, n);
            A_refine = std::make_unique<Dense>(Dense::create_strided(
                ctx, Layout::COL_MAJOR, args.refine_type,
                dim<2>{static_cast<size_t>(m), static_cast<size_t>(n)}));
            dispatch_precision(args.refine_type, [&](auto tag) {
                using T = decltype(tag);
                fill_dense_from_host<T>(args.device, *A_refine, a_host);
            });
        }
    }

    // Sketch + QR preconditioner (see the file header comment for why this
    // is always built on a throwaway CPU Context regardless of --device).
    Dense precond = Dense::create_strided(ctx, Layout::COL_MAJOR, args.precond_type,
                                          {static_cast<size_t>(n), static_cast<size_t>(n)});
    if (args.precond) {
        size_t k = 2 * static_cast<size_t>(n);
        // generate_sketch_qr_precond calls straight into LAPACKE_{s,d}geqrf/
        // {s,d}trtri -- there is no LAPACK half-precision QR, so the whole
        // sketch/QR/invert pipeline has to run at FP32 or FP64 regardless of
        // --precond-type; only the FINAL R^{-1} gets cast down to whatever
        // --precond-type actually is when it's copied onto `precond` below.
        NumericType build_type =
            (args.precond_type == NumericType::FP64) ? NumericType::FP64 : NumericType::FP32;
        std::printf("==> Building count-sketch + QR preconditioner (k=%zu = 2n rows, built at %s, applied at %s)\n",
                   k, precision_name(build_type), precision_name(args.precond_type));
        Context cpu_ctx{Device::CPU};
        std::vector<double> precond_host;
        dispatch_precision(build_type, [&](auto tag) {
            using T = decltype(tag);
            std::printf("    - staging a CPU copy of A (%d x %d)...\n", m, n);
            Dense A_cpu = Dense::create_strided(cpu_ctx, Layout::COL_MAJOR, build_type,
                                                {static_cast<size_t>(m), static_cast<size_t>(n)});
            fill_dense_from_host<T>(Device::CPU, A_cpu, a_host);

            std::printf("    - generating and applying the count sketch...\n");
            Sparse S = Sparse::create_csr(cpu_ctx, build_type, IntType::INT32,
                                          {k, static_cast<size_t>(m)}, k);
            generate_count_sketch(S);
            Dense sketched = Dense::create_strided(cpu_ctx, Layout::COL_MAJOR, build_type,
                                                   {k, static_cast<size_t>(n)});
            apply_count_sketch(S, A_cpu, sketched);

            std::printf("    - QR-factoring the %zu x %d sketch and inverting R...\n", k, n);
            generate_sketch_qr_precond(S, sketched);  // leading n x n block <- R^{-1}

            Dense precond_cpu = Dense::create_strided(cpu_ctx, Layout::COL_MAJOR, build_type,
                                                       {static_cast<size_t>(n), static_cast<size_t>(n)});
            auto vSketched = as_strided_view<T>(sketched);
            auto vPrecondCpu = as_strided_view<T>(precond_cpu);
            for (size_t j = 0; j < static_cast<size_t>(n); ++j)
                for (size_t i = 0; i < static_cast<size_t>(n); ++i)
                    vPrecondCpu.values_[i + j * vPrecondCpu.lead_dim_] =
                        vSketched.values_[i + j * vSketched.lead_dim_];

            read_dense_to_host<T>(Device::CPU, precond_cpu, precond_host);
        });
        std::printf("    - copying R^{-1} (%d x %d) onto %s at %s...\n", n, n,
                   args.device == Device::CPU ? "cpu" : "cuda", precision_name(args.precond_type));
        dispatch_precision(args.precond_type, [&](auto tag) {
            using T = decltype(tag);
            fill_dense_from_host<T>(args.device, precond, precond_host);
        });
        std::printf("    - preconditioner ready\n");
    }

    Workspace ws = args.precond
        ? create_lsqr_workspace(ctx, {static_cast<size_t>(m), static_cast<size_t>(n)}, args.value_type, args.precond_type)
        : create_lsqr_workspace(ctx, {static_cast<size_t>(m), static_cast<size_t>(n)}, args.value_type);
    if (args.precond) add_precond(precond, ws);
    ws.use_implicit_duplicate_ = args.implicit_duplicates;
    if (args.implicit_duplicates) {
        // args.mv_type holds the per-iteration compute tier here (fp32 or
        // fp16 -- see the post-parse block above), not a storage type.
        ws.mv_compute_type_     = args.mv_type;
        ws.mv_compute_type_set_ = true;
    }

    int iterations = 0;
    std::vector<double> residual_history;
    bool want_history = !args.residual_history_path.empty();
    auto t0 = std::chrono::steady_clock::now();
    lsqr_solve(ctx, A, b, x, ws, args.tol, args.max_iters, args.max_relres,
              &iterations, want_history ? &residual_history : nullptr,
              args.restarts, A_refine.get());
    if (args.device == Device::CUDA) cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (want_history) {
        FILE* hf = std::fopen(args.residual_history_path.c_str(), "w");
        if (!hf) {
            throw std::runtime_error("could not open '" + args.residual_history_path +
                                     "' for writing");
        }
        for (size_t i = 0; i < residual_history.size(); ++i)
            std::fprintf(hf, "%zu %.16e\n", i + 1, residual_history[i]);
        std::fclose(hf);
        std::printf("==> residual history (%zu entries) written to: %s\n",
                    residual_history.size(), args.residual_history_path.c_str());
    }

    // Residual: r = A*x (generic mv_apply -- always registered for any
    // (mv_type, refine_type, refine_type) triple, see dense_registries.cpp),
    // then ||r - b|| / ||b|| computed on the host. Uses A_refine (the
    // refine_type-precision duplicate) when available, for the same
    // reason lsqr_solve does internally -- otherwise this report would be
    // capped by A's own mv_type quantization noise, same as the
    // restart-boundary check would be without --explicit-duplicates. With
    // --implicit-duplicates instead (A itself stored as FP32_16, no
    // A_refine), mirror lsqr_solve's own initialize() and read A's SAME
    // buffer at its FP32 ("accurate") compute tier rather than the default
    // FP16 ("pure_half") tier every per-iteration matvec used.
    Dense r = Dense::create_strided(ctx, Layout::COL_MAJOR, args.refine_type, {static_cast<size_t>(m), 1});
    if (A_refine) {
        mv_apply(args.device, TransposeType::NO_TRANSPOSE, 1.0, *A_refine, x, 0.0, r);
    } else if (args.implicit_duplicates) {
        mv_apply(args.device, TransposeType::NO_TRANSPOSE, 1.0, A, x, 0.0, r, NumericType::FP32);
    } else {
        mv_apply(args.device, TransposeType::NO_TRANSPOSE, 1.0, A, x, 0.0, r);
    }

    std::vector<double> r_host, x_host;
    dispatch_precision(args.refine_type, [&](auto tag) {
        using T = decltype(tag);
        read_dense_to_host<T>(args.device, r, r_host);
        read_dense_to_host<T>(args.device, x, x_host);
    });

    double residual = rel_l2_diff(r_host, b_host);

    // x_host currently holds z (the column-equilibrated unknown, A'*z=b)
    // if --normalize-columns was used -- recover x = D^-1*z (D = diag
    // (col_norms)) before comparing against x_true, which is in the
    // ORIGINAL (unscaled) problem's space.
    if (args.normalize_columns) {
        for (int j = 0; j < n; ++j) x_host[j] /= col_norms[j];
    }
    double sol_error = rel_l2_diff(x_host, x_true_host);

    std::printf("==> iterations:        %d\n", iterations);
    std::printf("==> solve time:        %.3f ms\n", solve_ms);
    std::printf("==> final residual:    %.6e  (||A*x-b|| / ||b||)\n", residual);
    std::printf("==> solution error:    %.6e  (||x-x_true|| / ||x_true||)\n", sol_error);

    return 0;
}
