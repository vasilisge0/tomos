#include "dense_registries.hpp"

#include "fp32_16_registries.hpp"
#include "matrix_cpu_kernels.hpp"
#include "mm_cpu_kernels.hpp"
#include "set_values_cpu_kernels.hpp"

#if USE_CUDA
#include "context.hpp"
#include "mm_cuda_kernels.cuh"
#include "mv_cuda_kernels.cuh"
#include "set_values_cuda_kernels.cuh"
#endif

namespace tomos {
namespace dispatch {

template <Device dev, typename a_t, typename x_t, typename y_t>
void mv_strided(TransposeType trans, double alpha, tomos::Dense& A,
                tomos::Dense& x, double beta, tomos::Dense& y,
                float* /*workspace*/) {
    auto vA = as_strided_view<a_t>(A);
    auto vx = as_strided_view<x_t>(x);
    auto vy = as_strided_view<y_t>(y);
    if constexpr (dev == Device::CPU) {
        cpu::mv_strided<a_t, x_t, y_t>(trans, alpha, vA.size_, vA.values_,
                                        vA.lead_dim_, vx.values_, vx.lead_dim_,
                                        beta, vy.values_, vy.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::experimental::mv_strided<a_t, x_t, y_t>(trans, alpha, vA.size_, vA.values_,
                                        vA.lead_dim_, vx.values_, vx.lead_dim_,
                                        beta, vy.values_, vy.lead_dim_,
                                        cuda_cublas_handle());
    }
#endif
}

#if USE_CUDA
// Native FP32/FP32/FP32, compute_type == FP32 on CUDA: the tuned "TOMOS
// FP32" kernel (mv_fp32_tiled, see mv_cuda_kernels.cu/.cuh) instead of the
// generic mv_strided<float,float,float>/cublasSgemv path every other native
// triple still uses -- this is the kernel the GEVM plots actually measure
// as "TOMOS FP32". CUDA-only: no CPU equivalent exists, so (Device::CPU,
// FP32, FP32, FP32, FP32) stays on the generic path (OpenBLAS cblas_sgemv),
// same "no second implementation to register" contract lookup_mm already
// uses for its own CPU-only gaps.
void mv_fp32_tiled_impl(TransposeType trans, double alpha, tomos::Dense& A,
                        tomos::Dense& x, double beta, tomos::Dense& y,
                        float* /*workspace*/) {
    auto vA = as_strided_view<float>(A);
    auto vx = as_strided_view<float>(x);
    auto vy = as_strided_view<float>(y);
    cuda::mv_fp32_tiled(trans, alpha, vA.size_, vA.values_, vA.lead_dim_,
                        vx.values_, vx.lead_dim_, beta, vy.values_,
                        vy.lead_dim_);
}

// Native FP16/FP32/FP32, compute_type == FP16 on CUDA: mv_fp16_native_
// mixed (mv_cuda_kernels.cu/.cuh) instead of the generic mv_strided<
// __half,float,float>/gemv_generic_kernel path this triple would
// otherwise land on -- that generic path has no specialization for
// (half, float, float) (only the 3 fully-matched triples double/float/
// half-all-same do), so it silently upcasts A and x to double and
// accumulates there, which is NOT real fp16 compute despite A being
// stored as fp16. This is the kernel lsqr_precision_runner's own
// --explicit-duplicates (without --implicit-duplicates, which uses a
// completely different FP32_16-packed storage/dispatch path) now uses
// for its per-iteration matvec. CUDA-only, same "no second implementation
// to register" contract as mv_fp32_tiled_impl above.
//
// Was mv_fp16_native (the original, single-block-per-row, scalar-__half
// kernel) until mv_fp16_native_mixed -- a true mechanical duplicate of
// "TOMOS FP16" (mv_fp16_native_storage)'s entire col-split/half2-packed/
// "_best" dispatch tree, with x/y kept at float to match THIS slot's own
// (FP16,FP32,FP32,FP16) type contract exactly (mv_fp16_native_storage
// itself has all-half x/y, and swapping it in directly here would
// silently corrupt every x/y read/write, not just change precision) --
// measured faster on real A100 hardware once "TOMOS FP16" itself
// received the same col_splits/half2-packing work. mv_fp16_native is
// still defined and exported, just no longer registered here.
void mv_fp16_native_impl(TransposeType trans, double alpha, tomos::Dense& A,
                         tomos::Dense& x, double beta, tomos::Dense& y,
                         float* /*workspace*/) {
    auto vA = as_strided_view<__half>(A);
    auto vx = as_strided_view<float>(x);
    auto vy = as_strided_view<float>(y);
    cuda::mv_fp16_native_mixed(trans, alpha, vA.size_, vA.values_, vA.lead_dim_,
                               vx.values_, vx.lead_dim_, beta, vy.values_,
                               vy.lead_dim_);
}
#endif

template <Device dev, typename a_t, typename b_t, typename c_t>
void mm_strided(TransposeType transA, TransposeType transB, double alpha,
                tomos::Dense& A, tomos::Dense& B, double beta, tomos::Dense& C) {
    auto vA = as_strided_view<a_t>(A);
    auto vB = as_strided_view<b_t>(B);
    auto vC = as_strided_view<c_t>(C);
    if constexpr (dev == Device::CPU) {
        cpu::mm_strided<a_t, b_t, c_t>(transA, transB, alpha, vA.size_,
                                       vA.values_, vA.lead_dim_, vB.size_,
                                       vB.values_, vB.lead_dim_, beta,
                                       vC.values_, vC.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::experimental::mm_strided<a_t, b_t, c_t>(
            transA, transB, alpha, vA.size_, vA.values_, vA.lead_dim_,
            vB.size_, vB.values_, vB.lead_dim_, beta, vC.values_,
            vC.lead_dim_, cuda_cublas_handle());
    }
#endif
}

template <Device dev, typename value_t>
void set_values_strided(double val, tomos::Dense& target) {
    auto view = as_strided_view<value_t>(target);
    if constexpr (dev == Device::CPU) {
        cpu::set_values_impl<value_t>(static_cast<value_t>(val), view.size_,
                                      view.values_, view.lead_dim_);
    }
#if USE_CUDA
    else if constexpr (dev == Device::CUDA) {
        cuda::set_values_impl<value_t>(static_cast<value_t>(val), view.size_,
                                       view.values_, view.lead_dim_);
    }
#endif
}

const Dense& Dense::get() {
    static const Dense instance;
    return instance;
}

initialize_dense_fn Dense::lookup_initialize(MatrixFormat format, Device dev,
                                             NumericType numeric_type) const {
    auto fn = initialize_[encode(format, dev, numeric_type)];
    if (!fn) throw std::runtime_error("No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

free_dense_fn Dense::lookup_free(MatrixFormat format, Device dev,
                                 NumericType numeric_type) const {
    auto fn = free_[encode(format, dev, numeric_type)];
    if (!fn) throw std::runtime_error("No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

read_dense_fn Dense::lookup_read(MatrixFormat format, Device dev,
                                 NumericType numeric_type) const {
    auto fn = read_[encode(format, dev, numeric_type)];
    if (!fn) throw std::runtime_error("No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

copy_dense_fn Dense::lookup_copy(MatrixFormat source_format,
                                 MatrixFormat target_format, Device source_dev,
                                 Device target_dev, NumericType source_type,
                                 NumericType target_type) const {
    auto fn = copy_[encode_copy(source_format, target_format, source_dev,
                                target_dev, source_type, target_type)];
    if (!fn) throw std::runtime_error("No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

set_values_dense_fn Dense::lookup_set_values(Device dev,
                                             NumericType numeric_type) const {
    auto fn = set_values_[encode_set_values(dev, numeric_type)];
    if (!fn) throw std::runtime_error("No kernel registered for the requested device/type. "
            "Rebuild with -DDEVICE_BACKEND=USE_CUDA to enable CUDA support.");
    return fn;
}

mv_dense_fn Dense::lookup_mv(Device dev, NumericType type_A, NumericType type_x,
                              NumericType type_y, NumericType compute_type) const {
    auto fn = mv_[encode_mv(dev, type_A, type_x, type_y, compute_type)];
    if (!fn) throw std::runtime_error("No kernel registered for the requested device/type/compute "
            "combination. Either no such kernel exists (e.g. a non-native "
            "compute_type for a native FP storage type, which has only one "
            "implementation), or rebuild with -DDEVICE_BACKEND=USE_CUDA to "
            "enable CUDA support.");
    return fn;
}

mm_dense_fn Dense::lookup_mm(Device dev, NumericType type_A, NumericType type_B,
                             NumericType type_C, bool use_workspace) const {
    auto fn = mm_[encode_mm(dev, type_A, type_B, type_C, use_workspace)];
    if (!fn) throw std::runtime_error("No kernel registered for the requested device/type/workspace "
            "combination. Either no such kernel exists (e.g. a workspace "
            "variant for a native FP type, which has only one "
            "implementation), or rebuild with -DDEVICE_BACKEND=USE_CUDA to "
            "enable CUDA support.");
    return fn;
}

constexpr std::size_t Dense::encode_set_values(Device dev,
                                               NumericType numeric_type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types +
           static_cast<std::size_t>(numeric_type);
}

constexpr std::size_t Dense::encode_mv(Device dev, NumericType type_A,
                                       NumericType type_x, NumericType type_y,
                                       NumericType compute_type) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_numeric_types *
               num_registry_numeric_types +
           static_cast<std::size_t>(type_A) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_numeric_types +
           static_cast<std::size_t>(type_x) * num_registry_numeric_types *
               num_registry_numeric_types +
           static_cast<std::size_t>(type_y) * num_registry_numeric_types +
           static_cast<std::size_t>(compute_type);
}

constexpr std::size_t Dense::encode_mm(Device dev, NumericType type_A,
                                       NumericType type_B, NumericType type_C,
                                       bool use_workspace) {
    return static_cast<std::size_t>(dev) * num_registry_numeric_types *
               num_registry_numeric_types * num_registry_numeric_types * 2 +
           static_cast<std::size_t>(type_A) * num_registry_numeric_types *
               num_registry_numeric_types * 2 +
           static_cast<std::size_t>(type_B) * num_registry_numeric_types * 2 +
           static_cast<std::size_t>(type_C) * 2 +
           static_cast<std::size_t>(use_workspace);
}

constexpr std::size_t Dense::encode(MatrixFormat format, Device dev,
                                    NumericType numeric_type) {
    return static_cast<std::size_t>(format) * num_registry_devices *
               num_registry_numeric_types +
           static_cast<std::size_t>(dev) * num_registry_numeric_types +
           static_cast<std::size_t>(numeric_type);
}

constexpr std::size_t Dense::encode_copy(MatrixFormat source_format,
                                         MatrixFormat target_format,
                                         Device source_dev, Device target_dev,
                                         NumericType source_type,
                                         NumericType target_type) {
    return static_cast<std::size_t>(source_format) * num_registry_devices *
               num_registry_devices * num_registry_numeric_types *
               num_registry_numeric_types +
           static_cast<std::size_t>(target_format) * num_registry_devices *
               num_registry_numeric_types * num_registry_numeric_types +
           static_cast<std::size_t>(source_dev) * num_registry_numeric_types *
               num_registry_numeric_types +
           static_cast<std::size_t>(target_dev) * num_registry_numeric_types *
               num_registry_numeric_types +
           static_cast<std::size_t>(source_type) * num_registry_numeric_types +
           static_cast<std::size_t>(target_type);
}

Dense::Dense() {
    initialize_.fill(nullptr);
    free_.fill(nullptr);
    read_.fill(nullptr);
    copy_.fill(nullptr);
    set_values_.fill(nullptr);
    mv_.fill(nullptr);
    mm_.fill(nullptr);

    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto type_tag) {
            using T = typename decltype(type_tag)::type;
            initialize_[encode(MatrixFormat::STRIDED, dev,
                               numeric_type_of<T>)] =
                &initialize_strided_storage<dev, T>;
            free_[encode(MatrixFormat::STRIDED, dev, numeric_type_of<T>)] =
                &free_strided_storage<dev, T>;
            set_values_[encode_set_values(dev, numeric_type_of<T>)] =
                &set_values_strided<dev, T>;
        });

        // NumericType::FP32_16 stores two packed uint16_t halves (wh/wr, see
        // fp32_16_pack.hpp) per element in one uint32_t slot -- allocation,
        // freeing, and zero-fill are plain raw-storage operations, identical
        // in shape to every other NumericType, so the same generic templates
        // apply directly. GEMM/GEMV for this type go through the dedicated
        // dispatch::Fp32_16 registry instead of the generic mv_ table below,
        // since a Dense value_type of FP32_16 does not correspond to a
        // single arithmetic C++ type the mv_strided<a_t,x_t,y_t> template can
        // be instantiated with.
        initialize_[encode(MatrixFormat::STRIDED, dev,
                           numeric_type_of<uint32_t>)] =
            &initialize_strided_storage<dev, uint32_t>;
        free_[encode(MatrixFormat::STRIDED, dev, numeric_type_of<uint32_t>)] =
            &free_strided_storage<dev, uint32_t>;
        set_values_[encode_set_values(dev, numeric_type_of<uint32_t>)] =
            &set_values_strided<dev, uint32_t>;
    });

    // Native FP triples: compute_type has no independent meaning here (the
    // generic mv_strided path always computes at whatever mv_compute_t<a_t,
    // x_t,y_t> resolves to internally, not a caller-selectable precision),
    // so only compute_type == type_A (A's own storage format -- "no
    // override") is registered, same "one real implementation, one
    // registered slot" contract lookup_mm's use_workspace axis already uses.
    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto a_tag) {
            using AT = typename decltype(a_tag)::type;
            for_each_type(FPTypes{}, [&](auto x_tag) {
                using XT = typename decltype(x_tag)::type;
                for_each_type(FPTypes{}, [&](auto y_tag) {
                    using YT = typename decltype(y_tag)::type;
                    mv_[encode_mv(dev, numeric_type_of<AT>, numeric_type_of<XT>,
                                  numeric_type_of<YT>, numeric_type_of<AT>)] =
                        &mv_strided<dev, AT, XT, YT>;
                });
            });
        });
    });

#if USE_CUDA
    // Override: native FP32/FP32/FP32 on CUDA gets the tuned mv_fp32_tiled
    // kernel instead of the generic mv_strided/cublasSgemv registration the
    // loop above just filled in -- see mv_fp32_tiled_impl's own comment.
    mv_[encode_mv(Device::CUDA, NumericType::FP32, NumericType::FP32,
                  NumericType::FP32, NumericType::FP32)] = &mv_fp32_tiled_impl;
    // Override: native FP16/FP32/FP32 (compute_type FP16) on CUDA gets the
    // tuned mv_fp16_native kernel instead of the generic mv_strided<__half,
    // float,float>/gemv_generic_kernel double-accumulate fallback the loop
    // above just filled in -- see mv_fp16_native_impl's own comment.
    mv_[encode_mv(Device::CUDA, NumericType::FP16, NumericType::FP32,
                  NumericType::FP32, NumericType::FP16)] = &mv_fp16_native_impl;
#endif

    // NumericType::FP32_16 A against FP32 x/y: delegate to the dedicated
    // dispatch::Fp32_16 registry (same "goes through Fp32_16 instead of a
    // generic table" convention as lookup_mm's own FP32_16 delegation
    // below), not the native-triple loop above -- FP32_16 isn't a single
    // arithmetic C++ type mv_strided<a_t,x_t,y_t> can be instantiated with.
    // compute_type == FP16 routes to the "pure_half" (__half-only compute)
    // tier; compute_type == FP32 routes to "accurate" (wh+wr reconstructed,
    // float compute). Both are registered on CPU and CUDA -- unlike
    // lookup_mm's workspace tier, neither Fp32_16 mv variant is CUDA-only.
    // The "fast" tier (Fp32_16::lookup_mv_fast) has no compute_type slot of
    // its own here -- it stays reachable only via mv_apply_fp32_16_fast
    // directly, since this axis only distinguishes two compute precisions,
    // not the extra narrower-tuned-but-still-FP16 "fast" family alongside
    // "pure_half".
    mv_[encode_mv(Device::CPU, NumericType::FP32_16, NumericType::FP32,
                  NumericType::FP32, NumericType::FP16)] =
        Fp32_16::get().lookup_mv_pure_half(Device::CPU);
    mv_[encode_mv(Device::CPU, NumericType::FP32_16, NumericType::FP32,
                  NumericType::FP32, NumericType::FP32)] =
        Fp32_16::get().lookup_mv_accurate(Device::CPU);
#if USE_CUDA
    mv_[encode_mv(Device::CUDA, NumericType::FP32_16, NumericType::FP32,
                  NumericType::FP32, NumericType::FP16)] =
        Fp32_16::get().lookup_mv_pure_half(Device::CUDA);
    mv_[encode_mv(Device::CUDA, NumericType::FP32_16, NumericType::FP32,
                  NumericType::FP32, NumericType::FP32)] =
        Fp32_16::get().lookup_mv_accurate(Device::CUDA);
#endif

    // Native FP triples: only one real implementation each (the plain
    // mm_strided kernel), so only use_workspace=false is ever registered
    // here -- requesting use_workspace=true for one of these throws,
    // same "nothing registered" contract every other lookup already uses.
    for_each_value(ActiveDevices{}, [&](auto dev_tag) {
        constexpr Device dev = decltype(dev_tag)::value;
        for_each_type(FPTypes{}, [&](auto a_tag) {
            using AT = typename decltype(a_tag)::type;
            for_each_type(FPTypes{}, [&](auto b_tag) {
                using BT = typename decltype(b_tag)::type;
                for_each_type(FPTypes{}, [&](auto c_tag) {
                    using CT = typename decltype(c_tag)::type;
                    mm_[encode_mm(dev, numeric_type_of<AT>, numeric_type_of<BT>,
                                  numeric_type_of<CT>, /*use_workspace=*/false)] =
                        &mm_strided<dev, AT, BT, CT>;
                });
            });
        });
    });

    // NumericType::FP32_16 A against FP32 B/C: delegate to the dedicated
    // dispatch::Fp32_16 registry (same "goes through Fp32_16 instead of a
    // generic table" convention as initialize_/free_/set_values_ above),
    // not the native-triple loop just above -- FP32_16 isn't a single
    // arithmetic C++ type mm_strided<a_t,b_t,c_t> can be instantiated with.
    // "accurate" (always-correct, no workspace) is registered on both
    // devices; "workspace" (decode-into-scratch-then-native-GEMM) only on
    // CUDA -- Fp32_16::lookup_mm_workspace(Device::CPU) has nothing to
    // return (no CPU implementation exists), so that combination is
    // deliberately left unregistered here rather than eagerly calling a
    // lookup that would throw during this constructor.
    mm_[encode_mm(Device::CPU, NumericType::FP32_16, NumericType::FP32,
                  NumericType::FP32, /*use_workspace=*/false)] =
        Fp32_16::get().lookup_mm_accurate(Device::CPU);
#if USE_CUDA
    mm_[encode_mm(Device::CUDA, NumericType::FP32_16, NumericType::FP32,
                  NumericType::FP32, /*use_workspace=*/false)] =
        Fp32_16::get().lookup_mm_accurate(Device::CUDA);
    mm_[encode_mm(Device::CUDA, NumericType::FP32_16, NumericType::FP32,
                  NumericType::FP32, /*use_workspace=*/true)] =
        Fp32_16::get().lookup_mm_workspace(Device::CUDA);
#endif

    struct ReadEntry {
        MatrixFormat format;
        Device dev;
        NumericType numeric_type;
        read_dense_fn fn;
    };
    static const ReadEntry read_entries[] = {
        {MatrixFormat::STRIDED, Device::CPU, NumericType::FP64,
         &cpu::read_strided_storage<double>},
        {MatrixFormat::STRIDED, Device::CPU, NumericType::FP32,
         &cpu::read_strided_storage<float>},
    };
    for (auto& e : read_entries)
        read_[encode(e.format, e.dev, e.numeric_type)] = e.fn;

    for_each_type(FPTypes{}, [&](auto src_tag) {
        using SrcT = typename decltype(src_tag)::type;
        for_each_type(FPTypes{}, [&](auto tgt_tag) {
            using TgtT = typename decltype(tgt_tag)::type;
            copy_[encode_copy(MatrixFormat::STRIDED, MatrixFormat::STRIDED,
                              Device::CPU, Device::CPU, numeric_type_of<SrcT>,
                              numeric_type_of<TgtT>)] =
                &copy_strided_storage<Device::CPU, Device::CPU, SrcT, TgtT>;
        });
    });
}  // Dense constructor

}  // namespace dispatch
}  // namespace tomos