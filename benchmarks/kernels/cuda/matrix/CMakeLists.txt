find_package(CUDAToolkit REQUIRED)

add_executable(bench_mv_cuda_kernels bench_mv_cuda_kernels.cu)
set_property(TARGET bench_mv_cuda_kernels PROPERTY CUDA_STANDARD 17)
target_include_directories(bench_mv_cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
)
target_link_libraries(bench_mv_cuda_kernels
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
    benchmark::benchmark
)

add_executable(bench_mm_cuda_kernels bench_mm_cuda_kernels.cu)
set_property(TARGET bench_mm_cuda_kernels PROPERTY CUDA_STANDARD 17)
target_include_directories(bench_mm_cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
)
target_link_libraries(bench_mm_cuda_kernels
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cusparse
    CUDA::cudart
    benchmark::benchmark
)

add_executable(bench_fp32_16_cuda_kernels bench_fp32_16_cuda_kernels.cu)
set_property(TARGET bench_fp32_16_cuda_kernels PROPERTY CUDA_STANDARD 17)
target_include_directories(bench_fp32_16_cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(bench_fp32_16_cuda_kernels
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
    benchmark::benchmark
)

# Standalone correctness + performance verification for the mv_fp32_16_*
# GEMV kernels (wide/split-K, tiled, and register-blocked narrow_vec
# dispatch paths) against cuBLAS -- see verify_mv_fp32_16_gemv.cu. Not a
# google-benchmark executable: it generates its own (possibly large) test
# data directly on the device to avoid host memory limits, so it can be
# pointed at a GPU with a lot of headroom (e.g. an A100) via
# scripts/verify_mv_fp32_16_gemv.sh.
add_executable(verify_mv_fp32_16_gemv verify_mv_fp32_16_gemv.cu)
set_property(TARGET verify_mv_fp32_16_gemv PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mv_fp32_16_gemv PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mv_fp32_16_gemv
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Correctness verification for EXACTLY the NO_TRANSPOSE GEMV kernels
# scripts/plot_gemv_fp32_16_pure_half_vs_cublas_ref.m actually plots (5
# kernels, 5 shapes) -- much smaller/faster than verify_mv_fp32_16_gemv's
# own much larger sweep (which also covers every exploratory/negative-
# result kernel this session built along the way), for when the question
# is just "are the promoted kernels still correct" -- same idea as
# verify_mv_fp32_transpose_promoted just above, mirrored for this plot --
# see verify_gemv_fp32_16_plot_promoted.cu's own header comment.
add_executable(verify_gemv_fp32_16_plot_promoted verify_gemv_fp32_16_plot_promoted.cu)
set_property(TARGET verify_gemv_fp32_16_plot_promoted PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_gemv_fp32_16_plot_promoted PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_gemv_fp32_16_plot_promoted
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Correctness verification for EXACTLY the TRANSPOSE GEMV kernels
# scripts/plot_gemv_transpose_fp32_recon_vs_cublas_ref.m actually plots
# (5 kernels, 5 shapes) -- much smaller/faster than verify_mv_fp32_16_gemv's
# own ~50-kernel sweep, for when the question is just "are the promoted
# kernels still correct" -- see verify_mv_fp32_transpose_promoted.cu's
# own header comment.
add_executable(verify_mv_fp32_transpose_promoted verify_mv_fp32_transpose_promoted.cu)
set_property(TARGET verify_mv_fp32_transpose_promoted PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mv_fp32_transpose_promoted PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mv_fp32_transpose_promoted
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Correctness verification for EXACTLY the NO_TRANSPOSE GEMV kernels
# scripts/plot_gemv_fp32_recon_vs_cutlass_ref.m actually plots (5
# kernels, 5 shapes) -- much smaller/faster than verify_mv_fp32_16_gemv's
# own much larger sweep, for when the question is just "are the promoted
# kernels still correct" -- see verify_gemv_fp32_recon_plot_promoted.cu's
# own header comment.
add_executable(verify_gemv_fp32_recon_plot_promoted verify_gemv_fp32_recon_plot_promoted.cu)
set_property(TARGET verify_gemv_fp32_recon_plot_promoted PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_gemv_fp32_recon_plot_promoted PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_gemv_fp32_recon_plot_promoted
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Correctness verification for EXACTLY the TRANSPOSE GEVM 16-bit-compute
# kernels scripts/plot_gemv_transpose_fp32_16_pure_half_vs_cublas_ref.m
# actually plots (4 kernels, 5 shapes) -- see verify_gemv_fp32_16_pure_
# half_transpose_plot_promoted.cu's own header comment.
add_executable(verify_gemv_fp32_16_pure_half_transpose_plot_promoted
    verify_gemv_fp32_16_pure_half_transpose_plot_promoted.cu)
set_property(TARGET verify_gemv_fp32_16_pure_half_transpose_plot_promoted PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_gemv_fp32_16_pure_half_transpose_plot_promoted PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_gemv_fp32_16_pure_half_transpose_plot_promoted
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Correctness verification for EXACTLY the NO_TRANSPOSE GEMM full-fp32-
# reconstruction-tier kernels scripts/plot_gemm_format_32_16_compute_
# 32.m actually plots (6 kernels, 5 shapes) -- see verify_mm_fp32_recon_
# plot_promoted.cu's own header comment.
add_executable(verify_mm_fp32_recon_plot_promoted verify_mm_fp32_recon_plot_promoted.cu)
set_property(TARGET verify_mm_fp32_recon_plot_promoted PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mm_fp32_recon_plot_promoted PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mm_fp32_recon_plot_promoted
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Correctness verification for EXACTLY the NO_TRANSPOSE GEMM pure/
# composite (tensor-core) tier kernels scripts/plot_fp16_bf16_pure_vs_
# cutlass_ref.m actually plots (5 kernels, 5 shapes) -- see verify_mm_
# fp16_bf16_pure_plot_promoted.cu's own header comment.
add_executable(verify_mm_fp16_bf16_pure_plot_promoted verify_mm_fp16_bf16_pure_plot_promoted.cu)
set_property(TARGET verify_mm_fp16_bf16_pure_plot_promoted PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mm_fp16_bf16_pure_plot_promoted PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mm_fp16_bf16_pure_plot_promoted
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Square-only fp32 CUDA GEMM benchmark, scoped down from bench_mm_cuda_kernels
# so a build+run cycle while tuning a new fp32 GEMM kernel stays fast -- see
# bench_mm_fp32_square_cuda_kernels.cu.
add_executable(bench_mm_fp32_square_cuda_kernels bench_mm_fp32_square_cuda_kernels.cu)
set_property(TARGET bench_mm_fp32_square_cuda_kernels PROPERTY CUDA_STANDARD 17)
target_include_directories(bench_mm_fp32_square_cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
)
target_link_libraries(bench_mm_fp32_square_cuda_kernels
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
    benchmark::benchmark
)

# Standalone correctness + performance verification for the custom fp32
# square-GEMM CUDA kernel(s) against cuBLAS -- see verify_mm_fp32_gemm.cu.
# Not a google-benchmark executable, for the same reason as
# verify_mv_fp32_16_gemv above: it generates its own (possibly large)
# performance-sweep data directly on the device via
# scripts/verify_mm_fp32_gemm.sh.
add_executable(verify_mm_fp32_gemm verify_mm_fp32_gemm.cu)
set_property(TARGET verify_mm_fp32_gemm PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mm_fp32_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mm_fp32_gemm
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Standalone correctness + performance verification for the custom,
# un-encumbered fp16 square-GEMM CUDA kernel(s) (plain __half A/B/C, no
# FP32_16 packed storage) against cuBLAS's Hgemm -- see verify_mm_fp16_gemm.
# cu. Not a google-benchmark executable, for the same reason as
# verify_mm_fp32_gemm above.
add_executable(verify_mm_fp16_gemm verify_mm_fp16_gemm.cu)
set_property(TARGET verify_mm_fp16_gemm PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mm_fp16_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mm_fp16_gemm
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Square-only fp32_16 CUDA GEMM benchmark, scoped down from
# bench_fp32_16_cuda_kernels the same way bench_mm_fp32_square_cuda_kernels
# is scoped down from bench_mm_cuda_kernels -- see
# bench_mm_fp32_16_square_cuda_kernels.cu.
add_executable(bench_mm_fp32_16_square_cuda_kernels bench_mm_fp32_16_square_cuda_kernels.cu)
set_property(TARGET bench_mm_fp32_16_square_cuda_kernels PROPERTY CUDA_STANDARD 17)
target_include_directories(bench_mm_fp32_16_square_cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/benchmarks
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(bench_mm_fp32_16_square_cuda_kernels
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
    benchmark::benchmark
)

# Standalone correctness + performance verification for the custom fp32_16
# square-GEMM CUDA kernel(s) against cuBLAS -- see verify_mm_fp32_16_gemm.cu.
# Not a google-benchmark executable, for the same reason as
# verify_mm_fp32_gemm above.
add_executable(verify_mm_fp32_16_gemm verify_mm_fp32_16_gemm.cu)
set_property(TARGET verify_mm_fp32_16_gemm PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mm_fp32_16_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mm_fp32_16_gemm
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)

# Standalone correctness + performance verification for the custom
# fp32_16 "pure_half" (fp16-layout, fp16-compute) square-GEMM CUDA
# kernel(s) against cuBLAS's Hgemm -- see verify_mm_fp32_16_pure_half_gemm.
# cu. Not a google-benchmark executable, for the same reason as
# verify_mm_fp32_16_gemm above.
add_executable(verify_mm_fp32_16_pure_half_gemm verify_mm_fp32_16_pure_half_gemm.cu)
set_property(TARGET verify_mm_fp32_16_pure_half_gemm PROPERTY CUDA_STANDARD 17)
target_include_directories(verify_mm_fp32_16_pure_half_gemm PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/cuda/matrix
    ${CMAKE_SOURCE_DIR}/src/kernels/cpu/matrix
    ${CMAKE_SOURCE_DIR}/src/core/matrix
)
target_link_libraries(verify_mm_fp32_16_pure_half_gemm
    tomos_core
    tomos_cpu
    tomos_cuda
    CUDA::cublas
    CUDA::cudart
)
