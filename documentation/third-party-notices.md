# Third-Party Notices

This document enumerates the third-party software referenced or
incorporated in the development of TOMOS, for the purpose of
license-compliance review. For a general overview of the project, see
[`README.md`](../README.md). The full, verbatim license text for every
entry below that is genuinely redistributed (compiled into a binary TOMOS
ships) is reproduced in [`licenses/`](../licenses/) at the repository
root.

---

## Matrix Market I/O library (`mmio.c`/`mmio.h`)

**License:** Public domain
**Author:** Roldan Pozo, NIST (`pozo@nist.gov`), Fri Aug 15 16:29:47 EDT 1997
**Source:** <http://math.nist.gov/MatrixMarket>

**Description:** Vendored directly into this repository as `src/io/mmio.c`
and `src/io/mmio.h`, and compiled into `tomos_core`. This library provides
the `.mtx` file reading and writing functionality used throughout the
project's benchmark and LSQR tooling. Unlike every other entry in this
document, it consists of vendored source code residing within the
repository, rather than a build-time fetch or a link-only dependency. Both
files contain the original NIST copyright, permission, and warranty notice
within their header comments.

---

## NVIDIA CUDA Toolkit (cuBLAS, cuSPARSE, cuSOLVER, cuBLASLt, cuRAND, CUDA Runtime)

**License:** Proprietary; the NVIDIA CUDA Toolkit End User License
Agreement (EULA). This is not an open-source license, and is neither
BSD-3-Clause nor any OSI-approved license.
**Source:** Not applicable; the toolkit is closed-source and distributed
only as the CUDA Toolkit (<https://developer.nvidia.com/cuda-toolkit>).

**Description:** TOMOS's CUDA kernels (`tomos_cuda`) and solver code call
directly into `cublas`, `cusparse`, `cusolver`, `cublasLt`, `curand`,
`cudart`, and `cudadevrt`. No component of the CUDA Toolkit is vendored or
redistributed; TOMOS links against it at build time only, referencing
whichever CUDA Toolkit installation is present on the end user's own system
(`CUDA_HOME`). Anyone building or running TOMOS must separately install the
CUDA Toolkit and is thereby bound by NVIDIA's own End User License
Agreement.

---

## Thrust

**License:** Apache License 2.0 (a limited number of utility classes are
provided under the Boost Software License or a BSD license; see Thrust's
own `LICENSE` file).
**Copyright:** No single copyright line is present; copyright notices are
given within the license terms themselves rather than as a standalone
header.
**Source:** <https://github.com/NVIDIA/thrust>, header-only, distributed as
part of the CUDA Toolkit.
**Full license text:** [`licenses/thrust.txt`](../licenses/thrust.txt)

**Description:** Thrust is header-only and bundled with the CUDA Toolkit
(located under `CUDA_HOME/include`; it is neither fetched nor vendored
separately). It is included directly in
`src/kernels/cuda/sketch/countsketch_cuda_kernels.cu` and
`src/kernels/cuda/sketch/gaussian_sketch_cuda_kernels.cu`, and is compiled
into `tomos_cuda`. Unlike cuBLAS, cuSPARSE, and the other components listed
above, which are dynamically linked and never incorporated into TOMOS's own
binary, Thrust's template code is compiled directly into `tomos_cuda`'s
object code. For this reason it is listed as a separate entry rather than
being subsumed under the proprietary CUDA Toolkit license: it is a
distinct, Apache-licensed open-source component not covered by that End
User License Agreement. Thrust does not ship a `NOTICE` file, so no
additional material beyond the standard copyright and license text need be
propagated.

---

## OpenBLAS

**License:** BSD-3-Clause
**Copyright:** Copyright (c) 2011-2014, The OpenBLAS Project
**Source:** <https://github.com/OpenMathLib/OpenBLAS>, pinned at `v0.3.29`
**Full license text:** [`licenses/openblas.txt`](../licenses/openblas.txt)

**Description:** OpenBLAS is fetched automatically at configure time
(`dependencies/CMakeLists.txt`, via CMake's `FetchContent`), built from
source, and linked into the shipped `tomos_core` and `tomos_cpu` library
binaries. It provides the CBLAS and LAPACKE routines used by
`src/core/precond/sketch_qr.cpp` and
`src/kernels/cpu/dependencies/lapack.cpp`. This constitutes a genuine
redistribution, as OpenBLAS's compiled code becomes part of the binaries
that TOMOS ships.

---

## CUTLASS

**License:** BSD-3-Clause
**Copyright:** Copyright (c) 2017 - 2026, NVIDIA CORPORATION & AFFILIATES
**Source:** <https://github.com/NVIDIA/cutlass>

**Description:** CUTLASS is used in two distinct capacities. First, as an
external benchmark comparison: a small number of standalone files under
`benchmarks/kernels/cuda/matrix/` include CUTLASS headers directly and
instantiate `cutlass::gemm::device::Gemm` to obtain a same-process
throughput and correctness comparison. CUTLASS is not vendored in this
repository for this purpose; it is fetched separately by whoever executes
these scripts (via the `CUTLASS_DIR` environment variable), and the
resulting comparison executables are built independently of the
`tomos_core`, `tomos_cpu`, and `tomos_cuda` library targets. Second, as a
reference during kernel tuning: over the course of a single GEMM
kernel-tuning investigation confined to
`src/kernels/cuda/matrix/mm_cuda_kernels.cu`, specific technical facts
drawn from CUTLASS's source — including a tile shape, a lane-to-coordinate
mapping formula, a PTX instruction choice, and a pipeline-stage convention
— were used to inform the design of TOMOS's own, independently written
kernels. Each instance enumerated below is already documented in the
kernel file's own comments at the cited location; the table that follows
serves as an index into that existing documentation rather than as new
attribution text.

| # | Design decision | CUTLASS source cited | Applied as | Location |
|---|---|---|---|---|
| 1 | Lane-to-(row,col) mapping | `cutlass/layout/matrix.h`, `RowMajorInterleaved<2>::inverse()` | 2-line arithmetic formula replacing a plain divide/modulo split | `mm_cuda_kernels.cu:5378-5443` |
| 2 | ldmatrix PTX variant + offset toggle | `cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator`, `arch/memory_sm75.h` — via a standalone scratchpad that directly `#include`s real CUTLASS as ground truth | swapped `.trans`→non-transposed instruction, XOR offset fix | `mm_cuda_kernels.cu:15950-16229` |
| 3 | Instruction scheduling/interleaving | `cuobjdump` SASS diff vs CUTLASS's compiled mainloop; also noted CUTLASS has no `__launch_bounds__` | reordered load/compute source order; dropped a register ceiling | `mm_cuda_kernels.cu:1315-1346, 4242-4279, 6025-6041, 6669-6684, 25001-25059, 28281-28331` |
| 4 | Synchronous-load pipeline structure | `mma_pipelined.h`'s `gemm_iters` | rewrote pipeline to match (no cp.async) | `mm_cuda_kernels.cu:1563-1586` |
| 5 | Multistage priming/stage-count convention | `MmaMultistage`, confirmed `Stages=3` via `cuobjdump` on the built reference binary | wraparound-counter stage tracking; Stages=3 variants | `mm_cuda_kernels.cu:2025-2040, 15646-15663, 20780-20818, 24542-24567, 30591-30695` |
| 6 | Precomputed-stride pointer addressing | `PredicatedTileIterator`; corroborated via register-count comparison | replaced per-tile IMAD address recompute, many kernels | `mm_cuda_kernels.cu:2242-2263, 2486-2514, 2742-2746, 5595-5606, 19434-19485, 46109-46120, 46765-46778` |
| 7 | Accumulator/warp-tile sizing | SASS diff: CUTLASS's 128-float/thread, square 64×64 tile | BN changed 160→128 for an exact match | `mm_cuda_kernels.cu:18105-18186` |
| 8 | A-load thread→address mapping (fixed an L2 bug) | `PredicatedTileAccessIterator` + `cp_async` — wired up via real CUTLASS headers in a standalone repro | per-thread K/M mapping rewritten; fixed a confirmed 1.5x L2-sector excess | `mm_cuda_kernels.cu:18414-18609` |
| 9 | B-operand load mechanism | "zero STS, both operands via cp.async" | converted B load to direct cp.async; ~2x win | `mm_cuda_kernels.cu:18610-18648` |
| 10 | A-layout swizzle formula | `tensor_op_multiplicand_sm75.h`'s `TensorOpMultiplicand::operator()` | corrected an earlier self-invented, wrong swizzle | `mm_cuda_kernels.cu:23095-23167` |
| 11 | Epilogue vectorization width | matching config's `LinearCombination<..., 128/sizeof_bits<ElementC>>` | 8-half/float4 epilogue transfer | `mm_cuda_kernels.cu:19885-20377` |
| 12 | Serpentine mma.sync accumulate order | `mma_tensor_op.h`'s `MmaTensorOp::operator()` | reordered 32 mma.sync calls; `.reuse` 75%→97% | `mm_cuda_kernels.cu:27785-27858` |
| 13 | Full reference tile/warp/pipeline shape | `cutlass_fp16_gemm_ref.cu`'s actual config (128×256×32 tile, 64×64×32 warp, 3 stages) | built from scratch together as one kernel | `mm_cuda_kernels.cu:16838-16945` |
| 14 | B-swizzle (bigwarp family) | `tensor_op_multiplicand_sm75.h`, `RowMajorTensorOpMultiplicandCongruous<16,64>` | XOR partition swizzle for B | `mm_cuda_kernels.cu:47478-47528` |
| 15 | Warp-grid arrangement | CUTLASS's own `examples/14_ampere_tf32_tensorop_gemm` | 2×2 warp grid, low-occupancy variant | `mm_cuda_kernels.cu:44714-44748` |
| 16 | Register-level A/B double-buffering + spread cp.async issuance | `mma_multistage.h` — comment explicitly says "read verbatim via curl" | ported into the "bigwarp" kernel family (one sub-piece reverted after `compute-sanitizer` found a data race) | `mm_cuda_kernels.cu:44977-45404, 45530-45826, 46109-46778, 47100-47898` |
| 17 | Epilogue shared-memory staging | `epilogue/threadblock/epilogue.h` | stage C through shared memory before coalesced global write | `mm_cuda_kernels.cu:15111-15136` |

None of the instances enumerated above incorporate CUTLASS's source text
directly into TOMOS. In each case, a specific technical fact — a formula, a
configuration parameter, or an instruction choice — was identified in
CUTLASS's source and subsequently re-implemented in TOMOS's own,
independently structured kernel code, using TOMOS's own naming
conventions. Whether this practice constitutes the creation of a
derivative work under copyright law is a legal determination that falls
outside the scope of this document.

The same source file contains a small number of additional references to
"CUTLASS" that invoke it as general design inspiration without citing a
specific header, class, or formula (for example, a comment describing a
"cutlass-style" warp-tiling approach, or a reference to a "real CUTLASS
technique" in connection with an experiment that was never incorporated
into the shipped codebase). These references are omitted from the table
above because they do not cite a specific source; documentation of them is
available upon request.

---

## Google Test (gtest)

**License:** BSD-3-Clause
**Copyright:** Copyright 2008, Google Inc.
**Source:** <https://github.com/google/googletest>, pinned at commit
`03597a01ee50ed33e9dfd640b249b4be3799d395` (`test/CMakeLists.txt`)
**Full license text:** [`licenses/googletest.txt`](../licenses/googletest.txt)

**Description:** Google Test is fetched automatically at configure time via
CMake's `FetchContent`, built from source, and linked into every test
executable under `test/` (`GTest::gtest_main`). This is the same
redistribution situation as OpenBLAS: gtest's compiled code becomes part of
the shipped test binaries.

---

## Google Benchmark

**License:** Apache License 2.0
**Copyright:** No single copyright line is present; the project's
`AUTHORS` file attributes the work to multiple individual and
organizational contributors, originally including Google Inc.
**Source:** <https://github.com/google/benchmark>, pinned at `v1.9.1`
(`benchmarks/CMakeLists.txt`)
**Full license text:** [`licenses/google-benchmark.txt`](../licenses/google-benchmark.txt)

**Description:** Google Benchmark is fetched automatically at configure
time via CMake's `FetchContent`, built from source, and linked into every
`bench_*` executable under `benchmarks/` (`benchmark::benchmark`). Apache
License 2.0 differs from every other license family represented in this
document in that it includes an explicit patent grant and a
NOTICE-file-propagation clause (§4(d)); however, the google/benchmark
project does not ship a `NOTICE` file, so no additional material beyond
the standard copyright and license text need be propagated.
