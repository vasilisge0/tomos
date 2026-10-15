cd ../../build
rm CMakeCache.txt
cmake .. \
  -DDEVICE_BACKEND=CUDA \
  -DBLAS=USE_OPENBLAS \
  -DDEBUG=BUILD_WITHOUT_DEBUG_INFO \
  -DCMAKE_INSTALL_PREFIX=. \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --target test_dense -j4
cmake --build . --target test_sparse -j4
cmake --build . --target test_dense_cpu_kernels -j4
cmake --build . --target test_dense_cuda_kernels -j4
./test/core/matrix/test_dense
./test/core/matrix/test_sparse
./test/kernels/cpu/matrix/test_dense_cpu_kernels
./test/kernels/cuda/matrix/test_dense_cuda_kernels