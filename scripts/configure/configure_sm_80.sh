#!/bin/bash                                                                                                                                                         
CURRENT_DIR=$(pwd)
echo $CURRENT_DIR
cmake \
    -DCMAKE_INSTALL_PREFIX="$CURRENT_DIR/build"             \
    -DCUDA_INC="$CUDA_HOME/include"                         \
    -DCUDA_LIB="$CUDA_HOME/lib64"                           \
    -DCMAKE_CUDA_ARCHITECTURES="80"                         \
    -DRANDLS_INC="include/"                                 \
    -DRANDLS_LIB="$CURRENT_DIR/build"                       \
    -DBLAS_INC="$CURRENT_DIR/include/"                      \
    -DBLAS_LIB="$CURRENT_DIR/lib/"                          \
    -DBLAS="USE_OPENBLAS"                                   \
    -DDEVICE_BACKEND="CUDA"                                 \
    -DDEBUG="BUILD_WITHOUT_DEBUG_INFO"                      \
    -DVISUALS="BUILD_WITHOUT_VISUALS"                       \
    -DMEASURE_RUNTIME="BUILD_WITHOUT_RUNTIME_MEASUREMENT"   \
    -G "Unix Makefiles"                                     \
    -DBUILD_SHARED_LIBS=ON                                  \
    -DBUILD_WITHOUT_LAPACK=OFF                              \
    -DBUILD_LAPACKE=ON                                      \
    -DCMAKE_BUILD_TYPE=Release                              \
    -DCMAKE_Fortran_COMPILER=gfortran                       \
    -DNOFORTRAN=0                                           \
    -S . -B "$CURRENT_DIR/build" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
