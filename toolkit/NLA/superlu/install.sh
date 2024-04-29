#!/bin/sh

apt update
apt -y install cmake
apt -y install make
apt -y install gcc g++ gfortran
apt -y install libpthread-stubs0-dev

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/superlu_install -DTPL_BLAS_LIBRARIES=$PWD/../../install/openblas_install/lib/libopenblas.a
make -j4
make install
