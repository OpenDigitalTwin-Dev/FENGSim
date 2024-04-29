#!/bin/sh

apt update
apt -y install cmake
apt -y install make
apt -y install gcc g++ gfortran

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/lapack_install
make -j4
make install
