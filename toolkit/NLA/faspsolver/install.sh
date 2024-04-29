#!/bin/sh

apt update
apt -y install cmake
apt -y install make
apt -y install gcc g++

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/faspsolver_install
make
make install
