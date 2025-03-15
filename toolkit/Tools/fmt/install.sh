#!/bin/sh

apt -y update
apt -y install cmake make gcc g++

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/fmt_install
make -j4
make install
