#!/bin/sh

apt update
apt -y install cmake
apt -y install make
apt -y install gcc g++
apt -y install git
apt -y install zlib1g-dev
#apt -y install tcl-dev
#apt -y install tk-dev

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../netgen_install -DUSE_OCC=OFF -DUSE_PYTHON=OFF -DUSE_GUI=OFF
make -j4
make install
