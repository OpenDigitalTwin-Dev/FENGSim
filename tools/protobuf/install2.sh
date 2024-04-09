#!/bin/sh

apt update
apt -y install cmake
apt -y install make
apt -y install gcc
apt -y install g++
apt -y install automake
apt -y install libtool
./autogen.sh
rm -rf build
mkdir build
cd build
./../configure --prefix=$PWD/../../protobuf_install
make -j4
make install
