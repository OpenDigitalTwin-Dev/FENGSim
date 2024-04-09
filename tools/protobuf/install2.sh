#!/bin/sh

sudo apt update
sudo apt -y install cmake
sudo apt -y install make
sudo apt -y install gcc
sudo apt -y install g++
sudo apt -y install automake
sudo apt -y install libtool
./autogen.sh
rm -rf build
mkdir build
cd build
./../configure --prefix=$PWD/../../protobuf_install
make -j4
make install
