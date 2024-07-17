#!/bin/sh

sudo apt update
sudo apt -y install cmake make gcc g++

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/json_install
make -j4
make install
