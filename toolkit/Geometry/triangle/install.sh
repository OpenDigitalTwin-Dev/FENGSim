#!/bin/sh

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/triangle_install
make
make install
