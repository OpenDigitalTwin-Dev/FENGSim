#!/bin/sh

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../netgen_install -DUSE_OCC=OFF -DUSE_PYTHON=OFF
make -j4
make install
