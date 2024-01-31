#!/bin/sh

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../slice2mesh_install
make
make install
