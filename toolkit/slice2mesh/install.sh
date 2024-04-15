#!/bin/sh

# We need to install triangle and tetgen firstly.

# The slice2mesh needs cinolib, but cinolib doesn't need to compile.

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../slice2mesh_install
make -j4
make install
