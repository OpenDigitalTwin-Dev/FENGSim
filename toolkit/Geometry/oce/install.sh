#!/bin/sh

mkdir build
cd build
cmake .. -DVTK_DIR=$PWD/../../install/vtk_install/lib/cmake/vtk-8.1 -DOCE_INSTALL_PREFIX=$PWD/../../install/oce_install
make -j4
make install
