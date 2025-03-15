#!/bin/sh

mkdir build
cd build
cmake .. -DQt5_DIR=$PWD/../../../qt/5.12.12/lib/cmake/Qt5 -DCMAKE_INSTALL_PREFIX=$PWD/../../install/vtk_install
make -j4
make install
