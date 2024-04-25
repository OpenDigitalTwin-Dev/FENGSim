#!/bin/sh

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/cgal_install -DWITH_CGAL_Qt5=OFF
make -j4
make install
