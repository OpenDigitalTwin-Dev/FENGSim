#!/bin/sh

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/pcl_install -DWITH_QT=OFF
make -j4
make install
