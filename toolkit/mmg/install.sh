#!/bin/sh

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/mmg_install
make -j4
make install
