#!/bin/sh

sudo apt install libstb-dev

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../cura_engine_install 
make
make install
