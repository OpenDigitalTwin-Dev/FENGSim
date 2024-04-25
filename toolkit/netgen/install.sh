#!/bin/sh

sudo apt update
sudo apt -y install cmake
sudo apt -y install make
sudo apt -y install gcc g++
sudo apt -y install git
sudo apt -y install zlib1g-dev
#sudo apt -y install tcl-dev
#sudo apt -y install tk-dev

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/netgen_install -DUSE_OCC=OFF -DUSE_PYTHON=OFF -DUSE_GUI=OFF
make -j4
make install
