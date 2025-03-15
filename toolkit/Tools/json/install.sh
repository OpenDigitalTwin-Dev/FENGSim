#!/bin/sh

sudo apt update
sudo apt -y install cmake make gcc g++
sudo apt -y install p7zip-full

rm -rf json
7za x json.7z -o/$PWD/json

cd json

mkdir build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../../install/json_install
make -j4
make install
