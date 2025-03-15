#!/bin/sh

sudo apt update
sudo apt -y install p7zip-full
sudo apt -y install cmake make gcc g++

7za x eigenlib.7z -o/$PWD/

mkdir build
cd build
cmake .. -DVCG_BUILD_EXAMPLES=ON
make -j4
