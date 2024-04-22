#!/bin/sh

mkdir build
cd build
cmake .. -DVCG_BUILD_EXAMPLES=ON
make -j4
