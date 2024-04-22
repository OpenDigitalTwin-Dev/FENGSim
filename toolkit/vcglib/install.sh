#!/bin/sh

7za x eigenlib.7z -o/$PWD/

mkdir build
cd build
cmake .. -DVCG_BUILD_EXAMPLES=ON
make -j4
