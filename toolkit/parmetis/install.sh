#!/bin/sh

make config prefix=$PWD/../install/parmetis_install
make -j4
make install
