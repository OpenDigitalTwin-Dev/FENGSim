#!/bin/sh

#chang the following in Makefile
#gklib_path = ../NSM/extern/ALE/install/gk
#metis_path = ../NSM/extern/ALE/install/metis

make config prefix=$PWD/../install/parmetis_install
make -j4
make install
