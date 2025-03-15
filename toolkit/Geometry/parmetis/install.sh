#!/bin/sh

#chang the following in Makefile
#gklib_path = ../../NSM/extern/ALE/install/gk
#metis_path = ../../NSM/extern/ALE/install/metis

sudo apt update
sudo apt -y install make gcc g++ cmake libmpich-dev

make config prefix=$PWD/../install/parmetis_install
make -j4
make install
