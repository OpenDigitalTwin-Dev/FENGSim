#!/bin/sh

# >>> gcc g++ verison >=10 will have problems

sudo apt -y install gcc-9 g++-9 gcc-10 g++-10 cmake libmpich-dev build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

mkdir build
cd build
cmake .. -DQt5_DIR=$PWD/../../../Tools/qt/5.12.12/lib/cmake/Qt5 -DCMAKE_INSTALL_PREFIX=$PWD/../../install/vtk_install -DCMAKE_CXX_COMPILER=g++-9 -DCMAKE_C_COMPILER=gcc-9 -DVTK_Group_MPI=ON
make -j4
make install
