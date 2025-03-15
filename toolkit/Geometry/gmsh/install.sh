#!/bin/sh

mkdir build
cd build
cmake .. -DOCC_INC=$PWD/../../install/oce_install/include/oce -DCMAKE_INSTALL_PREFIX=$PWD/../../install/gmsh_install -DENABLE_BUILD_DYNAMIC=ON -DENABLE_BUILD_SHARED=ON -DENABLE_PETSC=OFF -DENABLE_FLTK=OFF -DENABLE_MMG3D=OFF -DENABLE_MED=OFF 
make -j4
make install
