#!/bin/sh

# basic
sudo apt update
sudo apt -y install cmake
sudo apt -y install libfreetype-dev
sudo apt -y install gfortran
sudo apt -y install liblapack-dev
sudo apt -y install libsuperlu-dev
sudo apt -y install libopenmpi-dev
sudo apt -y install p7zip-full

# qt install
rm -rf $PWD/../toolkit/qt/5.12.12
rm -rf $PWD/../toolkit/qt/icu
rm -rf $PWD/../toolkit/qt/qtcreator/bin
rm -rf $PWD/../toolkit/qt/qtcreator/lib
rm -rf $PWD/../toolkit/qt/qtcreator/libexec
rm -rf $PWD/../toolkit/qt/qtcreator/share
7za x ../toolkit/qt/5.12.12.7z -o/$PWD/../toolkit/qt
7za x ../toolkit/qt/icu.7z -o/$PWD/../toolkit/qt
7za x ../toolkit/qt/qtcreator/bin.7z -o/$PWD/../toolkit/qt/qtcreator
7za x ../toolkit/qt/qtcreator/lib.7z -o/$PWD/../toolkit/qt/qtcreator
7za x ../toolkit/qt/qtcreator/libexec.7z -o/$PWD/../toolkit/qt/qtcreator
7za x ../toolkit/qt/qtcreator/share.7z -o/$PWD/../toolkit/qt/qtcreator

sudo apt -y install libxcb-xinerama0-dev
sudo apt -y install libpcre2-dev
sudo apt -y install build-essential
sudo apt -y install libgl1-mesa-dev
sudo apt -y install libglu1-mesa-dev
sudo apt -y install freeglut3-dev
sudo apt -y install libxcb-xinput-dev
sudo apt -y install g++

# cgal
sudo apt -y install libgmp-dev
sudo apt -y install libmpfr-dev
sudo apt -y install libboost-all-dev

sed -i 11c"<value type=\"QString\" key=\"QMakePath\">"$PWD/../toolkit/qt/5.12.12/bin/qmake"</value>" $PWD/../toolkit/qt/qtcreator/share/qtcreator/QtProject/qtcreator/qtversion.xml

# vtk install
mkdir ../toolkit/vtk/build
cd ../toolkit/vtk/build
cmake .. -DQt5_DIR=$PWD/../../qt/5.12.12/lib/cmake/Qt5 -DCMAKE_INSTALL_PREFIX=$PWD/../../install/vtk_install
make -j4
make install

# oce install
cd ../../oce
mkdir build
cd build
cmake .. -DVTK_DIR=$PWD/../../install/vtk_install/lib/cmake/vtk-8.1 -DOCE_INSTALL_PREFIX=$PWD/../../install/oce_install
make -j4
make install

# gmsh install
cd ../../gmsh
mkdir build
cd build
cmake .. -DOCC_INC=$PWD/../../install/oce_install/include/oce -DCMAKE_INSTALL_PREFIX=$PWD/../../install/gmsh_install -DENABLE_BUILD_DYNAMIC=ON -DENABLE_BUILD_SHARED=ON -DENABLE_PETSC=OFF -DENABLE_FLTK=OFF -DENABLE_MMG3D=OFF -DENABLE_MED=OFF 
make -j4
make install

# cgal install
cd ../../cgal
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/cgal_install -DWITH_CGAL_Qt5=OFF
make -j4
make install

# pcl install
sudo apt -y install libeigen3-dev
sudo apt -y install libflann-dev
cd ../../pcl
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/pcl_install -DWITH_QT=OFF
make -j4
make install

# starter:ale
cd ../../../starter/ALE
mkdir build
cd build
cmake ..
make

# starter:cgal
cd ../../../starter/CGAL
mkdir data/output
mkdir build
cd build
cmake ..
make

# starter:gdt
cd ../../../starter/GDT
mkdir build
cd build
cmake ..
make

