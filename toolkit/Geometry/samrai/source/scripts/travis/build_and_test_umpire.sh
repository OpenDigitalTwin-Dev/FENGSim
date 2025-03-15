#!/bin/bash
set -x
env

threads=4

if [[ "$DO_BUILD" == "yes" ]] ; then
    base_path=$(pwd)

    # Umpire
    umpire_build=$base_path/umpire-build
    umpire_install=$base_path/umpire-install

    if [ ! -d $umpire_build ]; then
        git clone --recursive https://github.com/LLNL/Umpire || exit $?
        (cd Umpire && git checkout v0.3.5) || exit $?
        mkdir -p $umpire_install
        mkdir -p $umpire_build && cd $_
        cmake -DCMAKE_INSTALL_PREFIX=$umpire_install -DCMAKE_CXX_FLAGS=$CMAKE_EXTRA_FLAGS -DENABLE_CUDA=OFF -DENABLE_OPENMP=OFF -DENABLE_MPI=OFF -DENABLE_EXAMPLES=OFF -DENABLE_BENCHMARKS=OFF -DENABLE_TESTS=OFF -DCMAKE_CXX_COMPILER="$COMPILER" ../Umpire || exit $?
        make -j $threads install || exit $?
        cd $base_path
    fi

    # RAJA
    raja_build=$base_path/raja-build
    raja_install=$base_path/raja-install

    if [ ! -d $raja_build ]; then
        git clone --recursive https://github.com/LLNL/RAJA || exit $?
        (cd RAJA && git checkout v0.8.0) || exit $?
        mkdir -p $raja_install
        mkdir -p $raja_build && cd $_
        cmake -DCMAKE_INSTALL_PREFIX=$raja_install -DCMAKE_CXX_FLAGS=$CMAKE_EXTRA_FLAGS -DENABLE_CUDA=OFF -DENABLE_OPENMP=OFF -DENABLE_MPI=OFF -DENABLE_EXAMPLES=OFF -DENABLE_BENCHMARKS=OFF -DENABLE_TESTS=OFF -DCMAKE_CXX_COMPILER="$COMPILER" ../RAJA || exit $?
        make -j $threads install || exit $?
        cd $base_path
    fi

    # SAMRAI
    # Travis already cloned this to /home/llnl/SAMRAI = $base_path/SAMRAI -- see SAMRAI/.travis.yml for more details
    ls -lct $base_path
    samrai_source=$base_path
    cd $samrai_source && git submodule init && git submodule update || exit $?
    samrai_build=$base_path/SAMRAI-build
    mkdir -p $samrai_build && cd $_ || exit $?
    cmake -DENABLE_MPI=Off -DENABLE_CUDA=OFF -DENABLE_HDF5=Off -DENABLE_RAJA=ON -DRAJA_DIR=$raja_install/share/raja/cmake -DENABLE_UMPIRE=ON -Dumpire_DIR=$umpire_install/share/umpire/cmake -DCMAKE_CXX_COMPILER="$COMPILER" -DCMAKE_CXX_FLAGS=$CMAKE_EXTRA_FLAGS -DENABLE_EXAMPLES=ON -DENABLE_TESTS=ON -DCMAKE_Fortran_COMPILER=gfortran $samrai_source || exit $?
    make -j $threads || exit $?
    [[ "$DO_TEST" == "yes" ]] && ctest --extra-verbose -j $threads || exit $?
fi

exit 0
