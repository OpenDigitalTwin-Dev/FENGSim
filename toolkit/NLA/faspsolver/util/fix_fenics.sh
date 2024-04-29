#!/bin/bash
#set -x

echo " Do not run this if you are not Chensong Zhang "
#exit 127

GMPDIR="gmp-6.0"
FENICS_PREFIX="/Applications/FEniCS.app/Contents/Resources"
BACKUP_DIR="../FEniCS_gmplibs.orig"

if [ ! -d $GMPDIR ]; then
    hg clone https://gmplib.org/repo/$GMPDIR/ $GMPDIR
    cd $GMPDIR
else
    cd $GMPDIR
    hg pull ; hg update
fi

if [ ! $? ] ; then
    echo "ERROR: something is wrong in the install of libgmp*"
    exit 255;
fi


# generate configure script
if [ ! -f configure ] ; then
    sh .bootstrap
fi 

# make first 
./configure --prefix=$FENICS_PREFIX/  --enable-cxx=detect CC="/usr/bin/gcc" CXX="/usr/bin/g++" && make && make check 
### back up
if [  ! -d "$FENICS_PREFIX" ] ; then
    echo "ERROR: $FENICS_PREFIX does not exist. Exiting"
    exit 254;
else
    set -x 
    mkdir $BACKUP_DIR
    /bin/cp -nv $FENICS_PREFIX/lib/libgmp*  $BACKUP_DIR
    sudo make install
    set +x
fi
#install fortran using dragonegg macports which should be compatible
#with clang as descriped on th llvm web. 3.4 is the latest version as
#of today.

sudo port install dragonegg-3.4 
# sudo port select --set gcc dragonegg-3.4-gcc-4.6
#set +x
