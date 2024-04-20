#!/bin/sh

LD_LIBRARY_PATH=$PWD/../toolkit/qt/icu/usr/local/lib
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../toolkit/qt/icu/usr/local/lib
#_ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../toolkit/qt/5.12.12/plugins
export LD_LIBRARY_PATH

./../toolkit/qt/qtcreator/bin/qtcreator ./../starter/FENGSim/FENGSim.pro
