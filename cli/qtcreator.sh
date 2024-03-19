#!/bin/sh

LD_LIBRARY_PATH=$PWD/../tools/qt/icu/usr/local/lib
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../tools/qt/icu/usr/local/lib
_ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../tools/qt/5.12.12/plugins
export LD_LIBRARY_PATH

./../tools/qt/qtcreator/bin/qtcreator
