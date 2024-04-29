#!/bin/bash

if [ -d lib ]; then
    echo "Clean up executables in ./lib/ directory ..."
    cd lib
    rm -f *.a *.lib *.so
    cd ..
    echo "[OK]"
fi

if [ -d test ]; then
    echo "Clean up executables in ./test/ directory ..."
    cd test
    rm -f *.ex
    cd ..
    echo "[OK]"
fi

if [ -d tutorial ]; then
    echo "Clean up executables in ./tutorial/ directory ..."
    cd tutorial
    rm -f *.ex
    cd ..
    echo "[OK]"
fi

