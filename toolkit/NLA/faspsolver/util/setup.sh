#!/bin/bash

if [ -d lib ]; then
    echo "lib/ dir check [ok]"
else
    echo "Making a new directory lib/ automatically ..."
    mkdir lib
    echo "[OK]"
fi

if [ -d out ]; then
    echo "out/ dir check [ok]"
else
    echo "Making a new directory out/ automatically ..."
    mkdir out
    echo "[OK]"
fi

