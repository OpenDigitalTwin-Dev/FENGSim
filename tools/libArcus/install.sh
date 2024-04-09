#!/bin/sh

apt update
apt -y install python3.10-dev
apt -y install python3-sip-dev

rm -rf build
mkdir build
cd build
cmake .. \
      -D Protobuf_INCLUDE_DIR=$PWD/../../protobuf_install/include/ \
      -D Protobuf_LIBRARY_DEBUG=$PWD/../../protobuf_install/lib/ \
      -D Protobuf_LIBRARY_RELEASE=$PWD/../../protobuf_install/lib/ \
      -D Protobuf_LITE_LIBRARY_DEBUG=$PWD/../../protobuf_install/lib/ \
      -D Protobuf_LITE_LIBRARY_RELEASE=$PWD/../../protobuf_install/lib/ \
      -D Protobuf_PROTOC_EXECUTABLE=$PWD/../../protobuf_install/bin \
      -D Protobuf_PROTOC_LIBRARY_DEBUG=$PWD/../../protobuf_install/lib/ \
      -D Protobuf_PROTOC_LIBRARY_RELEAS=$PWD/../../protobuf_install/lib/
make -j4
