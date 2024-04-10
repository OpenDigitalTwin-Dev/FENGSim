#!/bin/sh

sudo apt install libstb-dev

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../cura_engine_install \
	  -D Arcus_DIR=$PWD/../../libarcus_install/lib/cmake/Arcus/ \
      -D Protobuf_INCLUDE_DIR=$PWD/../../protobuf_install/include/ \
      -D Protobuf_LIBRARY_DEBUG=$PWD/../../protobuf_install/lib/libprotobuf.so \
      -D Protobuf_LIBRARY_RELEASE=$PWD/../../protobuf_install/lib/libprotobuf.so \
      -D Protobuf_LITE_LIBRARY_DEBUG=$PWD/../../protobuf_install/lib/libprotobuf-lite.so \
      -D Protobuf_LITE_LIBRARY_RELEASE=$PWD/../../protobuf_install/lib/libprotobuf-lite.so \
      -D Protobuf_PROTOC_EXECUTABLE=$PWD/../../protobuf_install/bin/protoc \
      -D Protobuf_PROTOC_LIBRARY_DEBUG=$PWD/../../protobuf_install/lib/libprotoc.so \
      -D Protobuf_PROTOC_LIBRARY_RELEAS=$PWD/../../protobuf_install/lib/libprotoc.so
make -j4
make install
