#!/bin/sh


# We need to install protobuf and libarcus firstly.

sudo apt -y install libstb-dev

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/../../install/cura_engine_install \
	  -D Arcus_DIR=$PWD/../../install/libarcus_install/lib/cmake/Arcus/ \
      -D Protobuf_INCLUDE_DIR=$PWD/../../install/protobuf_install/include/ \
      -D Protobuf_LIBRARY_DEBUG=$PWD/../../install/protobuf_install/lib/libprotobuf.so \
      -D Protobuf_LIBRARY_RELEASE=$PWD/../../install/protobuf_install/lib/libprotobuf.so \
      -D Protobuf_LITE_LIBRARY_DEBUG=$PWD/../../install/protobuf_install/lib/libprotobuf-lite.so \
      -D Protobuf_LITE_LIBRARY_RELEASE=$PWD/../../install/protobuf_install/lib/libprotobuf-lite.so \
      -D Protobuf_PROTOC_EXECUTABLE=$PWD/../../install/protobuf_install/bin/protoc \
      -D Protobuf_PROTOC_LIBRARY_DEBUG=$PWD/../../install/protobuf_install/lib/libprotoc.so \
      -D Protobuf_PROTOC_LIBRARY_RELEAS=$PWD/../../install/protobuf_install/lib/libprotoc.so
make -j4
make install
