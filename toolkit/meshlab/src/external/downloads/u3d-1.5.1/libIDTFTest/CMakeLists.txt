cmake_minimum_required(VERSION 3.13)
project(libIDTFTest)

### Build settings
set(CMAKE_CXX_STANDARD 11)

set(SOURCES
	main.cpp)

set(INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../Samples/idtf/bimba.idtf")
set(OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../Samples/out_test/bimba.u3d")

add_executable(libIDTFTest
	${SOURCES})

target_link_libraries(
	libIDTFTest
	PUBLIC IDTF)

target_compile_definitions(libIDTFTest PRIVATE
	INPUT_FILE=\"${INPUT_FILE}\"
	OUTPUT_FILE=\"${OUTPUT_FILE}\")
