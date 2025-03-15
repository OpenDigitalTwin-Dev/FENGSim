set ( dlbg_sources
  main.cpp
  DLBGTest.cpp)

set(dlbg_depends testlib)
set(dlbg_depends ${dlbg_depends} ${SAMRAI_LIBRARIES})

# TODO CMake should resolve this dependency for us...
if (ENABLE_OPENMP)
  set(dlbg_depends ${dlbg_depends} openmp)
endif ()

if (ENABLE_CUDA)
  set(dlbg_depends ${dlbg_depends} cuda)
endif ()


blt_add_executable(
  NAME dlbg
  SOURCES ${dlbg_sources}
  DEPENDS_ON ${dlbg_depends})

target_compile_definitions(dlbg PUBLIC TESTING=1)

file (GLOB test_inputs ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/*.input)

samrai_add_tests(
  NAME dlbg
  EXECUTABLE dlbg
  INPUTS ${test_inputs}
  PARALLEL TRUE)
