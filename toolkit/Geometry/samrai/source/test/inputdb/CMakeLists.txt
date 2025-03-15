set (inputdb_sources
        inputdb.cpp)

blt_add_executable(
        NAME inputdb
        SOURCES ${inputdb_sources}
        DEPENDS_ON
          SAMRAI_tbox)

target_compile_definitions(
        inputdb
        PRIVATE
        NDIM=2 TESTING=1)

file (GLOB test_inputs ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/*.input)

samrai_add_tests(
  NAME inputdb
  EXECUTABLE inputdb
  INPUTS ${test_inputs}
  PARALLEL TRUE)
