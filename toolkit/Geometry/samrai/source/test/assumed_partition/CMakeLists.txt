blt_add_executable(
  NAME assumed_partition
  SOURCES main.cpp
  DEPENDS_ON
    SAMRAI_geom
    SAMRAI_hier
    SAMRAI_tbox)

target_include_directories( assumed_partition
  PUBLIC ${PROJECT_SOURCE_DIR}/source/test/assumed_partition)

file (GLOB test_inputs ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/*.input)

samrai_add_tests(
  NAME assumed_partition
  EXECUTABLE assumed_partition
  INPUTS ${test_inputs}
  PARALLEL TRUE)
