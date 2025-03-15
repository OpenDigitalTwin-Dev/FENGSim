set (oca_sources
  main.cpp)

set (oca_depends_on
    SAMRAI_hier
    SAMRAI_geom
    SAMRAI_tbox)

blt_add_executable(
  NAME overlap-connector-algorithm
  SOURCES ${oca_sources}
  DEPENDS_ON
    ${oca_depends_on})

target_compile_definitions(overlap-connector-algorithm PUBLIC TESTING=1)

file (GLOB test_inputs ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/*.input)

samrai_add_tests(
  NAME overlap-connector-algorithm
  EXECUTABLE overlap-connector-algorithm
  INPUTS ${test_inputs}
  PARALLEL TRUE)
