set (treesearch_sources
  main.cpp)

blt_add_executable(
  NAME treesearch
  SOURCES ${treesearch_sources}
  DEPENDS_ON
    SAMRAI_hier
    SAMRAI_tbox)

target_compile_definitions(treesearch PUBLIC TESTING=1)

file (GLOB test_inputs ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/*.input)

samrai_add_tests(
  NAME treesearch
  EXECUTABLE treesearch
  INPUTS ${test_inputs}
  PARALLEL TRUE)
