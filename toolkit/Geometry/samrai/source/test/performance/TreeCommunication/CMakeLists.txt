set (treecommunication_sources
  tcb.cpp)

blt_add_executable(
  NAME treecommunication
  SOURCES ${treecommunication_sources}
  DEPENDS_ON
    SAMRAI_tbox)

target_compile_definitions(treecommunication PUBLIC TESTING=1)

file (GLOB test_inputs ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/*.input)

samrai_add_tests(
  NAME treecommunication
  EXECUTABLE treecommunication
  INPUTS ${test_inputs}
  PARALLEL TRUE)

