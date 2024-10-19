blt_add_executable(
  NAME raja_api_tests
  SOURCES raja_api_tests.cpp
  DEPENDS_ON SAMRAI_hier SAMRAI_tbox RAJA cuda)
