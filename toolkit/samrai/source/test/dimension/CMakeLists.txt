set ( dimension_sources
  main.cpp)

blt_add_executable(
  NAME dimension
  SOURCES ${dimension_sources}
  DEPENDS_ON
    SAMRAI_tbox)

target_compile_definitions(dimension PUBLIC TESTING=1)

if(ENABLE_MPI)
  set(TASKS 1)
else()
  set(TASKS 0)
endif()

blt_add_test(
  NAME dimension
  COMMAND dimension
  NUM_MPI_TASKS ${TASKS})
