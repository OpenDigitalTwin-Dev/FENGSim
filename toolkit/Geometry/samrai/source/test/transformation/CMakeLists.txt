blt_add_executable(
  NAME transformation
  SOURCES main.cpp
  DEPENDS_ON
    SAMRAI_pdat
    SAMRAI_hier
    SAMRAI_tbox)

target_include_directories( transformation
  PUBLIC ${PROJECT_SOURCE_DIR}/source/test/transformation)

if(ENABLE_MPI)
  set(TASKS 1)
else()
  set(TASKS 0)
endif()

blt_add_test(
  NAME transformation
  COMMAND transformation
  NUM_MPI_TASKS ${TASKS})
