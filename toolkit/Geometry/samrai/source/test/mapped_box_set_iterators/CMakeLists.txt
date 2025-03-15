set (mapped_box_sources
        main-iterators.cpp)

blt_add_executable(
        NAME mapped_box_set_iterators
        SOURCES ${mapped_box_sources}
        DEPENDS_ON
          SAMRAI_hier
          SAMRAI_tbox)

if(ENABLE_MPI)
  set(TASKS 1)
else()
  set(TASKS 0)
endif()

blt_add_test(
  NAME mapped_box_set_iterators
  COMMAND mapped_box_set_iterators
  NUM_MPI_TASKS ${TASKS})
