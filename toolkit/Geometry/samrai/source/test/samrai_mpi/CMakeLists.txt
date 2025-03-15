set (mpi_on_sources
  main-runtime-on.cpp
  mpi-interface-tests.cpp)

set (mpi_off_sources
  main-runtime-off.cpp
  mpi-interface-tests.cpp)

set (mpi_disabled_sources
  main-runtime-disabled.cpp
  mpi-interface-tests.cpp)

if(ENABLE_MPI)
  set(TASKS 1)
else()
  set(TASKS 0)
endif()

blt_add_executable(
  NAME mpi_on
  SOURCES ${mpi_on_sources}
  DEPENDS_ON
    SAMRAI_tbox)

blt_add_executable(
  NAME mpi_off
  SOURCES ${mpi_off_sources}
  DEPENDS_ON
    SAMRAI_tbox)

blt_add_executable(
  NAME mpi_disabled
  SOURCES ${mpi_disabled_sources}
  DEPENDS_ON
    SAMRAI_tbox)

target_compile_definitions(mpi_on PUBLIC TESTING=1)
target_compile_definitions(mpi_off PUBLIC TESTING=1)
target_compile_definitions(mpi_disabled PUBLIC TESTING=1)

if(ENABLE_MPI)
  set(TASKS 1)
else()
  set(TASKS 0)
endif()

blt_add_test(
  NAME mpi_on
  COMMAND mpi_on
  NUM_MPI_TASKS ${TASKS})

blt_add_test(
  NAME mpi_off
  COMMAND mpi_off
  NUM_MPI_TASKS ${TASKS})

blt_add_test(
  NAME mpi_disabled
  COMMAND mpi_disabled)
