set (defaultlogger_sources
        defaultloggertest.cpp)

blt_add_executable(
        NAME defaultlogger
        SOURCES ${defaultlogger_sources}
        DEPENDS_ON
          SAMRAI_tbox)

set (userlogger_sources
        userloggertest.cpp)

blt_add_executable(
        NAME userlogger
        SOURCES ${userlogger_sources}
        DEPENDS_ON
        SAMRAI_tbox)

if(ENABLE_MPI)
  set(TASKS 1)
else()
  set(TASKS 0)
endif()

blt_add_test(
  NAME defaultlogger
  COMMAND defaultlogger
  NUM_MPI_TASKS ${TASKS})

blt_add_test(
  NAME userlogger
  COMMAND userlogger
  NUM_MPI_TASKS ${TASKS})
