# TODO This should be taken care of by CMake...
if (ENABLE_OPENMP)
  set (depends openmp)
endif ()

blt_add_executable(NAME restart-redistribute
  SOURCES main.C RedistributedRestartUtility.C
  DEPENDS_ON SAMRAI_tbox ${depends})

target_include_directories(
  restart-redistribute
  PRIVATE ${PROJECT_SOURCE_DIR}/source/)
