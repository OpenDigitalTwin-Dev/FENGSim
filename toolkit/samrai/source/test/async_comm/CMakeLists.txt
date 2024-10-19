blt_add_executable(
  NAME main-async_comm
  SOURCES main-async_comm.cpp
  DEPENDS_ON
    SAMRAI_tbox)

blt_add_executable(
  NAME main-peer_comm
  SOURCES main-peer_comm.cpp
  DEPENDS_ON
    SAMRAI_tbox)

target_include_directories( main-async_comm
  PUBLIC ${PROJECT_SOURCE_DIR}/source/test/main-async_comm)

target_include_directories( main-peer_comm
  PUBLIC ${PROJECT_SOURCE_DIR}/source/test/main-peer_comm)

target_compile_definitions(main-async_comm PUBLIC TESTING=1)

target_compile_definitions(main-peer_comm PUBLIC TESTING=1)

samrai_add_tests(
  NAME async_comm
  EXECUTABLE main-async_comm
  INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/async.default.input
  PARALLEL True)

samrai_add_tests(
  NAME async_comm_peer
  EXECUTABLE main-peer_comm
  INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/test_inputs/peer.default.input
  PARALLEL True)
