message(STATUS "Processing requested source directories")
add_subdirectory(SAMRAI)

message(STATUS "Enable Test Flag in source: ${ENABLE_SAMRAI_TESTS} " )
if (ENABLE_SAMRAI_TESTS)
   message(STATUS "Enable SAMRAI Tests Detected")
  add_subdirectory(test)
endif()
