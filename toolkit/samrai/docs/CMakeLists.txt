if (DOXYGEN_FOUND)
  set (top_srcdir ${CMAKE_SOURCE_DIR})
  blt_add_doxygen_target( samrai_doxygen )
endif ()
