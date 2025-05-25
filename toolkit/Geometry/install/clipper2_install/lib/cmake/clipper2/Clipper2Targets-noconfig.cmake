#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Clipper2::Clipper2" for configuration ""
set_property(TARGET Clipper2::Clipper2 APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Clipper2::Clipper2 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libClipper2.a"
  )

list(APPEND _cmake_import_check_targets Clipper2::Clipper2 )
list(APPEND _cmake_import_check_files_for_Clipper2::Clipper2 "${_IMPORT_PREFIX}/lib/libClipper2.a" )

# Import target "Clipper2::Clipper2Z" for configuration ""
set_property(TARGET Clipper2::Clipper2Z APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Clipper2::Clipper2Z PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libClipper2Z.a"
  )

list(APPEND _cmake_import_check_targets Clipper2::Clipper2Z )
list(APPEND _cmake_import_check_files_for_Clipper2::Clipper2Z "${_IMPORT_PREFIX}/lib/libClipper2Z.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
