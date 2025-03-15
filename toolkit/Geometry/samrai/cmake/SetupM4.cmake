include(CMakeParseArguments)

# Macro to process m4 files to generate Fortran
macro (process_m4)
  set(options)
  set(singleValueArgs NAME)
  set(multiValueArgs )

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)

  set(m4_args -DPDAT_FORTDIR=${PROJECT_SOURCE_DIR}/source/SAMRAI/pdat/fortran)

  add_custom_command(
    OUTPUT ${arg_NAME}.f
    COMMAND m4
    ARGS ${m4_args} -DFORTDIR=${CMAKE_CURRENT_SOURCE_DIR}/fortran ${CMAKE_CURRENT_SOURCE_DIR}/${arg_NAME}.m4 > ${arg_NAME}.f
    VERBATIM)

  set_source_files_properties(${arg_NAME}.f PROPERTIES GENERATED true)
  set_source_files_properties(${arg_NAME}.f PROPERTIES LANGUAGE Fortran)
endmacro ()
