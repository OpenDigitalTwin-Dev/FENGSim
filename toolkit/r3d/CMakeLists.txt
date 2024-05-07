

cmake_minimum_required(VERSION 3.13)

project(r3d C CXX)   # CXX needed for one of the tests



# Set a default R3D_MAX_VERTS that can be overridden on the command line
set(R3D_MAX_VERTS 512 CACHE STRING "Maximum number of vertices in a polyhedron")


# Unit testing
if (ENABLE_UNIT_TESTS)
  enable_testing()
endif ()


add_library(r3d)

# Alias (Daniel Pfeiffer, Effective CMake) - this allows other
# projects that use Pkg as a subproject to find_package(Nmspc::Pkg)
# which does nothing because Pkg is already part of the project

add_library(r3d::r3d ALIAS r3d)
set(r3d_LIBRARIES r3d::r3d CACHE STRING "R3D library target")


add_subdirectory(src)


# Wonton targets

install(TARGETS r3d
  EXPORT r3d_LIBRARIES
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
  INCLUDES DESTINATION include
  )

#-----------------------------------------------------------------------------
# Prepare output for configuration files to be used by projects importing Wonton
#-----------------------------------------------------------------------------

# Write a configuration file from template replacing only variables enclosed
# by the @ sign.

configure_file(${PROJECT_SOURCE_DIR}/config/r3dConfig.cmake.in 
  r3dConfig.cmake @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/r3dConfig.cmake
  DESTINATION lib/cmake/r3d)


# export targets

install(EXPORT r3d_LIBRARIES
  FILE r3dTargets.cmake
  NAMESPACE r3d::
  EXPORT_LINK_INTERFACE_LIBRARIES
  DESTINATION lib/cmake/r3d)



# Dynamically configured header files that contains defines like
# R3D_MAX_VERTS if enabled.

configure_file(${PROJECT_SOURCE_DIR}/config/r3d-config.h.in
  ${PROJECT_BINARY_DIR}/r3d-config.h @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/r3d-config.h
  DESTINATION include)

