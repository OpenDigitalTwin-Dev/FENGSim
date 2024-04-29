# - Try to find RT
# 
#  OUTPUT:
#  RT_FOUND        - system has RT
#  RT_INCLUDE_DIRS - include directories for RT
#  RT_LIBRARIES    - libraries for RT
#
#  Xiaozhe Hu
#  02/27/2013

message(STATUS "Checking for package 'RT'")

# Check for header file
find_path(RT_INCLUDE_DIRS time.h
 HINTS /usr/include /usr/local/include /opt/local/include
 DOC "Directory where the RT header is located"
 )
mark_as_advanced(RT_INCLUDE_DIRS)

# Check for RT library
find_library(RT_LIBRARIES rt
  HINTS /usr/lib /usr/local/lib /opt/local/lib
  DOC "The RT library"
  )
mark_as_advanced(RT_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RT
  "RT could not be found. Be sure to set RT_DIR."
  RT_LIBRARIES RT_INCLUDE_DIRS)
