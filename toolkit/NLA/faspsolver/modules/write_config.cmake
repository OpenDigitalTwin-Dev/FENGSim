#  
#  Writing configuration so that it can be used by regular makefiles 
#
#  Modified   20151017   --ltz
###################################################################
set(CONFIGMK ${PROJECT_SOURCE_DIR}/Config.mk)
message("-- Writing FASP configuration to ${CONFIGMK}")
# determinining library name; this is  a bit hard coding here
if(SHARED)
    set(FASP_LIBRARY_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
    set(FASP_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
else(SHARED)
    set(FASP_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(FASP_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif(SHARED)

file(WRITE  ${CONFIGMK} 
"
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# Automatically generated \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
\# Fast Auxiliary Space Preconditioners (FASP) 
\#
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
\# This file is rewritten when \"make config\" is run.
\# It is (s)included by test/Makefile and tutorial/Makefile
\#
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
fasp_prefix=${FASP_INSTALL_PREFIX}
fasp_library=${FASP_LIBRARY_PREFIX}${FASP_LIBRARY_NAME}${FASP_LIBRARY_SUFFIX}
CC=${CMAKE_C_COMPILER}
CXX=${CMAKE_CXX_COMPILER}
FC=${CMAKE_Fortran_COMPILER}
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
")
