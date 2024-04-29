## build test library

set(FASP_TLIB_DIR ${PROJECT_SOURCE_DIR}/test)
set(FASP_TINC_DIR ${PROJECT_SOURCE_DIR}/test/include)
set(RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test)

## requires math library
if(UNIX OR LINUX)
	set(ADD_MLIBS m)
endif()

include_directories(${FASP_TINC_DIR}
		    ${PROJECT_SOURCE_DIR}/base/include)

file(GLOB FASP_TLIB_SRC src/*.c)

# Build libfasptest as STATIC and do not install it. If we want
# SHARED, then we need to figure out how apple works with multiple
# rpaths put the shared lib in the same dir as the executables.

add_library(fasptest STATIC ${FASP_TLIB_SRC}) 

if(WITHFC)
    ## requires GNU Fortran library
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	    set(ADD_FLIBS gfortran)
    endif()
    file( GLOB FASP_TSRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/main" 
          main/*.c main/*.f90 )
else()
    file( GLOB FASP_TSRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/main" 
          main/*.c )
endif()

foreach( srcfile ${FASP_TSRC} )
    string( REGEX REPLACE "\\.[^\\.]*$" ".ex" execfile ${srcfile})
    add_executable( ${execfile} main/${srcfile})
# 
    target_link_libraries(${execfile} fasptest 
                          ${FASP_LIBRARY_NAME}
                          ${UMFPACK_LIBRARIES} 
                          ${SUPERLU_LIBRARIES}
                          ${METIS_LIBRARIES} 
                          ${MUMPS_LIBRARIES}
                          ${MKL_LIBRARIES}
                          ${STRUMPACK_LIBRARIES}
                          ${ADD_MLIBS}
                          ${ADD_FLIBS})
#
    install(TARGETS ${execfile} 
            DESTINATION ${PROJECT_SOURCE_DIR}/test)
endforeach( srcfile ${FASP_TSRC} )