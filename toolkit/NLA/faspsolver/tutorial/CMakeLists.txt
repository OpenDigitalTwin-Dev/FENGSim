## build tutorial programs

set(RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tutorial)

## requires math library
if(UNIX OR LINUX)
	set(ADD_MLIBS m)
endif()

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

include_directories(${PROJECT_SOURCE_DIR}/base/include)

foreach( srcfile ${FASP_TUTSRC} )
 	string( REGEX MATCH "\\.c$|\\.f90$|\\.cpp$" X ${srcfile} )
    if("${X}" STREQUAL ".c")
        string( REGEX REPLACE "\\.c$" "-c.ex" execfile ${srcfile} )
    elseif("${X}" STREQUAL ".f90")
        string( REGEX REPLACE "\\.f90$" "-f.ex" execfile ${srcfile} )
    elseif("${X}" STREQUAL ".cpp")
        string( REGEX REPLACE "\\.cpp$" "-cpp.ex" execfile ${srcfile} )
    endif()
    if(execfile)
        add_executable( ${execfile} main/${srcfile} )
        target_link_libraries(${execfile} 
                              ${FASP_LIBRARY_NAME}
                              ${UMFPACK_LIBRARIES} 
                              ${SUPERLU_LIBRARIES}
                              ${METIS_LIBRARIES}
                              ${MUMPS_LIBRARIES}
                              ${MKL_LIBRARIES}
                              ${ADD_MLIBS}
                              ${ADD_FLIBS})     
        install(TARGETS ${execfile} DESTINATION ${PROJECT_SOURCE_DIR}/tutorial)
    endif(execfile)
endforeach( srcfile ${FASP_TUTSRC} )