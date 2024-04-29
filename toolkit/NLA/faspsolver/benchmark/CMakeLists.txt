## build benchmark test

set(RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark)

## requires math library
if(UNIX OR LINUX)
    set(ADD_STDLIBS m)
endif()

file( GLOB FASP_BSRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/main" 
      main/*.c main/*.f main/*.f90 )

include_directories(${PROJECT_SOURCE_DIR}/base/include)

foreach( srcfile ${FASP_BSRC} )
    string( REGEX REPLACE "\\.[^\\.]*$" ".ex" execfile ${srcfile} )
    if(execfile)
        add_executable( ${execfile} main/${srcfile} )
        target_link_libraries( ${execfile} 
                               ${FASP_LIBRARY_NAME}
                               ${UMFPACK_LIBRARIES} 
                               ${SUPERLU_LIBRARIES} 
                               ${MUMPS_LIBRARIES}
                               ${MKL_LIBRARIES}
                               ${ADD_STDLIBS} )      
        install(TARGETS ${execfile} DESTINATION ${PROJECT_SOURCE_DIR}/benchmark)
    endif(execfile)
endforeach( srcfile ${FASP_BSRC} )
