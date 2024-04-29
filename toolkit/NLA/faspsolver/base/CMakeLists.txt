#
set(FASP_BASE_SRC_DIR "src")
#
set(FASP_EXTRA_SRC_DIR "extra")
#
file(GLOB FASP_EXTRA_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  
                                      ${FASP_EXTRA_SRC_DIR}/hb_io/*.c
                                      ${FASP_EXTRA_SRC_DIR}/interface/*.c)
#
file(GLOB FASP_BASE_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
                                     ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
                                     ${FASP_EXTRA_SRC_DIR}/include/*.h)
#
file(GLOB FASP_BASE_INL RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
                                 ${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl)
#
file(GLOB FASP_BASE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
                                     ${FASP_BASE_SRC_DIR}/*.c 
                                     ${FASP_BASE_SRC_DIR}/*.f 
                                     ${FASP_EXTRA_SOURCES} )
#
# Build libfasp
# depends also on the source, header, and inl files 
add_library(${FASP_LIBRARY_NAME} ${FASP_LIBRARY_TYPE} 
            ${FASP_BASE_SOURCES} ${FASP_BASE_INL} )

# Add include directories.
target_include_directories(${FASP_LIBRARY_NAME} 
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extra/include)

# install libfasp and header files. 
install(TARGETS ${FASP_LIBRARY_NAME}
        LIBRARY DESTINATION ${FASP_INSTALL_PREFIX}/lib
    	    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
        RUNTIME DESTINATION ${FASP_INSTALL_PREFIX}/lib
    	    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
        ARCHIVE DESTINATION ${FASP_INSTALL_PREFIX}/lib
    	    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)
file(GLOB FASP_INC  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
    install(FILES ${FASP_INC}
            DESTINATION ${FASP_INSTALL_PREFIX}/include
            PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)
