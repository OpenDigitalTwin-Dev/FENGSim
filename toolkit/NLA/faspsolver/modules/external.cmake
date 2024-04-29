##################################################################
# For UMFPACK
##################################################################
if(USE_UMFPACK)

    # set some path to the UMFPACK pacakge
    # metis is not part of suitesparse, so theremay be also some other metis dir.
    set(METIS_DIR "${SUITESPARSE_DIR}")

    find_package(UMFPACK)
    if (UMFPACK_FOUND)
        add_definitions("-DWITH_UMFPACK=1")
        include_directories(${UMFPACK_INCLUDE_DIRS})
    else(UMFPACK_FOUND)
        message("-- WARNING: UMFPACK was requested but not supported! Continue without it.")
    endif(UMFPACK_FOUND)

endif(USE_UMFPACK)

##################################################################
# For SuperLU
##################################################################
if(USE_SUPERLU)

    # set the path to find specific modules
    set(SUPERLU_DIR "${SUPERLU_DIR}")

    # try to find SuperLU
    find_package(SUPERLU)

    if (SUPERLU_FOUND)
        add_definitions("-DWITH_SuperLU=1")
        include_directories(${SUPERLU_INCLUDE_DIRS})
    else(SUPERLU_FOUND)
        message("-- WARNING: SuperLU was requested but not supported! Continue without it.")
    endif(SUPERLU_FOUND)

endif(USE_SUPERLU)

##################################################################
# For MUMPS
##################################################################
if(USE_MUMPS)

    # set the path to find specific modules
    set(MUMPS_DIR "${MUMPS_DIR}")

    # try to find MUMPS and METIS (as dependency)
    find_package(METIS)
    find_package(MUMPS)

    if (MUMPS_FOUND)
        add_definitions("-DWITH_MUMPS=1")
        include_directories(${MUMPS_INCLUDE_DIRS})
    else(MUMPS_FOUND)
        message("-- WARNING: MUMPS was requested but not supported! Continue without it.")
    endif(MUMPS_FOUND)

endif(USE_MUMPS)

##################################################################
# For Intel MKL PARDISO
##################################################################
if(USE_PARDISO)

    # set the path to find specific modules
    set(MKL_DIR "${MKL_DIR}")

    # try to find MKL
    find_package(MKL)

    if (MKL_FOUND)
        add_definitions("-DWITH_PARDISO=1")
        include_directories(${MKL_INCLUDE_DIRS})
    else(MKL_FOUND)
        message("-- WARNING: Intel MKL was requested but not supported! Continue without it.")
    endif(MKL_FOUND)

endif(USE_PARDISO)

##################################################################
# For STRUMPACK
##################################################################
if(USE_STRUMPACK)

    # set the path to find specific modules
    set(STRUMPACK_DIR "${STRUMPACK_DIR}")

    # try to find STRUMPACK
    find_package(STRUMPACK REQUIRED)

    if (STRUMPACK_FOUND)
        add_definitions("-DWITH_STRUMPACK=1")
        include_directories(${STRUMPACK_INCLUDE_DIRS})
    else(STRUMPACK_FOUND)
        message("-- WARNING: STRUMPACK was requested but not supported! Continue without it.")
    endif(STRUMPACK_FOUND)

endif(USE_STRUMPACK)

##################################################################
# For Doxygen
##################################################################
if(USE_DOXYGEN)

    find_package(Doxygen)

    if(DOXYGEN_FOUND)
        if(EXISTS ${FASP_SOURCE_DIR}/doc/fasp.Doxygen.cnf.in)
            configure_file(
                    ${FASP_SOURCE_DIR}/doc/fasp.Doxygen.cnf.in
                    ${CMAKE_CURRENT_BINARY_DIR}/fasp.Doxygen.cnf @ONLY)
            set(DOXY_EXEC "${DOXYGEN_EXECUTABLE}")
            if(DOXYWIZARD)
                find_program(WIZARD doxywizard)
                if(APPLE AND (NOT WIZARD) )
                    find_program(WIZARD
                            /Applications/Doxygen.app/Contents/MacOS/Doxywizard)
                endif()
                if(WIZARD)
                    set(DOXY_EXEC "${WIZARD}")
                endif()
            endif(DOXYWIZARD)
            add_custom_target(docs ${DOXY_EXEC}
                    ${CMAKE_CURRENT_BINARY_DIR}/fasp.Doxygen.cnf
                    WORKING_DIRECTORY
                    "${CMAKE_CURRENT_BINARY_DIR}"
                    COMMENT
                    "Generating FASP documentation (Doxygen)"
                    VERBATIM)
        else(EXISTS ${FASP_SOURCE_DIR}/doc/fasp.Doxygen.cnf.in)
            message("-- WARNING: Doxygen configuration file cannot be found! Continue without it.")
        endif(EXISTS ${FASP_SOURCE_DIR}/doc/fasp.Doxygen.cnf.in)
    endif(DOXYGEN_FOUND)

endif(USE_DOXYGEN)
