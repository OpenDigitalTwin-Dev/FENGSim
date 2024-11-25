MESSAGE( STATUS "U3D_USE_SYSTEM_EXTERNAL_LIBS:         " ${U3D_USE_SYSTEM_EXTERNAL_LIBS} )

IF(U3D_USE_SYSTEM_EXTERNAL_LIBS)

	# check zlib availibility
	find_package(ZLIB REQUIRED)

	# check png availibility
	find_package(PNG REQUIRED)
	include_directories(${PNG_INCLUDE_DIR})
	add_definitions(${PNG_DEFINITIONS})
	set(U3D_ADDITIONAL_LIBRARIES ${U3D_ADDITIONAL_LIBRARIES} ${PNG_LIBRARIES})

	# check jpeg availibility
	find_package(JPEG REQUIRED)
	include_directories(${JPEG_INCLUDE_DIR})
	set(U3D_ADDITIONAL_LIBRARIES ${U3D_ADDITIONAL_LIBRARIES} ${JPEG_LIBRARIES})

	SET_PROPERTY( SOURCE
		RTL/Platform/${U3D_PLATFORM}/Common/IFXOSLoader.cpp
		PROPERTY COMPILE_DEFINITIONS U3DPluginsPath="." U3DCorePath="${CMAKE_INSTALL_PREFIX}/${LIB_DESTINATION}" )
	IF(STDIO_HACK)
	SET_PROPERTY( SOURCE
		RTL/Component/Exporting/CIFXStdioWriteBufferX.cpp
		IDTF/ConverterDriver.cpp
		IDTF/File.cpp
		PROPERTY COMPILE_DEFINITIONS STDIO_HACK )
	SET_PROPERTY( SOURCE
		RTL/Platform/${U3D_PLATFORM}/Common/IFXOSLoader.cpp
		PROPERTY COMPILE_DEFINITIONS U3DPluginsPath="." U3DCorePath="${CMAKE_INSTALL_PREFIX}/${LIB_DESTINATION}" STDIO_HACK )
	ENDIF(STDIO_HACK)

ELSE(U3D_USE_SYSTEM_EXTERNAL_LIBS)
	#============================================================================
	# zlib
	#============================================================================

	include(CheckTypeSize)
	include(CheckFunctionExists)
	include(CheckIncludeFile)
	include(CheckCSourceCompiles)

	check_include_file(sys/types.h HAVE_SYS_TYPES_H)
	check_include_file(stdint.h    HAVE_STDINT_H)
	check_include_file(stddef.h    HAVE_STDDEF_H)

	#
	# Check to see if we have large file support
	#
	set(CMAKE_REQUIRED_DEFINITIONS -D_LARGEFILE64_SOURCE=1)
	check_type_size(off64_t OFF64_T)
	if(HAVE_OFF64_T)
	   add_definitions(-D_LARGEFILE64_SOURCE=1)
	endif()
	set(CMAKE_REQUIRED_DEFINITIONS) # clear variable

	#
	# Check for fseeko
	#
	check_function_exists(fseeko HAVE_FSEEKO)
	if(NOT HAVE_FSEEKO)
		add_definitions(-DNO_FSEEKO)
	endif()

	#
	# Check for unistd.h
	#
	check_include_file(unistd.h Z_HAVE_UNISTD_H)

	if(MSVC)
		add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
		add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
	endif()

	SET(ZLIB_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Dependencies/zlib)
	configure_file(${ZLIB_SOURCE_DIR}/zconf.h.cmakein
				   ${CMAKE_CURRENT_BINARY_DIR}/zconf.h @ONLY)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})
	include_directories(${ZLIB_SOURCE_DIR})


	set(ZLIB_PUBLIC_HDRS
		${CMAKE_CURRENT_BINARY_DIR}/zconf.h
		${ZLIB_SOURCE_DIR}/zlib.h
	)
	set(ZLIB_PRIVATE_HDRS
		${ZLIB_SOURCE_DIR}/crc32.h
		${ZLIB_SOURCE_DIR}/deflate.h
		${ZLIB_SOURCE_DIR}/gzguts.h
		${ZLIB_SOURCE_DIR}/inffast.h
		${ZLIB_SOURCE_DIR}/inffixed.h
		${ZLIB_SOURCE_DIR}/inflate.h
		${ZLIB_SOURCE_DIR}/inftrees.h
		${ZLIB_SOURCE_DIR}/trees.h
		${ZLIB_SOURCE_DIR}/zutil.h
	)
	set(ZLIB_SRCS
		${ZLIB_SOURCE_DIR}/adler32.c
		${ZLIB_SOURCE_DIR}/compress.c
		${ZLIB_SOURCE_DIR}/crc32.c
		${ZLIB_SOURCE_DIR}/deflate.c
		${ZLIB_SOURCE_DIR}/gzclose.c
		${ZLIB_SOURCE_DIR}/gzlib.c
		${ZLIB_SOURCE_DIR}/gzread.c
		${ZLIB_SOURCE_DIR}/gzwrite.c
		${ZLIB_SOURCE_DIR}/inflate.c
		${ZLIB_SOURCE_DIR}/infback.c
		${ZLIB_SOURCE_DIR}/inftrees.c
		${ZLIB_SOURCE_DIR}/inffast.c
		${ZLIB_SOURCE_DIR}/trees.c
		${ZLIB_SOURCE_DIR}/uncompr.c
		${ZLIB_SOURCE_DIR}/zutil.c
	)


	#============================================================================
	# png
	#============================================================================

	if(NOT WIN32)
		find_library(M_LIBRARY
			NAMES m
			PATHS /usr/lib /usr/local/lib)
		if(NOT M_LIBRARY)
			message(STATUS
			"math library 'libm' not found - floating point support disabled")
		endif()
	else()
		# not needed on windows
		set(M_LIBRARY "")
	endif()

	set(PNG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Dependencies/png)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	# OUR SOURCES
	set(libpng_public_hdrs
		${PNG_SOURCE_DIR}/png.h
		${PNG_SOURCE_DIR}/pngconf.h
		${PNG_SOURCE_DIR}/pnglibconf.h
	)
	set(libpng_sources
		${libpng_public_hdrs}
		${PNG_SOURCE_DIR}/pngdebug.h
		${PNG_SOURCE_DIR}/pnginfo.h
		${PNG_SOURCE_DIR}/pngpriv.h
		${PNG_SOURCE_DIR}/pngstruct.h
		${PNG_SOURCE_DIR}/png.c
		${PNG_SOURCE_DIR}/pngerror.c
		${PNG_SOURCE_DIR}/pngget.c
		${PNG_SOURCE_DIR}/pngmem.c
		${PNG_SOURCE_DIR}/pngpread.c
		${PNG_SOURCE_DIR}/pngread.c
		${PNG_SOURCE_DIR}/pngrio.c
		${PNG_SOURCE_DIR}/pngrtran.c
		${PNG_SOURCE_DIR}/pngrutil.c
		${PNG_SOURCE_DIR}/pngset.c
		${PNG_SOURCE_DIR}/pngtrans.c
		${PNG_SOURCE_DIR}/pngwio.c
		${PNG_SOURCE_DIR}/pngwrite.c
		${PNG_SOURCE_DIR}/pngwtran.c
		${PNG_SOURCE_DIR}/pngwutil.c
	)
	# SOME NEEDED DEFINITIONS

	if(MSVC)
		add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
	endif(MSVC)

	set(U3D_ADDITIONAL_LIBRARIES ${U3D_ADDITIONAL_LIBRARIES} ${M_LIBRARY})
	include_directories(${PNG_SOURCE_DIR})

	CHECK_INCLUDE_FILE(stdlib.h HAVE_STDLIB_H)

	set(JPEG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Dependencies/jpeg)
	configure_file(
		${JPEG_SOURCE_DIR}/jconfig.h.cmake
		${CMAKE_CURRENT_BINARY_DIR}/jconfig.h)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})
	include_directories(${JPEG_SOURCE_DIR})

	if(MSVC)
		add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	endif()

	set(JPEG_PUBLIC_HDRS
		${JPEG_SOURCE_DIR}/jerror.h
		${JPEG_SOURCE_DIR}/jmorecfg.h
		${JPEG_SOURCE_DIR}/jpeglib.h
		${CMAKE_CURRENT_BINARY_DIR}/jconfig.h
	)
	set(JPEG_PRIVATE_HDRS
		${JPEG_SOURCE_DIR}/cderror.h
		${JPEG_SOURCE_DIR}/jdct.h
		${JPEG_SOURCE_DIR}/jinclude.h
		${JPEG_SOURCE_DIR}/jmemsys.h
		${JPEG_SOURCE_DIR}/jpegint.h
		${JPEG_SOURCE_DIR}/jversion.h
		${JPEG_SOURCE_DIR}/transupp.h
	)

	# memmgr back ends: compile only one of these into a working library
	# (For now, let's use the mode that requires the image fit into memory.
	# This is the recommended mode for Win32 anyway.)
	SET(JPEG_systemdependent_SRCS ${JPEG_SOURCE_DIR}/jmemnobs.c)

	set(JPEG_SRCS
		${JPEG_SOURCE_DIR}/jaricom.c
		${JPEG_SOURCE_DIR}/jcapimin.c
		${JPEG_SOURCE_DIR}/jcapistd.c
		${JPEG_SOURCE_DIR}/jcarith.c
		${JPEG_SOURCE_DIR}/jccoefct.c
		${JPEG_SOURCE_DIR}/jccolor.c
		${JPEG_SOURCE_DIR}/jcdctmgr.c
		${JPEG_SOURCE_DIR}/jchuff.c
		${JPEG_SOURCE_DIR}/jcinit.c
		${JPEG_SOURCE_DIR}/jcmainct.c
		${JPEG_SOURCE_DIR}/jcmarker.c
		${JPEG_SOURCE_DIR}/jcmaster.c
		${JPEG_SOURCE_DIR}/jcomapi.c
		${JPEG_SOURCE_DIR}/jcparam.c
		${JPEG_SOURCE_DIR}/jcprepct.c
		${JPEG_SOURCE_DIR}/jcsample.c
		${JPEG_SOURCE_DIR}/jctrans.c
		${JPEG_SOURCE_DIR}/jdapimin.c
		${JPEG_SOURCE_DIR}/jdapistd.c
		${JPEG_SOURCE_DIR}/jdarith.c
		${JPEG_SOURCE_DIR}/jdatadst.c
		${JPEG_SOURCE_DIR}/jdatasrc.c
		${JPEG_SOURCE_DIR}/jdcoefct.c
		${JPEG_SOURCE_DIR}/jdcolor.c
		${JPEG_SOURCE_DIR}/jddctmgr.c
		${JPEG_SOURCE_DIR}/jdhuff.c
		${JPEG_SOURCE_DIR}/jdinput.c
		${JPEG_SOURCE_DIR}/jdmainct.c
		${JPEG_SOURCE_DIR}/jdmarker.c
		${JPEG_SOURCE_DIR}/jdmaster.c
		${JPEG_SOURCE_DIR}/jdmerge.c
		${JPEG_SOURCE_DIR}/jdpostct.c
		${JPEG_SOURCE_DIR}/jdsample.c
		${JPEG_SOURCE_DIR}/jdtrans.c
		${JPEG_SOURCE_DIR}/jerror.c
		${JPEG_SOURCE_DIR}/jfdctflt.c
		${JPEG_SOURCE_DIR}/jfdctfst.c
		${JPEG_SOURCE_DIR}/jfdctint.c
		${JPEG_SOURCE_DIR}/jidctflt.c
		${JPEG_SOURCE_DIR}/jidctfst.c
		${JPEG_SOURCE_DIR}/jidctint.c
		${JPEG_SOURCE_DIR}/jquant1.c
		${JPEG_SOURCE_DIR}/jquant2.c
		${JPEG_SOURCE_DIR}/jutils.c
		${JPEG_SOURCE_DIR}/jmemmgr.c)

	SET(DEPENDENCIES_SRCS ${ZLIB_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS} ${libpng_sources} ${JPEG_systemdependent_SRCS} ${JPEG_SRCS} ${JPEG_PUBLIC_HDRS} ${JPEG_PRIVATE_HDRS})
	SET_PROPERTY( SOURCE
		RTL/Platform/${U3D_PLATFORM}/Common/IFXOSLoader.cpp
		PROPERTY COMPILE_DEFINITIONS U3DPluginsPath="Plugins" U3DCorePath="." )
	IF(STDIO_HACK)
	SET_PROPERTY( SOURCE
		RTL/Component/Exporting/CIFXStdioWriteBufferX.cpp
		IDTF/ConverterDriver.cpp
		IDTF/File.cpp
		PROPERTY COMPILE_DEFINITIONS STDIO_HACK )
	SET_PROPERTY( SOURCE
		RTL/Platform/${U3D_PLATFORM}/Common/IFXOSLoader.cpp
		PROPERTY COMPILE_DEFINITIONS U3DPluginsPath="Plugins" U3DCorePath="." STDIO_HACK )
	ENDIF(STDIO_HACK)

ENDIF(U3D_USE_SYSTEM_EXTERNAL_LIBS)
