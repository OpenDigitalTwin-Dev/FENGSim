project(idtf-to-u3d-converter)
cmake_minimum_required(VERSION 3.8)

option(U3D_USE_SYSTEM_EXTERNAL_LIBS "Use system libraries for jpeg, png and z instead of bundled ones" OFF)

option(U3D_BUILD_STATIC_IDTF_LIB "Build IDTFLib statically" ON)

option(U3D_BUILD_LIBIDTF_TEST "Build a Test to check that the idtf to u3d converter used from library works" ON)

option(U3D_BUILD_IFX_IMPORTING "Build IFXImporting" OFF)

option(U3D_BUILD_IDTF_TO_U3D_BIN "Build a binary that allows to convert IDTF files to U3D" ON)

option(U3D_BUILD_IDTF_GEN_BIN "Build the IDTFGen binary" OFF)

option(U3D_BUILD_HELLO_WORLD "Build Hello U3D World" OFF)

option(U3D_INSTALL_HEADERS "Add headers when installing U3D" OFF)

option(U3D_INSTALL_LIBS "Add libraries when installing U3D" ON)

option(U3D_INSTALL_SAMPLES_AND_DOCS "Add samples and docs when installing U3D" OFF)

set (U3D_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT U3D_USE_SYSTEM_EXTERNAL_LIBS)
	SET( LIB_DESTINATION lib )
	SET( BIN_DESTINATION bin )
	SET( INCLUDE_DESTINATION include/u3d )
	SET( PLUGIN_DESTINATION lib )
	SET( SAMPLE_DESTINATION share/u3d/samples )
	SET( DOC_DESTINATION share/u3d/docs )
else()
	SET( LIB_DESTINATION u3d )
	SET( BIN_DESTINATION u3d )
	SET( INCLUDE_DESTINATION u3d/include )
	SET( PLUGIN_DESTINATION u3d )
	SET( SAMPLE_DESTINATION u3d/samples )
	SET( DOC_DESTINATION u3d/docs )
endif()

# makes sure to disable no undefined flags when building u3d libraries
if (NOT APPLE AND NOT MSVC)
	string(REPLACE "-Wl,--no-undefined" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
	string(REPLACE "-Wl,--no-undefined" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
endif()

# I do not want to support asm code, so disable it
# there were real problems with cpuid on Ubuntu 13.04 i386
ADD_DEFINITIONS(-DU3D_NO_ASM)

if(NOT MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif( NOT MSVC )

IF(APPLE)
	ADD_DEFINITIONS( -DMAC32 -fno-strict-aliasing )
	SET( U3D_PLATFORM Mac32 )
ENDIF(APPLE)

IF(WIN32)
	ADD_DEFINITIONS( -DUNICODE -D_UNICODE -D_CRT_SECURE_NO_DEPRECATE )
	SET( U3D_PLATFORM Win32 )
	LINK_LIBRARIES( winmm )
	UNSET( CMAKE_SHARED_LIBRARY_PREFIX )
	UNSET( CMAKE_SHARED_MODULE_PREFIX )
	UNSET( CMAKE_STATIC_LIBRARY_PREFIX )
	UNSET( CMAKE_IMPORT_LIBRARY_PREFIX )
ENDIF(WIN32)

IF(UNIX AND NOT APPLE)
	ADD_DEFINITIONS( -DLIN32 -fno-strict-aliasing )
	SET( U3D_PLATFORM Lin32 )
ENDIF(UNIX AND NOT APPLE)

# management of external libraries: png, jpeg and z
include(cmake/external.cmake)

MESSAGE( STATUS "CMAKE_INSTALL_PREFIX:         " ${CMAKE_INSTALL_PREFIX} )
MESSAGE( STATUS "LIB_DESTINATION:         " ${LIB_DESTINATION} )

# common header declaration (Component, Kernel and Platform)
include(cmake/common.cmake)

# IFXCoreStatic target
include(cmake/IFXCoreStatic.cmake)

# IFXCore target
include(cmake/IFXCore.cmake)

# IFXExporting target
include(cmake/IFXExporting.cmake)

# IFXImporting target
if (U3D_BUILD_IFX_IMPORTING)
	include(cmake/IFXImporting.cmake)
endif()

# IFXScheduling target
include(cmake/IFXScheduling.cmake)

# IDTFConverterBin target
if (U3D_BUILD_IDTF_TO_U3D_BIN)
	include(cmake/IDTFConverterBin.cmake)
endif()

# IDTFLib target
include(cmake/IDTFLib.cmake)

if(U3D_BUILD_LIBIDTF_TEST)
	add_subdirectory(libIDTFTest)
endif()

# IDTFGen target
if (U3D_BUILD_IDTF_GEN_BIN)
	include(cmake/IDTFGen.cmake)
endif()

# HelloWorld target
if (U3D_BUILD_HELLO_WORLD)
	include(cmake/HelloWorld.cmake)
endif()

if (U3D_INSTALL_HEADERS)
	INSTALL( DIRECTORY RTL/Component/Include/ RTL/Kernel/Include/ RTL/Platform/Include/
		DESTINATION ${INCLUDE_DESTINATION}
		FILES_MATCHING PATTERN "*.h"
	)
	INSTALL( FILES RTL/Platform/${U3D_PLATFORM}/IFXRendering/OpenGL/IFXOpenGLOS.h
		DESTINATION ${INCLUDE_DESTINATION}
	)
	INSTALL( FILES ${libIDTF_HDRS}
		DESTINATION ${INCLUDE_DESTINATION}
	)
endif()

if(U3D_INSTALL_SAMPLES_AND_DOCS)
	INSTALL( FILES
		Samples/SampleCode/HelloWorld.cpp
		Samples/SampleCode/IDTFGen.cpp
		Samples/SampleCode/Makefile.sample
		Samples/SampleCode/CMakeLists.txt
		Samples/SampleCode/vtkU3DExporter.cxx
		Samples/SampleCode/vtkU3DExporter.h
		Samples/SampleCode/vtkU3DExporterTest.cxx
		DESTINATION ${SAMPLE_DESTINATION}/SampleCode
	)
	INSTALL( DIRECTORY Samples/TestScenes
		DESTINATION ${SAMPLE_DESTINATION}
		PATTERN "Makef*" EXCLUDE
	)
	INSTALL( DIRECTORY Docs/
		DESTINATION ${DOC_DESTINATION}
		PATTERN "Makef*" EXCLUDE
	)
endif()

# uninstall target
if (NOT TARGET uninstall)
	configure_file(
		"${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
		"${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
		IMMEDIATE @ONLY)

	add_custom_target(uninstall
		COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake)
endif()
