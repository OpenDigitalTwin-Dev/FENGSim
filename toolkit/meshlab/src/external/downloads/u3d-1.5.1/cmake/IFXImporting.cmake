# IFXImporting
INCLUDE_DIRECTORIES(
	${U3D_DIR}/RTL/Component/Include
	${U3D_DIR}/RTL/Kernel/Include
	${U3D_DIR}/RTL/Platform/Include
	${U3D_DIR}/RTL/Component/Importing
	${U3D_DIR}/RTL/Dependencies/WildCards )
SET( IFXImporting_HDRS
	${Component_HDRS}
	${Kernel_HDRS}
	${Platform_HDRS}
	${U3D_DIR}/RTL/Component/Importing/CIFXAnimationModifierDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXAuthorCLODDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXBlockReaderX.h
	${U3D_DIR}/RTL/Component/Importing/CIFXBoneWeightsModifierDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXBTTHash.h
	${U3D_DIR}/RTL/Component/Importing/CIFXCLODModifierDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXDecoderChainX.h
	${U3D_DIR}/RTL/Component/Importing/CIFXDummyModifierDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXGlyphModifierDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXGroupDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXInternetReadBufferX.h
	${U3D_DIR}/RTL/Component/Importing/CIFXLightDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXLightResourceDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXLineSetDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXLoadManager.h
	${U3D_DIR}/RTL/Component/Importing/CIFXMaterialDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXModelDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXMotionDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXNodeDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXPointSetDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXShaderLitTextureDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXShadingModifierDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXStdioReadBufferX.h
	${U3D_DIR}/RTL/Component/Importing/CIFXSubdivisionModifierDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXTextureDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXViewDecoder.h
	${U3D_DIR}/RTL/Component/Importing/CIFXViewResourceDecoder.h
	${U3D_DIR}/RTL/Component/Importing/IFXInternetConnectionX.h
	${U3D_DIR}/RTL/Component/Importing/IFXInternetSessionX.h
	${U3D_DIR}/RTL/Component/Importing/IFXSocketStream.h
	${U3D_DIR}/RTL/Dependencies/WildCards/wcmatch.h
)
SET( IFXImporting_SRCS
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/IFXImporting/IFXImportingDllMain.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXAnimationModifierDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXAuthorCLODDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXAuthorCLODDecoder_P.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXAuthorCLODDecoder_S.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXBlockReaderX.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXBoneWeightsModifierDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXBTTHash.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXCLODModifierDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXDecoderChainX.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXDummyModifierDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXGlyphModifierDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXGroupDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXInternetReadBufferX.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXLightDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXLightResourceDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXLineSetDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXLoadManager.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXMaterialDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXModelDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXMotionDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXNodeDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXPointSetDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXShaderLitTextureDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXShadingModifierDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXStdioReadBufferX.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXSubdivisionModifierDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXTextureDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXViewDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/CIFXViewResourceDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/IFXImporting.cpp
	${U3D_DIR}/RTL/Component/Importing/IFXImportingGuids.cpp
	${U3D_DIR}/RTL/Component/Importing/IFXInternetConnectionX.cpp
	${U3D_DIR}/RTL/Component/Importing/IFXInternetSessionX.cpp
	${U3D_DIR}/RTL/Component/Importing/IFXModifierBaseDecoder.cpp
	${U3D_DIR}/RTL/Component/Importing/IFXSocketStream.cpp
	${U3D_DIR}/RTL/IFXCorePluginStatic/IFXCorePluginStatic.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSUtilities.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSLoader.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSSocket.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXCoreArray.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXCoreList.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXFastAllocator.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXListNode.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXMatrix4x4.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXQuaternion.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXString.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXUnitAllocator.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXVector3.cpp
	${U3D_DIR}/RTL/Kernel/DataTypes/IFXVector4.cpp
	${U3D_DIR}/RTL/Dependencies/WildCards/wcmatch.cpp
	${U3D_DIR}/RTL/Kernel/Common/IFXDebug.cpp
	)
IF(WIN32)
	SET( IMPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Win32/IFXImporting )
	ADD_LIBRARY( IFXImporting  SHARED
		${IFXImporting_SRCS} ${IFXImporting_HDRS} ${IMPORT_DIR}/IFXImporting.rc ${IMPORT_DIR}/IFXResource.h ${IMPORT_DIR}/IFXImporting.def )
	TARGET_LINK_LIBRARIES( IFXImporting IFXCore ws2_32 )
ENDIF(WIN32)
IF(APPLE)
	ADD_LIBRARY( IFXImporting  MODULE
		${IFXImporting_SRCS} ${IFXImporting_HDRS} )
	set_target_properties( IFXImporting  PROPERTIES
		LINK_FLAGS "${MY_LINK_FLAGS} -exported_symbols_list ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Mac32/IFXImporting/IFXImporting.def   -undefined dynamic_lookup" )
ENDIF(APPLE)
IF(UNIX AND NOT APPLE)
	ADD_LIBRARY( IFXImporting  MODULE
		${IFXImporting_SRCS} ${IFXImporting_HDRS} )
	set_target_properties( IFXImporting  PROPERTIES
		LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Lin32/IFXImporting/IFXImporting.list" )
ENDIF(UNIX AND NOT APPLE)

if(U3D_INSTALL_LIBS)
	install(
		TARGETS IFXImporting
		RUNTIME DESTINATION ${BIN_DESTINATION}
		ARCHIVE DESTINATION ${LIB_DESTINATION}
		LIBRARY DESTINATION ${PLUGIN_DESTINATION}
	)
endif()
