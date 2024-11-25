# IFXExporting
set(IFXExporting_Dirs
	${U3D_DIR}/RTL/Component/Include
	${U3D_DIR}/RTL/Kernel/Include
	${U3D_DIR}/RTL/Platform/Include
	${U3D_DIR}/RTL/Component/Exporting
	${U3D_DIR}/RTL/Dependencies/WildCards)

SET(IFXExporting_HDRS
	${Component_HDRS}
	${Kernel_HDRS}
	${Platform_HDRS}
	${U3D_DIR}/RTL/Component/Exporting/CIFXAnimationModifierEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXAuthorCLODEncoderX.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXAuthorGeomCompiler.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXBlockPriorityQueueX.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXBlockWriterX.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXBoneWeightsModifierEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXCLODModifierEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXDummyModifierEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXFileReferenceEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXGlyphModifierEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXGroupNodeEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXLightNodeEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXLightResourceEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXLineSetEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXMaterialResourceEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXModelNodeEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXMotionResourceEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXNodeBaseEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXPointSetEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXShaderLitTextureEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXShadingModifierEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXStdioWriteBufferX.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXSubdivisionModifierEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXViewNodeEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXViewResourceEncoder.h
	${U3D_DIR}/RTL/Component/Exporting/CIFXWriteManager.h
	${U3D_DIR}/RTL/Dependencies/WildCards/wcmatch.h
)

SET(IFXExporting_SRCS
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/IFXExporting/IFXExportingDllMain.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXAnimationModifierEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXAuthorCLODEncoderX.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXAuthorCLODEncoderX_P.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXAuthorCLODEncoderX_S.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXAuthorGeomCompiler.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXBlockPriorityQueueX.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXBlockWriterX.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXBoneWeightsModifierEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXCLODModifierEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXDummyModifierEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXFileReferenceEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXGlyphModifierEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXGroupNodeEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXLightNodeEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXLightResourceEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXLineSetEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXMaterialResourceEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXModelNodeEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXMotionResourceEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXNodeBaseEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXPointSetEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXShaderLitTextureEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXShadingModifierEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXStdioWriteBufferX.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXSubdivisionModifierEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXViewNodeEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXViewResourceEncoder.cpp
	${U3D_DIR}/RTL/Component/Exporting/CIFXWriteManager.cpp
	${U3D_DIR}/RTL/Component/Exporting/IFXExporting.cpp
	${U3D_DIR}/RTL/Component/Exporting/IFXExportingGuids.cpp
	${U3D_DIR}/RTL/IFXCorePluginStatic/IFXCorePluginStatic.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSUtilities.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSLoader.cpp
	${U3D_DIR}/RTL/Component/Base/IFXVertexMap.cpp
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
	SET(EXPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Win32/IFXExporting)
	ADD_LIBRARY(IFXExporting SHARED
		${IFXExporting_SRCS} ${IFXExporting_HDRS} ${EXPORT_DIR}/IFXExporting.rc ${EXPORT_DIR}/IFXResource.h ${EXPORT_DIR}/IFXExporting.def)
	TARGET_LINK_LIBRARIES(IFXExporting IFXCore)
ENDIF(WIN32)
IF(APPLE)
	ADD_LIBRARY(IFXExporting MODULE
		${IFXExporting_SRCS} ${IFXExporting_HDRS})
	set_target_properties( IFXExporting PROPERTIES
		LINK_FLAGS "${MY_LINK_FLAGS} -exported_symbols_list ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Mac32/IFXExporting/IFXExporting.def   -undefined dynamic_lookup")
ENDIF(APPLE)
IF(UNIX AND NOT APPLE)
	ADD_LIBRARY(IFXExporting  MODULE ${IFXExporting_SRCS} ${IFXExporting_HDRS})
	set_target_properties(IFXExporting PROPERTIES
		LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Lin32/IFXExporting/IFXExporting.list")
ENDIF(UNIX AND NOT APPLE)

target_include_directories(IFXExporting PRIVATE ${IFXExporting_Dirs})

if(U3D_INSTALL_LIBS)
	install(
		TARGETS IFXExporting
		RUNTIME DESTINATION ${BIN_DESTINATION}
		ARCHIVE DESTINATION ${LIB_DESTINATION}
		LIBRARY DESTINATION ${PLUGIN_DESTINATION}
	)
endif()
