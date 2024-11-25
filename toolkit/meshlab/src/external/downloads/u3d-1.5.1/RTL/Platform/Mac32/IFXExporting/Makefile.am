INCLUDES = -I../../../Component/Include -I../../../Kernel/Include -I../../../Platform/Include	\
	-I../../../Component/Exporting	\
	-I../../../Dependencies/WildCards

lib_LTLIBRARIES = libIFXExporting.la
libIFXExporting_la_LDFLAGS = -module -avoid-version -export-symbols IFXExporting.sym
libIFXExporting_la_CXXFLAGS = -DU3DPluginsPath=\".\" -DU3DCorePath=\"$(libdir)\" $(AM_CXXFLAGS)
libIFXExporting_la_SOURCES = \
	IFXExportingDllMain.cpp \
	../../../Component/Exporting/CIFXAnimationModifierEncoder.cpp	\
	../../../Component/Exporting/CIFXAuthorCLODEncoderX.cpp		\
	../../../Component/Exporting/CIFXAuthorCLODEncoderX_P.cpp	\
	../../../Component/Exporting/CIFXAuthorCLODEncoderX_S.cpp	\
	../../../Component/Exporting/CIFXAuthorGeomCompiler.cpp		\
	../../../Component/Exporting/CIFXBlockPriorityQueueX.cpp	\
	../../../Component/Exporting/CIFXBlockWriterX.cpp		\
	../../../Component/Exporting/CIFXBoneWeightsModifierEncoder.cpp	\
	../../../Component/Exporting/CIFXCLODModifierEncoder.cpp	\
	../../../Component/Exporting/CIFXDummyModifierEncoder.cpp	\
	../../../Component/Exporting/CIFXFileReferenceEncoder.cpp	\
	../../../Component/Exporting/CIFXGlyphModifierEncoder.cpp	\
	../../../Component/Exporting/CIFXGroupNodeEncoder.cpp		\
	../../../Component/Exporting/CIFXLightNodeEncoder.cpp		\
	../../../Component/Exporting/CIFXLightResourceEncoder.cpp	\
	../../../Component/Exporting/CIFXLineSetEncoder.cpp		\
	../../../Component/Exporting/CIFXMaterialResourceEncoder.cpp	\
	../../../Component/Exporting/CIFXModelNodeEncoder.cpp		\
	../../../Component/Exporting/CIFXMotionResourceEncoder.cpp	\
	../../../Component/Exporting/CIFXNodeBaseEncoder.cpp		\
	../../../Component/Exporting/CIFXPointSetEncoder.cpp		\
	../../../Component/Exporting/CIFXShaderLitTextureEncoder.cpp	\
	../../../Component/Exporting/CIFXShadingModifierEncoder.cpp	\
	../../../Component/Exporting/CIFXStdioWriteBufferX.cpp		\
	../../../Component/Exporting/CIFXSubdivisionModifierEncoder.cpp	\
	../../../Component/Exporting/CIFXViewNodeEncoder.cpp		\
	../../../Component/Exporting/CIFXViewResourceEncoder.cpp	\
	../../../Component/Exporting/CIFXWriteManager.cpp		\
	../../../Component/Exporting/IFXExporting.cpp			\
	../../../Component/Exporting/IFXExportingGuids.cpp		\
	../../../IFXCorePluginStatic/IFXCorePluginStatic.cpp		\
	../Common/IFXOSUtilities.cpp		\
	../Common/IFXOSLoader.cpp		\
	../../../Component/Base/IFXVertexMap.cpp		\
	../../../Kernel/DataTypes/IFXCoreArray.cpp	\
	../../../Kernel/DataTypes/IFXCoreList.cpp	\
	../../../Kernel/DataTypes/IFXFastAllocator.cpp	\
	../../../Kernel/DataTypes/IFXListNode.cpp	\
	../../../Kernel/DataTypes/IFXMatrix4x4.cpp	\
	../../../Kernel/DataTypes/IFXQuaternion.cpp	\
	../../../Kernel/DataTypes/IFXString.cpp		\
	../../../Kernel/DataTypes/IFXUnitAllocator.cpp	\
	../../../Kernel/DataTypes/IFXVector3.cpp	\
	../../../Kernel/DataTypes/IFXVector4.cpp	\
	../../../Dependencies/WildCards/wcmatch.cpp \
	../../../Kernel/Common/IFXDebug.cpp
libIFXExporting_la_LIBTOOLFLAGS = --tag=disable-static
