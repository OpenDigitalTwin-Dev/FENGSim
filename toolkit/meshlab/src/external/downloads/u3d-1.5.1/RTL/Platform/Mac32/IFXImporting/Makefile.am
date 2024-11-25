INCLUDES = -I../../../Component/Include -I../../../Kernel/Include -I../../../Platform/Include	\
	-I../../../Component/Importing	\
	-I../../../Dependencies/WildCards

lib_LTLIBRARIES = libIFXImporting.la
libIFXImporting_la_LDFLAGS = -module -avoid-version -export-symbols IFXImporting.sym
libIFXImporting_la_CXXFLAGS = -DU3DPluginsPath=\".\" -DU3DCorePath=\"$(libdir)\" $(AM_CXXFLAGS)
libIFXImporting_la_SOURCES = \
	IFXImportingDllMain.cpp	\
	../../../Component/Importing/CIFXAnimationModifierDecoder.cpp	\
	../../../Component/Importing/CIFXAuthorCLODDecoder.cpp		\
	../../../Component/Importing/CIFXAuthorCLODDecoder_P.cpp	\
	../../../Component/Importing/CIFXAuthorCLODDecoder_S.cpp	\
	../../../Component/Importing/CIFXBlockReaderX.cpp		\
	../../../Component/Importing/CIFXBoneWeightsModifierDecoder.cpp	\
	../../../Component/Importing/CIFXBTTHash.cpp			\
	../../../Component/Importing/CIFXCLODModifierDecoder.cpp	\
	../../../Component/Importing/CIFXDecoderChainX.cpp		\
	../../../Component/Importing/CIFXDummyModifierDecoder.cpp	\
	../../../Component/Importing/CIFXGlyphModifierDecoder.cpp	\
	../../../Component/Importing/CIFXGroupDecoder.cpp		\
	../../../Component/Importing/CIFXInternetReadBufferX.cpp	\
	../../../Component/Importing/CIFXLightDecoder.cpp		\
	../../../Component/Importing/CIFXLightResourceDecoder.cpp	\
	../../../Component/Importing/CIFXLineSetDecoder.cpp		\
	../../../Component/Importing/CIFXLoadManager.cpp		\
	../../../Component/Importing/CIFXMaterialDecoder.cpp		\
	../../../Component/Importing/CIFXModelDecoder.cpp		\
	../../../Component/Importing/CIFXMotionDecoder.cpp		\
	../../../Component/Importing/CIFXNodeDecoder.cpp		\
	../../../Component/Importing/CIFXPointSetDecoder.cpp		\
	../../../Component/Importing/CIFXShaderLitTextureDecoder.cpp	\
	../../../Component/Importing/CIFXShadingModifierDecoder.cpp	\
	../../../Component/Importing/CIFXStdioReadBufferX.cpp		\
	../../../Component/Importing/CIFXSubdivisionModifierDecoder.cpp	\
	../../../Component/Importing/CIFXTextureDecoder.cpp		\
	../../../Component/Importing/CIFXViewDecoder.cpp		\
	../../../Component/Importing/CIFXViewResourceDecoder.cpp	\
	../../../Component/Importing/IFXImporting.cpp			\
	../../../Component/Importing/IFXImportingGuids.cpp		\
	../../../Component/Importing/IFXInternetConnectionX.cpp		\
	../../../Component/Importing/IFXInternetSessionX.cpp		\
	../../../Component/Importing/IFXModifierBaseDecoder.cpp		\
	../../../Component/Importing/IFXSocketStream.cpp		\
	../../../IFXCorePluginStatic/IFXCorePluginStatic.cpp		\
	../Common/IFXOSUtilities.cpp				\
	../Common/IFXOSLoader.cpp				\
	../Common/IFXOSSocket.cpp				\
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
libIFXImporting_la_LIBTOOLFLAGS = --tag=disable-static
