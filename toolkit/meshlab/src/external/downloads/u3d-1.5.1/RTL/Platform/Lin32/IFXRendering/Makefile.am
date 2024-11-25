INCLUDES = -I../../../Component/Include -I../../../Kernel/Include -I../../../Platform/Include	\
	-I../../../Component/ModifierChain	\
	-I../../../Component/Culling		\
	-I../../../Component/Rendering		\
	-I../../../Component/Rendering/Null	\
	-I../../../Component/Rendering/OpenGL	\
	-IOpenGL				\
	-I../../../Dependencies/WildCards

lib_LTLIBRARIES = libIFXRendering.la
libIFXRendering_la_LDFLAGS = -module -avoid-version -export-symbols IFXRendering.sym
libIFXRendering_la_SOURCES = \
	IFXRenderingDllMain.cpp \
	../../../Component/Culling/CIFXOctreeCollection.cpp		\
	../../../Component/Culling/CIFXOctreeCuller.cpp			\
	../../../Component/Culling/CIFXScenegraphCuller.cpp		\
	../../../Component/Rendering/CIFXDeviceLight.cpp	\
	../../../Component/Rendering/CIFXDeviceTexture.cpp	\
	../../../Component/Rendering/CIFXDeviceTexUnit.cpp	\
	../../../Component/Rendering/CIFXRenderContext.cpp	\
	../../../Component/Rendering/CIFXRender.cpp		\
	../../../Component/Rendering/CIFXRenderDevice.cpp	\
	../../../Component/Rendering/CIFXRenderServices.cpp	\
	../../../Component/Rendering/IFXAAFilter.cpp		\
	../../../Component/Rendering/IFXRendering.cpp		\
	../../../Component/Rendering/IFXRenderingGuids.cpp	\
	../../../Component/Rendering/Null/CIFXDeviceLightNULL.cpp	\
	../../../Component/Rendering/Null/CIFXDeviceTextureNULL.cpp	\
	../../../Component/Rendering/Null/CIFXDeviceTexUnitNULL.cpp	\
	../../../Component/Rendering/Null/CIFXRenderDeviceNULL.cpp	\
	../../../Component/Rendering/Null/CIFXRenderNULL.cpp		\
	../../../Component/Rendering/OpenGL/CIFXDeviceLightOGL.cpp	\
	../../../Component/Rendering/OpenGL/CIFXDeviceTextureOGL.cpp	\
	../../../Component/Rendering/OpenGL/CIFXDeviceTexUnitOGL.cpp	\
	../../../Component/Rendering/OpenGL/CIFXOpenGL.cpp		\
	../../../Component/Rendering/OpenGL/CIFXRenderDeviceOGL.cpp	\
	../../../Component/Rendering/OpenGL/CIFXRenderOGL.cpp		\
	../../../Component/Rendering/OpenGL/IFXOpenGL.cpp		\
	OpenGL/CIFXOpenGLOS.cpp		\
	OpenGL/CIFXRenderOGLOS.cpp	\
	../../../IFXCorePluginStatic/IFXCorePluginStatic.cpp		\
	../../../Component/ModifierChain/CIFXSubject.cpp			\
	../Common/IFXOSUtilities.cpp		\
	../Common/IFXOSRender.cpp		\
	../Common/IFXOSRenderWindow.cpp		\
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
libIFXRendering_la_LIBTOOLFLAGS = --tag=disable-static
