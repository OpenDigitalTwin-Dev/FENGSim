# IFXCoreStatic
set(IFXCoreStatic_Dirs
	${U3D_DIR}/RTL/Component/Include
	${U3D_DIR}/RTL/Kernel/Include
	${U3D_DIR}/RTL/Platform/Include
	${U3D_DIR}/RTL/Component/Base
	${U3D_DIR}/RTL/Component/Rendering
	${U3D_DIR}/RTL/Dependencies/WildCards
)

SET(IFXCoreStatic_HDRS
	${Component_HDRS}
	${Kernel_HDRS}
	${Platform_HDRS}
	${U3D_DIR}/RTL/Component/Base/IFXVectorHasher.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXDeviceBase.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXDeviceLight.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXDeviceTexture.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXDeviceTexUnit.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXRender.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXRenderContext.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXRenderDevice.h
	${U3D_DIR}/RTL/Component/Rendering/CIFXRenderServices.h
	${U3D_DIR}/RTL/Component/Rendering/DX7/CIFXDeviceLightDX7.h
	${U3D_DIR}/RTL/Component/Rendering/DX7/CIFXDeviceTextureDX7.h
	${U3D_DIR}/RTL/Component/Rendering/DX7/CIFXDeviceTexUnitDX7.h
	${U3D_DIR}/RTL/Component/Rendering/DX7/CIFXDirectX7.h
	${U3D_DIR}/RTL/Component/Rendering/DX7/CIFXRenderDeviceDX7.h
	${U3D_DIR}/RTL/Component/Rendering/DX7/CIFXRenderDX7.h
	${U3D_DIR}/RTL/Component/Rendering/DX7/IFXRenderPCHDX7.h
	${U3D_DIR}/RTL/Component/Rendering/DX8/CIFXDeviceLightDX8.h
	${U3D_DIR}/RTL/Component/Rendering/DX8/CIFXDeviceTextureDX8.h
	${U3D_DIR}/RTL/Component/Rendering/DX8/CIFXDeviceTexUnitDX8.h
	${U3D_DIR}/RTL/Component/Rendering/DX8/CIFXDirectX8.h
	${U3D_DIR}/RTL/Component/Rendering/DX8/CIFXRenderDeviceDX8.h
	${U3D_DIR}/RTL/Component/Rendering/DX8/CIFXRenderDX8.h
	${U3D_DIR}/RTL/Component/Rendering/DX8/IFXRenderPCHDX8.h
	${U3D_DIR}/RTL/Component/Rendering/IFXAAFilter.h
	${U3D_DIR}/RTL/Component/Rendering/IFXRenderPCH.h
	${U3D_DIR}/RTL/Component/Rendering/Null/CIFXDeviceLightNULL.h
	${U3D_DIR}/RTL/Component/Rendering/Null/CIFXDeviceTextureNULL.h
	${U3D_DIR}/RTL/Component/Rendering/Null/CIFXDeviceTexUnitNULL.h
	${U3D_DIR}/RTL/Component/Rendering/Null/CIFXRenderDeviceNULL.h
	${U3D_DIR}/RTL/Component/Rendering/Null/CIFXRenderNULL.h
	${U3D_DIR}/RTL/Component/Rendering/Null/IFXRenderPCHNULL.h
	${U3D_DIR}/RTL/Component/Rendering/OpenGL/CIFXDeviceLightOGL.h
	${U3D_DIR}/RTL/Component/Rendering/OpenGL/CIFXDeviceTextureOGL.h
	${U3D_DIR}/RTL/Component/Rendering/OpenGL/CIFXDeviceTexUnitOGL.h
	${U3D_DIR}/RTL/Component/Rendering/OpenGL/CIFXOpenGL.h
	${U3D_DIR}/RTL/Component/Rendering/OpenGL/CIFXRenderDeviceOGL.h
	${U3D_DIR}/RTL/Component/Rendering/OpenGL/CIFXRenderOGL.h
	${U3D_DIR}/RTL/Component/Rendering/OpenGL/IFXRenderPCHOGL.h
	${U3D_DIR}/RTL/Dependencies/WildCards/wcmatch.h
)

SET(IFXCoreStatic_SRCS
	${U3D_DIR}/RTL/IFXCoreStatic/IFXCoreStatic.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSUtilities.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSLoader.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSRenderWindow.cpp
	${U3D_DIR}/RTL/Component/Common/IFXDids.cpp
	${U3D_DIR}/RTL/Component/Base/IFXCoincidentVertexMap.cpp
	${U3D_DIR}/RTL/Component/Base/IFXCornerIter.cpp
	${U3D_DIR}/RTL/Component/Base/IFXEuler.cpp
	${U3D_DIR}/RTL/Component/Base/IFXFatCornerIter.cpp
	${U3D_DIR}/RTL/Component/Base/IFXTransform.cpp
	${U3D_DIR}/RTL/Component/Base/IFXVectorHasher.cpp
	${U3D_DIR}/RTL/Component/Base/IFXVertexMap.cpp
	${U3D_DIR}/RTL/Component/Base/IFXVertexMapGroup.cpp
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

add_library(IFXCoreStatic STATIC ${IFXCoreStatic_SRCS} ${IFXCoreStatic_HDRS})
target_include_directories(IFXCoreStatic PRIVATE ${IFXCoreStatic_Dirs})

if(U3D_INSTALL_LIBS)
	install(
		TARGETS IFXCoreStatic
		ARCHIVE DESTINATION ${LIB_DESTINATION}
		LIBRARY DESTINATION ${LIB_DESTINATION}
	)
endif()
