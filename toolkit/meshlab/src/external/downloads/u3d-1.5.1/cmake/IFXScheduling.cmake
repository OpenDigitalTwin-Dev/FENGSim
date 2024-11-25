# IFXScheduling
set(IFXScheduling_Dirs
	${U3D_DIR}/RTL/Component/Include
	${U3D_DIR}/RTL/Kernel/Include
	${U3D_DIR}/RTL/Platform/Include
	${U3D_DIR}/RTL/Component/ModifierChain
	${U3D_DIR}/RTL/Component/SceneGraph
	${U3D_DIR}/RTL/Component/Scheduling
	${U3D_DIR}/RTL/Dependencies/WildCards
)

SET(IFXScheduling_HDRS
	${Component_HDRS}
	${Kernel_HDRS}
	${Platform_HDRS}
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXDidRegistry.h
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXModifier.h
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXModifierChain.h
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXModifierDataElementIter.h
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXModifierDataPacket.h
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXObserverStateTree.h
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXSubject.h
	${U3D_DIR}/RTL/Component/ModifierChain/CRedBlackTree.h
	${U3D_DIR}/RTL/Component/ModifierChain/IFXModifierChainInternal.h
	${U3D_DIR}/RTL/Component/ModifierChain/IFXModifierChainState.h
	${U3D_DIR}/RTL/Component/ModifierChain/IFXModifierDataPacketInternal.h
	${U3D_DIR}/RTL/Component/ModifierChain/IFXSet.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXBoundSphereDataElement.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXDevice.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXDummyModifier.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXFileReference.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXGroup.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXLight.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXLightResource.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXLightSet.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXMarker.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXMaterialResource.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXMixerConstruct.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXModel.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXMotionResource.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXNode.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXResourceClient.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXSceneGraph.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXShaderList.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXSimpleCollection.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXSimpleList.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXView.h
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXViewResource.h
	${U3D_DIR}/RTL/Component/SceneGraph/IFXSceneGraphPCH.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXClock.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXErrorInfo.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXNotificationInfo.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXNotificationManager.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXScheduler.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSchedulerInfo.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSimulationInfo.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSimulationManager.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSystemManager.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskCallback.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskData.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskManager.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskManagerNode.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskManagerView.h
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTimeManager.h
	${U3D_DIR}/RTL/Dependencies/WildCards/wcmatch.h
)

SET(IFXScheduling_SRCS
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/IFXScheduling/IFXSchedulingDllMain.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXClock.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXErrorInfo.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXNotificationInfo.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXNotificationManager.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXScheduler.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSchedulerInfo.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSimulationInfo.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSimulationManager.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXSystemManager.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskCallback.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskData.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskManager.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskManagerNode.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTaskManagerView.cpp
	${U3D_DIR}/RTL/Component/Scheduling/CIFXTimeManager.cpp
	${U3D_DIR}/RTL/Component/Scheduling/IFXScheduling.cpp
	${U3D_DIR}/RTL/Component/Scheduling/IFXSchedulingGuids.cpp
	${U3D_DIR}/RTL/IFXCorePluginStatic/IFXCorePluginStatic.cpp
	${U3D_DIR}/RTL/Platform/${U3D_PLATFORM}/Common/IFXOSUtilities.cpp
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXSubject.cpp
	${U3D_DIR}/RTL/Component/ModifierChain/CIFXModifier.cpp
	${U3D_DIR}/RTL/Component/SceneGraph/CIFXMarker.cpp
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
	SET(SCHED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Win32/IFXScheduling)
	ADD_LIBRARY(IFXScheduling SHARED
		${IFXScheduling_SRCS} ${IFXScheduling_HDRS} ${SCHED_DIR}/IFXScheduling.rc ${SCHED_DIR}/IFXResource.h ${SCHED_DIR}/IFXScheduling.def)
	TARGET_LINK_LIBRARIES(IFXScheduling IFXCore)
ENDIF(WIN32)
IF(APPLE)
	ADD_LIBRARY(IFXScheduling MODULE
		${IFXScheduling_SRCS} ${IFXScheduling_HDRS})
	set_target_properties(IFXScheduling PROPERTIES
		LINK_FLAGS "${MY_LINK_FLAGS} -exported_symbols_list ${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Mac32/IFXScheduling/IFXScheduling.def -undefined dynamic_lookup")
ENDIF(APPLE)
IF(UNIX AND NOT APPLE)
	ADD_LIBRARY(IFXScheduling  MODULE
		${IFXScheduling_SRCS} ${IFXScheduling_HDRS})
	set_target_properties(IFXScheduling  PROPERTIES
		LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/RTL/Platform/Lin32/IFXScheduling/IFXScheduling.list")
ENDIF(UNIX AND NOT APPLE)

target_include_directories(IFXScheduling PRIVATE ${IFXScheduling_Dirs})

if(U3D_INSTALL_LIBS)
	install(
		TARGETS IFXScheduling
		RUNTIME DESTINATION ${BIN_DESTINATION}
		ARCHIVE DESTINATION ${LIB_DESTINATION}
		LIBRARY DESTINATION ${PLUGIN_DESTINATION}
	)
endif()
