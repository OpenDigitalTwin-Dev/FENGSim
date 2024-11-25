INCLUDES = -I../../../Component/Include -I../../../Kernel/Include -I../../../Platform/Include	\
	-I../../../Component/ModifierChain	\
	-I../../../Component/SceneGraph		\
	-I../../../Component/Scheduling		\
	-I../../../Dependencies/WildCards

lib_LTLIBRARIES = libIFXScheduling.la
libIFXScheduling_la_LDFLAGS = -module -avoid-version -export-symbols IFXScheduling.sym
libIFXScheduling_la_SOURCES = \
	IFXSchedulingDllMain.cpp	\
	../../../Component/Scheduling/CIFXClock.cpp			\
	../../../Component/Scheduling/CIFXErrorInfo.cpp			\
	../../../Component/Scheduling/CIFXNotificationInfo.cpp		\
	../../../Component/Scheduling/CIFXNotificationManager.cpp	\
	../../../Component/Scheduling/CIFXScheduler.cpp			\
	../../../Component/Scheduling/CIFXSchedulerInfo.cpp		\
	../../../Component/Scheduling/CIFXSimulationInfo.cpp		\
	../../../Component/Scheduling/CIFXSimulationManager.cpp		\
	../../../Component/Scheduling/CIFXSystemManager.cpp		\
	../../../Component/Scheduling/CIFXTaskCallback.cpp		\
	../../../Component/Scheduling/CIFXTaskData.cpp			\
	../../../Component/Scheduling/CIFXTaskManager.cpp		\
	../../../Component/Scheduling/CIFXTaskManagerNode.cpp		\
	../../../Component/Scheduling/CIFXTaskManagerView.cpp		\
	../../../Component/Scheduling/CIFXTimeManager.cpp		\
	../../../Component/Scheduling/IFXScheduling.cpp			\
	../../../Component/Scheduling/IFXSchedulingGuids.cpp		\
	../../../IFXCorePluginStatic/IFXCorePluginStatic.cpp	\
	../Common/IFXOSUtilities.cpp	\
	../../../Component/ModifierChain/CIFXSubject.cpp			\
	../../../Component/ModifierChain/CIFXModifier.cpp			\
	../../../Component/SceneGraph/CIFXMarker.cpp				\
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
#	../../../Component/Base/IFXCoincidentVertexMap.cpp	
#	../../../Component/Base/IFXCornerIter.cpp		
#	../../../Component/Base/IFXEuler.cpp			
#	../../../Component/Base/IFXFatCornerIter.cpp		
#	../../../Component/Base/IFXTransform.cpp		
#	../../../Component/Base/IFXVectorHasher.cpp		
#	../../../Component/Base/IFXVertexMap.cpp		
#	../../../Component/Base/IFXVertexMapGroup.cpp		
libIFXScheduling_la_LIBTOOLFLAGS = --tag=disable-static
