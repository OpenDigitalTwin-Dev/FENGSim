INCLUDES = -I../../../Component/Include -I../../../Kernel/Include -I../../../Platform/Include	\
	-I../../../Component/Base	\
	-I../../../Component/Rendering	\
	-I../../../Dependencies/WildCards

lib_LTLIBRARIES = libIFXCoreStatic.la
libIFXCoreStatic_la_LDFLAGS = -static
libIFXCoreStatic_la_CXXFLAGS = -DU3DPluginsPath=\".\" -DU3DCorePath=\"$(libdir)\" $(AM_CXXFLAGS)
libIFXCoreStatic_la_SOURCES = \
	../../../IFXCoreStatic/IFXCoreStatic.cpp \
	../Common/IFXOSUtilities.cpp		\
	../Common/IFXOSLoader.cpp		\
	../Common/IFXOSRenderWindow.cpp		\
	../../../Component/Common/IFXDids.cpp		\
	../../../Component/Base/IFXCoincidentVertexMap.cpp	\
	../../../Component/Base/IFXCornerIter.cpp		\
	../../../Component/Base/IFXEuler.cpp			\
	../../../Component/Base/IFXFatCornerIter.cpp		\
	../../../Component/Base/IFXTransform.cpp		\
	../../../Component/Base/IFXVectorHasher.cpp		\
	../../../Component/Base/IFXVertexMap.cpp		\
	../../../Component/Base/IFXVertexMapGroup.cpp		\
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
