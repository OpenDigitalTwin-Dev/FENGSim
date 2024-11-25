# HelloWorld
ADD_EXECUTABLE( HelloU3DWorld
	Samples/SampleCode/HelloWorld.cpp
	${Component_HDRS} ${Kernel_HDRS} ${Platform_HDRS} )
target_link_libraries( HelloU3DWorld  IFXCoreStatic ${CMAKE_DL_LIBS} )
add_dependencies( HelloU3DWorld  IFXCoreStatic )

install(
	TARGETS HelloU3DWorld
	DESTINATION ${BIN_DESTINATION}
)
