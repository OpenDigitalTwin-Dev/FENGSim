# IDTFGen
ADD_EXECUTABLE( IDTFGen Samples/SampleCode/IDTFGen.cpp ${libIDTF_HDRS} )
target_link_libraries( IDTFGen  IDTF ${CMAKE_DL_LIBS} )
add_dependencies( IDTFGen  IDTF )

install(
	TARGETS IDTFGen
	DESTINATION ${BIN_DESTINATION}
)
