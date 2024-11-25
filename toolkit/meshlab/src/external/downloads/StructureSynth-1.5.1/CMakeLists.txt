cmake_minimum_required(VERSION 3.13)
project(StructureSynth)

### Dependencies
message(STATUS "Searching for required components")
find_package(OpenGL REQUIRED)

find_package(Qt5
	COMPONENTS Core Xml OpenGL
	REQUIRED)

### Build settings
set(CMAKE_CXX_STANDARD 11)

set(HEADERS
	StructureSynth/Model/Action.h
	StructureSynth/Model/AmbiguousRule.h
	StructureSynth/Model/Builder.h
	StructureSynth/Model/ColorPool.h
	StructureSynth/Model/CustomRule.h
	StructureSynth/Model/ExecutionStack.h
	StructureSynth/Model/PrimitiveClass.h
	StructureSynth/Model/PrimitiveRule.h
	StructureSynth/Model/RandomStreams.h
	StructureSynth/Model/Rule.h
	StructureSynth/Model/RuleRef.h
	StructureSynth/Model/RuleSet.h
	StructureSynth/Model/State.h
	StructureSynth/Model/Transformation.h
	StructureSynth/Model/TransformationLoop.h
	StructureSynth/Parser/EisenParser.h
	StructureSynth/Parser/Preprocessor.h
	StructureSynth/Parser/Tokenizer.h
	SyntopiaCore/Exceptions/Exception.h
	SyntopiaCore/GLEngine/Box.h
	SyntopiaCore/GLEngine/Dot.h
	SyntopiaCore/GLEngine/EngineWidget.h
	SyntopiaCore/GLEngine/Grid.h
	SyntopiaCore/GLEngine/Line.h
	SyntopiaCore/GLEngine/Mesh.h
	SyntopiaCore/GLEngine/Object3D.h
	SyntopiaCore/GLEngine/Raytracer/AtomicCounter.h
	SyntopiaCore/GLEngine/Raytracer/RayTracer.h
	SyntopiaCore/GLEngine/Raytracer/RenderThread.h
	SyntopiaCore/GLEngine/Raytracer/Sampler.h
	SyntopiaCore/GLEngine/Raytracer/VoxelStepper.h
	SyntopiaCore/GLEngine/RaytraceTriangle.h
	SyntopiaCore/GLEngine/Sphere.h
	SyntopiaCore/GLEngine/Triangle.h
	SyntopiaCore/Logging/ListWidgetLogger.h
	SyntopiaCore/Logging/Logging.h
	SyntopiaCore/Math/Matrix4.h
	SyntopiaCore/Math/Random.h
	SyntopiaCore/Math/Vector3.h
	SyntopiaCore/Misc/ColorUtils.h
	SyntopiaCore/Misc/MiniParser.h
	SyntopiaCore/Misc/Persistence.h
	SyntopiaCore/Misc/Version.h
	ThirdPartyCode/MersenneTwister/MersenneTwister.h
	StructureSynth/Model/Rendering/OpenGLRenderer.h
	StructureSynth/Model/Rendering/Renderer.h
	StructureSynth/Model/Rendering/TemplateRenderer.h
)

set(SOURCES
	StructureSynth/Model/Action.cpp
	StructureSynth/Model/AmbiguousRule.cpp
	StructureSynth/Model/Builder.cpp
	StructureSynth/Model/ColorPool.cpp
	StructureSynth/Model/CustomRule.cpp
	StructureSynth/Model/ExecutionStack.cpp
	StructureSynth/Model/PrimitiveClass.cpp
	StructureSynth/Model/PrimitiveRule.cpp
	StructureSynth/Model/RandomStreams.cpp
	StructureSynth/Model/Rule.cpp
	StructureSynth/Model/RuleRef.cpp
	StructureSynth/Model/RuleSet.cpp
	StructureSynth/Model/State.cpp
	StructureSynth/Model/Transformation.cpp
	StructureSynth/Model/TransformationLoop.cpp
	StructureSynth/Parser/EisenParser.cpp
	StructureSynth/Parser/Preprocessor.cpp
	StructureSynth/Parser/Tokenizer.cpp
	SyntopiaCore/GLEngine/Box.cpp
	SyntopiaCore/GLEngine/Dot.cpp
	SyntopiaCore/GLEngine/Grid.cpp
	SyntopiaCore/GLEngine/Line.cpp
	SyntopiaCore/GLEngine/Mesh.cpp
	SyntopiaCore/GLEngine/Object3D.cpp
	SyntopiaCore/GLEngine/Raytracer/AtomicCounter.cpp
	SyntopiaCore/GLEngine/Raytracer/RenderThread.cpp
	SyntopiaCore/GLEngine/Raytracer/Sampler.cpp
	SyntopiaCore/GLEngine/Raytracer/VoxelStepper.cpp
	SyntopiaCore/GLEngine/RaytraceTriangle.cpp
	SyntopiaCore/GLEngine/Triangle.cpp
	SyntopiaCore/Logging/ListWidgetLogger.cpp
	SyntopiaCore/Logging/Logging.cpp
	SyntopiaCore/Math/Matrix4.cpp
	SyntopiaCore/Math/Random.cpp
	SyntopiaCore/Math/Vector3.cpp
	SyntopiaCore/Misc/ColorUtils.cpp
	SyntopiaCore/Misc/MiniParser.cpp
	SyntopiaCore/Misc/Persistence.cpp
	SyntopiaCore/Misc/Version.cpp
	StructureSynth/Model/Rendering/Renderer.cpp
	StructureSynth/Model/Rendering/TemplateRenderer.cpp
)

add_library(structure-synth STATIC
	${SOURCES} ${HEADERS})

target_link_libraries(structure-synth PUBLIC
	Qt5::Core Qt5::Xml Qt5::OpenGL OpenGL::GLU)

target_include_directories(structure-synth SYSTEM PUBLIC
	"${CMAKE_CURRENT_LIST_DIR}")

set_property(TARGET structure-synth
	PROPERTY AUTOMOC TRUE)
