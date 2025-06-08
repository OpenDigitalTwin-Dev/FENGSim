cmake_minimum_required(VERSION 3.10)
project(ConcentricInfill)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 设置Clipper2安装路径

list(APPEND CMAKE_PREFIX_PATH 
    "../../toolkit/Geometry/install/clipper2_install/lib/cmake/clipper2"
    "../../toolkit/Geometry/install/clipper2_install/lib/cmake/clipper2utils"
)
#set(CMAKE_PREFIX_PATH "../../toolkit/Geometry/install/clipper2_install/lib/cmake/clipper2")

#find_package(Clipper2 REQUIRED)
# if(Clipper2_FOUND)
#     message(STATUS "Includes: ${Clipper2_INCLUDE_DIRS}")
#     message(STATUS "Libraries: ${Clipper2_LIBRARIES}")
# endif()


set(CLIPPER2_UTILS_INC
    src/Utils/clipper.svg.h
    src/Utils/ClipFileLoad.h
    src/Utils/ClipFileSave.h
    src/Utils/Timer.h
    src/Utils/Colors.h
    src/Utils/CommonUtils.h
  )

set(CLIPPER2_UTILS_SRC
    src/Utils/clipper.svg.cpp
    src/Utils/ClipFileLoad.cpp
    src/Utils/ClipFileSave.cpp
  ) 
   

find_package(Clipper2 REQUIRED)
if(Clipper2_FOUND)
   message(STATUS "Clipper2 found. Version: ${Clipper2_VERSION}")
   message(STATUS "Clipper2_DIR" ${Clipper2_DIR})
   message(STATUS "Clipper2_INCLUDE_DIRS" ${Clipper2_INCLUDE_DIRS})
   message(STATUS "Clipper2_LIBRARY_DIRS" ${Clipper2_LIBRARY_DIRS})
   message(STATUS "Clipper2_LIBRARIES" ${Clipper2_LIBRARIES})
   #message(STATUS "Include dirs: ${Clipper2_INCLUDE_DIRS}")

   #message(STATUS "Library paths: ${Clipper2_LIBRARIES}")
else()
   message(FATAL_ERROR "Clipper2 not found!")
endif()


find_package(Clipper2Utils REQUIRED)
if(Clipper2Utils_FOUND)
   message(STATUS "Clipper2Utils_INCLUDE_DIRS" ${Clipper2Utils_INCLUDE_DIRS})
   message(STATUS "Clipper2Utils_LIBRARY_DIRS" ${Clipper2Utils_LIBRARY_DIRS})
   message(STATUS "Clipper2Utils_LIBRARIES" ${Clipper2Utils_LIBRARIES})
   #message(STATUS "Includedirs: ${Clipper2Utils_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Clipper2Utils not found!")
endif()


#include_directories(${Clipper2_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/Utils)
#link_directories(${Clipper2_LIBRARIES})
include_directories(Utils)

add_executable(ConcentricInfill src/ConcentricInfill.cpp ${CLIPPER2_UTILS_INC} ${CLIPPER2_UTILS_SRC} )
target_link_libraries(ConcentricInfill PRIVATE Clipper2::Clipper2)
