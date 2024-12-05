cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(m++_api)

# >>> if use /usr/bin/mpicxx.openmpi don't need to inlcude
# >>> "/usr/include/x86_64-linux-gnu/openmpi/" as the following

#set(CMAKE_CXX_COMPILER /usr/bin/mpicxx.openmpi)

# >>> g++-12 and g++-13 have problems
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

find_package(MPI)
if(MPI_FOUND)
  message("MPI FOUND")
  message(${MPI_LIBRARIES})
  include_directories(${MPI_INCLUDE_DIRS})
  link_directories(${MPI_LIBRARY_DIRS})
  add_definitions(${MPI_DEFINITIONS})
endif()

include_directories("/usr/include/x86_64-linux-gnu/openmpi/")

include_directories("src")
aux_source_directory(solver/src solver_DIR_SRCS)
add_executable (ElasticitySolver ${solver_DIR_SRCS} src/Identify.C src/MatrixGraph.C src/Algebra.C src/Shape.C src/Point.C src/Parallel.C src/Time.C src/ctools.C src/IO.C src/Mesh.C src/Cell.C src/Distribution.C src/Plot.C src/Quadrature.C src/DoF.C src/Sparse.C src/Interface.C src/Discretization.C src/LinearSolver.C src/DD.C src/Preconditioner.C src/TimeSeries.C src/ESolver.C src/Transfer.C src/Element.C src/FaceElement.C src/Newton.C src/Newmark.C src/Euler.C src/Small.C src/Schur.C)
target_link_libraries (ElasticitySolver /usr/lib/x86_64-linux-gnu/libblas.so /usr/lib/x86_64-linux-gnu/liblapack.so ${MPI_LIBRARIES} /usr/lib/x86_64-linux-gnu/libsuperlu.so)
set_target_properties(ElasticitySolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

file(COPY "log" DESTINATION ${CMAKE_BINARY_DIR})
file(COPY "data/vtk" DESTINATION ${CMAKE_BINARY_DIR}/data)
file(COPY "conf" DESTINATION ${CMAKE_BINARY_DIR})
file(COPY "solver/conf" DESTINATION ${CMAKE_BINARY_DIR}/solver)

