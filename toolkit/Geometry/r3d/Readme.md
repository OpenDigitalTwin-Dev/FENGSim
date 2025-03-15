# `r3d`

Routines for fast, geometrically robust clipping operations and analytic volume/moment computations 
over polytopes in 2D and 3D (as well as experimental ND). This software forms the kernel for an exact 
general remeshing scheme. Also includes physically conservative voxelization 
(volume sampling) of polytopes to a Cartesian grid in 2D and 3D.

As described in 
[Powell & Abel (2015)](http://www.sciencedirect.com/science/article/pii/S0021999115003563) and
[LA-UR-15-26964](la-ur-15-26964.pdf). For information about the API itself, see
[LA-UR-15-26964](la-ur-15-26964.pdf). The now-deprecated version of the code used for 
[Powell & Abel (2015)](http://www.sciencedirect.com/science/article/pii/S0021999115003563) 
lives in `deprecated`.

---

### Features:

- Robustly clip polytopes against planes.

- Compute volumes and moments over polytopes using the optimal recursive method of
[Koehl (2012)](https://www.computer.org/csdl/trans/tp/2012/11/ttp2012112158.pdf).

- Voxelize 2D and 3D polytopes onto a Cartesian grid by calculating the exact coordinate moments
  of the intersections between the polytope and each underlying grid cell.

- Utility functions for orientation tests, box initialization, conversion between polyhedral
  representations, and more.

- A set of rigorous unit-tests, located in `src/tests`.
These tests also serve as examples of how to use `r3d`. 

- All declarations and documentation are located in `r3d.h`, `v3d.h`, `r2d.h`, and `v2d.h`.

- For computational efficiency, R3D use a statically size array to
  store vertices of 3D polyhedra. This defaults to 512 but can be
  expanded or contracted by a CMake command line specification (`cmake
  -DR3D_MAX_VERTS=N ..`). This autogenerates a config file
  `r3d-config.h` at build time which is included in `r3d.h`. (Note: This makes it impossible to use a simple Makefile to compile R3D)

- To improve accuracy of moment calculations, polytops can be shifted to the origin by
  setting -DSHIFT_POLY=True in CMake configuration options. This option is particularly
  beneficial for small polytops located far from the origin, but it is computationally
  costly for high-order moments.

---

### Building:

-   Basic build

   `mkdir build`  # make a build directory
 
   `cd build`
 
   `cmake -DENABLE_UNIT_TESTS=ON ..`
 
   `make`
 
   `make test`    # to test if R3D is working correctly


-  CMake configuration options

   Debug build:       `cmake -DCMAKE_BUILD_TYPE=Debug ..`

   Release build:     `cmake -DCMAKE_BUILD_TYPE=Release ..`

   Tests:             `cmake -DENABLE_UNIT_TESTS=[ON|OFF] ..`

   Installation dir:  `cmake -DCMAKE_INSTALL_PREFIX=<r3d_install_dir> ..`

   Set R3D_MAX_VERTS: `cmake -DR3D_MAX_VERTS=N ..` where N can be any number. 

- To link to R3D

  - Using CMake
  
  R3D's build system now writes out relevant CMake configuration
  information in `<r3d_install_dir>/lib/cmake/r3d/r3dConfig.cmake`. It
  also exports a library target called `r3d::r3d` which can be
  specified as a dependency and automatically get the locations of the
  library, the name of the library (even if it is static or shared),
  the locations of the include files etc.
  
  To use this method, call `find_package(r3d)` in the calling app's
  CMake build system and specify `-Dr3d_ROOT=<r3d_install_dir>` in the
  calling app's CMake command line invocation. Then specify `r3d` as a
  dependency like so:
  
  `find_package(r3d)
  target_link_libraries(MYAPP PRIVATE ${r3d_LIBRARIES})`
  
  
  CMake will do the rest to find the includes and link in the right libraries.
  
  
  - Manually

	`#include <r3d.h>`, `#include <r2d.h>`, `#include <v3d.h>` etc. in your code as you require
	
	Point the build system to include path of the install directory
	
	Link to libr3d.a (or libr3d.so) as required (`-lr3d`)


---

### Licensing: 

`r3d.c`, `r3d.h`, `r2d.c`, `r2d.h`, `rNd.c`, `rNd.h`, and contents of `tests` 
Copyright (C) 2015, DOE and Los Alamos National Security, LLC.

`v3d.c`, `v3d.h`, `v2d.c`, `v2d.h`, and contents of `deprecated` Copyright (C) 2015, Stanford University, 
through SLAC National Accelerator Laboratory.

See source file headers for full license text. All code is open-source, subject to terms of the
respective license. We request that you cite 
[Powell & Abel (2015)](http://www.sciencedirect.com/science/article/pii/S0021999115003563) and
[LA-UR-15-26964](la-ur-15-26964.pdf) when using this code for research purposes.


