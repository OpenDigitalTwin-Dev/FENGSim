#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
extern "C" {
#include "r3d.h"
}

int main() {

#if 0
  {
	//////////////////////////////////////////
	// make sure we have a working program
	//////////////////////////////////////////

	std::cout << "hello world" << std::endl;
  }
#endif

#if 0
  {
	//////////////////////////////////////////
	// vectors
	//////////////////////////////////////////

	r3d_rvec3 v = {1., 2., 1.};
	std::cout << "v " << v.x << " " << v.y << " " << v.z << " " << std::endl;

	std::cout << "v indexed " << v.xyz[0] << " " << v.xyz[1] << " " 
		<< v.xyz[2] << " " << std::endl;
  }
#endif

#if 0
  {
	//////////////////////////////////////////
	// planes
	//////////////////////////////////////////

	// define a plane the long way
	r3d_plane plane;
	plane.n = {1., 1., -1.};
	plane.d = 1.;
	std::cout << "plane 1 " << plane.n.xyz[0] << " " << plane.n.xyz[1] 
		<< " " << plane.n.xyz[2] << " " << std::endl;

	// define a plane in one step
	r3d_plane plane2 = {{2., 2., 3.}, 1.};
	std::cout << "plane 2 " << plane2.n.xyz[0] << " " << plane2.n.xyz[1] 
		<< " " << plane2.n.xyz[2] << " " << std::endl;
  }
#endif

#if 0
  {
	//////////////////////////////////////////
	// vertices, It is unlikely we will ever call this directly. It is created
	// automatically by helper functions like r3d_init_tet
	//////////////////////////////////////////

	r3d_vertex vertex = {{0, 1., 2}, {5., 6., 7.}};
	std::cout << "vertex " << vertex.pos.x << " " << vertex.pos.y << " "
		<< vertex.pos.z << " " << std::endl;
  }
#endif

  {
          
	//////////////////////////////////////////
	// tetrahedron, test clip
	//////////////////////////////////////////

	// start with raw vertices
	r3d_rvec3 verts[4] = {
		{0., 0., 0.},
		{1., 0., 0.},
		{0., 1., 0.},
		{0., 0., 1.},
	};

	// turn the raw vertices into a poly
	r3d_poly poly;
	std::cout << std::endl;
	r3d_init_tet(&poly, verts);
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

	// clip the poly
	// the definition of the plane is n.v+d=0, which is why d needs to be negative
	r3d_plane p1 = {{1, 0, 0}, -.5}; // this convention intersects
	r3d_clip(&poly, &p1, 1);
	std::cout << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// tetrahedron, test split
	//////////////////////////////////////////

	// start with raw vertices
	r3d_rvec3 verts[4] = {
		{0., 0., 0.},
		{1., 0., 0.},
		{0., 1., 0.},
		{0., 0., 1.},
	};

	// turn the raw vertices into a poly
	r3d_poly poly;
	r3d_init_tet(&poly, verts);

	std::cout << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	// define the plane
	// the definition of the plane is n.v+d=0, which is why d needs to be negative
	r3d_plane p1 = {{1, 0, 0}, -.5}; // this convention intersects

	// split the poly
	r3d_poly out_pos, out_neg;
	r3d_split(&poly, 1, p1, &out_pos, &out_neg);

	std::cout << "out_pos" << std::endl;
	r3d_print(&out_pos);
	std::cout << std::endl;

	std::cout << "neg" << std::endl;
	r3d_print(&out_neg);

  }

  {

	//////////////////////////////////////////
	// make a cube, the easiest way is to make a bound box
	//////////////////////////////////////////

	// start with raw vertices
	r3d_rvec3 rbounds[2] = {
		{0., 0., 0.},
		{1., 1., 1.},
	};

	// turn the raw vertices into a poly
	r3d_poly poly;
	r3d_init_box(&poly, rbounds);

	std::cout << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// make a cube, using the r3d_init_poly interface
	//////////////////////////////////////////

	// start with raw vertices
	r3d_rvec3 vertices[8] = {
		{0., 0., 0.},
		{1., 0., 0.},
		{1., 1., 0.},
		{0., 1., 0.},
		{0., 0., 1.},
		{1., 0., 1.},
		{1., 1., 1.},
		{0., 1., 1.},
	};

	r3d_int nvertices = 8, nfaces = 6;

	// define connectivity
	r3d_int rawinds[6][4] = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, 
		{3, 0, 4, 7}, {0, 3, 2, 1}, {4, 5, 6, 7}};

	r3d_int numvertsperface[6] = {4, 4, 4, 4, 4, 4};

	// turn the raw vertices into a polym
	r3d_poly poly;

	// make a double-pointer for the faces
	r3d_int *faceinds[nfaces];
	for (r3d_int f = 0; f < nfaces; ++f)
		faceinds[f] = &rawinds[f][0];

	r3d_init_poly(&poly, vertices, nvertices, faceinds, numvertsperface, nfaces);

	std::cout << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// make a cube, using the r3d_poly interface with list initialization and
	// std::vector
	// this method works even if the individual lists aren't the same length
	//////////////////////////////////////////

	// start with raw vertices
	r3d_rvec3 vertices[8] = {
		{0., 0., 0.},
		{1., 0., 0.},
		{1., 1., 0.},
		{0., 1., 0.},
		{0., 0., 1.},
		{1., 0., 1.},
		{1., 1., 1.},
		{0., 1., 1.},
	};

	r3d_int nvertices = 8, nfaces = 6;

	// define connectivity
	std::vector<std::vector<r3d_int>> rawinds = {
		{0, 1, 5, 4},
		{1, 2, 6, 5},
		{2, 3, 7, 6},
		{3, 0, 4, 7},
		{0, 3, 2, 1},
		{4, 5, 6, 7}};

	r3d_int numvertsperface[6] = {4, 4, 4, 4, 4, 4};

	// turn the raw vertices into a polym
	r3d_poly poly;

	// make a double-pointer for the faces
	r3d_int *faceinds[nfaces];
	for (r3d_int f = 0; f < nfaces; ++f)
		faceinds[f] = &rawinds[f][0];

	r3d_init_poly(&poly, vertices, nvertices, faceinds, numvertsperface, nfaces);

	std::cout << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << "poly is good: " << r3d_is_good(&poly) << std::endl;
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// cube, test split
	//////////////////////////////////////////

	// start with raw vertices
	r3d_rvec3 rbounds[2] = {
		{0., 0., 0.},
		{1., 1., 1.},
	};

	// turn the raw vertices into a poly
	r3d_poly poly;
	r3d_init_box(&poly, rbounds);

	std::cout << std::endl << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

	// define the plane
	// the definition of the plane is n.v+d=0, which is why d needs to be negative
	r3d_plane p1 = {{1, 0, 0}, -.5}; // this convention intersects

	// split the poly
	r3d_poly out_pos, out_neg;
	r3d_split(&poly, 1, p1, &out_pos, &out_neg);

	std::cout << "out_pos" << std::endl;
	r3d_print(&out_pos);
	std::cout << std::endl;

	r3d_init_brep(&out_pos, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);
	std::cout << std::endl;

	std::cout << "out_neg" << std::endl;
	r3d_print(&out_neg);
	std::cout << std::endl;

	r3d_init_brep(&out_neg, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// make a double cube and intersect with a plane so it cuts both cubes
	//////////////////////////////////////////

	const r3d_int nvertices = 16;
	const r3d_int nfaces = 12;

	// start with raw vertices
	r3d_rvec3 vertices[nvertices] = {
		{0., 0., 0.},
		{1., 0., 0.},
		{1., 1., 0.},
		{0., 1., 0.},
		{0., 0., 1.},
		{1., 0., 1.},
		{1., 1., 1.},
		{0., 1., 1.},
		{2., 0., 0.},
		{3., 0., 0.},
		{3., 1., 0.},
		{2., 1., 0.},
		{2., 0., 1.},
		{3., 0., 1.},
		{3., 1., 1.},
		{2., 1., 1.},
	};

	// define connectivity
	r3d_int rawinds[nfaces][4] = {
		{0, 1, 5, 4},
		{1, 2, 6, 5},
		{2, 3, 7, 6},
		{3, 0, 4, 7},
		{0, 3, 2, 1},
		{4, 5, 6, 7},
		{8, 9, 13, 12},
		{9, 10, 14, 13},
		{10, 11, 15, 14},
		{11, 8, 12, 15},
		{8, 11, 10, 9},
		{12, 13, 14, 15}};

	r3d_int numvertsperface[nfaces] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

	// turn the raw vertices into a poly
	r3d_poly poly;

	// make a double-pointer for the faces
	r3d_int *faceinds[nfaces];
	for (r3d_int f = 0; f < nfaces; ++f)
		faceinds[f] = &rawinds[f][0];

	r3d_init_poly(&poly, vertices, nvertices, faceinds, numvertsperface, nfaces);

	std::cout << std::endl << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	// get the volume
	r3d_real volume;
	r3d_reduce(&poly, &volume, 0);

	std::cout << "poly is good: " << r3d_is_good(&poly) 
		<< " and has volume " << volume << std::endl << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

	// clip in the z normal direction, should be the upper half of two cubes
	r3d_plane p1 = {{0, 0, 1.}, -.5}; 
	r3d_clip(&poly, &p1, 1);

	// get the volume
	r3d_reduce(&poly, &volume, 0);

	std::cout << "clipped poly is good: " << r3d_is_good(&poly)
		<< " and has volume " << volume << std::endl << std::endl;

	std::cout << "clipped poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// make a dimpled cube
	//////////////////////////////////////////

	const r3d_int nvertices = 9;
	const r3d_int nfaces = 9;
	const r3d_real dimple = -1.; // theoretical volume is 1+dimple/3

	// start with raw vertices
	r3d_rvec3 vertices[nvertices] = {
		{0., 0., 0.},
		{1., 0., 0.},
		{1., 1., 0.},
		{0., 1., 0.},
		{0., 0., 1.},
		{1., 0., 1.},
		{1., 1., 1.},
		{0., 1., 1.},
		{.5, .5, 1. + dimple}};

	// define connectivity
	std::vector<std::vector<r3d_int>> rawinds = {
		{0, 1, 5, 4},
		{1, 2, 6, 5},
		{2, 3, 7, 6},
		{3, 0, 4, 7},
		{0, 3, 2, 1},
		{4, 5, 8},
		{5, 6, 8},
		{6, 7, 8},
		{7, 4, 8}};

	r3d_int numvertsperface[nfaces] = {4, 4, 4, 4, 4, 3, 3, 3, 3};

	// turn the raw vertices into a poly
	r3d_poly poly;

	// make a double-pointer for the faces
	r3d_int *faceinds[nfaces];
	for (r3d_int f = 0; f < nfaces; ++f)
		faceinds[f] = &rawinds[f][0];

	r3d_init_poly(&poly, vertices, nvertices, faceinds, numvertsperface, nfaces);

        std::cout << std::endl << "input poly" << std::endl;
        r3d_print(&poly);
        std::cout << std::endl;

	// get the volume
        r3d_real volume;
        r3d_reduce(&poly, &volume, 0);

	std::cout << "poly is good: " << r3d_is_good(&poly) << " and has volume "
		<< volume << std::endl << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

	// clip in the z normal direction,
	r3d_plane p1 = {{0, 0, 1.}, -.5};
	r3d_clip(&poly, &p1, 1);

	// get the volume
	r3d_reduce(&poly, &volume, 0);

	std::cout << std::endl << "clipped poly is good: " << r3d_is_good(&poly)
		<< " and has volume " << volume << std::endl << std::endl;

	std::cout << "clipped poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// make a pyramid with an octagonal base
	// the vertex is degree 8
	//////////////////////////////////////////

	const r3d_int nvertices = 9;
	const r3d_int nfaces = 9;

	// start with raw vertices
	r3d_rvec3 vertices[nvertices] = {
		{2., 1., 0.},
		{1., 2., 0.},
		{-1., 2., 0.},
		{-2., 1., 0.},
		{-2., -1., 0.},
		{-1., -2., 0.},
		{1., -2., 0.},
		{2., -1., 0.},
		{0., 0., 1.}};

	// define connectivity
	std::vector<std::vector<r3d_int>> rawinds = {
		{0, 1, 8},
		{1, 2, 8},
		{2, 3, 8},
		{3, 4, 8},
		{4, 5, 8},
		{5, 6, 8},
		{6, 7, 8},
		{7, 0, 8},
		{7, 6, 5, 4, 3, 2, 1, 0}};

	r3d_int numvertsperface[nfaces] = {3, 3, 3, 3, 3, 3, 3, 3, 8};

	// turn the raw vertices into a poly
	r3d_poly poly;

	// make a double-pointer for the faces
	r3d_int *faceinds[nfaces];
	for (r3d_int f = 0; f < nfaces; ++f)
		faceinds[f] = &rawinds[f][0];

	r3d_init_poly(&poly, vertices, nvertices, faceinds, numvertsperface, nfaces);

	std::cout << std::endl << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

	//////////////////////////////////////////
	// make a double pyramid with an octagonal center
	//////////////////////////////////////////

	const r3d_int nvertices = 10;
	const r3d_int nfaces = 16;

	// start with raw vertices
	r3d_rvec3 vertices[nvertices] = {
		{2., 1., 0.},
		{1., 2., 0.},
		{-1., 2., 0.},
		{-2., 1., 0.},
		{-2., -1., 0.},
		{-1., -2., 0.},
		{1., -2., 0.},
		{2., -1., 0.},
		{0., 0., 1.},
		{0., 0., -1.}};

	// define connectivity
	std::vector<std::vector<r3d_int>> rawinds = {
		{0, 1, 8},
		{1, 2, 8},
		{2, 3, 8},
		{3, 4, 8},
		{4, 5, 8},
		{5, 6, 8},
		{6, 7, 8},
		{7, 0, 8},
		{7, 6, 9},
		{6, 5, 9},
		{5, 4, 9},
		{4, 3, 9},
		{3, 2, 9},
		{2, 1, 9},
		{1, 0, 9},
		{0, 7, 9}};

	r3d_int numvertsperface[nfaces] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};

	// turn the raw vertices into a poly
	r3d_poly poly;

	// make a double-pointer for the faces
	r3d_int *faceinds[nfaces];
	for (r3d_int f = 0; f < nfaces; ++f)
		faceinds[f] = &rawinds[f][0];

	r3d_init_poly(&poly, vertices, nvertices, faceinds, numvertsperface, nfaces);

	std::cout << std::endl << "input poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }

  {

        //////////////////////////////////////////
        // make a topological cube with a non-planar top
        // cut the cube with a plane whose normal is the z axis
        // if offset is negative the clipped polyhedron should
        // have 2 components (tetrahedrons)
        //////////////////////////////////////////

        const r3d_real offset = -1.;	// theoretical volume is 1+dimple/3

        // start with raw vertices
        r3d_rvec3 vertices[8] = {
                {0., 0., 0.}, 
                {1., 0., 0.}, 
                {1., 1., 0.}, 
                {0., 1., 0.},
                {0., 0., 1.}, 
                {1., 0., 1. + offset}, 
                {1., 1., 1.}, 
                {0., 1., 1. + offset},
        };

        r3d_int nvertices = 8, nfaces = 6;

	// define connectivity
	std::vector<std::vector<r3d_int>> rawinds = {
		{0, 1, 5, 4}, 
		{1, 2, 6, 5},
		{2, 3, 7, 6}, 
		{3, 0, 4, 7},
		{0, 3, 2, 1}, 
		{4, 5, 6, 7}};

	r3d_int numvertsperface[6] = {4, 4, 4, 4, 4, 4};

	// turn the raw vertices into a polym
	r3d_poly poly;

	// make a double-pointer for the faces
	r3d_int *faceinds[nfaces];
	for (r3d_int f = 0; f < nfaces; ++f) faceinds[f] = &rawinds[f][0];

	r3d_init_poly(&poly, vertices, nvertices, faceinds, numvertsperface, nfaces);

	std::cout << "\ninput poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_int numcomponents;
	r3d_brep *brep;
	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

	// clip the poly
	r3d_plane p1 = {{0, 0, 1.}, -.5};
	r3d_clip(&poly, &p1, 1);

	// get the volume
	r3d_real volume;
	r3d_reduce(&poly, &volume, 0);

	std::cout << "\nclipped poly is good: " << r3d_is_good(&poly)
		<< " and has volume " << volume << std::endl << std::endl;

	std::cout << "clipped poly" << std::endl;
	r3d_print(&poly);
	std::cout << std::endl;

	r3d_init_brep(&poly, &brep, &numcomponents);
	r3d_print_brep(&brep, numcomponents);
	r3d_free_brep(&brep, numcomponents);

  }
  
  return 0;
}
