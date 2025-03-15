/**************************************************************
 *
 *    clip.c
 *    
 *    Devon Powell
 *    26 May 2015
 *
 *    Example of the clipping and reduction functionality of r3d
 *    on convex polyhedra. 
 *    Initializes a cube, clips against a plane, then integrates moments
 *    over the resulting polyhedron and prints the results.
 *    See r3d.h for detailed documentation.
 *    
 *    Copyright (C) 2014 Stanford University.
 *    See License.txt for more information.
 *
 *************************************************************/

#include <stdio.h>
#include <math.h>
#include "r3d.h"

// normalizes a vector
// for making clip plane normals
#define dot(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define norm(v) {					\
	r3d_real tmplen = sqrt(dot(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}

int main() {

	// variable declarations for counters and such
	r3d_int m, f, v;


	//// Initialize a unit cube that we will next clip and reduce ////

	// r3d_poly struct contains vertex buffer and other info
	r3d_poly poly;
	// initialize the cube
	r3d_rvec3 rbounds[2] = {
		{-0.5, -0.5, -0.5}, 
		{0.5, 0.5, 0.5} 
	};
	r3du_init_box(&poly, rbounds);
	printf("Initialized a cube with vertices:\n");
	for(v = 0; v < poly.nverts; ++v) printf("  ( %f , %f , %f )\n", poly.verts[v].pos.x, poly.verts[v].pos.y, poly.verts[v].pos.z);


	//// reduce and print moments before clipping //// 
	
	r3d_real moments[10];
	r3d_real polyorder = 2; // 0 for volume only, 1 for linear, 2 for quadratic 
	r3d_reduce(&poly, polyorder, moments);
	printf("Polyhedron moments (pre-clip):\n");
	printf("  Integral[ 1 dV ] = %f\n", moments[0]);
	printf("  Integral[ x dV ] = %f\n", moments[1]);
	printf("  Integral[ y dV ] = %f\n", moments[2]);
	printf("  Integral[ z dV ] = %f\n", moments[3]);
	printf("  Integral[ x^2 dV ] = %f\n", moments[4]);
	printf("  Integral[ y^2 dV ] = %f\n", moments[5]);
	printf("  Integral[ z^2 dV ] = %f\n", moments[6]);
	printf("  Integral[ x*y dV ] = %f\n", moments[7]);
	printf("  Integral[ x*z dV ] = %f\n", moments[8]);
	printf("  Integral[ y*z dV ] = %f\n", moments[9]);


	//// Initialize a list of clip faces (up to 4 at a time in the current implementation) ////
	
	r3d_plane faces[4];
	r3d_int nfaces = 1;
	// set and normalize an arbitrary face normal
	faces[0].n.x = 1.0;
	faces[0].n.y = -1.0;
	faces[0].n.z = 0.8;
	norm(faces[0].n);
	// set the distance to the origin
	faces[0].d = 0.1;
	printf("Initialized %d clip planes:\n", nfaces);
	for(f = 0; f < nfaces; ++f) printf("  n = ( %f , %f , %f ), d = %f\n", faces[f].n.x, faces[f].n.y, faces[f].n.z, faces[f].d);


	//// Prepare for clipping by setting clip plane distances and bit flags ////
	
	// check all vertices against each clip plane
	r3d_real perp_dist;
	for(v = 0; v < poly.nverts; ++v) {
		poly.verts[v].orient.fflags = 0x00;
		// set bit flags and geometry for faces to be clipped against
		for(f = 0; f < nfaces; ++f) {
			perp_dist = faces[f].d + dot(poly.verts[v].pos, faces[f].n);
			poly.verts[v].orient.fdist[f] = perp_dist;
			if(perp_dist > 0.0) poly.verts[v].orient.fflags |= (1 << f);
		}
		// set bit flags for faces to be ignored
		for(f = nfaces; f < 4; ++f) poly.verts[v].orient.fflags |= (1 << f);
	}

	// initialize comparison flags
	// bitwise comparisons are an easy way to decide if/how
	// the polyhedron needs to be processed
	unsigned char orcmp = 0x00;
	unsigned char andcmp = 0x0f;
	for(v = 0; v < poly.nverts; ++v) {
		orcmp |= poly.verts[v].orient.fflags; 
		andcmp &= poly.verts[v].orient.fflags; 
	}


	//// Reduce the voxel if necessary ////
	
	if(andcmp == 0x0f) {
		// the polyhedron is entirely inside all clip planes,
		// so no clipping is required.
		printf("The polyhedron is entirely inside all clip planes. No reduction necessary.\n");
	}
	else if(orcmp == 0x0f) {
		// the polyhedron is intersected by the clip planes
		// need to clip and reduce a nontrivial polyhedron
		printf("Clipping and reducing...\n");
		r3d_clip_tet(&poly, andcmp);
		r3d_reduce(&poly, polyorder, moments);
		printf("  done.\n");
	}
	else {
		// the polyhedron is entirely outside of a clip plane.
		// Ignore it.
		printf("The voxel is entirely outside of at least one clip plane. No reduction necessary.\n");
		for(m = 0; m < 10; ++m) moments[m] = 0.0;
	}


	//// Print the results ////
	printf("Polyhedron moments (post-clip):\n");
	printf("  Integral[ 1 dV ] = %f\n", moments[0]);
	printf("  Integral[ x dV ] = %f\n", moments[1]);
	printf("  Integral[ y dV ] = %f\n", moments[2]);
	printf("  Integral[ z dV ] = %f\n", moments[3]);
	printf("  Integral[ x^2 dV ] = %f\n", moments[4]);
	printf("  Integral[ y^2 dV ] = %f\n", moments[5]);
	printf("  Integral[ z^2 dV ] = %f\n", moments[6]);
	printf("  Integral[ x*y dV ] = %f\n", moments[7]);
	printf("  Integral[ x*z dV ] = %f\n", moments[8]);
	printf("  Integral[ y*z dV ] = %f\n", moments[9]);

	return 0;
}
