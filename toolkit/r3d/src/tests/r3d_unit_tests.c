/*
 *
 *		r3d_unit_tests.c 
 *		
 *		Definitions for r3d unit tests. 
 *		
 *		Devon Powell
 *		31 August 2015
 *
 *		This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 *		Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). 
 *		All rights in the program are reserved by the DOE and Los Alamos National Security, LLC.  
 *		Permission is granted to the public to copy and use this software without charge, provided that 
 *		this Notice and any statement of authorship are reproduced on all copies.  Neither the U.S. 
 *		Government nor LANS makes any warranty, express or implied, or assumes any liability 
 *		or responsibility for the use of this software.
 *
 */

#include "test_helpers.h"
#include "utest.h"
#include <math.h>

// numerical tolerances for pass/warn/fail tests
#define TOL_WARN 1.0e-8
#define TOL_FAIL 1.0e-4

// minimum volume allowed for test polyhedra 
#define MIN_VOL 1.0e-8

// number of trials per test
#define NUM_TRIALS 10

// for recursive tests, the stack capacity and 
// maximum recursion depth allowed
#define STACK_SIZE 64
#define MAX_DEPTH 16

// order of polynomial integration for all tests 
#define POLY_ORDER 2


// --                 unit tests                    -- //

void test_split_tets_thru_centroid() {

	// a very basic sanity check. Splits a tet through its centroid
	// and checks to see whether the two resulting volumes add up to equal the original

	// variables: the polyhedra and their moments
	r3d_int m, i;
	r3d_rvec3 verts[4];
	r3d_real om[R3D_NUM_MOMENTS(POLY_ORDER)], m1[R3D_NUM_MOMENTS(POLY_ORDER)], m2[R3D_NUM_MOMENTS(POLY_ORDER)];
	r3d_plane splane;

	// generate a random tet and clip plane
	r3d_int ntets = 128;
	r3d_poly opoly[ntets], poly1[ntets], poly2[ntets];
	for(i = 0; i < ntets; ++i) {
		rand_tet_3d(verts, MIN_VOL);
		r3d_init_tet(&opoly[i], verts);
		if(i == 13)
			splane = thru_cent_3d(&opoly[i]);
	}

	// split them all about the same plane
	r3d_split(opoly, ntets, splane, poly1, poly2);

	for(i = 0; i < ntets; ++i) {

		// reduce the original and its two parts
		r3d_reduce(&opoly[i], om, POLY_ORDER);
		r3d_reduce(&poly1[i], m1, POLY_ORDER);
		r3d_reduce(&poly2[i], m2, POLY_ORDER);
	
		// make sure the sum of moments equals the original 
		for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
			ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
			EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
		}

		//printf(" original = %f, parts = %f %f, sum = %f\n", om[0], m1[0], m2[0], m1[0]+m2[0]);
		
		// make sure neither of the two resulting volumes is larger than the original
		// (within some tolerance)
		ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
		EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
		ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
		EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));
	}
}

void test_split_nonconvex() {

	// a very basic sanity check. Splits a nonconvex poly (a zig-zag prism of sorts)
	// and checks to see whether the two resulting volumes add up to equal the original
	
	// variables: the polyhedra and their moments
	r3d_int m, v;
	r3d_plane splane;
	r3d_poly opoly, poly1, poly2;
	r3d_real om[R3D_NUM_MOMENTS(POLY_ORDER)], m1[R3D_NUM_MOMENTS(POLY_ORDER)], m2[R3D_NUM_MOMENTS(POLY_ORDER)];

	// explicitly create the nonconvex poly
#define NZIGS 3
#define ZOFF 0.1
	opoly.nverts = 4*NZIGS;
	for(v = 0; v < NZIGS; ++v) {
		opoly.verts[v].pos.x = 1.0*v;
		opoly.verts[v].pos.y = ZOFF + 1.0*(v%2);
		opoly.verts[v].pos.z = 0.0;
		opoly.verts[v+NZIGS].pos.x = 1.0*(NZIGS-v-1);
		opoly.verts[v+NZIGS].pos.y = 1.0*((NZIGS-v-1)%2);
		opoly.verts[v+NZIGS].pos.z = 0.0;
		opoly.verts[v+2*NZIGS].pos.x = 1.0*v;
		opoly.verts[v+2*NZIGS].pos.y = ZOFF + 1.0*(v%2);
		opoly.verts[v+2*NZIGS].pos.z = 1.0;
		opoly.verts[v+3*NZIGS].pos.x = 1.0*(NZIGS-v-1);
		opoly.verts[v+3*NZIGS].pos.y = 1.0*((NZIGS-v-1)%2);
		opoly.verts[v+3*NZIGS].pos.z = 1.0;
	}
	for(v = 0; v < 2*NZIGS; ++v) {
		opoly.verts[v].pnbrs[0] = (v+1)%(2*NZIGS); 
		opoly.verts[v].pnbrs[1] = (v+2*NZIGS-1)%(2*NZIGS); 
		opoly.verts[v].pnbrs[2] = (v+2*NZIGS); 
		opoly.verts[v+2*NZIGS].pnbrs[0] = (v+2*NZIGS-1)%(2*NZIGS)+2*NZIGS; 
		opoly.verts[v+2*NZIGS].pnbrs[1] = (v+1)%(2*NZIGS)+2*NZIGS; 
		opoly.verts[v+2*NZIGS].pnbrs[2] = v; 
	}

	// split along the x-axis (two single connected components this direction)
	poly1 = opoly;
	poly2 = opoly;
	splane.n.x = 1.0;
	splane.n.y = 0.0;
	splane.n.z = 0.0;
	splane.d = -0.5*(NZIGS-1);
	r3d_clip(&poly1, &splane, 1);
	splane.n.x *= -1;
	splane.n.y *= -1;
	splane.n.z *= -1;
	splane.d *= -1;
	r3d_clip(&poly2, &splane, 1);
	r3d_reduce(&opoly, om, POLY_ORDER);
	r3d_reduce(&poly1, m1, POLY_ORDER);
	r3d_reduce(&poly2, m2, POLY_ORDER);
	for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
		ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
		EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
	}
	ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
	EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
	ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
	EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));

	// split along the z-axis (two single connected components this direction)
	poly1 = opoly;
	poly2 = opoly;
	splane.n.x = 0.0;
	splane.n.y = 0.0;
	splane.n.z = 1.0;
	splane.d = -0.5;
	r3d_clip(&poly1, &splane, 1);
	splane.n.x *= -1;
	splane.n.y *= -1;
	splane.n.z *= -1;
	splane.d *= -1;
	r3d_clip(&poly2, &splane, 1);
	r3d_reduce(&opoly, om, POLY_ORDER);
	r3d_reduce(&poly1, m1, POLY_ORDER);
	r3d_reduce(&poly2, m2, POLY_ORDER);
	for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
		ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
		EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
	}
	ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
	EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
	ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
	EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));

	// split along the y-axis (multiple connected components this direction)
	poly1 = opoly;
	poly2 = opoly;
	splane.n.x = 0.0;
	splane.n.y = 1.0;
	splane.n.z = 0.0;
	splane.d = -0.5*(1.0+ZOFF);
	r3d_clip(&poly1, &splane, 1);
	splane.n.x *= -1;
	splane.n.y *= -1;
	splane.n.z *= -1;
	splane.d *= -1;
	r3d_clip(&poly2, &splane, 1);
	r3d_reduce(&opoly, om, POLY_ORDER);
	r3d_reduce(&poly1, m1, POLY_ORDER);
	r3d_reduce(&poly2, m2, POLY_ORDER);
	for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
		ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
		EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
	}
	ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
	EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
	ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
	EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));

}


void test_recursive_splitting_nondegenerate() {

	// recursively splits a polyhedron (starting from a tet) through the centroid,
	// checking to see that the resulting volumes add up properly.

	// explicit stack-based implementation
	r3d_int nstack, depth, t, m;
	r3d_poly polystack[STACK_SIZE];
	r3d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r3d_rvec3 verts[4];
	r3d_plane splane;
	r3d_poly opoly, poly1, poly2;
	r3d_real om[R3D_NUM_MOMENTS(POLY_ORDER)], m1[R3D_NUM_MOMENTS(POLY_ORDER)], m2[R3D_NUM_MOMENTS(POLY_ORDER)];

	// do many trials
	printf("Recursively splitting %d tetrahedra, maximum splits per tet is %d.\n", NUM_TRIALS, MAX_DEPTH);
	for(t = 0; t < NUM_TRIALS; ++t) {

		// generate a random tet
		rand_tet_3d(verts, MIN_VOL);
		r3d_init_tet(&opoly, verts);
	
		// push the starting tet to the stack
		nstack = 0;
		polystack[nstack] = opoly;
		depthstack[nstack] = 0;
		++nstack;	
	
		// recursively split the poly
		while(nstack > 0) {
	
			// pop the stack
			--nstack;
			opoly = polystack[nstack];
			depth = depthstack[nstack];
	
			// generate a randomly oriented plane
			// through the centroid of the poly
			splane = thru_cent_3d(&opoly);

			// split the poly by making two copies of the original poly
			// and them clipping them against the same plane, with one
			// oriented oppositely
			poly1 = opoly;
			poly2 = opoly;
			r3d_clip(&poly1, &splane, 1);
			splane.n.x *= -1;
			splane.n.y *= -1;
			splane.n.z *= -1;
			splane.d *= -1;
			r3d_clip(&poly2, &splane, 1);

			// reduce the original and its two parts
			r3d_reduce(&opoly, om, POLY_ORDER);
			r3d_reduce(&poly1, m1, POLY_ORDER);
			r3d_reduce(&poly2, m2, POLY_ORDER);

			// make sure the sum of moments equals the original 
			for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
				ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
				EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
			}
			
			// make sure neither of the two resulting volumes is larger than the original
			// (within some tolerance)
			ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
			EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
			ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
			EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));

			//printf("nstack = %d, depth = %d, opoly = %.10e, p1 = %.10e, p2 = %.10e, err = %.10e\n", 
					//nstack, depth, om[0], m1[0], m2[0], fabs(1.0 - om[0]/(m1[0] + m2[0])));
	
			// push the children to the stack if they have
			// an acceptably large volume
			if(depth < MAX_DEPTH) {
				if(m1[0] > MIN_VOL) {
					polystack[nstack] = poly1;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
				if(m2[0] > MIN_VOL) {
					polystack[nstack] = poly2;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
			}
		}
	}
}

void test_recursive_splitting_degenerate() {

	// recursively splits a polyhedron (starting from a tet) with a cut plane
	// that is always degenerate with the tet in some way,
	// checking to see that the resulting volumes add up properly.

	// explicit stack-based implementation
	r3d_int nstack, depth, t, chopt, m;
	r3d_poly polystack[STACK_SIZE];
	r3d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r3d_rvec3 verts[4];
	r3d_plane splane;
	r3d_poly opoly, poly1, poly2;
	r3d_real om[R3D_NUM_MOMENTS(POLY_ORDER)], m1[R3D_NUM_MOMENTS(POLY_ORDER)], m2[R3D_NUM_MOMENTS(POLY_ORDER)];

	// do many trials
	printf("Recursively splitting %d tetrahedra, maximum splits per tet is %d.\n", NUM_TRIALS, MAX_DEPTH);
	for(t = 0; t < NUM_TRIALS; ++t) {

		// generate a random tet
		rand_tet_3d(verts, MIN_VOL);
		r3d_init_tet(&opoly, verts);
	
		// push the starting tet to the stack
		nstack = 0;
		polystack[nstack] = opoly;
		depthstack[nstack] = 0;
		++nstack;	
	
		// recursively split the poly
		while(nstack > 0) {
	
			// pop the stack
			--nstack;
			opoly = polystack[nstack];
			depth = depthstack[nstack];
	
			// generate a random plane from one of a few
			// possible degenerate configurations, ensuring that it
			// has a valid unit normal
			chopt = rand_int(6);
			do {
				splane = choptions_3d[chopt](&opoly);
			} while(splane.n.x == 0.0 && splane.n.y == 0.0 && splane.n.z == 0.0);

			// split the poly by making two copies of the original poly
			// and them clipping them against the same plane, with one
			// oriented oppositely
			poly1 = opoly;
			poly2 = opoly;
			r3d_clip(&poly1, &splane, 1);
			splane.n.x *= -1;
			splane.n.y *= -1;
			splane.n.z *= -1;
			splane.d *= -1;
			r3d_clip(&poly2, &splane, 1);

			// reduce the original and its two parts
			r3d_reduce(&opoly, om, POLY_ORDER);
			r3d_reduce(&poly1, m1, POLY_ORDER);
			r3d_reduce(&poly2, m2, POLY_ORDER);
		
			// make sure the sum of moments equals the original 
			for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
				ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
				EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
			}
		
			// make sure neither of the two resulting volumes is larger than the original
			// (within some tolerance)
			ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
			EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
			ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
			EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));
	
			//printf("nstack = %d, depth = %d, opoly = %.10e, p1 = %.10e, p2 = %.10e, err = %.10e\n", 
					//nstack, depth, om[0], m1[0], m2[0], fabs(1.0 - om[0]/(m1[0] + m2[0])));
	
			// push the children to the stack if they have
			// an acceptably large volume
			if(depth < MAX_DEPTH) {
				if(m1[0] > MIN_VOL) {
					polystack[nstack] = poly1;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
				if(m2[0] > MIN_VOL) {
					polystack[nstack] = poly2;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
			}
		}
	}
}

void test_recursive_splitting_degenerate_perturbed() {

	// recursively splits a polyhedron (starting from a tet) with a cut plane
	// that is always degenerate with the tet in some way,
	// checking to see that the resulting volumes add up properly.
	// In this one, the cut plane is perturbed by a small amount.

#define MIN_PERTURB_ORDER (-17)
#define MAX_PERTURB_ORDER (-1)

	// explicit stack-based implementation
	r3d_int nstack, depth, t, chopt, m;
	r3d_poly polystack[STACK_SIZE];
	r3d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r3d_rvec3 verts[4];
	r3d_plane splane;
	r3d_poly opoly, poly1, poly2;
	r3d_real om[R3D_NUM_MOMENTS(POLY_ORDER)], m1[R3D_NUM_MOMENTS(POLY_ORDER)], m2[R3D_NUM_MOMENTS(POLY_ORDER)];
	r3d_real perturb;

	// do many trials
	printf("Recursively splitting %d tetrahedra, maximum splits per tet is %d.\n", NUM_TRIALS, MAX_DEPTH);
	for(t = 0; t < NUM_TRIALS; ++t) {

		// compute the order of magnitude by which to perturb the clip plane,
		// determined by the trial number
		perturb = pow(10, MIN_PERTURB_ORDER + t%(MAX_PERTURB_ORDER - MIN_PERTURB_ORDER));
		//printf("omag = %d, pow = %.10e\n", MIN_PERTURB_ORDER + t%(MAX_PERTURB_ORDER - MIN_PERTURB_ORDER), perturb);

		// generate a random tet
		rand_tet_3d(verts, MIN_VOL);
		r3d_init_tet(&opoly, verts);
	
		// push the starting tet to the stack
		nstack = 0;
		polystack[nstack] = opoly;
		depthstack[nstack] = 0;
		++nstack;	
	
		// recursively split the poly
		while(nstack > 0) {
	
			// pop the stack
			--nstack;
			opoly = polystack[nstack];
			depth = depthstack[nstack];
	
			// generate a random plane from one of a few
			// possible degenerate configurations, ensuring that it
			// has a valid unit normal
			chopt = rand_int(6);
			do {
				splane = choptions_3d[chopt](&opoly);
			} while(splane.n.x == 0.0 && splane.n.y == 0.0 && splane.n.z == 0.0);

			// randomly perturb the plane
			splane.n.x *= 1.0 + perturb*(rand_uniform() - 0.5);
			splane.n.y *= 1.0 + perturb*(rand_uniform() - 0.5);
			splane.n.z *= 1.0 + perturb*(rand_uniform() - 0.5);
			splane.d *= 1.0 + perturb*(rand_uniform() - 0.5);
	
			// split the poly by making two copies of the original poly
			// and them clipping them against the same plane, with one
			// oriented oppositely
			poly1 = opoly;
			poly2 = opoly;
			r3d_clip(&poly1, &splane, 1);
			splane.n.x *= -1;
			splane.n.y *= -1;
			splane.n.z *= -1;
			splane.d *= -1;
			r3d_clip(&poly2, &splane, 1);

			// reduce the original and its two parts
			r3d_reduce(&opoly, om, POLY_ORDER);
			r3d_reduce(&poly1, m1, POLY_ORDER);
			r3d_reduce(&poly2, m2, POLY_ORDER);
			
			// make sure the sum of moments equals the original 
			for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
				ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
				EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
			}
		
			// make sure neither of the two resulting volumes is larger than the original
			// (within some tolerance)
			ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
			EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
			ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
			EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));
	
			//printf("nstack = %d, depth = %d, opoly = %.10e, p1 = %.10e, p2 = %.10e, err = %.10e\n", 
					//nstack, depth, om[0], m1[0], m2[0], fabs(1.0 - om[0]/(m1[0] + m2[0])));
	
			// push the children to the stack if they have
			// an acceptably large volume
			if(depth < MAX_DEPTH) {
				if(m1[0] > MIN_VOL) {
					polystack[nstack] = poly1;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
				if(m2[0] > MIN_VOL) {
					polystack[nstack] = poly2;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
			}
		}
	}
}



void test_tet_tet_timing() {

	// Intersects pairs of tetrahedra, for timing purposes only.

#undef NUM_TRIALS
#define NUM_TRIALS 100000

	// variables: the polyhedra and their moments
	r3d_poly poly; 
	r3d_rvec3 verts[4];
	r3d_plane faces[4];
	r3d_real om[R3D_NUM_MOMENTS(POLY_ORDER)];

	// for timing only, no assertions
	r3d_int trial;
	printf("Intersecting %d pairs of tetrahedra.\n", NUM_TRIALS);
	for(trial = 0; trial < NUM_TRIALS; ++trial) {

		// generate the first random tet
		rand_tet_3d(verts, MIN_VOL);
		r3d_init_tet(&poly, verts);

		// generate the second random tet
		rand_tet_3d(verts, MIN_VOL);
		r3d_tet_faces_from_verts(faces, verts);

		// clip the first tet against the faces of the second
		r3d_clip(&poly, faces, 4);

		// find the moments (up to quadratic order) of the clipped poly
		r3d_reduce(&poly, om, POLY_ORDER);

	}
}


void test_torus_load_and_chop() {

	// recursively splits a polyhedron (starting from a torus) with a cut plane
	// that is always degenerate with the poly in some way,
	// checking to see that the resulting volumes add up properly.
	// This checks non-convex geometry, and also r3d_init_poly() when more than
	// three edges per vertex are needed.
	
	r3d_int i, j, f, m;

	// explicit stack-based implementation
	r3d_int nstack, depth, chopt;
	r3d_poly polystack[STACK_SIZE];
	r3d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r3d_plane splane;
	r3d_poly opoly, poly1, poly2;
	r3d_real om[R3D_NUM_MOMENTS(POLY_ORDER)], m1[R3D_NUM_MOMENTS(POLY_ORDER)], m2[R3D_NUM_MOMENTS(POLY_ORDER)];
	
	// torus parameters
#define NTHETA 3
#define NPHI 3
#define R_MAJ 1.0
#define R_MIN 0.1
#define TWOPI 6.28318530718

	r3d_int nverts = NTHETA*NPHI;
	r3d_rvec3 vertices[nverts];

	// generate the vertex coordinates
	for(i = 0; i < NPHI; ++i)
	for(j = 0; j < NTHETA; ++j) {
		vertices[i*NTHETA + j].x = (R_MAJ + R_MIN*cos(j*TWOPI/NTHETA))*cos(i*TWOPI/NPHI);
		vertices[i*NTHETA + j].y = (R_MAJ + R_MIN*cos(j*TWOPI/NTHETA))*sin(i*TWOPI/NPHI);
		vertices[i*NTHETA + j].z = R_MIN*sin(j*TWOPI/NTHETA);

		// shift away from the origin so that none of the moments are identically zero
		vertices[i*NTHETA + j].x += 0.5; 
		vertices[i*NTHETA + j].y += 0.7; 
		vertices[i*NTHETA + j].z += 0.9; 
	}

#if 0
	// generate quadrilateral faces
	r3d_int nfaces = NTHETA*NPHI;
	r3d_int vertsperface[nfaces];
	r3d_int rawinds[nfaces][4];
	for(i = 0; i < NPHI; ++i)
	for(j = 0; j < NTHETA; ++j) {
		vertsperface[i*NTHETA + j] = 4;
		rawinds[i*NTHETA + j][0] = (i)*NTHETA + (j);
		rawinds[i*NTHETA + j][1] = ((i+1)%NPHI)*NTHETA + (j);
		rawinds[i*NTHETA + j][2] = ((i+1)%NPHI)*NTHETA + ((j+1)%NTHETA);
		rawinds[i*NTHETA + j][3] = (i)*NTHETA + ((j+1)%NTHETA);
	}
#else
	// generate triangular faces
	r3d_int nfaces = 2*NTHETA*NPHI;
	r3d_int vertsperface[nfaces];
	r3d_int rawinds[nfaces][3];
	for(i = 0; i < NPHI; ++i)
	for(j = 0; j < NTHETA; ++j) {
		vertsperface[2*(i*NTHETA + j)] = 3;
		rawinds[2*(i*NTHETA + j)][0] = (i)*NTHETA + (j);
		rawinds[2*(i*NTHETA + j)][1] = ((i+1)%NPHI)*NTHETA + (j);
		rawinds[2*(i*NTHETA + j)][2] = ((i+1)%NPHI)*NTHETA + ((j+1)%NTHETA);
		vertsperface[2*(i*NTHETA + j)+1] = 3;
		rawinds[2*(i*NTHETA + j)+1][0] = (i)*NTHETA + (j);
		rawinds[2*(i*NTHETA + j)+1][1] = ((i+1)%NPHI)*NTHETA + ((j+1)%NTHETA);
		rawinds[2*(i*NTHETA + j)+1][2] = (i)*NTHETA + ((j+1)%NTHETA);
	}
#endif

	// make a double-pointer for the faces
	r3d_int* faceinds[nfaces];
	for(f = 0; f < nfaces; ++f) faceinds[f] = &rawinds[f][0];

	// initialize a general polyhedron
	r3d_init_poly(&opoly, vertices, nverts, faceinds, vertsperface, nfaces);

	// push the torus to the stack
	nstack = 0;
	polystack[nstack] = opoly;
	depthstack[nstack] = 0;
	++nstack;	

	// recursively split the poly
	while(nstack > 0) {

		// pop the stack
		--nstack;
		opoly = polystack[nstack];
		depth = depthstack[nstack];

		// generate a random plane from one of a few
		// possible degenerate configurations, ensuring that it
		// has a valid unit normal
		chopt = rand_int(6);
		do {
			splane = choptions_3d[chopt](&opoly);
		} while(splane.n.x == 0.0 && splane.n.y == 0.0 && splane.n.z == 0.0);

		// split the poly by making two copies of the original poly
		// and them clipping them against the same plane, with one
		// oriented oppositely
		poly1 = opoly;
		poly2 = opoly;
		r3d_clip(&poly1, &splane, 1);
		splane.n.x *= -1;
		splane.n.y *= -1;
		splane.n.z *= -1;
		splane.d *= -1;
		r3d_clip(&poly2, &splane, 1);

		// reduce the original and its two parts
		r3d_reduce(&opoly, om, POLY_ORDER);
		r3d_reduce(&poly1, m1, POLY_ORDER);
		r3d_reduce(&poly2, m2, POLY_ORDER);
	
		// make sure the sum of moments equals the original 
		for(m = 0; m < R3D_NUM_MOMENTS(POLY_ORDER); ++m) {
			ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
			EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
		}
	
		// make sure neither of the two resulting volumes is larger than the original
		// (within some tolerance)
		ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
		EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
		ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
		EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));

		//printf("nstack = %d, depth = %d, opoly = %.10e, p1 = %.10e, p2 = %.10e, err = %.10e\n", 
				//nstack, depth, om[0], m1[0], m2[0], fabs(1.0 - om[0]/(m1[0] + m2[0])));

		// push the children to the stack if they have
		// an acceptably large volume
		if(depth < MAX_DEPTH) {
			if(m1[0] > MIN_VOL) {
				polystack[nstack] = poly1;
				depthstack[nstack] = depth + 1;
				++nstack;	
			}
			if(m2[0] > MIN_VOL) {
				polystack[nstack] = poly2;
				depthstack[nstack] = depth + 1;
				++nstack;	
			}
		}
	}
}

void test_voxelization() {

	// Test r3d_voxelize() by checking that the voxelized moments
	// do indeed sum to those of the original input

#undef POLY_ORDER
#define POLY_ORDER 4
#define NGRID 23

	// vars
	r3d_int i, j, v, curorder, mind;
	r3d_long gg; 
	r3d_int nmom = R3D_NUM_MOMENTS(POLY_ORDER);
	r3d_real voxsum, tmom[nmom];
	r3d_poly poly;
	r3d_rvec3 verts[4];

	// create a random tet in the unit box
	rand_tet_3d(verts, MIN_VOL);
	for(v = 0; v < 4; ++v)
	for(i = 0; i < 3; ++i) {
		verts[v].xyz[i] += 1.0;
		verts[v].xyz[i] *= 0.5;
	}
	r3d_init_tet(&poly, verts);

	// get its original moments for reference
	r3d_reduce(&poly, tmom, POLY_ORDER);

	// voxelize it
	r3d_rvec3 dx = {{1.0/NGRID, 1.0/NGRID, 1.0/NGRID}};
	r3d_dvec3 ibox[2];
	r3d_get_ibox(&poly, ibox, dx);
	printf("Voxelizing a tetrahedron to a grid with dx = %f %f %f and moments of order %d\n", dx.x, dx.y, dx.z, POLY_ORDER);
	printf("Minimum index box = %d %d %d to %d %d %d\n", ibox[0].i, ibox[0].j, ibox[0].k, ibox[1].i, ibox[1].j, ibox[1].k);
	r3d_int nvoxels = (ibox[1].i-ibox[0].i)*(ibox[1].j-ibox[0].j)*(ibox[1].k-ibox[0].k);
	r3d_real* grid = (r3d_real *) calloc(nvoxels*nmom, sizeof(r3d_real));
	r3d_voxelize(&poly, ibox, grid, dx, POLY_ORDER);
	
	// make sure the sum of each moment equals the original 
	for(curorder = 0, mind = 0; curorder <= POLY_ORDER; ++curorder) {
		//printf("Order = %d\n", curorder);
		for(i = curorder; i >= 0; --i)
		for(j = curorder - i; j >= 0; --j, ++mind) {
			//k = curorder - i - j;
			voxsum = 0.0;
			for(gg = 0; gg < nvoxels; ++gg) voxsum += grid[nmom*gg+mind];
			//printf(" Int[ x^%d y^%d z^%d dV ] original = %.10e, voxsum = %.10e, error = %.10e\n", 
					//i, j, k, tmom[mind], voxsum, fabs(1.0 - tmom[mind]/voxsum));
			ASSERT_EQ(tmom[mind], voxsum, TOL_FAIL);
			EXPECT_EQ(tmom[mind], voxsum, TOL_WARN);
		}
	}
	free(grid);
}


void test_moments() {

	// check the moments against an analytic test case
	// (an axis-aligned box) up to some arbitrary order 
	
#undef POLY_ORDER
#undef NUM_TRIALS
#define POLY_ORDER 20
#define NUM_TRIALS 1000
	
	r3d_int i, j, k, mind, curorder;
	r3d_poly poly;
	r3d_rvec3 box[2];
	r3d_real moments[R3D_NUM_MOMENTS(POLY_ORDER)];
	r3d_real exact;

	// initialize the box
  	box[0].x = 1.0;
  	box[0].y = 1.0;
  	box[0].z = 1.0;
  	box[1].x = 1.5;
  	box[1].y = 2.0;
  	box[1].z = 2.5;
	r3d_init_box(&poly, box);


	r3d_int trial;
	printf("Computing moments of order %d, %d trials.\n", POLY_ORDER, NUM_TRIALS);
	for(trial = 0; trial < NUM_TRIALS; ++trial) {

		// get the moments from r3d_reduce
		r3d_reduce(&poly, moments, POLY_ORDER);
	
		// Check all the moments against the analytic solution 
		// NOTE: This is the order that r3d_reduce puts out! 
		for(curorder = 0, mind = 0; curorder <= POLY_ORDER; ++curorder) {
			//printf("Order = %d\n", curorder);
			for(i = curorder; i >= 0; --i)
			for(j = curorder - i; j >= 0; --j, ++mind) {
				k = curorder - i - j;
				exact = 1.0/((i+1)*(j+1)*(k+1))*(pow(box[1].x, i+1) - pow(box[0].x, i+1))
					*(pow(box[1].y, j+1) - pow(box[0].y, j+1))*(pow(box[1].z, k+1) - pow(box[0].z, k+1));
				//printf(" Int[ x^%d y^%d z^%d dV ] = %.10e, analytic = %.10e, frac = %f, error = %.10e\n", 
						//i, j, k, moments[mind], exact, moments[mind]/exact, fabs(1.0 - moments[mind]/exact));
				ASSERT_EQ(moments[mind], exact, TOL_FAIL);
				EXPECT_EQ(moments[mind], exact, TOL_WARN);
			}
		}
	}

}

// -- user-implemented functions declared in utest.h -- //

void register_all_tests() {

	register_test(test_split_tets_thru_centroid, "split_tets_thru_centroid");
	register_test(test_split_nonconvex, "split_nonconvex");
	register_test(test_recursive_splitting_nondegenerate, "recursive_splitting_nondegenerate");
	register_test(test_recursive_splitting_degenerate, "recursive_splitting_degenerate");
	register_test(test_recursive_splitting_degenerate_perturbed, "recursive_splitting_degenerate_perturbed");
	register_test(test_tet_tet_timing, "tet_tet_timing");
	register_test(test_torus_load_and_chop, "torus_load_and_chop");
	register_test(test_moments, "moments");
	register_test(test_voxelization, "voxelization");

}

void setup() {

	// no print buffer
	setbuf(stdout, NULL);

	// random number seed
	srand(time(NULL));
	//srand(10291986);

	// initialize random clip plane options
	choptions_3d[0] = thru_cent_3d;
	choptions_3d[1] = thru_face_3d;
	choptions_3d[2] = thru_edge_cent_3d;
	choptions_3d[3] = thru_edge_rand_3d;
	choptions_3d[4] = thru_vert_cent_3d;
	choptions_3d[5] = thru_vert_rand_3d;
}

/////////////////////////////////////////////////////////
