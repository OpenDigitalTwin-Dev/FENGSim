/*
 *
 *		r2d_unit_tests.c 
 *		
 *		Definitions for r2d unit tests. 
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
#include "r2d.h"
#include "utest.h"
#include <math.h>

// numerical tolerances for pass/warn/fail tests
#define TOL_WARN 1.0e-8
#define TOL_FAIL 1.0e-4

// minimum volume allowed for test polyhedra 
#define MIN_AREA 1.0e-8

// number of trials per test
#define NUM_TRIALS 100

// for recursive tests, the stack capacity and 
// maximum recursion depth allowed
#define STACK_SIZE 64
#define MAX_DEPTH 16

// order of polynomial integration for all tests 
#define POLY_ORDER 2


// --                 unit tests                    -- //



void test_split_tris_thru_centroid() {

	// a very basic sanity check. Splits a tet through its centroid
	// and checks to see whether the two resulting volumes add up to equal the original

	// variables: the polyhedra and their moments
	r2d_int m, i;
	r2d_rvec2 verts[3];
	r2d_real om[R2D_NUM_MOMENTS(POLY_ORDER)], m1[R2D_NUM_MOMENTS(POLY_ORDER)], m2[R2D_NUM_MOMENTS(POLY_ORDER)];
	r2d_plane splane;

	// generate a random tet and clip plane
	r2d_int ntris = 128;
	r2d_poly opoly[ntris], poly1[ntris], poly2[ntris];
	for(i = 0; i < ntris; ++i) {
		rand_tri_2d(verts, MIN_AREA);
		r2d_init_poly(&opoly[i], verts, 3);
		if(i == 13)
			splane = thru_cent_2d(&opoly[i]);
	}

	// split them all about the same plane
	r2d_split(opoly, ntris, splane, poly1, poly2);

	for(i = 0; i < ntris; ++i) {

		// reduce the original and its two parts
		r2d_reduce(&opoly[i], om, POLY_ORDER);
		r2d_reduce(&poly1[i], m1, POLY_ORDER);
		r2d_reduce(&poly2[i], m2, POLY_ORDER);
	
		// make sure the sum of moments equals the original 
		for(m = 0; m < R2D_NUM_MOMENTS(POLY_ORDER); ++m) {
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

void test_recursive_splitting_nondegenerate() {

	// recursively splits a polygon (starting from a tet) through the centroid,
	// checking to see that the resulting volumes add up properly.

	// explicit stack-based implementation
	r2d_int nstack, depth, t, m;
	r2d_poly polystack[STACK_SIZE];
	r2d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r2d_rvec2 verts[3];
	r2d_plane splane;
	r2d_poly opoly, poly1, poly2;
	r2d_real om[R2D_NUM_MOMENTS(POLY_ORDER)], m1[R2D_NUM_MOMENTS(POLY_ORDER)], m2[R2D_NUM_MOMENTS(POLY_ORDER)];

	// do many trials
	printf("Recursively splitting %d triangles, maximum splits per tri is %d.\n", NUM_TRIALS, MAX_DEPTH);
	for(t = 0; t < NUM_TRIALS; ++t) {

		// generate a random tet
		rand_tri_2d(verts, MIN_AREA);
		r2d_init_poly(&opoly, verts, 3);
	
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
			splane = thru_cent_2d(&opoly);

			// split the poly by making two copies of the original poly
			// and them clipping them against the same plane, with one
			// oriented oppositely
			poly1 = opoly;
			poly2 = opoly;
			r2d_clip(&poly1, &splane, 1);
			splane.n.x *= -1;
			splane.n.y *= -1;
			splane.d *= -1;
			r2d_clip(&poly2, &splane, 1);

			// reduce the original and its two parts
			r2d_reduce(&opoly, om, POLY_ORDER);
			r2d_reduce(&poly1, m1, POLY_ORDER);
			r2d_reduce(&poly2, m2, POLY_ORDER);

			// make sure the sum of moments equals the original 
			for(m = 0; m < R2D_NUM_MOMENTS(POLY_ORDER); ++m) {
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
				if(m1[0] > MIN_AREA) {
					polystack[nstack] = poly1;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
				if(m2[0] > MIN_AREA) {
					polystack[nstack] = poly2;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
			}
		}
	}
}

void test_recursive_splitting_degenerate() {

	// recursively splits a polygon (starting from a tet) with a cut plane
	// that is always degenerate with the tet in some way,
	// checking to see that the resulting volumes add up properly.

	// explicit stack-based implementation
	r2d_int nstack, depth, t, chopt, m;
	r2d_poly polystack[STACK_SIZE];
	r2d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r2d_rvec2 verts[3];
	r2d_plane splane;
	r2d_poly opoly, poly1, poly2;
	r2d_real om[R2D_NUM_MOMENTS(POLY_ORDER)], m1[R2D_NUM_MOMENTS(POLY_ORDER)], m2[R2D_NUM_MOMENTS(POLY_ORDER)];

	// do many trials
	printf("Recursively splitting %d triangles, maximum splits per tri is %d.\n", NUM_TRIALS, MAX_DEPTH);
	for(t = 0; t < NUM_TRIALS; ++t) {

		// generate a random tet
		rand_tri_2d(verts, MIN_AREA);
		r2d_init_poly(&opoly, verts, 3);
	
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
			chopt = rand_int(4);
			do {
				splane = choptions_2d[chopt](&opoly);
			} while(splane.n.x == 0.0 && splane.n.y == 0.0);
	
			// split the poly by making two copies of the original poly
			// and them clipping them against the same plane, with one
			// oriented oppositely
			poly1 = opoly;
			poly2 = opoly;
			r2d_clip(&poly1, &splane, 1);
			splane.n.x *= -1;
			splane.n.y *= -1;
			splane.d *= -1;
			r2d_clip(&poly2, &splane, 1);

			// reduce the original and its two parts
			r2d_reduce(&opoly, om, POLY_ORDER);
			r2d_reduce(&poly1, m1, POLY_ORDER);
			r2d_reduce(&poly2, m2, POLY_ORDER);
		
			// make sure the sum of moments equals the original 
			for(m = 0; m < R2D_NUM_MOMENTS(POLY_ORDER); ++m) {
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
				if(m1[0] > MIN_AREA) {
					polystack[nstack] = poly1;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
				if(m2[0] > MIN_AREA) {
					polystack[nstack] = poly2;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
			}
		}
	}
}

void test_recursive_splitting_degenerate_perturbed() {

	// recursively splits a polygon (starting from a tet) with a cut plane
	// that is always degenerate with the tet in some way,
	// checking to see that the resulting volumes add up properly.
	// In this one, the cut plane is perturbed by a small amount.

#define MIN_PERTURB_ORDER (-17)
#define MAX_PERTURB_ORDER (-1)

	// explicit stack-based implementation
	r2d_int nstack, depth, t, chopt, m;
	r2d_poly polystack[STACK_SIZE];
	r2d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r2d_rvec2 verts[3];
	r2d_plane splane;
	r2d_poly opoly, poly1, poly2;
	r2d_real om[R2D_NUM_MOMENTS(POLY_ORDER)], m1[R2D_NUM_MOMENTS(POLY_ORDER)], m2[R2D_NUM_MOMENTS(POLY_ORDER)];
	r2d_real perturb;

	// do many trials
	printf("Recursively splitting %d triangles, maximum splits per tri is %d.\n", NUM_TRIALS, MAX_DEPTH);
	for(t = 0; t < NUM_TRIALS; ++t) {

		// compute the order of magnitude by which to perturb the clip plane,
		// determined by the trial number
		perturb = pow(10, MIN_PERTURB_ORDER + t%(MAX_PERTURB_ORDER - MIN_PERTURB_ORDER));
		//printf("omag = %d, pow = %.10e\n", MIN_PERTURB_ORDER + t%(MAX_PERTURB_ORDER - MIN_PERTURB_ORDER), perturb);

		// generate a random tet
		rand_tri_2d(verts, MIN_AREA);
		r2d_init_poly(&opoly, verts, 3);
	
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
			chopt = rand_int(4);
			do {
				splane = choptions_2d[chopt](&opoly);
			} while(splane.n.x == 0.0 && splane.n.y == 0.0);

			// randomly perturb the plane
			splane.n.x *= 1.0 + perturb*(rand_uniform() - 0.5);
			splane.n.y *= 1.0 + perturb*(rand_uniform() - 0.5);
			splane.d *= 1.0 + perturb*(rand_uniform() - 0.5);
	
			// split the poly by making two copies of the original poly
			// and them clipping them against the same plane, with one
			// oriented oppositely
			poly1 = opoly;
			poly2 = opoly;
			r2d_clip(&poly1, &splane, 1);
			splane.n.x *= -1;
			splane.n.y *= -1;
			splane.d *= -1;
			r2d_clip(&poly2, &splane, 1);

			// reduce the original and its two parts
			r2d_reduce(&opoly, om, POLY_ORDER);
			r2d_reduce(&poly1, m1, POLY_ORDER);
			r2d_reduce(&poly2, m2, POLY_ORDER);
			
			// make sure the sum of moments equals the original 
			for(m = 0; m < R2D_NUM_MOMENTS(POLY_ORDER); ++m) {
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
				if(m1[0] > MIN_AREA) {
					polystack[nstack] = poly1;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
				if(m2[0] > MIN_AREA) {
					polystack[nstack] = poly2;
					depthstack[nstack] = depth + 1;
					++nstack;	
				}
			}
		}
	}
}



void test_tri_tri_timing() {

	// Intersects pairs of tetrahedra, for timing purposes only.

#undef NUM_TRIALS
#define NUM_TRIALS 100000

	// variables: the polyhedra and their moments
	r2d_poly poly; 
	r2d_rvec2 verts[3];
	r2d_plane faces[3];
	r2d_real om[R2D_NUM_MOMENTS(POLY_ORDER)];

	// for timing only, no assertions
	r2d_int trial;
	printf("Intersecting %d pairs of triangles.\n", NUM_TRIALS);
	for(trial = 0; trial < NUM_TRIALS; ++trial) {

		// generate the first random tet
		rand_tri_2d(verts, MIN_AREA);
		r2d_init_poly(&poly, verts, 3);

		// generate the second random tet
		rand_tri_2d(verts, MIN_AREA);
		r2d_poly_faces_from_verts(faces, verts, 3);

		// clip the first tet against the faces of the second
		r2d_clip(&poly, faces, 3);

		// find the moments (up to quadratic order) of the clipped poly
		r2d_reduce(&poly, om, POLY_ORDER);

	}
}


void test_random_verts() {

	// recursively splits a polygon with randomly generated verts with a cut plane
	// that is always degenerate with the poly in some way,
	// checking to see that the resulting volumes add up properly.
	
#define NVERTS 20

	r2d_int v, m; 

	// explicit stack-based implementation
	r2d_int nstack, depth, chopt;
	r2d_poly polystack[STACK_SIZE];
	r2d_int depthstack[STACK_SIZE];

	// variables: the polyhedra and their moments
	r2d_plane splane;
	r2d_rvec2 verts[NVERTS];
	r2d_poly opoly, poly1, poly2;
	r2d_real om[R2D_NUM_MOMENTS(POLY_ORDER)], m1[R2D_NUM_MOMENTS(POLY_ORDER)], m2[R2D_NUM_MOMENTS(POLY_ORDER)];
	
	// randomly generate vertices
	// Note: This will be nasty, tangled, and nonconvex
	for(v = 0; v < NVERTS; ++v)
		verts[v] = rand_uvec_2d();

	// initialize a general polygon
	r2d_init_poly(&opoly, verts, NVERTS);

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
		chopt = rand_int(4);
		do {
			splane = choptions_2d[chopt](&opoly);
		} while(splane.n.x == 0.0 && splane.n.y == 0.0);
		//splane = thru_cent_2d(&opoly);

		// split the poly by making two copies of the original poly
		// and them clipping them against the same plane, with one
		// oriented oppositely
		poly1 = opoly;
		poly2 = opoly;
		r2d_clip(&poly1, &splane, 1);
		splane.n.x *= -1;
		splane.n.y *= -1;
		splane.d *= -1;
		r2d_clip(&poly2, &splane, 1);

		// reduce the original and its two parts
		r2d_reduce(&opoly, om, POLY_ORDER);
		r2d_reduce(&poly1, m1, POLY_ORDER);
		r2d_reduce(&poly2, m2, POLY_ORDER);
	
		// make sure the sum of moments equals the original 
		for(m = 0; m < R2D_NUM_MOMENTS(POLY_ORDER); ++m) {
			ASSERT_EQ(om[m], m1[m] + m2[m], TOL_FAIL);
			EXPECT_EQ(om[m], m1[m] + m2[m], TOL_WARN);
		}
	
		// make sure neither of the two resulting volumes is larger than the original
		// (within some tolerance)
		// Note: Not here! We allow negative volumes and moments for inside-out polygons...
		//ASSERT_LT(m1[0], om[0]*(1.0 + TOL_FAIL));
		//EXPECT_LT(m1[0], om[0]*(1.0 + TOL_WARN));
		//ASSERT_LT(m2[0], om[0]*(1.0 + TOL_FAIL));
		//EXPECT_LT(m2[0], om[0]*(1.0 + TOL_WARN));

		//printf("nstack = %d, depth = %d, opoly = %.10e, p1 = %.10e, p2 = %.10e, err = %.10e\n", 
				//nstack, depth, om[0], m1[0], m2[0], fabs(1.0 - om[0]/(m1[0] + m2[0])));

		// push the children to the stack if they have
		// an acceptably large volume
		if(depth < MAX_DEPTH) {
			if(fabs(m1[0]) > MIN_AREA) {
				polystack[nstack] = poly1;
				depthstack[nstack] = depth + 1;
				++nstack;	
			}
			if(fabs(m2[0]) > MIN_AREA) {
				polystack[nstack] = poly2;
				depthstack[nstack] = depth + 1;
				++nstack;	
			}
		}
	}
}

void test_rasterization() {

	// Test r2d_rasterize() by checking that the rasterized moments
	// do indeed sum to those of the original input

#undef POLY_ORDER
#define POLY_ORDER 3
#define NGRID 31 

	// vars
	r2d_int i, v, curorder, mind;
	r2d_long gg; 
	r2d_int nmom = R2D_NUM_MOMENTS(POLY_ORDER);
	r2d_real voxsum, tmom[nmom];
	r2d_poly poly;
	r2d_rvec2 verts[4];

	// create a random tet in the unit box
	rand_tri_2d(verts, MIN_AREA);
	for(v = 0; v < 4; ++v)
	for(i = 0; i < 2; ++i) {
		verts[v].xy[i] += 1.0;
		verts[v].xy[i] *= 0.5;
	}
	r2d_init_poly(&poly, verts, 3);

	// get its original moments for reference
	r2d_reduce(&poly, tmom, POLY_ORDER);

	// rasterize it
	r2d_rvec2 dx = {{1.0/NGRID, 1.0/NGRID}};
	r2d_dvec2 ibox[2];
	r2d_get_ibox(&poly, ibox, dx);
	printf("Rasterizing a triangle to a grid with dx = %f %f and moments of order %d\n", dx.x, dx.y, POLY_ORDER);
	printf("Minimum index box = %d %d to %d %d\n", ibox[0].i, ibox[0].j, ibox[1].i, ibox[1].j);
	r2d_int npix = (ibox[1].i-ibox[0].i)*(ibox[1].j-ibox[0].j);
	r2d_real* grid = (r2d_real *) calloc(npix*nmom, sizeof(r2d_real));
	r2d_rasterize(&poly, ibox, grid, dx, POLY_ORDER);

	// print out an ASCII check of the rasterization
	r2d_int ii, jj;
	r2d_int ni = ibox[1].i-ibox[0].i;
	r2d_int nj = ibox[1].j-ibox[0].j;
	for(ii = 0; ii < ni; ++ii) {
		for(jj = 0; jj < nj; ++jj) {
			if(grid[nmom*(nj*ii+jj)+0] > 0.8*dx.x*dx.y)
				printf("X");
			else if(grid[nmom*(nj*ii+jj)+0] > 0.0)
				printf(".");
			else
				printf(" ");
		}
		printf("\n");
	}

	
	// make sure the sum of each moment equals the original 
	for(curorder = 0, mind = 0; curorder <= POLY_ORDER; ++curorder) {
		//printf("Order = %d\n", curorder);
		for(i = curorder; i >= 0; --i, ++mind) {
			//j = curorder - i;
			voxsum = 0.0;
			for(gg = 0; gg < npix; ++gg) voxsum += grid[nmom*gg+mind];
			//printf(" Int[ x^%d y^%d dV ] original = %.10e, voxsum = %.10e, error = %.10e\n", 
					//i, j, tmom[mind], voxsum, fabs(1.0 - tmom[mind]/voxsum));
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
#define POLY_ORDER 20
	
	r2d_int i, j, mind, curorder;
	r2d_poly poly;
	r2d_rvec2 box[2];
	r2d_real moments[R2D_NUM_MOMENTS(POLY_ORDER)];
	r2d_real exact;

	// initialize the box
  	box[0].x = -1.0;
  	box[0].y = 1.0;
  	box[1].x = 1.5;
  	box[1].y = 2.0;
	r2d_init_box(&poly, box);

	// get the moments from r2d_reduce
	r2d_reduce(&poly, moments, POLY_ORDER);

	// Check all the moments against the analytic solution 
	// NOTE: This is the order that r2d_reduce puts out! 
	for(curorder = 0, mind = 0; curorder <= POLY_ORDER; ++curorder) {
		//printf("Order = %d\n", curorder);
		for(i = curorder; i >= 0; --i, ++mind) {
			j = curorder - i;
			exact = 1.0/((i+1)*(j+1))*(pow(box[1].x, i+1) - pow(box[0].x, i+1))
				*(pow(box[1].y, j+1) - pow(box[0].y, j+1));
			//printf(" Int[ x^%d y^%d dA ] = %.10e, analytic = %.10e, frac = %f, error = %.10e\n", 
					//i, j, moments[mind], exact, moments[mind]/exact, fabs(1.0 - moments[mind]/exact));
			ASSERT_EQ(moments[mind], exact, TOL_FAIL);
			EXPECT_EQ(moments[mind], exact, TOL_WARN);
		}
	}
}

// -- user-implemented functions declared in utest.h -- //

void register_all_tests() {

	register_test(test_split_tris_thru_centroid, "split_tris_thru_centroid");
	register_test(test_recursive_splitting_nondegenerate, "recursive_splitting_nondegenerate");
	register_test(test_recursive_splitting_degenerate, "recursive_splitting_degenerate");
	register_test(test_recursive_splitting_degenerate_perturbed, "recursive_splitting_degenerate_perturbed");
	register_test(test_tri_tri_timing, "test_tri_tri_timing");
	register_test(test_random_verts, "random_verts");
	register_test(test_rasterization, "rasterization");
	register_test(test_moments, "moments");

}

void setup() {

	// random number seed
	srand(time(NULL));
	//srand(10291989);
	setbuf(stdout, NULL);

	// initialize random clip plane options
	choptions_2d[0] = thru_cent_2d;
	choptions_2d[1] = thru_edge_2d;
	choptions_2d[2] = thru_vert_cent_2d;
	choptions_2d[3] = thru_vert_rand_2d;
}

/////////////////////////////////////////////////////////
