/*
 *  
 *
 *  	r3d.c
 *
 *  	Devon Powell
 *  	8 February 2015
 *
 *  	See Readme.md and r3d.h for usage.
 * 
 *  	Copyright (C) 2014 Stanford University.
 *  	See License.txt for more information.
 *
 *
 */

#include "r3d.h"

#include <string.h>
#include <math.h>

/**
 *  useful macros for r3d
 */

#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667

// tells us which bit signals a clipped vertex
// the last one - leaving seven for flagging faces
#define CLIP_MASK 0x80

// macros for vector manipulation
#define dot(va, vb) (va.x*vb.x + va.y*vb.y + va.z*vb.z)
#define wav(va, wa, vb, wb, vr) {			\
	vr.x = (wa*va.x + wb*vb.x)/(wa + wb);	\
	vr.y = (wa*va.y + wb*vb.y)/(wa + wb);	\
	vr.z = (wa*va.z + wb*vb.z)/(wa + wb);	\
}
#define norm(v) {					\
	r3d_real tmplen = sqrt(dot(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
	v.z /= (tmplen + 1.0e-299);		\
}

/**
 * r3d
 */

#ifdef USE_TREE
// tree node for recursive splitting
typedef struct {
	r3d_int imin, jmin, kmin;
	r3d_int ioff, joff, koff;
	r3d_orientation orient[8];
} r3d_treenode;
#endif // USE_TREE 

void r3d_voxelize_tet(r3d_plane* faces, r3d_dest_grid* grid) {

	// var declarations
	r3d_real moments[10];
	r3d_real locvol, gor;
	r3d_real xmin, ymin, zmin;
	r3d_rvec3 gpt;
	r3d_int i, j, k, ii, jj, kk, m, mmax;
	unsigned char v, f;
	unsigned char orcmp, andcmp;

	// local access to common grid parameters
	r3d_int polyorder = grid->polyorder;
	r3d_dvec3 n = grid->n;
	r3d_rvec3 d = grid->d;

// macros for grid access
#define gind(ii, jj, kk) ((n.j+1)*(n.k+1)*(ii) + (n.k+1)*(jj) + (kk))
#define vind(ii, jj, kk) (n.j*n.k*(ii) + n.k*(jj) + (kk))

// tree/storage buffers, as needed
#ifdef USE_TREE
	r3d_treenode treestack[128];
	r3d_int ntreestack;
	r3d_treenode curnode;
#else
	r3d_long vv0;
	r3d_long vv[8];
	r3d_orientation* orient = grid->orient;
#endif

	// get the high moment index 
	mmax = r3d_num_moments[polyorder];

	// zero the moments 
	for(m = 0; m < mmax; ++m) 
		memset((void*) grid->moments[m], 0, n.i*n.j*n.k*sizeof(r3d_real));

	// voxel bounds in shifted coordinates
	// TODO: make a template voxel and stick with it
	r3d_rvec3 rbounds[2] = {
		{-0.5*d.x, -0.5*d.y, -0.5*d.z}, 
		{0.5*d.x, 0.5*d.y, 0.5*d.z} 
	};

	// the voxel polyhedron
	r3d_poly voxel;

#ifndef USE_TREE // !USE_TREE

	// check all grid vertices in the patch against each tet face
	for(i = 0; i <= n.i; ++i)
	for(j = 0; j <= n.j; ++j)
	for(k = 0; k <= n.k; ++k) {
		// flag the vertex for each face it lies inside
		// also save its distance from each face
		gpt.x = i*d.x; gpt.y = j*d.y; gpt.z = k*d.z;
		vv0 = gind(i, j, k);
		orient[vv0].fflags = 0x00;
		for(f = 0; f < 4; ++f) {
			gor = faces[f].d + dot(gpt, faces[f].n);
			if(gor > 0.0) orient[vv0].fflags |= (1 << f);
			orient[vv0].fdist[f] = gor;
		}
	}

	// iterate over all voxels in the patch
	for(i = 0; i < n.i; ++i)
	for(j = 0; j < n.j; ++j)
	for(k = 0; k < n.k; ++k) {

		// precompute flattened grid indices
		for(ii = 0; ii < 2; ++ii)
		for(jj = 0; jj < 2; ++jj)
		for(kk = 0; kk < 2; ++kk)
			vv[4*ii+2*jj+kk] = gind(i+ii, j+jj, k+kk);
	
		// check inclusion of each voxel within the tet
		orcmp = 0x00;
        andcmp = 0x0f;
		for(v = 0; v < 8; ++v) {
			orcmp |= orient[vv[v]].fflags; 
			andcmp &= orient[vv[v]].fflags; 
		}

		// if the voxel is entirely inside the tet
		if(andcmp == 0x0f) {

			// calculate moments
			// TODO: recursively, somehow...
			locvol = d.x*d.y*d.z;
			moments[0] = locvol; // 1
			if(polyorder >= 1) {
				moments[1] = locvol*d.x*(i + 0.5); // x
				moments[2] = locvol*d.y*(j + 0.5); // y
				moments[3] = locvol*d.z*(k + 0.5); // z
			}
			if(polyorder >= 2) {
				moments[4] = locvol*ONE_THIRD*d.x*d.x*(1 + 3*i + 3*i*i); // x*x
				moments[5] = locvol*ONE_THIRD*d.y*d.y*(1 + 3*j + 3*j*j); // y*y
				moments[6] = locvol*ONE_THIRD*d.z*d.z*(1 + 3*k + 3*k*k); // z*z
				moments[7] = locvol*0.25*d.x*d.y*(1 + 2*i)*(1 + 2*j); // x*y
				moments[8] = locvol*0.25*d.y*d.z*(1 + 2*j)*(1 + 2*k); // y*z
				moments[9] = locvol*0.25*d.x*d.z*(1 + 2*i)*(1 + 2*k); // z*x
			}

			//reduce to main grid
			for(m = 0; m < mmax; ++m) 
				grid->moments[m][vind(i, j, k)] = moments[m]; 

			continue;
		}	

		// if the voxel crosses the boundary of the tet
		if(orcmp == 0x0f) {

			// initialize the unit cube connectivity 
			// TODO: make this a simple copy operation?
			r3du_init_box(&voxel, rbounds);
			for(v = 0; v < 8; ++v)
				voxel.verts[v].orient = orient[vv[v]];

			// clip and reduce the voxel
			r3d_clip_tet(&voxel, andcmp);
			r3d_reduce(&voxel, polyorder, moments);

			// the cross-terms arising from using an offset box
			// must be taken into account in the absolute moment integrals
			// TODO: recursively, somehow...
			if(polyorder >= 1) {
				xmin = (i + 0.5)*d.x;
				ymin = (j + 0.5)*d.y;
				zmin = (k + 0.5)*d.z;
			}
			if(polyorder >= 2) {
				moments[4] += 2.0*xmin*moments[1] + xmin*xmin*moments[0];
				moments[5] += 2.0*ymin*moments[2] + ymin*ymin*moments[0];
				moments[6] += 2.0*zmin*moments[3] + zmin*zmin*moments[0];
				moments[7] += xmin*moments[2] + ymin*moments[1] + xmin*ymin*moments[0];
				moments[8] += ymin*moments[3] + zmin*moments[2] + ymin*zmin*moments[0];
				moments[9] += xmin*moments[3] + zmin*moments[1] + xmin*zmin*moments[0];
			}
			if(polyorder >= 1) {
				moments[1] += xmin*moments[0];
				moments[2] += ymin*moments[0];
				moments[3] += zmin*moments[0];
			}

			// reduce to main grid
			for(m = 0; m < mmax; ++m) 
				grid->moments[m][vind(i, j, k)] = moments[m];

			continue;	
		}
	}

#else // USE_TREE
	// TODO: collapse this down??
	
	// get the initial face orientations for each corner of the node
	gpt.x = 0.0;
	gpt.y = 0.0;
	gpt.z = 0.0;
	curnode.orient[0].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[0].fflags |= (1 << f);
		curnode.orient[0].fdist[f] = gor;
	}
	gpt.x = n.i*d.x;
	curnode.orient[1].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[1].fflags |= (1 << f);
		curnode.orient[1].fdist[f] = gor;
	}
	gpt.y = n.j*d.y;
	curnode.orient[2].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[2].fflags |= (1 << f);
		curnode.orient[2].fdist[f] = gor;
	}
	gpt.x = 0.0;
	curnode.orient[3].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[3].fflags |= (1 << f);
		curnode.orient[3].fdist[f] = gor;
	}
	gpt.y = 0.0;
	gpt.z = n.k*d.z;
	curnode.orient[4].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[4].fflags |= (1 << f);
		curnode.orient[4].fdist[f] = gor;
	}
	gpt.x = n.i*d.x;
	curnode.orient[5].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[5].fflags |= (1 << f);
		curnode.orient[5].fdist[f] = gor;
	}
	gpt.y = n.j*d.y;
	curnode.orient[6].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[6].fflags |= (1 << f);
		curnode.orient[6].fdist[f] = gor;
	}
	gpt.x = 0.0;
	curnode.orient[7].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[7].fflags |= (1 << f);
		curnode.orient[7].fdist[f] = gor;
	}
	curnode.imin = 0;
	curnode.jmin = 0;
	curnode.kmin = 0;
	curnode.ioff = n.i;
	curnode.joff = n.j;
	curnode.koff = n.k;

	ntreestack = 0;
	treestack[ntreestack++] = curnode;

	while(ntreestack > 0) {

		// pop the top node
		curnode = treestack[--ntreestack];

		orcmp = 0x00;
        andcmp = 0x0f;
		for(v = 0; v < 8; ++v) {
			orcmp |= curnode.orient[v].fflags; 
			andcmp &= curnode.orient[v].fflags; 
		}

		if(andcmp == 0x0f) {
			// all cells in this leaf are fully contained

			locvol = d.x*d.y*d.z;
			for(i = curnode.imin; i < curnode.imin + curnode.ioff; ++i)
			for(j = curnode.jmin; j < curnode.jmin + curnode.joff; ++j)
			for(k = curnode.kmin; k < curnode.kmin + curnode.koff; ++k) {

				moments[0] = locvol; // 1
				
				if(polyorder >= 1) {
					moments[1] = locvol*d.x*(i + 0.5); // x
					moments[2] = locvol*d.y*(j + 0.5); // y
					moments[3] = locvol*d.z*(k + 0.5); // z
				}
				if(polyorder >= 2) {
					moments[4] = locvol*ONE_THIRD*d.x*d.x*(1 + 3*i + 3*i*i); // x*x
					moments[5] = locvol*ONE_THIRD*d.y*d.y*(1 + 3*j + 3*j*j); // y*y
					moments[6] = locvol*ONE_THIRD*d.z*d.z*(1 + 3*k + 3*k*k); // z*z
					moments[7] = locvol*0.25*d.x*d.y*(1 + 2*i)*(1 + 2*j); // x*y
					moments[8] = locvol*0.25*d.y*d.z*(1 + 2*j)*(1 + 2*k); // y*z
					moments[9] = locvol*0.25*d.x*d.z*(1 + 2*i)*(1 + 2*k); // z*x
				}

				//reduce to main grid
				for(m = 0; m < mmax; ++m) 
					grid->moments[m][vind(i, j, k)] = moments[m]; 

			}
			continue;
		}
		if(orcmp != 0x0f) {

			// the leaf lies entirely outside the tet
			// skip it
			continue;
		}
		if(curnode.ioff == 1 && curnode.joff == 1 && curnode.koff == 1) {

			// we've reached a single cell that straddles the tet boundary 
			// Clip and voxelize it.
			i = curnode.imin;
			j = curnode.jmin;
			k = curnode.kmin;

			// initialize the unit cube connectivity 
			r3du_init_box(&voxel, rbounds);
			for(v = 0; v < voxel.nverts; ++v)
				voxel.verts[v].orient = curnode.orient[v];

			// clip and reduce
			r3d_clip_tet(&voxel, andcmp);
			r3d_reduce(&voxel, polyorder, moments);

			// the cross-terms arising from using an offset box
			// must be taken into account in the absolute moment integrals
			if(polyorder >= 1) {
				xmin = (i + 0.5)*d.x;
				ymin = (j + 0.5)*d.y;
				zmin = (k + 0.5)*d.z;
			}
			if(polyorder >= 2) {
				moments[4] += 2.0*xmin*moments[1] + xmin*xmin*moments[0];
				moments[5] += 2.0*ymin*moments[2] + ymin*ymin*moments[0];
				moments[6] += 2.0*zmin*moments[3] + zmin*zmin*moments[0];
				moments[7] += xmin*moments[2] + ymin*moments[1] + xmin*ymin*moments[0];
				moments[8] += ymin*moments[3] + zmin*moments[2] + ymin*zmin*moments[0];
				moments[9] += xmin*moments[3] + zmin*moments[1] + xmin*zmin*moments[0];
			}
			if(polyorder >= 1) {
				moments[1] += xmin*moments[0];
				moments[2] += ymin*moments[0];
				moments[3] += zmin*moments[0];
			}

			// reduce to main grid
			for(m = 0; m < mmax; ++m) 
				grid->moments[m][vind(i, j, k)] = moments[m];

			continue;	
		}

		// else, split the node along its longest dimension
		// and push to the the stack
		i = curnode.ioff/2;
		j = curnode.joff/2;
		k = curnode.koff/2;
		if(i >= j && i >= k) {

			// LEFT NODE
			treestack[ntreestack].imin = curnode.imin;
			treestack[ntreestack].ioff = i;
			treestack[ntreestack].jmin = curnode.jmin;
			treestack[ntreestack].joff = curnode.joff;
			treestack[ntreestack].kmin = curnode.kmin;
			treestack[ntreestack].koff = curnode.koff;
			treestack[ntreestack].orient[0] = curnode.orient[0];
			treestack[ntreestack].orient[3] = curnode.orient[3];
			treestack[ntreestack].orient[4] = curnode.orient[4];
			treestack[ntreestack].orient[7] = curnode.orient[7];

			// RIGHT NODE
			treestack[ntreestack+1].imin = curnode.imin + i;
			treestack[ntreestack+1].ioff = curnode.ioff - i;
			treestack[ntreestack+1].jmin = curnode.jmin;
			treestack[ntreestack+1].joff = curnode.joff;
			treestack[ntreestack+1].kmin = curnode.kmin;
			treestack[ntreestack+1].koff = curnode.koff;
			treestack[ntreestack+1].orient[1] = curnode.orient[1];
			treestack[ntreestack+1].orient[2] = curnode.orient[2];
			treestack[ntreestack+1].orient[5] = curnode.orient[5];
			treestack[ntreestack+1].orient[6] = curnode.orient[6];

			// FILL IN COMMON POINTS
			gpt.x = d.x*(curnode.imin + i);
			gpt.y = d.y*curnode.jmin;
			gpt.z = d.z*curnode.kmin;
			treestack[ntreestack].orient[1].fflags = 0x00;
			treestack[ntreestack+1].orient[0].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[1].fdist[f] = gor;
				treestack[ntreestack+1].orient[0].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[1].fflags |= (1 << f);
					treestack[ntreestack+1].orient[0].fflags |= (1 << f);
				}
			}
			gpt.y = d.y*(curnode.jmin + curnode.joff);
			treestack[ntreestack].orient[2].fflags = 0x00;
			treestack[ntreestack+1].orient[3].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[2].fdist[f] = gor;
				treestack[ntreestack+1].orient[3].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[2].fflags |= (1 << f);
					treestack[ntreestack+1].orient[3].fflags |= (1 << f);
				}
			}
			gpt.z = d.z*(curnode.kmin + curnode.koff);
			treestack[ntreestack].orient[6].fflags = 0x00;
			treestack[ntreestack+1].orient[7].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[6].fdist[f] = gor;
				treestack[ntreestack+1].orient[7].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[6].fflags |= (1 << f);
					treestack[ntreestack+1].orient[7].fflags |= (1 << f);
				}
			}
			gpt.y = d.y*curnode.jmin;
			treestack[ntreestack].orient[5].fflags = 0x00;
			treestack[ntreestack+1].orient[4].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[5].fdist[f] = gor;
				treestack[ntreestack+1].orient[4].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[5].fflags |= (1 << f);
					treestack[ntreestack+1].orient[4].fflags |= (1 << f);
				}
			}
			ntreestack += 2;
			continue;
		}
		if(j >= i && j >= k) {
			// LEFT NODE
			treestack[ntreestack].imin = curnode.imin;
			treestack[ntreestack].ioff = curnode.ioff;
			treestack[ntreestack].jmin = curnode.jmin;
			treestack[ntreestack].joff = j;
			treestack[ntreestack].kmin = curnode.kmin;
			treestack[ntreestack].koff = curnode.koff;
			treestack[ntreestack].orient[0] = curnode.orient[0];
			treestack[ntreestack].orient[1] = curnode.orient[1];
			treestack[ntreestack].orient[4] = curnode.orient[4];
			treestack[ntreestack].orient[5] = curnode.orient[5];

			// RIGHT NODE
			treestack[ntreestack+1].imin = curnode.imin;
			treestack[ntreestack+1].ioff = curnode.ioff;
			treestack[ntreestack+1].jmin = curnode.jmin + j;
			treestack[ntreestack+1].joff = curnode.joff - j;
			treestack[ntreestack+1].kmin = curnode.kmin;
			treestack[ntreestack+1].koff = curnode.koff;
			treestack[ntreestack+1].orient[2] = curnode.orient[2];
			treestack[ntreestack+1].orient[3] = curnode.orient[3];
			treestack[ntreestack+1].orient[6] = curnode.orient[6];
			treestack[ntreestack+1].orient[7] = curnode.orient[7];

			// FILL IN COMMON POINTS
			gpt.x = d.x*curnode.imin;
			gpt.y = d.y*(curnode.jmin + j);
			gpt.z = d.z*curnode.kmin;
			treestack[ntreestack].orient[3].fflags = 0x00;
			treestack[ntreestack+1].orient[0].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[3].fdist[f] = gor;
				treestack[ntreestack+1].orient[0].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[3].fflags |= (1 << f);
					treestack[ntreestack+1].orient[0].fflags |= (1 << f);
				}
			}
			gpt.x = d.x*(curnode.imin + curnode.ioff);
			treestack[ntreestack].orient[2].fflags = 0x00;
			treestack[ntreestack+1].orient[1].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[2].fdist[f] = gor;
				treestack[ntreestack+1].orient[1].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[2].fflags |= (1 << f);
					treestack[ntreestack+1].orient[1].fflags |= (1 << f);
				}
			}
			gpt.z = d.z*(curnode.kmin + curnode.koff);
			treestack[ntreestack].orient[6].fflags = 0x00;
			treestack[ntreestack+1].orient[5].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[6].fdist[f] = gor;
				treestack[ntreestack+1].orient[5].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[6].fflags |= (1 << f);
					treestack[ntreestack+1].orient[5].fflags |= (1 << f);
				}
			}
			gpt.x = d.x*curnode.imin;
			treestack[ntreestack].orient[7].fflags = 0x00;
			treestack[ntreestack+1].orient[4].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[7].fdist[f] = gor;
				treestack[ntreestack+1].orient[4].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[7].fflags |= (1 << f);
					treestack[ntreestack+1].orient[4].fflags |= (1 << f);
				}
			}
			ntreestack += 2;
			continue;
		}
		if(k >= i && k >= j) {

			// LEFT NODE
			treestack[ntreestack].imin = curnode.imin;
			treestack[ntreestack].ioff = curnode.ioff;
			treestack[ntreestack].jmin = curnode.jmin;
			treestack[ntreestack].joff = curnode.joff;
			treestack[ntreestack].kmin = curnode.kmin;
			treestack[ntreestack].koff = k;
			treestack[ntreestack].orient[0] = curnode.orient[0];
			treestack[ntreestack].orient[1] = curnode.orient[1];
			treestack[ntreestack].orient[2] = curnode.orient[2];
			treestack[ntreestack].orient[3] = curnode.orient[3];

			// RIGHT NODE
			treestack[ntreestack+1].imin = curnode.imin;
			treestack[ntreestack+1].ioff = curnode.ioff;
			treestack[ntreestack+1].jmin = curnode.jmin;
			treestack[ntreestack+1].joff = curnode.joff;
			treestack[ntreestack+1].kmin = curnode.kmin + k;
			treestack[ntreestack+1].koff = curnode.koff - k;
			treestack[ntreestack+1].orient[4] = curnode.orient[4];
			treestack[ntreestack+1].orient[5] = curnode.orient[5];
			treestack[ntreestack+1].orient[6] = curnode.orient[6];
			treestack[ntreestack+1].orient[7] = curnode.orient[7];

			// FILL IN COMMON POINTS
			gpt.x = d.x*curnode.imin;
			gpt.y = d.y*curnode.jmin;
			gpt.z = d.z*(curnode.kmin + k);
			treestack[ntreestack].orient[4].fflags = 0x00;
			treestack[ntreestack+1].orient[0].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[4].fdist[f] = gor;
				treestack[ntreestack+1].orient[0].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[4].fflags |= (1 << f);
					treestack[ntreestack+1].orient[0].fflags |= (1 << f);
				}
			}
			gpt.x = d.x*(curnode.imin + curnode.ioff);
			treestack[ntreestack].orient[5].fflags = 0x00;
			treestack[ntreestack+1].orient[1].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[5].fdist[f] = gor;
				treestack[ntreestack+1].orient[1].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[5].fflags |= (1 << f);
					treestack[ntreestack+1].orient[1].fflags |= (1 << f);
				}
			}
			gpt.y = d.y*(curnode.jmin + curnode.joff);
			treestack[ntreestack].orient[6].fflags = 0x00;
			treestack[ntreestack+1].orient[2].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[6].fdist[f] = gor;
				treestack[ntreestack+1].orient[2].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[6].fflags |= (1 << f);
					treestack[ntreestack+1].orient[2].fflags |= (1 << f);
				}
			}
			gpt.x = d.x*curnode.imin;
			treestack[ntreestack].orient[7].fflags = 0x00;
			treestack[ntreestack+1].orient[3].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[7].fdist[f] = gor;
				treestack[ntreestack+1].orient[3].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[7].fflags |= (1 << f);
					treestack[ntreestack+1].orient[3].fflags |= (1 << f);
				}
			}
			ntreestack += 2;
			continue;
		}
	}

#endif // USE_TREE

}

void r3d_clip_tet(r3d_poly* poly, unsigned char andcmp) {

	// TODO: Why is the updated version slower??

	// variable declarations
	unsigned char v, vnext, f, ff, np, vcur, onv;
	unsigned char fmask, ffmask;

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
			
	for(f = 0; f < 4; ++f) {

		fmask = (1 << f);
		if(andcmp & fmask) continue;

		// for each edge that crosses a clip plane, insert a new vertex
		// TODO: Do away with CLIP_MASK
		onv = *nverts;
		for(vcur = 0; vcur < onv; ++vcur) {
			if(vertbuffer[vcur].orient.fflags & CLIP_MASK) continue;
			if(!(vertbuffer[vcur].orient.fflags & fmask)) {
				vertbuffer[vcur].orient.fflags |= CLIP_MASK;
				continue;
			}	
			for(np = 0; np < 3; ++np) {
				vnext = vertbuffer[vcur].pnbrs[np];
				if(vertbuffer[vnext].orient.fflags & fmask) continue;
				wav(vertbuffer[vcur].pos, -vertbuffer[vnext].orient.fdist[f],
					vertbuffer[vnext].pos, vertbuffer[vcur].orient.fdist[f],
					vertbuffer[*nverts].pos);
				vertbuffer[vcur].pnbrs[np] = *nverts;
				vertbuffer[*nverts].pnbrs[0] = vcur;
				vertbuffer[*nverts].orient.fflags = 0x00;
				for(ff = f + 1; ff < 4; ++ff) {
					ffmask = (1 << ff); 
					if(andcmp & ffmask) continue;
					vertbuffer[*nverts].orient.fdist[ff] = 
							(vertbuffer[vnext].orient.fdist[ff]*vertbuffer[vcur].orient.fdist[f] 
							- vertbuffer[vnext].orient.fdist[f]*vertbuffer[vcur].orient.fdist[ff])
							/(vertbuffer[vcur].orient.fdist[f] - vertbuffer[vnext].orient.fdist[f]);
					if(vertbuffer[*nverts].orient.fdist[ff] > 0.0) vertbuffer[*nverts].orient.fflags |= ffmask;
				}
				(*nverts)++;
			}
		}

		// for each new vert, traverse around the graph
		// and insert new edges
		for(v = onv; v < *nverts; ++v) {
			vcur = v;
			vnext = vertbuffer[vcur].pnbrs[0];
			while(vnext < onv) {
				for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
				vcur = vnext;
				vnext = vertbuffer[vcur].pnbrs[(np+1)%3];
			}
			vertbuffer[v].pnbrs[2] = vnext;
			vertbuffer[vnext].pnbrs[1] = v;
		}

		// TODO: compress the vertex list??

	}
}

void r3d_reduce(r3d_poly* poly, r3d_int polyorder, r3d_real* moments) {

	// var declarations
	r3d_real locvol;
	unsigned char v, np, m;
	unsigned char vcur, vnext, pnext, vstart;
	r3d_rvec3 v0, v1, v2; 

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
	
	// for keeping track of which edges have been traversed
	unsigned char emarks[R3D_MAX_VERTS][3];
	memset((void*) &emarks, 0, sizeof(emarks));

	// stack for edges
	r3d_int nvstack;
	unsigned char vstack[2*R3D_MAX_VERTS];

	// zero the moments
	for(m = 0; m < 10; ++m)
		moments[m] = 0.0;

	// find the first unclipped vertex
	vcur = R3D_MAX_VERTS;
	for(v = 0; vcur == R3D_MAX_VERTS && v < *nverts; ++v) 
		if(!(vertbuffer[v].orient.fflags & CLIP_MASK)) vcur = v;
	
	// return if all vertices have been clipped
	if(vcur == R3D_MAX_VERTS) return;

	// stack implementation
	nvstack = 0;
	vstack[nvstack++] = vcur;
	vstack[nvstack++] = 0;

	while(nvstack > 0) {
		
		pnext = vstack[--nvstack];
		vcur = vstack[--nvstack];

		// skip this edge if we have marked it
		if(emarks[vcur][pnext]) continue;

		// initialize face looping
		emarks[vcur][pnext] = 1;
		vstart = vcur;
		v0 = vertbuffer[vstart].pos;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// move to the second edge
		for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
		vcur = vnext;
		pnext = (np+1)%3;
		emarks[vcur][pnext] = 1;
		vnext = vertbuffer[vcur].pnbrs[pnext];
		vstack[nvstack++] = vcur;
		vstack[nvstack++] = (pnext+1)%3;

		// make a triangle fan using edges
		// and first vertex
		while(vnext != vstart) {

			v2 = vertbuffer[vcur].pos;
			v1 = vertbuffer[vnext].pos;

			locvol = ONE_SIXTH*(-(v2.x*v1.y*v0.z) + v1.x*v2.y*v0.z + v2.x*v0.y*v1.z
				   	- v0.x*v2.y*v1.z - v1.x*v0.y*v2.z + v0.x*v1.y*v2.z); 

			moments[0] += locvol; 
			if(polyorder >= 1) {
				moments[1] += locvol*0.25*(v0.x + v1.x + v2.x);
				moments[2] += locvol*0.25*(v0.y + v1.y + v2.y);
				moments[3] += locvol*0.25*(v0.z + v1.z + v2.z);
			}
			if(polyorder >= 2) {
				moments[4] += locvol*0.1*(v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + v1.x*v2.x + v0.x*(v1.x + v2.x));
				moments[5] += locvol*0.1*(v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + v1.y*v2.y + v0.y*(v1.y + v2.y));
				moments[6] += locvol*0.1*(v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + v1.z*v2.z + v0.z*(v1.z + v2.z));
				moments[7] += locvol*0.05*(v2.x*v0.y + v2.x*v1.y + 2*v2.x*v2.y + v0.x*(2*v0.y + v1.y + v2.y) + v1.x*(v0.y + 2*v1.y + v2.y));
				moments[8] += locvol*0.05*(v2.y*v0.z + v2.y*v1.z + 2*v2.y*v2.z + v0.y*(2*v0.z + v1.z + v2.z) + v1.y*(v0.z + 2*v1.z + v2.z));
				moments[9] += locvol*0.05*(v2.x*v0.z + v2.x*v1.z + 2*v2.x*v2.z + v0.x*(2*v0.z + v1.z + v2.z) + v1.x*(v0.z + 2*v1.z + v2.z));
			}

			// move to the next edge
			for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
			vcur = vnext;
			pnext = (np+1)%3;
			emarks[vcur][pnext] = 1;
			vnext = vertbuffer[vcur].pnbrs[pnext];
			vstack[nvstack++] = vcur;
			vstack[nvstack++] = (pnext+1)%3;
		}
	}
}

/*
 * r3du: utility functions for r3d
 */

void r3du_tet_faces_from_verts(r3d_rvec3* verts, r3d_plane* faces) {
	r3d_rvec3 tmpcent;
	faces[0].n.x = ((verts[3].y - verts[1].y)*(verts[2].z - verts[1].z) 
			- (verts[2].y - verts[1].y)*(verts[3].z - verts[1].z));
	faces[0].n.y = ((verts[2].x - verts[1].x)*(verts[3].z - verts[1].z) 
			- (verts[3].x - verts[1].x)*(verts[2].z - verts[1].z));
	faces[0].n.z = ((verts[3].x - verts[1].x)*(verts[2].y - verts[1].y) 
			- (verts[2].x - verts[1].x)*(verts[3].y - verts[1].y));
	norm(faces[0].n);
	tmpcent.x = ONE_THIRD*(verts[1].x + verts[2].x + verts[3].x);
	tmpcent.y = ONE_THIRD*(verts[1].y + verts[2].y + verts[3].y);
	tmpcent.z = ONE_THIRD*(verts[1].z + verts[2].z + verts[3].z);
	faces[0].d = -dot(faces[0].n, tmpcent);

	faces[1].n.x = ((verts[2].y - verts[0].y)*(verts[3].z - verts[2].z) 
			- (verts[2].y - verts[3].y)*(verts[0].z - verts[2].z));
	faces[1].n.y = ((verts[3].x - verts[2].x)*(verts[2].z - verts[0].z) 
			- (verts[0].x - verts[2].x)*(verts[2].z - verts[3].z));
	faces[1].n.z = ((verts[2].x - verts[0].x)*(verts[3].y - verts[2].y) 
			- (verts[2].x - verts[3].x)*(verts[0].y - verts[2].y));
	norm(faces[1].n);
	tmpcent.x = ONE_THIRD*(verts[2].x + verts[3].x + verts[0].x);
	tmpcent.y = ONE_THIRD*(verts[2].y + verts[3].y + verts[0].y);
	tmpcent.z = ONE_THIRD*(verts[2].z + verts[3].z + verts[0].z);
	faces[1].d = -dot(faces[1].n, tmpcent);

	faces[2].n.x = ((verts[1].y - verts[3].y)*(verts[0].z - verts[3].z) 
			- (verts[0].y - verts[3].y)*(verts[1].z - verts[3].z));
	faces[2].n.y = ((verts[0].x - verts[3].x)*(verts[1].z - verts[3].z) 
			- (verts[1].x - verts[3].x)*(verts[0].z - verts[3].z));
	faces[2].n.z = ((verts[1].x - verts[3].x)*(verts[0].y - verts[3].y) 
			- (verts[0].x - verts[3].x)*(verts[1].y - verts[3].y));
	norm(faces[2].n);
	tmpcent.x = ONE_THIRD*(verts[3].x + verts[0].x + verts[1].x);
	tmpcent.y = ONE_THIRD*(verts[3].y + verts[0].y + verts[1].y);
	tmpcent.z = ONE_THIRD*(verts[3].z + verts[0].z + verts[1].z);
	faces[2].d = -dot(faces[2].n, tmpcent);

	faces[3].n.x = ((verts[0].y - verts[2].y)*(verts[1].z - verts[0].z) 
			- (verts[0].y - verts[1].y)*(verts[2].z - verts[0].z));
	faces[3].n.y = ((verts[1].x - verts[0].x)*(verts[0].z - verts[2].z) 
			- (verts[2].x - verts[0].x)*(verts[0].z - verts[1].z));
	faces[3].n.z = ((verts[0].x - verts[2].x)*(verts[1].y - verts[0].y) 
			- (verts[0].x - verts[1].x)*(verts[2].y - verts[0].y));
	norm(faces[3].n);
	tmpcent.x = ONE_THIRD*(verts[0].x + verts[1].x + verts[2].x);
	tmpcent.y = ONE_THIRD*(verts[0].y + verts[1].y + verts[2].y);
	tmpcent.z = ONE_THIRD*(verts[0].z + verts[1].z + verts[2].z);
	faces[3].d = -dot(faces[3].n, tmpcent);
}

void r3du_init_box(r3d_poly* poly, r3d_rvec3 rbounds[2]) {

	// direct access to vertex buffer
	r3d_vertex* vertbuffer = poly->verts; 
	r3d_int* nverts = &poly->nverts; 
	
	// vertices are in row-major order
	// copy locations
	r3d_int i, j, k; 
	for(i = 0; i < 2; ++i)
	for(j = 0; j < 2; ++j)
	for(k = 0; k < 2; ++k) {
		vertbuffer[4*i+2*j+k].pos.x = rbounds[i].x;
		vertbuffer[4*i+2*j+k].pos.y = rbounds[j].y;
		vertbuffer[4*i+2*j+k].pos.z = rbounds[k].z;
	}

	// fill out vertex connectivity explicitly
	vertbuffer[0].pnbrs[0] = 4;	
	vertbuffer[0].pnbrs[1] = 1;	
	vertbuffer[0].pnbrs[2] = 2;	
	vertbuffer[1].pnbrs[0] = 5;	
	vertbuffer[1].pnbrs[1] = 3;	
	vertbuffer[1].pnbrs[2] = 0;	
	vertbuffer[2].pnbrs[0] = 6;	
	vertbuffer[2].pnbrs[1] = 0;	
	vertbuffer[2].pnbrs[2] = 3;	
	vertbuffer[3].pnbrs[0] = 7;	
	vertbuffer[3].pnbrs[1] = 2;	
	vertbuffer[3].pnbrs[2] = 1;	
	vertbuffer[4].pnbrs[0] = 0;	
	vertbuffer[4].pnbrs[1] = 6;	
	vertbuffer[4].pnbrs[2] = 5;	
	vertbuffer[5].pnbrs[0] = 1;	
	vertbuffer[5].pnbrs[1] = 4;	
	vertbuffer[5].pnbrs[2] = 7;	
	vertbuffer[6].pnbrs[0] = 2;	
	vertbuffer[6].pnbrs[1] = 7;	
	vertbuffer[6].pnbrs[2] = 4;	
	vertbuffer[7].pnbrs[0] = 3;	
	vertbuffer[7].pnbrs[1] = 5;	
	vertbuffer[7].pnbrs[2] = 6;	
	*nverts = 8;
}

r3d_real r3du_orient(r3d_rvec3 pa, r3d_rvec3 pb, r3d_rvec3 pc, r3d_rvec3 pd) {
	r3d_real adx, bdx, cdx;
	r3d_real ady, bdy, cdy;
	r3d_real adz, bdz, cdz;
	adx = pa.x - pd.x;
	bdx = pb.x - pd.x;
	cdx = pc.x - pd.x;
	ady = pa.y - pd.y;
	bdy = pb.y - pd.y;
	cdy = pc.y - pd.y;
	adz = pa.z - pd.z;
	bdz = pb.z - pd.z;
	cdz = pc.z - pd.z;
	return -ONE_SIXTH*(adx * (bdy * cdz - bdz * cdy)
			+ bdx * (cdy * adz - cdz * ady)
			+ cdx * (ady * bdz - adz * bdy));
}

