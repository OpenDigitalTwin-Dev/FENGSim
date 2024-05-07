/*
 *  
 *
 *  	r2d.c
 *
 *  	Devon Powell
 *  	8 February 2015
 *
 *  	See Readme.md and r2d.h for usage.
 * 
 *  	Copyright (C) 2014 Stanford University.
 *  	See License.txt for more information.
 *
 *
 */

#include "r2d.h"

#include <string.h>
#include <math.h>

/**
 *  useful macros for r2d
 */

#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
#define ONE_TWLFTH 0.0833333333333333333333333333333333333333333333333333

// tells us which bit signals a clipped vertex
// the last one - leaving seven to flag faces
#define CLIP_MASK 0x80 

// macros for vector manipulation
#define dot(va, vb) (va.x*vb.x + va.y*vb.y)
#define wav(va, wa, vb, wb, vr) {			\
	vr.x = (wa*va.x + wb*vb.x)/(wa + wb);	\
	vr.y = (wa*va.y + wb*vb.y)/(wa + wb);	\
}
#define norm(v) {					\
	r2d_real tmplen = sqrt(dot(v, v));	\
	v.x /= (tmplen + 1.0e-299);		\
	v.y /= (tmplen + 1.0e-299);		\
}

/**
 * r2d
 */

#ifdef USE_TREE
// tree node for recursive splitting
typedef struct {
	r2d_int imin, jmin;
	r2d_int ioff, joff;
	r2d_orientation orient[4];
} r2d_treenode;
#endif // USE_TREE 

void r2d_rasterize_quad(r2d_plane* faces, r2d_dest_grid* grid) {

	// variables used in this function
	r2d_real moments[6];
	r2d_real gor, locvol, xmin, ymin;
	r2d_rvec2 gpt;
	r2d_int i, j, m, mmax;
	unsigned char v, f;
	unsigned char orcmp, andcmp;

	// local access to common grid parameters
	r2d_int polyorder = grid->polyorder;
	r2d_dvec2 n = grid->n;
	r2d_rvec2 d = grid->d;

// macros for grid access
#define gind(ii, jj) ((n.j+1)*(ii) + (jj))
#define vind(ii, jj) (n.j*(ii) + (jj))

// macros for grid access
#ifdef USE_TREE
	// stack for the tree
	r2d_treenode treestack[128];
	r2d_int ntreestack;
	r2d_treenode curnode;
#else
	r2d_long vv[4];
	r2d_long vv0;
	r2d_orientation* orient = grid->orient;
#endif

	// get the high moment index 
	mmax = r2d_num_moments[polyorder];

	// zero the moments 
	for(m = 0; m < mmax; ++m) 
		memset((void*) grid->moments[m], 0, n.i*n.j*sizeof(r2d_real));


	// voxel bounds in shifted coordinates
	r2d_rvec2 rbounds[2] = {
		{-0.5*d.x, -0.5*d.y}, {0.5*d.x, 0.5*d.y}
	};

	// pixel polygon
	r2d_poly pixel;

#ifndef USE_TREE // !USE_TREE

	// check all grid vertices in the patch against each tet face
	for(i = 0; i <= n.i; ++i)
	for(j = 0; j <= n.j; ++j) {
		// flag the vertex for each face it lies inside
		// also save its distance from each face
		gpt.x = i*d.x; gpt.y = j*d.y;
		vv0 = gind(i, j);
		orient[vv0].fflags = 0x00;
		for(f = 0; f < 4; ++f) {
			gor = faces[f].d + dot(gpt, faces[f].n);
			if(gor > 0.0) orient[vv0].fflags |= (1 << f);
			orient[vv0].fdist[f] = gor;
		}
	}

	// iterate over all voxels in the patch
	for(i = 0; i < n.i; ++i)
	for(j = 0; j < n.j; ++j) {

		// precompute flattened grid indices
		vv[0] = gind(i+1, j);
		vv[1] = gind(i+1, j+1);
		vv[2] = gind(i, j+1);
		vv[3] = gind(i, j);
	
		// check inclusion of each voxel within the tet
		orcmp = 0x00;
        andcmp = 0x0f;
		for(v = 0; v < 4; ++v) {
			orcmp |= orient[vv[v]].fflags; 
			andcmp &= orient[vv[v]].fflags; 
		}

		if(andcmp == 0x0f) {
			// the voxel is entirely inside the quad
#ifndef NO_REDUCTION
			locvol = d.x*d.y;
			moments[0] = locvol;
			if(polyorder >= 1) {
				moments[1] = locvol*d.x*(i + 0.5); // x
				moments[2] = locvol*d.y*(j + 0.5); // y
			}
			if(polyorder >= 2) {
				moments[3] = locvol*ONE_THIRD*d.x*d.x*(1 + 3*i + 3*i*i); // x*x
				moments[4] = locvol*ONE_THIRD*d.y*d.y*(1 + 3*j + 3*j*j); // y*y
				moments[5] = locvol*0.25*d.x*d.y*(1 + 2*i)*(1 + 2*j); // x*y
			}

			//reduce to main grid
			for(m = 0; m < mmax; ++m) 
				grid->moments[m][vind(i, j)] = moments[m]; 

#endif // NO_REDUCTION
		}	
		else if(orcmp == 0x0f) {
			// the voxel crosses the boundary of the tet
			// need to process further
			
			// initialize the unit cube connectivity 
			r2du_init_box(&pixel, rbounds);
			for(v = 0; v < pixel.nverts; ++v)
				pixel.verts[v].orient = orient[vv[v]];

			// Clipping
#ifndef NO_CLIPPING
			r2d_clip_quad(&pixel, andcmp);
#endif // NO_CLIPPING

#ifndef NO_REDUCTION
			r2d_reduce(&pixel, polyorder, moments);

			// the cross-terms arising from using an offset box
			// must be taken into account in the absolute moment integrals
			if(polyorder >= 1) {
				xmin = (i + 0.5)*d.x;
				ymin = (j + 0.5)*d.y;
			}
			if(polyorder >= 2) {
				moments[3] += 2.0*xmin*moments[1] + xmin*xmin*moments[0];
				moments[4] += 2.0*ymin*moments[2] + ymin*ymin*moments[0];
				moments[5] += xmin*moments[2] + ymin*moments[1] + xmin*ymin*moments[0];
			}
			if(polyorder >= 1) {
				moments[1] += xmin*moments[0];
				moments[2] += ymin*moments[0];
			}

			// reduce to main grid
			for(m = 0; m < mmax; ++m) 
				grid->moments[m][vind(i, j)] = moments[m];

#endif // NO_REDUCTION
		}
		else {
			// pixel is entirely outside of the quad
			// ignore it
			continue;
		}
	}

#else // USE_TREE
	
	// get the initial face orientations for each corner of the node
	gpt.x = n.i*d.x;
	gpt.y = 0.0;
	curnode.orient[0].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[0].fflags |= (1 << f);
		curnode.orient[0].fdist[f] = gor;
	}
	gpt.y = n.j*d.y;
	curnode.orient[1].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[1].fflags |= (1 << f);
		curnode.orient[1].fdist[f] = gor;
	}
	gpt.x = 0.0;
	curnode.orient[2].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[2].fflags |= (1 << f);
		curnode.orient[2].fdist[f] = gor;
	}
	gpt.y = 0.0;
	curnode.orient[3].fflags = 0x00;
	for(f = 0; f < 4; ++f) {
		gor = faces[f].d + dot(gpt, faces[f].n);
		if(gor > 0.0) curnode.orient[3].fflags |= (1 << f);
		curnode.orient[3].fdist[f] = gor;
	}

	curnode.imin = 0;
	curnode.jmin = 0;
	curnode.ioff = n.i;
	curnode.joff = n.j;

	ntreestack = 0;
	treestack[ntreestack++] = curnode;
	while(ntreestack > 0) {

		// pop the top node
		curnode = treestack[--ntreestack];

		orcmp = 0x00;
        andcmp = 0x0f;
		for(v = 0; v < 4; ++v) {
			orcmp |= curnode.orient[v].fflags; 
			andcmp &= curnode.orient[v].fflags; 
		}

		if(andcmp == 0x0f) {
			
			// all cells in this leaf are fully contained
#ifndef NO_REDUCTION
			locvol = d.x*d.y;
			for(i = curnode.imin; i < curnode.imin + curnode.ioff; ++i)
			for(j = curnode.jmin; j < curnode.jmin + curnode.joff; ++j) {

				moments[0] = locvol; // 1
				if(polyorder >= 1) {
					moments[1] = locvol*d.x*(i + 0.5); // x
					moments[2] = locvol*d.y*(j + 0.5); // y
				}
				if(polyorder >= 2) {
					moments[3] = locvol*ONE_THIRD*d.x*d.x*(1 + 3*i + 3*i*i); // x*x
					moments[4] = locvol*ONE_THIRD*d.y*d.y*(1 + 3*j + 3*j*j); // y*y
					moments[5] = locvol*0.25*d.x*d.y*(1 + 2*i)*(1 + 2*j); // x*y
				}

				//reduce to main grid
				for(m = 0; m < mmax; ++m) 
					grid->moments[m][vind(i, j)] = moments[m]; 

			}
#endif // NO_REDUCTION
			continue;
		}
		if(orcmp != 0x0f) {
			// the leaf lies entirely outside the tet
			// skip it
			continue;
		}
		if(curnode.ioff == 1 && curnode.joff == 1) {
			// we've reached a single cell that straddles the tet boundary 
			// Clip and voxelize it.
			i = curnode.imin;
			j = curnode.jmin;
			
			// initialize the unit cube connectivity 
			r2du_init_box(&pixel, rbounds);
			for(v = 0; v < pixel.nverts; ++v)
				pixel.verts[v].orient = curnode.orient[v];

			// Clipping
#ifndef NO_CLIPPING
			r2d_clip_quad(&pixel, andcmp);
#endif // NO_CLIPPING

#ifndef NO_REDUCTION
			r2d_reduce(&pixel, polyorder, moments);

			// the cross-terms arising from using an offset box
			// must be taken into account in the absolute moment integrals
			if(polyorder >= 1) {
				xmin = (i + 0.5)*d.x;
				ymin = (j + 0.5)*d.y;
			}
			if(polyorder >= 2) {
				moments[3] += 2.0*xmin*moments[1] + xmin*xmin*moments[0];
				moments[4] += 2.0*ymin*moments[2] + ymin*ymin*moments[0];
				moments[5] += xmin*moments[2] + ymin*moments[1] + xmin*ymin*moments[0];
			}
			if(polyorder >= 1) {
				moments[1] += xmin*moments[0];
				moments[2] += ymin*moments[0];
			}

			// reduce to main grid
			for(m = 0; m < mmax; ++m) 
				grid->moments[m][vind(i, j)] = moments[m];

#endif // NO_REDUCTION

			continue;	
		}

		// else, split the node along its longest dimension
		// and push to the the stack
		i = curnode.ioff/2;
		j = curnode.joff/2;
		if(i >= j) {

			// LEFT NODE
			treestack[ntreestack].imin = curnode.imin;
			treestack[ntreestack].ioff = i;
			treestack[ntreestack].jmin = curnode.jmin;
			treestack[ntreestack].joff = curnode.joff;
			treestack[ntreestack].orient[2] = curnode.orient[2];
			treestack[ntreestack].orient[3] = curnode.orient[3];

			// RIGHT NODE
			treestack[ntreestack+1].imin = curnode.imin + i;
			treestack[ntreestack+1].ioff = curnode.ioff - i;
			treestack[ntreestack+1].jmin = curnode.jmin;
			treestack[ntreestack+1].joff = curnode.joff;
			treestack[ntreestack+1].orient[0] = curnode.orient[0];
			treestack[ntreestack+1].orient[1] = curnode.orient[1];

			// FILL IN COMMON POINTS
			gpt.x = d.x*(curnode.imin + i);
			gpt.y = d.y*curnode.jmin;
			treestack[ntreestack].orient[0].fflags = 0x00;
			treestack[ntreestack+1].orient[3].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[0].fdist[f] = gor;
				treestack[ntreestack+1].orient[3].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[0].fflags |= (1 << f);
					treestack[ntreestack+1].orient[3].fflags |= (1 << f);
				}
			}
			gpt.y = d.y*(curnode.jmin + curnode.joff);
			treestack[ntreestack].orient[1].fflags = 0x00;
			treestack[ntreestack+1].orient[2].fflags = 0x00;
			for(f = 0; f < 4; ++f) {
				gor = faces[f].d + dot(gpt, faces[f].n);
				treestack[ntreestack].orient[1].fdist[f] = gor;
				treestack[ntreestack+1].orient[2].fdist[f] = gor;
				if(gor > 0.0) {
					treestack[ntreestack].orient[1].fflags |= (1 << f);
					treestack[ntreestack+1].orient[2].fflags |= (1 << f);
				}
			}
			ntreestack += 2;
			continue;
		}
		else { // j > i

			// LEFT NODE
			treestack[ntreestack].imin = curnode.imin;
			treestack[ntreestack].ioff = curnode.ioff;
			treestack[ntreestack].jmin = curnode.jmin;
			treestack[ntreestack].joff = j;
			treestack[ntreestack].orient[0] = curnode.orient[0];
			treestack[ntreestack].orient[3] = curnode.orient[3];

			// RIGHT NODE
			treestack[ntreestack+1].imin = curnode.imin;
			treestack[ntreestack+1].ioff = curnode.ioff;
			treestack[ntreestack+1].jmin = curnode.jmin + j;
			treestack[ntreestack+1].joff = curnode.joff - j;
			treestack[ntreestack+1].orient[1] = curnode.orient[1];
			treestack[ntreestack+1].orient[2] = curnode.orient[2];

			// FILL IN COMMON POINTS
			gpt.x = d.x*curnode.imin;
			gpt.y = d.y*(curnode.jmin + j);
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
			gpt.x = d.x*(curnode.imin + curnode.ioff);
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
			ntreestack += 2;
			continue;
		}
	}

#endif // USE_TREE
}

void r2d_clip_quad(r2d_poly* poly, unsigned char andcmp) {

	unsigned char v, f, ff;
	unsigned char vstart, vnext, vcur, searching;
	unsigned char fmask, ffmask;
	
	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// loop over faces
	for(f = 0; f < 4; ++f) {

		fmask = (1 << f);
		if(andcmp & fmask) continue;

		// find the first vertex lying outside of the face
		// only need to find one (taking advantage of convexity)
		vcur = R2D_MAX_VERTS;
		for(v = 0; vcur >= R2D_MAX_VERTS && v < *nverts; ++v) 
			if(!(vertbuffer[v].orient.fflags & (CLIP_MASK | fmask))) vcur = v;
		if(vcur >= R2D_MAX_VERTS) continue;
				
		// traverse to the right
		vstart = vcur;
		searching = 1;
		while(searching) { 
			vnext = vertbuffer[vcur].pnbrs[1];
			if(fmask & vertbuffer[vnext].orient.fflags) {
				// vnext is inside the face

				// compute the intersection point using a weighted
				// average of perpendicular distances to the plane
				wav(vertbuffer[vcur].pos, vertbuffer[vnext].orient.fdist[f],
					vertbuffer[vnext].pos, -vertbuffer[vcur].orient.fdist[f],
					vertbuffer[*nverts].pos);
				
				vertbuffer[*nverts].pnbrs[1] = vnext;

				// reciprocal connetivity
				vertbuffer[vnext].pnbrs[0] = *nverts;

				// do face intersections and flags
				vertbuffer[*nverts].orient.fflags = 0x00;
				for(ff = f + 1; ff < 4; ++ff) {

					// skip if all initial verts are inside ff
					ffmask = (1 << ff); 
					if(andcmp & ffmask) continue;

					// weighted average keeps us in a relative coordinate system
					
					vertbuffer[*nverts].orient.fdist[ff] = 
						(vertbuffer[vcur].orient.fdist[ff]*vertbuffer[vnext].orient.fdist[f] 
							- vertbuffer[vnext].orient.fdist[ff]*vertbuffer[vcur].orient.fdist[f])
							/(vertbuffer[vnext].orient.fdist[f] - vertbuffer[vcur].orient.fdist[f]);
					if(vertbuffer[*nverts].orient.fdist[ff] > 0.0) vertbuffer[*nverts].orient.fflags |= ffmask;
				}
				++(*nverts);
				searching = 0;
			}
			vertbuffer[vcur].orient.fflags |= CLIP_MASK;
			vcur = vnext;
		}

		// traverse to the left
		vcur = vstart;
		searching = 1;
		while(searching) { 
			vnext = vertbuffer[vcur].pnbrs[0];
			if(fmask & vertbuffer[vnext].orient.fflags) {
				// vnext is inside the face

				// compute the intersection point using a weighted
				// average of perpendicular distances to the plane
				wav(vertbuffer[vcur].pos, vertbuffer[vnext].orient.fdist[f],
					vertbuffer[vnext].pos, -vertbuffer[vcur].orient.fdist[f],
					vertbuffer[*nverts].pos);
				
				vertbuffer[*nverts].pnbrs[0] = vnext;

				// reciprocal connetivity
				vertbuffer[vnext].pnbrs[1] = *nverts;

				// do face intersections and flags
				vertbuffer[*nverts].orient.fflags = 0x00;
				for(ff = f + 1; ff < 4; ++ff) {

					// skip if all verts are inside ff
					ffmask = (1 << ff); 
					if(andcmp & ffmask) continue;

					// weighted average keeps us in a relative coordinate system
					vertbuffer[*nverts].orient.fdist[ff] = 
						(vertbuffer[vcur].orient.fdist[ff]*vertbuffer[vnext].orient.fdist[f] 
							- vertbuffer[vnext].orient.fdist[ff]*vertbuffer[vcur].orient.fdist[f])
							/(vertbuffer[vnext].orient.fdist[f] - vertbuffer[vcur].orient.fdist[f]);
					if(vertbuffer[*nverts].orient.fdist[ff] > 0.0) vertbuffer[*nverts].orient.fflags |= ffmask;

				}
				++(*nverts);
				searching = 0;
			}
			vertbuffer[vcur].orient.fflags |= CLIP_MASK;
			vcur = vnext;
		}
		vertbuffer[*nverts-2].pnbrs[0] = *nverts-1; 
		vertbuffer[*nverts-1].pnbrs[1] = *nverts-2;
	}
}


void r2d_reduce(r2d_poly* poly, r2d_int polyorder, r2d_real* moments) {

	r2d_real locvol;
	unsigned char v, m;
	r2d_rvec2 v0, v1; 

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// zero the moments
	for(m = 0; m < 6; ++m)
		moments[m] = 0.0;

	// iterate over vertices and compute sum over simplices
	for(v = 0; v < *nverts; ++v) {
		
		if(vertbuffer[v].orient.fflags & CLIP_MASK) continue;
		v0 = vertbuffer[v].pos;
		v1 = vertbuffer[vertbuffer[v].pnbrs[1]].pos;
		locvol = 0.5*(v0.x*v1.y - v0.y*v1.x); 

		moments[0] += locvol; 
		if(polyorder >= 1) {
			moments[1] += locvol*ONE_THIRD*(v0.x + v1.x);
			moments[2] += locvol*ONE_THIRD*(v0.y + v1.y);
		}
		if(polyorder >= 2) {
			moments[3] += locvol*ONE_SIXTH*(v0.x*v0.x + v1.x*v1.x + v0.x*v1.x);
			moments[4] += locvol*ONE_SIXTH*(v0.y*v0.y + v1.y*v1.y + v0.y*v1.y);
			moments[5] += locvol*ONE_TWLFTH*(v0.x*(2.0*v0.y + v1.y) + v1.x*(v0.y + 2.0*v1.y));
		}
	}
}

/*
 * r2du: utility functions for r2d
 */

void r2du_init_box(r2d_poly* poly, r2d_rvec2 rbounds[2]) {

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 
	
	*nverts = 4;
	vertbuffer[0].pnbrs[0] = 3;	
	vertbuffer[0].pnbrs[1] = 1;	
	vertbuffer[1].pnbrs[0] = 0;	
	vertbuffer[1].pnbrs[1] = 2;	
	vertbuffer[2].pnbrs[0] = 1;	
	vertbuffer[2].pnbrs[1] = 3;	
	vertbuffer[3].pnbrs[0] = 2;	
	vertbuffer[3].pnbrs[1] = 0;	
	vertbuffer[0].pos.x = rbounds[1].x; 
	vertbuffer[0].pos.y = rbounds[0].y; 
	vertbuffer[1].pos.x = rbounds[1].x; 
	vertbuffer[1].pos.y = rbounds[1].y; 
	vertbuffer[2].pos.x = rbounds[0].x; 
	vertbuffer[2].pos.y = rbounds[1].y; 
	vertbuffer[3].pos.x = rbounds[0].x; 
	vertbuffer[3].pos.y = rbounds[0].y; 
}

void r2du_faces_from_verts(r2d_rvec2* verts, r2d_int nverts, r2d_plane* faces) {

	// TODO: warn for convexity?

	// compute unit face normals and distances to origin
	r2d_rvec2 v0, v1, tmpcent;
	r2d_int f;

	// Assumes vertices are CCW
	for(f = 0; f < nverts; ++f) {
		v0 = verts[f];
		v1 = verts[(f+1)%nverts];
		faces[f].n.x = -(v1.y - v0.y);
		faces[f].n.y = (v1.x - v0.x);
		norm(faces[f].n);
		tmpcent.x = 0.5*(v0.x + v1.x);
		tmpcent.y = 0.5*(v0.y + v1.y);
		faces[f].d = -dot(faces[f].n, tmpcent);
	}
}

r2d_real r2du_orient(r2d_rvec2 pa, r2d_rvec2 pb, r2d_rvec2 pc) {
	return 0.5*((pc.x - pa.x)*(pb.y - pa.y) - (pb.x - pa.x)*(pc.y - pa.y)); 
}

