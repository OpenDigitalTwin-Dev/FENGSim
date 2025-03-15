/*
 *
 *		rNd.c
 *		
 *		See rNd.h for usage.
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

#include "rNd.h"
#include <string.h>
#include <math.h>
#include <stdio.h>


void rNd_clip(rNd_poly* poly, rNd_plane* planes, rNd_int nplanes) {

	// direct access to vertex buffer
	rNd_vertex* vertbuffer = poly->verts; 
	rNd_int* nverts = &poly->nverts; 
	rNd_int* nfaces = &poly->nfaces; 
	if(*nverts <= 0) return;

	// variable declarations
	rNd_int i, v, p, np, np0, np1, onv, vcur, fcur, vprev, vnext, vstart, 
			pnext, numunclipped, fadj, prevnewvert, pprev, ftmp;

	// signed distances to the clipping plane
	rNd_real sdists[RND_MAX_VERTS];
	rNd_real smin, smax;

	// for marking clipped vertices
	rNd_int clipped[RND_MAX_VERTS];

	// loop over each clip plane
	for(p = 0; p < nplanes; ++p) {

		// calculate signed distances to the clip plane
		onv = *nverts;
		smin = 1.0e30;
		smax = -1.0e30;
		memset(&clipped, 0, sizeof(clipped));
		for(v = 0; v < onv; ++v) {
			sdists[v] = planes[p].d;
			for(i = 0; i < RND_DIM; ++i)
				sdists[v] += vertbuffer[v].pos.xyz[i]*planes[p].n.xyz[i];
			if(sdists[v] < smin) smin = sdists[v];
			if(sdists[v] > smax) smax = sdists[v];
			if(sdists[v] < 0.0) clipped[v] = 1;
		}

		// skip this face if the poly lies entirely on one side of it 
		if(smin >= 0.0) continue;
		if(smax <= 0.0) {
			*nverts = 0;
			return;
		}

		// check all edges and insert new vertices on the bisected edges 
		// Also map existing 2-face IDs to the new vertices
		for(vcur = 0; vcur < onv; ++vcur) {
			if(clipped[vcur]) continue;
			for(np = 0; np < RND_DIM; ++np) {
				vnext = vertbuffer[vcur].pnbrs[np];
				if(!clipped[vnext]) continue;
				vertbuffer[*nverts].pnbrs[0] = vcur;
				vertbuffer[vcur].pnbrs[np] = *nverts;
				for(i = 0; i < RND_DIM; ++i) // weighted average of vertex positions
					vertbuffer[*nverts].pos.xyz[i] = (vertbuffer[vnext].pos.xyz[i]*sdists[vcur] 
							- vertbuffer[vcur].pos.xyz[i]*sdists[vnext])/(sdists[vcur] - sdists[vnext]);
				for(np0 = 0, np1 = 1; np0 < RND_DIM; ++np0) { 
					if(np0 == np) continue;
					vertbuffer[*nverts].finds[0][np1] = vertbuffer[vcur].finds[np][np0];
					vertbuffer[*nverts].finds[np1][0] = vertbuffer[vcur].finds[np][np0];
					++np1;
				}
				// mark everything else with -1 to indicate it needs attention
				// TODO: does this re-insert the new faces with the correct parity
				// around the new vertex?
				for(np0 = 1; np0 < RND_DIM; ++np0) vertbuffer[*nverts].pnbrs[np0] = -1;
				for(np0 = 1; np0 < RND_DIM; ++np0)
				for(np1 = 1; np1 < RND_DIM; ++np1)
					vertbuffer[*nverts].finds[np0][np1] = -1;
				(*nverts)++;
			}
		}

		// TODO: doesn't work for 2D,,, no 3-cells to walk over

		// search completely around the boundary of all existing 3-cells 
		// to close off all new 2-faces
		for(vstart = onv; vstart < *nverts; ++vstart)
		for(np0 = 1; np0 < RND_DIM; ++np0)
		for(np1 = np0+1; np1 < RND_DIM; ++np1) {
			if(vertbuffer[vstart].finds[np0][np1] >= 0) continue;
			fcur = vertbuffer[vstart].finds[0][np0];
			fadj = vertbuffer[vstart].finds[0][np1];
			vprev = vstart;
			vcur = vertbuffer[vstart].pnbrs[0]; 
			prevnewvert = vstart;
			do {
				if(vcur >= onv) {
					for(pprev = 1; pprev < RND_DIM; ++pprev)
						if(vertbuffer[vcur].finds[0][pprev] == fcur) break;
					for(pnext = 1; pnext < RND_DIM; ++pnext)
						if(vertbuffer[prevnewvert].finds[0][pnext] == fcur) break;
					for(np = 1; np < RND_DIM; ++np)
						if(vertbuffer[vcur].finds[0][np] == fadj) break;
					vertbuffer[vcur].pnbrs[pprev] = prevnewvert;
					vertbuffer[prevnewvert].pnbrs[pnext] = vcur;
					vertbuffer[vcur].finds[pprev][np] = *nfaces;
					vertbuffer[vcur].finds[np][pprev] = *nfaces;
					ftmp = fcur;
					fcur = fadj;
					fadj = ftmp;
					prevnewvert = vcur;
					vprev = vcur;
					vcur = vertbuffer[vcur].pnbrs[0];
				}
				for(pprev = 0; pprev < RND_DIM; ++pprev) 
					if(vertbuffer[vcur].pnbrs[pprev] == vprev) break;
				for(pnext = 0; pnext < RND_DIM; ++pnext) {
					if(pnext == pprev) continue;
					if(vertbuffer[vcur].finds[pprev][pnext] == fcur) break;
				}
				for(np = 0; np < RND_DIM; ++np) {
					if(np == pprev) continue;
					if(vertbuffer[vcur].finds[np][pprev] == fadj) break;
				}
				fadj = vertbuffer[vcur].finds[np][pnext];
				vprev = vcur;
				vcur = vertbuffer[vcur].pnbrs[pnext];
			} while(vcur != vstart);
			for(pprev = 1; pprev < RND_DIM; ++pprev)
				if(vertbuffer[vcur].finds[0][pprev] == fcur) break;
			for(pnext = 1; pnext < RND_DIM; ++pnext)
				if(vertbuffer[prevnewvert].finds[0][pnext] == fcur) break;
			for(np = 1; np < RND_DIM; ++np)
				if(vertbuffer[vcur].finds[0][np] == fadj) break;
			vertbuffer[vcur].pnbrs[pprev] = prevnewvert;
			vertbuffer[prevnewvert].pnbrs[pnext] = vcur;
			vertbuffer[vcur].finds[pprev][np] = *nfaces;
			vertbuffer[vcur].finds[np][pprev] = *nfaces;
			(*nfaces)++;
		}

		// go through and compress the vertex list, removing clipped verts
		// and re-indexing accordingly (reusing `clipped` to re-index everything)
		numunclipped = 0;
		for(v = 0; v < *nverts; ++v) {
			if(!clipped[v]) {
				vertbuffer[numunclipped] = vertbuffer[v];
				clipped[v] = numunclipped++;
			}
		}
		*nverts = numunclipped;
		for(v = 0; v < *nverts; ++v) 
			for(np = 0; np < RND_DIM; ++np)
				vertbuffer[v].pnbrs[np] = clipped[vertbuffer[v].pnbrs[np]];
	}
}



rNd_real reduce_helper(rNd_poly* poly, rNd_int v, rNd_int d, rNd_int processed[RND_DIM], rNd_rvec ltd[RND_DIM]) {

	// TODO: clean this up
	// TODO: Make it nonrecursive? More memory-efficient too??

	rNd_int i, j, dd;
	rNd_real ltdsum = 0.0;

	if(d == RND_DIM) {

		//printf("Full LTD!\n");

		ltdsum = 1.0;
		for(dd = 0; dd < RND_DIM; ++dd) {

			rNd_real dot = 0.0;
			
			for(j = 0; j < RND_DIM; ++j)
				dot += ltd[dd].xyz[j]*poly->verts[v].pos.xyz[j];

			ltdsum *= 1.0*dot/(dd+1);

			//for(j = 0; j < RND_DIM; ++j)
				//printf(" %.5e", ltd[dd][j]);
			//printf("\n");

		}
			//printf("det = %.5e\n", determinant(ltd));

		return ltdsum;
	}

	for(i = 0; i < RND_DIM; ++i) {
		if(processed[i]) continue;

		// copy processed
		rNd_int proc2[RND_DIM];
		for(j = 0; j < RND_DIM; ++j)
			proc2[j] = processed[j];
		proc2[i] = 1;

		for(j = 0; j < RND_DIM; ++j)
			ltd[d].xyz[j] = poly->verts[v].pos.xyz[j] - poly->verts[poly->verts[v].pnbrs[i]].pos.xyz[j];

		// TODO: explore the robustness of the orthogonalization

		for(dd = 0; dd < d; ++dd) {
			rNd_real dot = 0.0;
			for(j = 0; j < RND_DIM; ++j)
				dot += ltd[d].xyz[j]*ltd[dd].xyz[j];
			for(j = 0; j < RND_DIM; ++j)
				ltd[d].xyz[j] -= dot*ltd[dd].xyz[j];
		}

		rNd_real len = 0.0;
		for(j = 0; j < RND_DIM; ++j)
			len += ltd[d].xyz[j]*ltd[d].xyz[j];

		len = sqrt(len);

		for(j = 0; j < RND_DIM; ++j)
			ltd[d].xyz[j] /= len;

		//if(d == RND_DIM - 1) {
			//for(j = 0; j < RND_DIM; ++j)
				//ltd[d][j] *= -1;
		//}

		
		ltdsum += reduce_helper(poly, v, d+1, proc2, ltd);

	}

	return ltdsum;

}

void rNd_reduce(rNd_poly* poly, rNd_real* moments, rNd_int polyorder) {

#if 1
	rNd_int v;

	// direct access to vertex buffer
	//rNd_vertex* vertbuffer = poly->verts; 
	rNd_int* nverts = &poly->nverts; 
	
	moments[0] = 0.0;

	// sum over all vertex LTD terms
	for(v = 0; v < *nverts; ++v) {

		rNd_int processed[RND_DIM];
		memset(&processed, 0, sizeof(processed));

		rNd_rvec ltd[RND_DIM];

		rNd_real vtmp = reduce_helper(poly, v, 0, processed, ltd);

		//printf("Finished vert %d, vtmp = %.5e\n", v, vtmp);

		moments[0] += vtmp;

	}
#else

	rNd_real det, dtmp;
	rNd_int perm[RND_DIM], c[RND_DIM];
	rNd_int n, i, j, itmp;
	det = 0.0;
	for(i = 0; i < RND_DIM; ++i) {
		perm[i] = i;
		c[i] = 0;
	}

	rNd_int np = 0;
	printf("permutation %d:", np);
	for(i = 0; i < RND_DIM; ++i) printf(" %d", perm[i]);
	printf("\n");


	for(n = 1; n < RND_DIM;) {
		if(c[n] < n) {
			j = (n%2)*c[n];
			itmp = perm[j];
			perm[j] = perm[n];
			perm[n] = itmp;
			c[n]++;
			n = 1;

			++np;
			printf("permutation %d:", np);
			for(i = 0; i < RND_DIM; ++i) printf(" %d", perm[i]);
			printf("\n");


		}
		else c[n++] = 0;
	}
#endif

}



rNd_int rNd_is_good(rNd_poly* poly) {

	rNd_int v, vc, np, npp, rcur, insubset;
	rNd_int nvstack;
	rNd_int vct[RND_MAX_VERTS];
	rNd_int stack[RND_MAX_VERTS];
	rNd_int regions[RND_MAX_VERTS];
	rNd_int subset[RND_DIM-1];

	// direct access to vertex buffer
	rNd_vertex* vertbuffer = poly->verts; 
	rNd_int* nverts = &poly->nverts; 

	/////////////////////
	// easy checks first 
	/////////////////////
	
	memset(&vct, 0, sizeof(vct));
	for(v = 0; v < *nverts; ++v) {
		for(np = 0; np < RND_DIM; ++np) {
			// return if any verts point to themselves
			if(vertbuffer[v].pnbrs[np]  == v) {
				printf("Self-pointing vertex.\n");
				return 0;
			}
			// return if any edges are obviously invalid
			if(vertbuffer[v].pnbrs[np] >= *nverts) {
				printf("Bad pointer.\n");
				return 0;
			}
			// return if any verts point to the same other vert twice
			for(npp = np+1; npp < RND_DIM; ++npp)
				if(vertbuffer[v].pnbrs[np] == vertbuffer[v].pnbrs[npp]) {
					printf("Double edge.\n");
					return 0;
				}
			// count edges per vertex
			vct[vertbuffer[v].pnbrs[np]]++;
		}
	}
	
	// return false if any vertices are pointed to 
	// by more or fewer than DIM other vertices
	for(v = 0; v < *nverts; ++v) if(vct[v] != RND_DIM) {
		printf("Bad edge count: count[%d] = %d.\n", v, vct[v]);
		return 0;
	}

	/////////////////////////////////////////////////////////
	// check for k-vertex-connectedness (Balinski's theorem)
	/////////////////////////////////////////////////////////

	// handle multiply-connected polyhedra by testing each 
	// component separately. Flood-fill starting from each vertex
	// to give each connected region a unique ID.
	rcur = 1;
	memset(&regions, 0, sizeof(regions));
	for(v = 0; v < *nverts; ++v) {
		if(regions[v]) continue;
		nvstack = 0;
		stack[nvstack++] = v;
		while(nvstack > 0) {
			vc = stack[--nvstack];
			if(regions[vc]) continue;
			regions[vc] = rcur;
			for(np = 0; np < RND_DIM; ++np)
				stack[nvstack++] = vertbuffer[vc].pnbrs[np];
		}
		++rcur;
	}

	// iterate over all possible subsets of the vertices
	// with size DIM-1
	for(v = 0; v < RND_DIM-1; ++v) subset[v] = v;
	while(subset[RND_DIM-2] < *nverts) {

		// get the current region and make sure all verts in the subset
		// are in the same one
		rcur = regions[subset[0]];
		for(v = 1; v < RND_DIM-1; ++v) 
			if(regions[subset[v]] != rcur) goto next_subset; 
	
		// use vct to mark visited verts
		// mask out the selected subset 
		memset(&vct, 0, sizeof(vct));
		for(v = 0; v < RND_DIM-1; ++v) vct[subset[v]] = 1;
	
		// pick a starting vert in the same connected component
		// but not in the selected subset
		for(vc = 0; vc < *nverts; ++vc) {
			if(regions[vc] != rcur) continue;
			insubset = 0;
			for(v = 0; v < RND_DIM-1; ++v)
				if(vc == subset[v]) {
					insubset = 1;
					break;
				}
			if(!insubset) break;
		}
		
		// flood-fill from vc to make sure the graph is 
		// still connected when va and vb are masked
		nvstack = 0;
		stack[nvstack++] = vc;
		while(nvstack > 0) {
			vc = stack[--nvstack];
			if(vct[vc]) continue;
			vct[vc] = 1;
			for(np = 0; np < RND_DIM; ++np)
				stack[nvstack++] = vertbuffer[vc].pnbrs[np];
		}

		// if any verts in the region rcur were untouched, 
		// the graph is not DIM-vertex-connected and hence an invalid polyhedron
		for(v = 0; v < *nverts; ++v) if(regions[v] == rcur && !vct[v]) {
			printf("Not %d-vertex-connected.\n", RND_DIM);
			return 0;
		}

		// update the index list to the next lexicographic permutation 
		next_subset:
		for(v = 0; v < RND_DIM-2; ++v) 
			if(subset[v] < subset[v+1]-1) {
				subset[v]++;
				break;
			}
		if(v == RND_DIM-2) subset[v]++;
		for(--v; v >= 0; --v) subset[v] = v;
	}

	return 1;
}

void rNd_rotate(rNd_poly* poly, rNd_real theta, rNd_int ax1, rNd_int ax2) {
	rNd_int v;
	rNd_rvec tmp;
	rNd_real sine = sin(theta);
	rNd_real cosine = cos(theta);
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;
		poly->verts[v].pos.xyz[ax1] = cosine*tmp.xyz[ax1] - sine*tmp.xyz[ax2]; 
		poly->verts[v].pos.xyz[ax2] = sine*tmp.xyz[ax1] + cosine*tmp.xyz[ax2]; 
	}
}

void rNd_translate(rNd_poly* poly, rNd_rvec shift) {
	rNd_int v, i;
	for(v = 0; v < poly->nverts; ++v)
	for(i = 0; i < RND_DIM; ++i)
		poly->verts[v].pos.xyz[i] += shift.xyz[i];
}

void rNd_scale(rNd_poly* poly, rNd_rvec scale) {
	rNd_int v, i;
	for(v = 0; v < poly->nverts; ++v)
	for(i = 0; i < RND_DIM; ++i)
		poly->verts[v].pos.xyz[i] *= scale.xyz[i];
}

void rNd_shear(rNd_poly* poly, rNd_real shear, rNd_int axb, rNd_int axs) {
	rNd_int v;
	for(v = 0; v < poly->nverts; ++v)
		poly->verts[v].pos.xyz[axb] += shear*poly->verts[v].pos.xyz[axs];
}

void rNd_affine(rNd_poly* poly, rNd_real mat[RND_DIM+1][RND_DIM+1]) {
	rNd_int v, i, j;
	rNd_rvec tmp;
	rNd_real w;
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;
		// affine transformation
		for(i = 0; i < RND_DIM; ++i)
		for(j = 0; j < RND_DIM; ++j)
			poly->verts[v].pos.xyz[i] = tmp.xyz[j]*mat[i][j];
		for(i = 0; i < RND_DIM; ++i)
			poly->verts[v].pos.xyz[i] += mat[i][RND_DIM];
		// homogeneous divide if w != 1, i.e. in a perspective projection
		w = 0.0;
		for(j = 0; j < RND_DIM; ++j)
			w += tmp.xyz[j]*mat[RND_DIM][j];
		for(i = 0; i < RND_DIM; ++i)
			poly->verts[v].pos.xyz[i] /= w; 
	}
}

void rNd_init_simplex(rNd_poly* poly, rNd_rvec verts[RND_DIM+1]) {

	rNd_int v, i;
	rNd_int v0, v1, v2, np0, np1, np2, f;

	// direct access to vertex buffer
	rNd_vertex* vertbuffer = poly->verts; 
	rNd_int* nverts = &poly->nverts; 
	rNd_int* nfaces = &poly->nfaces; 

	// set up vertex positions and connectivity
	*nverts = RND_DIM+1;
	for(v = 0; v < RND_DIM+1; ++v) {
		for(i = 0; i < RND_DIM; ++i) {
			vertbuffer[v].pos.xyz[i] = verts[v].xyz[i];
			vertbuffer[v].pnbrs[i] = (v+i+1)%(RND_DIM+1);
		}
	}

	// TODO: set up reverse pnbrs??
	// could speed up traversal...

	// set up 2-faces
	// made up of every possible unique set of 3 vertices
	for(v0 = 0, f = 0; v0 < *nverts; ++v0)
	for(v1 = v0+1; v1 < *nverts; ++v1)
	for(v2 = v1+1; v2 < *nverts; ++v2, ++f) {
		for(np1 = 0; np1 < RND_DIM; ++np1) if(vertbuffer[v0].pnbrs[np1] == v1) break;
		for(np2 = 0; np2 < RND_DIM; ++np2) if(vertbuffer[v0].pnbrs[np2] == v2) break;
		vertbuffer[v0].finds[np1][np2] = f;
		vertbuffer[v0].finds[np2][np1] = f;
		for(np0 = 0; np0 < RND_DIM; ++np0) if(vertbuffer[v1].pnbrs[np0] == v0) break;
		for(np2 = 0; np2 < RND_DIM; ++np2) if(vertbuffer[v1].pnbrs[np2] == v2) break;
		vertbuffer[v1].finds[np0][np2] = f;
		vertbuffer[v1].finds[np2][np0] = f;
		for(np0 = 0; np0 < RND_DIM; ++np0) if(vertbuffer[v2].pnbrs[np0] == v0) break;
		for(np1 = 0; np1 < RND_DIM; ++np1) if(vertbuffer[v2].pnbrs[np1] == v1) break;
		vertbuffer[v2].finds[np0][np1] = f;
		vertbuffer[v2].finds[np1][np0] = f;
	}
	*nfaces = f;
}

void rNd_init_box(rNd_poly* poly, rNd_rvec rbounds[2]) {

	rNd_int i, v, np, np1, stride;

	// direct access to vertex buffer
	rNd_vertex* vertbuffer = poly->verts; 
	rNd_int* nverts = &poly->nverts; 
	rNd_int* nfaces = &poly->nfaces; 

	// the bit-hacky way
	*nverts = (1 << RND_DIM);
	for(v = 0; v < (1 << RND_DIM); ++v)
	for(i = 0; i < RND_DIM; ++i) {
		stride = (1 << i);
		vertbuffer[v].pos.xyz[i] = rbounds[(v & stride) > 0].xyz[i];
		vertbuffer[v].pnbrs[i] = (v ^ stride); 
	}
 	
	// TODO: bit hacks for this too!!
	// TODO: make faces unique!!
	*nfaces = 0;
	for(np = 0; np < RND_DIM; ++np)
	for(np1 = np+1; np1 < RND_DIM; ++np1, ++(*nfaces)) 
	for(v = 0; v < (1 << RND_DIM); ++v) {
		vertbuffer[v].finds[np][np1] = *nfaces;
		vertbuffer[v].finds[np1][np] = *nfaces;
	}
}

#if 0
void r3d_tet_faces_from_verts(r3d_plane* faces, r3d_rvec3* verts) {
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

void r3d_box_faces_from_verts(r3d_plane* faces, r3d_rvec3* rbounds) {
	faces[0].n.x = 0.0; faces[0].n.y = 0.0; faces[0].n.z = 1.0; faces[0].d = rbounds[0].z; 
	faces[2].n.x = 0.0; faces[2].n.y = 1.0; faces[2].n.z = 0.0; faces[2].d = rbounds[0].y; 
	faces[4].n.x = 1.0; faces[4].n.y = 0.0; faces[4].n.z = 0.0; faces[4].d = rbounds[0].x; 
	faces[1].n.x = 0.0; faces[1].n.y = 0.0; faces[1].n.z = -1.0; faces[1].d = rbounds[1].z; 
	faces[3].n.x = 0.0; faces[3].n.y = -1.0; faces[3].n.z = 0.0; faces[3].d = rbounds[1].y; 
	faces[5].n.x = -1.0; faces[5].n.y = 0.0; faces[5].n.z = 0.0; faces[5].d = rbounds[1].x; 
}

#endif


rNd_real rNd_det(rNd_real mat[RND_DIM][RND_DIM]) {

	// Uses Heap's algorithm to nonrecursively iterate over all 
	// index permutations, using the Leibniz formula to explicity 
	// calculate the determinant. This is O(RND_DIM!), so be careful!

	rNd_real det, dtmp;
	rNd_int perm[RND_DIM], c[RND_DIM];
	rNd_int n, i, j, itmp, parity;
	parity = 2*(RND_DIM%2)-1; // TODO: Is Bourke's determinant wrong??
	det = 0.0;
	for(i = 0; i < RND_DIM; ++i) {
		perm[i] = i;
		c[i] = 0;
		dtmp = 1.0*parity;
		for(j = 0; j < RND_DIM; ++j)
			dtmp *= mat[j][(i+j)%RND_DIM];
	}
	det += dtmp;
	for(n = 1; n < RND_DIM;) {
		if(c[n] < n) {
			j = (n%2)*c[n];
			itmp = perm[j];
			perm[j] = perm[n];
			perm[n] = itmp;
			c[n]++;
			n = 1;
			parity *= -1;
			for(i = 0; i < RND_DIM; ++i) {
				dtmp = 1.0*parity;
				for(j = 0; j < RND_DIM; ++j)
					dtmp *= mat[j][(i+perm[j])%RND_DIM];
			}
			det += dtmp;
		}
		else c[n++] = 0;
	}
	return det;
}

rNd_real rNd_orient(rNd_rvec verts[RND_DIM+1]) {

	rNd_int i, j;
	rNd_real fac;
	rNd_real mat[RND_DIM][RND_DIM];

	// subtract one vertex from the rest
	for(i = 0; i < RND_DIM; ++i)
	for(j = 0; j < RND_DIM; ++j)
		mat[i][j] = verts[i+1].xyz[j]-verts[0].xyz[j];

	// get the factorial
	fac = 1.0;
	for(i = 1; i <= RND_DIM; ++i) fac *= i;
	
	// return the determinant
	return rNd_det(mat)/fac;
}

void rNd_print(rNd_poly* poly) {
	rNd_int v, i;
	for(v = 0; v < poly->nverts; ++v) {
		printf("vert %d, pos =", v);
		for(i = 0; i < RND_DIM; ++i) printf(" %.5e", poly->verts[v].pos.xyz[i]);
		printf(", nbrs =");
		for(i = 0; i < RND_DIM; ++i) printf(" %d", poly->verts[v].pnbrs[i]);
		printf("\n");
	}
}


