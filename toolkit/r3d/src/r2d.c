/*
 *
 *		r2d.c
 *		
 *		Devon Powell
 *		31 August 2015
 *		
 *		See r2d.h for usage.
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

#include "r2d.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// useful macros
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
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


void r2d_clip(r2d_poly* poly, r2d_plane* planes, r2d_int nplanes) {

	// variable declarations
	r2d_int v, p, np, onv, vstart, vcur, vnext, numunclipped; 

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 
	if(*nverts <= 0) return;

	// signed distances to the clipping plane
	r2d_real sdists[R2D_MAX_VERTS];
	r2d_real smin, smax;

	// for marking clipped vertices
	r2d_int clipped[R2D_MAX_VERTS];

	// loop over each clip plane
	for(p = 0; p < nplanes; ++p) {
	
		// calculate signed distances to the clip plane
		onv = *nverts;
		smin = 1.0e30;
		smax = -1.0e30;
		memset(&clipped, 0, sizeof(clipped));
		for(v = 0; v < onv; ++v) {
			sdists[v] = planes[p].d + dot(vertbuffer[v].pos, planes[p].n);
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
		for(vcur = 0; vcur < onv; ++vcur) {
			if(clipped[vcur]) continue;
			for(np = 0; np < 2; ++np) {
				vnext = vertbuffer[vcur].pnbrs[np];
				if(!clipped[vnext]) continue;
				vertbuffer[*nverts].pnbrs[1-np] = vcur;
				vertbuffer[*nverts].pnbrs[np] = -1;
				vertbuffer[vcur].pnbrs[np] = *nverts;
				wav(vertbuffer[vcur].pos, -sdists[vnext],
					vertbuffer[vnext].pos, sdists[vcur],
					vertbuffer[*nverts].pos);
				(*nverts)++;
			}
		}

		// for each new vert, search around the poly for its new neighbors
		// and doubly-link everything
		for(vstart = onv; vstart < *nverts; ++vstart) {
			if(vertbuffer[vstart].pnbrs[1] >= 0) continue;
			vcur = vertbuffer[vstart].pnbrs[0];
			do {
				vcur = vertbuffer[vcur].pnbrs[0]; 
			} while(vcur < onv);
			vertbuffer[vstart].pnbrs[1] = vcur;
			vertbuffer[vcur].pnbrs[0] = vstart;
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
		for(v = 0; v < *nverts; ++v) {
			vertbuffer[v].pnbrs[0] = clipped[vertbuffer[v].pnbrs[0]];
			vertbuffer[v].pnbrs[1] = clipped[vertbuffer[v].pnbrs[1]];
		}	
	}
}

void r2d_split(r2d_poly* inpolys, r2d_int npolys, r2d_plane plane, r2d_poly* out_pos, r2d_poly* out_neg) {

	// direct access to vertex buffer
	r2d_int* nverts;
	r2d_int p;
	r2d_vertex* vertbuffer;
	r2d_int v, np, onv, vcur, vnext, vstart, nright, cside;
	r2d_rvec2 newpos;
	r2d_int side[R2D_MAX_VERTS];
	r2d_real sdists[R2D_MAX_VERTS];
	r2d_poly* outpolys[2];

	for(p = 0; p < npolys; ++p) {

		nverts = &inpolys[p].nverts;
		vertbuffer = inpolys[p].verts; 
		outpolys[0] = &out_pos[p];
		outpolys[1] = &out_neg[p];
		if(*nverts <= 0) {
			memset(&out_pos[p], 0, sizeof(r2d_poly));
			memset(&out_neg[p], 0, sizeof(r2d_poly));
			continue;
		} 


		// calculate signed distances to the clip plane
		nright = 0;
		memset(&side, 0, sizeof(side));
		for(v = 0; v < *nverts; ++v) {
			sdists[v] = plane.d + dot(vertbuffer[v].pos, plane.n);
			sdists[v] *= -1;
			if(sdists[v] < 0.0) {
				side[v] = 1;
				nright++;
			}
		}
	
		// return if the poly lies entirely on one side of it 
		if(nright == 0) {
			*(outpolys[0]) = inpolys[p]; 
			outpolys[1]->nverts = 0;
			continue;	
		}
		if(nright == *nverts) {
			*(outpolys[1]) = inpolys[p];
			outpolys[0]->nverts = 0;
			continue;
		}
	
		// check all edges and insert new vertices on the bisected edges 
		onv = *nverts; 
		for(vcur = 0; vcur < onv; ++vcur) {
			if(side[vcur]) continue;
			for(np = 0; np < 2; ++np) {
				vnext = vertbuffer[vcur].pnbrs[np];
				if(!side[vnext]) continue;
				wav(vertbuffer[vcur].pos, -sdists[vnext],
					vertbuffer[vnext].pos, sdists[vcur],
					newpos);
				vertbuffer[*nverts].pos = newpos;
				vertbuffer[vcur].pnbrs[np] = *nverts;
				vertbuffer[*nverts].pnbrs[np] = -1;
				vertbuffer[*nverts].pnbrs[1-np] = vcur;
				(*nverts)++;
				side[*nverts] = 1;
				vertbuffer[*nverts].pos = newpos;
				vertbuffer[*nverts].pnbrs[1-np] = -1;
				vertbuffer[*nverts].pnbrs[np] = vnext;
				vertbuffer[vnext].pnbrs[1-np] = *nverts;
				(*nverts)++;
			}
		}
	
		// for each new vert, search around the poly for its new neighbors
		// and doubly-link everything
		for(vstart = onv; vstart < *nverts; ++vstart) {
			if(vertbuffer[vstart].pnbrs[1] >= 0) continue;
			vcur = vertbuffer[vstart].pnbrs[0];
			do {
				vcur = vertbuffer[vcur].pnbrs[0]; 
			} while(vcur < onv);
			vertbuffer[vstart].pnbrs[1] = vcur;
			vertbuffer[vcur].pnbrs[0] = vstart;
		}
	
		// copy and compress vertices into their new buffers
		// reusing side[] for reindexing
		onv = *nverts;
		outpolys[0]->nverts = 0;
		outpolys[1]->nverts = 0;
		for(v = 0; v < onv; ++v) {
			cside = side[v];
			outpolys[cside]->verts[outpolys[cside]->nverts] = vertbuffer[v];
			side[v] = (outpolys[cside]->nverts)++;
		}
	
		for(v = 0; v < outpolys[0]->nverts; ++v) 
			for(np = 0; np < 2; ++np)
				outpolys[0]->verts[v].pnbrs[np] = side[outpolys[0]->verts[v].pnbrs[np]];
		for(v = 0; v < outpolys[1]->nverts; ++v) 
			for(np = 0; np < 2; ++np)
				outpolys[1]->verts[v].pnbrs[np] = side[outpolys[1]->verts[v].pnbrs[np]];
	}
}

void r2d_reduce(r2d_poly* poly, r2d_real* moments, r2d_int polyorder) {

	// var declarations
	r2d_int vcur, vnext, m, i, j, corder;
	r2d_real twoa;
	r2d_rvec2 v0, v1, vc;

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// zero the moments
	for(m = 0; m < R2D_NUM_MOMENTS(polyorder); ++m) moments[m] = 0.0;

	if(*nverts <= 0) return;

#ifdef SHIFT_POLY
	// translate a polygon to the origin for increased accuracy
	// (this will increase computational cost, in particular for higher moments)
	vc = r2d_poly_center(poly);
#endif

	// Storage for coefficients
	// keep two layers of the triangle of coefficients
	r2d_int prevlayer = 0;
	r2d_int curlayer = 1;
	r2d_real D[polyorder+1][2];
	r2d_real C[polyorder+1][2];

	// iterate over edges and compute a sum over simplices 
	for(vcur = 0; vcur < *nverts; ++vcur) {

		vnext = vertbuffer[vcur].pnbrs[0];
		v0 = vertbuffer[vcur].pos;
		v1 = vertbuffer[vnext].pos;

#ifdef SHIFT_POLY
		v0.x = v0.x - vc.x;
		v0.y = v0.y - vc.y;
		v1.x = v1.x - vc.x;
		v1.y = v1.y - vc.y;
#endif

		twoa = (v0.x*v1.y - v0.y*v1.x);

		// calculate the moments
		// using the fast recursive method of Koehl (2012)
		// essentially building a set of Pascal's triangles, one layer at a time

		// base case
		D[0][prevlayer] = 1.0;
		C[0][prevlayer] = 1.0;
		moments[0] += 0.5*twoa;

		// build up successive polynomial orders
		for(corder = 1, m = 1; corder <= polyorder; ++corder) {
			for(i = corder; i >= 0; --i, ++m) {
				j = corder - i;
				C[i][curlayer] = 0; 
				D[i][curlayer] = 0;  
				if(i > 0) {
					C[i][curlayer] += v1.x*C[i-1][prevlayer];
					D[i][curlayer] += v0.x*D[i-1][prevlayer]; 
				}
				if(j > 0) {
					C[i][curlayer] += v1.y*C[i][prevlayer];
					D[i][curlayer] += v0.y*D[i][prevlayer]; 
				}
				D[i][curlayer] += C[i][curlayer]; 
				moments[m] += twoa*D[i][curlayer];
			}
			curlayer = 1 - curlayer;
			prevlayer = 1 - prevlayer;
		}
	}

	// reuse C to recursively compute the leading multinomial coefficients
	C[0][prevlayer] = 1.0;
	for(corder = 1, m = 1; corder <= polyorder; ++corder) {
		for(i = corder; i >= 0; --i, ++m) {
			j = corder - i;
			C[i][curlayer] = 0.0; 
			if(i > 0) C[i][curlayer] += C[i-1][prevlayer];
			if(j > 0) C[i][curlayer] += C[i][prevlayer];
			moments[m] /= C[i][curlayer]*(corder+1)*(corder+2);
		}
		curlayer = 1 - curlayer;
		prevlayer = 1 - prevlayer;
	}

#ifdef SHIFT_POLY
	r2d_shift_moments(moments, polyorder, vc);
#endif
}

void r2d_shift_moments(r2d_real* moments, r2d_int polyorder, r2d_rvec2 vc) {

	// var declarations
	r2d_int m, i, j, corder;
	r2d_int mm, mi, mj, mcorder;

	// store moments of a shifted polygon
	r2d_real moments2[R2D_NUM_MOMENTS(polyorder)];
	for(m = 0; m < R2D_NUM_MOMENTS(polyorder); ++m) moments2[m] = 0.0;

	// calculate and save Pascal's triangle
	r2d_real B[polyorder+1][polyorder+1];
	B[0][0] = 1.0;
	for(corder = 1, m = 1; corder <= polyorder; ++corder) {
		for(i = corder; i >= 0; --i, ++m) {
			j = corder - i;
			B[i][corder] = 1.0;
			if(i > 0 && j > 0) B[i][corder] = B[i][corder-1] + B[i-1][corder-1];
		}
	}

	// shift moments back to the original position using
	// \int_\Omega x^i y^j d\vec r =
	// \int_\omega (x+\xi)^i (y+\eta)^j d\vec r =
	// \sum_{a,b,c=0}^{i,j,k} \binom{i}{a} \binom{j}{b}
	// \xi^{i-a} \eta^{j-b} \int_\omega x^a y^b d\vec r
	for(corder = 1, m = 1; corder <= polyorder; ++corder) {
		for(i = corder; i >= 0; --i, ++m) {
			j = corder - i;
			for(mcorder = 0, mm = 0; mcorder <= corder; ++mcorder) {
				for(mi = mcorder; mi >= 0; --mi, ++mm) {
					mj = mcorder - mi;
					if (mi <= i && mj <= j ) {
						moments2[m] += B[mi][i] * B[mj][j] * pow(vc.x,(i-mi)) * pow(vc.y,(j-mj)) * moments[mm];
					}
				}
			}
		}
	}

	// assign shifted moments
	for(m = 1; m < R2D_NUM_MOMENTS(polyorder); ++m)
		moments[m] = moments2[m];
}

r2d_rvec2 r2d_poly_center(r2d_poly* poly) {

	// var declarations
	r2d_int vcur;
	r2d_rvec2 vc;

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts;
	r2d_int* nverts = &poly->nverts;

	vc.x = 0.0;
	vc.y = 0.0;
	for(vcur = 0; vcur < *nverts; ++vcur) {
		vc.x += vertbuffer[vcur].pos.x;
		vc.y += vertbuffer[vcur].pos.y;
	}
	vc.x /= *nverts;
	vc.y /= *nverts;

	return vc;
}

r2d_int r2d_is_good(r2d_poly* poly) {

	r2d_int v;
	r2d_int vct[R2D_MAX_VERTS];

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// consistency check
	memset(&vct, 0, sizeof(vct));
	for(v = 0; v < *nverts; ++v) {

		// return false if vertices share an edge with themselves 
		// or if any edges are obviously invalid
		if(vertbuffer[v].pnbrs[0] == vertbuffer[v].pnbrs[1]) return 0;
		if(vertbuffer[v].pnbrs[0] >= *nverts) return 0;
		if(vertbuffer[v].pnbrs[1] >= *nverts) return 0;

		vct[vertbuffer[v].pnbrs[0]]++;
		vct[vertbuffer[v].pnbrs[1]]++;
	}
	
	// return false if any vertices are pointed to 
	// by more or fewer than two other vertices
	for(v = 0; v < *nverts; ++v) if(vct[v] != 2) return 0;

	return 1;
}

void r2d_rotate(r2d_poly* poly, r2d_real theta) {
	r2d_int v;
	r2d_rvec2 tmp;
	r2d_real sine = sin(theta);
	r2d_real cosine = cos(theta);
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;
		poly->verts[v].pos.x = cosine*tmp.x - sine*tmp.y; 
		poly->verts[v].pos.x = sine*tmp.x + cosine*tmp.y; 
	}
}

void r2d_translate(r2d_poly* poly, r2d_rvec2 shift) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.x += shift.x;
		poly->verts[v].pos.y += shift.y;
	}
}

void r2d_scale(r2d_poly* poly, r2d_real scale) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.x *= scale;
		poly->verts[v].pos.y *= scale;
	}
}

void r2d_shear(r2d_poly* poly, r2d_real shear, r2d_int axb, r2d_int axs) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.xy[axb] += shear*poly->verts[v].pos.xy[axs];
	}
}

void r2d_affine(r2d_poly* poly, r2d_real mat[3][3]) {
	r2d_int v;
	r2d_rvec2 tmp;
	r2d_real w;
	for(v = 0; v < poly->nverts; ++v) {
		tmp = poly->verts[v].pos;

		// affine transformation
		poly->verts[v].pos.x = tmp.x*mat[0][0] + tmp.y*mat[0][1] + mat[0][2];
		poly->verts[v].pos.y = tmp.x*mat[1][0] + tmp.y*mat[1][1] + mat[1][2];
		w = tmp.x*mat[2][0] + tmp.y*mat[2][1] + mat[2][2];
	
		// homogeneous divide if w != 1, i.e. in a perspective projection
		poly->verts[v].pos.x /= w;
		poly->verts[v].pos.y /= w;
	}
}


void r2d_init_box(r2d_poly* poly, r2d_rvec2 rbounds[2]) {

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 
	
	*nverts = 4;
	vertbuffer[0].pnbrs[0] = 1;	
	vertbuffer[0].pnbrs[1] = 3;	
	vertbuffer[1].pnbrs[0] = 2;	
	vertbuffer[1].pnbrs[1] = 0;	
	vertbuffer[2].pnbrs[0] = 3;	
	vertbuffer[2].pnbrs[1] = 1;	
	vertbuffer[3].pnbrs[0] = 0;	
	vertbuffer[3].pnbrs[1] = 2;	
	vertbuffer[0].pos.x = rbounds[0].x; 
	vertbuffer[0].pos.y = rbounds[0].y; 
	vertbuffer[1].pos.x = rbounds[1].x; 
	vertbuffer[1].pos.y = rbounds[0].y; 
	vertbuffer[2].pos.x = rbounds[1].x; 
	vertbuffer[2].pos.y = rbounds[1].y; 
	vertbuffer[3].pos.x = rbounds[0].x; 
	vertbuffer[3].pos.y = rbounds[1].y; 

}


void r2d_init_poly(r2d_poly* poly, r2d_rvec2* vertices, r2d_int numverts) {

	// direct access to vertex buffer
	r2d_vertex* vertbuffer = poly->verts; 
	r2d_int* nverts = &poly->nverts; 

	// init the poly
	*nverts = numverts;
	r2d_int v;
	for(v = 0; v < *nverts; ++v) {
		vertbuffer[v].pos = vertices[v];
		vertbuffer[v].pnbrs[0] = (v+1)%(*nverts);
		vertbuffer[v].pnbrs[1] = (*nverts+v-1)%(*nverts);
	}
}


void r2d_box_faces_from_verts(r2d_plane* faces, r2d_rvec2* rbounds) {
	faces[0].n.x = 0.0; faces[0].n.y = 1.0; faces[0].d = rbounds[0].y; 
	faces[1].n.x = 1.0; faces[1].n.y = 0.0; faces[1].d = rbounds[0].x; 
	faces[2].n.x = 0.0; faces[2].n.y = -1.0; faces[2].d = rbounds[1].y; 
	faces[3].n.x = -1.0; faces[3].n.y = 0.0; faces[3].d = rbounds[1].x; 
}

void r2d_poly_faces_from_verts(r2d_plane* faces, r2d_rvec2* vertices, r2d_int numverts) {

	// dummy vars
	r2d_int f;
	r2d_rvec2 p0, p1;

	// calculate a centroid and a unit normal for each face 
	for(f = 0; f < numverts; ++f) {

		p0 = vertices[f];
		p1 = vertices[(f+1)%numverts];

		// normal of the edge
		faces[f].n.x = p0.y - p1.y;
		faces[f].n.y = p1.x - p0.x;

		// normalize the normals and set the signed distance to origin
		norm(faces[f].n);
		faces[f].d = -dot(faces[f].n, p0);

	}
}

r2d_real r2d_orient(r2d_rvec2 pa, r2d_rvec2 pb, r2d_rvec2 pc) {
	return 0.5*((pa.x - pc.x)*(pb.y - pc.y) - (pb.x - pc.x)*(pa.y - pc.y)); 
}

void r2d_print(r2d_poly* poly) {
	r2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		printf("  vertex %d: pos = ( %.10e , %.10e ), nbrs = %d %d\n", 
				v, poly->verts[v].pos.x, poly->verts[v].pos.y, poly->verts[v].pnbrs[0], poly->verts[v].pnbrs[1]);
	}
}

