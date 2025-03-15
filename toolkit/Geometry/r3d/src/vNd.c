/*
 *
 *  	vNd.c
 *
 *  	See vNd.h for usage.
 *
 *  	Devon Powell
 *  	9 August 2016
 *
 *		Copyright (c) 2015, The Board of Trustees of the Leland Stanford Junior University, 
 *		through SLAC National Accelerator Laboratory (subject to receipt of any required approvals 
 *		from the U.S. Dept. of Energy). All rights reserved. Redistribution and use in source and 
 *		binary forms, with or without modification, are permitted provided that the following 
 *		conditions are met: 
 *		(1) Redistributions of source code must retain the above copyright notice, 
 *		this list of conditions and the following disclaimer. 
 *		(2) Redistributions in binary form must reproduce the above copyright notice, 
 *		this list of conditions and the following disclaimer in the documentation and/or other 
 *		materials provided with the distribution. 
 *		(3) Neither the name of the Leland Stanford Junior University, SLAC National Accelerator 
 *		Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to endorse 
 *		or promote products derived from this software without specific prior written permission. 
 *		
 *		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
 *		OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
 *		MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 *		COPYRIGHT OWNER, THE UNITED STATES GOVERNMENT, OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 *		INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 *		LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
 *		BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 *		STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
 *		USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *		
 *		You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to 
 *		the features, functionality or performance of the source code ("Enhancements") to anyone; 
 *		however, if you choose to make your Enhancements available either publicly, or directly to 
 *		SLAC National Accelerator Laboratory, without imposing a separate written license agreement 
 *		for such Enhancements, then you hereby grant the following license: a non-exclusive, 
 *		royalty-free perpetual license to install, use, modify, prepare derivative works, 
 *		incorporate into other computer software, distribute, and sublicense such Enhancements or 
 *		derivative works thereof, in binary and source code form.
 *
 */

#include "vNd.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

// TODO: make this a generic "split" routine that just takes a plane.
void rNd_split(rNd_poly* inpoly, rNd_poly** outpolys, rNd_real coord, rNd_int ax);

void rNd_voxelize(rNd_poly* poly, rNd_dvec ibox[2], rNd_real* dest_grid, rNd_rvec d, rNd_int polyorder) {

	rNd_int i, j, m, ncell, spax, dmax, nstack, siz;
	rNd_long gind;
	rNd_poly* children[2];
	rNd_dvec gridsz, stride;

	// return if any parameters are bad 
	// calculate grid size 
	if(!poly || poly->nverts <= 0 || !dest_grid) return;
	for(i = 0; i < RND_DIM; ++i) {
		gridsz.ijk[i] = ibox[1].ijk[i]-ibox[0].ijk[i];	
		if(gridsz.ijk[i] <= 0) return; 
	}

	// get the number of grid cells
	// initialize the grid strides for easy row-major indexing
	ncell = 1; 
	for(i = 0; i < RND_DIM; ++i) {
		ncell *= gridsz.ijk[i]; 
		stride.ijk[i] = 1;
		for(j = i+1; j < RND_DIM; ++j) stride.ijk[i] *= gridsz.ijk[j]; 
	}

	// Only zeroth-order moments for now
	// TODO: rNd_int nmom = RND_NUM_MOMENTS(polyorder);
	rNd_int nmom = 1; 
	rNd_real moments[nmom];
	
	// explicit stack-based implementation
	// stack size should never overflow
	struct {
		rNd_poly poly;
		rNd_dvec ibox[2];
	} stack[(rNd_int)(ceil(log2(ncell))+RND_DIM+1)];

	// push the original polyhedron onto the stack
	// and recurse until child polyhedra occupy single voxels
	nstack = 0;
	stack[nstack].poly = *poly;
	memcpy(stack[nstack].ibox, ibox, 2*sizeof(rNd_dvec));
	nstack++;
	while(nstack > 0) {

		// pop the stack
		// if the leaf is empty, skip it
		--nstack;
		if(stack[nstack].poly.nverts <= 0) continue;
		
		// find the longest axis along which to split 
		dmax = 0;
		spax = 0;
		for(i = 0; i < RND_DIM; ++i) {
			siz = stack[nstack].ibox[1].ijk[i]-stack[nstack].ibox[0].ijk[i];
			if(siz > dmax) {
				dmax = siz; 
				spax = i;
			}	
		}

		// if all three axes are only one voxel long, reduce the single voxel to the dest grid
		if(dmax == 1) {
			rNd_reduce(&stack[nstack].poly, moments, polyorder);

			// calculate the grid index by dotting with the strides
			// TODO: give this its own macro?
			gind = 0;
			for(i = 0; i < RND_DIM; ++i) 
				gind += (stack[nstack].ibox[0].ijk[i]-ibox[0].ijk[i])*stride.ijk[i];

			// TODO: cell shifting for accuracy
			for(m = 0; m < nmom; ++m) dest_grid[nmom*gind+m] += moments[m];
			
			continue;
		}

		// split the poly and push children to the stack
		children[0] = &stack[nstack].poly;
		children[1] = &stack[nstack+1].poly;
		rNd_split(&stack[nstack].poly, children, d.xyz[spax]*(stack[nstack].ibox[0].ijk[spax]+dmax/2), spax);
		memcpy(stack[nstack+1].ibox, stack[nstack].ibox, 2*sizeof(rNd_dvec));
		stack[nstack].ibox[1].ijk[spax] -= dmax-dmax/2; 
		stack[nstack+1].ibox[0].ijk[spax] += dmax/2;
		nstack += 2;
	}
}

void rNd_split(rNd_poly* inpoly, rNd_poly** outpolys, rNd_real coord, rNd_int ax) {

	// naive split for now
	*outpolys[0] = *inpoly;
	*outpolys[1] = *inpoly;
	rNd_plane splane;
	memset(&splane, 0, sizeof(splane));
	splane.n.xyz[ax] = -1.0;
	splane.d = coord;
	rNd_clip(outpolys[0], &splane, 1);
	splane.n.xyz[ax] *= -1; 
	splane.d *= -1;
	rNd_clip(outpolys[1], &splane, 1);

#if 0

	// direct access to vertex buffer
	if(inpoly->nverts <= 0) return;
	rNd_int* nverts = &inpoly->nverts;
	rNd_vertex* vertbuffer = inpoly->verts; 
	rNd_int v, np, npnxt, onv, vcur, vnext, vstart, pnext, nright, cside;
	rNd_rvec newpos;
	rNd_int side[RND_MAX_VERTS];
	rNd_real sdists[RND_MAX_VERTS];

	// calculate signed distances to the clip plane
	nright = 0;
	memset(&side, 0, sizeof(side));
	for(v = 0; v < *nverts; ++v) {
			//sdists[v] = splane.d + rNd_dot(vertbuffer[v].pos, splane.n);
		sdists[v] = coord - vertbuffer[v].pos.xyz[ax];
		if(sdists[v] < 0.0) {
			side[v] = 1;
			nright++;
		}
	}

	// return if the poly lies entirely on one side of it 
	if(nright == 0) {
		*(outpolys[0]) = *inpoly;
		outpolys[1]->nverts = 0;
		return;
	}
	if(nright == *nverts) {
		*(outpolys[1]) = *inpoly;
		outpolys[0]->nverts = 0;
		return;
	}

	// check all edges and insert new vertices on the bisected edges 
	onv = inpoly->nverts;
	for(vcur = 0; vcur < onv; ++vcur) {
		if(side[vcur]) continue;
		for(np = 0; np < 3; ++np) {
			vnext = vertbuffer[vcur].pnbrs[np];
			if(!side[vnext]) continue;
			wav(vertbuffer[vcur].pos, -sdists[vnext],
				vertbuffer[vnext].pos, sdists[vcur],
				newpos);
			vertbuffer[*nverts].pos = newpos;
			vertbuffer[*nverts].pnbrs[0] = vcur;
			vertbuffer[vcur].pnbrs[np] = *nverts;
			(*nverts)++;
			vertbuffer[*nverts].pos = newpos;
			side[*nverts] = 1;
			vertbuffer[*nverts].pnbrs[0] = vnext;
			for(npnxt = 0; npnxt < 3; ++npnxt) 
				if(vertbuffer[vnext].pnbrs[npnxt] == vcur) break;
			vertbuffer[vnext].pnbrs[npnxt] = *nverts;
			(*nverts)++;
		}
	}

	// for each new vert, search around the faces for its new neighbors
	// and doubly-link everything
	for(vstart = onv; vstart < *nverts; ++vstart) {
		vcur = vstart;
		vnext = vertbuffer[vcur].pnbrs[0];
		do {
			for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
			vcur = vnext;
			pnext = (np+1)%3;
			vnext = vertbuffer[vcur].pnbrs[pnext];
		} while(vcur < onv);
		vertbuffer[vstart].pnbrs[2] = vcur;
		vertbuffer[vcur].pnbrs[1] = vstart;
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
		for(np = 0; np < 3; ++np)
			outpolys[0]->verts[v].pnbrs[np] = side[outpolys[0]->verts[v].pnbrs[np]];
	for(v = 0; v < outpolys[1]->nverts; ++v) 
		for(np = 0; np < 3; ++np)
			outpolys[1]->verts[v].pnbrs[np] = side[outpolys[1]->verts[v].pnbrs[np]];
#endif
}

void rNd_get_ibox(rNd_poly* poly, rNd_dvec ibox[2], rNd_rvec d) {
	rNd_int i, v;
	rNd_rvec rbox[2];
	for(i = 0; i < RND_DIM; ++i) {
		rbox[0].xyz[i] = 1.0e30;
		rbox[1].xyz[i] = -1.0e30;
	}
	for(v = 0; v < poly->nverts; ++v) {
		for(i = 0; i < RND_DIM; ++i) {
			if(poly->verts[v].pos.xyz[i] < rbox[0].xyz[i]) rbox[0].xyz[i] = poly->verts[v].pos.xyz[i];
			if(poly->verts[v].pos.xyz[i] > rbox[1].xyz[i]) rbox[1].xyz[i] = poly->verts[v].pos.xyz[i];
		}
	}
	for(i = 0; i < RND_DIM; ++i) {
		ibox[0].ijk[i] = floor(rbox[0].xyz[i]/d.xyz[i]);
		ibox[1].ijk[i] = ceil(rbox[1].xyz[i]/d.xyz[i]);
	}
}

void rNd_clamp_ibox(rNd_poly* poly, rNd_dvec ibox[2], rNd_dvec clampbox[2], rNd_rvec d) {
	rNd_int i, nboxclip;
	rNd_plane boxfaces[6];
	nboxclip = 0;
	memset(boxfaces, 0, sizeof(boxfaces));
	for(i = 0; i < RND_DIM; ++i) {
		if(ibox[1].ijk[i] <= clampbox[0].ijk[i] || ibox[0].ijk[i] >= clampbox[1].ijk[i]) {
			memset(ibox, 0, 2*sizeof(rNd_dvec));
			poly->nverts = 0;
			return;
		}
		if(ibox[0].ijk[i] < clampbox[0].ijk[i]) {
			ibox[0].ijk[i] = clampbox[0].ijk[i];
			boxfaces[nboxclip].d = -clampbox[0].ijk[i]*d.xyz[i];
			boxfaces[nboxclip].n.xyz[i] = 1.0;
			nboxclip++;
		}
		if(ibox[1].ijk[i] > clampbox[1].ijk[i]) {
			ibox[1].ijk[i] = clampbox[1].ijk[i];
			boxfaces[nboxclip].d = clampbox[1].ijk[i]*d.xyz[i];
			boxfaces[nboxclip].n.xyz[i] = -1.0;
			nboxclip++;
		}	
	}
	if(nboxclip) rNd_clip(poly, boxfaces, nboxclip);
}
