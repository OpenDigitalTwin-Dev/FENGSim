/**************************************************************
 *
 *    voxelize.c
 *    
 *    Devon Powell
 *    26 May 2015
 *
 *    Example of the voxelization functionality of r3d.
 *    Voxelizes a single tet and prints the results.
 *    See r3d.h for detailed documentation.
 *    
 *    Copyright (C) 2014 Stanford University.
 *    See License.txt for more information.
 *
 *************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "r3d.h"

int main() {

    // variable declarations for counters and such
    r3d_int m, v, i, j, k;


    //// Initialize the tetrahedron to be voxelized ////
	
    // just some arbitrary vertex positions
    r3d_rvec3 verts[4] = {
	{0.0, 0.0, 0.0}, 
	{1.0, 0.0, 0.0}, 
	{0.0, 1.0, 0.0}, 
	{0.0, 0.0, 1.0} 
    };
    printf("Initialized a tetrahedron with vertices:\n");
    for(v = 0; v < 4; ++v) printf("  ( %f , %f , %f )\n", verts[v].x, verts[v].y, verts[v].z);


    //// Initialize the destination grid ////
	
    // This struct contains grid parameters, 
    // as well as buffers for the voxelized moments
    r3d_dest_grid vox_grid;
    // make a 64^3 grid for our example
    vox_grid.n.i = 64;
    vox_grid.n.j = 64;
    vox_grid.n.k = 64;
    // grid cell spacing in each dimension, for a unit-cube
    // Important: vox_grid coordinates begin at the origin!
    vox_grid.d.x = 1.0/vox_grid.n.i;
    vox_grid.d.y = 1.0/vox_grid.n.j;
    vox_grid.d.z = 1.0/vox_grid.n.k;
    // polynomial order to be voxelized
    // 0 for constant, 1 for linear, 2 for quadratic
    vox_grid.polyorder = 2;
    // allocate buffers
    vox_grid.bufsz = (r3d_long) (1.5*vox_grid.n.i*vox_grid.n.j*vox_grid.n.k); // a bit of overkill
    r3d_int mmax = r3d_num_moments[vox_grid.polyorder]; // number of moments for a given polynomial order
    for(m = 0; m < mmax; ++m) // moment buffers
	vox_grid.moments[m] = (r3d_real*) malloc(vox_grid.bufsz*sizeof(r3d_real));
    vox_grid.orient = (r3d_orientation*) malloc(vox_grid.bufsz*sizeof(r3d_orientation)); // internal buffer for geometric checks
    printf("Initialized the destination grid:\n");
    printf("  res = ( %d , %d , %d )\n", vox_grid.n.i, vox_grid.n.j, vox_grid.n.k);
    printf("  dx = ( %f , %f , %f )\n", vox_grid.d.x, vox_grid.d.y, vox_grid.d.z);
    printf("  bounds = ( %f , %f , %f ) to ( %f , %f , %f )\n", 
	   0.0, 0.0, 0.0, vox_grid.d.x*vox_grid.n.i, vox_grid.d.y*vox_grid.n.j, vox_grid.d.z*vox_grid.n.k);


    //// Voxelize the tetrahedron onto the main grid ////

    // make sure it is oriented with positive volume
    r3d_real tetvol = r3du_orient(verts[0], verts[1], verts[2], verts[3]);
    if(tetvol < 0.0) {
	r3d_rvec3 swap = verts[2];
	verts[2] = verts[3];
	verts[3] = swap;
	tetvol = -tetvol;
    }
    // get the tet in its face representation
    r3d_plane faces[4];
    r3du_tet_faces_from_verts(&verts[0], &faces[0]);
    // voxelize the tet
    printf("Voxelizing...\n");
    r3d_voxelize_tet(faces, &vox_grid);
    printf("  done.\n");


    //// Check the results ////
	
    // sum moments over the voxel buffer
    r3d_real moments_vox[10];
    for(m = 0; m < 10; ++m)
	moments_vox[m] = 0.0;

#define vox_ind(ii, jj, kk) (vox_grid.n.j*vox_grid.n.k*(ii) + vox_grid.n.k*(jj) + (kk))

    for(i = 0; i < vox_grid.n.i; ++i) 
	for(j = 0; j < vox_grid.n.j; ++j)
	    for(k = 0; k < vox_grid.n.k; ++k) {
		for(m = 0; m < mmax; ++m) {
		    moments_vox[m] += vox_grid.moments[m][vox_ind(i, j, k)];
		}
	    }

    int n_cell = 0;
    for(i = 0; i < vox_grid.n.i; ++i) 
	for(j = 0; j < vox_grid.n.j; ++j)
	    for(k = 0; k < vox_grid.n.k; ++k) {
		if (vox_grid.moments[0][vox_ind(i, j, k)]>0)
		    //printf("%f",vox_grid.moments[0][vox_ind(i, j, k)]);
		    n_cell++;
	    }
    
    
    FILE *fp;
    fp = fopen("vox_example.vtk","w");
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Structured Grid by Portage\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d float\n", n_cell*8);
    for(i = 0; i < vox_grid.n.i; ++i) {
	for(j = 0; j < vox_grid.n.j; ++j) {
	    for(k = 0; k < vox_grid.n.k; ++k) {
		if (vox_grid.moments[0][vox_ind(i, j, k)]>0) {
		    double p1[3];
		    p1[0] = 0 + i*vox_grid.d.x;
		    p1[1] = 0 + j*vox_grid.d.y;
		    p1[2] = 0 + k*vox_grid.d.z;
		    double p2[3];
		    p2[0] = vox_grid.d.x + i*vox_grid.d.x;
		    p2[1] = 0 + j*vox_grid.d.y;
		    p2[2] = 0 + k*vox_grid.d.z;
		    double p3[3];
		    p3[0] = 0 + i*vox_grid.d.x;
		    p3[1] = vox_grid.d.y + j*vox_grid.d.y;
		    p3[2] = 0 + k*vox_grid.d.z;
		    double p4[3];
		    p4[0] = vox_grid.d.x + i*vox_grid.d.x;
		    p4[1] = vox_grid.d.y + j*vox_grid.d.y;
		    p4[2] = 0 + k*vox_grid.d.z;
		    double p5[3];
		    p5[0] = 0 + i*vox_grid.d.x;
		    p5[1] = 0 + j*vox_grid.d.y;
		    p5[2] = vox_grid.d.z + k*vox_grid.d.z;
		    double p6[3];
		    p6[0] = vox_grid.d.x + i*vox_grid.d.x;
		    p6[1] = 0 + j*vox_grid.d.y;
		    p6[2] = vox_grid.d.z + k*vox_grid.d.z;
		    double p7[3];
		    p7[0] = 0 + i*vox_grid.d.x;
		    p7[1] = vox_grid.d.y + j*vox_grid.d.y;
		    p7[2] = vox_grid.d.z + k*vox_grid.d.z;
		    double p8[3];
		    p8[0] = vox_grid.d.x + i*vox_grid.d.x;
		    p8[1] = vox_grid.d.y + j*vox_grid.d.y;
		    p8[2] = vox_grid.d.z + k*vox_grid.d.z;
		    fprintf(fp, "%f %f %f\n", p1[0], p1[1], p1[2]);
		    fprintf(fp, "%f %f %f\n", p2[0], p2[1], p2[2]);
		    fprintf(fp, "%f %f %f\n", p3[0], p3[1], p3[2]);
		    fprintf(fp, "%f %f %f\n", p4[0], p4[1], p4[2]);
		    fprintf(fp, "%f %f %f\n", p5[0], p5[1], p5[2]);
		    fprintf(fp, "%f %f %f\n", p6[0], p6[1], p6[2]);
		    fprintf(fp, "%f %f %f\n", p7[0], p7[1], p7[2]);
		    fprintf(fp, "%f %f %f\n", p8[0], p8[1], p8[2]);
		}   
	    }
	}
    }
    
    fprintf(fp, "CELLS %d %d\n", n_cell, n_cell*9);
    for (int i=0; i<n_cell; i++) {
        fprintf(fp, "8 %d %d %d %d %d %d %d %d\n", 0+i*8, 1+i*8, 2+i*8, 3+i*8, 4+i*8, 5+i*8, 6+i*8, 7+i*8);
    }
    fprintf(fp, "CELL_TYPES %d\n", n_cell);
    for (int i=0; i<n_cell; i++) {
        fprintf(fp, "11\n"); 
    }
    fclose(fp);
    
    printf("Moments (sum over voxels):\n");
    printf("  Integral[ 1 dV ] = %f\n",   moments_vox[0]);
    printf("  Integral[ x dV ] = %f\n",   moments_vox[1]);
    printf("  Integral[ y dV ] = %f\n",   moments_vox[2]);
    printf("  Integral[ z dV ] = %f\n",   moments_vox[3]);
    printf("  Integral[ x^2 dV ] = %f\n", moments_vox[4]);
    printf("  Integral[ y^2 dV ] = %f\n", moments_vox[5]);
    printf("  Integral[ z^2 dV ] = %f\n", moments_vox[6]);
    printf("  Integral[ x*y dV ] = %f\n", moments_vox[7]);
    printf("  Integral[ y*z dV ] = %f\n", moments_vox[8]);
    printf("  Integral[ z*x dV ] = %f\n", moments_vox[9]);
	
    // compute the moments directly from the input tetrahedron
    r3d_real moments_tet[10];
    r3d_rvec3 v0 = verts[0];
    r3d_rvec3 v1 = verts[1];
    r3d_rvec3 v2 = verts[2];
    r3d_rvec3 v3 = verts[3];
    moments_tet[0] = tetvol;
    moments_tet[1] = tetvol*0.25*(v0.x + v1.x + v2.x + v3.x);
    moments_tet[2] = tetvol*0.25*(v0.y + v1.y + v2.y + v3.y);
    moments_tet[3] = tetvol*0.25*(v0.z + v1.z + v2.z + v3.z);
    moments_tet[4] = tetvol*0.1*(v0.x*v0.x + v1.x*v1.x + v2.x*v2.x + v3.x*v3.x + v0.x*v1.x + v0.x*v2.x + v0.x*v3.x + v1.x*v2.x + v1.x*v3.x + v2.x*v3.x);
    moments_tet[5] = tetvol*0.1*(v0.y*v0.y + v1.y*v1.y + v2.y*v2.y + v3.y*v3.y + v0.y*v1.y + v0.y*v2.y + v0.y*v3.y + v1.y*v2.y + v1.y*v3.y + v2.y*v3.y);
    moments_tet[6] = tetvol*0.1*(v0.z*v0.z + v1.z*v1.z + v2.z*v2.z + v3.z*v3.z + v0.z*v1.z + v0.z*v2.z + v0.z*v3.z + v1.z*v2.z + v1.z*v3.z + v2.z*v3.z);
    moments_tet[7] = tetvol*0.05*(2*v0.x*v0.y + v1.x*v0.y + v2.x*v0.y + v3.x*v0.y + v0.x*v1.y + 2*v1.x*v1.y + v2.x*v1.y + v3.x*v1.y 
				  + v0.x*v2.y + v1.x*v2.y + 2*v2.x*v2.y + v3.x*v2.y + v0.x*v3.y + v1.x*v3.y + v2.x*v3.y + 2*v3.x*v3.y);
    moments_tet[8] = tetvol*0.05*(2*v0.y*v0.z + v1.y*v0.z + v2.y*v0.z + v3.y*v0.z + v0.y*v1.z + 2*v1.y*v1.z + v2.y*v1.z + v3.y*v1.z 
				  + v0.y*v2.z + v1.y*v2.z + 2*v2.y*v2.z + v3.y*v2.z + v0.y*v3.z + v1.y*v3.z + v2.y*v3.z + 2*v3.y*v3.z);
    moments_tet[9] = tetvol*0.05*(2*v0.x*v0.z + v1.x*v0.z + v2.x*v0.z + v3.x*v0.z + v0.x*v1.z + 2*v1.x*v1.z + v2.x*v1.z + v3.x*v1.z 
				  + v0.x*v2.z + v1.x*v2.z + 2*v2.x*v2.z + v3.x*v2.z + v0.x*v3.z + v1.x*v3.z + v2.x*v3.z + 2*v3.x*v3.z);
    printf("Moments (directly from input tetrahedron):\n");
    printf("  Integral[ 1 dV ] = %f\n",   moments_tet[0]);
    printf("  Integral[ x dV ] = %f\n",   moments_tet[1]);
    printf("  Integral[ y dV ] = %f\n",   moments_tet[2]);
    printf("  Integral[ z dV ] = %f\n",   moments_tet[3]);
    printf("  Integral[ x^2 dV ] = %f\n", moments_tet[4]);
    printf("  Integral[ y^2 dV ] = %f\n", moments_tet[5]);
    printf("  Integral[ z^2 dV ] = %f\n", moments_tet[6]);
    printf("  Integral[ x*y dV ] = %f\n", moments_tet[7]);
    printf("  Integral[ y*z dV ] = %f\n", moments_tet[8]);
    printf("  Integral[ z*x dV ] = %f\n", moments_tet[9]);
	
    // print errors
    printf("Errors:\n");
    printf("  Integral[ 1 dV ] = %e\n",   1.0 - moments_vox[0]/moments_tet[0]);
    printf("  Integral[ x dV ] = %e\n",   1.0 - moments_vox[1]/moments_tet[1]);
    printf("  Integral[ y dV ] = %e\n",   1.0 - moments_vox[2]/moments_tet[2]);
    printf("  Integral[ z dV ] = %e\n",   1.0 - moments_vox[3]/moments_tet[3]);
    printf("  Integral[ x^2 dV ] = %e\n", 1.0 - moments_vox[4]/moments_tet[4]);
    printf("  Integral[ y^2 dV ] = %e\n", 1.0 - moments_vox[5]/moments_tet[5]);
    printf("  Integral[ z^2 dV ] = %e\n", 1.0 - moments_vox[6]/moments_tet[6]);
    printf("  Integral[ x*y dV ] = %e\n", 1.0 - moments_vox[7]/moments_tet[7]);
    printf("  Integral[ y*z dV ] = %e\n", 1.0 - moments_vox[8]/moments_tet[8]);
    printf("  Integral[ z*x dV ] = %e\n", 1.0 - moments_vox[9]/moments_tet[9]);
	

    //// free the destination grid buffers ////
	
    for(m = 0; m < mmax; ++m)
	free(vox_grid.moments[m]);
    free(vox_grid.orient);

    return 0;
}
