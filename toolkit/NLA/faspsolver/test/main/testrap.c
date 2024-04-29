/*! \file  testrap.c
 *
 *  \brief Test speed of RAP implementations
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"
#include "poisson_fem.h"
#include "misc.h"
#include "mesh.h"

/* Test functions f and u for the Poisson's equation */
#include "testfct_poisson.inl"

static void rap_setup(AMG_data *mgl, AMG_param *param)
{
    const int print_level=param->print_level;
    const int m=mgl[0].A.row, n=mgl[0].A.col, nnz=mgl[0].A.nnz; 
    SHORT max_levels=param->max_levels;   
    SHORT level=0;   
    iCSRmat S;
    ivector vertices=fasp_ivec_create(m); // stores level info

    REAL setup_start, setup_end;

    if (print_level>8) printf("rap_setup: %d, %d, %d\n",m,n,nnz);

    // initialize ILU parameters
    mgl->ILU_levels = param->ILU_levels;
    ILU_param iluparam;
    if (param->ILU_levels>0) {
        iluparam.print_level = param->print_level;
        iluparam.ILU_lfil    = param->ILU_lfil;
        iluparam.ILU_droptol = param->ILU_droptol;
        iluparam.ILU_relax   = param->ILU_relax;
        iluparam.ILU_type    = param->ILU_type;
    }
    
    while ((mgl[level].A.row>param->coarse_dof) && (level<max_levels-1)) {

        /*-- setup ILU decomposition if necessary */
        if (level<param->ILU_levels) fasp_ilu_dcsr_setup(&mgl[level].A,&mgl[level].LU,&iluparam);
        
        /*-- Coarsening and form the structure of interpolation --*/
        fasp_amg_coarsening_rs(&mgl[level].A, &vertices, &mgl[level].P, &S, param);
        
        /*-- Form interpolation --*/
        fasp_amg_interp(&mgl[level].A, &vertices, &mgl[level].P, &S, param);
        
        /*-- Form coarse level stiffness matrix --*/
        fasp_dcsr_trans(&mgl[level].P, &mgl[level].R);
        
        /*-- Form coarse level stiffness matrix --*/    
        printf("--------- Level %d ---------- \n", level);

        fasp_gettime(&setup_start);
        fasp_blas_dcsr_rap(&mgl[level].R, &mgl[level].A, &mgl[level].P, &mgl[level+1].A);
        fasp_gettime(&setup_end);

        fasp_cputime("RAP1", setup_end - setup_start);

        fasp_dcsr_free(&mgl[level+1].A);
        
        fasp_gettime(&setup_start);
        fasp_blas_dcsr_ptap(&mgl[level].R, &mgl[level].A, &mgl[level].P, &mgl[level+1].A);
        fasp_gettime(&setup_end);

        fasp_cputime("RAP2", setup_end - setup_start);

        level++;

    }
    
    // setup total level number and current level
    mgl[0].num_levels = max_levels = level+1;
    mgl[0].w = fasp_dvec_create(m); 
    
    for (level=1; level<max_levels; level++) {
        int m = mgl[level].A.row;
        mgl[level].num_levels = max_levels;         
        mgl[level].b = fasp_dvec_create(m);
        mgl[level].x = fasp_dvec_create(m);
        mgl[level].w = fasp_dvec_create(m); 
    }
    
    fasp_ivec_free(&vertices);
}

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for testing sparse RAP subroutines.
 *
 * \author Chensong Zhang
 * \date   04/28/2010
 */
int main(int argc, const char * argv[]) 
{       
    dCSRmat A;
    dvector b;
    dvector uh;
    ivector dof;
    Mesh mesh;
    Mesh_aux mesh_aux;
    int i;
    
    char *inputfile="ini/input.dat";
    input_param Input;
    fasp_param_input(inputfile,&Input);
    
    AMG_param amgparam; // parameters for AMG
    fasp_param_amg_set(&amgparam,&Input);
    
    // Assemble matrix and right-hand side
    FEM_param pt;// parameter for testfem
    FEM_param_init(&pt);
    mesh_init(&mesh, "../data/mesh.dat");
    mesh_aux_build(&mesh, &mesh_aux);

    for (i=0;i<pt.refine_lvl;++i) mesh_refine(&mesh, &mesh_aux);
    setup_poisson(&A, &b, &mesh, &mesh_aux, &pt, &uh, &dof);
    
    {   // Check sprap speed
        const int nnz=A.nnz, m=A.row, n=A.col;
        
        AMG_data *mgl=fasp_amg_data_create(amgparam.max_levels);
        mgl[0].A=fasp_dcsr_create(m,n,nnz); fasp_dcsr_cp(&A,&mgl[0].A);
        mgl[0].b=fasp_dvec_create(n); fasp_dvec_cp(&b,&mgl[0].b);   
        mgl[0].x=fasp_dvec_create(n);
        
        rap_setup(mgl, &amgparam);
    }
    
    // Clean up memory
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    fasp_dvec_free(&uh);
    fasp_ivec_free(&dof);
    mesh_free(&mesh);
    mesh_aux_free(&mesh_aux);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
