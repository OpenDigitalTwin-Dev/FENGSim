/*! \file  testheat.c
 *
 *  \brief The main test function for FASP FEM assembling.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2011--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>
#include "fasp.h"
#include "fasp_functs.h"
#include "misc.h"
#include "mesh.h"
#include "assemble.h"
#include "heat_fem.h"

#define DIM 2

/* Test functions f and u for the heat transfer's equation */
#include "testfct_heat.inl"

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for testing FASP FEM assembling.
 *
 * \author Feiteng Huang
 * \date   04/01/2011
 *
 * Modified by Feiteng Huang on 04/05/2012, for refine & assemble
 */
int main (int argc, const char * argv[]) 
{
    // Set default values
    int status = FASP_SUCCESS;
    int print_usage;
    double mesh_refine_s, mesh_refine_e, assemble_s, assemble_e;

    FEM_param pt; // parameter for testfem
    FEM_param_init(&pt);
    print_usage = FEM_param_set(argc, argv, &pt);

    double dt = pt.T/pt.nt; // time step size
    
    if (print_usage) {
        printf("\nUsage: %s [<options>]\n", argv[0]);
        printf("  -output <val>    : mesh output option [default: 0]\n");
        printf("  -meshin <val>    : input mesh [default: ../data/mesh.dat]\n");
        printf("  -meshout <val>   : output mesh [default: ../data/mesh_?.dat]\n");
        printf("  -refine <val>    : refine level [default: 8]\n");
        printf("  -assemble <val>  : assemble option [default: ab]\n"); 
        printf("                     ab  |  assemble the stiff matrix & mass matrix & rhs;\n");
        printf("                      a  |  assemble the stiff matrix & mass matrix;\n");
        printf("                      b  |  assemble the rhs;\n");
        printf("  -nt     <val>    : time steps [default: 0]\n");
        printf("  -T      <val>    : T_end [default: 1.0]\n");
        printf("  -quad_rhs <val>  : quad points for rhs [default: 3]\n");
        printf("  -quad_mat <val>  : quad points for mat [default: 1]\n");
        printf("  -help            : print this help message\n\n");
        exit(status);
    }
    
    // variables
    Mesh mesh;
    Mesh_aux mesh_aux;
    Bd_apply_info bdinfo;
    
    dCSRmat A_heat, Mass, A; // A_heat = matrix of heat transfer
                             // Mass   = mass matrix 
                             // A      = matrix after boundary condition applied
    
    dvector b_heat, b, bh;   // b_heat = right hand side for all timesteps of heat transfer 
                             // b      = final rhs
                             // bh     = b_heat in 1-step
    
    dvector uh_heat, u0, uh; // uh_heat = boundary value for all timesteps of heat transfer
                             // u0      = last time-step 
                             // uh      = uh_heat in 1-step
    
    int i, it;
    double *l2error, *rhs_pro; //rhs_pro = back Euler's right hand side
    dvector x; //solution of the system
    
    // Step 1: reading mesh info
    mesh_init (&mesh, "../data/mesh.dat");

    // If there is already mesh_aux data available, you can use the following fct to init it:    
    //    mesh_aux_init (&mesh, &mesh_aux, "mesh.aux");
    // otherwise, just build the auxiliary information for the mesh
    mesh_aux_build(&mesh, &mesh_aux);
    
    // Step 2: refinement
    fasp_gettime(&mesh_refine_s);
    for (i=0;i<pt.refine_lvl;++i) mesh_refine(&mesh, &mesh_aux);
    fasp_gettime(&mesh_refine_e);
    fasp_cputime("Mesh refinement", mesh_refine_e - mesh_refine_s);

    // Step 3: assemble for linear system
    fasp_gettime(&assemble_s);
    setup_heat(&A_heat, &Mass, &b_heat, &mesh, &mesh_aux, &pt, &uh_heat, &bdinfo, dt);
    fasp_gettime(&assemble_e);
    fasp_cputime("Assembling", assemble_e - assemble_s);
    
    // Step 3.5: clean auxiliary mesh info
    mesh_aux_free(&mesh_aux);
    
    // init
    uh = fasp_dvec_create(mesh.node.row);
    bh = fasp_dvec_create(mesh.node.row);
    u0 = fasp_dvec_create(mesh.node.row);
    rhs_pro = (double *)fasp_mem_calloc(mesh.node.row, sizeof(double));
    l2error = (double *)fasp_mem_calloc(pt.nt, sizeof(double));
    double p[DIM+1];
    p[DIM] = 0;
    for (i = 0;i < u0.row;++i) {
        for (it = 0;it < DIM;++it) p[it] = mesh.node.val[i][it];
        u0.val[i] = u(p)/dt;
    }
    
    // Step 4: loop over it: apply boundary condition, solve system and get L2 error
    for (it = 0; it < pt.nt; ++it) {

        for (i = 0;i < uh.row;++i) {
            uh.val[i] = uh_heat.val[i+it*mesh.node.row];
            bh.val[i] = b_heat.val[i+it*mesh.node.row];
        }
        fasp_blas_dcsr_mxv(&Mass, u0.val, rhs_pro);//get backward Euler's rhs
        for (i = 0;i < u0.row;++i) bh.val[i] += rhs_pro[i];
        extractNondirichletMatrix(&A_heat, &bh, &A, &b, &(mesh.node_bd), 
                                  &(bdinfo.bd), &(bdinfo.dof), &(bdinfo.idx), &uh);
        
        // Print problem size
        printf("A: m = %d, n = %d, nnz = %d\n", A.row, A.col, A.nnz);
        printf("b: n = %d\n", b.row);
        
        // Solve A x = b with AMG 
        {
            AMG_param amgparam; // parameters for AMG
            
            fasp_param_amg_init(&amgparam); // set AMG param with default values
            amgparam.print_level = PRINT_SOME; // print some AMG message
            amgparam.maxit = 100; // max iter number = 100 
            
            fasp_dvec_alloc(A.row, &x); 
            fasp_dvec_set(A.row,&x,0.0);
            
            fasp_solver_amg(&A, &b, &x, &amgparam);
            
        }
        
        for ( i=0; i<bdinfo.dof.row; ++i)
            uh.val[bdinfo.dof.val[i]] = x.val[i];
        
        for ( i=0; i<u0.row; ++i)
            u0.val[i] = uh.val[i]/dt;
        
        l2error[it] = get_l2_error_heat(&(mesh.node), &(mesh.elem), &uh, pt.num_qp_rhs, dt*(it+1));
        
        fasp_dcsr_free(&A);
        fasp_dvec_free(&b);
    }
    
    printf("\n==============================================================\n");
    for (it = 0;it < pt.nt; ++it) {
        printf("L2 error of FEM at t=%1.5fs is %g\n", dt*(it+1), l2error[it]);
    }
    printf("==============================================================\n");
    
    // Clean up memory
    mesh_free(&mesh);
    fasp_ivec_free(&(bdinfo.dof));
    fasp_ivec_free(&(bdinfo.bd));
    fasp_ivec_free(&(bdinfo.idx));
    fasp_dcsr_free(&A_heat);
    fasp_dcsr_free(&Mass);
    fasp_dvec_free(&b_heat);
    fasp_dvec_free(&bh);
    fasp_dvec_free(&uh);
    fasp_dvec_free(&uh_heat);
    fasp_dvec_free(&u0);
    fasp_dvec_free(&x);
    
    fasp_mem_free(rhs_pro);
    fasp_mem_free(l2error);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
