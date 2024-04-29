/*! \file  FemParam.c
 *
 *  \brief Miscellaneous functions for setting parameters of FEM
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "misc.h"

/**
 * \fn int FEM_param_set (int argc, const char *argv, FEM_param *pt)
 *
 * \brief read input from arguments
 *
 * \param argc       Number of arg input
 * \param argv       Input arguments
 * \param pt         Parameters to be set
 *
 * \author Feiteng Huang
 * \date   04/05/2009
 */
int FEM_param_set(int argc, const char *argv [], FEM_param * pt)
{
    int arg_index = 1;
    int print_usage = 0;
    int input_flag = 1;
    
    while (arg_index < argc) {
        
        if (argc%2 == 0) {
            print_usage = 1;
            break;
        }
        
        if ( strcmp(argv[arg_index], "-help") == 0 ) {
            print_usage = 1;
            break;
        }
        
        if ( strcmp(argv[arg_index], "-meshin") == 0 ) {
            arg_index ++;
            strcpy(pt->meshIn, argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-meshout") == 0 ) {
            arg_index ++;
            strcpy(pt->meshOut, argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-assemble") == 0 ) {
            arg_index ++;
            strcpy(pt->option, argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-refine") == 0 ) {
            arg_index ++;
            pt->refine_lvl = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-nt") == 0 ) {
            arg_index ++;
            pt->nt = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-T") == 0 ) {
            arg_index ++;
            pt->T = atof(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-output") == 0 ) {
            arg_index ++;
            pt->mesh_out = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-quad_rhs") == 0 ) {
            arg_index ++;
            pt->num_qp_rhs = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if ( strcmp(argv[arg_index], "-quad_mat") == 0 ) {
            arg_index ++;
            pt->num_qp_mat = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if (input_flag) {
            print_usage = 1;
            break;
        }
        input_flag = 1;
    }
    
    return print_usage;
}

/**
 * \fn void FEM_param_init (FEM_param *pt)
 *
 * \brief Initialize input arguments
 *
 * \param pt     Parameters to be set
 *
 * \author Feiteng Huang
 * \date   04/05/2009
 */
void FEM_param_init (FEM_param *pt)
{
    strcpy(pt->meshIn, "./data/testmesh.dat");
    strcpy(pt->meshOut, "./out/mesh_out.dat");
    strcpy(pt->option, "ab");
    
    pt->refine_lvl  = 8;    // default value
    pt->nt          = 1;    // time steps
    pt->T           = 1.0;  // final time
    pt->mesh_out    = 0;    // output mesh or not
    pt->num_qp_rhs  = 3;    // for P1 FEM, smooth right-hand-side
    pt->num_qp_mat  = 1;    // for P1 FEM, stiffness matrix
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
