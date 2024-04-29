/*! \file misc.h
 *  \brief miscellaneous for test.
 */

#ifndef _MISC_H_
#define _MISC_H_

#include <string.h>
#include <stdlib.h>

/** 
 * \struct FEM_param
 * \brief param for FEM test problems
 */ 
typedef struct FEM_param{
    
    char meshIn[128];
    char meshOut[128];
    char option[128];

    int refine_lvl;
    int nt;
    double T;
    int mesh_out;
    int num_qp_rhs;
    int num_qp_mat;
    
} FEM_param;

void FEM_param_init (FEM_param *pt);

int FEM_param_set (int argc, const char *argv [], FEM_param * pt);

#endif
