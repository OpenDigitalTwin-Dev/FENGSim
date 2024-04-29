/*! \file poisson_fem.h
 *  \brief Main header file for the Finite Element Methods
 */

#ifndef _SETUP_POISSON_H_
#define _SETUP_POISSON_H_

#include "misc.h"
#include "mesh.h"
#include "assemble.h"
#include "fasp.h"

extern REAL f(REAL *p);
extern REAL u(REAL *p);

INT setup_poisson (dCSRmat *A,
                   dvector *b, 
                   Mesh *mesh, 
                   Mesh_aux *mesh_aux, 
                   FEM_param *pt, 
                   dvector *ptr_uh, 
                   ivector *dof);

REAL get_l2_error_poisson (ddenmat *node,
                           idenmat *elem,
                           dvector *uh,
                           INT num_qp);

#endif
