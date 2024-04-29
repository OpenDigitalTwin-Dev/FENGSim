/*! \file  ItrSmootherCSRcr.c
 *
 *  \brief Smoothers for dCSRmat matrices using compatible relaxation
 *
 *  \note  Restricted smoothers for compatible relaxation, C/F smoothing, etc.
 *
 *  \note  This file contains Level-2 (Itr) functions. It requires:
 *         AuxMessage.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  // TODO: Need to optimize routines here! --Chensong
 */

#include <math.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_smoother_dcsr_gscr (INT pt, INT n, REAL *u, INT *ia, INT *ja, 
 *                                   REAL *a, REAL *b, INT L, INT *CF)
 *
 * \brief Gauss Seidel method restriced to a block 
 *
 * \param pt   Relax type, e.g., cpt, fpt, etc.. 
 * \param n    Number of variables
 * \param u    Iterated solution 
 * \param ia   Row pointer
 * \param ja   Column index
 * \param a    Pointers to sparse matrix values in CSR format
 * \param b    Pointer to right hand side
 * \param L    Number of iterations
 * \param CF   Marker for C, F points 
 *
 * \author James Brannick
 * \date   09/07/2010
 *
 * \note Gauss Seidel CR smoother (Smoother_Type = 99)
 */
void fasp_smoother_dcsr_gscr (INT   pt,
                              INT   n,
                              REAL *u,
                              INT  *ia,
                              INT  *ja,
                              REAL *a,
                              REAL *b, 
                              INT   L,
                              INT  *CF)
{ 
    INT i,j,k,l;
    REAL t, d=0;
    
    for (l=0;l<L;++l) {
        for (i=0;i<n;++i) {
            if (CF[i] == pt) { 
                t=b[i];
                for (k=ia[i];k<ia[i+1];++k) {
                    j=ja[k];
                    if (CF[j] == pt) {
                        if (i!=j) {
                            t-=a[k]*u[j]; 
                        }
                        else { 
                            d=a[k];
                        }
                        if (ABS(d)>SMALLREAL) { 
                            u[i]=t/d;
                        }
                        else {
                            printf("### ERROR: Diagonal entry_%d (%e) close to 0!\n",
                                   i, d);
                            fasp_chkerr(ERROR_MISC, __FUNCTION__);
                        }
                    }
                }
            }
            else {
                u[i]=0.e0;
            }
        } 
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
