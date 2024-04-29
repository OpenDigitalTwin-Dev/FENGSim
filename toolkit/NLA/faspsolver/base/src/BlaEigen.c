/*! \file  BlaEigen.c
 *
 *  \brief Computing the extreme eigenvalues
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxVector.c, BlaArray.c, BlaSpmvCSR.c, and BlaVector.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn REAL fasp_dcsr_maxeig (const dCSRmat *A, const REAL tol, const INT maxit)
 *
 * \brief Approximate the largest eigenvalue of A by the power method
 *
 * \param A      Pointer to the dCSRmat matrix
 * \param tol    Tolerance for stopping the power method
 * \param maxit  Max number of iterations
 *
 * \return       Largest eigenvalue
 *
 * \author Xiaozhe Hu
 * \date   01/25/2011
 */
REAL fasp_dcsr_maxeig (const dCSRmat  *A,
                       const REAL      tol,
                       const INT       maxit)
{
    REAL eigenvalue = 0.0, temp = 1.0, L2_norm_y;
    dvector x, y;
    int i;

    fasp_dvec_alloc(A->row, &x);
    fasp_dvec_rand(A->row,&x);
    fasp_blas_darray_ax(A->row, 1.0/fasp_blas_dvec_norm2(&x), x.val);
    
    fasp_dvec_alloc(A->row, &y);
    
    for ( i = maxit; i--; ) {
        // y = Ax;
        fasp_blas_dcsr_mxv(A, x.val, y.val);
        
        // y/||y||
        L2_norm_y = fasp_blas_dvec_norm2(&y);
        fasp_blas_darray_ax(A->row, 1.0/L2_norm_y, y.val);
        
        // eigenvalue = y'Ay;
        eigenvalue = fasp_blas_dcsr_vmv(A, y.val, y.val);
        
        // convergence test
        if ( (ABS(eigenvalue - temp)/ABS(temp)) < tol ) break;
        
        fasp_dvec_cp(&y, &x);
        temp = eigenvalue;
    }
    
    // clean up memory
    fasp_dvec_free(&x);
    fasp_dvec_free(&y);
    
    return eigenvalue;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
