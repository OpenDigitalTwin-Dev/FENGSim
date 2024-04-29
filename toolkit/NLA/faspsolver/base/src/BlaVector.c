/*! \file  BlaVector.c
 *
 *  \brief BLAS1 operations for vectors
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMessage.c, AuxThreads.c, and BlaArray.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_blas_dvec_axpy (const REAL a, const dvector *x, dvector *y)
 *
 * \brief y = a*x + y
 *
 * \param a   REAL factor a
 * \param x   Pointer to dvector x
 * \param y   Pointer to dvector y
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_blas_dvec_axpy (const REAL     a,
                          const dvector *x,
                          dvector       *y)
{
    const INT   m   = x->row;
    const REAL *xpt = x->val;
    REAL       *ypt = y->val;

    SHORT use_openmp = FALSE;
    INT   i;
    
#ifdef _OPENMP
    INT myid, mybegin, myend, nthreads;
    if ( m > OPENMP_HOLDS ) {
        use_openmp = TRUE;
        nthreads = fasp_get_num_threads();
    }
#endif
    
    if ( y->row != m ) {
        printf("### ERROR: Vectors have different dimensions!\n");
        fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
    }
    
    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid,mybegin,myend,i) num_threads(nthreads)
        {
            myid = omp_get_thread_num();
            fasp_get_start_end (myid, nthreads, m, &mybegin, &myend);
            for ( i = mybegin; i < myend; ++i ) ypt[i] += a*xpt[i];
        }
#endif
    }
    else {
        for ( i = 0; i < m; ++i ) ypt[i] += a*xpt[i];
    }
}

/**
 * \fn void fasp_blas_dvec_axpyz (const REAL a, const dvector *x,
 *                                const dvector *y, dvector *z)
 *
 * \brief z = a*x + y, z is a third vector (z is cleared)
 *
 * \param a   REAL factor a
 * \param x   Pointer to dvector x
 * \param y   Pointer to dvector y
 * \param z   Pointer to dvector z
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_blas_dvec_axpyz (const REAL     a,
                           const dvector *x,
                           const dvector *y,
                           dvector       *z)
{
    const INT     m = x->row;
    const REAL *xpt = x->val, *ypt = y->val;
    REAL       *zpt = z->val;
    
    if ( y->row != m ) {
        printf("### ERROR: Vectors have different dimensions!\n");
        fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
    }
    
    z->row = m;
    
    memcpy (zpt, ypt, m*sizeof(REAL));
    fasp_blas_darray_axpy (m, a, xpt, zpt);
}

/**
 * \fn REAL fasp_blas_dvec_norm1 (const dvector *x)
 *
 * \brief L1 norm of dvector x
 *
 * \param x   Pointer to dvector x
 *
 * \return    L1 norm of x
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
REAL fasp_blas_dvec_norm1 (const dvector *x)
{
    const INT   length = x->row;
    const REAL *xpt = x->val;
    
    register REAL onenorm = 0.0;
    SHORT use_openmp = FALSE;
    INT   i;
    
#ifdef _OPENMP
    if ( length > OPENMP_HOLDS ) {
        use_openmp = TRUE;
    }
#endif
    
    if ( use_openmp ) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:onenorm) private(i)
#endif
        for ( i = 0; i < length; ++i ) onenorm += ABS(xpt[i]);
    }
    else {
        for ( i = 0; i < length; ++i ) onenorm += ABS(xpt[i]);
    }
    
    return onenorm;
}

/**
 * \fn REAL fasp_blas_dvec_norm2 (const dvector *x)
 *
 * \brief L2 norm of dvector x
 *
 * \param x   Pointer to dvector x
 *
 * \return    L2 norm of x
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 */
REAL fasp_blas_dvec_norm2 (const dvector *x)
{
    const INT   length = x->row;
    const REAL *xpt = x->val;
    
    register REAL twonorm = 0.0;
    SHORT use_openmp = FALSE;
    INT   i;
    
#ifdef _OPENMP
    if ( length > OPENMP_HOLDS ) use_openmp = TRUE;
#endif
    
    if ( use_openmp ) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:twonorm) private(i)
#endif
        for ( i = 0; i < length; ++i ) twonorm += xpt[i]*xpt[i];
    }
    else {
        for ( i = 0; i < length; ++i ) twonorm += xpt[i]*xpt[i];
    }
    
    return sqrt(twonorm);
}

/**
 * \fn REAL fasp_blas_dvec_norminf (const dvector *x)
 *
 * \brief Linf norm of dvector x
 *
 * \param x   Pointer to dvector x
 *
 * \return    L_inf norm of x
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 */
REAL fasp_blas_dvec_norminf (const dvector *x)
{
    const INT      length=x->row;
    const REAL    *xpt=x->val;
    
    register REAL  infnorm = 0.0;
    register INT   i;
    
    for ( i = 0; i < length; ++i ) infnorm = MAX(infnorm, ABS(xpt[i]));
    
    return infnorm;
}

/**
 * \fn REAL fasp_blas_dvec_dotprod (const dvector *x, const dvector *y)
 *
 * \brief Inner product of two vectors (x,y)
 *
 * \param x   Pointer to dvector x
 * \param y   Pointer to dvector y
 *
 * \return    Inner product
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
REAL fasp_blas_dvec_dotprod (const dvector *x,
                             const dvector *y)
{
    const INT   length = x->row;
    const REAL *xpt = x->val, *ypt = y->val;
    
    register REAL value = 0.0;
    SHORT use_openmp = FALSE;
    INT   i;
    
#ifdef _OPENMP
    if ( length > OPENMP_HOLDS ) use_openmp = TRUE;
#endif
    
    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:value) private(i)
#endif
        for ( i = 0; i < length; ++i ) value += xpt[i] * ypt[i];
    }
    else {
        for ( i = 0; i < length; ++i ) value += xpt[i] * ypt[i];
    }
    
    return value;
}

/**
 * \fn REAL fasp_blas_dvec_relerr (const dvector *x, const dvector *y)
 *
 * \brief Relative difference between two dvector x and y
 *
 * \param x   Pointer to dvector x
 * \param y   Pointer to dvector y
 *
 * \return    Relative difference ||x-y||/||x||
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
REAL fasp_blas_dvec_relerr (const dvector *x,
                            const dvector *y)
{
    const INT   length = x->row;
    const REAL *xpt = x->val, *ypt = y->val;
    
    SHORT use_openmp = FALSE;
    REAL diff = 0.0, temp = 0.0;
    INT   i;
    
    if ( length != y->row ) {
        printf("### ERROR: Vectors have different dimensions!\n");
        fasp_chkerr(ERROR_DATA_STRUCTURE, __FUNCTION__);
    }
    
#ifdef _OPENMP
    if ( length > OPENMP_HOLDS ) use_openmp = TRUE;
#endif
    
    if ( use_openmp ) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:temp,diff) private(i)
#endif
        for ( i = 0; i < length; ++i ) {
            temp += xpt[i]*xpt[i];
            diff += pow(xpt[i]-ypt[i],2);
        }
    }
    else {
        for ( i = 0; i < length; ++i ) {
            temp += xpt[i]*xpt[i];
            diff += pow(xpt[i]-ypt[i],2);
        }
    }
    
    return sqrt(diff/temp);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
