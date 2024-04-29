/*! \file  AuxVector.c
 *
 *  \brief Simple vector operations -- init, set, copy, etc
 *
 *  \note  This file contains Level-0 (Aux) functions. It requires:
 *         AuxThreads.c
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
 * \fn SHORT fasp_dvec_isnan (const dvector *u)
 *
 * \brief Check a dvector whether there is NAN
 *
 * \param u    Pointer to dvector
 *
 * \return     Return TRUE if there is NAN
 *
 * \author Chensong Zhang
 * \date   2013/03/31
 */
SHORT fasp_dvec_isnan (const dvector *u)
{
    INT i;
    
    for ( i = 0; i < u->row; i++ ) {
        if ( isnan(u->val[i]) ) return TRUE;
    }

    return FALSE;
}

/**
 * \fn dvector fasp_dvec_create (const INT m)
 *
 * \brief Create dvector data space of REAL type
 *
 * \param m    Number of rows
 *
 * \return u   The new dvector
 *
 * \author Chensong Zhang 
 * \date   2010/04/06
 */
dvector fasp_dvec_create (const INT m)
{
    dvector u;
    
    u.row = m;
    u.val = (REAL *)fasp_mem_calloc(m,sizeof(REAL));
    
    return u;
}

/**
 * \fn ivector fasp_ivec_create (const INT m)
 *
 * \brief Create vector data space of INT type
 *
 * \param m   Number of rows
 *
 * \return u  The new ivector
 *
 * \author Chensong Zhang 
 * \date   2010/04/06
 */
ivector fasp_ivec_create (const INT m)
{    
    ivector u;
    
    u.row = m;
    u.val = (INT *)fasp_mem_calloc(m,sizeof(INT)); 
    
    return u;
}

/**
 * \fn void fasp_dvec_alloc (const INT m, dvector *u)
 *
 * \brief Create dvector data space of REAL type
 *
 * \param m    Number of rows
 * \param u    Pointer to dvector (OUTPUT)
 *
 * \author Chensong Zhang 
 * \date   2010/04/06
 */
void fasp_dvec_alloc (const INT  m,
                      dvector   *u)
{    
    u->row = m;
    u->val = (REAL*)fasp_mem_calloc(m,sizeof(REAL)); 
    
    return;
}

/**
 * \fn void fasp_ivec_alloc (const INT m, ivector *u)
 *
 * \brief Create vector data space of INT type
 *
 * \param m   Number of rows
 * \param u   Pointer to ivector (OUTPUT)
 *
 * \author Chensong Zhang 
 * \date   2010/04/06
 */
void fasp_ivec_alloc (const INT  m,
                      ivector   *u)
{    
    
    u->row = m;
    u->val = (INT*)fasp_mem_calloc(m,sizeof(INT));
    
    return;
}

/**
 * \fn void fasp_dvec_free (dvector *u)
 *
 * \brief Free vector data space of REAL type
 *
 * \param u   Pointer to dvector which needs to be deallocated
 *
 * \author Chensong Zhang
 * \date   2010/04/03  
 */
void fasp_dvec_free (dvector *u)
{    
    if ( u == NULL ) return;
    
    fasp_mem_free(u->val); u->val = NULL; u->row = 0;
}

/**
 * \fn void fasp_ivec_free (ivector *u)
 *
 * \brief Free vector data space of INT type
 *
 * \param u   Pointer to ivector which needs to be deallocated
 *
 * \author Chensong Zhang
 * \date   2010/04/03  
 *
 * \note This function is same as fasp_dvec_free except input type.
 */
void fasp_ivec_free (ivector *u)
{    
    if ( u == NULL ) return;
    
    fasp_mem_free(u->val); u->val = NULL; u->row = 0;
}

/**
 * \fn void fasp_dvec_rand (const INT n, dvector *x)
 *
 * \brief Generate fake random REAL vector in the range from 0 to 1
 *
 * \param n    Size of the vector
 * \param x    Pointer to dvector
 * 
 * \note Sample usage: 
 * \par
 *   dvector xapp;
 * \par
 *   fasp_dvec_create(100,&xapp);
 * \par
 *   fasp_dvec_rand(100,&xapp);
 * \par
 *   fasp_dvec_print(100,&xapp);
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 */
void fasp_dvec_rand (const INT  n,
                     dvector   *x)
{
    const INT va = 0;
    const INT vb = n;
    
    INT s=1, i,j;
    
    srand(s);
    for ( i = 0; i < n; ++i ) {
        j = 1 + (INT) (((REAL)n)*rand()/(RAND_MAX+1.0));
        x->val[i] = (((REAL)j)-va)/(vb-va);
    }
    x->row = n;
}

/**
 * \fn void fasp_dvec_set (INT n, dvector *x, const REAL val)
 *
 * \brief Initialize dvector x[i]=val for i=0:n-1
 *
 * \param n      Number of variables
 * \param x      Pointer to dvector
 * \param val    Initial value for the vector
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012    
 */
void fasp_dvec_set (INT         n,
                    dvector    *x,
                    const REAL  val)
{
    INT   i;
    REAL *xpt = x->val;
    
    if ( n > 0 ) x->row = n;
    else n = x->row;
	
#ifdef _OPENMP 
    // variables for OpenMP 
    INT myid, mybegin, myend;
    INT nthreads = fasp_get_num_threads();
#endif
   
    if (val == 0.0) {
        
#ifdef _OPENMP 
        if (n > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend) 
            for (myid = 0; myid < nthreads; myid++ ) {
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                memset(&xpt[mybegin], 0x0, sizeof(REAL)*(myend-mybegin));
            }
        }
        else {
#endif
            memset(xpt, 0x0, sizeof(REAL)*n);
#ifdef _OPENMP
        }
#endif
        
    }
    
    else {
        
#ifdef _OPENMP 
        if (n > OPENMP_HOLDS) {
#pragma omp parallel for private(myid, mybegin, myend) 
            for (myid = 0; myid < nthreads; myid++ ) {
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i=mybegin; i<myend; ++i) xpt[i]=val;
            }
        }
        else {
#endif
            for (i=0; i<n; ++i) xpt[i]=val;
#ifdef _OPENMP
        }
#endif
        
    }
}

/**
 * \fn void fasp_ivec_set (INT n, ivector *u, const INT m)
 *
 * \brief Set ivector value to be m
 *
 * \param n    Number of variables
 * \param m    Integer value of ivector
 * \param u    Pointer to ivector (MODIFIED)
 *
 * \author Chensong Zhang
 * \date   04/03/2010  
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012    
 */
void fasp_ivec_set (INT        n,
                    ivector   *u,
                    const INT  m)
{    
    SHORT nthreads = 1, use_openmp = FALSE;
    INT   i;
    
    if ( n > 0 ) u->row = n;
    else n = u->row;

#ifdef _OPENMP
	if ( n > OPENMP_HOLDS ) {
        use_openmp = TRUE;
        nthreads = fasp_get_num_threads();
	}
#endif
    
	if (use_openmp) {
		INT mybegin, myend, myid;
#ifdef _OPENMP 
#pragma omp parallel for private(myid, mybegin, myend, i) 
#endif
        for (myid = 0; myid < nthreads; myid++ ) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i=mybegin; i<myend; ++i) u->val[i] = m;
        }        
	}    
	else {
        for (i=0; i<n; ++i) u->val[i] = m;
	}
}

/**
 * \fn void fasp_dvec_cp (const dvector *x, dvector *y)
 *
 * \brief Copy dvector x to dvector y
 *
 * \param x  Pointer to dvector
 * \param y  Pointer to dvector (MODIFIED)
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 */
void fasp_dvec_cp (const dvector  *x,
                   dvector        *y)
{
    y->row = x->row;
    memcpy(y->val,x->val,x->row*sizeof(REAL));
}

/**
 * \fn REAL fasp_dvec_maxdiff (const dvector *x, const dvector *y)
 *
 * \brief Maximal difference of two dvector x and y
 *
 * \param  x    Pointer to dvector
 * \param  y    Pointer to dvector
 *
 * \return      Maximal norm of x-y
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 *
 * Modified by chunsheng Feng, Zheng Li
 * \date   06/30/2012
 */
REAL fasp_dvec_maxdiff (const dvector *x,
                        const dvector *y)
{
    const INT length = x->row;
    const REAL *xpt = x->val, *ypt = y->val;
    
    SHORT use_openmp = FALSE;
    INT   i;
    REAL  Linf = 0.0, diffi = 0.0;

#ifdef _OPENMP 
    INT myid, mybegin, myend, nthreads;
    if ( length > OPENMP_HOLDS ) {
        use_openmp = TRUE;
        nthreads = fasp_get_num_threads();
    }
#endif
    
    if(use_openmp) {
#ifdef _OPENMP 
        REAL temp = 0.;
#pragma omp parallel firstprivate(temp) private(myid, mybegin, myend, i, diffi) 
        {
            myid = omp_get_thread_num();
            fasp_get_start_end(myid, nthreads, length, &mybegin, &myend);
            for(i=mybegin; i<myend; i++) {
                if ((diffi = ABS(xpt[i]-ypt[i])) > temp) temp = diffi;
            }
#pragma omp critical
            if(temp > Linf) Linf = temp;
        }
#endif
    }
    else {
        for (i=0; i<length; ++i) {
            if ((diffi = ABS(xpt[i]-ypt[i])) > Linf) Linf = diffi;
        }
    }
    
    return Linf;
}

/**
 * \fn void fasp_dvec_symdiagscale (dvector *b, const dvector *diag)
 *
 * \brief Symmetric diagonal scaling D^{-1/2}b
 *
 * \param b       Pointer to dvector
 * \param diag    Pointer to dvector: the diagonal entries
 *
 * \author Xiaozhe Hu
 * \date   01/31/2011
 */
void fasp_dvec_symdiagscale (dvector        *b,
                             const dvector  *diag)
{
    // information about dvector
    const INT    n = b->row;
    REAL      *val = b->val;
    
    // local variables
    SHORT use_openmp = FALSE;
    INT   i;
    
    if ( diag->row != n ) {
        printf("### ERROR: Sizes of diag = %d != dvector = %d!", diag->row, n);
        fasp_chkerr(ERROR_MISC, __FUNCTION__);
    }
    
#ifdef _OPENMP 
    INT mybegin, myend, myid, nthreads;
    if ( n > OPENMP_HOLDS ){
        use_openmp = TRUE;
        nthreads = fasp_get_num_threads();
    }
#endif

    if (use_openmp) {
#ifdef _OPENMP 
#pragma omp parallel for private(myid, mybegin,myend) 
        for (myid = 0; myid < nthreads; myid++ ) {
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i=mybegin; i<myend; ++i) val[i] = val[i]/sqrt(diag->val[i]);
        }        
#endif
    }    
    else {
        for (i=0; i<n; ++i) val[i] = val[i]/sqrt(diag->val[i]);
    }
    
    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
