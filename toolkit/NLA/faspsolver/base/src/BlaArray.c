/*! \file  BlaArray.c
 *
 *  \brief BLAS1 operations for arrays
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
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
 * \fn inline void fasp_blas_darray_ax (const INT n, const REAL a, REAL *x)
 *
 * \brief x = a*x
 *
 * \param n    Number of variables
 * \param a    Factor a
 * \param x    Pointer to x
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 *
 * \warning x is reused to store the resulting array!
 */
void fasp_blas_darray_ax(const INT n, const REAL a, REAL* x)
{
    if (a == 1.0) return; // do nothing

    {
        SHORT use_openmp = FALSE;
        INT   i;

#ifdef _OPENMP
        INT myid, mybegin, myend, nthreads;
        if (n > OPENMP_HOLDS) {
            use_openmp = TRUE;
            nthreads   = fasp_get_num_threads();
        }
#endif

        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i) x[i] *= a;
            }
#endif
        } else {
            for (i = 0; i < n; ++i) x[i] *= a;
        }
    }
}

/**
 * \fn void fasp_blas_darray_axpy (const INT n, const REAL a,
 *                                 const REAL *x, REAL *y)
 *
 * \brief y = a*x + y
 *
 * \param n    Number of variables
 * \param a    Factor a
 * \param x    Pointer to x
 * \param y    Pointer to y, reused to store the resulting array
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_blas_darray_axpy(const INT n, const REAL a, const REAL* x, REAL* y)
{
    SHORT use_openmp = FALSE;
    INT   i;

#ifdef _OPENMP
    INT myid, mybegin, myend, nthreads;
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (a == 1.0) {
        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i) y[i] += x[i];
            }
#endif
        } else {
            for (i = 0; i < n; ++i) y[i] += x[i];
        }
    }

    else if (a == -1.0) {
        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i) y[i] -= x[i];
            }
#endif
        } else {
            for (i = 0; i < n; ++i) y[i] -= x[i];
        }
    }

    else {
        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i) y[i] += a * x[i];
            }
#endif
        } else {
            for (i = 0; i < n; ++i) y[i] += a * x[i];
        }
    }
}

/**
 * \fn void fasp_blas_ldarray_axpy(const INT n, const REAL a,
 *                                 const REAL *x, LONGREAL *y)
 *
 * \brief y = a*x + y
 *
 * \param n    Number of variables
 * \param a    Factor a
 * \param x    Pointer to x
 * \param y    Pointer to y, reused to store the resulting array
 *
 * \author Ting Lai
 * \date   05/09/2022
 */
void fasp_blas_ldarray_axpy(const INT n, const REAL a, const REAL* x, LONGREAL* y)
{
    SHORT use_openmp = FALSE;
    INT   i;

#ifdef _OPENMP
    INT myid, mybegin, myend, nthreads;
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (a == 1.0) {
        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i) y[i] += x[i];
            }
#endif
        } else {
            for (i = 0; i < n; ++i) y[i] += x[i];
        }
    }

    else if (a == -1.0) {
        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i) y[i] -= x[i];
            }
#endif
        } else {
            for (i = 0; i < n; ++i) y[i] -= x[i];
        }
    }

    else {
        if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
            {
                myid = omp_get_thread_num();
                fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
                for (i = mybegin; i < myend; ++i) y[i] += a * x[i];
            }
#endif
        } else {
            for (i = 0; i < n; ++i) y[i] += a * x[i];
        }
    }
}

/**
 * \fn void fasp_blas_darray_axpy_nc2 (const REAL a, const REAL *x, REAL *y)
 *
 * \brief y = a*x + y, length of x and y should be 2
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array
 * \param y   Pointer to the destination array
 *
 * \author Xiaozhe Hu
 * \date   18/11/2011
 */
void fasp_blas_darray_axpy_nc2(const REAL a, const REAL* x, REAL* y)
{
    y[0] += a * x[0];
    y[1] += a * x[1];

    y[2] += a * x[2];
    y[3] += a * x[3];
}

/**
 * \fn void fasp_blas_darray_axpy_nc3 (const REAL a, const REAL *x, REAL *y)
 *
 * \brief y = a*x + y, length of x and y should be 3
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array
 * \param y   Pointer to the destination array
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
void fasp_blas_darray_axpy_nc3(const REAL a, const REAL* x, REAL* y)
{
    y[0] += a * x[0];
    y[1] += a * x[1];
    y[2] += a * x[2];

    y[3] += a * x[3];
    y[4] += a * x[4];
    y[5] += a * x[5];

    y[6] += a * x[6];
    y[7] += a * x[7];
    y[8] += a * x[8];
}

/**
 * \fn void fasp_blas_darray_axpy_nc5 (const REAL a, const REAL *x, REAL *y)
 *
 * \brief y = a*x + y, length of x and y should be 5
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array
 * \param y   Pointer to the destination array
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
void fasp_blas_darray_axpy_nc5(const REAL a, const REAL* x, REAL* y)
{
    y[0] += a * x[0];
    y[1] += a * x[1];
    y[2] += a * x[2];
    y[3] += a * x[3];
    y[4] += a * x[4];

    y[5] += a * x[5];
    y[6] += a * x[6];
    y[7] += a * x[7];
    y[8] += a * x[8];
    y[9] += a * x[9];

    y[10] += a * x[10];
    y[11] += a * x[11];
    y[12] += a * x[12];
    y[13] += a * x[13];
    y[14] += a * x[14];

    y[15] += a * x[15];
    y[16] += a * x[16];
    y[17] += a * x[17];
    y[18] += a * x[18];
    y[19] += a * x[19];

    y[20] += a * x[20];
    y[21] += a * x[21];
    y[22] += a * x[22];
    y[23] += a * x[23];
    y[24] += a * x[24];
}

/**
 * \fn void fasp_blas_darray_axpy_nc7 (const REAL a, const REAL *x, REAL *y)
 *
 * \brief y = a*x + y, length of x and y should be 7
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array
 * \param y   Pointer to the destination array
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
void fasp_blas_darray_axpy_nc7(const REAL a, const REAL* x, REAL* y)
{
    y[0] += a * x[0];
    y[1] += a * x[1];
    y[2] += a * x[2];
    y[3] += a * x[3];
    y[4] += a * x[4];
    y[5] += a * x[5];
    y[6] += a * x[6];

    y[7] += a * x[7];
    y[8] += a * x[8];
    y[9] += a * x[9];
    y[10] += a * x[10];
    y[11] += a * x[11];
    y[12] += a * x[12];
    y[13] += a * x[13];

    y[14] += a * x[14];
    y[15] += a * x[15];
    y[16] += a * x[16];
    y[17] += a * x[17];
    y[18] += a * x[18];
    y[19] += a * x[19];
    y[20] += a * x[20];

    y[21] += a * x[21];
    y[22] += a * x[22];
    y[23] += a * x[23];
    y[24] += a * x[24];
    y[25] += a * x[25];
    y[26] += a * x[26];
    y[27] += a * x[27];

    y[28] += a * x[28];
    y[29] += a * x[29];
    y[30] += a * x[30];
    y[31] += a * x[31];
    y[32] += a * x[32];
    y[33] += a * x[33];
    y[34] += a * x[34];

    y[35] += a * x[35];
    y[36] += a * x[36];
    y[37] += a * x[37];
    y[38] += a * x[38];
    y[39] += a * x[39];
    y[40] += a * x[40];
    y[41] += a * x[41];

    y[42] += a * x[42];
    y[43] += a * x[43];
    y[44] += a * x[44];
    y[45] += a * x[45];
    y[46] += a * x[46];
    y[47] += a * x[47];
    y[48] += a * x[48];
}

/**
 * \fn void fasp_blas_darray_axpyz (const INT n, const REAL a, const REAL *x,
 *                                  const REAL *y, REAL *z)
 *
 * \brief z = a*x + y
 *
 * \param n    Number of variables
 * \param a    Factor a
 * \param x    Pointer to x
 * \param y    Pointer to y
 * \param z    Pointer to z
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_blas_darray_axpyz(const INT n, const REAL a, const REAL* x, const REAL* y,
                            REAL* z)
{
    SHORT use_openmp = FALSE;
    INT   i;

#ifdef _OPENMP
    INT myid, mybegin, myend, nthreads;
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
        {
            myid = omp_get_thread_num();
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) z[i] = a * x[i] + y[i];
        }
#endif
    } else {
        for (i = 0; i < n; ++i) z[i] = a * x[i] + y[i];
    }
}

/**
 * \fn void fasp_blas_darray_axpyz_nc2 (const REAL a, const REAL *x,
 *                                      const REAL *y, REAL *z)
 *
 * \brief z = a*x + y, length of x, y and z should be 2
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array 1
 * \param y   Pointer to the original array 2
 * \param z   Pointer to the destination array
 *
 * \author Xiaozhe Hu
 * \date   18/11/2011
 */
void fasp_blas_darray_axpyz_nc2(const REAL a, const REAL* x, const REAL* y, REAL* z)
{
    z[0] = a * x[0] + y[0];
    z[1] = a * x[1] + y[1];

    z[2] = a * x[2] + y[2];
    z[3] = a * x[3] + y[3];
}

/**
 * \fn void fasp_blas_darray_axpyz_nc3 (const REAL a, const REAL *x,
 *                                      const REAL *y, REAL *z)
 *
 * \brief z = a*x + y, length of x, y and z should be 3
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array 1
 * \param y   Pointer to the original array 2
 * \param z   Pointer to the destination array
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
void fasp_blas_darray_axpyz_nc3(const REAL a, const REAL* x, const REAL* y, REAL* z)
{
    z[0] = a * x[0] + y[0];
    z[1] = a * x[1] + y[1];
    z[2] = a * x[2] + y[2];

    z[3] = a * x[3] + y[3];
    z[4] = a * x[4] + y[4];
    z[5] = a * x[5] + y[5];

    z[6] = a * x[6] + y[6];
    z[7] = a * x[7] + y[7];
    z[8] = a * x[8] + y[8];
}

/**
 * \fn void fasp_blas_darray_axpyz_nc5 (const REAL a, const REAL *x,
 *                                      const REAL *y, REAL *z)
 *
 * \brief z = a*x + y, length of x, y and z should be 5
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array 1
 * \param y   Pointer to the original array 2
 * \param z   Pointer to the destination array
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
void fasp_blas_darray_axpyz_nc5(const REAL a, const REAL* x, const REAL* y, REAL* z)
{
    z[0] = a * x[0] + y[0];
    z[1] = a * x[1] + y[1];
    z[2] = a * x[2] + y[2];
    z[3] = a * x[3] + y[3];
    z[4] = a * x[4] + y[4];

    z[5] = a * x[5] + y[5];
    z[6] = a * x[6] + y[6];
    z[7] = a * x[7] + y[7];
    z[8] = a * x[8] + y[8];
    z[9] = a * x[9] + y[9];

    z[10] = a * x[10] + y[10];
    z[11] = a * x[11] + y[11];
    z[12] = a * x[12] + y[12];
    z[13] = a * x[13] + y[13];
    z[14] = a * x[14] + y[14];

    z[15] = a * x[15] + y[15];
    z[16] = a * x[16] + y[16];
    z[17] = a * x[17] + y[17];
    z[18] = a * x[18] + y[18];
    z[19] = a * x[19] + y[19];

    z[20] = a * x[20] + y[20];
    z[21] = a * x[21] + y[21];
    z[22] = a * x[22] + y[22];
    z[23] = a * x[23] + y[23];
    z[24] = a * x[24] + y[24];
}

/**
 * \fn void fasp_blas_darray_axpyz_nc7 (const REAL a, const REAL *x,
 *                                      const REAL *y, REAL *z)
 *
 * \brief z = a*x + y, length of x, y and z should be 7
 *
 * \param a   REAL factor a
 * \param x   Pointer to the original array 1
 * \param y   Pointer to the original array 2
 * \param z   Pointer to the destination array
 *
 * \author Xiaozhe Hu, Shiquan Zhang
 * \date   05/01/2010
 */
void fasp_blas_darray_axpyz_nc7(const REAL a, const REAL* x, const REAL* y, REAL* z)
{
    z[0] = a * x[0] + y[0];
    z[1] = a * x[1] + y[1];
    z[2] = a * x[2] + y[2];
    z[3] = a * x[3] + y[3];
    z[4] = a * x[4] + y[4];
    z[5] = a * x[5] + y[5];
    z[6] = a * x[6] + y[6];

    z[7]  = a * x[7] + y[7];
    z[8]  = a * x[8] + y[8];
    z[9]  = a * x[9] + y[9];
    z[10] = a * x[10] + y[10];
    z[11] = a * x[11] + y[11];
    z[12] = a * x[12] + y[12];
    z[13] = a * x[13] + y[13];

    z[14] = a * x[14] + y[14];
    z[15] = a * x[15] + y[15];
    z[16] = a * x[16] + y[16];
    z[17] = a * x[17] + y[17];
    z[18] = a * x[18] + y[18];
    z[19] = a * x[19] + y[19];
    z[20] = a * x[20] + y[20];

    z[21] = a * x[21] + y[21];
    z[22] = a * x[22] + y[22];
    z[23] = a * x[23] + y[23];
    z[24] = a * x[24] + y[24];
    z[25] = a * x[25] + y[25];
    z[26] = a * x[26] + y[26];
    z[27] = a * x[27] + y[27];

    z[28] = a * x[28] + y[28];
    z[29] = a * x[29] + y[29];
    z[30] = a * x[30] + y[30];
    z[31] = a * x[31] + y[31];
    z[32] = a * x[32] + y[32];
    z[33] = a * x[33] + y[33];
    z[34] = a * x[34] + y[34];

    z[35] = a * x[35] + y[35];
    z[36] = a * x[36] + y[36];
    z[37] = a * x[37] + y[37];
    z[38] = a * x[38] + y[38];
    z[39] = a * x[39] + y[39];
    z[40] = a * x[40] + y[40];
    z[41] = a * x[41] + y[41];

    z[42] = a * x[42] + y[42];
    z[43] = a * x[43] + y[43];
    z[44] = a * x[44] + y[44];
    z[45] = a * x[45] + y[45];
    z[46] = a * x[46] + y[46];
    z[47] = a * x[47] + y[47];
    z[48] = a * x[48] + y[48];
}

/**
 * \fn void fasp_blas_darray_axpby (const INT n, const REAL a, const REAL *x,
 *                                  const REAL b, REAL *y)
 *
 * \brief y = a*x + b*y
 *
 * \param n    Number of variables
 * \param a    Factor a
 * \param x    Pointer to x
 * \param b    Factor b
 * \param y    Pointer to y, reused to store the resulting array
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
void fasp_blas_darray_axpby(const INT n, const REAL a, const REAL* x, const REAL b,
                            REAL* y)
{
    SHORT use_openmp = FALSE;
    INT   i;

#ifdef _OPENMP
    INT myid, mybegin, myend, nthreads;
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel private(myid, mybegin, myend, i) num_threads(nthreads)
        {
            myid = omp_get_thread_num();
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) y[i] = a * x[i] + b * y[i];
        }
#endif
    } else {
        for (i = 0; i < n; ++i) y[i] = a * x[i] + b * y[i];
    }
}

/**
 * \fn REAL fasp_blas_darray_norm1 (const INT n, const REAL *x)
 *
 * \brief L1 norm of array x
 *
 * \param n    Number of variables
 * \param x    Pointer to x
 *
 * \return     L1 norm of x
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
REAL fasp_blas_darray_norm1(const INT n, const REAL* x)
{
    register REAL onenorm = 0.0;
    INT           i;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : onenorm) private(i)
#endif
    for (i = 0; i < n; ++i) onenorm += ABS(x[i]);

    return onenorm;
}

/**
 * \fn REAL fasp_blas_darray_norm2 (const INT n, const REAL *x)
 *
 * \brief L2 norm of array x
 *
 * \param n    Number of variables
 * \param x    Pointer to x
 *
 * \return     L2 norm of x
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
REAL fasp_blas_darray_norm2(const INT n, const REAL* x)
{
    register REAL twonorm = 0.0;
    INT           i;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : twonorm) private(i)
#endif
    for (i = 0; i < n; ++i) twonorm += x[i] * x[i];

    return sqrt(twonorm);
}

/**
 * \fn REAL fasp_blas_darray_norminf (const INT n, const REAL *x)
 *
 * \brief Linf norm of array x
 *
 * \param n    Number of variables
 * \param x    Pointer to x
 *
 * \return     L_inf norm of x
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Zheng Li on 06/28/2012
 */
REAL fasp_blas_darray_norminf(const INT n, const REAL* x)
{
    SHORT         use_openmp = FALSE;
    register REAL infnorm    = 0.0;
    INT           i;

#ifdef _OPENMP
    INT myid, mybegin, myend, nthreads;
    if (n > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    if (use_openmp) {
#ifdef _OPENMP
        REAL infnorm_loc = 0.0;
#pragma omp parallel firstprivate(infnorm_loc) private(myid, mybegin, myend, i)
        {
            myid = omp_get_thread_num();
            fasp_get_start_end(myid, nthreads, n, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) infnorm_loc = MAX(infnorm_loc, ABS(x[i]));

            if (infnorm_loc > infnorm) {
#pragma omp critical
                infnorm = MAX(infnorm_loc, infnorm);
            }
        }
#endif
    } else {
        for (i = 0; i < n; ++i) infnorm = MAX(infnorm, ABS(x[i]));
    }

    return infnorm;
}

/**
 * \fn REAL fasp_blas_darray_dotprod (const INT n, const REAL *x, const REAL *y)
 *
 * \brief Inner product of two arraies x and y
 *
 * \param n    Number of variables
 * \param x    Pointer to x
 * \param y    Pointer to y
 *
 * \return     Inner product (x,y)
 *
 * \author Chensong Zhang
 * \date   07/01/2009
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012
 */
REAL fasp_blas_darray_dotprod(const INT n, const REAL* x, const REAL* y)
{
    SHORT         use_openmp = FALSE;
    register REAL value      = 0.0;
    INT           i;

#ifdef _OPENMP
    if (n > OPENMP_HOLDS) use_openmp = TRUE;
#endif

    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : value) private(i)
#endif
        for (i = 0; i < n; ++i) value += x[i] * y[i];
    } else {
        for (i = 0; i < n; ++i) value += x[i] * y[i];
    }

    return value;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
