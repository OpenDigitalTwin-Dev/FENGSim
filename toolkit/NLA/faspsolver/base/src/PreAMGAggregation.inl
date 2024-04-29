/*! \file  PreAMGAggregation.inl
 *
 *  \brief Utilities for aggregation methods
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreAMGSetupSA.c, PreAMGSetupUA.c, PreAMGSetupSABSR.c,
 *         and PreAMGSetupUABSR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2012--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  \warning This file is also used in FASP4BLKOIL!!!
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"
#include "math.h"

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn double dense_matrix_norm_fro(double* A, int m, int n)
 *
 * \brief Calculate the Frobenius-norm of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return Frobenius-norm
 *
 * \author Li Zhao
 * \date   05/21/2023
 */
static double dense_matrix_norm_fro(double* A, int m, int n)
{
    double norm = 0.0;
    int    i;
    for (i = 0; i < m * n; i++) {
        norm += A[i] * A[i];
    }
    return (sqrt(norm));
}

/**
 * \fn double dense_matrix_norm_fro22(double* A, int m, int n)
 *
 * \brief Calculate the Frobenius-norm of A(0:1, 0:1) a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return Frobenius-norm
 *
 * \author Li Zhao
 * \date   06/06/2023
 */
static double dense_matrix_norm_fro22(double* A, int m, int n)
{
    double norm = 0.0, sum = 0.0;
    int    i, j;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            sum += A[i * n + j] * A[i * n + j];
        }
    }
    norm = sqrt(sum);

    return (norm);
}

/**
 * \fn double dense_matrix_norm_fro22_rest(double* A, int m, int n)
 *
 * \brief Calculate the Frobenius-norm of A(0:1, 0:1) a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return Frobenius-norm
 *
 * \author Li Zhao
 * \date   06/06/2023
 */
static double dense_matrix_norm_fro22_rest(double* A, int m, int n)
{
    double norm = 0.0, sum = 0.0;
    int    i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (!(i > 1 && j > 1)) sum += A[i * n + j] * A[i * n + j];
        }
    }
    norm = sqrt(sum);

    return (norm);
}

/**
 * \fn double dense_matrix_norm_fro_weight(double* A, int m, int n)
 *
 * \brief Calculate the Frobenius-norm with weight of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return Frobenius-norm
 *
 * \author Li Zhao
 * \date   06/06/2023
 */
static double dense_matrix_norm_fro_weight(double* A, int m, int n)
{
    double norm = 0.0, sum = 0.0, max_val = 0.0;
    int    i, j;
    // double w[4][4] = {{0, 1, 1, 1}, {0, 0, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}};
    // double w[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    // double w[4][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    double w[4][4] = {{0, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            // sum += w[i][j] * fabs(A[i * n + j]);
            sum += w[i][j] * A[i * n + j] * A[i * n + j];
            // max_val = MAX(max_val, w[i][j] * fabs(A[i * n + j]));
        }
    }
    // norm = sum / 5.0;
    norm = sqrt(sum);
    // norm = max_val;

    return (norm);
}

/**
 * \fn double dense_matrix_norm1(double* A, int m, int n)
 *
 * \brief Calculate the 1-norm (sum columns) of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return 1-norm
 *
 * \author Li Zhao
 * \date   05/21/2023
 */
static double dense_matrix_norm1(double* A, int m, int n)
{
    double norm = 0.0, sum;
    int    i, j;
    for (j = 0; j < n; j++) {
        sum = 0.0;
        for (i = 0; i < m; i++) {
            sum += fabs(A[i * m + j]);
        }
        norm = MAX(norm, sum);
    }
    return (norm);
}

/**
 * \fn double dense_matrix_norm_inf(double* A, int m, int n)
 *
 * \brief Calculate the infity-norm (sum row) of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return infity-norm
 *
 * \author Li Zhao
 * \date   05/21/2023
 */
static double dense_matrix_norm_inf(double* A, int m, int n)
{
    double norm = 0.0, sum;
    int    i, j;
    for (i = 0; i < m; i++) {
        sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += fabs(A[i * m + j]);
        }
        norm = MAX(norm, sum);
    }
    return (norm);
}

/**
 * \fn double dense_matrix_max (double* A, int m, int n)
 *
 * \brief Calculate the maximumn value of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return maximumn value
 *
 * \author Li Zhao
 * \date   05/21/2023
 */
static double dense_matrix_max(double* A, int m, int n)
{
    double norm = 0.0, sum;
    int    i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            norm = MAX(norm, fabs(A[i * m + j]));
        }
    }

    if (norm < SMALLREAL) printf("%e\n", norm);
    return (norm);
}

/**
 * \fn double dense_matrix_get00 (double* A, int m, int n)
 *
 * \brief get the (0, 0) entry of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return A(0, 0)
 *
 * \author Li Zhao
 * \date   05/21/2023
 */
static double dense_matrix_get00(double* A, int m, int n) { return (A[0]); }

/**
 * \fn double dense_matrix_getii (double* A, int m, int n)
 *
 * \brief get the (i, i) entry of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return A(i, i)
 *
 * \author Li Zhao
 * \date   06/06/2023
 */
static double dense_matrix_getii(double* A, int m, int n, int i)
{
    if (i >= n || i >= m) {
        printf("%s i = %d >= m = %d, AMG_aggregation_norm_type error!\n", __FUNCTION__,
               i, m);
        exit(0);
    }

    return (A[i + i * n]);
}

/**
 * \fn double dense_matrix_norms (double* A, int m, int n, int type)
 *
 * \brief Calculate the infity-norm (sum row) of a dense matrix
 *
 * \param A          Pointer to the dense matrices, i.e., A[m*n]
 * \param m          The number of rows    in the matrix
 * \param n          The number of columns in the matrix
 *
 * \return infity-norm
 *
 * \author Li Zhao
 * \date   05/21/2023
 */
static double dense_matrix_norms(double* A, int m, int n, int type)
{
    double norm = 0.0;
    switch (type) {
        case 0:
            norm = dense_matrix_max(A, m, n);
            break;

        case 1:
            norm = dense_matrix_norm1(A, m, n);
            break;

        case 2:
            printf("%s norm type: %d don't implemented!\n", __FUNCTION__, type);
            exit(0);
            break;

        case 3:
            norm = dense_matrix_norm_inf(A, m, n);
            break;

        case 4:
            norm = dense_matrix_norm_fro22(A, m, n);
            break;

        case 5:
            norm = dense_matrix_norm_fro22_rest(A, m, n);
            break;

        case 6:
            norm = dense_matrix_norm_fro_weight(A, m, n);
            break;

        case 11:
            norm = dense_matrix_get00(A, m, n);
            break;

        case 22:
            norm = dense_matrix_getii(A, m, n, 1);
            break;

        case 33:
            norm = dense_matrix_getii(A, m, n, 2);
            break;

        case 44:
            norm = dense_matrix_getii(A, m, n, 3);
            break;

            // case 55:
            //     norm = dense_matrix_getii(A, m, n, 4);
            //     break;

        default:
            norm = dense_matrix_norm_fro(A, m, n);
            break;
    }
    return (norm);
}

/**
 * \fn static SHORT aggregation_vmb (dCSRmat *A, ivector *vertices, AMG_param *param,
 *                                   const INT NumLevels, dCSRmat *Neigh,
 *                                   INT *NumAggregates)
 *
 * \brief Form aggregation based on strong coupled neighbors
 *
 * \param A                 Pointer to the coefficient matrices
 * \param vertices          Pointer to the aggregation of vertices
 * \param param             Pointer to AMG parameters
 * \param NumLevels         Level number
 * \param Neigh             Pointer to strongly coupled neighbors
 * \param NumAggregates     Pointer to number of aggregations
 *
 * \author Xiaozhe Hu
 * \date   09/29/2009
 *
 * \note Setup A, P, PT and levels using the unsmoothed aggregation algorithm;
 *       Refer to P. Vanek, J. Madel and M. Brezina
 *       "Algebraic Multigrid on Unstructured Meshes", 1994
 *
 * Modified by Zheng Li, Chensong Zhang on 07/29/2014
 */
static SHORT aggregation_vmb(dCSRmat*   A,
                             ivector*   vertices,
                             AMG_param* param,
                             const INT  NumLevels,
                             dCSRmat*   Neigh,
                             INT*       NumAggregates)
{
    const INT   row = A->row, col = A->col, nnz = A->IA[row] - A->IA[0];
    const INT * AIA = A->IA, *AJA = A->JA;
    const REAL* Aval            = A->val;
    const INT   max_aggregation = param->max_aggregation;

    // return status
    SHORT status = FASP_SUCCESS;

    // local variables
    INT  num_left = row;
    INT  subset, count;
    INT* num_each_agg;

    REAL  strongly_coupled, strongly_coupled2;
    INT   i, j, index, row_start, row_end;
    INT * NIA, *NJA;
    REAL* Nval;

    dvector diag;
    fasp_dcsr_getdiag(0, A, &diag); // get the diagonal entries

    if (GE(param->tentative_smooth, SMALLREAL)) {
        strongly_coupled = param->strong_coupled * pow(0.5, NumLevels - 1);
    } else {
        strongly_coupled = param->strong_coupled;
    }
    strongly_coupled2 = pow(strongly_coupled, 2);

    /*------------------------------------------*/
    /*    Form strongly coupled neighborhood    */
    /*------------------------------------------*/
    fasp_dcsr_alloc(row, col, nnz, Neigh);

    NIA  = Neigh->IA;
    NJA  = Neigh->JA;
    Nval = Neigh->val;

#if 0 // Li Zhao
#ifdef _OPENMP
#pragma omp parallel for if (row > OPENMP_HOLDS)
#endif
    for (i = row; i >= 0; i--)
        NIA[i] = AIA[i]; // why ?, here do not require it!, Li Zhao, 05/21/2023
#endif

    for (index = i = 0; i < row; ++i) {
        NIA[i]    = index;
        row_start = AIA[i];
        row_end   = AIA[i + 1];
        for (j = row_start; j < row_end; ++j) {
            if ((AJA[j] == i) ||
                (pow(Aval[j], 2) >=
                 strongly_coupled2 * ABS(diag.val[i] * diag.val[AJA[j]]))) {
                NJA[index]  = AJA[j];
                Nval[index] = Aval[j];
                index++;
            }
        }
    }
    NIA[row] = index;

    Neigh->nnz = index;
    Neigh->JA  = (INT*)fasp_mem_realloc(Neigh->JA, (Neigh->IA[row]) * sizeof(INT));
    Neigh->val = (REAL*)fasp_mem_realloc(Neigh->val, (Neigh->IA[row]) * sizeof(REAL));

    // NIA = Neigh->IA;
    // NJA = Neigh->JA;

    fasp_dvec_free(&diag);

    /*------------------------------------------*/
    /*             Initialization               */
    /*------------------------------------------*/
    fasp_ivec_alloc(row, vertices);
    fasp_iarray_set(row, vertices->val, -2);
    *NumAggregates = 0;

    /*-------------*/
    /*   Step 1.   */
    /*-------------*/
    for (i = 0; i < row; ++i) {
        if ((AIA[i + 1] - AIA[i]) == 1) {
            vertices->val[i] = UNPT;
            num_left--;
        } else {
            subset    = TRUE;
            row_start = NIA[i];
            row_end   = NIA[i + 1];
            for (j = row_start; j < row_end; ++j) {
                if (vertices->val[NJA[j]] >= UNPT) {
                    subset = FALSE;
                    break;
                }
            }
            if (subset) {
                count            = 0;
                vertices->val[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (count < max_aggregation)) {
                        vertices->val[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }

    /*-------------*/
    /*   Step 2.   */
    /*-------------*/
    INT* temp_C = (INT*)fasp_mem_calloc(row, sizeof(INT));

    if (*NumAggregates < MIN_CDOF) {
        status = ERROR_AMG_COARSEING;
        goto END;
    }

    num_each_agg = (INT*)fasp_mem_calloc(*NumAggregates, sizeof(INT));

    // for ( i = 0; i < *NumAggregates; i++ ) num_each_agg[i] = 0; // initialize

    for (i = row; i--;) {
        temp_C[i] = vertices->val[i];
        if (vertices->val[i] >= 0) num_each_agg[vertices->val[i]]++;
    }

    for (i = 0; i < row; ++i) {
        if (vertices->val[i] < UNPT) {
            row_start = NIA[i];
            row_end   = NIA[i + 1];

#if 1
            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > UNPT &&
                    num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
                    vertices->val[i] = temp_C[NJA[j]];
                    num_left--;
                    num_each_agg[temp_C[NJA[j]]]++;
                    break;
                }
            }
#else
            // try: if more than one such set exists, choose the one to which node i
            // has the strongest coupling,
            // The result is even worse, Li Zhao
            REAL strongest_coupling = 0.0;
            INT  this_id            = -1;
            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > UNPT &&
                    num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
                    if (fabs(Nval[j]) > strongest_coupling) {
                        strongest_coupling = fabs(Nval[j]);
                        this_id            = j;
                    }
                }
            }

            if (this_id > -1) {
                vertices->val[i] = temp_C[NJA[this_id]];
                num_left--;
                num_each_agg[temp_C[NJA[this_id]]]++;
            }
#endif
        }
    }

    // printf("%s, Step 3 num_left = %d\n", __FUNCTION__, num_left);

    /*-------------*/
    /*   Step 3.   */
    /*-------------*/

    // Optimal Implementation, Li Zhao
    // REAL start_time, end_time;
    // fasp_gettime(&start_time);
#if 0
    if (num_left > 0) {
        ivector left_map;
        fasp_ivec_alloc(row, &left_map);
        left_map.row = 0;
        for (i = 0; i < row; ++i) {
            if (vertices->val[i] < UNPT) {
                left_map.val[left_map.row++] = i;
            }
        }
        // printf("left_map.row = %d\n", left_map.row);

        int g;
        while (num_left > 0) {
            for (g = 0; g < left_map.row; g++) {
                i = left_map.val[g];
                if (vertices->val[i] < UNPT) {
                    count            = 0;
                    vertices->val[i] = *NumAggregates;
                    num_left--;
                    count++;
                    row_start = NIA[i];
                    row_end   = NIA[i + 1];
                    for (j = row_start; j < row_end; ++j) {
                        if ((NJA[j] != i) && (vertices->val[NJA[j]] < UNPT) &&
                            (count < max_aggregation)) {
                            vertices->val[NJA[j]] = *NumAggregates;
                            num_left--;
                            count++;
                        }
                    }
                    (*NumAggregates)++;
                }
            }
        }

        fasp_ivec_free(&left_map);
    }

#else
    while (num_left > 0) {
        for (i = 0; i < row; ++i) {
            if (vertices->val[i] < UNPT) {
                count            = 0;
                vertices->val[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (vertices->val[NJA[j]] < UNPT) &&
                        (count < max_aggregation)) {
                        vertices->val[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }
#endif

    // fasp_gettime(&end_time);
    // printf("Step 3 time: %f\n", end_time - start_time);

    // printf("%s, NumAggregates = %d\n", __FUNCTION__, *NumAggregates);

    fasp_mem_free(num_each_agg);
    num_each_agg = NULL;

END:
    fasp_mem_free(temp_C);
    temp_C = NULL;

    return status;
}

/**
 * \fn static SHORT aggregation_vmb_bsr (dBSRmat *A, ivector *vertices,
 *                                       AMG_param *param,
 *                                       const INT NumLevels, dCSRmat *Neigh,
 *                                       INT *NumAggregates)
 *
 * \brief Form aggregation based on strong coupled neighbors for BSR matrices
 *
 * \param A                 Pointer to the coefficient BSR matrices
 * \param vertices          Pointer to the aggregation of vertices
 * \param param             Pointer to AMG parameters
 * \param NumLevels         Level number
 * \param Neigh             Pointer to strongly coupled neighbors
 * \param NumAggregates     Pointer to number of aggregations
 *
 * \author Li Zhao
 * \date   05/21/2023
 *
 * \note Setup A, P, PT and levels using the unsmoothed aggregation algorithm;
 *       Refer to
 *       (1) P. Vanek, J. Madel and M. Brezina.
 *       "Algebraic Multigrid on Unstructured Meshes", 1994.
 *       (2) P. Vanek, J. Madel and M. Brezina.
 *       "Algebraic Multigrid by Smoothed Aggregation for Second and Fourth Order
 *        Elliptic Problems", 1995
 *
 */
static SHORT aggregation_vmb_bsr(dBSRmat*   A,
                                 ivector*   vertices,
                                 AMG_param* param,
                                 const INT  NumLevels,
                                 dCSRmat*   Neigh,
                                 INT*       NumAggregates)
{
    const INT  row = A->ROW, col = A->COL, nnz = A->NNZ; // A->IA[row] - A->IA[0];
    const INT *AIA = A->IA, *AJA = A->JA;
    REAL*      Aval = A->val;
    const INT  nb = A->nb, nb2 = nb * nb;
    const INT  max_aggregation = param->max_aggregation;
    const INT  norm_type       = param->aggregation_norm_type;
    // printf("%s aggregation_norm_type: %d\n", __FUNCTION__, norm_type);
    // type = -1: fro-norm, 0: 0-norm (max value), 1: 1-norm, 2: 2-norm, 3: inf-norm
    //  11: get pressure block A_11
    //  22: get temperature block A_22
    //  33: get N1 block A_33
    //  44: get N2 block A_33

    // return status
    SHORT status = FASP_SUCCESS;

    // local variables
    INT  num_left = row;
    INT  subset, count;
    INT* num_each_agg;

    REAL  strongly_coupled, strongly_coupled2, Aij_norm;
    INT   i, j, index, row_start, row_end;
    INT * NIA, *NJA;
    REAL* Nval;

    dvector diag_block      = fasp_dvec_create(row * nb2);
    dvector diag_block_norm = fasp_dvec_create(row);
    fasp_dbsr_getdiag(0, A, diag_block.val); // get the block diagonal entries

    /*-----------------------------------------------*/
    /*   Calculate norm of block diagonal entries    */
    /*-----------------------------------------------*/
#ifdef _OPENMP
#pragma omp parallel for if (row > OPENMP_HOLDS)
#endif
    for (i = 0; i < row; i++) {
        diag_block_norm.val[i] =
            dense_matrix_norms(&diag_block.val[i * nb2], nb, nb, norm_type);
    }

    if (GE(param->tentative_smooth, SMALLREAL)) {
        strongly_coupled = param->strong_coupled * pow(0.5, NumLevels - 1);
    } else {
        strongly_coupled = param->strong_coupled;
    }
    strongly_coupled2 = pow(strongly_coupled, 2);

    /*------------------------------------------*/
    /*    Form strongly coupled neighborhood    */
    /*------------------------------------------*/
    fasp_dcsr_alloc(row, col, nnz, Neigh);

    NIA  = Neigh->IA;
    NJA  = Neigh->JA;
    Nval = Neigh->val;

    for (index = i = 0; i < row; ++i) {
        NIA[i]    = index;
        row_start = AIA[i];
        row_end   = AIA[i + 1];
        for (j = row_start; j < row_end; ++j) {
            Aij_norm = dense_matrix_norms(&Aval[j * nb2], nb, nb, norm_type);

            // printf("(%d, %d), (%e >? %e[%d] * %e[%d] = %e)\n", i, AJA[j],
            //        pow(Aij_norm, 2), diag_block_norm.val[i], i,
            //        diag_block_norm.val[AJA[j]], AJA[j],
            //        diag_block_norm.val[i] * diag_block_norm.val[AJA[j]]);

#if 1
            if ((AJA[j] == i) ||
                (pow(Aij_norm, 2) >=
                 strongly_coupled2 *
                     ABS(diag_block_norm.val[i] * diag_block_norm.val[AJA[j]]))) {
                // if ((AJA[j] == i) ||
                //     (Aij_norm >= strongly_coupled * ABS(diag_block_norm.val[i]))) {
                NJA[index]  = AJA[j];
                Nval[index] = Aij_norm;
                index++;
            }
#else
            if ((AJA[j] == i)) {
                NJA[index]  = AJA[j];
                Nval[index] = Aij_norm;
                index++;
            } else {

                if (pow(Aij_norm, 2) >=
                    strongly_coupled2 *
                        ABS(diag_block_norm.val[i] * diag_block_norm.val[AJA[j]])) {
                    NJA[index]  = AJA[j];
                    Nval[index] = Aij_norm;
                    index++;
                } else {
                    //! 将丢掉的元素加到对角线上
                    // fasp_blas_smat_add(&Aval[row_start * nb2], &Aval[j * nb2],
                    // nb, 1.0,
                    //                    1.0, &Aval[row_start * nb2]);
                    // 清空
                    fasp_blas_smat_add(&Aval[j * nb2], &Aval[j * nb2], nb, 1.0, -1.0,
                                       &Aval[j * nb2]);
                }
            }
#endif
        }
    }
    NIA[row] = index;

    Neigh->nnz = index;
    // Neigh->JA  = (INT*)fasp_mem_realloc(Neigh->JA, (Neigh->IA[row]) * sizeof(INT));
    // Neigh->val = (REAL*)fasp_mem_realloc(Neigh->val, (Neigh->IA[row]) *
    // sizeof(REAL));

    fasp_dvec_free(&diag_block);
    fasp_dvec_free(&diag_block_norm);

    /*------------------------------------------*/
    /*             Initialization               */
    /*------------------------------------------*/
    fasp_ivec_alloc(row, vertices);
    fasp_iarray_set(row, vertices->val, -2);
    *NumAggregates = 0;

    // int* row_num    = (int*)malloc(row * sizeof(int));
    // int* sort_index = (int*)malloc(row * sizeof(int));
    // int  ii;
    // for (i = 0; i < row; ++i) {
    //     row_num[i]    = NIA[i + 1] - NIA[i];
    //     sort_index[i] = i;
    // }
    // fasp_aux_iQuickSortIndex(row_num, 0, row - 1, sort_index);

    // for (i = 0; i < row; ++i) {
    //     printf("row_num[%d]=%d => sort row_num[%d]=%d\n", i, row_num[i],
    //     sort_index[i],
    //            row_num[sort_index[i]]);
    // }
    // exit(0);

    /*-------------*/
    /*   Step 1.   */
    /*-------------*/
    for (i = 0; i < row; ++i) {
        // for (i = row - 1; i >= 0; i--) {
        // for (ii = 0; ii < row; ++ii) {
        //     i = sort_index[row - 1 - ii];

        if ((AIA[i + 1] - AIA[i]) == 1) {
            vertices->val[i] = UNPT;
            num_left--;
        } else {
            subset    = TRUE;
            row_start = NIA[i];
            row_end   = NIA[i + 1];
            for (j = row_start; j < row_end; ++j) {
                if (vertices->val[NJA[j]] >= UNPT) {
                    subset = FALSE;
                    break;
                }
            }
            if (subset) {
                count            = 0;
                vertices->val[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (count < max_aggregation)) {
                        vertices->val[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }

    // printf("DEBUG1: %s NumAggregates = %d\n", __FUNCTION__, *NumAggregates);
    // free(row_num);
    // free(sort_index);

    /*-------------*/
    /*   Step 2.   */
    /*-------------*/
    INT* temp_C = (INT*)fasp_mem_calloc(row, sizeof(INT));

    if (*NumAggregates < MIN_CDOF) {
        status = ERROR_AMG_COARSEING;
        printf("DEBUG: %s NumAggregates = %d\n", __FUNCTION__, *NumAggregates);
        goto END;
    }

    num_each_agg = (INT*)fasp_mem_calloc(*NumAggregates, sizeof(INT));

    // for ( i = 0; i < *NumAggregates; i++ ) num_each_agg[i] = 0; // initialize

    for (i = row; i--;) {
        temp_C[i] = vertices->val[i];
        if (vertices->val[i] >= 0) num_each_agg[vertices->val[i]]++;
    }

    for (i = 0; i < row; ++i) {
        if (vertices->val[i] < UNPT) {
            row_start = NIA[i];
            row_end   = NIA[i + 1];

#if 1
            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > UNPT &&
                    num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
                    vertices->val[i] = temp_C[NJA[j]];
                    num_left--;
                    num_each_agg[temp_C[NJA[j]]]++;
                    break;
                }
            }
#else
            // try: if more than one such set exists, choose the one to which node i
            // has the strongest coupling,
            // The result is even worse, Li Zhao
            REAL strongest_coupling = 0.0;
            INT  this_id            = -1;
            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > UNPT &&
                    num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
                    if (fabs(Nval[j]) > strongest_coupling) {
                        strongest_coupling = fabs(Nval[j]);
                        this_id            = j;
                    }
                }
            }

            if (this_id > -1) {
                vertices->val[i] = temp_C[NJA[this_id]];
                num_left--;
                num_each_agg[temp_C[NJA[this_id]]]++;
            }
#endif
        }
    }

    // printf("DEBUG2: %s NumAggregates = %d\n", __FUNCTION__, *NumAggregates);
    // printf("%s, Step 3 num_left = %d\n", __FUNCTION__, num_left);

    /*-------------*/
    /*   Step 3.   */
    /*-------------*/

    // Optimal Implementation, Li Zhao
    // REAL start_time, end_time;
    // fasp_gettime(&start_time);
#if 0
    if (num_left > 0) {
        ivector left_map;
        fasp_ivec_alloc(row, &left_map);
        left_map.row = 0;
        for (i = 0; i < row; ++i) {
            if (vertices->val[i] < UNPT) {
                left_map.val[left_map.row++] = i;
            }
        }
        // printf("left_map.row = %d\n", left_map.row);

        int g;
        while (num_left > 0) {
            for (g = 0; g < left_map.row; g++) {
                i = left_map.val[g];
                if (vertices->val[i] < UNPT) {
                    count            = 0;
                    vertices->val[i] = *NumAggregates;
                    num_left--;
                    count++;
                    row_start = NIA[i];
                    row_end   = NIA[i + 1];
                    for (j = row_start; j < row_end; ++j) {
                        if ((NJA[j] != i) && (vertices->val[NJA[j]] < UNPT) &&
                            (count < max_aggregation)) {
                            vertices->val[NJA[j]] = *NumAggregates;
                            num_left--;
                            count++;
                        }
                    }
                    (*NumAggregates)++;
                }
            }
        }

        fasp_ivec_free(&left_map);
    }

#else
    while (num_left > 0) {
        for (i = 0; i < row; ++i) {
            if (vertices->val[i] < UNPT) {
                count            = 0;
                vertices->val[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (vertices->val[NJA[j]] < UNPT) &&
                        (count < max_aggregation)) {
                        vertices->val[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }
#endif

    // fasp_gettime(&end_time);
    // printf("Step 3 time: %f\n", end_time - start_time);

    // printf("%s, NumAggregates = %d\n", __FUNCTION__, *NumAggregates);

    fasp_mem_free(num_each_agg);
    num_each_agg = NULL;

END:
    fasp_mem_free(temp_C);
    temp_C = NULL;

    return status;
}

/**
 * \fn static SHORT aggregation_vmb_bsr_sa (dBSRmat *A, ivector *vertices,
 *                                          AMG_param *param,
 *                                          const INT NumLevels, dBSRmat *Neigh,
 *                                          INT *NumAggregates)
 *
 * \brief Form aggregation based on strong coupled neighbors for BSR matrices
 *
 * \param A                 Pointer to the coefficient BSR matrices
 * \param vertices          Pointer to the aggregation of vertices
 * \param param             Pointer to AMG parameters
 * \param NumLevels         Level number
 * \param Neigh             Pointer to strongly coupled neighbors (BSR matrices)
 * \param NumAggregates     Pointer to number of aggregations
 *
 * \author Li Zhao
 * \date   06/01/2023
 *
 * \note Setup A, P, PT and levels using the unsmoothed aggregation algorithm;
 *       Refer to
 *       (1) P. Vanek, J. Madel and M. Brezina.
 *       "Algebraic Multigrid on Unstructured Meshes", 1994.
 *       (2) P. Vanek, J. Madel and M. Brezina.
 *       "Algebraic Multigrid by Smoothed Aggregation for Second and Fourth Order
 *        Elliptic Problems", 1995
 *
 * The difference from "aggregation_vmb_bsr" lies in the format of the output "Neigh"
 * matrix (BSR), this will be used to filt A for Smoothed Aggregation.
 */
static SHORT aggregation_vmb_bsr_sa(dBSRmat*   A,
                                    ivector*   vertices,
                                    AMG_param* param,
                                    const INT  NumLevels,
                                    dBSRmat*   Neigh,
                                    INT*       NumAggregates)
{
    const INT  row = A->ROW, col = A->COL, nnz = A->NNZ; // A->IA[row] - A->IA[0];
    const INT *AIA = A->IA, *AJA = A->JA;
    REAL*      Aval = A->val;
    const INT  nb = A->nb, nb2 = nb * nb;
    const INT  max_aggregation = param->max_aggregation;
    // type = -1: fro-norm, 0: 0-norm (max value), 1: 1-norm, 2: 2-norm, 3: inf-norm
    //  4: get pressure block A_00
    const INT norm_type = -1;

    // return status
    SHORT status = FASP_SUCCESS;

    // local variables
    INT  num_left = row;
    INT  subset, count;
    INT* num_each_agg;

    REAL  strongly_coupled, strongly_coupled2, Aij_norm;
    INT   i, j, index, row_start, row_end;
    INT * NIA, *NJA;
    REAL* Nval;

    dvector diag_block      = fasp_dvec_create(row * nb2);
    dvector diag_block_norm = fasp_dvec_create(row);
    fasp_dbsr_getdiag(0, A, diag_block.val); // get the block diagonal entries

    /*-----------------------------------------------*/
    /*   Calculate norm of block diagonal entries    */
    /*-----------------------------------------------*/
#ifdef _OPENMP
#pragma omp parallel for if (row > OPENMP_HOLDS)
#endif
    for (i = 0; i < row; i++) {
        diag_block_norm.val[i] =
            dense_matrix_norms(&diag_block.val[i * nb2], nb, nb, norm_type);
    }

    if (GE(param->tentative_smooth, SMALLREAL)) {
        strongly_coupled = param->strong_coupled * pow(0.5, NumLevels - 1);
    } else {
        strongly_coupled = param->strong_coupled;
    }
    strongly_coupled2 = pow(strongly_coupled, 2);

    /*------------------------------------------*/
    /*    Form strongly coupled neighborhood    */
    /*------------------------------------------*/
    // fasp_dcsr_alloc(row, col, nnz, Neigh);
    fasp_dbsr_alloc(row, col, nnz, nb, 0, Neigh);

    NIA  = Neigh->IA;
    NJA  = Neigh->JA;
    Nval = Neigh->val;

    for (index = i = 0; i < row; ++i) {
        NIA[i]    = index;
        row_start = AIA[i];
        row_end   = AIA[i + 1];
        for (j = row_start; j < row_end; ++j) {
            Aij_norm = dense_matrix_norms(&Aval[j * nb2], nb, nb, norm_type);
            if ((AJA[j] == i) ||
                (pow(Aij_norm, 2) >=
                 strongly_coupled2 *
                     ABS(diag_block_norm.val[i] * diag_block_norm.val[AJA[j]]))) {
                NJA[index] = AJA[j];
                // Nval[index] = Aij_norm;

                // copy A_ij to N_ij
                memcpy(Nval + (index * nb2), Aval + (j * nb2), nb2 * sizeof(REAL));

                index++;
            }
        }
    }
    NIA[row] = index;

    Neigh->NNZ = index;
    // Neigh->JA  = (INT*)fasp_mem_realloc(Neigh->JA, (Neigh->IA[row]) * sizeof(INT));
    // Neigh->val = (REAL*)fasp_mem_realloc(Neigh->val, (Neigh->IA[row]) *
    // sizeof(REAL));

    fasp_dvec_free(&diag_block);
    fasp_dvec_free(&diag_block_norm);

    /*------------------------------------------*/
    /*             Initialization               */
    /*------------------------------------------*/
    fasp_ivec_alloc(row, vertices);
    fasp_iarray_set(row, vertices->val, -2);
    *NumAggregates = 0;

    /*-------------*/
    /*   Step 1.   */
    /*-------------*/
    for (i = 0; i < row; ++i) {
        if ((AIA[i + 1] - AIA[i]) == 1) {
            vertices->val[i] = UNPT;
            num_left--;
        } else {
            subset    = TRUE;
            row_start = NIA[i];
            row_end   = NIA[i + 1];
            for (j = row_start; j < row_end; ++j) {
                if (vertices->val[NJA[j]] >= UNPT) {
                    subset = FALSE;
                    break;
                }
            }
            if (subset) {
                count            = 0;
                vertices->val[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (count < max_aggregation)) {
                        vertices->val[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }

    /*-------------*/
    /*   Step 2.   */
    /*-------------*/
    INT* temp_C = (INT*)fasp_mem_calloc(row, sizeof(INT));

    if (*NumAggregates < MIN_CDOF) {
        status = ERROR_AMG_COARSEING;
        goto END;
    }

    num_each_agg = (INT*)fasp_mem_calloc(*NumAggregates, sizeof(INT));

    // for ( i = 0; i < *NumAggregates; i++ ) num_each_agg[i] = 0; // initialize

    for (i = row; i--;) {
        temp_C[i] = vertices->val[i];
        if (vertices->val[i] >= 0) num_each_agg[vertices->val[i]]++;
    }

    for (i = 0; i < row; ++i) {
        if (vertices->val[i] < UNPT) {
            row_start = NIA[i];
            row_end   = NIA[i + 1];

#if 1
            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > UNPT &&
                    num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
                    vertices->val[i] = temp_C[NJA[j]];
                    num_left--;
                    num_each_agg[temp_C[NJA[j]]]++;
                    break;
                }
            }
#else
            // try: if more than one such set exists, choose the one to which node i
            // has the strongest coupling,
            // The result is even worse, Li Zhao
            REAL strongest_coupling = 0.0;
            INT  this_id            = -1;
            for (j = row_start; j < row_end; ++j) {
                if (temp_C[NJA[j]] > UNPT &&
                    num_each_agg[temp_C[NJA[j]]] < max_aggregation) {
                    if (fabs(Nval[j]) > strongest_coupling) {
                        strongest_coupling = fabs(Nval[j]);
                        this_id            = j;
                    }
                }
            }

            if (this_id > -1) {
                vertices->val[i] = temp_C[NJA[this_id]];
                num_left--;
                num_each_agg[temp_C[NJA[this_id]]]++;
            }
#endif
        }
    }

    /*-------------*/
    /*   Step 3.   */
    /*-------------*/
    while (num_left > 0) {
        for (i = 0; i < row; ++i) {
            if (vertices->val[i] < UNPT) {
                count            = 0;
                vertices->val[i] = *NumAggregates;
                num_left--;
                count++;
                row_start = NIA[i];
                row_end   = NIA[i + 1];
                for (j = row_start; j < row_end; ++j) {
                    if ((NJA[j] != i) && (vertices->val[NJA[j]] < UNPT) &&
                        (count < max_aggregation)) {
                        vertices->val[NJA[j]] = *NumAggregates;
                        num_left--;
                        count++;
                    }
                }
                (*NumAggregates)++;
            }
        }
    }

    fasp_mem_free(num_each_agg);
    num_each_agg = NULL;

END:
    fasp_mem_free(temp_C);
    temp_C = NULL;

    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
