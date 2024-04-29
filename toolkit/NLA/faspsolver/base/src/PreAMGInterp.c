/*! \file  PreAMGInterp.c
 *
 *  \brief Direct and standard interpolations for classical AMG
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxThreads.c,
 *         and PreAMGInterpEM.c
 *
 *  Reference:
 *         U. Trottenberg, C. W. Oosterlee, and A. Schuller
 *         Multigrid (Appendix A: An Intro to Algebraic Multigrid)
 *         Academic Press Inc., San Diego, CA, 2001
 *         With contributions by A. Brandt, P. Oswald and K. Stuben.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static void interp_DIR(dCSRmat*, ivector*, dCSRmat*, AMG_param*);
static void interp_RDC(dCSRmat*, ivector*, dCSRmat*, AMG_param*);
static void interp_STD(dCSRmat*, ivector*, dCSRmat*, iCSRmat*, AMG_param*);
static void interp_EXT(dCSRmat*, ivector*, dCSRmat*, iCSRmat*, AMG_param*);
static void amg_interp_trunc(dCSRmat*, AMG_param*);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_amg_interp (dCSRmat *A, ivector *vertices, dCSRmat *P,
 *                           iCSRmat *S, AMG_param *param)
 *
 * \brief Generate interpolation operator P
 *
 * \param A          Pointer to dCSRmat coefficient matrix (index starts from 0)
 * \param vertices   Indicator vector for the C/F splitting of the variables
 * \param P          Prolongation (input: nonzero pattern, output: prolongation)
 * \param S          Strong connection matrix
 * \param param      AMG parameters
 *
 * \author  Xuehai Huang, Chensong Zhang
 * \date    04/04/2010
 *
 * Modified by Xiaozhe Hu on 05/23/2012: add S as input
 * Modified by Chensong Zhang on 09/12/2012: clean up and debug interp_RS
 * Modified by Chensong Zhang on 05/14/2013: reconstruct the code
 */
void fasp_amg_interp(
    dCSRmat* A, ivector* vertices, dCSRmat* P, iCSRmat* S, AMG_param* param)
{
    const INT coarsening_type = param->coarsening_type;
    INT       interp_type     = param->interpolation_type;

    // make sure standard interpolation is used for aggressive coarsening
    if (coarsening_type == COARSE_AC) interp_type = INTERP_STD;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    switch (interp_type) {

        case INTERP_DIR: // Direct interpolation
            interp_DIR(A, vertices, P, param);
            break;

        case INTERP_STD: // Standard interpolation
            interp_STD(A, vertices, P, S, param);
            break;

        case INTERP_EXT: // Extended interpolation
            interp_EXT(A, vertices, P, S, param);
            break;

        case INTERP_ENG: // Energy-min interpolation defined in PreAMGInterpEM.c
            fasp_amg_interp_em(A, vertices, P, param);
            break;

        case INTERP_RDC: // Reduction-based interpolation
            interp_RDC(A, vertices, P, param);
            break;

        default:
            fasp_chkerr(ERROR_AMG_INTERP_TYPE, __FUNCTION__);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void amg_interp_trunc (dCSRmat *P, AMG_param *param)
 *
 * \brief Truncation step for prolongation operators
 *
 * \param P        Prolongation (input: full, output: truncated)
 * \param param    Pointer to AMG_param: AMG parameters
 *
 * \author Chensong Zhang
 * \date   05/14/2013
 *
 * Originally by Xuehai Huang, Chensong Zhang on 01/31/2009
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012: add OMP support
 * Modified by Chensong Zhang on 05/14/2013: rewritten
 */
static void amg_interp_trunc(dCSRmat* P, AMG_param* param)
{
    const INT  row    = P->row;
    const INT  nnzold = P->nnz;
    const INT  prtlvl = param->print_level;
    const REAL eps_tr = param->truncation_threshold;

    // local variables
    INT  num_nonzero = 0;   // number of non zeros after truncation
    REAL Min_neg, Max_pos;  // min negative and max positive entries
    REAL Fac_neg, Fac_pos;  // factors for negative and positive entries
    REAL Sum_neg, TSum_neg; // sum and truncated sum of negative entries
    REAL Sum_pos, TSum_pos; // sum and truncated sum of positive entries

    INT index1 = 0, index2 = 0, begin, end;
    INT i, j;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    for (i = 0; i < row; ++i) {

        begin = P->IA[i];
        end   = P->IA[i + 1];

        P->IA[i] = num_nonzero;
        Min_neg = Max_pos = 0;
        Sum_neg = Sum_pos = 0;
        TSum_neg = TSum_pos = 0;

        // 1. Summations of positive and negative entries
        for (j = begin; j < end; ++j) {

            if (P->val[j] > 0) {
                Sum_pos += P->val[j];
                Max_pos = MAX(Max_pos, P->val[j]);
            }

            else {
                Sum_neg += P->val[j];
                Min_neg = MIN(Min_neg, P->val[j]);
            }
        }

        // Truncate according to max and min values!!!
        Max_pos *= eps_tr;
        Min_neg *= eps_tr;

        // 2. Set JA of truncated P
        for (j = begin; j < end; ++j) {

            if (P->val[j] >= Max_pos) {
                num_nonzero++;
                P->JA[index1++] = P->JA[j];
                TSum_pos += P->val[j];
            }

            else if (P->val[j] <= Min_neg) {
                num_nonzero++;
                P->JA[index1++] = P->JA[j];
                TSum_neg += P->val[j];
            }
        }

        // 3. Compute factors and set values of truncated P
        if (TSum_pos > SMALLREAL) {
            Fac_pos = Sum_pos / TSum_pos; // factor for positive entries
        } else {
            Fac_pos = 1.0;
        }

        if (TSum_neg < -SMALLREAL) {
            Fac_neg = Sum_neg / TSum_neg; // factor for negative entries
        } else {
            Fac_neg = 1.0;
        }

        for (j = begin; j < end; ++j) {

            if (P->val[j] >= Max_pos)
                P->val[index2++] = P->val[j] * Fac_pos;

            else if (P->val[j] <= Min_neg)
                P->val[index2++] = P->val[j] * Fac_neg;
        }
    }

    // resize the truncated prolongation P
    P->nnz = P->IA[row] = num_nonzero;
    P->JA               = (INT*)fasp_mem_realloc(P->JA, num_nonzero * sizeof(INT));
    P->val              = (REAL*)fasp_mem_realloc(P->val, num_nonzero * sizeof(REAL));

    if (prtlvl >= PRINT_MOST) {
        printf("NNZ in prolongator: before truncation = %10d, after = %10d\n", nnzold,
               num_nonzero);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif
}

/**
 * @brief Reduction-based AMG interpolation
 *
 * @author Yan Xie
 * @date   2022/12/06
 *
 * use all coarse points as interpolation points
 * D_ii = (2 - 1/theta) * A_ii, |A_ii| > theta * sum(|A_ij|) for j as F-points
 * P_F = - D_FF^-1 * A_FC
 */
static void interp_RDC(dCSRmat* A, ivector* vertices, dCSRmat* P, AMG_param* param)
{
    INT  row = A->row;
    INT* vec = vertices->val;

    // local variables
    INT        i, j, k, l, index, idiag;
    const REAL alpha = 2.0 - 1.0 / param->theta;

    // first pass: P->val
    for (index = i = 0; i < row; ++i) {
        if (vec[i] == CGPT) { // identity for coarse points
            P->val[index++] = 1.0;
        } else { // interpolation for fine points
            for (j = A->IA[i]; j < A->IA[i + 1]; ++j) {
                if (A->JA[j] == i) {
                    idiag = j; // found diagonal entry
                    break;
                }
            }
            REAL Dii = alpha * A->val[idiag];
            // fill in entries for P
            for (j = A->IA[i]; j < A->IA[i + 1]; ++j) {
                if (vec[A->JA[j]] == CGPT) {
                    P->val[index++] = -A->val[j] / Dii;
                }
            }
        }
    }

    // second pass: P->JA renumber column index for coarse space
    INT* cindex = (INT*)fasp_mem_calloc(row, sizeof(INT));
    // index of coarse points from fine-space to coarse-space
    for (index = i = 0; i < row; ++i) {
        if (vec[i] == CGPT) cindex[i] = index++;
    }
    for (i = 0; i < P->nnz; ++i) P->JA[i] = cindex[P->JA[i]];

    // clean up
    fasp_mem_free(cindex);
}

/**
 * \fn static void interp_DIR (dCSRmat *A, ivector *vertices, dCSRmat *P,
 *                             AMG_param *param)
 *
 * \brief Direct interpolation
 *
 * \param A          Pointer to dCSRmat: the coefficient matrix (index starts from 0)
 * \param vertices   Indicator vector for the C/F splitting of the variables
 * \param P          Prolongation (input: nonzero pattern, output: prolongation)
 * \param param      Pointer to AMG_param: AMG parameters
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   01/31/2009
 *
 * Modified by Chunsheng Feng on 07/17/2018: Fix a potential bug for "row =
 * MIN(P->IA[P->row], row)" Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012: add
 * OMP support Modified by Chensong Zhang on 09/12/2012: compare with the old version
 * Modified by Chensong Zhang on 05/14/2013: reconstruct the code
 * Modified by Chensong Zhang on 01/13/2017: make this as the default
 */
static void interp_DIR(dCSRmat* A, ivector* vertices, dCSRmat* P, AMG_param* param)
{
    INT  row = A->row;
    INT* vec = vertices->val;

    // local variables
    SHORT IS_STRONG;   // is the variable strong coupled to i?
    INT   num_pcouple; // number of positive strong couplings
    INT   begin_row, end_row;
    INT   i, j, k, l, index = 0, idiag;

    // a_minus and a_plus for Neighbors and Prolongation support
    REAL amN, amP, apN, apP;
    REAL alpha, beta, aii = 0.0;

    // indices of C-nodes
    INT* cindex = (INT*)fasp_mem_calloc(row, sizeof(INT));

    SHORT use_openmp = FALSE;

#ifdef _OPENMP
    INT myid, mybegin, myend, stride_i, nthreads;
    //    row = MIN(P->IA[P->row], row);
    if (MIN(P->IA[P->row], row) > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // Step 1. Fill in values for interpolation operator P
    if (use_openmp) {
#ifdef _OPENMP
        stride_i = row / nthreads;
#pragma omp parallel private(myid, mybegin, myend, i, begin_row, end_row, idiag, aii,  \
                             amN, amP, apN, apP, num_pcouple, j, k, alpha, beta, l)    \
    num_threads(nthreads)
        {
            myid    = omp_get_thread_num();
            mybegin = myid * stride_i;
            if (myid < nthreads - 1)
                myend = mybegin + stride_i;
            else
                myend = row;
            aii = 0.0;

            for (i = mybegin; i < myend; ++i) {
                begin_row = A->IA[i];
                end_row   = A->IA[i + 1] - 1;
                for (idiag = begin_row; idiag <= end_row; idiag++) {
                    if (A->JA[idiag] == i) {
                        aii = A->val[idiag];
                        break;
                    }
                }
                if (vec[i] == 0) { // if node i is on fine grid
                    amN = 0, amP = 0, apN = 0, apP = 0, num_pcouple = 0;
                    for (j = begin_row; j <= end_row; ++j) {
                        if (j == idiag) continue;
                        for (k = P->IA[i]; k < P->IA[i + 1]; ++k) {
                            if (P->JA[k] == A->JA[j]) break;
                        }
                        if (A->val[j] > 0) {
                            apN += A->val[j];
                            if (k < P->IA[i + 1]) {
                                apP += A->val[j];
                                num_pcouple++;
                            }
                        } else {
                            amN += A->val[j];
                            if (k < P->IA[i + 1]) {
                                amP += A->val[j];
                            }
                        }
                    } // j

                    // avoid division by zero for amP and apP
                    amP   = (amP < -SMALLREAL) ? amP : -SMALLREAL;
                    apP   = (apP > SMALLREAL) ? apP : SMALLREAL;
                    alpha = amN / amP;
                    if (num_pcouple > 0) {
                        beta = apN / apP;
                    } else {
                        beta = 0;
                        aii += apN;
                    }
                    for (j = P->IA[i]; j < P->IA[i + 1]; ++j) {
                        k = P->JA[j];
                        for (l = A->IA[i]; l < A->IA[i + 1]; l++) {
                            if (A->JA[l] == k) break;
                        }
                        if (A->val[l] > 0) {
                            P->val[j] = -beta * A->val[l] / aii;
                        } else {
                            P->val[j] = -alpha * A->val[l] / aii;
                        }
                    }
                } else if (vec[i] == 2) // if node i is a special fine node
                {

                } else { // if node i is on coarse grid
                    P->val[P->IA[i]] = 1;
                }
            }
        }
#endif
    }

    else {

        for (i = 0; i < row; ++i) {

            begin_row = A->IA[i];
            end_row   = A->IA[i + 1];

            // find diagonal entry first!!!
            for (idiag = begin_row; idiag < end_row; idiag++) {
                if (A->JA[idiag] == i) {
                    aii = A->val[idiag];
                    break;
                }
            }

            if (vec[i] == FGPT) { // fine grid nodes

                amN = amP = apN = apP = 0.0;

                num_pcouple = 0;

                for (j = begin_row; j < end_row; ++j) {

                    if (j == idiag) continue; // skip diagonal

                    // check a point strong-coupled to i or not
                    IS_STRONG = FALSE;
                    for (k = P->IA[i]; k < P->IA[i + 1]; ++k) {
                        if (P->JA[k] == A->JA[j]) {
                            IS_STRONG = TRUE;
                            break;
                        }
                    }

                    if (A->val[j] > 0) {
                        apN += A->val[j]; // sum up positive entries
                        if (IS_STRONG) {
                            apP += A->val[j];
                            num_pcouple++;
                        }
                    } else {
                        amN += A->val[j]; // sum up negative entries
                        if (IS_STRONG) {
                            amP += A->val[j];
                        }
                    }
                } // end for j

                // set weight factors
                // avoid division by zero for amP and apP
                amP   = (amP < -SMALLREAL) ? amP : -SMALLREAL;
                apP   = (apP > SMALLREAL) ? apP : SMALLREAL;
                alpha = amN / amP;
                if (num_pcouple > 0) {
                    beta = apN / apP;
                } else {
                    beta = 0.0;
                    aii += apN;
                }

                // keep aii inside the loop to avoid floating pt error! --Chensong
                for (j = P->IA[i]; j < P->IA[i + 1]; ++j) {
                    k = P->JA[j];
                    for (l = A->IA[i]; l < A->IA[i + 1]; l++) {
                        if (A->JA[l] == k) break;
                    }
                    if (A->val[l] > 0) {
                        P->val[j] = -beta * A->val[l] / aii;
                    } else {
                        P->val[j] = -alpha * A->val[l] / aii;
                    }
                }

            } // end if vec

            else if (vec[i] == CGPT) { // coarse grid nodes
                P->val[P->IA[i]] = 1.0;
            }
        }
    }

    // Step 2. Generate coarse level indices and set values of P.JA
    for (index = i = 0; i < row; ++i) {
        if (vec[i] == CGPT) cindex[i] = index++;
    }
    P->col = index;

    if (use_openmp) {
#ifdef _OPENMP
        stride_i = P->IA[P->row] / nthreads;
#pragma omp parallel private(myid, mybegin, myend, i, j) num_threads(nthreads)
        {
            myid    = omp_get_thread_num();
            mybegin = myid * stride_i;
            if (myid < nthreads - 1)
                myend = mybegin + stride_i;
            else
                myend = P->IA[P->row];
            for (i = mybegin; i < myend; ++i) {
                j        = P->JA[i];
                P->JA[i] = cindex[j];
            }
        }
#endif
    } else {
        for (i = 0; i < P->nnz; ++i) {
            j        = P->JA[i];
            P->JA[i] = cindex[j];
        }
    }

    // clean up
    fasp_mem_free(cindex);
    cindex = NULL;

    // Step 3. Truncate the prolongation operator to reduce cost
    amg_interp_trunc(P, param);
}

/**
 * \fn static void interp_STD (dCSRmat *A, ivector *vertices, dCSRmat *P,
 *                             iCSRmat *S, AMG_param *param)
 *
 * \brief Standard interpolation
 *
 * \param A          Pointer to dCSRmat: the coefficient matrix (index starts from 0)
 * \param vertices   Indicator vector for the C/F splitting of the variables
 * \param P          Interpolation matrix (input: nnz pattern, output: prolongation)
 * \param S          Strong connection matrix
 * \param param      Pointer to AMG_param: AMG parameters
 *
 * \author Kai Yang, Xiaozhe Hu
 * \date   05/21/2012
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/17/2012: add OMP support
 * Modified by Chensong Zhang on 05/15/2013: reconstruct the code
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 12/25/2013: check C1 Criterion
 */
static void interp_STD(dCSRmat* A, ivector* vertices, dCSRmat* P, iCSRmat* S, AMG_param* param)
{
    const INT row = A->row;
    INT*      vec = vertices->val;

    // local variables
    INT  i, j, k, l, m, index;
    REAL alpha = 1.0, factor, alN, alP;
    REAL akk, akl, aik, aki;

    // indices for coarse neighbor node for every node
    INT* cindex = (INT*)fasp_mem_calloc(row, sizeof(INT));

    // indices from column number to index in nonzeros in i-th row
    INT* rindi = (INT*)fasp_mem_calloc(2 * row, sizeof(INT));

    // indices from column number to index in nonzeros in k-th row
    INT* rindk = (INT*)fasp_mem_calloc(2 * row, sizeof(INT));

    // sums of strongly connected C neighbors
    REAL* csum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

#if RS_C1
    // sums of all neighbors except ISPT
    REAL* psum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));
#endif

    // sums of all neighbors
    REAL* nsum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

    // diagonal entries
    REAL* diag = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

    // coefficients hat a_ij for relevant CGPT of the i-th node
    REAL* Ahat = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

    // Step 0. Prepare diagonal, Cs-sum, and N-sum
    fasp_iarray_set(row, cindex, -1);
    fasp_darray_set(row, csum, 0.0);
    fasp_darray_set(row, nsum, 0.0);

    for (i = 0; i < row; i++) {

        // set flags for strong-connected C nodes
        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {
            k = S->JA[j];
            if (vec[k] == CGPT) cindex[k] = i;
        }

        for (j = A->IA[i]; j < A->IA[i + 1]; j++) {
            k = A->JA[j];

            if (cindex[k] == i) csum[i] += A->val[j]; // strong C-couplings

            if (k == i) diag[i] = A->val[j];
#if RS_C1
            else {
                nsum[i] += A->val[j];
                if (vec[k] != ISPT) {
                    psum[i] += A->val[j];
                }
            }
#else
            else
                nsum[i] += A->val[j];
#endif
        }
    }

    // Step 1. Fill in values for interpolation operator P
    for (i = 0; i < row; i++) {

        if (vec[i] == FGPT) {
#if RS_C1
            alN = psum[i];
#else
            alN = nsum[i];
#endif
            alP = csum[i];

            // form the reverse indices for i-th row
            for (j = A->IA[i]; j < A->IA[i + 1]; j++) rindi[A->JA[j]] = j;

            // clean up Ahat for relevant nodes only
            for (j = P->IA[i]; j < P->IA[i + 1]; j++) Ahat[P->JA[j]] = 0.0;

            // set values of Ahat
            Ahat[i] = diag[i];

            for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

                k   = S->JA[j];
                aik = A->val[rindi[k]];

                if (vec[k] == CGPT)
                    Ahat[k] += aik;

                else if (vec[k] == FGPT) {

                    akk = diag[k];

                    // form the reverse indices for k-th row
                    for (m = A->IA[k]; m < A->IA[k + 1]; m++) rindk[A->JA[m]] = m;

                    factor = aik / akk;

                    // visit the strong-connected C neighbors of k, compute
                    // Ahat in the i-th row, set aki if found
                    aki = 0.0;
#if 0 // modified by Xiaoqiang Yue 12/25/2013
                    for ( m = S->IA[k]; m < S->IA[k+1]; m++ ) {
                        l   = S->JA[m];
                        akl = A->val[rindk[l]];
                        if ( vec[l] == CGPT ) Ahat[l] -= factor * akl;
                        else if ( l == i ) {
                            aki = akl; Ahat[l] -= factor * aki;
                        }
                    } // end for m
#else
                    for (m = A->IA[k]; m < A->IA[k + 1]; m++) {
                        if (A->JA[m] == i) {
                            aki = A->val[m];
                            Ahat[i] -= factor * aki;
                        }
                    } // end for m
#endif
                    for (m = S->IA[k]; m < S->IA[k + 1]; m++) {
                        l   = S->JA[m];
                        akl = A->val[rindk[l]];
                        if (vec[l] == CGPT) Ahat[l] -= factor * akl;
                    } // end for m

                    // compute Cs-sum and N-sum for Ahat
                    alN -= factor * (nsum[k] - aki + akk);
                    alP -= factor * csum[k];

                } // end if vec[k]

            } // end for j

            // Originally: alpha = alN/alP, do this only if P is not empty!
            if (P->IA[i] < P->IA[i + 1]) alpha = alN / alP;

            // How about positive entries? --Chensong
            for (j = P->IA[i]; j < P->IA[i + 1]; j++) {
                k         = P->JA[j];
                P->val[j] = -alpha * Ahat[k] / Ahat[i];
            }

        }

        else if (vec[i] == CGPT) {
            P->val[P->IA[i]] = 1.0;
        }

    } // end for i

    // Step 2. Generate coarse level indices and set values of P.JA
    for (index = i = 0; i < row; ++i) {
        if (vec[i] == CGPT) cindex[i] = index++;
    }
    P->col = index;

#ifdef _OPENMP
#pragma omp parallel for private(i, j) if (P->IA[P->row] > OPENMP_HOLDS)
#endif
    for (i = 0; i < P->IA[P->row]; ++i) {
        j        = P->JA[i];
        P->JA[i] = cindex[j];
    }

    // clean up
    fasp_mem_free(cindex);
    cindex = NULL;
    fasp_mem_free(rindi);
    rindi = NULL;
    fasp_mem_free(rindk);
    rindk = NULL;
    fasp_mem_free(nsum);
    nsum = NULL;
    fasp_mem_free(csum);
    csum = NULL;
    fasp_mem_free(diag);
    diag = NULL;
    fasp_mem_free(Ahat);
    Ahat = NULL;

#if RS_C1
    fasp_mem_free(psum);
    psum = NULL;
#endif

    // Step 3. Truncate the prolongation operator to reduce cost
    amg_interp_trunc(P, param);
}

/**
 * \fn static void interp_EXT (dCSRmat *A, ivector *vertices, dCSRmat *P,
 *                             iCSRmat *S, AMG_param *param)
 *
 * \brief Extended interpolation
 *
 * \param A          Pointer to dCSRmat: the coefficient matrix (index starts from 0)
 * \param vertices   Indicator vector for the C/F splitting of the variables
 * \param P          Interpolation matrix (input: nnz pattern, output: prolongation)
 * \param S          Strong connection matrix
 * \param param      Pointer to AMG_param: AMG parameters
 *
 * \author Zheng Li, Chensong Zhang
 * \date   11/21/2014
 *
 * \todo  Need to be fixed!  --zcs
 */
static void interp_EXT(dCSRmat* A, ivector* vertices, dCSRmat* P, iCSRmat* S, AMG_param* param)
{
    const INT row = A->row;
    INT*      vec = vertices->val;

    // local variables
    INT  i, j, k, l, m, index;
    REAL alpha = 1.0, factor, alN, alP;
    REAL akk, akl, aik, aki;

    // indices for coarse neighbor node for every node
    INT* cindex = (INT*)fasp_mem_calloc(row, sizeof(INT));

    // indices from column number to index in nonzeros in i-th row
    INT* rindi = (INT*)fasp_mem_calloc(2 * row, sizeof(INT));

    // indices from column number to index in nonzeros in k-th row
    INT* rindk = (INT*)fasp_mem_calloc(2 * row, sizeof(INT));

    // sums of strongly connected C neighbors
    REAL* csum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

#if RS_C1
    // sums of all neighbors except ISPT
    REAL* psum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));
#endif
    // sums of all neighbors
    REAL* nsum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

    // diagonal entries
    REAL* diag = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

    // coefficients hat a_ij for relevant CGPT of the i-th node
    REAL* Ahat = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

    // Step 0. Prepare diagonal, Cs-sum, and N-sum
    fasp_iarray_set(row, cindex, -1);
    fasp_darray_set(row, csum, 0.0);
    fasp_darray_set(row, nsum, 0.0);

    for (i = 0; i < row; i++) {

        // set flags for strong-connected C nodes
        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {
            k = S->JA[j];
            if (vec[k] == CGPT) cindex[k] = i;
        }

        for (j = A->IA[i]; j < A->IA[i + 1]; j++) {
            k = A->JA[j];

            if (cindex[k] == i) csum[i] += A->val[j]; // strong C-couplings

            if (k == i) diag[i] = A->val[j];
#if RS_C1
            else {
                nsum[i] += A->val[j];
                if (vec[k] != ISPT) {
                    psum[i] += A->val[j];
                }
            }
#else
            else
                nsum[i] += A->val[j];
#endif
        }
    }

    // Step 1. Fill in values for interpolation operator P
    for (i = 0; i < row; i++) {

        if (vec[i] == FGPT) {
#if RS_C1
            alN = psum[i];
#else
            alN = nsum[i];
#endif
            alP = csum[i];

            // form the reverse indices for i-th row
            for (j = A->IA[i]; j < A->IA[i + 1]; j++) rindi[A->JA[j]] = j;

            // clean up Ahat for relevant nodes only
            for (j = P->IA[i]; j < P->IA[i + 1]; j++) Ahat[P->JA[j]] = 0.0;

            // set values of Ahat
            Ahat[i] = diag[i];

            for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

                k   = S->JA[j];
                aik = A->val[rindi[k]];

                if (vec[k] == CGPT)
                    Ahat[k] += aik;

                else if (vec[k] == FGPT) {

                    akk = diag[k];

                    // form the reverse indices for k-th row
                    for (m = A->IA[k]; m < A->IA[k + 1]; m++) rindk[A->JA[m]] = m;

                    factor = aik / akk;

                    // visit the strong-connected C neighbors of k, compute
                    // Ahat in the i-th row, set aki if found
                    aki = 0.0;
#if 0 // modified by Xiaoqiang Yue 12/25/2013
                    for ( m = S->IA[k]; m < S->IA[k+1]; m++ ) {
                        l   = S->JA[m];
                        akl = A->val[rindk[l]];
                        if ( vec[l] == CGPT ) Ahat[l] -= factor * akl;
                        else if ( l == i ) {
                            aki = akl; Ahat[l] -= factor * aki;
                        }
                    } // end for m
#else
                    for (m = A->IA[k]; m < A->IA[k + 1]; m++) {
                        if (A->JA[m] == i) {
                            aki = A->val[m];
                            Ahat[i] -= factor * aki;
                        }
                    } // end for m
#endif
                    for (m = S->IA[k]; m < S->IA[k + 1]; m++) {
                        l   = S->JA[m];
                        akl = A->val[rindk[l]];
                        if (vec[l] == CGPT) Ahat[l] -= factor * akl;
                    } // end for m

                    // compute Cs-sum and N-sum for Ahat
                    alN -= factor * (nsum[k] - aki + akk);
                    alP -= factor * csum[k];

                } // end if vec[k]

            } // end for j

            // Originally: alpha = alN/alP, do this only if P is not empty!
            if (P->IA[i] < P->IA[i + 1]) alpha = alN / alP;

            // How about positive entries? --Chensong
            for (j = P->IA[i]; j < P->IA[i + 1]; j++) {
                k         = P->JA[j];
                P->val[j] = -alpha * Ahat[k] / Ahat[i];
            }

        }

        else if (vec[i] == CGPT) {
            P->val[P->IA[i]] = 1.0;
        }

    } // end for i

    // Step 2. Generate coarse level indices and set values of P.JA
    for (index = i = 0; i < row; ++i) {
        if (vec[i] == CGPT) cindex[i] = index++;
    }
    P->col = index;

#ifdef _OPENMP
#pragma omp parallel for private(i, j) if (P->IA[P->row] > OPENMP_HOLDS)
#endif
    for (i = 0; i < P->IA[P->row]; ++i) {
        j        = P->JA[i];
        P->JA[i] = cindex[j];
    }

    // clean up
    fasp_mem_free(cindex);
    cindex = NULL;
    fasp_mem_free(rindi);
    rindi = NULL;
    fasp_mem_free(rindk);
    rindk = NULL;
    fasp_mem_free(nsum);
    nsum = NULL;
    fasp_mem_free(csum);
    csum = NULL;
    fasp_mem_free(diag);
    diag = NULL;
    fasp_mem_free(Ahat);
    Ahat = NULL;

#if RS_C1
    fasp_mem_free(psum);
    psum = NULL;
#endif

    // Step 3. Truncate the prolongation operator to reduce cost
    amg_interp_trunc(P, param);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
