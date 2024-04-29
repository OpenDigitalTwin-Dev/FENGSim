/*! \file  PreAMGCoarsenRS.c
 *
 *  \brief Coarsening with a modified Ruge-Stuben strategy.
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxMemory.c, AuxMessage.c, AuxThreads.c, AuxVector.c,
 *         BlaSparseCSR.c, and PreAMGCoarsenCR.c
 *
 *  Reference:
 *         Multigrid by U. Trottenberg, C. W. Oosterlee and A. Schuller
 *         Appendix P475 A.7 (by A. Brandt, P. Oswald and K. Stuben)
 *         Academic Press Inc., San Diego, CA, 2001.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "PreAMGUtil.inl"
#include <math.h>

static INT cfsplitting_cls(dCSRmat*, iCSRmat*, ivector*);
static INT cfsplitting_clsp(dCSRmat*, iCSRmat*, ivector*);
static INT cfsplitting_agg(dCSRmat*, iCSRmat*, ivector*, INT);
static INT cfsplitting_mis(iCSRmat*, ivector*, ivector*);
static INT clean_ff_couplings(iCSRmat*, ivector*, INT, INT);
static INT compress_S(iCSRmat*);

static void strong_couplings(dCSRmat*, iCSRmat*, AMG_param*);
static void form_P_pattern_dir(dCSRmat*, iCSRmat*, ivector*, INT, INT);
static void form_P_pattern_std(dCSRmat*, iCSRmat*, ivector*, INT, INT);
static void ordering1(iCSRmat*, ivector*);

static void form_P_pattern_rdc(dCSRmat*, dCSRmat*, double*, ivector*, INT, INT);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn SHORT fasp_amg_coarsening_rs (dCSRmat *A, ivector *vertices, dCSRmat *P,
 *                                   iCSRmat *S, AMG_param *param)
 *
 * \brief Standard and aggressive coarsening schemes
 *
 * \param A          Pointer to dCSRmat: Coefficient matrix (index starts from 0)
 * \param vertices   Indicator vector for the C/F splitting of the variables
 * \param P          Interpolation matrix (nonzero pattern only)
 * \param S          Strong connection matrix
 * \param param      Pointer to AMG_param: AMG parameters
 *
 * \return           FASP_SUCCESS if successed; otherwise, error information.
 *
 * \author Xuehai Huang, Chensong Zhang, Xiaozhe Hu, Ludmil Zikatanov
 * \date   09/06/2010
 *
 * \note vertices = 0: fine; 1: coarse; 2: isolated or special
 *
 * Modified by Xiaozhe Hu on 05/23/2011: add strength matrix as an argument
 * Modified by Xiaozhe Hu on 04/24/2013: modify aggressive coarsening
 * Modified by Chensong Zhang on 04/28/2013: remove linked list
 * Modified by Chensong Zhang on 05/11/2013: restructure the code
 */
SHORT fasp_amg_coarsening_rs(
    dCSRmat* A, ivector* vertices, dCSRmat* P, iCSRmat* S, AMG_param* param)
{
    const SHORT coarse_type = param->coarsening_type;
    const INT   agg_path    = param->aggressive_path;
    const INT   row         = A->row;

    // local variables
    SHORT interp_type = param->interpolation_type;
    INT   col         = 0;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

#if DEBUG_MODE > 1
    printf("### DEBUG: Step 1. Find strong connections ......\n");
#endif

    // make sure standard interp is used for aggressive coarsening
    if (coarse_type == COARSE_AC) interp_type = INTERP_STD;

    // find strong couplings and return them in S
    strong_couplings(A, S, param);

#if DEBUG_MODE > 1
    printf("### DEBUG: Step 2. C/F splitting ......\n");
#endif
    //   printf("### DEBUG: Step 2. C/F splitting ......\n");
    //    printf("coarse_type:%d \n", coarse_type);
    switch (coarse_type) {

        case COARSE_RSP: // Classical coarsening with positive connections
            col = cfsplitting_clsp(A, S, vertices);
            break;

        case COARSE_AC: // Aggressive coarsening
            col = cfsplitting_agg(A, S, vertices, agg_path);
            break;

        case COARSE_CR: // Compatible relaxation
            col = fasp_amg_coarsening_cr(0, row - 1, A, vertices, param);
            break;

        case COARSE_MIS: // Maximal independent set
            {
                ivector order = fasp_ivec_create(row);
                compress_S(S);
                ordering1(S, &order);
                col = cfsplitting_mis(S, vertices, &order);
                fasp_ivec_free(&order);
                break;
            }

        default: // Classical coarsening
            col = cfsplitting_cls(A, S, vertices);
    }

#if DEBUG_MODE > 1
    printf("### DEBUG: col = %d\n", col);
#endif
    //    printf("### DEBUG: col = %d\n", col);
    if (col <= 0) return ERROR_UNKNOWN;

#if DEBUG_MODE > 1
    printf("### DEBUG: Step 3. Find support of C points ......\n");
#endif

    switch (interp_type) {

        case INTERP_DIR: // Direct interpolation or ...
        case INTERP_ENG: // Energy-min interpolation
            col = clean_ff_couplings(S, vertices, row, col);
            form_P_pattern_dir(P, S, vertices, row, col);
            break;

        case INTERP_STD: // Standard interpolation
        case INTERP_EXT: // Extended interpolation
            form_P_pattern_std(P, S, vertices, row, col);
            break;

        case INTERP_RDC: // Reduction-based amg interpolation
            {
                // printf("### DEBUG: Reduction-based interpolation\n");
                INT     i;
                double* theta = (double*)fasp_mem_calloc(row, sizeof(double));
                form_P_pattern_rdc(P, A, theta, vertices, row, col);
                // theta will be used to
                // 1. compute entries of interpolation matrix
                // 2. calculate relaxation parameter
                // 3. approximate convergence factor
                param->theta = 1.0;
                for (i = 0; i < row; ++i)
                    if (theta[i] < param->theta) param->theta = theta[i];
                printf("### DEBUG: theta = %e\n", param->theta);
                fasp_mem_free(theta);
                theta      = NULL;
                double eps = (2 - 2 * param->theta) / (2 * param->theta - 1);
                // assume: two-grid
                double nu = (param->presmooth_iter + param->postsmooth_iter) / 2.0;
                double conv_factor = (eps / (1 + eps)) *
                                     (1 + pow(eps, 2 * nu - 1) / pow(2 + eps, 2 * nu));
                conv_factor = sqrt(conv_factor);
                printf("### DEBUG: Upper bound for conv_factor = %e\n", conv_factor);
                if (param->theta <= 0.5) {
                    REAL reset_value = 0.5 + 1e-5;
                    printf("### WARNING: theta = %e <= 0.5, use %e instead \n",
                           param->theta, reset_value);
                    param->theta = reset_value;
                }
                if (param->theta >= 0.0) {
                    // weighted Jacobi smoother only on F-points
                    REAL theta  = param->theta;
                    REAL eps    = (2 - 2 * theta) / (2 * theta - 1);
                    REAL sigma  = 2 / (2 + eps);
                    REAL weight = sigma / (2 - 1 / theta);
                    printf("### DEBUG: theta = %e, eps = %e, sigma = %e\n", theta, eps,
                           sigma);

                    // set reduction-based AMG smoother parameters
                    if (param->smoother == SMOOTHER_JACOBIF) {
                        param->relaxation = weight;
                        printf("### DEBUG: Weight for JACOBI_F = %e\n", weight);
                    }
                }
                break;
            }

        default:
            fasp_chkerr(ERROR_AMG_INTERP_TYPE, __FUNCTION__);
    }

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void strong_couplings (dCSRmat *A, iCSRmat *S, AMG_param *param)
 *
 * \brief Generate the set of all strong negative couplings
 *
 * \param A          Coefficient matrix, the index starts from zero
 * \param S          Strong connection matrix
 * \param param      AMG parameters
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   09/06/2010
 *
 * \note   For flexibility, we do NOT compress S here!!! It is due to the C/F
 *         splitting routines to decide when to compress S.
 *
 * Modified by Chensong Zhang on 05/11/2013: restructure the code
 */
static void strong_couplings(dCSRmat* A, iCSRmat* S, AMG_param* param)
{
    const SHORT coarse_type = param->coarsening_type;
    const REAL  max_row_sum = param->max_row_sum;
    const REAL  epsilon_str = param->strong_threshold;
    const INT   row = A->row, col = A->col, row1 = row + 1;
    const INT   nnz = A->nnz;

    INT * ia = A->IA, *ja = A->JA;
    REAL* aj = A->val;

    // local variables
    INT  i, j, begin_row, end_row;
    REAL row_scl, row_sum;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (row > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // get the diagonal entry of A: assume all connections are strong
    dvector diag;
    fasp_dcsr_getdiag(0, A, &diag);

    // copy the structure of A to S
    S->row = row;
    S->col = col;
    S->nnz = nnz;
    S->val = NULL;
    S->IA  = (INT*)fasp_mem_calloc(row1, sizeof(INT));
    S->JA  = (INT*)fasp_mem_calloc(nnz, sizeof(INT));
    fasp_iarray_cp(row1, ia, S->IA);
    fasp_iarray_cp(nnz, ja, S->JA);

    if (use_openmp) {

        // This part is still old! Need to be updated. --Chensong 09/18/2016

        INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, row_scl, row_sum, begin_row, \
                                 end_row, j)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, row, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {

                // Compute most negative entry in each row and row sum
                row_scl = row_sum = 0.0;
                begin_row         = ia[i];
                end_row           = ia[i + 1];
                for (j = begin_row; j < end_row; j++) {
                    row_scl = MIN(row_scl, aj[j]);
                    row_sum += aj[j];
                }

                // Find diagonal entries of S and remove them later
                for (j = begin_row; j < end_row; j++) {
                    if (ja[j] == i) {
                        S->JA[j] = -1;
                        break;
                    }
                }

                // Mark entire row as weak couplings if strongly diagonal-dominant
                if (ABS(row_sum) > max_row_sum * ABS(diag.val[i])) {
                    for (j = begin_row; j < end_row; j++) S->JA[j] = -1;
                } else {
                    for (j = begin_row; j < end_row; j++) {
                        // If a_{ij} >= \epsilon_{str} * \min a_{ij}, the connection
                        // j->i is set to be weak; positive entries result in weak
                        // connections
                        if (A->val[j] >= epsilon_str * row_scl) S->JA[j] = -1;
                    }
                }

            } // end for i
        }     // end for myid

    }

    else {

        for (i = 0; i < row; ++i) {

            // Compute row scale and row sum
            row_scl = row_sum = 0.0;
            begin_row         = ia[i];
            end_row           = ia[i + 1];

            for (j = begin_row; j < end_row; j++) {

                // Originally: Not consider positive entries
                // row_sum += aj[j];
                // Now changed to --Chensong 05/17/2013
                row_sum += ABS(aj[j]);

                // Originally: Not consider positive entries
                // row_scl = MAX(row_scl, -aj[j]); // smallest negative
                // Now changed to --Chensong 06/01/2013
                if (ja[j] != i) row_scl = MAX(row_scl, ABS(aj[j])); // largest abs
            }

            // Multiply by the strength threshold
            row_scl *= epsilon_str;
            // printf("row_sum:%e, row_scl:%e \n", row_sum, row_scl);
            // Find diagonal entries of S and remove them later
            for (j = begin_row; j < end_row; j++) {
                if (ja[j] == i) {
                    S->JA[j] = -1;
                    break;
                }
            }

            // Mark entire row as weak couplings if strongly diagonal-dominant
            // Originally: Not consider positive entries
            // if ( ABS(row_sum) > max_row_sum * ABS(diag.val[i]) ) {
            // Now changed to --Chensong 05/17/2013
            /*  printf(" Mark entire row:%e, \n",  (2 - max_row_sum) *
             * ABS(diag.val[i]));*/
            if (row_sum < (2 - max_row_sum) * ABS(diag.val[i])) {

                for (j = begin_row; j < end_row; j++) S->JA[j] = -1;

            } else {

                switch (coarse_type) {

                    case COARSE_RSP: // consider positive off-diag as well
                        for (j = begin_row; j < end_row; j++) {
                            if (ABS(A->val[j]) <= row_scl) S->JA[j] = -1;
                        }
                        break;

                    default: // only consider n-couplings
                        for (j = begin_row; j < end_row; j++) {
                            if (-A->val[j] <= row_scl) S->JA[j] = -1;
                            // printf("j : %d , val[j] : %e", j, A->val[j]);
                        }
                        break;
                }
            }
        } // end for i

    } // end if openmp

    fasp_dvec_free(&diag);
}

/**
 * \fn static INT compress_S (iCSRmat *S)
 *
 * \brief Remove weak couplings from S (marked as -1)
 *
 * \param S        Strong connection matrix (in: with weak, out: without weak)
 *
 * \return Number of cols of P
 *
 * \author Chensong Zhang
 * \date   05/16/2013
 *
 * \note   Compression is done in-place. Used by the C/F splitting schemes!
 */
static INT compress_S(iCSRmat* S)
{
    const INT row = S->row;
    INT*      ia  = S->IA;

    // local variables
    INT index, i, j, begin_row, end_row;

    // compress S: remove weak connections and form strong coupling matrix
    for (index = i = 0; i < row; ++i) {

        begin_row = ia[i];
        end_row   = ia[i + 1];

        ia[i] = index;
        for (j = begin_row; j < end_row; j++) {
            if (S->JA[j] > -1) S->JA[index++] = S->JA[j]; // strong couplings
        }
    }

    S->nnz = S->IA[row] = index;

    if (S->nnz <= 0) {
        return ERROR_UNKNOWN;
    } else {
        return FASP_SUCCESS;
    }
}

/**
 * \fn static void rem_positive_ff (dCSRmat *A, iCSRmat *Stemp, ivector *vertices)
 *
 * \brief Update interpolation support for positive strong couplings
 *
 * \param A            Coefficient matrix, the index starts from zero
 * \param Stemp        Original strong connection matrix
 * \param vertices     Indicator vector for the C/F splitting of the variables
 *
 * \author Chensong Zhang
 * \date   06/07/2013
 */
static void rem_positive_ff(dCSRmat* A, iCSRmat* Stemp, ivector* vertices)
{
    const INT row = A->row;
    INT *     ia = A->IA, *vec = vertices->val;

    REAL row_scl, max_entry;
    INT  i, j, ji, max_index;

    for (i = 0; i < row; ++i) {

        if (vec[i] != FGPT) continue; // skip non F-variables

        row_scl = 0.0;
        for (ji = ia[i]; ji < ia[i + 1]; ++ji) {
            j = A->JA[ji];
            if (j == i) continue;                    // skip diagonal
            row_scl = MAX(row_scl, ABS(A->val[ji])); // max abs entry
        }                                            // end for ji
        row_scl *= 0.75;

        // looking for strong F-F connections
        max_index = -1;
        max_entry = 0.0;
        for (ji = ia[i]; ji < ia[i + 1]; ++ji) {
            j = A->JA[ji];
            if (j == i) continue;         // skip diagonal
            if (vec[j] != FGPT) continue; // skip F-C connections
            if (A->val[ji] > row_scl) {
                Stemp->JA[ji] = j;
                if (A->val[ji] > max_entry) {
                    max_entry = A->val[ji];
                    max_index = j; // max positive entry
                }
            }
        } // end for ji

        // mark max positive entry as C-point
        if (max_index != -1) vec[max_index] = CGPT;

    } // end for i
}

/**
 * \fn static INT cfsplitting_cls (dCSRmat *A, iCSRmat *S, ivector *vertices)
 *
 * \brief Find coarse level variables (classic C/F splitting)
 *
 * \param A            Coefficient matrix, the index starts from zero
 * \param S            Strong connection matrix
 * \param vertices     Indicator vector for the C/F splitting of the variables
 *
 * \return Number of cols of P
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   09/06/2010
 *
 * \note   Coarsening Phase ONE: find coarse level points
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/24/2012: add OMP support.
 * Modified by Chensong Zhang on 07/06/2012: fix a data type bug.
 * Modified by Chensong Zhang on 05/11/2013: restructure the code.
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 12/25/2013: check C1 criterion.
 */
static INT cfsplitting_cls(dCSRmat* A, iCSRmat* S, ivector* vertices)
{
    const INT row = A->row;

    // local variables
    INT  col = 0;
    INT  maxmeas, maxnode, num_left = 0;
    INT  measure, newmeas;
    INT  i, j, k, l;
    INT  myid, mybegin, myend;
    INT* vec   = vertices->val;
    INT* work  = (INT*)fasp_mem_calloc(3 * row, sizeof(INT));
    INT *lists = work, *where = lists + row, *lambda = where + row;

#if RS_C1
    INT  set_empty = 1;
    INT  jkeep     = 0, cnt, index;
    INT  row_end_S, ji, row_end_S_nabor, jj;
    INT* graph_array = lambda;
#else
    INT* ia = A->IA;
#endif

    LinkList LoL_head = NULL, LoL_tail = NULL, list_ptr = NULL;

    SHORT nthreads = 1, use_openmp = FALSE;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

#ifdef _OPENMP
    if (row > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // 0. Compress S and form S_transpose

    col = compress_S(S);

    if (col < 0) goto FINISHED; // compression failed!!!

    iCSRmat ST;
    fasp_icsr_trans(S, &ST);

    // 1. Initialize lambda
    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, row, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) lambda[i] = ST.IA[i + 1] - ST.IA[i];
        }
    } else {
        for (i = 0; i < row; ++i) lambda[i] = ST.IA[i + 1] - ST.IA[i];
    }

    // 2. Before C/F splitting algorithm starts, filter out the variables which
    //    have no connections at all and mark them as special F-variables.
    if (use_openmp) {

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : num_left) private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, row, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
#if RS_C1 // Check C1 criteria or not
                if (S->IA[i + 1] == S->IA[i])
#else
                if ((ia[i + 1] - ia[i]) <= 1)
#endif
                {
                    vec[i]    = ISPT; // set i as an ISOLATED fine node
                    lambda[i] = 0;
                } else {
                    vec[i] = UNPT; // set i as a undecided node
                    num_left++;
                }
            }
        } // end for myid

    }

    else {

        for (i = 0; i < row; ++i) {

#if RS_C1
            if (S->IA[i + 1] == S->IA[i])
#else
            if ((ia[i + 1] - ia[i]) <= 1)
#endif
            {
                vec[i]    = ISPT; // set i as an ISOLATED fine node
                lambda[i] = 0;
            } else {
                vec[i] = UNPT; // set i as a undecided node
                num_left++;
            }
        } // end for i
    }

    // 3. Form linked list for lambda (max to min)
    for (i = 0; i < row; ++i) {

        if (vec[i] == ISPT) continue; // skip isolated variables

        measure = lambda[i];

        if (measure > 0) {
            enter_list(&LoL_head, &LoL_tail, lambda[i], i, lists, where);
        } else {

            if (measure < 0) printf("### WARNING: Negative lambda[%d]!\n", i);

            // Set variables with non-positive measure as F-variables
            vec[i] = FGPT; // no strong connections, set i as fine node
            --num_left;

            // Update lambda and linked list after i->F
            for (k = S->IA[i]; k < S->IA[i + 1]; ++k) {
                j = S->JA[k];
                if (vec[j] == ISPT) continue; // skip isolate variables
                if (j < i) {
                    newmeas = lambda[j];
                    if (newmeas > 0) {
                        remove_node(&LoL_head, &LoL_tail, newmeas, j, lists, where);
                    }
                    newmeas = ++(lambda[j]);
                    enter_list(&LoL_head, &LoL_tail, newmeas, j, lists, where);
                } else {
                    newmeas = ++(lambda[j]);
                }
            }

        } // end if measure

    } // end for i

    // 4. Main loop
    while (num_left > 0) {

        // pick $i\in U$ with $\max\lambda_i: C:=C\cup\{i\}, U:=U\\{i\}$
        maxnode = LoL_head->head;
        maxmeas = lambda[maxnode];
        if (maxmeas == 0) printf("### WARNING: Head of the list has measure 0!\n");

        vec[maxnode]    = CGPT; // set maxnode as coarse node
        lambda[maxnode] = 0;
        --num_left;
        remove_node(&LoL_head, &LoL_tail, maxmeas, maxnode, lists, where);
        col++;

        // for all $j\in S_i^T\cap U: F:=F\cup\{j\}, U:=U\backslash\{j\}$
        for (i = ST.IA[maxnode]; i < ST.IA[maxnode + 1]; ++i) {

            j = ST.JA[i];

            if (vec[j] != UNPT) continue; // skip decided variables

            vec[j] = FGPT; // set j as fine node
            remove_node(&LoL_head, &LoL_tail, lambda[j], j, lists, where);
            --num_left;

            // Update lambda and linked list after j->F
            for (l = S->IA[j]; l < S->IA[j + 1]; l++) {
                k = S->JA[l];
                if (vec[k] == UNPT) { // k is unknown
                    remove_node(&LoL_head, &LoL_tail, lambda[k], k, lists, where);
                    newmeas = ++(lambda[k]);
                    enter_list(&LoL_head, &LoL_tail, newmeas, k, lists, where);
                }
            }

        } // end for i

        // Update lambda and linked list after maxnode->C
        for (i = S->IA[maxnode]; i < S->IA[maxnode + 1]; ++i) {

            j = S->JA[i];

            if (vec[j] != UNPT) continue; // skip decided variables

            measure = lambda[j];
            remove_node(&LoL_head, &LoL_tail, measure, j, lists, where);
            lambda[j] = --measure;

            if (measure > 0) {
                enter_list(&LoL_head, &LoL_tail, measure, j, lists, where);
            } else { // j is the only point left, set as fine variable
                vec[j] = FGPT;
                --num_left;

                // Update lambda and linked list after j->F
                for (l = S->IA[j]; l < S->IA[j + 1]; l++) {
                    k = S->JA[l];
                    if (vec[k] == UNPT) { // k is unknown
                        remove_node(&LoL_head, &LoL_tail, lambda[k], k, lists, where);
                        newmeas = ++(lambda[k]);
                        enter_list(&LoL_head, &LoL_tail, newmeas, k, lists, where);
                    }
                } // end for l
            }     // end if

        } // end for

    } // end while

#if RS_C1

    // C/F splitting of RS coarsening check C1 Criterion
    fasp_iarray_set(row, graph_array, -1);
    for (i = 0; i < row; i++) {
        if (vec[i] == FGPT) {
            row_end_S = S->IA[i + 1];
            for (ji = S->IA[i]; ji < row_end_S; ji++) {
                j = S->JA[ji];
                if (vec[j] == CGPT) {
                    graph_array[j] = i;
                }
            }
            cnt = 0;
            for (ji = S->IA[i]; ji < row_end_S; ji++) {
                j = S->JA[ji];
                if (vec[j] == FGPT) {
                    set_empty       = 1;
                    row_end_S_nabor = S->IA[j + 1];
                    for (jj = S->IA[j]; jj < row_end_S_nabor; jj++) {
                        index = S->JA[jj];
                        if (graph_array[index] == i) {
                            set_empty = 0;
                            break;
                        }
                    }
                    if (set_empty) {
                        if (cnt == 0) {
                            vec[j] = CGPT;
                            col++;
                            graph_array[j] = i;
                            jkeep          = j;
                            cnt            = 1;
                        } else {
                            vec[i]     = CGPT;
                            vec[jkeep] = FGPT;
                            break;
                        }
                    }
                }
            }
        }
    }

#endif

    fasp_icsr_free(&ST);

    if (LoL_head) {
        list_ptr            = LoL_head;
        LoL_head->prev_node = NULL;
        LoL_head->next_node = NULL;
        LoL_head            = list_ptr->next_node;
        fasp_mem_free(list_ptr);
        list_ptr = NULL;
    }

FINISHED:
    fasp_mem_free(work);
    work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return col;
}

/**
 * \fn static INT cfsplitting_clsp (dCSRmat *A, iCSRmat *S, ivector *vertices)
 *
 * \brief Find coarse level variables (C/F splitting with positive connections)
 *
 * \param A            Coefficient matrix, the index starts from zero
 * \param S            Strong connection matrix
 * \param vertices     Indicator vector for the C/F splitting of the variables
 *
 * \return Number of cols of P
 *
 * \author Chensong Zhang
 * \date   05/16/2013
 *
 * \note   Compared with cfsplitting_cls, cfsplitting_clsp has an extra step for
 *         checking strong positive couplings and pick some of them as C.
 *
 * Modified by Chensong Zhang on 06/07/2013: restructure the code
 */
static INT cfsplitting_clsp(dCSRmat* A, iCSRmat* S, ivector* vertices)
{
    const INT row = A->row;

    // local variables
    INT col = 0;
    INT maxmeas, maxnode, num_left = 0;
    INT measure, newmeas;
    INT i, j, k, l;
    INT myid, mybegin, myend;

    INT *ia = A->IA, *vec = vertices->val;
    INT* work  = (INT*)fasp_mem_calloc(3 * row, sizeof(INT));
    INT *lists = work, *where = lists + row, *lambda = where + row;

    LinkList LoL_head = NULL, LoL_tail = NULL, list_ptr = NULL;

    SHORT nthreads = 1, use_openmp = FALSE;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

#ifdef _OPENMP
    if (row > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // 0. Compress S and form S_transpose (not complete, just IA and JA)
    iCSRmat Stemp;
    Stemp.row = S->row;
    Stemp.col = S->col;
    Stemp.nnz = S->nnz;
    Stemp.IA  = (INT*)fasp_mem_calloc(S->row + 1, sizeof(INT));
    fasp_iarray_cp(S->row + 1, S->IA, Stemp.IA);
    Stemp.JA = (INT*)fasp_mem_calloc(S->nnz, sizeof(INT));
    fasp_iarray_cp(S->nnz, S->JA, Stemp.JA);

    if (compress_S(S) < 0) goto FINISHED; // compression failed!!!

    iCSRmat ST;
    fasp_icsr_trans(S, &ST);

    // 1. Initialize lambda
    if (use_openmp) {
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, row, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) lambda[i] = ST.IA[i + 1] - ST.IA[i];
        }
    } else {
        for (i = 0; i < row; ++i) lambda[i] = ST.IA[i + 1] - ST.IA[i];
    }

    // 2. Before C/F splitting algorithm starts, filter out the variables which
    //    have no connections at all and mark them as special F-variables.
    if (use_openmp) {

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : num_left) private(myid, mybegin, myend, i)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, row, &mybegin, &myend);
            for (i = mybegin; i < myend; i++) {
                if ((ia[i + 1] - ia[i]) <= 1) {
                    vec[i]    = ISPT; // set i as an ISOLATED fine node
                    lambda[i] = 0;
                } else {
                    vec[i] = UNPT; // set i as a undecided node
                    num_left++;
                }
            }
        } // end for myid

    } else {

        for (i = 0; i < row; ++i) {
            if ((ia[i + 1] - ia[i]) <= 1) {
                vec[i]    = ISPT; // set i as an ISOLATED fine node
                lambda[i] = 0;
            } else {
                vec[i] = UNPT; // set i as a undecided node
                num_left++;
            }
        } // end for i
    }

    // 3. Form linked list for lambda (max to min)
    for (i = 0; i < row; ++i) {

        if (vec[i] == ISPT) continue; // skip isolated variables

        measure = lambda[i];

        if (measure > 0) {
            enter_list(&LoL_head, &LoL_tail, lambda[i], i, lists, where);
        } else {

            if (measure < 0) printf("### WARNING: Negative lambda[%d]!\n", i);

            // Set variables with non-positive measure as F-variables
            vec[i] = FGPT; // no strong connections, set i as fine node
            --num_left;

            // Update lambda and linked list after i->F
            for (k = S->IA[i]; k < S->IA[i + 1]; ++k) {

                j = S->JA[k];
                if (vec[j] == ISPT) continue; // skip isolate variables

                if (j < i) { // only look at the previous points!!
                    newmeas = lambda[j];
                    if (newmeas > 0) {
                        remove_node(&LoL_head, &LoL_tail, newmeas, j, lists, where);
                    }
                    newmeas = ++(lambda[j]);
                    enter_list(&LoL_head, &LoL_tail, newmeas, j, lists, where);
                } else { // will be checked later on
                    newmeas = ++(lambda[j]);
                } // end if

            } // end for k

        } // end if measure

    } // end for i

    // 4. Main loop
    while (num_left > 0) {

        // pick $i\in U$ with $\max\lambda_i: C:=C\cup\{i\}, U:=U\\{i\}$
        maxnode = LoL_head->head;
        maxmeas = lambda[maxnode];
        if (maxmeas == 0) printf("### WARNING: Head of the list has measure 0!\n");

        vec[maxnode]    = CGPT; // set maxnode as coarse node
        lambda[maxnode] = 0;
        --num_left;
        remove_node(&LoL_head, &LoL_tail, maxmeas, maxnode, lists, where);
        col++;

        // for all $j\in S_i^T\cap U: F:=F\cup\{j\}, U:=U\backslash\{j\}$
        for (i = ST.IA[maxnode]; i < ST.IA[maxnode + 1]; ++i) {

            j = ST.JA[i];

            if (vec[j] != UNPT) continue; // skip decided variables

            vec[j] = FGPT; // set j as fine node
            remove_node(&LoL_head, &LoL_tail, lambda[j], j, lists, where);
            --num_left;

            // Update lambda and linked list after j->F
            for (l = S->IA[j]; l < S->IA[j + 1]; l++) {
                k = S->JA[l];
                if (vec[k] == UNPT) { // k is unknown
                    remove_node(&LoL_head, &LoL_tail, lambda[k], k, lists, where);
                    newmeas = ++(lambda[k]);
                    enter_list(&LoL_head, &LoL_tail, newmeas, k, lists, where);
                }
            }

        } // end for i

        // Update lambda and linked list after maxnode->C
        for (i = S->IA[maxnode]; i < S->IA[maxnode + 1]; ++i) {

            j = S->JA[i];

            if (vec[j] != UNPT) continue; // skip decided variables

            measure = lambda[j];
            remove_node(&LoL_head, &LoL_tail, measure, j, lists, where);
            lambda[j] = --measure;

            if (measure > 0) {
                enter_list(&LoL_head, &LoL_tail, measure, j, lists, where);
            } else { // j is the only point left, set as fine variable
                vec[j] = FGPT;
                --num_left;

                // Update lambda and linked list after j->F
                for (l = S->IA[j]; l < S->IA[j + 1]; l++) {
                    k = S->JA[l];
                    if (vec[k] == UNPT) { // k is unknown
                        remove_node(&LoL_head, &LoL_tail, lambda[k], k, lists, where);
                        newmeas = ++(lambda[k]);
                        enter_list(&LoL_head, &LoL_tail, newmeas, k, lists, where);
                    }
                } // end for l
            }     // end if

        } // end for

    } // end while

    fasp_icsr_free(&ST);

    if (LoL_head) {
        list_ptr            = LoL_head;
        LoL_head->prev_node = NULL;
        LoL_head->next_node = NULL;
        LoL_head            = list_ptr->next_node;
        fasp_mem_free(list_ptr);
        list_ptr = NULL;
    }

    // Enforce F-C connections. Adding this step helps for the ExxonMobil test
    // problems! Need more tests though --Chensong 06/08/2013
    // col = clean_ff_couplings(S, vertices, row, col);

    rem_positive_ff(A, &Stemp, vertices);

    if (compress_S(&Stemp) < 0) goto FINISHED; // compression failed!!!

    S->row = Stemp.row;
    S->col = Stemp.col;
    S->nnz = Stemp.nnz;

    fasp_mem_free(S->IA);
    S->IA = Stemp.IA;
    fasp_mem_free(S->JA);
    S->JA = Stemp.JA;

FINISHED:
    fasp_mem_free(work);
    work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return col;
}

/**
 * \fn static void strong_couplings_agg1 (dCSRmat *A, iCSRmat *S, iCSRmat *Sh,
 *                                        ivector *vertices, ivector *CGPT_index,
 *                                        ivector *CGPT_rindex)
 *
 * \brief Generate the set of all strong negative or absolute couplings using
 *        aggressive coarsening A1
 *
 * \param A            Coefficient matrix, the index starts from zero
 * \param S            Strong connection matrix
 * \param Sh           Strong couplings matrix between coarse grid points
 * \param vertices     Type of variables--C/F splitting
 * \param CGPT_index   Index of CGPT from CGPT to all points
 * \param CGPT_rindex  Index of CGPT from all points to CGPT
 *
 * \author Kai Yang, Xiaozhe Hu
 * \date   09/06/2010
 *
 * Modified by Chensong Zhang on 05/13/2013: restructure the code
 */
static void strong_couplings_agg1(dCSRmat* A,
                                  iCSRmat* S,
                                  iCSRmat* Sh,
                                  ivector* vertices,
                                  ivector* CGPT_index,
                                  ivector* CGPT_rindex)
{
    const INT row = A->row;

    // local variables
    INT  i, j, k;
    INT  num_c, count, ci, cj, ck, fj, cck;
    INT *cp_index, *cp_rindex, *visited;
    INT* vec = vertices->val;

    // count the number of coarse grid points
    for (num_c = i = 0; i < row; i++) {
        if (vec[i] == CGPT) num_c++;
    }

    // for the reverse indexing of coarse grid points
    fasp_ivec_alloc(row, CGPT_rindex);
    cp_rindex = CGPT_rindex->val;

    // generate coarse grid point index
    fasp_ivec_alloc(num_c, CGPT_index);
    cp_index = CGPT_index->val;
    for (j = i = 0; i < row; i++) {
        if (vec[i] == CGPT) {
            cp_index[j]  = i;
            cp_rindex[i] = j;
            j++;
        }
    }

    // allocate space for Sh
    Sh->row = Sh->col = num_c;
    Sh->val = Sh->JA = NULL;
    Sh->IA           = (INT*)fasp_mem_calloc(Sh->row + 1, sizeof(INT));

    // record the number of times some coarse point is visited
    visited = (INT*)fasp_mem_calloc(num_c, sizeof(INT));
    fasp_iarray_set(num_c, visited, -1);

    /**********************************************/
    /* step 1: Find first the structure IA of Sh  */
    /**********************************************/

    Sh->IA[0] = 0;

    for (ci = 0; ci < Sh->row; ci++) {

        i = cp_index[ci]; // find the index of the ci-th coarse grid point

        // number of coarse point that i is strongly connected to w.r.t. S(p,2)
        count = 0;

        // visit all the fine neighbors that ci is strongly connected to
        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

            fj = S->JA[j];

            if (vec[fj] == CGPT && fj != i) {
                cj = cp_rindex[fj];
                if (visited[cj] != ci) {
                    visited[cj] = ci; // mark as strongly connected from ci
                    count++;
                }

            }

            else if (vec[fj] == FGPT) { // fine grid point,

                // find all the coarse neighbors that fj is strongly connected to
                for (k = S->IA[fj]; k < S->IA[fj + 1]; k++) {
                    ck = S->JA[k];
                    if (vec[ck] == CGPT && ck != i) { // it is a coarse grid point
                        if (cp_rindex[ck] >= num_c) {
                            printf("### ERROR: ck=%d, num_c=%d, out of bound!\n", ck,
                                   num_c);
                            fasp_chkerr(ERROR_AMG_COARSEING, __FUNCTION__);
                        }
                        cck = cp_rindex[ck];

                        if (visited[cck] != ci) {
                            visited[cck] = ci; // mark as strongly connected from ci
                            count++;
                        }
                    } // end if
                }     // end for k

            } // end if

        } // end for j

        Sh->IA[ci + 1] = Sh->IA[ci] + count;

    } // end for i

    /*************************/
    /* step 2: Find JA of Sh */
    /*************************/

    fasp_iarray_set(num_c, visited, -1); // reset visited

    Sh->nnz = Sh->IA[Sh->row];
    Sh->JA  = (INT*)fasp_mem_calloc(Sh->nnz, sizeof(INT));

    for (ci = 0; ci < Sh->row; ci++) {

        i     = cp_index[ci]; // find the index of the i-th coarse grid point
        count = Sh->IA[ci];   // count for coarse points

        // visit all the fine neighbors that ci is strongly connected to
        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

            fj = S->JA[j];

            if (vec[fj] == CGPT && fj != i) {
                cj = cp_rindex[fj];
                if (visited[cj] != ci) { // not visited yet
                    visited[cj]   = ci;
                    Sh->JA[count] = cj;
                    count++;
                }
            } else if (vec[fj] == FGPT) { // fine grid point,
                // find all the coarse neighbors that fj is strongly connected to
                for (k = S->IA[fj]; k < S->IA[fj + 1]; k++) {
                    ck = S->JA[k];
                    if (vec[ck] == CGPT && ck != i) { // coarse grid point
                        cck = cp_rindex[ck];
                        if (visited[cck] != ci) { // not visited yet
                            visited[cck]  = ci;
                            Sh->JA[count] = cck;
                            count++;
                        }
                    } // end if
                }     // end for k
            }         // end if

        } // end for j

        if (count != Sh->IA[ci + 1]) {
            printf("### WARNING: Inconsistent numbers of nonzeros!\n ");
        }

    } // end for ci

    fasp_mem_free(visited);
    visited = NULL;
}

/**
 * \fn static void strong_couplings_agg2 (dCSRmat *A, iCSRmat *S, iCSRmat *Sh,
 *                                        ivector *vertices, ivector *CGPT_index,
 *                                        ivector *CGPT_rindex)
 *
 * \brief Generate the set of all strong negative or absolute couplings using
 *        aggressive coarsening A2
 *
 * \param A            Coefficient matrix, the index starts from zero
 * \param S            Strong connection matrix
 * \param Sh           Strong couplings matrix between coarse grid points
 * \param vertices     Type of variables--C/F splitting
 * \param CGPT_index   Index of CGPT from CGPT to all points
 * \param CGPT_rindex  Index of CGPT from all points to CGPT
 *
 * \author Xiaozhe Hu
 * \date   04/24/2013
 *
 * \note   The difference between strong_couplings_agg1 and strong_couplings_agg2
 *         is that strong_couplings_agg1 uses one path to determine strongly coupled
 *         C points while strong_couplings_agg2 uses two paths to determine strongly
 *         coupled C points. Usually strong_couplings_agg1 gives more aggressive
 *         coarsening!
 *
 * Modified by Chensong Zhang on 05/13/2013: restructure the code
 */
static void strong_couplings_agg2(dCSRmat* A,
                                  iCSRmat* S,
                                  iCSRmat* Sh,
                                  ivector* vertices,
                                  ivector* CGPT_index,
                                  ivector* CGPT_rindex)
{
    const INT row = A->row;

    // local variables
    INT  i, j, k;
    INT  num_c, count, ci, cj, ck, fj, cck;
    INT *cp_index, *cp_rindex, *visited;
    INT* vec = vertices->val;

    // count the number of coarse grid points
    for (num_c = i = 0; i < row; i++) {
        if (vec[i] == CGPT) num_c++;
    }

    // for the reverse indexing of coarse grid points
    fasp_ivec_alloc(row, CGPT_rindex);
    cp_rindex = CGPT_rindex->val;

    // generate coarse grid point index
    fasp_ivec_alloc(num_c, CGPT_index);
    cp_index = CGPT_index->val;
    for (j = i = 0; i < row; i++) {
        if (vec[i] == CGPT) {
            cp_index[j]  = i;
            cp_rindex[i] = j;
            j++;
        }
    }

    // allocate space for Sh
    Sh->row = Sh->col = num_c;
    Sh->val = Sh->JA = NULL;
    Sh->IA           = (INT*)fasp_mem_calloc(Sh->row + 1, sizeof(INT));

    // record the number of times some coarse point is visited
    visited = (INT*)fasp_mem_calloc(num_c, sizeof(INT));
    memset(visited, 0, sizeof(INT) * num_c);

    /**********************************************/
    /* step 1: Find first the structure IA of Sh  */
    /**********************************************/

    Sh->IA[0] = 0;

    for (ci = 0; ci < Sh->row; ci++) {

        i = cp_index[ci]; // find the index of the ci-th coarse grid point

        // number of coarse point that i is strongly connected to w.r.t. S(p,2)
        count = 0;

        // visit all the fine neighbors that ci is strongly connected to
        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

            fj = S->JA[j];

            if (vec[fj] == CGPT && fj != i) {
                cj = cp_rindex[fj];
                if (visited[cj] != ci + 1) { // not visited yet
                    visited[cj] = ci + 1;    // mark as strongly connected from ci
                    count++;
                }
            }

            else if (vec[fj] == FGPT) { // fine grid point

                // find all the coarse neighbors that fj is strongly connected to
                for (k = S->IA[fj]; k < S->IA[fj + 1]; k++) {

                    ck = S->JA[k];

                    if (vec[ck] == CGPT && ck != i) { // coarse grid point
                        if (cp_rindex[ck] >= num_c) {
                            printf("### ERROR: ck=%d, num_c=%d, out of bound!\n", ck,
                                   num_c);
                            fasp_chkerr(ERROR_AMG_COARSEING, __FUNCTION__);
                        }
                        cck = cp_rindex[ck];

                        if (visited[cck] == ci + 1) {
                            // visited already!
                        } else if (visited[cck] == -ci - 1) {
                            visited[cck] = ci + 1; // mark as strongly connected from ci
                            count++;
                        } else {
                            visited[cck] = -ci - 1; // mark as visited
                        }

                    } // end if vec[ck]

                } // end for k

            } // end if vec[fj]

        } // end for j

        Sh->IA[ci + 1] = Sh->IA[ci] + count;

    } // end for i

    /*************************/
    /* step 2: Find JA of Sh */
    /*************************/

    memset(visited, 0, sizeof(INT) * num_c); // reset visited

    Sh->nnz = Sh->IA[Sh->row];
    Sh->JA  = (INT*)fasp_mem_calloc(Sh->nnz, sizeof(INT));

    for (ci = 0; ci < Sh->row; ci++) {

        i     = cp_index[ci]; // find the index of the i-th coarse grid point
        count = Sh->IA[ci];   // count for coarse points

        // visit all the fine neighbors that ci is strongly connected to
        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

            fj = S->JA[j];

            if (vec[fj] == CGPT && fj != i) {
                cj = cp_rindex[fj];
                if (visited[cj] != ci + 1) { // not visited yet
                    visited[cj]   = ci + 1;
                    Sh->JA[count] = cj;
                    count++;
                }
            }

            else if (vec[fj] == FGPT) { // fine grid point

                // find all the coarse neighbors that fj is strongly connected to
                for (k = S->IA[fj]; k < S->IA[fj + 1]; k++) {

                    ck = S->JA[k];

                    if (vec[ck] == CGPT && ck != i) { // coarse grid point
                        cck = cp_rindex[ck];
                        if (visited[cck] == ci + 1) {
                            // visited before
                        } else if (visited[cck] == -ci - 1) {
                            visited[cck]  = ci + 1;
                            Sh->JA[count] = cck;
                            count++;
                        } else {
                            visited[cck] = -ci - 1;
                        }
                    } // end if vec[ck]

                } // end for k

            } // end if vec[fj]

        } // end for j

        if (count != Sh->IA[ci + 1]) {
            printf("### WARNING: Inconsistent numbers of nonzeros!\n ");
        }

    } // end for ci

    fasp_mem_free(visited);
    visited = NULL;
}

/**
 * \fn static INT cfsplitting_agg (dCSRmat *A, iCSRmat *S, ivector *vertices,
 *                                 INT aggressive_path)
 *
 * \brief Find coarse level variables (C/F splitting): aggressive
 *
 * \param A                Coefficient matrix, the index starts from zero
 * \param S                Strong connection matrix
 * \param vertices         Indicator vector for the C/F splitting of the variables
 * \param aggressive_path  Aggressive path
 *
 * \return Number of cols of P
 *
 * \author Kai Yang, Xiaozhe Hu
 * \date   09/06/2010
 *
 * Modified by Chensong Zhang on 07/05/2012: Fix a data type bug
 * Modified by Chunsheng Feng, Zheng Li on 10/13/2012
 * Modified by Xiaozhe Hu on 04/24/2013: modify aggressive coarsening
 * Modified by Chensong Zhang on 05/13/2013: restructure the code
 */
static INT
cfsplitting_agg(dCSRmat* A, iCSRmat* S, ivector* vertices, INT aggressive_path)
{
    const INT row = A->row;
    INT       col = 0; // initialize col(P): returning output

    // local variables
    INT * vec = vertices->val, *cp_index;
    INT   maxmeas, maxnode, num_left = 0;
    INT   measure, newmeas;
    INT   i, j, k, l, m, ci, cj, ck, cl, num_c;
    SHORT IS_CNEIGH;

    INT* work  = (INT*)fasp_mem_calloc(3 * row, sizeof(INT));
    INT *lists = work, *where = lists + row, *lambda = where + row;

    ivector  CGPT_index, CGPT_rindex;
    LinkList LoL_head = NULL, LoL_tail = NULL, list_ptr = NULL;

    // Sh is for the strong coupling matrix between temporary CGPTs
    // ShT is the transpose of Sh
    // Snew is for combining the information from S and Sh
    iCSRmat ST, Sh, ShT;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    /************************************************************/
    /* Coarsening Phase ONE: find temporary coarse level points */
    /************************************************************/

    num_c = cfsplitting_cls(A, S, vertices);
    fasp_icsr_trans(S, &ST);

    /************************************************************/
    /* Coarsening Phase TWO: find real coarse level points      */
    /************************************************************/

    // find Sh, the strong coupling between coarse grid points S(path,2)
    if (aggressive_path < 2)
        strong_couplings_agg1(A, S, &Sh, vertices, &CGPT_index, &CGPT_rindex);
    else
        strong_couplings_agg2(A, S, &Sh, vertices, &CGPT_index, &CGPT_rindex);

    fasp_icsr_trans(&Sh, &ShT);

    CGPT_index.row  = num_c;
    CGPT_rindex.row = row;
    cp_index        = CGPT_index.val;

    // 1. Initialize lambda
#ifdef _OPENMP
#pragma omp parallel for if (num_c > OPENMP_HOLDS)
#endif
    for (ci = 0; ci < num_c; ++ci) lambda[ci] = ShT.IA[ci + 1] - ShT.IA[ci];

    // 2. Form linked list for lambda (max to min)
    for (ci = 0; ci < num_c; ++ci) {

        i       = cp_index[ci];
        measure = lambda[ci];

        if (vec[i] == ISPT) continue; // skip isolated points

        if (measure > 0) {
            enter_list(&LoL_head, &LoL_tail, lambda[ci], ci, lists, where);
            num_left++;
        } else {
            if (measure < 0) printf("### WARNING: Negative lambda[%d]!\n", i);

            vec[i] = FGPT; // set i as fine node

            // update the lambda value in the CGPT neighbor of i
            for (ck = Sh.IA[ci]; ck < Sh.IA[ci + 1]; ++ck) {

                cj = Sh.JA[ck];
                j  = cp_index[cj];

                if (vec[j] == ISPT) continue;

                if (cj < ci) {
                    newmeas = lambda[cj];
                    if (newmeas > 0) {
                        remove_node(&LoL_head, &LoL_tail, newmeas, cj, lists, where);
                        num_left--;
                    }
                    newmeas = ++(lambda[cj]);
                    enter_list(&LoL_head, &LoL_tail, newmeas, cj, lists, where);
                    num_left++;
                } else {
                    newmeas = ++(lambda[cj]);
                } // end if cj<ci

            } // end for ck

        } // end if

    } // end for ci

    // 3. Main loop
    while (num_left > 0) {

        // pick $i\in U$ with $\max\lambda_i: C:=C\cup\{i\}, U:=U\\{i\}$
        maxnode = LoL_head->head;
        maxmeas = lambda[maxnode];
        if (maxmeas == 0) printf("### WARNING: Head of the list has measure 0!\n");

        // mark maxnode as real coarse node, labelled as number 3
        vec[cp_index[maxnode]] = 3;
        --num_left;
        remove_node(&LoL_head, &LoL_tail, maxmeas, maxnode, lists, where);
        lambda[maxnode] = 0;
        col++; // count for the real coarse node after aggressive coarsening

        // for all $j\in S_i^T\cap U: F:=F\cup\{j\}, U:=U\backslash\{j\}$
        for (ci = ShT.IA[maxnode]; ci < ShT.IA[maxnode + 1]; ++ci) {

            cj = ShT.JA[ci];
            j  = cp_index[cj];

            if (vec[j] != CGPT) continue; // skip if j is not C-point

            vec[j] = 4; // set j as 4--fake CGPT
            remove_node(&LoL_head, &LoL_tail, lambda[cj], cj, lists, where);
            --num_left;

            // update the measure for neighboring points
            for (cl = Sh.IA[cj]; cl < Sh.IA[cj + 1]; cl++) {
                ck = Sh.JA[cl];
                k  = cp_index[ck];
                if (vec[k] == CGPT) { // k is temporary CGPT
                    remove_node(&LoL_head, &LoL_tail, lambda[ck], ck, lists, where);
                    newmeas = ++(lambda[ck]);
                    enter_list(&LoL_head, &LoL_tail, newmeas, ck, lists, where);
                }
            }

        } // end for ci

        // Update lambda and linked list after maxnode->C
        for (ci = Sh.IA[maxnode]; ci < Sh.IA[maxnode + 1]; ++ci) {

            cj = Sh.JA[ci];
            j  = cp_index[cj];

            if (vec[j] != CGPT) continue; // skip if j is not C-point

            measure = lambda[cj];
            remove_node(&LoL_head, &LoL_tail, measure, cj, lists, where);
            lambda[cj] = --measure;

            if (measure > 0) {
                enter_list(&LoL_head, &LoL_tail, measure, cj, lists, where);
            } else {
                vec[j] = 4; // set j as fake CGPT variable
                --num_left;
                for (cl = Sh.IA[cj]; cl < Sh.IA[cj + 1]; cl++) {
                    ck = Sh.JA[cl];
                    k  = cp_index[ck];
                    if (vec[k] == CGPT) { // k is temporary CGPT
                        remove_node(&LoL_head, &LoL_tail, lambda[ck], ck, lists, where);
                        newmeas = ++(lambda[ck]);
                        enter_list(&LoL_head, &LoL_tail, newmeas, ck, lists, where);
                    }
                } // end for l
            }     // end if

        } // end for

    } // while

    // 4. reorganize the variable type: mark temporary CGPT--1 and fake CGPT--4 as
    //    FGPT; mark real CGPT--3 to be CGPT
#ifdef _OPENMP
#pragma omp parallel for if (row > OPENMP_HOLDS)
#endif
    for (i = 0; i < row; i++) {
        if (vec[i] == CGPT || vec[i] == 4) vec[i] = FGPT;
    }

#ifdef _OPENMP
#pragma omp parallel for if (row > OPENMP_HOLDS)
#endif
    for (i = 0; i < row; i++) {
        if (vec[i] == 3) vec[i] = CGPT;
    }

    /************************************************************/
    /* Coarsening Phase THREE: all the FGPTs which have no CGPT */
    /* neighbors within distance 2. Change them into CGPT such  */
    /* that the standard interpolation works!                   */
    /************************************************************/

    for (i = 0; i < row; i++) {

        if (vec[i] != FGPT) continue;

        IS_CNEIGH = FALSE; // whether there exist CGPT neighbors within distance of 2

        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

            if (IS_CNEIGH) break;

            k = S->JA[j];

            if (vec[k] == CGPT) {
                IS_CNEIGH = TRUE;
            } else if (vec[k] == FGPT) {
                for (l = S->IA[k]; l < S->IA[k + 1]; l++) {
                    m = S->JA[l];
                    if (vec[m] == CGPT) {
                        IS_CNEIGH = TRUE;
                        break;
                    }
                } // end for l
            }

        } // end for j

        // no CGPT neighbors in distance <= 2, mark i as CGPT
        if (!IS_CNEIGH) {
            vec[i] = CGPT;
            col++;
        }

    } // end for i

    if (LoL_head) {
        list_ptr            = LoL_head;
        LoL_head->prev_node = NULL;
        LoL_head->next_node = NULL;
        LoL_head            = list_ptr->next_node;
        fasp_mem_free(list_ptr);
        list_ptr = NULL;
    }

    fasp_ivec_free(&CGPT_index);
    fasp_ivec_free(&CGPT_rindex);
    fasp_icsr_free(&Sh);
    fasp_icsr_free(&ST);
    fasp_icsr_free(&ShT);
    fasp_mem_free(work);
    work = NULL;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return col;
}

/**
 * \fn static INT clean_ff_couplings (iCSRmat *S, ivector *vertices,
 *                                    INT row, INT col)
 *
 * \brief Clear some of the FF connections
 *
 * \param S            Strong connection matrix
 * \param vertices     Indicator vector for the C/F splitting of the variables
 * \param row          Number of rows of P
 * \param col          Number of columns of P
 *
 * \return Number of cols of P
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   09/06/2010
 *
 * \note   Coarsening Phase TWO: remove some F-F connections by F->C. Need to be
 *         applied in direct and energy-min interpolations to make sure C-neighbors
 *         exist for each F-point!
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/24/2012: add OMP support
 * Modified by Chensong Zhang on 05/12/2013: restructure the code
 */
static INT clean_ff_couplings(iCSRmat* S, ivector* vertices, INT row, INT col)
{
    // local variables
    INT* vec       = vertices->val;
    INT* cindex    = (INT*)fasp_mem_calloc(row, sizeof(INT));
    INT  set_empty = TRUE, C_i_nonempty = FALSE;
    INT  ci_tilde = -1, ci_tilde_mark = -1;
    INT  ji, jj, i, j, index;

    fasp_iarray_set(row, cindex, -1);

    for (i = 0; i < row; ++i) {

        if (vec[i] != FGPT) continue; // skip non F-variables

        for (ji = S->IA[i]; ji < S->IA[i + 1]; ++ji) {
            j = S->JA[ji];
            if (vec[j] == CGPT)
                cindex[j] = i; // mark C-neighbors
            else
                cindex[j] = -1; // reset cindex --Chensong 06/02/2013
        }

        if (ci_tilde_mark != i) ci_tilde = -1; //???

        for (ji = S->IA[i]; ji < S->IA[i + 1]; ++ji) {

            j = S->JA[ji];

            if (vec[j] != FGPT) continue; // skip non F-variables

            // check whether there is a C-connection
            set_empty = TRUE;
            for (jj = S->IA[j]; jj < S->IA[j + 1]; ++jj) {
                index = S->JA[jj];
                if (cindex[index] == i) {
                    set_empty = FALSE;
                    break;
                }
            } // end for jj

            // change the point i (if only F-F exists) to C
            if (set_empty) {
                if (C_i_nonempty) {
                    vec[i] = CGPT;
                    col++;
                    if (ci_tilde > -1) {
                        vec[ci_tilde] = FGPT;
                        col--;
                        ci_tilde = -1;
                    }
                    C_i_nonempty = FALSE;
                    break;
                } else { // temporary set j->C and roll back
                    vec[j] = CGPT;
                    col++;
                    ci_tilde      = j;
                    ci_tilde_mark = i;
                    C_i_nonempty  = TRUE;
                    i--; // roll back to check i-point again
                    break;
                } // end if C_i_nonempty
            }     // end if set_empty

        } // end for ji

    } // end for i

    fasp_mem_free(cindex);
    cindex = NULL;

    return col;
}

REAL rabs(REAL x) { return (x > 0) ? x : -x; }
/**
 * @brief Generate sparsity pattern of prolongation for reduction-based amg
 * interpolation
 *
 * @param theta D_ii = (2 - 1/theta) * A_ii, |A_ii| > theta * sum(|A_ij|) for j as
 * F-points
 * @note P generated here uses the same index for columns as A
 * @author Yan Xie
 * @date   2022/12/06
 *
 * use all coarse points as interpolation points
 */
static void form_P_pattern_rdc(
    dCSRmat* P, dCSRmat* A, double* theta, ivector* vertices, INT row, INT col)
{
    // local variables
    INT* vec = vertices->val;
    INT  i, j, k, index = 0;

    // Initialize P matrix
    P->row   = row;
    P->col   = col;
    P->IA    = (INT*)fasp_mem_calloc(row + 1, sizeof(INT));
    P->IA[0] = 0;

    /* Generate sparsity pattern of P & calculate theta */
    // firt pass: P->IA & theta
    for (i = 0; i < row; ++i) {
        if (vec[i] == CGPT) { // identity interpolation for C-points
            P->IA[i + 1] = P->IA[i] + 1;
            theta[i]     = 1000000.0;
        } else { // D_FF^-1 * A_FC for F-points
            P->IA[i + 1]   = P->IA[i];
            double sum     = 0.0;
            INT    diagptr = -1;
            for (k = A->IA[i]; k < A->IA[i + 1]; ++k) {
                j = A->JA[k];
                if (vec[j] == CGPT)
                    P->IA[i + 1]++;
                else {
                    sum += rabs(A->val[k]);
                }
                if (j == i) diagptr = k;
            }
            if (diagptr > -1)
                theta[i] = rabs(A->val[diagptr]) / sum;
            else {
                printf("### ERROR: Diagonal element is zero! [%s:%d]\n", __FUNCTION__,
                       __LINE__);
                theta[i] = 1000000.0;
            }
            if (theta[i] < 0.5) {
                printf("WARNING: theta[%d] = %f < 0.5! not diagonal dominant!\n", i,
                       theta[i]);
                // view the matrix row entries
                // printf("### DEBUG: row %d: ", i);
                int    ii;
                int    jj;
                double sum_row = 0.0;
                for (ii = A->IA[i]; ii < A->IA[i + 1]; ++ii) {
                    jj = A->JA[ii];
                    sum_row += A->val[ii];
                    // printf("A[%d,%d] = %f, ", i, jj, A->val[ii]);
                }
                printf("(no abs op)sum_row = %f\n", sum_row);
            }
        }
    }

    // second pass: P->JA
    // TODO: given nnz, we can combine the two passes
    P->nnz = P->IA[row];
    P->JA  = (INT*)fasp_mem_calloc(P->nnz, sizeof(INT));
    for (i = 0; i < row; ++i) {
        if (vec[i] == CGPT) {   // identity interpolation for C-points
            P->JA[index++] = i; // use index on fine grid (need to be replaced with
                                // coarse grid index later)
        } else {                // D_FF^-1 * A_FC for F-points
            for (k = A->IA[i]; k < A->IA[i + 1]; ++k) {
                j = A->JA[k];
                if (vec[j] == CGPT) P->JA[index++] = j;
            }
        }
    }

    // val allocated here
    P->val = (REAL*)fasp_mem_calloc(P->nnz, sizeof(REAL));
}

/**
 * \fn static void form_P_pattern_dir (dCSRmat *P, iCSRmat *S, ivector *vertices,
 *                                     INT row, INT col)
 *
 * \brief Generate sparsity pattern of prolongation for direct interpolation
 *
 * \param P         Pointer to the prolongation matrix
 * \param S         Pointer to the set of all strong couplings matrix
 * \param vertices  Pointer to the type of variables
 * \param row       Number of rows of P
 * \param col       Number of cols of P
 *
 * \author Xuehai Huang, Chensong Zhang
 * \date   09/06/2010
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/23/2012: add OMP support
 * Modified by Chensong Zhang on 05/13/2013: restructure the code
 */
static void
form_P_pattern_dir(dCSRmat* P, iCSRmat* S, ivector* vertices, INT row, INT col)
{
    // local variables
    INT  i, j, k, index;
    INT* vec = vertices->val;

    SHORT nthreads = 1, use_openmp = FALSE;

#ifdef _OPENMP
    if (row > OPENMP_HOLDS) {
        use_openmp = TRUE;
        nthreads   = fasp_get_num_threads();
    }
#endif

    // Initialize P matrix
    P->row = row;
    P->col = col;
    P->IA  = (INT*)fasp_mem_calloc(row + 1, sizeof(INT));

    // step 1: Find the structure IA of P first: using P as a counter
    if (use_openmp) {

        INT mybegin, myend, myid;
#ifdef _OPENMP
#pragma omp parallel for private(myid, mybegin, myend, i, j, k)
#endif
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, row, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                switch (vec[i]) {
                    case FGPT: // fine grid points
                        for (j = S->IA[i]; j < S->IA[i + 1]; j++) {
                            k = S->JA[j];
                            if (vec[k] == CGPT) P->IA[i + 1]++;
                        }
                        break;

                    case CGPT: // coarse grid points
                        P->IA[i + 1] = 1;
                        break;

                    default: // treat everything else as isolated
                        P->IA[i + 1] = 0;
                        break;
                }
            }
        }

    }

    else {

        for (i = 0; i < row; ++i) {
            switch (vec[i]) {
                case FGPT: // fine grid points
                    for (j = S->IA[i]; j < S->IA[i + 1]; j++) {
                        k = S->JA[j];
                        if (vec[k] == CGPT) P->IA[i + 1]++;
                    }
                    break;

                case CGPT: // coarse grid points
                    P->IA[i + 1] = 1;
                    break;

                default: // treat everything else as isolated
                    P->IA[i + 1] = 0;
                    break;
            }
        } // end for i

    } // end if

    // Form P->IA from the counter P
    for (i = 0; i < P->row; ++i) P->IA[i + 1] += P->IA[i];
    P->nnz = P->IA[P->row] - P->IA[0];

    // step 2: Find the structure JA of P
    P->JA  = (INT*)fasp_mem_calloc(P->nnz, sizeof(INT));
    P->val = (REAL*)fasp_mem_calloc(P->nnz, sizeof(REAL));

    for (index = i = 0; i < row; ++i) {
        if (vec[i] == FGPT) { // fine grid point
            for (j = S->IA[i]; j < S->IA[i + 1]; j++) {
                k = S->JA[j];
                if (vec[k] == CGPT) P->JA[index++] = k;
            }                      // end for j
        }                          // end if
        else if (vec[i] == CGPT) { // coarse grid point -- one entry only
            P->JA[index++] = i;
        }
    }
}

/**
 * \fn static void form_P_pattern_std (dCSRmat *P, iCSRmat *S, ivector *vertices,
 *                                     INT row, INT col)
 *
 * \brief Generate sparsity pattern of prolongation for standard interpolation
 *
 * \param P         Pointer to the prolongation matrix
 * \param S         Pointer to the set of all strong couplings matrix
 * \param vertices  Pointer to the type of variables
 * \param row       Number of rows of P
 * \param col       Number of cols of P
 *
 * \author Kai Yang, Xiaozhe Hu
 * \date   05/21/2012
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/13/2012: add OMP support
 * Modified by Chensong Zhang on 05/13/2013: restructure the code
 */
static void
form_P_pattern_std(dCSRmat* P, iCSRmat* S, ivector* vertices, INT row, INT col)
{
    // local variables
    INT  i, j, k, l, h, index;
    INT* vec = vertices->val;

    // number of times a C-point is visited
    INT* visited = (INT*)fasp_mem_calloc(row, sizeof(INT));

    P->row = row;
    P->col = col;
    P->IA  = (INT*)fasp_mem_calloc(row + 1, sizeof(INT));

    fasp_iarray_set(row, visited, -1);

    // Step 1: Find the structure IA of P first: use P as a counter
    for (i = 0; i < row; ++i) {

        if (vec[i] == FGPT) { // if node i is a F point
            for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

                k = S->JA[j];

                // if neighbor of i is a C point, good
                if ((vec[k] == CGPT) && (visited[k] != i)) {
                    visited[k] = i;
                    P->IA[i + 1]++;
                }

                // if k is a F point and k is not i, look for indirect C neighbors
                else if ((vec[k] == FGPT) && (k != i)) {
                    for (l = S->IA[k]; l < S->IA[k + 1]; l++) { // neighbors of k
                        h = S->JA[l];
                        if ((vec[h] == CGPT) && (visited[h] != i)) {
                            visited[h] = i;
                            P->IA[i + 1]++;
                        }
                    } // end for(l=S->IA[k];l<S->IA[k+1];l++)
                }     // end if (vec[k]==CGPT)

            } // end for (j=S->IA[i];j<S->IA[i+1];j++)
        }

        else if (vec[i] == CGPT) { // if node i is a C point
            P->IA[i + 1] = 1;
        }

        else { // treat everything else as isolated points
            P->IA[i + 1] = 0;
        } // end if (vec[i]==FGPT)

    } // end for (i=0;i<row;++i)

    // Form P->IA from the counter P
    for (i = 0; i < P->row; ++i) P->IA[i + 1] += P->IA[i];
    P->nnz = P->IA[P->row] - P->IA[0];

    // Step 2: Find the structure JA of P
    P->JA  = (INT*)fasp_mem_calloc(P->nnz, sizeof(INT));
    P->val = (REAL*)fasp_mem_calloc(P->nnz, sizeof(REAL));

    fasp_iarray_set(row, visited, -1); // re-init visited array

    for (i = 0; i < row; ++i) {

        if (vec[i] == FGPT) { // if node i is a F point

            index = 0;

            for (j = S->IA[i]; j < S->IA[i + 1]; j++) {

                k = S->JA[j];

                // if neighbor k of i is a C point
                if ((vec[k] == CGPT) && (visited[k] != i)) {
                    visited[k]              = i;
                    P->JA[P->IA[i] + index] = k;
                    index++;
                }

                // if neighbor k of i is a F point and k is not i
                else if ((vec[k] == FGPT) && (k != i)) {
                    for (l = S->IA[k]; l < S->IA[k + 1]; l++) { // neighbors of k
                        h = S->JA[l];
                        if ((vec[h] == CGPT) && (visited[h] != i)) {
                            visited[h]              = i;
                            P->JA[P->IA[i] + index] = h;
                            index++;
                        }

                    } // end for (l=S->IA[k];l<S->IA[k+1];l++)

                } // end if (vec[k]==CGPT)

            } // end for (j=S->IA[i];j<S->IA[i+1];j++)
        }

        else if (vec[i] == CGPT) {
            P->JA[P->IA[i]] = i;
        }
    }

    // clean up
    fasp_mem_free(visited);
    visited = NULL;
}

/**
 * \fn static INT cfsplitting_mis (iCSRmat *S, ivector *vertices, ivector *order)
 *
 * \brief Find coarse level variables (C/F splitting): MIS
 *
 * \param S                Strong connection matrix
 * \param vertices         Indicator vector for the C/F splitting of the variables
 * \param order        order of vertices
 *
 * \return Number of cols of P
 *
 * \author Hongxuan Zhang, Xiaozhe Hu
 * \date   10/12/2014
 */
static INT cfsplitting_mis(iCSRmat* S, ivector* vertices, ivector* order)
{
    const INT n = S->row;

    INT  col = 0;
    INT* ord = order->val;
    INT* vec = vertices->val;
    INT* IS  = S->IA;
    INT* JS  = S->JA;

    INT i, j, ind;
    INT row_begin, row_end;

    fasp_ivec_set(n, vertices, UNPT);

    for (i = 0; i < n; i++) {
        ind = ord[i];
        if (vec[ind] == UNPT) {
            vec[ind]  = CGPT;
            row_begin = IS[ind];
            row_end   = IS[ind + 1];
            for (j = row_begin; j < row_end; j++) {
                if (vec[JS[j]] == CGPT) {
                    vec[ind] = FGPT;
                    break;
                }
            }
            if (vec[ind] == CGPT) {
                col++;
                for (j = row_begin; j < row_end; j++) {
                    vec[JS[j]] = FGPT;
                }
            }
        }
    }
    return col;
}

/**
 * \fn static void ordering1 (iCSRmat *S, ivector *order)
 *
 * \brief reorder the vertices of A base on their degrees.
 *
 * \param S            Strong connection matrix
 * \param order        order of vertices (output)
 *
 * \note The vertex with highest degree will appear first. Other vertices will use
 *       nature order.
 *
 * \author Hongxuan Zhang
 * \date   10/12/2014
 */
static void ordering1(iCSRmat* S, ivector* order)
{
    const INT n   = order->row;
    INT*      IS  = S->IA;
    INT*      ord = order->val;
    INT       maxind, maxdeg, degree;
    INT       i;

    for (i = 0; i < n; i++) ord[i] = i;

    for (maxind = maxdeg = i = 0; i < n; i++) {
        degree = IS[i + 1] - IS[i];
        if (degree > maxdeg) {
            maxind = i;
            maxdeg = degree;
        }
    }

    ord[0]      = maxind;
    ord[maxind] = 0;

    return;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
