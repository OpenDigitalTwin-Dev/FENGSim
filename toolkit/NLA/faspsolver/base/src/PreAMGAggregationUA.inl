/*! \file  PreAMGAggregationUA.inl
 *
 *  \brief Utilities for unsmoothed aggregation methods
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreAMGSetupUA.c and PreAMGSetupUABSR.c
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

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void pair_aggregate_init (const dCSRmat *A, const SHORT checkdd,
 *                                      const REAL kaptg, INT *iperm,
 *                                      ivector *vertices, REAL *s)
 *
 * \brief Initial vertices for first pass aggregation
 *
 * \param A         Pointer to the coefficient matrices
 * \param checkdd   Pointer to diagonal dominant checking
 * \param iperm     Pointer to large positive off-diagonal rows.
 * \param vertices  Pointer to the aggregation of vertices
 * \param s         Pointer to off-diagonal row sum
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/23/2014
 */
static void pair_aggregate_init (const dCSRmat  *A,
                                 const SHORT     checkdd,
                                 const REAL      kaptg,
                                 INT            *iperm,
                                 ivector        *vertices,
                                 REAL           *s)
{
    INT i, j, col;
    INT row = A->row;
    INT *ia = A->IA;
    INT *ja = A->JA;
    REAL *val = A->val;
    REAL strong_hold, aij, aii, rowsum, absrowsum, max;

    REAL *colsum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));
    REAL *colmax = (REAL*)fasp_mem_calloc(row, sizeof(REAL));
    REAL *abscolsum = (REAL*)fasp_mem_calloc(row, sizeof(REAL));

    strong_hold = kaptg/(kaptg - 2.0);

    for (i=0; i<row; ++i) {
        for (j=ia[i]+1; j<ia[i+1]; ++j) {
            col = ja[j];
            aij = val[j];
            colsum[col] += aij;
            colmax[col] = MAX(colmax[col], aij);
            if (checkdd) abscolsum[col] += ABS(aij);
        }
    }

    for (i=0; i<row; ++i) {
        rowsum = 0.0; max = 0.0; absrowsum = 0.0;
        aii = val[ia[i]];
        for (j=ia[i]+1; j<ia[i+1]; ++j) {
            aij = val[j];
            rowsum += aij;
            max = MAX(max, aij);
            if (checkdd) absrowsum += ABS(aij);
        }
        rowsum = 0.5*(colsum[i] + rowsum);
        max = MAX(colmax[i], max);
        if (checkdd) absrowsum = 0.5*(abscolsum[i] + absrowsum);

        s[i] = -rowsum;

        if (aii > strong_hold*absrowsum) {
            vertices->val[i] = G0PT;
        }
        else {
            vertices->val[i] = UNPT;
            if (max > 0.45*aii) iperm[i] = -1;
        }
    }

    fasp_mem_free(colsum);    colsum    = NULL;
    fasp_mem_free(colmax);    colmax    = NULL;
    fasp_mem_free(abscolsum); abscolsum = NULL;
}

/**
 * \fn static void pair_aggregate_init2 (const dCSRmat *A, ivector *map,
 *                                       ivector *vertices, REAL *s1, REAL *s)
 *
 * \brief Initial vertices for second pass aggregation based on initial matrix
 *
 * \param A         Pointer to the coefficient matrices
 * \param map       Pointer to mapping form fine nodes to coarse nodes
 * \param vertices  Pointer to the aggregation of vertices
 * \param s1        Pointer to off-diagonal row sum of initial matrix
 * \param s         Pointer to off-diagonal row sum of temporary coarse matrix
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/23/2014
 */
static void pair_aggregate_init2 (const dCSRmat  *A,
                                  ivector        *map,
                                  ivector        *vertices,
                                  REAL           *s1,
                                  REAL           *s)
{
    INT i, j, k, col, nc;
    INT *ia = A->IA;
    INT *ja = A->JA;
    REAL *val = A->val;

    REAL si;
    const INT NumAggregates = map->row/2;

    for (i=0; i<NumAggregates; ++i) {
        j = map->val[2*i];
        si = 0;
        si = si + s1[j];
        for (k=ia[j]; k<ia[j+1]; ++k) {
            col = ja[k];
            nc = vertices->val[col];
            if ((nc==i) && (col != j)) si += val[k];
        }
        j = map->val[2*i+1];
        if (j < 0) {
            s[i] = si;
            continue;
        }
        si = si + s1[j];
        for (k=ia[j]; k<ia[j+1]; ++k) {
            col = ja[k];
            nc = vertices->val[col];
            if ((nc==i) && (col != j)) si += val[k];
        }
        s[i] = si;
    }
}

/**
 * \fn static void form_pairwise (const dCSRmat *A, const INT pair,
 *                                const REAL k_tg, ivector *vertices,
 *                                INT *NumAggregates)
 *
 * \brief Form aggregation based on pairwise matching
 *
 * \param A                 Pointer to the coefficient matrices
 * \param pair              Number of pairs in matching
 * \param vertices          Pointer to the aggregation of vertices
 * \param NumAggregates     Pointer to number of aggregations
 *
 * \author Xiaoping Li, Zheng Li, Chensong Zhang
 * \date   04/21/2014
 *
 * \note Refer to Artem Napov and Yvan Notay "An algebraic multigrid
 *       method with guaranteed convergence rate" 2011.
 */
static void form_pairwise (const dCSRmat  *A,
                           const INT       pair,
                           const REAL      k_tg,
                           ivector        *vertices,
                           INT            *NumAggregates)
{
    const INT row  = A->row;

    const INT  *AIA  = A->IA;
    const INT  *AJA  = A->JA;
    const REAL *Aval = A->val;

    INT   i, j, row_start, row_end;
    REAL  sum;

    INT   col, index = 0;
    REAL  mu, min_mu, aii, ajj, aij;
    REAL  temp1, temp2, temp3, temp4;

    /*---------------------------------------------------------*/
    /* Step 1. select extremely strong diagonal dominate rows  */
    /*         and store in G0.                                */
    /*         G0        : vertices->val[i]=G0PT               */
    /*         Remaining : vertices->val[i]=UNPT               */
    /*---------------------------------------------------------*/

    fasp_ivec_alloc(row, vertices);

    if ( pair == 1 ) {
        for ( i = 0; i < row; i++ ) {
            sum = 0.0;
            row_start = AIA[i];
            row_end = AIA[i+1];

            for ( j = row_start+1; j < row_end; j++) sum += ABS(Aval[j]);

            if ( Aval[AIA[i]] >= ((k_tg+1.)/(k_tg-1.))*sum) {
                vertices->val[i] = G0PT;
            }
            else {
                vertices->val[i] = UNPT;
            }
        }
    }
    else {
        fasp_iarray_set(row, vertices->val, UNPT);
    }

    /*---------------------------------------------------------*/
    /* Step 2. compute row sum (off-diagonal) for each vertex  */
    /*---------------------------------------------------------*/

    REAL *s = (REAL *)fasp_mem_calloc(row, sizeof(REAL));

    for ( i = 0; i < row; i++ ) {
        s[i] = 0.0;

        if ( vertices->val[i] == G0PT ) continue;

        row_start = AIA[i]; row_end = AIA[i+1];
        for ( j = row_start + 1; j < row_end; j++ ) s[i] -= Aval[j];
    }

    /*---------------------------------------------------------*/
    /* Step 3. start the pairwise aggregation                  */
    /*---------------------------------------------------------*/

    *NumAggregates = 0;

    for ( i = 0; i < row; i++ ) {

        if ( vertices->val[i] != UNPT ) continue;

        min_mu = BIGREAL;

        row_start = AIA[i]; row_end = AIA[i+1];

        aii = Aval[row_start];

        for ( j= row_start + 1; j < row_end; j++ ) {
            col = AJA[j];
            if ( vertices->val[col] != UNPT ) continue;

            aij = Aval[j];
            ajj = Aval[AIA[col]];

            temp1 = aii+s[i]+2*aij;
            temp2 = ajj+s[col]+2*aij;
            temp2 = 1.0/temp1+1.0/temp2;

            temp3 = MAX(ABS(aii-s[i]), SMALLREAL); // avoid temp3 to be zero
            temp4 = MAX(ABS(ajj-s[col]), SMALLREAL); // avoid temp4 to be zero
            temp4 = -aij+1./(1.0/temp3+1.0/temp4);
            // avoid temp4 to be zero
            if ( ABS(temp4) < SMALLREAL ) 
                temp4 = (temp4>0)? SMALLREAL:-SMALLREAL;

            mu    = (-aij+1.0/temp2) / temp4;

            if ( min_mu > mu ) {
                min_mu = mu;
                index  = col;
            }
        }

        vertices->val[i] = *NumAggregates;

        if ( min_mu <= k_tg ) vertices->val[index] = *NumAggregates;

        *NumAggregates += 1;
    }

    fasp_mem_free(s); s = NULL;
}

/**
 * \fn static void form_boolean_p (const ivector *vertices, dCSRmat *tentp,
 *                                 const INT NumAggregates)
 *
 * \brief Form aggregation based on strong coupled neighbors
 *
 * \param vertices           Pointer to the aggregation of vertices
 * \param tentp              Pointer to the prolongation operators
 * \param NumAggregates      Number of aggregations
 *
 * \author Xiaozhe Hu
 * \date   09/29/2009
 */
static void form_boolean_p (const ivector  *vertices,
                            dCSRmat        *tentp,
                            const INT       NumAggregates)
{
    INT i, j;

    /* Form tentative prolongation */
    tentp->row = vertices->row;
    tentp->col = NumAggregates;
    tentp->nnz = vertices->row;
    tentp->IA  = (INT *)fasp_mem_calloc(tentp->row+1,sizeof(INT));

    // local variables
    INT       *IA   = tentp->IA;
    INT       *vval = vertices->val;
    const INT  row  = tentp->row;

    // first run
    for ( i = 0, j = 0; i < row; i++ ) {
        IA[i] = j;
        if (vval[i] > UNPT) j++;
    }
    IA[row] = j;

    // allocate memory for P
    tentp->nnz = j;
    tentp->JA  = (INT *)fasp_mem_calloc(tentp->nnz, sizeof(INT));
    tentp->val = (REAL *)fasp_mem_calloc(tentp->nnz, sizeof(REAL));

    INT  *JA = tentp->JA;
    REAL *val = tentp->val;

    // second run
    for (i = 0, j = 0; i < row; i ++) {
        IA[i] = j;
        if (vval[i] > UNPT) {
            JA[j] = vval[i];
            val[j] = 1.0;
            j ++;
        }
    }
}

/**
 * \fn static SHORT aggregation_symmpair (dCSRmat *A, AMG_param *param,
 *                                        const INT level, ivector *vertices,
 *                                        INT *NumAggregates)
 *
 * \brief AMG coarsening based on symmetric pairwise matching aggregation
 *
 * \param mgl               Pointer to multigrid data
 * \param param             Pointer to AMG parameters
 * \param level             Level number
 * \param vertices          Pointer to the aggregation of vertices
 * \param NumAggregates     Pointer to number of aggregations
 *
 * \author Xiaoping Li, Zheng Li, Chensong Zhang
 * \date   04/21/2014
 *
 * \note Setup A, P, PT and levels using the pairwise aggregation;
 *       Refer to A. Napov and Y. Notay
 *       "An algebraic multigrid method with guaranteed convergence rate", 2012
 *
 * Modified by Chensong Zhang, Zheng Li on 07/29/2014
 */
static SHORT aggregation_symmpair (AMG_data   *mgl,
                                   AMG_param  *param,
                                   const INT   level,
                                   ivector    *vertices,
                                   INT        *NumAggregates)
{
    const INT  pair_number = param->pair_number;
    dCSRmat  * ptrA = &mgl[level].A;
    REAL       quality_bound = param->quality_bound;

    INT        i, j, k, num_agg = 0, aggindex;
    INT        lvl = level;
    REAL       isorate;

    SHORT      dopass = 0, domin = 0;
    SHORT      status = FASP_SUCCESS;

    INT bandwidth = fasp_dcsr_bandwidth(&mgl[level].A);
    if (bandwidth > 5.0)
        param->quality_bound = quality_bound = 1.0*bandwidth;

    for ( i = 1; i <= pair_number; ++i ) {

        /*-- generate aggregations by pairwise matching --*/
        form_pairwise(ptrA, i, quality_bound, &vertices[lvl], &num_agg);

        /*-- check number of aggregates in the first pass --*/
        if ( i == 1 && num_agg < MIN_CDOF ) {
            for ( domin=k=0; k<ptrA->row; k++ ) {
                if ( vertices[lvl].val[k] == G0PT ) domin ++;
            }
            isorate = (REAL)num_agg/domin;
            if ( isorate < 0.1 ) {
                status = ERROR_AMG_COARSEING; goto END;
            }
        }

        if ( i < pair_number ) {

            /*-- Form Prolongation --*/
            form_boolean_p(&vertices[lvl], &mgl[lvl].P, num_agg);

            /*-- Perform aggressive coarsening only up to the specified level --*/
            if ( mgl[lvl].P.col < MIN_CDOF ) break;

            /*-- Form restriction --*/
            fasp_dcsr_trans(&mgl[lvl].P, &mgl[lvl].R);

            /*-- Form coarse level stiffness matrix --*/
            fasp_blas_dcsr_rap_agg(&mgl[lvl].R, ptrA, &mgl[lvl].P, &mgl[lvl+1].A);

            ptrA = &mgl[lvl+1].A;

            fasp_dcsr_free(&mgl[lvl].P);
            fasp_dcsr_free(&mgl[lvl].R);
        }
        lvl ++; dopass ++;
    }

    // Form global aggregation indices
    if ( dopass > 1 ) {
        for ( i = 0; i < mgl[level].A.row; ++i ) {
            aggindex = vertices[level].val[i];
            if ( aggindex < 0 ) continue;
            for ( j = 1; j < dopass; ++j ) aggindex = vertices[level+j].val[aggindex];
            vertices[level].val[i] = aggindex;
        }
    }
    *NumAggregates = num_agg;

    /*-- clean memory --*/
    for ( i = 1; i < dopass; ++i ) {
        fasp_dcsr_free(&mgl[level+i].A);
        fasp_ivec_free(&vertices[level+i]);
    }

END:
    return status;
}

/**
 * \fn static SHORT cholesky_factorization_check (REAL W[8][8],
 *                                                const INT agg_size)
 *
 * \brief Cholesky factorization
 *
 * \param W        Pointer to the coefficient matrices
 * \param agg_size Size of aggregate
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/23/2014
 */
static SHORT cholesky_factorization_check (REAL      W[8][8],
                                           const INT agg_size)
{
    REAL T;
    SHORT status = 0;

    if (agg_size >= 8) {
        if (W[7][7] <= 0.0) return status;
        W[6][6] = W[6][6] - (W[6][7]/W[7][7]) * W[6][7];
        T = W[4][6]/W[6][6];
        W[5][6] = W[5][6] - T * W[6][7];
        W[5][5] = W[5][5] - T * W[5][7];
        T = W[4][7]/W[7][7];
        W[4][6] = W[4][6] - T * W[6][7];
        W[4][5] = W[4][5] - T * W[5][7];
        W[4][4] = W[4][4] - T * W[4][7];
        T = W[3][7]/W[7][7];
        W[3][6] = W[3][6] - T * W[6][7];
        W[3][5] = W[3][5] - T * W[5][7];
        W[3][4] = W[3][4] - T * W[4][7];
        W[3][3] = W[3][3] - T * W[3][7];
        T = W[2][7]/W[7][7];
        W[2][6] = W[2][6] - T * W[6][7];
        W[2][5] = W[2][5] - T * W[5][7];
        W[3][5] = W[3][5] - T * W[4][7];
        W[2][3] = W[2][3] - T * W[3][7];
        W[2][2] = W[2][2] - T * W[2][7];
        T = W[1][7]/W[7][7];
        W[1][6] = W[1][6] - T * W[6][7];
        W[1][5] = W[1][5] - T * W[5][7];
        W[1][4] = W[1][4] - T * W[4][7];
        W[1][3] = W[1][3] - T * W[3][7];
        W[1][2] = W[1][2] - T * W[2][7];
        W[1][1] = W[1][1] - T * W[1][7];
        T = W[0][7]/W[7][7];
        W[0][6] = W[0][6] - T * W[6][7];
        W[0][5] = W[0][5] - T * W[5][7];
        W[0][4] = W[0][4] - T * W[4][7];
        W[0][3] = W[0][3] - T * W[3][7];
        W[0][2] = W[0][2] - T * W[2][7];
        W[0][1] = W[0][1] - T * W[1][7];
        W[0][0] = W[0][0] - T * W[0][7];
    }
    if (agg_size >= 7) {
        if (W[6][6] <= 0.0) return status;
        W[5][5] = W[5][5] - (W[5][6]/W[6][6]) * W[5][6];
        T = W[4][6]/W[6][6];
        W[4][5] = W[4][5] - T * W[5][6];
        W[4][4] = W[4][4] - T * W[4][6];
        T = W[3][6]/W[6][6];
        W[3][5] = W[3][5] - T * W[5][6];
        W[3][4] = W[3][4] - T * W[4][6];
        W[3][3] = W[3][3] - T * W[3][6];
        T = W[2][6]/W[6][6];
        W[2][5] = W[2][5] - T * W[5][6];
        W[3][5] = W[3][5] - T * W[4][6];
        W[2][3] = W[2][3] - T * W[3][6];
        W[2][2] = W[2][2] - T * W[2][6];
        T = W[1][6]/W[6][6];
        W[1][5] = W[1][5] - T * W[5][6];
        W[1][4] = W[1][4] - T * W[4][6];
        W[1][3] = W[1][3] - T * W[3][6];
        W[1][2] = W[1][2] - T * W[2][6];
        W[1][1] = W[1][1] - T * W[1][6];
        T = W[0][6]/W[6][6];
        W[0][5] = W[0][5] - T * W[5][6];
        W[0][4] = W[0][4] - T * W[4][6];
        W[0][3] = W[0][3] - T * W[3][6];
        W[0][2] = W[0][2] - T * W[2][6];
        W[0][1] = W[0][1] - T * W[1][6];
        W[0][0] = W[0][0] - T * W[0][6];
    }
    if (agg_size >= 6) {
        if (W[5][5] <= 0.0) return status;
        W[4][4] = W[4][4] - (W[4][5]/W[5][5]) * W[4][5];
        T = W[3][5]/W[5][5];
        W[3][4] = W[3][4] - T * W[4][5];
        W[3][3] = W[3][3] - T * W[3][5];
        T = W[2][5]/W[5][5];
        W[3][5] = W[3][5] - T * W[4][5];
        W[2][3] = W[2][3] - T * W[3][5];
        W[2][2] = W[2][2] - T * W[2][5];
        T = W[1][5]/W[5][5];
        W[1][4] = W[1][4] - T * W[4][5];
        W[1][3] = W[1][3] - T * W[3][5];
        W[1][2] = W[1][2] - T * W[2][5];
        W[1][1] = W[1][1] - T * W[1][5];
        T = W[0][5]/W[5][5];
        W[0][4] = W[0][4] - T * W[4][5];
        W[0][3] = W[0][3] - T * W[3][5];
        W[0][2] = W[0][2] - T * W[2][5];
        W[0][1] = W[0][1] - T * W[1][5];
        W[0][0] = W[0][0] - T * W[0][5];
    }
    if (agg_size >= 5) {
        if (W[4][4] <= 0.0) return status;
        W[3][3] = W[3][3] - (W[3][4]/W[4][4]) * W[3][4];
        T = W[3][5]/W[4][4];
        W[2][3] = W[2][3] - T * W[3][4];
        W[2][2] = W[2][2] - T * W[3][5];
        T = W[1][4]/W[4][4];
        W[1][3] = W[1][3] - T * W[3][4];
        W[1][2] = W[1][2] - T * W[3][5];
        W[1][1] = W[1][1] - T * W[1][4];
        T = W[0][4]/W[4][4];
        W[0][3] = W[0][3] - T * W[3][4];
        W[0][2] = W[0][2] - T * W[3][5];
        W[0][1] = W[0][1] - T * W[1][4];
        W[0][0] = W[0][0] - T * W[0][4];
    }
    if (agg_size >= 4) {
        if (W[3][3] <= 0.0) return status;
        W[2][2] = W[2][2] - (W[2][3]/W[3][3]) * W[2][3];
        T = W[1][3]/W[3][3];
        W[1][2] = W[1][2] - T * W[2][3];
        W[1][1] = W[1][1] - T * W[1][3];
        T = W[0][3]/W[3][3];
        W[0][2] = W[0][2] - T * W[2][3];
        W[0][1] = W[0][1] - T * W[1][3];
        W[0][0] = W[0][0] - T * W[0][3];
    }
    if (agg_size >= 3) {
        if (W[2][2] <= 0.0) return status;
        W[1][1] = W[1][1] - (W[1][2]/W[2][2]) * W[1][2];
        T = W[0][2]/W[2][2];
        W[0][1] = W[0][1] - T * W[1][2];
        W[0][0] = W[0][0] - T * W[0][2];
    }
    if (agg_size >= 2) {
        if (W[1][1] <= 0.0) return status;
        W[0][0] = W[0][0] - (W[0][1]/W[1][1]) * W[0][1];
    }
    if (agg_size >= 1) {
        if (W[0][0] <= 0.0) return status;
    }

    status = 1;
    return status;
}

/**
 * \fn static SHORT aggregation_quality (const dCSRmat *A, const ivector *tentmap,
 *                                       const REAL *s, const INT root,
 *                                       const INT pair, const INT dopass)
 *
 * \brief From local matrix corresponding to each aggregate
 *
 * \param A                 Pointer to the coefficient matrices
 * \param tentmap           Pointer to the map of first pass
 * \param s                 Pointer to off-diagonal row sum
 * \param root              Root node of each aggregate
 * \param pair              Associate node of each aggregate
 * \param dopass            Number of pass
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/23/2014
 *
 * Use a similar method as in AGMG; refer to Yvan Notay's AGMG-3.2.0.
 */
static SHORT aggregation_quality (const dCSRmat  *A,
                                  const ivector  *tentmap,
                                  const REAL     *s,
                                  const INT       root,
                                  const INT       pair,
                                  const INT       dopass,
                                  const REAL      quality_bound)
{
    const INT  *IA  = A->IA;
    const INT  *JA  = A->JA;
    const REAL *val = A->val;
    const INT  *map = tentmap->val;

    REAL bnd1       = 2*1.0/quality_bound;
    REAL bnd2       = 1.0-bnd1;
    REAL alpha      = 1.0, beta = 1.0;
    INT  agg_size   = 1;

    REAL W[8][8], v[4], sig[4], AG[4];
    INT fnode[4];
    INT i, j, l, k, m, status, flag, jj;

    if (dopass == 2) {
        if (map[2*root+1] < -1) {
            if (map[2*pair+1] < -1) {
                fnode[0] = map[2*root];
                fnode[1] = map[2*pair];
                agg_size = 2;
            }
            else {
                fnode[0] = map[2*root];
                fnode[1] = map[2*pair];
                fnode[2] = map[2*pair+1];
                agg_size = 3;
            }
        }
        else {
            if (map[2*pair+1] < -1) {
                fnode[0] = map[2*root];
                fnode[1] = map[2*root+1];
                fnode[2] = map[2*pair];
                agg_size = 3;
            }
            else {
                fnode[0] = map[2*root];
                fnode[1] = map[2*root+1];
                fnode[2] = map[2*pair];
                fnode[3] = map[2*pair+1];
                agg_size = 4;
            }
        }
    }

    flag = 1;

    while (flag) {
        flag = 0;
        for(i=1; i<agg_size; ++i) {
            if (fnode[i] < fnode[i-1]) {
                jj = fnode[i];
                fnode[i] = fnode[i-1];
                fnode[i-1] = jj;
                flag = 1;
            }
        }
    }

    for (j=0; j<agg_size; ++j) {
        jj = fnode[j];
        sig[j] = s[jj];
        W[j][j]= val[IA[jj]];
        AG[j]  = W[j][j]-sig[j];
        for (l=j+1; l<agg_size; ++l) {
            W[j][l]=0.0;
            W[l][j]=0.0;
        }

        for (k=IA[jj]; k<IA[jj+1]; ++k) {
            if (JA[k]>jj)
                for (l=j+1; l<agg_size; ++l) {
                    m = fnode[l];
                    if (JA[k]==m) {
                        alpha=val[k]/2;
                        W[j][l]=alpha;
                        W[l][j]=alpha;
                        break;
                    }
                }
        }

        for (k=IA[jj]; k<IA[jj+1]; ++k) {
            if (JA[k] < jj)
                for (l=0; l<j; ++l) {
                    m = fnode[l];
                    if (JA[k] == m) {
                        alpha = val[k]/2;
                        W[j][l] = W[j][l]+alpha;
                        W[l][j] = W[j][l];
                        break;
                    }
                }
        }
    }

    for (j=0; j<agg_size; ++j) {
        for (k=0; k<agg_size; ++k) {
            if (j != k) sig[j] += W[j][k];
        }
        if (sig[j] < 0.0)  AG[j] = AG[j] + 2*sig[j];
        v[j] = W[j][j];
        W[j][j] = bnd2*W[j][j]-ABS(sig[j]);

        if (j == 0) {
            beta = v[j];
            alpha = ABS(AG[j]);
        }
        else {
            beta = beta + v[j];
            alpha = MAX(alpha,ABS(AG[j]));
        }
    }
    beta = bnd1/beta;

    for (j=0; j<agg_size; ++j) {
        for (k=0; k<agg_size; ++k) {
            W[j][k] = W[j][k] + beta*v[j]*v[k];
        }
    }

    if (alpha < 1.5e-8*beta) {
        agg_size --;
    }

    status = cholesky_factorization_check(W, agg_size);

    return status;
}

/**
 * \fn void nsympair_1stpass (const dCSRmat * A, const REAL k_tg,
 *                            ivector *vertices, ivector *map, REAL*s,
 *                            INT *NumAggregates)
 *
 * \brief Form initial pass aggregation for non-symmetric problem
 *
 * \param A                Pointer to the coefficient matrices
 * \param k_tg             Two-grid convergence parameter
 * \param vertices         Pointer to the aggregation of vertices
 * \param map              Pointer to the map index of fine nodes to coarse nodes
 * \param s                Pointer to off-diagonal row sum
 * \param NumAggregates    Pointer to number of aggregations
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/23/2014
 *
 * \note  Refer to Yvan Notay "Aggregation-based algebraic multigrid
 *        for convection-diffusion equations" 2011.
 */
static void nsympair_1stpass (const dCSRmat * A,
                              const REAL      k_tg,
                              ivector       * vertices,
                              ivector       * map,
                              REAL          * s,
                              INT           * NumAggregates)
{
    const INT   row  = A->row;
    const INT  *AIA  = A->IA;
    const INT  *AJA  = A->JA;
    const REAL *Aval = A->val;

    INT i, j, row_start, row_end, nc, col, k, ipair, node, checkdd;
    REAL mu, aii, ajj, aij, aji, tent, vals, val = 0;
    REAL del1, del2, eta1, eta2, sig1, sig2, rsi, rsj, epsr, del12;

    nc = 0;
    i = 0;
    node = 0;
    checkdd = 1;

    /*---------------------------------------------------------*/
    /* Step 1. select extremely strong diagonal dominate rows  */
    /*         and store in G0.                                */
    /*---------------------------------------------------------*/

    fasp_ivec_alloc(row, vertices);
    fasp_ivec_alloc(2*row, map);

    INT *iperm = (INT *)fasp_mem_calloc(row, sizeof(INT));

    /*---------------------------------------------------------*/
    /* Step 2. compute row sum (off-diagonal) for each vertex  */
    /*---------------------------------------------------------*/

    /* G0:vertices->val[i]=G0PT, Remain: vertices->val[i]=UNPT */
    pair_aggregate_init(A, checkdd, k_tg, iperm, vertices, s);

    /*-----------------------------------------*/
    /* Step 3. start the pairwise aggregation  */
    /*-----------------------------------------*/
    while ( node < row ) {

        // deal with G0 type node
        if ( vertices->val[i] == G0PT ) {
            node ++;
            i ++;
            continue;
        }

        // skip nodes if they have been determined
        if ( vertices->val[i] != UNPT ) { i ++ ; continue;}

        vertices->val[i] = nc;
        map->val[2*nc] = i;
        node ++;

        // check whether node has large off-diagonal positive node or not
        if ( iperm[i] == -1 ) {
            map->val[2*nc+1] = -1;
            nc ++;
            i ++;
            continue;
        }

        ipair = -1;

        row_start = AIA[i]; row_end = AIA[i+1];

        aii = Aval[row_start];

        for ( j = row_start + 1; j < row_end; j++ ) {
            col = AJA[j];
            if ( vertices->val[col] != UNPT || iperm[col] == -1 ) continue;

            aij = Aval[j];
            ajj = Aval[AIA[col]];
            aji = 0.0;

            for (k = AIA[col]; k < AIA[col+1]; ++k) {
                if (AJA[k] == i) {
                    aji = Aval[k];
                    break;
                }
            }

            vals = -0.5*(aij+aji);

            rsi = -s[i] + aii;
            rsj = -s[col] + ajj;

            eta1 = 2*aii;
            eta2 = 2*ajj;

            sig1 = s[i]-vals;
            sig2 = s[col]-vals;

            if (sig1 > 0) {
                del1 = rsi;
            } else {
                del1 = rsi+2*sig1;
            }
            if (sig2 > 0) {
                del2 = rsj;
            } else {
                del2 = rsj+2*sig2;
            }
            if (vals > 0.0) {
                epsr = 1.49e-8*vals;
                if ((ABS(del1) < epsr) && (ABS(del2) < epsr)) {
                    mu = (eta1*eta2)/(vals*(eta1+eta2));
                } else if (ABS(del1) < epsr) {
                    if (del2 < -epsr) continue;
                    mu = (eta1*eta2)/(vals*(eta1+eta2));
                } else if (ABS(del2) < epsr) {
                    if (del1 < -epsr) continue;
                    mu = (eta1*eta2)/(vals*(eta1+eta2));
                } else {
                    del12 = del1 + del2;
                    if (del12 < -epsr) continue;
                    if (del12 == 0.0) continue;
                    mu = vals + del1*del2/del12;
                    if (mu <= 0.0) continue;
                    mu = ((eta1*eta2)/(eta1+eta2))/mu;
                }
            }
            else {
                if (del1 <= 0.0 || del2 <= 0.0) continue;
                mu = vals + del1*del2/(del1+del2);
                if (mu <= 0.0) continue;
                vals = (eta1*eta2)/(eta1+eta2);
                mu = vals/mu;
            }

            if (mu > k_tg) continue;

            tent = mu;

            if (ipair == -1) {
                ipair = col;
                val = tent;
            }
            else if ( (tent-val) < -0.06 ) {
                ipair = col;
                val = tent;
            }
        }

        if (ipair == -1) {
            map->val[2*nc+1] = -2;
        }
        else {
            vertices->val[ipair] = nc;
            map->val[2*nc+1] = ipair;
            node ++;
        }

        nc++;
        i ++;
    }

    map->row = 2*nc;

	if ( nc > 0 ) map->val = (INT*)fasp_mem_realloc(map->val, sizeof(INT)*map->row);

    *NumAggregates = nc;

    fasp_mem_free(iperm); iperm = NULL;
}

/**
 * \fn void nsympair_2ndpass (const dCSRmat *A, dCSRmat *tmpA, const REAL k_tg,
 *                            INT dopass, ivector *map1, ivector *vertices1, 
 *                            ivector *vertices, ivector *map, REAL *s1, INT *NumAggregates)
 *
 * \brief Form second pass aggregation for non-symmetric problem
 *
 * \param A          Pointer to the coefficient matrices
 * \param tmpA       Pointer to the first pass aggregation coarse matrix
 * \param dopass     Pointer to the number of pass
 * \param map1       Pointer to the map index of fine nodes to coarse nodes in
 *                   initial pass
 * \param vertices1  Pointer to the aggregation of vertices in initial pass
 * \param vertices   Pointer to the aggregation of vertices in the second pass
 * \param map        Pointer to the map index of fine nodes to coarse nodes in
 *                   the second pass
 * \param s1         Pointer to off-diagonal row sum of matrix
 * \param NumAggregates    Pointer to number of aggregations
 *
 * \author Zheng Li, Chensong Zhang
 * \date   12/23/2014
 *
 * \note  Refer to Yvan Notay "Aggregation-based algebraic multigrid
 *        for convection-diffusion equations" 2011.
 */
static void nsympair_2ndpass (const dCSRmat  *A,
                              dCSRmat        *tmpA,
                              const REAL      k_tg,
                              INT             dopass,
                              ivector        *map1,
                              ivector        *vertices1,
                              ivector        *vertices,
                              ivector        *map,
                              REAL           *s1,
                              INT            *NumAggregates)
{
    INT i, j, k, l, m, ijtent;
    const INT row = tmpA->row;
    const INT *AIA = tmpA->IA;
    const INT *AJA = tmpA->JA;
    const REAL *Aval = tmpA->val;

    REAL *Tval;
    INT *Tnode;

    INT  col,ipair,Tsize, row_start, row_end, Semipd, nc, node;
    REAL mu, aii, ajj, aij, tmp, aji, vals, val = 0;
    REAL del1, del2, eta1, eta2, sig1, sig2, rsi, rsj, epsr,del12;

    Tval  = (REAL*)fasp_mem_calloc(row, sizeof(REAL));
    Tnode = (INT*)fasp_mem_calloc(row, sizeof(INT));

    fasp_ivec_alloc(2*row, map);
    fasp_ivec_alloc(row, vertices);

    nc = node = 0;

    REAL *s = (REAL *)fasp_mem_calloc(row, sizeof(REAL));

    pair_aggregate_init2(A, map1, vertices1, s1, s);

    fasp_ivec_set(0, vertices, UNPT);

    i = 0;

    while (node < row) {

        // check nodes whether are aggregated
        if ( vertices->val[i] != UNPT ) {
            i++;
            continue;
        }

        vertices->val[i] = nc;
        map->val[2*nc] = i;

        node ++;
        // if node isolated in first pass will be isolated in second pass
        if (map1->val[2*i+1] == -1) {
            map->val[2*nc+1] = -1;
            nc ++;
            i ++;
            continue;
        }

        ipair = -1;
        Tsize = 0;

        row_start = AIA[i]; row_end = AIA[i+1];

        aii = Aval[row_start];

        for ( j= row_start + 1; j < row_end; j++ ) {
            col = AJA[j];

            if ( vertices->val[col] != UNPT || map1->val[2*col+1] == -1) continue;

            aji = 0.0;
            aij = Aval[j];
            ajj = Aval[AIA[col]];

            for (k = AIA[col]; k < AIA[col+1]; ++k) {
                if (AJA[k]==i) {
                    aji = Aval[k];
                    break;
                }
            }

            vals = -0.5*(aij+aji);
            rsi = -s[i] + aii;
            rsj = -s[col] + ajj;
            eta1 = 2*aii;
            eta2 = 2*ajj;

            sig1 = s[i]-vals;
            sig2 = s[col]-vals;

            if (sig1 > 0) {
                del1 = rsi;
            } else {
                del1 = rsi+2*sig1;
            }
            if (sig2 > 0) {
                del2 = rsj;
            } else {
                del2 = rsj+2*sig2;
            }
            if (vals > 0.0) {
                epsr=1.49e-8*vals;
                if ((ABS(del1) < epsr) && (ABS(del2) < epsr)) {
                    mu = (eta1*eta2)/(vals*(eta1+eta2));
                } else if (ABS(del1) < epsr) {
                    if (del2 < -epsr) continue;
                    mu = (eta1*eta2)/(vals*(eta1+eta2));
                } else if (ABS(del2) < epsr) {
                    if (del1 < -epsr) continue;
                    mu = (eta1*eta2)/(vals*(eta1+eta2));
                } else {
                    del12 = del1 + del2;
                    if (del12 < -epsr) continue;
                    if (del12 == 0.0) continue;
                    mu = vals + del1*del2/del12;
                    if (mu <= 0.0) continue;
                    mu = ((eta1*eta2)/(eta1+eta2))/mu;
                }
            }
            else {
                if (del1 <= 0.0 || del2 <= 0.0) continue;
                mu = vals + del1*del2/(del1+del2);
                if (mu <= 0.0) continue;
                aij = (eta1*eta2)/(eta1+eta2);
                mu = aij/mu;
            }
            if (mu > k_tg) continue;

            tmp = mu;

            if (ipair == -1) {
                ipair = col;
                val = tmp;
            }
            else if ( (tmp-val) < -0.06 ) {
                Tnode[Tsize] = ipair;
                Tval[Tsize]  = val;
                ipair = col;
                val = tmp;
                Tsize ++;

            }
            else {
                Tnode[Tsize] = col;
                Tval[Tsize]  = tmp;
                Tsize ++;
            }
        }

        if (ipair == -1) {
            map->val[2*nc+1] = -2;
            nc ++;
            i ++;
            continue;
        }

        while (Tsize >= 0) {
            Semipd = aggregation_quality(A, map1, s1, i, ipair, dopass, k_tg);
            if (!Semipd) {
                ipair = -1;
                l = 0; m = 0; ijtent = 0;
                while (l < Tsize) {
                    if (Tnode[m] >= 0) {
                        tmp = Tval[m];
                        if (ipair == -1) {
                            val   = tmp;
                            ipair = Tnode[m];
                            ijtent= m;
                        }
                        else if ((tmp-val) < -0.06 ) {
                            val = tmp;
                            ipair = Tnode[m];
                            ijtent = m;
                        }
                        l++;
                    }
                    m++;
                }
                Tsize--;
                Tnode[ijtent]=-1;
            }
            else {
                break;
            }
        }

        if (ipair == -1) {
            map->val[2*nc+1] = -2;
        }
        else {
            vertices->val[ipair] = nc;
            map->val[2*nc+1] = ipair;
            node ++;
        }

        i ++;
        nc ++;
    }

    for (i=0; i<row; ++i) s1[i] = s[i];

    if ( nc > 0 ) map->val = (INT*)fasp_mem_realloc(map->val, sizeof(INT)*2*nc);
    map->row = 2*nc;

    *NumAggregates = nc;

    fasp_mem_free(s);      s     = NULL;
    fasp_mem_free(Tnode);  Tnode = NULL;
    fasp_mem_free(Tval);   Tval  = NULL;
}

/**
 * \fn static SHORT aggregation_nsympair (dCSRmat *A, AMG_param *param,
 *                                        const INT level, ivector *vertices,
 *                                        INT *NumAggregates)
 *
 * \brief AMG coarsening based on unsymmetric pairwise matching aggregation
 *
 * \param mgl               Pointer to multigrid data
 * \param param             Pointer to AMG parameters
 * \param level             Level number
 * \param vertices          Pointer to the aggregation of vertices
 * \param NumAggregates     Pointer to number of aggregations
 *
 * \author Xiaoping Li, Zheng Li, Chensong Zhang
 * \date   04/21/2014
 *
 * \note Setup A, P, PT and levels using the pairwise aggregation;
 *       Refer to A. Napov and Y. Notay
 *       "An algebraic multigrid method with guaranteed convergence rate", 2012
 *
 * Modified by Chensong Zhang, Zheng Li on 07/29/2014
 */
static SHORT aggregation_nsympair (AMG_data   *mgl,
                                   AMG_param  *param,
                                   const INT   level,
                                   ivector    *vertices,
                                   INT        *NumAggregates)
{
    const INT  pair_number = param->pair_number;
    dCSRmat  * ptrA = &mgl[level].A;
    REAL       quality_bound = param->quality_bound;

    INT        i, j, k, num_agg = 0, aggindex;
    INT        lvl = level;
    REAL       isorate;

    SHORT      dopass = 0, domin = 0;
    SHORT      status = FASP_SUCCESS;

    ivector  map1, map2;
    REAL *s = (REAL*)fasp_mem_calloc(ptrA->row, sizeof(REAL));

    for ( i = 1; i <= pair_number; ++i ) {

        if ( i == 1 ) {
            nsympair_1stpass(ptrA, quality_bound, &vertices[lvl], &map1, s, &num_agg);
        }
        else {
			nsympair_2ndpass(&mgl[level].A, ptrA, quality_bound, i, &map1,
                             &vertices[lvl-1], &vertices[lvl], &map2, s, &num_agg);
        }

        /*-- check number of aggregates in the first pass --*/
        if ( i == 1 && num_agg < MIN_CDOF ) {
            for ( domin=k=0; k<ptrA->row; k++ ) {
                if ( vertices[lvl].val[k] == G0PT ) domin ++;
            }
            // Ratio b/w num of aggregates and those cannot fit in aggregates
            isorate = (REAL)num_agg/domin;
            if ( isorate < 0.1 ) {
                status = ERROR_AMG_COARSEING; goto END;
            }
        }

        if ( i < pair_number ) {
            /*-- Form Prolongation --*/
            form_boolean_p(&vertices[lvl], &mgl[lvl].P, num_agg);

			/*-- Perform aggressive coarsening only up to the specified level --*/
            if ( mgl[lvl].P.col < MIN_CDOF ) break;

			/*-- Form restriction --*/
            fasp_dcsr_trans(&mgl[lvl].P, &mgl[lvl].R);

			/*-- Form coarse level stiffness matrix --*/
            fasp_blas_dcsr_rap_agg(&mgl[lvl].R, ptrA, &mgl[lvl].P, &mgl[lvl+1].A);

            ptrA = &mgl[lvl+1].A;

            fasp_dcsr_free(&mgl[lvl].P);
            fasp_dcsr_free(&mgl[lvl].R);
        }
        lvl ++; dopass ++;
    }

    // Form global aggregation indices
    if ( dopass > 1 ) {
        for ( i = 0; i < mgl[level].A.row; ++i ) {
            aggindex = vertices[level].val[i];
            if ( aggindex < 0 ) continue;
            for ( j = 1; j < dopass; ++j ) aggindex = vertices[level+j].val[aggindex];
            vertices[level].val[i] = aggindex;
        }
    }
    *NumAggregates = num_agg;

    /*-- clean memory --*/
    for ( i = 1; i < dopass; ++i ) {
        fasp_dcsr_free(&mgl[level+i].A);
        fasp_ivec_free(&vertices[level+i]);
    }

    fasp_ivec_free(&map1);
    fasp_ivec_free(&map2);
    fasp_mem_free(s); s = NULL;

END:
    return status;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
