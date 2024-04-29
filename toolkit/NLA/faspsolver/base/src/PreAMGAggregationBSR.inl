/*! \file  PreAMGAggregationBSR.inl
 *
 *  \brief Utilities for aggregation methods for BSR matrices
 *
 *  \note  This file contains Level-4 (Pre) functions, which are used in:
 *         PreAMGSetupSABSR.c and PreAMGSetupUABSR.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2014--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static dCSRmat condenseBSR (const dBSRmat *A)
 *
 * \brief Form a dCSRmat matrix from a dBSRmat matrix: use the (1,1)-entry
 *
 * \param A    Pointer to the BSR format matrix
 *
 * \return     dCSRmat matrix if succeed, NULL if fail
 *
 * \author Xiaozhe Hu
 * \date   03/16/2012
 */
static dCSRmat condenseBSR(const dBSRmat* A)
{
    // information about A
    const INT   ROW = A->ROW;
    const INT   COL = A->COL;
    const INT   NNZ = A->NNZ;
    const SHORT nc  = A->nb;
    const SHORT nc2 = nc * nc;
    const REAL  TOL = 1e-8;

    const REAL* val = A->val;
    const INT*  IA  = A->IA;
    const INT*  JA  = A->JA;

    // (1,1) block
    dCSRmat P_csr = fasp_dcsr_create(ROW, COL, NNZ);
    REAL*   Pval  = P_csr.val;
    memcpy(P_csr.JA, JA, NNZ * sizeof(INT));
    memcpy(P_csr.IA, IA, (ROW + 1) * sizeof(INT));

#ifdef _OPENMP
    INT i;

#pragma omp parallel for if (NNZ > OPENMP_HOLDS)
    for (i = NNZ - 1; i >= 0; i--) Pval[i] = val[i * nc2];

#else
    INT i, j;

    for (i = NNZ, j = NNZ * nc2 - nc2 + (0 * nc + 0); i--; j -= nc2) Pval[i] = val[j];
#endif

    // compress CSR format
    fasp_dcsr_compress_inplace(&P_csr, TOL);

    // return P
    return P_csr;
}

/**
 * \fn static dCSRmat condenseBSRLinf (const dBSRmat *A)
 *
 * \brief Form a dCSRmat matrix from a dBSRmat matrix: use inf-norm of each block
 *
 * \param A    Pointer to the BSR format matrix
 *
 * \return     dCSRmat matrix if succeed, NULL if fail
 *
 * \author Xiaozhe Hu
 * \date   05/25/2014
 */
static dCSRmat condenseBSRLinf(const dBSRmat* A)
{
    // information about A
    const INT   ROW = A->ROW;
    const INT   COL = A->COL;
    const INT   NNZ = A->NNZ;
    const SHORT nc  = A->nb;
    const SHORT nc2 = nc * nc;
    const REAL  TOL = 1e-8;

    const REAL* val = A->val;
    const INT*  IA  = A->IA;
    const INT*  JA  = A->JA;

    // CSR matrix
    dCSRmat Acsr = fasp_dcsr_create(ROW, COL, NNZ);
    REAL*   Aval = Acsr.val;

    // get structure
    memcpy(Acsr.JA, JA, NNZ * sizeof(INT));
    memcpy(Acsr.IA, IA, (ROW + 1) * sizeof(INT));

    INT i, j, k;
    INT row_start, row_end;

    for (i = 0; i < ROW; i++) {

        row_start = A->IA[i];
        row_end   = A->IA[i + 1];

        for (k = row_start; k < row_end; k++) {
            j       = A->JA[k];
            Aval[k] = fasp_smat_Linf(val + k * nc2, nc);
            if (i != j) Aval[k] = -Aval[k];
        }
    }

    // compress CSR format
    fasp_dcsr_compress_inplace(&Acsr, TOL);

    // return CSR matrix
    return Acsr;
}

/**
 * \fn static void form_boolean_p_bsr (const ivector *vertices, dBSRmat *tentp,
 *                                     const AMG_data_bsr *mgl,
 *                                     const INT NumAggregates)
 *
 * \brief Form boolean prolongations in dBSRmat (assume constant vector is in
 *        the null space)
 *
 * \param vertices           Pointer to the aggregation of vertices
 * \param tentp              Pointer to the prolongation operators
 * \param mgl                Pointer to AMG levels
 * \param NumAggregates      Number of aggregations
 *
 * \author Xiaozhe Hu
 * \date   05/27/2014
 */
static void form_boolean_p_bsr(const ivector*      vertices,
                               dBSRmat*            tentp,
                               const AMG_data_bsr* mgl,
                               const INT           NumAggregates)
{
    INT i, j;

    /* Form tentative prolongation */
    tentp->ROW = vertices->row;
    tentp->COL = NumAggregates;
    tentp->nb  = mgl->A.nb;
    INT nb2    = tentp->nb * tentp->nb;

    tentp->IA = (INT*)fasp_mem_calloc(tentp->ROW + 1, sizeof(INT));

    // local variables
    INT*  IA = tentp->IA;
    INT*  JA;
    REAL* val;
    INT*  vval = vertices->val;

    const INT row = tentp->ROW;

    // first run
    for (i = 0, j = 0; i < row; i++) {
        IA[i] = j;
        if (vval[i] > -1) {
            j++;
        }
    }
    IA[row] = j;

    // allocate
    tentp->NNZ = j;

    tentp->JA = (INT*)fasp_mem_calloc(tentp->NNZ, sizeof(INT));

    tentp->val = (REAL*)fasp_mem_calloc(tentp->NNZ * nb2, sizeof(REAL));

    JA  = tentp->JA;
    val = tentp->val;

    // second run
    for (i = 0, j = 0; i < row; i++) {
        IA[i] = j;
        if (vval[i] > -1) {
            JA[j] = vval[i];
            fasp_smat_identity(&(val[j * nb2]), tentp->nb, nb2);
            j++;
        }
    }
}

/**
 * \fn static void form_tentative_p_bsr1 (const ivector *vertices, dBSRmat *tentp,
 *                                        const AMG_data_bsr *mgl, const INT
 * NumAggregates, const const INT dim, REAL **basis)
 *
 * \brief Form tentative prolongation for BSR format matrix (use general basis for
 *        the null space)
 *
 * \param vertices           Pointer to the aggregation of vertices
 * \param tentp              Pointer to the prolongation operators
 * \param mgl                Pointer to AMG levels
 * \param NumAggregates      Number of aggregations
 * \param dim                Dimension of the near kernel space
 * \param basis              Pointer to the basis of the near kernel space
 *
 * \author Xiaozhe Hu
 * \date   05/27/2014
 */
static void form_tentative_p_bsr1(const ivector*      vertices,
                                  dBSRmat*            tentp,
                                  const AMG_data_bsr* mgl,
                                  const INT           NumAggregates,
                                  const INT           dim,
                                  REAL**              basis)
{
    INT i, j, k;

    INT p, q;

    const INT nnz_row = dim / mgl->A.nb; // nonzeros per row

    /* Form tentative prolongation */
    tentp->ROW    = vertices->row;
    tentp->COL    = NumAggregates * nnz_row;
    tentp->nb     = mgl->A.nb;
    const INT nb  = tentp->nb;
    const INT nb2 = nb * nb;

    tentp->IA = (INT*)fasp_mem_calloc(tentp->ROW + 1, sizeof(INT));

    // local variables
    INT*  IA = tentp->IA;
    INT*  JA;
    REAL* val;

    const INT* vval = vertices->val;
    const INT  row  = tentp->ROW;

    // first run
    for (i = 0, j = 0; i < row; i++) {
        IA[i] = j;
        if (vval[i] > -1) {
            j = j + nnz_row;
        }
    }
    IA[row] = j;

    // allocate
    tentp->NNZ = j;
    tentp->JA  = (INT*)fasp_mem_calloc(tentp->NNZ, sizeof(INT));
    tentp->val = (REAL*)fasp_mem_calloc(tentp->NNZ * nb2, sizeof(REAL));

    JA  = tentp->JA;
    val = tentp->val;

    // second run
    for (i = 0, j = 0; i < row; i++) {
        IA[i] = j;
        if (vval[i] > -1) {

            for (k = 0; k < nnz_row; k++) {

                JA[j] = vval[i] * nnz_row + k;

                for (p = 0; p < nb; p++) {

                    for (q = 0; q < nb; q++) {

                        val[j * nb2 + p * nb + q] = basis[k * nb + p][i * nb + q];
                    }
                }

                j++;
            }
        }
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
