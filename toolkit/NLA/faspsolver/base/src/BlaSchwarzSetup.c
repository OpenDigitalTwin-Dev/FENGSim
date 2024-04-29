/*! \file  BlaSchwarzSetup.c
 *
 *  \brief Setup phase for the Schwarz methods
 *
 *  \note  This file contains Level-1 (Bla) functions. It requires:
 *         AuxMemory.c, AuxVector.c, BlaSparseCSR.c, BlaSparseUtil.c,
 *         and KryPvgmres.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>
#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

static void SWZ_level(const INT, dCSRmat*, INT*, INT*, INT*, INT*, const INT);
static void SWZ_block(SWZ_data*, const INT, const INT*, const INT*, INT*);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_swz_dcsr_setup (SWZ_data *swzdata, SWZ_param *swzparam)
 *
 * \brief Setup phase for the Schwarz methods
 *
 * \param swzdata    Pointer to the Schwarz data
 * \param swzparam   Type of the Schwarz method
 *
 * \return           FASP_SUCCESS if succeed
 *
 * \author Ludmil, Xiaozhe Hu
 * \date   03/22/2011
 *
 * Modified by Zheng Li on 10/09/2014
 */
INT fasp_swz_dcsr_setup(SWZ_data* swzdata, SWZ_param* swzparam)
{
    // information about A
    dCSRmat A = swzdata->A;
    INT     n = A.row;

    INT blksolver = swzparam->SWZ_blksolver;
    INT maxlev    = swzparam->SWZ_maxlvl;

    // local variables
    INT     i;
    INT     inroot = -10, nsizei = -10, nsizeall = -10, nlvl = 0;
    INT*    jb = NULL;
    ivector MIS;

    // data for Schwarz method
    INT  nblk;
    INT *iblock = NULL, *jblock = NULL, *mask = NULL, *maxa = NULL;

    // return
    INT flag = FASP_SUCCESS;

    swzdata->swzparam = swzparam;

#if DEBUG_MODE > 0
    printf("### DEBUG: [-Begin-] %s ...\n", __FUNCTION__);
#endif

    // allocate memory
    maxa   = (INT*)fasp_mem_calloc(n, sizeof(INT));
    mask   = (INT*)fasp_mem_calloc(n, sizeof(INT));
    iblock = (INT*)fasp_mem_calloc(n, sizeof(INT));
    jblock = (INT*)fasp_mem_calloc(n, sizeof(INT));

    nsizeall = 0;
    memset(mask, 0, sizeof(INT) * n);
    memset(iblock, 0, sizeof(INT) * n);
    memset(maxa, 0, sizeof(INT) * n);

    maxa[0] = 0;

    // select root nodes
    MIS = fasp_sparse_mis(&A);

    /*-------------------------------------------*/
    // find the blocks
    /*-------------------------------------------*/

    // first pass: do a maxlev level sets out for each node
    for (i = 0; i < MIS.row; i++) {
        inroot = MIS.val[i];
        SWZ_level(inroot, &A, mask, &nlvl, maxa, jblock, maxlev);
        nsizei = maxa[nlvl];
        nsizeall += nsizei;
    }

#if DEBUG_MODE > 1
    printf("### DEBUG: nsizeall = %d\n", nsizeall);
#endif

    // calculated the size of jblock up to here
    jblock = (INT*)fasp_mem_realloc(jblock, (nsizeall + n) * sizeof(INT));

    // second pass: redo the same again, but this time we store in jblock
    maxa[0]   = 0;
    iblock[0] = 0;
    nsizeall  = 0;
    jb        = jblock;
    for (i = 0; i < MIS.row; i++) {
        inroot = MIS.val[i];
        SWZ_level(inroot, &A, mask, &nlvl, maxa, jb, maxlev);
        nsizei        = maxa[nlvl];
        iblock[i + 1] = iblock[i] + nsizei;
        nsizeall += nsizei;
        jb += nsizei;
    }
    nblk = MIS.row;

#if DEBUG_MODE > 1
    printf("### DEBUG: nsizeall = %d, %d\n", nsizeall, iblock[nblk]);
#endif

    /*-------------------------------------------*/
    //  LU decomposition of blocks
    /*-------------------------------------------*/

    memset(mask, 0, sizeof(INT) * n);

    swzdata->blk_data = (dCSRmat*)fasp_mem_calloc(nblk, sizeof(dCSRmat));

    SWZ_block(swzdata, nblk, iblock, jblock, mask);

    // Setup for each block solver
    switch (blksolver) {

#if WITH_MUMPS
        case SOLVER_MUMPS:
            {
                /* use MUMPS direct solver on each block */
                dCSRmat*    blk = swzdata->blk_data;
                Mumps_data* mumps =
                    (Mumps_data*)fasp_mem_calloc(nblk, sizeof(Mumps_data));
                for (i = 0; i < nblk; ++i)
                    mumps[i] = fasp_mumps_factorize(&blk[i], NULL, NULL, PRINT_NONE);
                swzdata->mumps = mumps;

                break;
            }
#endif

#if WITH_UMFPACK
        case SOLVER_UMFPACK:
            {
                /* use UMFPACK direct solver on each block */
                dCSRmat* blk     = swzdata->blk_data;
                void**   numeric = (void**)fasp_mem_calloc(nblk, sizeof(void*));
                dCSRmat  Ac_tran;
                for (i = 0; i < nblk; ++i) {
                    Ac_tran = fasp_dcsr_create(blk[i].row, blk[i].col, blk[i].nnz);
                    fasp_dcsr_transz(&blk[i], NULL, &Ac_tran);
                    fasp_dcsr_cp(&Ac_tran, &blk[i]);
                    numeric[i] = fasp_umfpack_factorize(&blk[i], 0);
                }
                swzdata->numeric = numeric;
                fasp_dcsr_free(&Ac_tran);

                break;
            }
#endif

        default:
            {
                /* do nothing for iterative methods */
            }
    }

#if DEBUG_MODE > 1
    printf("### DEBUG: n = %d, #blocks = %d, max block size = %d\n", n, nblk,
           swzdata->maxbs);
#endif

    /*-------------------------------------------*/
    //  return
    /*-------------------------------------------*/
    swzdata->nblk     = nblk;
    swzdata->iblock   = iblock;
    swzdata->jblock   = jblock;
    swzdata->mask     = mask;
    swzdata->maxa     = maxa;
    swzdata->SWZ_type = swzparam->SWZ_type;

#if DEBUG_MODE > 0
    printf("### DEBUG: [--End--] %s ...\n", __FUNCTION__);
#endif

    return flag;
}

/**
 * \fn void fasp_dcsr_swz_forward (SWZ_data  *swzdata, SWZ_param *swzparam,
 *                                 dvector *x, dvector *b)
 *
 * \brief Schwarz smoother: forward sweep
 *
 * \param swzdata Pointer to the Schwarz data
 * \param swzparam   Pointer to the Schwarz parameter
 * \param x       Pointer to solution vector
 * \param b       Pointer to right hand
 *
 * \author Zheng Li, Chensong Zhang
 * \date   2014/10/5
 */
void fasp_dcsr_swz_forward(SWZ_data* swzdata, SWZ_param* swzparam, dvector* x,
                           dvector* b)
{
    INT i, j, iblk, ki, kj, kij, is, ibl0, ibl1, nloc, iaa, iab;

    // Schwarz partition
    INT      nblk      = swzdata->nblk;
    dCSRmat* blk       = swzdata->blk_data;
    INT*     iblock    = swzdata->iblock;
    INT*     jblock    = swzdata->jblock;
    INT*     mask      = swzdata->mask;
    INT      blksolver = swzparam->SWZ_blksolver;

    // Schwarz data
    dCSRmat A   = swzdata->A;
    INT*    ia  = A.IA;
    INT*    ja  = A.JA;
    REAL*   val = A.val;

    // Local solution and right hand vectors
    dvector rhs = swzdata->rhsloc1;
    dvector u   = swzdata->xloc1;

#if WITH_UMFPACK
    void** numeric = swzdata->numeric;
#endif

#if WITH_MUMPS
    Mumps_data* mumps = swzdata->mumps;
#endif

    for (is = 0; is < nblk; ++is) {
        // Form the right hand of eack block
        ibl0 = iblock[is];
        ibl1 = iblock[is + 1];
        nloc = ibl1 - ibl0;
        for (i = 0; i < nloc; ++i) {
            iblk     = ibl0 + i;
            ki       = jblock[iblk];
            mask[ki] = i + 1;
        }

        for (i = 0; i < nloc; ++i) {
            iblk       = ibl0 + i;
            ki         = jblock[iblk];
            rhs.val[i] = b->val[ki];
            iaa        = ia[ki] - 1;
            iab        = ia[ki + 1] - 1;
            for (kij = iaa; kij < iab; ++kij) {
                kj = ja[kij] - 1;
                j  = mask[kj];
                if (j == 0) {
                    rhs.val[i] -= val[kij] * x->val[kj];
                }
            }
        }

        // Solve each block
        switch (blksolver) {

#if WITH_MUMPS
            case SOLVER_MUMPS:
                {
                    /* use MUMPS direct solver on each block */
                    fasp_mumps_solve(&blk[is], &rhs, &u, mumps[is], 0);
                    break;
                }
#endif

#if WITH_UMFPACK
            case SOLVER_UMFPACK:
                {
                    /* use UMFPACK direct solver on each block */
                    fasp_umfpack_solve(&blk[is], &rhs, &u, numeric[is], 0);
                    break;
                }
#endif
            default:
                /* use iterative solver on each block */
                u.row   = blk[is].row;
                rhs.row = blk[is].row;
                fasp_dvec_set(u.row, &u, 0);
                fasp_solver_dcsr_pvgmres(&blk[is], &rhs, &u, NULL, 1e-8, 1e-20, 100, 20,
                                         1, 0);
        }

        // zero the mask so that everyting is as it was
        for (i = 0; i < nloc; ++i) {
            iblk       = ibl0 + i;
            ki         = jblock[iblk];
            mask[ki]   = 0;
            x->val[ki] = u.val[i];
        }
    }
}

/**
 * \fn void fasp_dcsr_swz_backward (SWZ_data  *swzdata, SWZ_param *swzparam,
 *                                  dvector *x, dvector *b)
 *
 * \brief Schwarz smoother: backward sweep
 *
 * \param swzdata Pointer to the Schwarz data
 * \param swzparam   Pointer to the Schwarz parameter
 * \param x       Pointer to solution vector
 * \param b       Pointer to right hand
 *
 * \author Zheng Li, Chensong Zhang
 * \date   2014/10/5
 */
void fasp_dcsr_swz_backward(SWZ_data* swzdata, SWZ_param* swzparam, dvector* x,
                            dvector* b)
{
    INT i, j, iblk, ki, kj, kij, is, ibl0, ibl1, nloc, iaa, iab;

    // Schwarz partition
    INT      nblk      = swzdata->nblk;
    dCSRmat* blk       = swzdata->blk_data;
    INT*     iblock    = swzdata->iblock;
    INT*     jblock    = swzdata->jblock;
    INT*     mask      = swzdata->mask;
    INT      blksolver = swzparam->SWZ_blksolver;

    // Schwarz data
    dCSRmat A   = swzdata->A;
    INT*    ia  = A.IA;
    INT*    ja  = A.JA;
    REAL*   val = A.val;

    // Local solution and right hand vectors
    dvector rhs = swzdata->rhsloc1;
    dvector u   = swzdata->xloc1;

#if WITH_UMFPACK
    void** numeric = swzdata->numeric;
#endif

#if WITH_MUMPS
    Mumps_data* mumps = swzdata->mumps;
#endif

    for (is = nblk - 1; is >= 0; --is) {
        // Form the right hand of eack block
        ibl0 = iblock[is];
        ibl1 = iblock[is + 1];
        nloc = ibl1 - ibl0;
        for (i = 0; i < nloc; ++i) {
            iblk     = ibl0 + i;
            ki       = jblock[iblk];
            mask[ki] = i + 1;
        }

        for (i = 0; i < nloc; ++i) {
            iblk       = ibl0 + i;
            ki         = jblock[iblk];
            rhs.val[i] = b->val[ki];
            iaa        = ia[ki] - 1;
            iab        = ia[ki + 1] - 1;
            for (kij = iaa; kij < iab; ++kij) {
                kj = ja[kij] - 1;
                j  = mask[kj];
                if (j == 0) {
                    rhs.val[i] -= val[kij] * x->val[kj];
                }
            }
        }

        // Solve each block
        switch (blksolver) {

#if WITH_MUMPS
            case SOLVER_MUMPS:
                {
                    /* use MUMPS direct solver on each block */
                    fasp_mumps_solve(&blk[is], &rhs, &u, mumps[is], 0);
                    break;
                }
#endif

#if WITH_UMFPACK
            case SOLVER_UMFPACK:
                {
                    /* use UMFPACK direct solver on each block */
                    fasp_umfpack_solve(&blk[is], &rhs, &u, numeric[is], 0);
                    break;
                }
#endif
            default:
                /* use iterative solver on each block */
                rhs.row = blk[is].row;
                u.row   = blk[is].row;
                fasp_dvec_set(u.row, &u, 0);
                fasp_solver_dcsr_pvgmres(&blk[is], &rhs, &u, NULL, 1e-8, 1e-20, 100, 20,
                                         1, 0);
        }

        // zero the mask so that everyting is as it was
        for (i = 0; i < nloc; ++i) {
            iblk       = ibl0 + i;
            ki         = jblock[iblk];
            mask[ki]   = 0;
            x->val[ki] = u.val[i];
        }
    }
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static void SWZ_level (const INT inroot, dCSRmat *A, INT *mask, INT *nlvl,
 *                            INT *iblock, INT *jblock, const INT maxlev)
 *
 * \brief Form the level hierarchy of input root node
 *
 * \param inroot  Root node
 * \param A       Pointer to CSR matrix
 * \param mask    Pointer to flag array
 * \param nlvl    The number of levels to expand from root node
 * \param iblock  Pointer to vertices number of each level
 * \param jblock  Pointer to vertices of each level
 * \param maxlev  The maximal number of levels to expand from root node
 *
 * \author Zheng Li
 * \date   2014/09/29
 */
static void SWZ_level(const INT inroot, dCSRmat* A, INT* mask, INT* nlvl, INT* iblock,
                      INT* jblock, const INT maxlev)
{
    INT* ia  = A->IA;
    INT* ja  = A->JA;
    INT  nnz = A->nnz;
    INT  i, j, lvl, lbegin, lvlend, nsize, node;
    INT  jstrt, jstop, nbr, lvsize;

    // This is diagonal
    if (ia[inroot + 1] - ia[inroot] <= 1) {
        lvl                 = 0;
        iblock[lvl]         = 0;
        jblock[iblock[lvl]] = inroot;
        lvl++;
        iblock[lvl] = 1;
    } else {
        // input node as root node (level 0)
        lvl       = 0;
        jblock[0] = inroot;
        lvlend    = 0;
        nsize     = 1;
        // mark root node
        mask[inroot] = 1;

        lvsize = nnz;

        // form the level hierarchy for root node(level1, level2, ... maxlev)
        while (lvsize > 0 && lvl < maxlev) {
            lbegin      = lvlend;
            lvlend      = nsize;
            iblock[lvl] = lbegin;
            lvl++;
            for (i = lbegin; i < lvlend; ++i) {
                node  = jblock[i];
                jstrt = ia[node] - 1;
                jstop = ia[node + 1] - 1;
                for (j = jstrt; j < jstop; ++j) {
                    nbr = ja[j] - 1;
                    if (mask[nbr] == 0) {
                        jblock[nsize] = nbr;
                        mask[nbr]     = lvl;
                        nsize++;
                    }
                }
            }
            lvsize = nsize - lvlend;
        }

        iblock[lvl] = nsize;

        // reset mask array
        for (i = 0; i < nsize; ++i) {
            node       = jblock[i];
            mask[node] = 0;
        }
    }

    *nlvl = lvl;
}

/**
 * \fn static void SWZ_block (SWZ_data *swzdata, const INT nblk,
 *                            const INT *iblock, const INT *jblock, INT *mask)
 *
 * \brief Form Schwarz partition data
 *
 * \param swzdata Pointer to the Schwarz data
 * \param nblk    Number of partitions
 * \param iblock  Pointer to number of vertices on each level
 * \param jblock  Pointer to vertices of each level
 * \param mask    Pointer to flag array
 *
 * \author Zheng Li, Chensong Zhang
 * \date   2014/09/29
 */
static void SWZ_block(SWZ_data* swzdata, const INT nblk, const INT* iblock,
                      const INT* jblock, INT* mask)
{
    INT i, j, iblk, ki, kj, kij, is, ibl0, ibl1, nloc, iaa, iab;
    INT maxbs = 0, count, nnz;

    dCSRmat  A   = swzdata->A;
    dCSRmat* blk = swzdata->blk_data;

    INT*  ia  = A.IA;
    INT*  ja  = A.JA;
    REAL* val = A.val;

    // get maximal block size
    for (is = 0; is < nblk; ++is) {
        ibl0  = iblock[is];
        ibl1  = iblock[is + 1];
        nloc  = ibl1 - ibl0;
        maxbs = MAX(maxbs, nloc);
    }

    swzdata->maxbs = maxbs;

    // allocate memory for each sub_block's right hand
    swzdata->xloc1   = fasp_dvec_create(maxbs);
    swzdata->rhsloc1 = fasp_dvec_create(maxbs);

    for (is = 0; is < nblk; ++is) {
        ibl0  = iblock[is];
        ibl1  = iblock[is + 1];
        nloc  = ibl1 - ibl0;
        count = 0;
        for (i = 0; i < nloc; ++i) {
            iblk = ibl0 + i;
            ki   = jblock[iblk];
            iaa  = ia[ki] - 1;
            iab  = ia[ki + 1] - 1;
            count += iab - iaa;
            mask[ki] = i + 1;
        }

        blk[is]       = fasp_dcsr_create(nloc, nloc, count);
        blk[is].IA[0] = 0;
        nnz           = 0;

        for (i = 0; i < nloc; ++i) {
            iblk = ibl0 + i;
            ki   = jblock[iblk];
            iaa  = ia[ki] - 1;
            iab  = ia[ki + 1] - 1;
            for (kij = iaa; kij < iab; ++kij) {
                kj = ja[kij] - 1;
                j  = mask[kj];
                if (j != 0) {
                    blk[is].JA[nnz]  = j - 1;
                    blk[is].val[nnz] = val[kij];
                    nnz++;
                }
            }
            blk[is].IA[i + 1] = nnz;
        }

        blk[is].nnz = nnz;

        // zero the mask so that everyting is as it was
        for (i = 0; i < nloc; ++i) {
            iblk     = ibl0 + i;
            ki       = jblock[iblk];
            mask[ki] = 0;
        }
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
