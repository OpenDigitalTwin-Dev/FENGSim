/*! \file  PreBSR.c
 *
 *  \brief Preconditioners for dBSRmat matrices
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxArray.c, AuxParam.c, AuxThreads.c, AuxVector.c, BlaSmallMat.c,
 *         BlaSpmvBSR.c, BlaSpmvCSR.c, KrySPcg.c, KrySPvgmres.c, PreMGCycle.c,
 *         and PreMGRecurAMLI.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
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

#include "PreMGUtil.inl"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_precond_dbsr_diag (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zhou Zhiyang, Xiaozhe Hu
 * \date   10/26/2010
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/24/2012
 *
 * \note Works for general nb (Xiaozhe)
 */
void fasp_precond_dbsr_diag(REAL* r, REAL* z, void* data)
{
    precond_diag_bsr* diag = (precond_diag_bsr*)data;
    const INT         nb   = diag->nb;

    switch (nb) {

        case 2:
            fasp_precond_dbsr_diag_nc2(r, z, diag);
            break;
        case 3:
            fasp_precond_dbsr_diag_nc3(r, z, diag);
            break;

        case 4:
            fasp_precond_dbsr_diag_nc4(r, z, diag);
            break;

        case 5:
            fasp_precond_dbsr_diag_nc5(r, z, diag);
            break;

        case 7:
            fasp_precond_dbsr_diag_nc7(r, z, diag);
            break;

        default:
            {
                REAL*     diagptr = diag->diag.val;
                const INT nb2     = nb * nb;
                const INT m       = diag->diag.row / nb2;
                INT       i;

#ifdef _OPENMP
                if (m > OPENMP_HOLDS) {
                    INT myid, mybegin, myend;
                    INT nthreads = fasp_get_num_threads();
#pragma omp parallel for private(myid, mybegin, myend, i)
                    for (myid = 0; myid < nthreads; myid++) {
                        fasp_get_start_end(myid, nthreads, m, &mybegin, &myend);
                        for (i = mybegin; i < myend; ++i) {
                            fasp_blas_smat_mxv(&(diagptr[i * nb2]), &(r[i * nb]),
                                               &(z[i * nb]), nb);
                        }
                    }
                } else {
#endif
                    for (i = 0; i < m; ++i) {
                        fasp_blas_smat_mxv(&(diagptr[i * nb2]), &(r[i * nb]),
                                           &(z[i * nb]), nb);
                    }
#ifdef _OPENMP
                }
#endif
                break;
            }
    }
}

/**
 * \fn void fasp_precond_dbsr_diag_nc2 (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r.
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zhou Zhiyang, Xiaozhe Hu
 * \date   11/18/2011
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/24/2012
 *
 * \note Works for 2-component (Xiaozhe)
 */
void fasp_precond_dbsr_diag_nc2(REAL* r, REAL* z, void* data)
{
    precond_diag_bsr* diag    = (precond_diag_bsr*)data;
    REAL*             diagptr = diag->diag.val;

    INT       i;
    const INT m = diag->diag.row / 4;

#ifdef _OPENMP
    if (m > OPENMP_HOLDS) {
        INT myid, mybegin, myend;
        INT nthreads = fasp_get_num_threads();
#pragma omp parallel for private(myid, mybegin, myend, i)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, m, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                fasp_blas_smat_mxv_nc2(&(diagptr[i * 4]), &(r[i * 2]), &(z[i * 2]));
            }
        }
    } else {
#endif
        for (i = 0; i < m; ++i) {
            fasp_blas_smat_mxv_nc2(&(diagptr[i * 4]), &(r[i * 2]), &(z[i * 2]));
        }
#ifdef _OPENMP
    }
#endif
}

/**
 * \fn void fasp_precond_dbsr_diag_nc3 (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r.
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zhou Zhiyang, Xiaozhe Hu
 * \date 01/06/2011
 *
 * Modified by Chunsheng Feng Xiaoqiang Yue on 05/24/2012
 *
 * \note Works for 3-component (Xiaozhe)
 */
void fasp_precond_dbsr_diag_nc3(REAL* r, REAL* z, void* data)
{
    precond_diag_bsr* diag    = (precond_diag_bsr*)data;
    REAL*             diagptr = diag->diag.val;

    const INT m = diag->diag.row / 9;
    INT       i;

#ifdef _OPENMP
    if (m > OPENMP_HOLDS) {
        INT myid, mybegin, myend;
        INT nthreads = fasp_get_num_threads();
#pragma omp parallel for private(myid, mybegin, myend, i)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, m, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                fasp_blas_smat_mxv_nc3(&(diagptr[i * 9]), &(r[i * 3]), &(z[i * 3]));
            }
        }
    } else {
#endif
        for (i = 0; i < m; ++i) {
            fasp_blas_smat_mxv_nc3(&(diagptr[i * 9]), &(r[i * 3]), &(z[i * 3]));
        }
#ifdef _OPENMP
    }
#endif
}

/**
 * \fn void fasp_precond_dbsr_diag_nc4 (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r.
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zhou Zhiyang, Xiaozhe Hu
 * \date 01/06/2011
 *
 * Modified by Chunsheng Feng Xiaoqiang Yue on 05/24/2012
 *
 * \note Works for 4-component (Li Zhao)
 */
void fasp_precond_dbsr_diag_nc4(REAL* r, REAL* z, void* data)
{
    precond_diag_bsr* diag    = (precond_diag_bsr*)data;
    REAL*             diagptr = diag->diag.val;

    const INT m = diag->diag.row / 16;
    INT       i;

#ifdef _OPENMP
    if (m > OPENMP_HOLDS) {
        INT myid, mybegin, myend;
        INT nthreads = fasp_get_num_threads();
#pragma omp parallel for private(myid, mybegin, myend, i)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, m, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                fasp_blas_smat_mxv_nc4(&(diagptr[i * 16]), &(r[i * 4]), &(z[i * 4]));
            }
        }
    } else {
#endif
        for (i = 0; i < m; ++i) {
            fasp_blas_smat_mxv_nc4(&(diagptr[i * 16]), &(r[i * 4]), &(z[i * 4]));
        }
#ifdef _OPENMP
    }
#endif
}

/**
 * \fn void fasp_precond_dbsr_diag_nc5 (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r.
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zhou Zhiyang, Xiaozhe Hu
 * \date   01/06/2011
 *
 * Modified by Chunsheng Feng, Xiaoqiang Yue on 05/24/2012
 *
 * \note Works for 5-component (Xiaozhe)
 */
void fasp_precond_dbsr_diag_nc5(REAL* r, REAL* z, void* data)
{
    precond_diag_bsr* diag    = (precond_diag_bsr*)data;
    REAL*             diagptr = diag->diag.val;

    const INT m = diag->diag.row / 25;
    INT       i;

#ifdef _OPENMP
    if (m > OPENMP_HOLDS) {
        INT myid, mybegin, myend;
        INT nthreads = fasp_get_num_threads();
#pragma omp parallel for private(myid, mybegin, myend, i)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, m, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                fasp_blas_smat_mxv_nc5(&(diagptr[i * 25]), &(r[i * 5]), &(z[i * 5]));
            }
        }
    } else {
#endif
        for (i = 0; i < m; ++i) {
            fasp_blas_smat_mxv_nc5(&(diagptr[i * 25]), &(r[i * 5]), &(z[i * 5]));
        }
#ifdef _OPENMP
    }
#endif
}

/**
 * \fn void fasp_precond_dbsr_diag_nc7 (REAL *r, REAL *z, void *data)
 *
 * \brief Diagonal preconditioner z=inv(D)*r.
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zhou Zhiyang, Xiaozhe Hu
 * \date   01/06/2011
 *
 * Modified by Chunsheng Feng Xiaoqiang Yue on 05/24/2012
 *
 * \note Works for 7-component (Xiaozhe)
 */
void fasp_precond_dbsr_diag_nc7(REAL* r, REAL* z, void* data)
{
    precond_diag_bsr* diag    = (precond_diag_bsr*)data;
    REAL*             diagptr = diag->diag.val;

    const INT m = diag->diag.row / 49;
    INT       i;

#ifdef _OPENMP
    if (m > OPENMP_HOLDS) {
        INT myid, mybegin, myend;
        INT nthreads = fasp_get_num_threads();
#pragma omp parallel for private(myid, mybegin, myend, i)
        for (myid = 0; myid < nthreads; myid++) {
            fasp_get_start_end(myid, nthreads, m, &mybegin, &myend);
            for (i = mybegin; i < myend; ++i) {
                fasp_blas_smat_mxv_nc7(&(diagptr[i * 49]), &(r[i * 7]), &(z[i * 7]));
            }
        }
    } else {
#endif
        for (i = 0; i < m; ++i) {
            fasp_blas_smat_mxv_nc7(&(diagptr[i * 49]), &(r[i * 7]), &(z[i * 7]));
        }
#ifdef _OPENMP
    }
#endif
}

/**
 * \fn void fasp_precond_dbsr_ilu (REAL *r, REAL *z, void *data)
 *
 * \brief ILU preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Shiquan Zhang, Xiaozhe Hu
 * \date   11/09/2010
 *
 * \note Works for general nb (Xiaozhe)
 */
void fasp_precond_dbsr_ilu(REAL* r, REAL* z, void* data)
{
    const ILU_data* iludata = (ILU_data*)data;
    const INT       m = iludata->row, mm1 = m - 1, mm2 = m - 2, memneed = 2 * m;
    const INT       nb = iludata->nb, nb2 = nb * nb, size = m * nb;

    INT*  ijlu = iludata->ijlu;
    REAL* lu   = iludata->luval;

    INT   ib, ibstart, ibstart1;
    INT   i, j, jj, begin_row, end_row;
    REAL *zz, *zr, *mult;

    if (iludata->nwork < memneed) {
        printf("### ERROR: Need %d memory, only %d available!\n", memneed,
               iludata->nwork);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    zz   = iludata->work;
    zr   = zz + size;
    mult = zr + size;

    memcpy(zr, r, size * sizeof(REAL));

    switch (nb) {

        case 1:

            // forward sweep: solve unit lower matrix equation L*zz=zr
            zz[0] = zr[0];
            for (i = 1; i <= mm1; ++i) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                for (j = begin_row; j <= end_row; ++j) {
                    jj = ijlu[j];
                    if (jj < i)
                        zr[i] -= lu[j] * zz[jj];
                    else
                        break;
                }
                zz[i] = zr[i];
            }

            // backward sweep: solve upper matrix equation U*z=zz
            z[mm1] = zz[mm1] * lu[mm1];
            for (i = mm2; i >= 0; i--) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                for (j = end_row; j >= begin_row; j--) {
                    jj = ijlu[j];
                    if (jj > i)
                        zz[i] -= lu[j] * z[jj];
                    else
                        break;
                }
                z[i] = zz[i] * lu[i];
            }

            break; // end (if nb==1)

        case 3:

            // forward sweep: solve unit lower matrix equation L*zz=zr
            zz[0] = zr[0];
            zz[1] = zr[1];
            zz[2] = zr[2];

            for (i = 1; i <= mm1; ++i) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb;
                for (j = begin_row; j <= end_row; ++j) {
                    jj = ijlu[j];
                    if (jj < i) {
                        fasp_blas_smat_mxv_nc3(&(lu[j * nb2]), &(zz[jj * nb]), mult);
                        for (ib = 0; ib < nb; ++ib) zr[ibstart + ib] -= mult[ib];
                    } else
                        break;
                }

                zz[ibstart]     = zr[ibstart];
                zz[ibstart + 1] = zr[ibstart + 1];
                zz[ibstart + 2] = zr[ibstart + 2];
            }

            // backward sweep: solve upper matrix equation U*z=zz
            ibstart  = mm1 * nb2;
            ibstart1 = mm1 * nb;
            fasp_blas_smat_mxv_nc3(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]));

            for (i = mm2; i >= 0; i--) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb2;
                ibstart1  = i * nb;
                for (j = end_row; j >= begin_row; j--) {
                    jj = ijlu[j];
                    if (jj > i) {
                        fasp_blas_smat_mxv_nc3(&(lu[j * nb2]), &(z[jj * nb]), mult);
                        for (ib = 0; ib < nb; ++ib) zz[ibstart1 + ib] -= mult[ib];
                    }

                    else
                        break;
                }

                fasp_blas_smat_mxv_nc3(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]));
            }

            break; // end (if nb=3)

        case 5:

            // forward sweep: solve unit lower matrix equation L*zz=zr
            fasp_darray_cp(nb, &(zr[0]), &(zz[0]));

            for (i = 1; i <= mm1; ++i) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb;
                for (j = begin_row; j <= end_row; ++j) {
                    jj = ijlu[j];
                    if (jj < i) {
                        fasp_blas_smat_mxv_nc5(&(lu[j * nb2]), &(zz[jj * nb]), mult);
                        for (ib = 0; ib < nb; ++ib) zr[ibstart + ib] -= mult[ib];
                    } else
                        break;
                }

                fasp_darray_cp(nb, &(zr[ibstart]), &(zz[ibstart]));
            }

            // backward sweep: solve upper matrix equation U*z=zz
            ibstart  = mm1 * nb2;
            ibstart1 = mm1 * nb;
            fasp_blas_smat_mxv_nc5(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]));

            for (i = mm2; i >= 0; i--) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb2;
                ibstart1  = i * nb;
                for (j = end_row; j >= begin_row; j--) {
                    jj = ijlu[j];
                    if (jj > i) {
                        fasp_blas_smat_mxv_nc5(&(lu[j * nb2]), &(z[jj * nb]), mult);
                        for (ib = 0; ib < nb; ++ib) zz[ibstart1 + ib] -= mult[ib];
                    }

                    else
                        break;
                }

                fasp_blas_smat_mxv_nc5(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]));
            }

            break; // end (if nb==5)

        case 7:

            // forward sweep: solve unit lower matrix equation L*zz=zr
            fasp_darray_cp(nb, &(zr[0]), &(zz[0]));

            for (i = 1; i <= mm1; ++i) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb;
                for (j = begin_row; j <= end_row; ++j) {
                    jj = ijlu[j];
                    if (jj < i) {
                        fasp_blas_smat_mxv_nc7(&(lu[j * nb2]), &(zz[jj * nb]), mult);
                        for (ib = 0; ib < nb; ++ib) zr[ibstart + ib] -= mult[ib];
                    } else
                        break;
                }

                fasp_darray_cp(nb, &(zr[ibstart]), &(zz[ibstart]));
            }

            // backward sweep: solve upper matrix equation U*z=zz
            ibstart  = mm1 * nb2;
            ibstart1 = mm1 * nb;
            fasp_blas_smat_mxv_nc7(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]));

            for (i = mm2; i >= 0; i--) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb2;
                ibstart1  = i * nb;
                for (j = end_row; j >= begin_row; j--) {
                    jj = ijlu[j];
                    if (jj > i) {
                        fasp_blas_smat_mxv_nc7(&(lu[j * nb2]), &(z[jj * nb]), mult);
                        for (ib = 0; ib < nb; ++ib) zz[ibstart1 + ib] -= mult[ib];
                    }

                    else
                        break;
                }

                fasp_blas_smat_mxv_nc7(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]));
            }

            break; // end (if nb==7)

        default:

            // forward sweep: solve unit lower matrix equation L*zz=zr
            fasp_darray_cp(nb, &(zr[0]), &(zz[0]));

            for (i = 1; i <= mm1; ++i) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb;
                for (j = begin_row; j <= end_row; ++j) {
                    jj = ijlu[j];
                    if (jj < i) {
                        fasp_blas_smat_mxv(&(lu[j * nb2]), &(zz[jj * nb]), mult, nb);
                        for (ib = 0; ib < nb; ++ib) zr[ibstart + ib] -= mult[ib];
                    } else
                        break;
                }

                fasp_darray_cp(nb, &(zr[ibstart]), &(zz[ibstart]));
            }

            // backward sweep: solve upper matrix equation U*z=zz
            ibstart  = mm1 * nb2;
            ibstart1 = mm1 * nb;
            fasp_blas_smat_mxv(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]), nb);

            for (i = mm2; i >= 0; i--) {
                begin_row = ijlu[i];
                end_row   = ijlu[i + 1] - 1;
                ibstart   = i * nb2;
                ibstart1  = i * nb;
                for (j = end_row; j >= begin_row; j--) {
                    jj = ijlu[j];
                    if (jj > i) {
                        fasp_blas_smat_mxv(&(lu[j * nb2]), &(z[jj * nb]), mult, nb);
                        for (ib = 0; ib < nb; ++ib) zz[ibstart1 + ib] -= mult[ib];
                    }

                    else
                        break;
                }

                fasp_blas_smat_mxv(&(lu[ibstart]), &(zz[ibstart1]), &(z[ibstart1]), nb);
            }

            break; // end everything else
    }

    return;
}

/**
 * \fn void fasp_precond_dbsr_ilu_mc_omp (REAL *r, REAL *z, void *data)
 *
 * \brief Multi-thread Parallel ILU preconditioner based on graph coloring
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zheng Li
 * \date   12/04/2016
 *
 * \note Only works for nb 1, 2, and 3 (Zheng)
 */
void fasp_precond_dbsr_ilu_mc_omp(REAL* r, REAL* z, void* data)
{
#ifdef _OPENMP
    const ILU_data* iludata = (ILU_data*)data;
    const INT       m = iludata->row, memneed = 2 * m;
    const INT       nb = iludata->nb, nb2 = nb * nb, size = m * nb;

    INT*  ijlu    = iludata->ijlu;
    REAL* lu      = iludata->luval;
    INT   ncolors = iludata->nlevL;
    INT*  ic      = iludata->ilevL;

    INT   ib, ibstart, ibstart1;
    INT   i, j, jj, k, begin_row, end_row;
    REAL *zz, *zr, *mult;

    if (iludata->nwork < memneed) {
        printf("### ERROR: Need %d memory, only %d available!\n", memneed,
               iludata->nwork);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    zz = iludata->work;
    zr = zz + size;

    memcpy(zr, r, size * sizeof(REAL));

    switch (nb) {

        case 1:
            // forward sweep: solve unit lower matrix equation L*zz=zr
            for (k = 0; k < ncolors; ++k) {
#pragma omp parallel for private(i, begin_row, end_row, j, jj)
                for (i = ic[k]; i < ic[k + 1]; ++i) {
                    begin_row = ijlu[i];
                    end_row   = ijlu[i + 1] - 1;
                    for (j = begin_row; j <= end_row; ++j) {
                        jj = ijlu[j];
                        if (jj < i)
                            zr[i] -= lu[j] * zz[jj];
                        else
                            break;
                    }
                    zz[i] = zr[i];
                }
            }
            // backward sweep: solve upper matrix equation U*z=zz
            for (k = ncolors - 1; k >= 0; k--) {
#pragma omp parallel for private(i, begin_row, end_row, j, jj)
                for (i = ic[k + 1] - 1; i >= ic[k]; i--) {
                    begin_row = ijlu[i];
                    end_row   = ijlu[i + 1] - 1;
                    for (j = end_row; j >= begin_row; j--) {
                        jj = ijlu[j];
                        if (jj > i)
                            zz[i] -= lu[j] * z[jj];
                        else
                            break;
                    }
                    z[i] = zz[i] * lu[i];
                }
            }

            break; // end (if nb==1)

        case 2:

            for (k = 0; k < ncolors; ++k) {
#pragma omp parallel private(i, begin_row, end_row, ibstart, j, jj, ib, mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (i = ic[k]; i < ic[k + 1]; ++i) {
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb;
                        for (j = begin_row; j <= end_row; ++j) {
                            jj = ijlu[j];
                            if (jj < i) {
                                fasp_blas_smat_mxv_nc2(&(lu[j * nb2]), &(zz[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zr[ibstart + ib] -= mult[ib];
                            } else
                                break;
                        }

                        zz[ibstart]     = zr[ibstart];
                        zz[ibstart + 1] = zr[ibstart + 1];
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            for (k = ncolors - 1; k >= 0; k--) {
#pragma omp parallel private(i, begin_row, end_row, ibstart, ibstart1, j, jj, ib, mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (i = ic[k + 1] - 1; i >= ic[k]; i--) {
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb2;
                        ibstart1  = i * nb;
                        for (j = end_row; j >= begin_row; j--) {
                            jj = ijlu[j];
                            if (jj > i) {
                                fasp_blas_smat_mxv_nc2(&(lu[j * nb2]), &(z[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zz[ibstart1 + ib] -= mult[ib];
                            }

                            else
                                break;
                        }

                        fasp_blas_smat_mxv_nc2(&(lu[ibstart]), &(zz[ibstart1]),
                                               &(z[ibstart1]));
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            break; // end (if nb=2)
        case 3:

            for (k = 0; k < ncolors; ++k) {
#pragma omp parallel private(i, begin_row, end_row, ibstart, j, jj, ib, mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (i = ic[k]; i < ic[k + 1]; ++i) {
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb;
                        for (j = begin_row; j <= end_row; ++j) {
                            jj = ijlu[j];
                            if (jj < i) {
                                fasp_blas_smat_mxv_nc3(&(lu[j * nb2]), &(zz[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zr[ibstart + ib] -= mult[ib];
                            } else
                                break;
                        }

                        zz[ibstart]     = zr[ibstart];
                        zz[ibstart + 1] = zr[ibstart + 1];
                        zz[ibstart + 2] = zr[ibstart + 2];
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            for (k = ncolors - 1; k >= 0; k--) {
#pragma omp parallel private(i, begin_row, end_row, ibstart, ibstart1, j, jj, ib, mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (i = ic[k + 1] - 1; i >= ic[k]; i--) {
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb2;
                        ibstart1  = i * nb;
                        for (j = end_row; j >= begin_row; j--) {
                            jj = ijlu[j];
                            if (jj > i) {
                                fasp_blas_smat_mxv_nc3(&(lu[j * nb2]), &(z[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zz[ibstart1 + ib] -= mult[ib];
                            }

                            else
                                break;
                        }

                        fasp_blas_smat_mxv_nc3(&(lu[ibstart]), &(zz[ibstart1]),
                                               &(z[ibstart1]));
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            break; // end (if nb=3)

        default:
            {
                if (nb > 3) {
                    printf("### ERROR: Multi-thread Parallel ILU for %d components \
                       has not yet been implemented!!!",
                           nb);
                    fasp_chkerr(ERROR_UNKNOWN, __FUNCTION__);
                }
                break;
            }
    }

    return;
#endif
}

/**
 * \fn void fasp_precond_dbsr_ilu_ls_omp (REAL *r, REAL *z, void *data)
 *
 * \brief Multi-thread Parallel ILU preconditioner based on level schedule strategy
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Zheng Li
 * \date   12/04/2016
 *
 * \note Only works for nb 1, 2, and 3 (Zheng)
 * \note works forall nb = 1, 2, ..., added by Li Zhao, 09/19/2022
 */
void fasp_precond_dbsr_ilu_ls_omp(REAL* r, REAL* z, void* data)
{
#ifdef _OPENMP
    const ILU_data* iludata = (ILU_data*)data;
    const INT       m = iludata->row, memneed = 2 * m;
    const INT       nb = iludata->nb, nb2 = nb * nb, size = m * nb;

    INT*  ijlu  = iludata->ijlu;
    REAL* lu    = iludata->luval;
    INT   nlevL = iludata->nlevL;
    INT*  ilevL = iludata->ilevL;
    INT*  jlevL = iludata->jlevL;
    INT   nlevU = iludata->nlevU;
    INT*  ilevU = iludata->ilevU;
    INT*  jlevU = iludata->jlevU;

    INT   ib, ibstart, ibstart1;
    INT   i, ii, j, jj, k, begin_row, end_row;
    REAL *zz, *zr, *mult;

    if (iludata->nwork < memneed) {
        printf("### ERROR: Need %d memory, only %d available!\n", memneed,
               iludata->nwork);
        fasp_chkerr(ERROR_ALLOC_MEM, __FUNCTION__);
    }

    zz = iludata->work;
    zr = zz + size;
    // mult = zr + size;

    memcpy(zr, r, size * sizeof(REAL));

    switch (nb) {

        case 1:
            // forward sweep: solve unit lower matrix equation L*zz=zr
            for (k = 0; k < nlevL; ++k) {
#pragma omp parallel for private(i, ii, begin_row, end_row, j, jj)
                for (ii = ilevL[k]; ii < ilevL[k + 1]; ++ii) {
                    i         = jlevL[ii];
                    begin_row = ijlu[i];
                    end_row   = ijlu[i + 1] - 1;
                    for (j = begin_row; j <= end_row; ++j) {
                        jj = ijlu[j];
                        if (jj < i)
                            zr[i] -= lu[j] * zz[jj];
                        else
                            break;
                    }
                    zz[i] = zr[i];
                }
            }
            // backward sweep: solve upper matrix equation U*z=zz
            for (k = 0; k < nlevU; k++) {
#pragma omp parallel for private(i, ii, begin_row, end_row, j, jj)
                for (ii = ilevU[k + 1] - 1; ii >= ilevU[k]; ii--) {
                    i         = jlevU[ii];
                    begin_row = ijlu[i];
                    end_row   = ijlu[i + 1] - 1;
                    for (j = end_row; j >= begin_row; j--) {
                        jj = ijlu[j];
                        if (jj > i)
                            zz[i] -= lu[j] * z[jj];
                        else
                            break;
                    }
                    z[i] = zz[i] * lu[i];
                }
            }

            break; // end (if nb==1)

        case 2:

            for (k = 0; k < nlevL; ++k) {
#pragma omp parallel private(i, ii, begin_row, end_row, ibstart, j, jj, ib, mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (ii = ilevL[k]; ii < ilevL[k + 1]; ++ii) {
                        i         = jlevL[ii];
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb;
                        for (j = begin_row; j <= end_row; ++j) {
                            jj = ijlu[j];
                            if (jj < i) {
                                fasp_blas_smat_mxv_nc2(&(lu[j * nb2]), &(zz[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zr[ibstart + ib] -= mult[ib];
                            } else
                                break;
                        }

                        zz[ibstart]     = zr[ibstart];
                        zz[ibstart + 1] = zr[ibstart + 1];
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            for (k = 0; k < nlevU; k++) {
#pragma omp parallel private(i, ii, begin_row, end_row, ibstart, ibstart1, j, jj, ib,  \
                                 mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (ii = ilevU[k + 1] - 1; ii >= ilevU[k]; ii--) {
                        i         = jlevU[ii];
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb2;
                        ibstart1  = i * nb;
                        for (j = end_row; j >= begin_row; j--) {
                            jj = ijlu[j];
                            if (jj > i) {
                                fasp_blas_smat_mxv_nc2(&(lu[j * nb2]), &(z[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zz[ibstart1 + ib] -= mult[ib];
                            }

                            else
                                break;
                        }

                        fasp_blas_smat_mxv_nc2(&(lu[ibstart]), &(zz[ibstart1]),
                                               &(z[ibstart1]));
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            break; // end (if nb=2)
        case 3:

            for (k = 0; k < nlevL; ++k) {
#pragma omp parallel private(i, ii, begin_row, end_row, ibstart, j, jj, ib, mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (ii = ilevL[k]; ii < ilevL[k + 1]; ++ii) {
                        i         = jlevL[ii];
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb;
                        for (j = begin_row; j <= end_row; ++j) {
                            jj = ijlu[j];
                            if (jj < i) {
                                fasp_blas_smat_mxv_nc3(&(lu[j * nb2]), &(zz[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zr[ibstart + ib] -= mult[ib];
                            } else
                                break;
                        }

                        zz[ibstart]     = zr[ibstart];
                        zz[ibstart + 1] = zr[ibstart + 1];
                        zz[ibstart + 2] = zr[ibstart + 2];
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            for (k = 0; k < nlevU; k++) {
#pragma omp parallel private(i, ii, begin_row, end_row, ibstart, ibstart1, j, jj, ib,  \
                                 mult)
                {
                    mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                    for (ii = ilevU[k + 1] - 1; ii >= ilevU[k]; ii--) {
                        i         = jlevU[ii];
                        begin_row = ijlu[i];
                        end_row   = ijlu[i + 1] - 1;
                        ibstart   = i * nb2;
                        ibstart1  = i * nb;
                        for (j = end_row; j >= begin_row; j--) {
                            jj = ijlu[j];
                            if (jj > i) {
                                fasp_blas_smat_mxv_nc3(&(lu[j * nb2]), &(z[jj * nb]),
                                                       mult);
                                for (ib = 0; ib < nb; ++ib)
                                    zz[ibstart1 + ib] -= mult[ib];
                            }

                            else
                                break;
                        }

                        fasp_blas_smat_mxv_nc3(&(lu[ibstart]), &(zz[ibstart1]),
                                               &(z[ibstart1]));
                    }

                    fasp_mem_free(mult);
                    mult = NULL;
                }
            }

            break; // end (if nb=3)

        default:
            {

                for (k = 0; k < nlevL; ++k) {
#pragma omp parallel private(i, ii, begin_row, end_row, ibstart, j, jj, ib, mult)
                    {
                        mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                        for (ii = ilevL[k]; ii < ilevL[k + 1]; ++ii) {
                            i         = jlevL[ii];
                            begin_row = ijlu[i];
                            end_row   = ijlu[i + 1] - 1;
                            ibstart   = i * nb;
                            for (j = begin_row; j <= end_row; ++j) {
                                jj = ijlu[j];
                                if (jj < i) {
                                    fasp_blas_smat_mxv(&(lu[j * nb2]), &(zz[jj * nb]),
                                                       mult, nb);
                                    for (ib = 0; ib < nb; ++ib)
                                        zr[ibstart + ib] -= mult[ib];
                                } else
                                    break;
                            }

                            for (j = 0; j < nb; j++)
                                zz[ibstart + j] =
                                    zr[ibstart + j]; // Li Zhao, 09/19/2022
                        }

                        fasp_mem_free(mult);
                        mult = NULL;
                    }
                }

                for (k = 0; k < nlevU; k++) {
#pragma omp parallel private(i, ii, begin_row, end_row, ibstart, ibstart1, j, jj, ib,  \
                                 mult)
                    {
                        mult = (REAL*)fasp_mem_calloc(nb, sizeof(REAL));
#pragma omp for
                        for (ii = ilevU[k + 1] - 1; ii >= ilevU[k]; ii--) {
                            i         = jlevU[ii];
                            begin_row = ijlu[i];
                            end_row   = ijlu[i + 1] - 1;
                            ibstart   = i * nb2;
                            ibstart1  = i * nb;
                            for (j = end_row; j >= begin_row; j--) {
                                jj = ijlu[j];
                                if (jj > i) {
                                    fasp_blas_smat_mxv(&(lu[j * nb2]), &(z[jj * nb]),
                                                       mult, nb);
                                    for (ib = 0; ib < nb; ++ib)
                                        zz[ibstart1 + ib] -= mult[ib];
                                }

                                else
                                    break;
                            }

                            fasp_blas_smat_mxv(&(lu[ibstart]), &(zz[ibstart1]),
                                               &(z[ibstart1]), nb);
                        }

                        fasp_mem_free(mult);
                        mult = NULL;
                    }
                }

                break;

                /*
                if (nb > 3) {
                    printf("### ERROR: Multi-thread Parallel ILU for %d components \
                           has not yet been implemented!!!", nb);
                    fasp_chkerr(ERROR_UNKNOWN, __FUNCTION__);
                }
                break;
                */
            }
    }

    return;
#endif
}

/**
 * \fn void fasp_precond_dbsr_amg (REAL *r, REAL *z, void *data)
 *
 * \brief AMG preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   08/07/2011
 */
void fasp_precond_dbsr_amg(REAL* r, REAL* z, void* data)
{
    precond_data_bsr* predata = (precond_data_bsr*)data;
    const INT         row     = predata->mgl_data[0].A.ROW;
    const INT         nb      = predata->mgl_data[0].A.nb;
    const INT         maxit   = predata->maxit;
    const INT         m       = row * nb;

    INT i;

    AMG_param amgparam;
    fasp_param_amg_init(&amgparam);
    amgparam.cycle_type       = predata->cycle_type;
    amgparam.smoother         = predata->smoother;
    amgparam.smooth_order     = predata->smooth_order;
    amgparam.presmooth_iter   = predata->presmooth_iter;
    amgparam.postsmooth_iter  = predata->postsmooth_iter;
    amgparam.relaxation       = predata->relaxation;
    amgparam.coarse_scaling   = predata->coarse_scaling;
    amgparam.tentative_smooth = predata->tentative_smooth;
    amgparam.ILU_levels       = predata->mgl_data->ILU_levels;

    AMG_data_bsr* mgl = predata->mgl_data;
    mgl->b.row        = m;
    fasp_darray_cp(m, r, mgl->b.val); // residual is an input
    mgl->x.row = m;
    fasp_dvec_set(m, &mgl->x, 0.0);

    for (i = maxit; i--;) fasp_solver_mgcycle_bsr(mgl, &amgparam);

    fasp_darray_cp(m, mgl->x.val, z);
}

/**
 * \fn void fasp_precond_dbsr_amg_nk (REAL *r, REAL *z, void *data)
 *
 * \brief AMG with extra near kernel solve preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   05/26/2014
 */
void fasp_precond_dbsr_amg_nk(REAL* r, REAL* z, void* data)
{
    precond_data_bsr* predata = (precond_data_bsr*)data;
    const INT         row     = predata->mgl_data[0].A.ROW;
    const INT         nb      = predata->mgl_data[0].A.nb;
    const INT         maxit   = predata->maxit;
    const INT         m       = row * nb;

    INT i;

    dCSRmat* A_nk = predata->A_nk;
    dCSRmat* P_nk = predata->P_nk;
    dCSRmat* R_nk = predata->R_nk;

    fasp_darray_set(m, z, 0.0);

    // local variables
    dvector r_nk, z_nk;
    fasp_dvec_alloc(A_nk->row, &r_nk);
    fasp_dvec_alloc(A_nk->row, &z_nk);

    //----------------------
    // extra kernel solve
    //----------------------
    // r_nk = R_nk*r
    fasp_blas_dcsr_mxv(R_nk, r, r_nk.val);

    // z_nk = A_nk^{-1}*r_nk
#if WITH_UMFPACK // use UMFPACK directly
    fasp_solver_umfpack(A_nk, &r_nk, &z_nk, 0);
#else
    fasp_coarse_itsolver(A_nk, &r_nk, &z_nk, 1e-12, 0);
#endif

    // z = z + P_nk*z_nk;
    fasp_blas_dcsr_aAxpy(1.0, P_nk, z_nk.val, z);

    //----------------------
    // AMG solve
    //----------------------
    AMG_param amgparam;
    fasp_param_amg_init(&amgparam);
    amgparam.cycle_type       = predata->cycle_type;
    amgparam.smoother         = predata->smoother;
    amgparam.smooth_order     = predata->smooth_order;
    amgparam.presmooth_iter   = predata->presmooth_iter;
    amgparam.postsmooth_iter  = predata->postsmooth_iter;
    amgparam.relaxation       = predata->relaxation;
    amgparam.coarse_scaling   = predata->coarse_scaling;
    amgparam.tentative_smooth = predata->tentative_smooth;
    amgparam.ILU_levels       = predata->mgl_data->ILU_levels;

    AMG_data_bsr* mgl = predata->mgl_data;
    mgl->b.row        = m;
    fasp_darray_cp(m, r, mgl->b.val); // residual is an input
    mgl->x.row = m;                   // fasp_dvec_set(m,&mgl->x,0.0);
    fasp_darray_cp(m, z, mgl->x.val);

    for (i = maxit; i--;) fasp_solver_mgcycle_bsr(mgl, &amgparam);

    fasp_darray_cp(m, mgl->x.val, z);

    //----------------------
    // extra kernel solve
    //----------------------
    // r = r - A*z
    fasp_blas_dbsr_aAxpy(-1.0, &(predata->mgl_data[0].A), z, mgl->b.val);

    // r_nk = R_nk*r
    fasp_blas_dcsr_mxv(R_nk, mgl->b.val, r_nk.val);

    // z_nk = A_nk^{-1}*r_nk
#if WITH_UMFPACK // use UMFPACK directly
    fasp_solver_umfpack(A_nk, &r_nk, &z_nk, 0);
#else
    fasp_coarse_itsolver(A_nk, &r_nk, &z_nk, 1e-12, 0);
#endif

    // z = z + P_nk*z_nk;
    fasp_blas_dcsr_aAxpy(1.0, P_nk, z_nk.val, z);
}

double PreSmoother_time_zl  = 0.0;
double PostSmoother_time_zl = 0.0;
double Krylov_time_zl       = 0.0;
double Coarsen_time_zl      = 0.0;
double AMLI_cycle_time_zl   = 0.0;

/**
 * \fn void fasp_precond_dbsr_namli (REAL *r, REAL *z, void *data)
 *
 * \brief Nonlinear AMLI-cycle AMG preconditioner
 *
 * \param r     Pointer to the vector needs preconditioning
 * \param z     Pointer to preconditioned vector
 * \param data  Pointer to precondition data
 *
 * \author Xiaozhe Hu
 * \date   02/06/2012
 */
void fasp_precond_dbsr_namli(REAL* r, REAL* z, void* data)
{
    precond_data_bsr* pcdata     = (precond_data_bsr*)data;
    const INT         row        = pcdata->mgl_data[0].A.ROW;
    const INT         nb         = pcdata->mgl_data[0].A.nb;
    const INT         maxit      = pcdata->maxit;
    const SHORT       num_levels = pcdata->max_levels;
    const INT         m          = row * nb;

    INT i;

    AMG_param amgparam;
    fasp_param_amg_init(&amgparam);
    fasp_param_precbsr_to_amg(&amgparam, pcdata);

    AMG_data_bsr* mgl = pcdata->mgl_data;
    mgl->b.row        = m;
    fasp_darray_cp(m, r, mgl->b.val); // residual is an input
    mgl->x.row = m;
    fasp_dvec_set(m, &mgl->x, 0.0);

    // REAL start_time, end_time; //! zhaoli
    // fasp_gettime(&start_time); //! zhaoli

    for (i = maxit; i--;) fasp_solver_namli_bsr(mgl, &amgparam, 0, num_levels);

    // fasp_gettime(&end_time);                                         //! zhaoli
    // AMLI_cycle_time_zl += end_time - start_time;
    // printf("nonlinear AMLI-cycle time: %.4f\n", AMLI_cycle_time_zl); //! zhaoli
    // printf("PreSmoother_time_zl: %.4f\n", PreSmoother_time_zl);      //! zhaoli
    // printf("PostSmoother_time_zl: %.4f\n", PostSmoother_time_zl);    //! zhaoli
    // printf("Krylov_time_zl: %.4f\n", Krylov_time_zl);                //! zhaoli
    // printf("Coarsen_time_zl: %.4f\n", Coarsen_time_zl);              //! zhaoli

    fasp_darray_cp(m, mgl->x.val, z);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
