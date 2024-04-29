/*! \file  PreAMGCoarsenCR.c
 *
 *  \brief Coarsening with Brannick-Falgout strategy.
 *
 *  \note  This file contains Level-4 (Pre) functions. It requires:
 *         AuxMemory.c, AuxThreads.c, and ItrSmootherCSRcr.c
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 *
 *  // TODO: Not completed! --Chensong
 */

#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"

#define AMG_COARSEN_CR

/*---------------------------------*/
/*--  Declare Private Functions  --*/
/*---------------------------------*/

#include "PreAMGUtil.inl"

static INT GraphAdd(Link *, INT *, INT *, INT, INT);
static INT GraphRemove(Link *, INT *, INT *, INT );
static INT indset(INT, INT, INT, INT *, INT *, INT, INT *, REAL *);

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn INT fasp_amg_coarsening_cr (const INT i_0, const INT i_n, dCSRmat *A,
 *                                 ivector *vertices, AMG_param *param)
 *
 * \brief CR coarsening
 *
 * \param i_0       Starting index
 * \param i_n       Ending index
 * \param A         Pointer to dCSRmat: the coefficient matrix (index starts from 0)
 * \param vertices  Pointer to CF, 0: Fpt (current level) or 1: Cpt
 * \param param     Pointer to AMG_param: AMG parameters
 *
 * \return          Number of coarse level points
 *
 * \author James Brannick
 * \date   04/21/2010
 *
 * \note vertices = 0: fine; 1: coarse; 2: isolated or special
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/14/2012
 */
INT fasp_amg_coarsening_cr (const INT   i_0,
                            const INT   i_n,
                            dCSRmat    *A,
                            ivector    *vertices,
                            AMG_param  *param)
{    
    const SHORT prtlvl = param->print_level;
    
    // local variables
    INT   cand=0,cpt=-1,fpt=1;        // internal labeling
    INT   nc,ns=1;                    // # cpts, # stages
    INT   i,j,in1,nu=3,num1 = nu-1;   // nu is number of cr sweeps
    INT  *cf=NULL,*ia=NULL,*ja=NULL;
    
    REAL  temp0=0.0e0,temp1=0.0e0,rho=0.0e0,tg=8.0e-01;
    REAL *a=NULL;
    
    /* WORKING MEMORY -- b not needed, remove later */
    REAL *b=NULL,*u=NULL,*ma=NULL;

    ia = A->IA;
    ja = A->JA;
    a  = A->val;

    if (i_0 == 0) {
        in1 = i_n+1;
    } else {
        in1 = i_n;
    }
    
    /* CF, RHS, INITIAL GUESS, and MEAS. ARRAY */
    cf = (INT*)fasp_mem_calloc(in1,sizeof(INT));
    b  = (REAL*)fasp_mem_calloc(in1,sizeof(REAL));
    u  = (REAL*)fasp_mem_calloc(in1,sizeof(REAL));
    ma = (REAL*)fasp_mem_calloc(in1,sizeof(REAL));
    
#ifdef _OPENMP
#pragma omp parallel for if(i_n>OPENMP_HOLDS) 
#endif
    for(i=i_0;i<=i_n;++i) {
        b[i] = 0.0e0; // ZERO RHS
        cf[i] = fpt;  // ALL FPTS
    }
    
    /** CR STAGES */
    while (TRUE) {

        nc = 0;
#ifdef _OPENMP
#pragma omp parallel for if(i_n>OPENMP_HOLDS) 
#endif
        for(i=i_0; i<=i_n; ++i) {
            if (cf[i] == cpt) {
                nc += 1;
                u[i] = 0.0e0;
            } else {
                u[i] = 1.0e0;
            }
        }

        for (i=i_0;i<=nu;++i) {

            if (i == num1)
                for (j = i_0; j<= i_n; ++j) {
                    if (cf[j] == fpt) {
                        temp0 += u[j]*u[j];
                    }
                }
            fasp_smoother_dcsr_gscr(fpt,i_n,u,ia,ja,a,b,1,cf);
        }

#ifdef _OPENMP
#pragma omp parallel for reduction(+:temp1) if(i_n>OPENMP_HOLDS)
#endif
        for (i = i_0; i<= i_n; ++i) {
            if (cf[i] == fpt) {
                temp1 += u[i]*u[i];
            }
        }
        rho = sqrt(temp1)/sqrt(temp0);

        if ( prtlvl > PRINT_MIN ) printf("rho=%2.13lf\n",rho);

        if ( rho > tg ) {
            /* FORM CAND. SET & COMPUTE IND SET */
            temp0 = 0.0e0;

            for (i = i_0; i<= i_n; ++i) {
                temp1 = fabs(u[i]);
                if (cf[i] == cpt && temp1 > 0.0e0) {
                    temp0 = temp1; // max.
                }
            }
            if (ns == 1) {
                temp1 = pow(0.3, nu);
            } else {
                temp1 = 0.5;
            }

#ifdef _OPENMP
#pragma omp parallel for if(i_n>OPENMP_HOLDS)
#endif
            for (i = i_0; i <= i_n; ++i) {
                if (cf[i] == fpt && fabs(u[i])/temp0 > temp1 && ia[i+1]-ia[i] > 1)
                    cf[i] = cand;
            }
            temp1 = 0.0e0;
            indset(cand,cpt,fpt,ia,ja,i_n,cf,ma);
            ns++;
        }
        else {
            /* back to fasp labeling */

#ifdef _OPENMP
#pragma omp parallel for if(i_n>OPENMP_HOLDS)
#endif
            for (i = i_0; i<= i_n; ++i) {
                if (cf[i] == cpt) {
                    cf[i] = 1; // cpt
                } else {
                    cf[i] = 0; // fpt
                }
                // printf("cf[%i] = %i\n",i,cf[i]);
            }
            vertices->row=i_n;
            if ( prtlvl >= PRINT_MORE ) printf("vertices = %i\n",vertices->row);
            vertices->val= cf;
            if ( prtlvl >= PRINT_MORE ) printf("nc=%i\n",nc);
            break;
        }
    }
    
    fasp_mem_free(u);   u  = NULL;
    fasp_mem_free(b);   b  = NULL;
    fasp_mem_free(ma);  ma = NULL;
    
    return nc;
}

/*---------------------------------*/
/*--      Private Functions      --*/
/*---------------------------------*/

/**
 * \fn static INT GraphAdd (Link *list, INT *head, INT *tail, INT index, INT istack)
 * \brief graphadd
 */
static INT GraphAdd (Link  *list,
                     INT   *head,
                     INT   *tail,
                     INT    index,
                     INT    istack)
{
    INT prev = tail[-istack];
    
    list[index].prev = prev;
    if (prev < 0)
        head[-istack] = index;
    else
        list[prev].next = index;
    list[index].next = -istack;
    tail[-istack] = index;
    
    return 0;
}

/**
 * \fn static INT GraphRemove (Link *list, INT *head, INT *tail, INT index)
 * \brief GraphRemove
 */
static INT GraphRemove (Link   *list,
                        INT    *head,
                        INT    *tail,
                        INT     index)
{
    INT prev = list[index].prev;
    INT next = list[index].next;
    
    if (prev < 0)
        head[prev] = next;
    else
        list[prev].next = next;
    if (next < 0)
        tail[next] = prev;
    else
        list[next].prev = prev;
    
    return 0;
}

/**
 * \fn static INT indset (INT cand, INT cpt, INT fpt, INT *ia, INT *ja, INT n,
 *                        INT *cf, REAL *ma)
 * \brief Find independent set of the graph
 *
 * \param cand ???
 * \param cpt ???
 * \param fpt ???
 * \param ia ???
 * \param ja ???
 * \param n ???
 * \param cf contains CF list
 * \param ma contains candidate set info.
 *
 * \return ???
 *
 * Modified by Chunsheng Feng, Zheng Li on 10/14/2012
 *
 * TODO: Fix the missing Doxygen parameters
 */
static INT indset (INT   cand,
                   INT   cpt,
                   INT   fpt,
                   INT  *ia,
                   INT  *ja,
                   INT   n,
                   INT  *cf,
                   REAL *ma)
{
    /* ma: candidates >= 1, cpts = -1, otherwise = 0
     * Note: graph contains candidates only */
    
    Link *list;
    INT  *head, *head_mem;
    INT  *tail, *tail_mem;
    
    INT i, ji, jj, jl, index, istack, stack_size;

    for (istack = i = 0; i < n; ++i) {

        if (cf[i] == cand) {
            ma[i] = 1;
            for (ji = ia[i]+1; ji < ia[i+1]; ++ji) {
                jj = ja[ji];
                if (cf[jj] != cpt) {
                    ma[i]++;
                }
            }

            if (ma[i] > istack) {
                istack = (INT) ma[i];
            }
        }
        else if (cf[i] == cpt) {
            ma[i] = -1;
        }
        else {
            ma[i] = 0;
        }
    }

    stack_size = 2*istack;

    /* INITIALIZE GRAPH */
    list = (Link*)fasp_mem_calloc(n,sizeof(Link));
    head_mem = (INT*)fasp_mem_calloc(stack_size,sizeof(INT));
    tail_mem = (INT*)fasp_mem_calloc(stack_size,sizeof(INT));
    head = head_mem + stack_size;
    tail = tail_mem + stack_size;

#ifdef _OPENMP
#pragma omp parallel for if(stack_size>OPENMP_HOLDS) 
#endif
    for (i = -1; i >= -stack_size; i--) {
        head[i] = i;
        tail[i] = i;
    }

#ifdef _OPENMP
#pragma omp parallel for if(stack_size>OPENMP_HOLDS) 
#endif
    for (i = 0; i < n; ++i) {
        if (ma[i] > 0) {
            GraphAdd(list, head, tail, i, (INT) ma[i]);
        }
    }

    while (istack > 0) {
        /* i with maximal measure is at the head of the stacks */
        i = head[-istack];
        /* make i a c-point */
        cf[i] = cpt;
        ma[i] = -1;
        /* remove i from graph */
        GraphRemove(list, head, tail, i);

        /* update neighbors and neighbors-of-neighbors */
        for (ji = ia[i]+1; ji < ia[i+1]; ++ji) {
            jj = ja[ji];
            /* if not "decided" c or f */
            if (ma[jj] > -1) {
                /* if a candidate, remove jj from graph */
                if (ma[jj] > 0) {
                    GraphRemove(list, head, tail, jj);
                }
                /* make jj an f-point and mark "decided" */
                cf[jj] = fpt;
                ma[jj] = -1;

                for (jl = ia[jj]+1; jl < ia[jj+1]; jl++) {
                    index = ja[jl];
                    /* if a candidate, increase likehood of being chosen */
                    if (ma[index] > 0) {
                        ma[index]++;
                        /* move index in graph */
                        GraphRemove(list, head, tail, index);
                        GraphAdd(list, head, tail, index, (INT) ma[index]);
                        if (ma[index] > istack) {
                            istack = (INT) ma[index];
                        }
                    }
                }
            }
        }

        /* reset istack to point to the biggest non-empty stack */
        for ( ; istack > 0; istack-- ) {
            /* if non-negative, break */
            if (head[-istack] > -1) {
                break;
            }
        }
    }

    fasp_mem_free(list);     list     = NULL;
    fasp_mem_free(head_mem); head_mem = NULL;
    fasp_mem_free(tail_mem); tail_mem = NULL;

    return 0;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
