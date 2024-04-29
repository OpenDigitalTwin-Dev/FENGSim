/*! \file  AuxMessage.c
 *
 *  \brief Output some useful messages
 *
 *  \note  This file contains Level-0 (Aux) functions.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_itinfo (const INT ptrlvl, const INT stop_type, const INT iter,
 *                       const REAL relres, const REAL absres, const REAL factor)
 *
 * \brief Print out iteration information for iterative solvers
 *
 * \param ptrlvl     Level for output
 * \param stop_type  Type of stopping criteria
 * \param iter       Number of iterations
 * \param relres     Relative residual of different kinds
 * \param absres     Absolute residual of different kinds
 * \param factor     Contraction factor
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 *
 * Modified by Chensong Zhang on 03/28/2013: Output initial guess
 * Modified by Chensong Zhang on 04/05/2013: Fix a typo
 */
void fasp_itinfo (const INT   ptrlvl,
                  const INT   stop_type,
                  const INT   iter,
                  const REAL  relres,
                  const REAL  absres,
                  const REAL  factor)
{
    if ( ptrlvl >= PRINT_SOME ) {
        
        if ( iter > 0 ) {
            printf("%6d | %13.6e   | %13.6e  | %10.4f\n", iter, relres, absres, factor);
        }
        else { // iter = 0: initial guess
            printf("-----------------------------------------------------------\n");
            switch (stop_type) {
                case STOP_REL_RES:
                    printf("It Num |   ||r||/||b||   |     ||r||      |  Conv. Factor\n");
                    break;
                case STOP_REL_PRECRES:
                    printf("It Num | ||r||_B/||b||_B |    ||r||_B     |  Conv. Factor\n");
                    break;
                case STOP_MOD_REL_RES:
                    printf("It Num |   ||r||/||x||   |     ||r||      |  Conv. Factor\n");
                    break;
            }
            printf("-----------------------------------------------------------\n");
            printf("%6d | %13.6e   | %13.6e  |     -.-- \n", iter, relres, absres);
        } // end if iter
        
    } // end if ptrlvl
}

/**
 * \fn void void fasp_amgcomplexity (const AMG_data *mgl, const SHORT prtlvl)
 *
 * \brief Print level and complexity information of AMG
 *
 * \param mgl      Multilevel hierachy for AMG
 * \param prtlvl   How much information to print
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 */
void fasp_amgcomplexity (const AMG_data  *mgl,
                         const SHORT      prtlvl)
{
    const SHORT   max_levels = mgl->num_levels;
    SHORT         level;
    REAL          gridcom = 0.0, opcom = 0.0;
    
    if ( prtlvl >= PRINT_SOME ) {
        
        printf("-----------------------------------------------------------\n");
        printf("  Level   Num of rows   Num of nonzeros   Avg. NNZ / row   \n");
        printf("-----------------------------------------------------------\n");

        for ( level = 0; level < max_levels; ++level) {
            const REAL AvgNNZ = (REAL) mgl[level].A.nnz/mgl[level].A.row;
            printf("%5d %13d %17d %14.2f\n",
                   level, mgl[level].A.row, mgl[level].A.nnz, AvgNNZ);
            gridcom += mgl[level].A.row;
            opcom   += mgl[level].A.nnz;

#if 0 // Save coarser linear systems for debugging purposes --Chensong
            char matA[max_levels], rhsb[max_levels];
            if (level > 0) {
                sprintf(matA, "A%d.coo", level);
                sprintf(rhsb, "b%d.coo", level);
                fasp_dcsrvec_write2(matA, rhsb, &(mgl[level].A), &(mgl[level].b));
            }
#endif
        }
        printf("-----------------------------------------------------------\n");
        
        gridcom /= mgl[0].A.row;
        opcom   /= mgl[0].A.nnz;
        printf("  Grid complexity = %.3f  |", gridcom);
        printf("  Operator complexity = %.3f\n", opcom);
        
        printf("-----------------------------------------------------------\n");
    }
}

/**
 * \fn void void fasp_amgcomplexity_bsr (const AMG_data_bsr *mgl,
 *                                       const SHORT prtlvl)
 *
 * \brief Print complexities of AMG method for BSR matrices
 *
 * \param mgl      Multilevel hierachy for AMG
 * \param prtlvl   How much information to print
 *
 * \author Chensong Zhang
 * \date   05/10/2013
 */
void fasp_amgcomplexity_bsr (const AMG_data_bsr  *mgl,
                             const SHORT          prtlvl)
{
    const SHORT  max_levels = mgl->num_levels;
    SHORT        level;
    REAL         gridcom = 0.0, opcom = 0.0;
    
    if ( prtlvl >= PRINT_SOME ) {
        
        printf("-----------------------------------------------------------\n");
        printf("  Level   Num of rows   Num of nonzeros   Avg. NNZ / row   \n");
        printf("-----------------------------------------------------------\n");
        
        for ( level = 0; level < max_levels; ++level ) {
            const REAL AvgNNZ = (REAL) mgl[level].A.NNZ/mgl[level].A.ROW;
            printf("%5d  %13d  %17d  %14.2f\n",
                   level,mgl[level].A.ROW, mgl[level].A.NNZ, AvgNNZ);
            gridcom += mgl[level].A.ROW;
            opcom   += mgl[level].A.NNZ;
        }
        printf("-----------------------------------------------------------\n");
        
        gridcom /= mgl[0].A.ROW;
        opcom   /= mgl[0].A.NNZ;
        printf("  Grid complexity = %.3f  |", gridcom);
        printf("  Operator complexity = %.3f\n", opcom);
        
        printf("-----------------------------------------------------------\n");
        
    }
}

/**
 * \fn void void fasp_cputime (const char *message, const REAL cputime)
 *
 * \brief Print CPU walltime
 *
 * \param message   Some string to print out
 * \param cputime   Walltime since start to end
 *
 * \author Chensong Zhang
 * \date   04/10/2012
 */
void fasp_cputime (const char  *message,
                   const REAL   cputime)
{
    printf("%s costs %.4f seconds\n", message, cputime);
}

/**
 * \fn void fasp_message (const INT ptrlvl, const char *message)
 *
 * \brief Print output information if necessary
 *
 * \param ptrlvl   Level for output
 * \param message  Error message to print
 *
 * \author Chensong Zhang
 * \date   11/16/2009
 */
void fasp_message (const INT    ptrlvl,
                   const char  *message)
{
    if ( ptrlvl > PRINT_NONE ) printf("%s", message);
}

/**
 * \fn void fasp_chkerr (const SHORT status, const char *fctname)
 *
 * \brief Check error status and print out error messages before quit
 *
 * \param status   Error status
 * \param fctname  Function name where this routine is called
 *
 * \author Chensong Zhang
 * \date   01/10/2012
 */
void fasp_chkerr (const SHORT  status,
                  const char  *fctname)
{
    if ( status >= 0 ) return; // No error found!!!
    
    switch ( status ) {
        case ERROR_READ_FILE:
            printf("### ERROR: Cannot read file! [%s]\n", fctname);
            break;
        case ERROR_OPEN_FILE:
            printf("### ERROR: Cannot open file! [%s]\n", fctname);
            break;
        case ERROR_WRONG_FILE:
            printf("### ERROR: Unknown file format! [%s]\n", fctname);
            break;
        case ERROR_INPUT_PAR:
            printf("### ERROR: Unknown input argument! [%s]\n", fctname);
            break;
        case ERROR_REGRESS:
            printf("### ERROR: Regression test failed! [%s]\n", fctname);
            break;
        case ERROR_ALLOC_MEM:
            printf("### ERROR: Cannot allocate memory! [%s]\n", fctname);
            break;
        case ERROR_NUM_BLOCKS:
            printf("### ERROR: Unexpected number of blocks! [%s]\n", fctname);
            break;
        case ERROR_DATA_STRUCTURE:
            printf("### ERROR: Wrong data structure! [%s]\n", fctname);
            break;
        case ERROR_DATA_ZERODIAG:
            printf("### ERROR: Matrix has zero diagonal entries! [%s]\n", fctname);
            break;
        case ERROR_DUMMY_VAR:
            printf("### ERROR: Unknown input argument! [%s]\n", fctname);
            break;
        case ERROR_AMG_INTERP_TYPE:
            printf("### ERROR: Unknown AMG interpolation type! [%s]\n", fctname);
            break;
        case ERROR_AMG_COARSE_TYPE:
            printf("### ERROR: Unknown AMG coarsening type! [%s]\n", fctname);
            break;
        case ERROR_AMG_SMOOTH_TYPE:
            printf("### ERROR: Unknown AMG smoother type! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_TYPE:
            printf("### ERROR: Unknown solver type! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_PRECTYPE:
            printf("### ERROR: Unknown preconditioner type! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_STAG:
            printf("### ERROR: Solver stagnation! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_SOLSTAG:
            printf("### ERROR: Solution close to zero! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_TOLSMALL:
            printf("### ERROR: Convergence tolerance too small! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_ILUSETUP:
            printf("### ERROR: ILU setup failed! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_MAXIT:
            printf("### ERROR: Max iteration number reached! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_EXIT:
            printf("### ERROR: Iterative solver failed! [%s]\n", fctname);
            break;
        case ERROR_SOLVER_MISC:
            printf("### ERROR: Unknown solver runtime error! [%s]\n", fctname);
            break;
        case ERROR_MISC:
            printf("### ERROR: Miscellaneous error! [%s]\n", fctname);
            break;
        case ERROR_QUAD_TYPE:
            printf("### ERROR: Unknown quadrature rules! [%s]\n", fctname);
            break;
        case ERROR_QUAD_DIM:
            printf("### ERROR: Num of quad points not supported! [%s]\n", fctname);
            break;
        case ERROR_UNKNOWN:
            printf("### ERROR: Unknown error! [%s]\n", fctname);
            break;
        default:
            break;
    }
    
    exit(status);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
