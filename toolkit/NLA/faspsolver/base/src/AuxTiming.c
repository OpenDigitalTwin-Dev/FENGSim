/*! \file  AuxTiming.c
 *
 *  \brief Timing subroutines
 *
 *  \note  This file contains Level-0 (Aux) functions.
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fasp.h"
#include "fasp_functs.h"
// #include "sys/time.h" //! call gettimeofday() for struct timeval on Linux system.

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void fasp_gettime (REAL *time)
 *
 * \brief Get system time
 *
 * \author Chunsheng Feng, Zheng LI
 * \date   11/10/2012
 *
 * Modified by Chensong Zhang on 09/22/2014: Use CLOCKS_PER_SEC for cross-platform
 */
void fasp_gettime(REAL* time)
{
    if (time != NULL) {
#ifdef _OPENMP
        *time = omp_get_wtime();
#else
         *time = (REAL)clock() / CLOCKS_PER_SEC;

        //struct timeval t;
        //gettimeofday(&t, NULL); // Get system time
        //*time = t.tv_sec + t.tv_usec / 1000000.0;
#endif
    }
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
