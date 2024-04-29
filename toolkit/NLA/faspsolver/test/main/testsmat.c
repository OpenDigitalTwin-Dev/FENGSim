/*! \file  testsmat.c
 *
 *  \brief Test smat (dense) matrix operations
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2015--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"
#include "fasp_functs.h"

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for testing matrix properties.
 *
 * \author Chensong Zhang
 * \date   04/03/2015
 */
int main(int argc, const char * argv[]) 
{
    REAL mat[4] = {2.0, 1.0, 0.0, 2.0};
    // REAL mat[4] = {0.0, 2.0, 2.0, 1.0};

    fasp_smat_invp_nc(mat,2);
    
    printf("%e   %e\n", mat[0], mat[1]);
    printf("%e   %e\n", mat[2], mat[3]);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
