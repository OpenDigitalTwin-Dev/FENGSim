/*! \file  XtrSamg.c
 *
 *  \brief Interface to SAMG solvers
 *
 *  Reference for SAMG:
 *  http://www.scai.fraunhofer.de/geschaeftsfelder/nuso/produkte/samg.html
 *
 *  \warning This interface has *only* been tested for SAMG24a1 (2010 version)!
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2010--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <math.h>
#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/*---------------------------------*/
/*--      Public Functions       --*/
/*---------------------------------*/

/**
 * \fn void dvector2SAMGInput (dvector *vec, char *filename)
 * \brief Write a dvector to disk file in SAMG format (coordinate format)
 *
 * \param vec        Pointer to the dvector
 * \param filename   File name for input
 *
 * \author Zhiyang Zhou
 * \date   08/25/2010
 */
void dvector2SAMGInput (dvector *vec,
                        char *filename)
{
    INT m = vec->row, i;
    
    FILE *fp=fopen(filename,"w");
    if ( fp == NULL ) {
        printf("### ERROR: Opening file %s failed!\n",filename);
        exit(ERROR_OPEN_FILE);
    }
    
    printf("%s: writing vector to `%s'...\n", __FUNCTION__, filename);
    
    for (i=0;i<m;++i) fprintf(fp,"%0.15le\n",vec->val[i]);
    
    fclose(fp);
}

/**
 * \fn INT dCSRmat2SAMGInput (dCSRmat *A, char *filefrm, char *fileamg)
 * \brief Write SAMG Input data from a sparse matrix of CSR format.
 *
 * \param A         Pointer to the dCSRmat matrix
 * \param filefrm   Name of the .frm file
 * \param fileamg   Name of the .amg file
 *
 * \author Zhiyang Zhou
 * \date 2010/08/25
 */
INT dCSRmat2SAMGInput (dCSRmat *A,
                       char *filefrm,
                       char *fileamg)
{
    FILE    *fp            = NULL;
    INT      file_base     = 1;
    
    REAL    *A_data        = A -> val;
    INT     *A_i           = A -> IA;
    INT     *A_j           = A -> JA;
    INT      num_rowsA     = A -> row;
    INT      num_nonzeros  = A_i[num_rowsA] - A_i[0];
    
    INT      matrix_type   = 0;
    INT      rowsum_type   = 0;
    INT      symmetry_type = 0;
    
    INT      i,j;
    REAL     rowsum;
    
    fasp_dcsr_diagpref(A);
    
    /* check symmetry type of the matrix */
    symmetry_type = fasp_check_symm(A);
    
    /* check rowsum type of the matrix */
    for (i = 0; i < num_rowsA; ++i) {
        rowsum = 0.0;
        for (j = A_i[i]; j < A_i[i+1]; ++j) {
            rowsum += A_data[j];
        }
        if (rowsum*rowsum > 0.0) {
            rowsum_type = 1;
            break;
        }
    }
    
    /* Get the matrix type of A */
    if (symmetry_type == 0) {
        if (rowsum_type == 0)
            matrix_type = 11;
        else
            matrix_type = 12;
    }
    else {
        if (rowsum_type == 0)
            matrix_type = 21;
        else
            matrix_type = 22;
    }
    
    /* write the *.frm file */
    fp = fopen(filefrm, "w");
    fprintf(fp, "%s   %d\n", "f", 4);
    fprintf(fp, "%d %d %d %d %d\n", num_nonzeros, num_rowsA, matrix_type, 1, 0);
    fclose(fp);
    
    /* write the *.amg file */
    fp = fopen(fileamg, "w");
    for (j = 0; j <= num_rowsA; ++j) {
        fprintf(fp, "%d\n", A_i[j] + file_base);
    }
    for (j = 0; j < num_nonzeros; ++j) {
        fprintf(fp, "%d\n", A_j[j] + file_base);
    }
    if (A_data) {
        for (j = 0; j < num_nonzeros; ++j) {
            fprintf(fp, "%.15le\n", A_data[j]); // we always use "%.15le\n"
        }
    }
    else {
        fprintf(fp, "### WARNING: No matrix data!\n");
    }
    fclose(fp);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
