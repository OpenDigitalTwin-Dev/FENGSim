/*! \file  testmat.c
 *
 *  \brief Test matrix properties
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include <time.h>

#include "fasp.h"
#include "fasp_functs.h"

/**
 * \fn int main (int argc, const char * argv[])
 *
 * \brief This is the main function for testing matrix properties.
 *
 * \author Chensong Zhang
 * \date   03/31/2009
 * 
 * Modified by Chensong Zhang on 03/19/2012 
 */
int main(int argc, const char * argv[]) 
{       
    char *inputfile="ini/input.dat";
    input_param Input;
    fasp_param_input(inputfile,&Input);
    
    char filename1[512], *datafile1;    
    memcpy(filename1,Input.workdir,STRLEN);
    
    char filename2[512], *datafile2;    
    memcpy(filename2,Input.workdir,STRLEN);

    // Read matrix for testing
    dCSRmat A;
    dvector b;
    
    datafile1="csrmat_FE.dat";
    strcat(filename1,datafile1);
    datafile2="rhs_FE.dat";
    strcat(filename2,datafile2);        
    fasp_dcsrvec_read2(filename1, filename2, &A, &b);
        
    // Check sparse pattern
    char *bmpfile="out/matrixsubplot.bmp";  /* Output the matrix as BMP file */
    fasp_dcsr_subplot(&A, bmpfile, 200);

    bmpfile="out/matrix.bmp";   /* Output the matrix as BMP file */
    fasp_dcsr_plot(&A, bmpfile);
    
    // Check symmetry
    fasp_check_symm(&A);
    
    // Check diagonal positivity
    fasp_check_diagpos(&A);
    
    // Check diagonal dominance
    fasp_check_diagdom(&A);
    
    // Output matrix in COO format  
    char *matfile="out/matrix.out"; /* Output the matrix in coordinate format */    
    fasp_dcoo_write(matfile, &A);
    
    // Clean up memory
    fasp_dcsr_free(&A);
    fasp_dvec_free(&b);
    
    return FASP_SUCCESS;
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
