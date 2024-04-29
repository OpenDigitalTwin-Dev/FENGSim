/*! \file  testfdm2d.cpp
 *
 *  \brief Test program to generate FDM linear system for 2D Heat equation.
 *
 *  Consider a two-dimensional Heat/Poisson equation
 *
 * \f[
 *   \frac{du}{dt}-u_{xx}-u_{yy} = f(x,y,t)\ \ in\ \Omega = (0,1)\times(0,1)
 * \f]
 * \f[
 *             u(x,y,0) = 0\ \ \ \ \ \ in\ \Omega = (0,1)\times(0,1)
 * \f]
 * \f[
 *                    u = 0\ \ \ \ \ \ \ \ \ on\  \partial\Omega
 * \f]
 *
 *  where f(x,y,t) = \f$ 2*\pi^2*sin(\pi*x)*sin(\pi*y)*t + sin(\pi*x)*sin(\pi*y) \f$,
 *  and the solution function can be expressed by
 *
 *          \f$ u(x,y,t) = sin(pi*x)*sin(pi*y)*t. \f$
 *
 *---------------------------------------------------------------------------------
 *  Copyright (C) 2009--Present by the FASP team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *---------------------------------------------------------------------------------
 */

#include "fasp.h"

extern "C"
{
#include "fasp_functs.h"
#include "poisson_fdm.h"
}

int
main( int argc, char *argv[])
{
    double tStart, tEnd;
    int TTest       = 1;
    int arg_index   = 1;
    int print_usage = 0;

    char filename[120];

    const char *FileDir = NULL;

    fsls_BandMatrix *A    = NULL;
    fsls_CSRMatrix  *Acsr = NULL;
    fsls_XVector    *b    = NULL;
    fsls_XVector    *u    = NULL;

    int nx,ny,ngrid,nt;
    int rb = 0;
    int test = 0;
    int input_flag  = 1;
    const char* order = "normal";
    const char* op = "csr";

    nx = 11;
    ny = 11;
    nt = 0;
    FileDir = "./out";

    while (arg_index < argc)
    {
        if (argc%2 == 0) {
            print_usage = 1;
            break;
        }

        if ( strcmp(argv[arg_index], "-help") == 0 ) {
            print_usage = 1;
            break;
        }

        if ( strcmp(argv[arg_index], "-nx") == 0 ) {
            arg_index ++;
            nx = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-ny") == 0 ) {
            arg_index ++;
            ny = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-nt") == 0 ) {
            arg_index ++;
            nt = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-test") == 0 ) {
            arg_index ++;
            test = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-order") == 0 ) {
            arg_index ++;
            order = argv[arg_index++];
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-op") == 0 ) {
            arg_index ++;
            op = argv[arg_index++];
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if (input_flag) {
            print_usage = 1;
            break;
        }
    }

    if (print_usage) {
        printf("\n  Usage: %s [<options>]\n", argv[0]);
        printf("  -nx    <val> : number of interior nodes in x-direction [default: 11]\n");
        printf("  -ny    <val> : number of interior nodes in y-direction [default: 11]\n");
        printf("  -nt    <val> : number of interior nodes in t-direction [default: 0]\n");
        printf("  -test  <val> : 1->Lapack routine test for fdm; 0->no test for FDM [default: 0]\n");
        printf("  -order <val> : rb->Red-Black ordering for the d.o.f [default:  normal]\n");
        printf("  -op    <val> : csr->CSR or coo->COO format for the output matrix [default: CSR]\n");
        printf("  -help        : print this help message\n\n");
        exit(1);
    }
    
    if ( strcmp(order,"rb") == 0 ) rb = 1;

    ngrid = nx*ny;

    printf("\n ++++++++++++ (nx,ny,nt,test,order) = (%d,%d,%d,%d,%s)  ngrid = %d ++++++++++\n\n",
           nx,ny,nt,test,order,ngrid);

    /*-----------------------------------------------------
     * construct a linear system
     *----------------------------------------------------*/
    if (TTest) fasp_gettime(&tStart);

    if (rb) fsls_BuildLinearSystem_5pt2d_rb(nt, nx, ny, &A, &b, &u);
    else fsls_BuildLinearSystem_5pt2d(nt, nx, ny, &A, &b, &u);

    if (TTest) {
        fasp_gettime(&tEnd);
        printf("\n >>> total time: %.3f seconds\n\n",mytime(tStart, tEnd));
    }

    fsls_Band2CSRMatrix(A, &Acsr);

    if ( strcmp(op,"csr") == 0 ) {
        sprintf(filename, "%s/csrmat_%dX%d.dat",FileDir,nx,ny);
        fsls_CSRMatrixPrint(Acsr, filename);
    }
    if ( strcmp(op,"coo") == 0 ) {
        sprintf(filename, "%s/coomat_%dX%d.dat",FileDir,nx,ny);
        fsls_COOMatrixPrint(Acsr, filename);
    }

    sprintf(filename, "%s/sp_%dX%d_%s.dat",FileDir,nx,ny,order);
    fsls_MatrixSPGnuplot(Acsr, filename);

    sprintf(filename, "%s/rhs_%dX%d.dat",FileDir,nx,ny);
    fsls_XVectorPrint(b, filename);

    sprintf(filename, "%s/sol_%dX%d.dat",FileDir,nx,ny);
    fsls_XVectorPrint(u, filename);

    /*------------------------------------------------------
     * free some staff
     *-----------------------------------------------------*/
    fsls_BandMatrixDestroy(A);
    fsls_CSRMatrixDestroy(Acsr);
    fsls_XVectorDestroy(b);
    fsls_XVectorDestroy(u);

    return(0);
}

/*---------------------------------*/
/*--        End of File          --*/
/*---------------------------------*/
