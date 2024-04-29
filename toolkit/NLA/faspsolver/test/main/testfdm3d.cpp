/*! \file  testfdm3d.cpp
 *
 *  \brief Test program to generate FDM linear system for 3D Possion Problem.
 *
 *  Consider a three-dimensional Possion equation
 *
 * \f[
 *   \frac{du}{dt}-u_{xx}-u_{yy}-u_{zz} = f(x,y,z,t)\
 *   \ in\ \Omega = (0,1)\times(0,1)\times(0,1)
 * \f]
 * \f[
 *           u(x,y,z,0) = 0\ \ \ \ \ \ in\ \Omega
 * \f]
 * \f[
 *                    u = 0\ \ \ \ \ \ \ \ \ on\  \partial\Omega
 * \f]
 *
 *  where f(x,y,z,t) = \f$3*\pi^2*u(x,y,z,t) + sin(\pi*x)*sin(\pi*y)*sin(\pi*z)\f$,
 *  and the solution function can be expressed by
 *
 *        \f$u(x,y,z,t) = sin(\pi*x)*sin(\pi*y)*sin(\pi*z)\f$
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
    
    const char *MatFile = NULL;
    const char *RhsFile = NULL;
    const char *SolFile = NULL;
    const char *SPFile  = NULL;
    
    fsls_BandMatrix *A    = NULL;
    fsls_CSRMatrix  *Acsr = NULL;
    fsls_XVector    *b    = NULL;
    fsls_XVector    *u    = NULL;
    
    int nx,ny,nz,ngrid,nt;
    int test = 0;
    int input_flag  = 1;
    const char* order = "normal";
    const char* op = "csr";

    nx = 10;
    ny = 10;
    nz = 10;
    nt = 0;

    while (arg_index < argc)
    {
        if (argc%2 == 0)
        {
            print_usage = 1;
            break;
        }

        if ( strcmp(argv[arg_index], "-help") == 0 )
        {
            print_usage = 1;
            break;
        }

        if ( strcmp(argv[arg_index], "-nx") == 0 )
        {
            arg_index ++;
            nx = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-ny") == 0 )
        {
            arg_index ++;
            ny = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-nz") == 0 )
        {
            arg_index ++;
            nz = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-nt") == 0 )
        {
            arg_index ++;
            nt = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-test") == 0 )
        {
            arg_index ++;
            test = atoi(argv[arg_index++]);
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-order") == 0 )
        {
            arg_index ++;
            order = argv[arg_index++];
            input_flag = 0;
        }
        if (arg_index >= argc) break;

        if ( strcmp(argv[arg_index], "-op") == 0 )
        {
            arg_index ++;
            op = argv[arg_index++];
            input_flag = 0;
        }
        if (arg_index >= argc) break;
        
        if (input_flag)
        {
            print_usage = 1;
            break;
        }
    }

    if (print_usage)
    {
        printf("\n  Usage: %s [<options>]\n\n", argv[0]);
        printf("  -nx    <val> : number of interier nodes in x-direction [default: 10]\n");
        printf("  -ny    <val> : number of interier nodes in y-direction [default: 10]\n");
        printf("  -nz    <val> : number of interier nodes in z-direction [default: 10]\n");
        printf("  -nt    <val> : number of interier nodes in t-direction [default:  0]\n");
        printf("  -test  <val> : 1->lapack routine test for fdm;0->no test for FDM [default:  0]\n");
        printf("  -op <val>    : csr->CSR format for the output matrix\n         coo->COO format for the output matrix [default:  CSR]\n");
        printf("  -help        : print this help message\n\n");
        exit(1);
    }

    ngrid = nx*ny*nz;
    
    printf("\n ++++++++ (nx,ny,nz,nt,test,order) = (%d,%d,%d,%d,%d,%s)  ngrid = %d +++++++\n\n",
           nx,ny,nz,nt,test,order,ngrid);

    MatFile = "./out/mat_";
    RhsFile = "./out/rhs_";
    SolFile = "./out/sol_";
    SPFile  = "./out/sp_";

    /*-----------------------------------------------------
     * construct a linear system
     *----------------------------------------------------*/
    if (TTest) fasp_gettime(&tStart);

    fsls_BuildLinearSystem_7pt3d(nt, nx, ny, nz, &A, &b, &u);

    if (TTest)
    {
        fasp_gettime(&tEnd);
        printf("\n >>> total time: %.3f seconds\n\n",mytime(tStart,tEnd));
    }

    fsls_Band2CSRMatrix(A, &Acsr);

    if ( strcmp(op,"csr") == 0 )
    {
        sprintf(filename, "%scsr_%dX%dX%d.dat",MatFile,nx,ny,nz);
        fsls_CSRMatrixPrint(Acsr,filename);
    }
    if ( strcmp(op,"coo") == 0 )
    {
        sprintf(filename, "%scoo_%dX%dX%d.dat",MatFile,nx,ny,nz);
        fsls_COOMatrixPrint(Acsr,filename);
    }

    sprintf(filename, "%s%dX%dX%d_%s.dat",SPFile,nx,ny,nz,order);
    fsls_MatrixSPGnuplot( Acsr, filename );

    sprintf(filename, "%s%dX%dX%d.dat",RhsFile,nx,ny,nz);
    fsls_XVectorPrint(b, filename);

    sprintf(filename, "%s%dX%dX%d.dat",SolFile,nx,ny,nz);
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
