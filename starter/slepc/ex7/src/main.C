/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a generalized eigensystem Ax=kBx with matrices loaded from a file.\n"
  "The command line options are:\n"
  "  -f1 <filename> -f2 <filename>, PETSc binary files containing A and B.\n"
  "  -evecs <filename>, output file to save computed eigenvectors.\n"
  "  -ninitial <nini>, number of user-provided initial guesses.\n"
  "  -finitial <filename>, binary file containing <nini> vectors.\n"
  "  -nconstr <ncon>, number of user-provided constraints.\n"
  "  -fconstr <filename>, binary file containing <ncon> vectors.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            A,B;             /* matrices */
  EPS            eps;             /* eigenproblem solver context */
  ST             st;
  KSP            ksp;
  EPSType        type;
  PetscReal      tol;
  Vec            xr,xi,*Iv,*Cv;
  PetscInt       nev,maxit,i,its,lits,nconv,nini=0,ncon=0;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,evecs,ishermitian,terse;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the matrices that define the eigensystem, Ax=kBx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized eigenproblem stored in file.\n\n"));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f1",filename,sizeof(filename),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate a file name for matrix A with the -f1 option");

#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrices from binary files...\n"));
#else
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrices from binary files...\n"));
#endif
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscOptionsGetString(NULL,NULL,"-f2",filename,sizeof(filename),&flg));
  if (flg) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
    PetscCall(MatSetFromOptions(B));
    PetscCall(MatLoad(B,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Matrix B was not provided, setting B=I\n\n"));
    B = NULL;
  }

  PetscCall(MatCreateVecs(A,NULL,&xr));
  PetscCall(MatCreateVecs(A,NULL,&xi));

  /*
     Read user constraints if available
  */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nconstr",&ncon,&flg));
  if (flg) {
    PetscCheck(ncon>0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The number of constraints must be >0");
    PetscCall(PetscOptionsGetString(NULL,NULL,"-fconstr",filename,sizeof(filename),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must specify the name of the file storing the constraints");
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    PetscCall(VecDuplicateVecs(xr,ncon,&Cv));
    for (i=0;i<ncon;i++) PetscCall(VecLoad(Cv[i],viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /*
     Read initial guesses if available
  */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ninitial",&nini,&flg));
  if (flg) {
    PetscCheck(nini>0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"The number of initial vectors must be >0");
    PetscCall(PetscOptionsGetString(NULL,NULL,"-finitial",filename,sizeof(filename),&flg));
    PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must specify the name of the file containing the initial vectors");
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
    PetscCall(VecDuplicateVecs(xr,nini,&Iv));
    for (i=0;i<nini;i++) PetscCall(VecLoad(Iv[i],viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators. In this case, it is a generalized eigenvalue problem
  */
  PetscCall(EPSSetOperators(eps,A,B));

  /*
     If the user provided initial guesses or constraints, pass them here
  */
  PetscCall(EPSSetInitialSpace(eps,nini,Iv));
  PetscCall(EPSSetDeflationSpace(eps,ncon,Cv));

  /*
     Set solver parameters at runtime
  */
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSSolve(eps));

  /*
     Optional: Get some information from the solver and display it
  */
  PetscCall(EPSGetIterationNumber(eps,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));
  PetscCall(EPSGetST(eps,&st));
  PetscCall(STGetKSP(st,&ksp));
  PetscCall(KSPGetTotalIterations(ksp,&lits));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method: %" PetscInt_FMT "\n",lits));
  PetscCall(EPSGetType(eps,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  PetscCall(EPSGetDimensions(eps,&nev,NULL,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
  PetscCall(EPSGetTolerances(eps,&tol,&maxit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Show detailed info unless -terse option is given by user
   */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  /*
     Save eigenvectors, if requested
  */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-evecs",filename,sizeof(filename),&evecs));
  PetscCall(EPSGetConverged(eps,&nconv));
  if (nconv>0 && evecs) {
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer));
    PetscCall(EPSIsHermitian(eps,&ishermitian));
    for (i=0;i<nconv;i++) {
      PetscCall(EPSGetEigenvector(eps,i,xr,xi));
      PetscCall(VecView(xr,viewer));
#if !defined(PETSC_USE_COMPLEX)
      if (!ishermitian) PetscCall(VecView(xi,viewer));
#endif
    }
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /*
     Free work space
  */
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&xr));
  PetscCall(VecDestroy(&xi));
  if (nini > 0) PetscCall(VecDestroyVecs(nini,&Iv));
  if (ncon > 0) PetscCall(VecDestroyVecs(ncon,&Cv));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62b.petsc -eps_nev 4 -terse
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

   test:
      suffix: ciss_1
      args: -f1 ${DATAFILESPATH}/matrices/complex/mhd1280a.petsc -f2 ${DATAFILESPATH}/matrices/complex/mhd1280b.petsc -eps_type ciss -eps_ciss_usest 0 -eps_ciss_quadrule chebyshev -rg_type ring -rg_ring_center 0 -rg_ring_radius .49 -rg_ring_width 0.2 -rg_ring_startangle .25 -rg_ring_endangle .49 -terse -eps_max_it 1
      requires: double complex datafilespath !defined(PETSC_USE_64BIT_INDICES)
      timeoutfactor: 2

   test:
      suffix: 3 # test problem (A,A)
      args: -f1 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -f2 ${SLEPC_DIR}/share/slepc/datafiles/matrices/bfw62a.petsc -eps_nev 4 -terse
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
