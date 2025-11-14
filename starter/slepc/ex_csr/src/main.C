#include <slepceps.h>
#include <petscvec.h>
#include <petscmat.h>

#include <iostream>

int main(int argc,char **argv) {
    Mat            A, B;
    PetscInt       m = 5, n = 5;
    PetscInt       Istart, Iend;
    PetscMPIInt    rank;
    EPS            eps;
    
    SlepcInitialize(&argc,&argv,NULL,NULL);
	
    // Compressed Sparse Row
    // [ 2, -1,  0,  0,  0]
    // [-1,  2, -1,  0,  0] 
    // [ 0, -1,  2, -1,  0]
    // [ 0,  0, -1,  2, -1]
    // [ 0,  0,  0, -1,  2]

    // [ 4,  1,  0,  0,  0]
    // [ 1,  4,  1,  0,  0] 
    // [ 0,  1,  4,  1,  0]
    // [ 0,  0,  1,  4,  1]
    // [ 0,  0,  0,  1,  4]
    
    PetscInt row_ptr_A[6] = {0, 2, 5, 8, 11, 13};
    PetscInt col_ind_A[13] = {
        0, 1,          
        0, 1, 2,       
        1, 2, 3,       
        2, 3, 4,       
        3, 4           
    };
    PetscScalar values_A[13] = {
         2.0, -1.0,  
        -1.0,  2.0, -1.0,
        -1.0,  2.0, -1.0,
        -1.0,  2.0, -1.0,
        -1.0,  2.0       
    };
    
    PetscInt row_ptr_B[6] = {0, 2, 5, 8, 11, 13};
    PetscInt col_ind_B[13] = {
        0, 1,          
        0, 1, 2,       
        1, 2, 3,       
        2, 3, 4,       
        3, 4           
    };
    PetscScalar values_B[13] = {
         4.0, 1.0,  
         1.0, 4.0, 1.0,
         1.0, 4.0, 1.0,
         1.0, 4.0, 1.0,
	 1.0, 4.0       
    };

    /*! create A */
    MatCreate(PETSC_COMM_WORLD,&A);
    MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);
    MatSetType(A,MATAIJ); 
    MatSetFromOptions(A);  
    MatSetUp(A);
    PetscInt *d_nnz = NULL;
    PetscMalloc1(m,&d_nnz);
    for (PetscInt i=0; i<m; i++) {
        d_nnz[i] = row_ptr_A[i+1] - row_ptr_A[i];
    }
    MatMPIAIJSetPreallocation(A,0,d_nnz,0,NULL);

    /*! create B */
    MatCreate(PETSC_COMM_WORLD,&B);
    MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);
    MatSetType(B,MATAIJ); 
    MatSetFromOptions(B);  
    MatSetUp(B);
    for (PetscInt i=0; i<m; i++) {
        d_nnz[i] = row_ptr_B[i+1] - row_ptr_B[i];
    }
    MatMPIAIJSetPreallocation(B,0,d_nnz,0,NULL);
    
    PetscFree(d_nnz);

    /*! give A's values*/
    MatGetOwnershipRange(A, &Istart, &Iend);
    //std::cout << Istart << " " << Iend << std::endl;
    for (PetscInt i=Istart; i<Iend; i++) {
        PetscInt ncols = row_ptr_A[i+1] - row_ptr_A[i];
        PetscInt *cols = &col_ind_A[row_ptr_A[i]];
        PetscScalar *vals = &values_A[row_ptr_A[i]];
	/*! the second parameter in MatSetValues means just set 1 row values*/
        MatSetValues(A, 1, &i, ncols, cols, vals, INSERT_VALUES);
    }

    /*! give B's values*/
    MatGetOwnershipRange(B, &Istart, &Iend);
    //std::cout << Istart << " " << Iend << std::endl;
    for (PetscInt i=Istart; i<Iend; i++) {
        PetscInt ncols = row_ptr_B[i+1] - row_ptr_B[i];
        PetscInt *cols = &col_ind_B[row_ptr_B[i]];
        PetscScalar *vals = &values_B[row_ptr_B[i]];
	/*! the second parameter in MatSetValues means just set 1 row values*/
        MatSetValues(B, 1, &i, ncols, cols, vals, INSERT_VALUES);
    }
    
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
    MatView(B, PETSC_VIEWER_STDOUT_WORLD);

    /*! Ax=Mb */
    EPSCreate(PETSC_COMM_WORLD,&eps);
    EPSSetOperators(eps,A,B);
    EPSSetFromOptions(eps);
    EPSSolve(eps);

    /*! output eigenvalues and eigenvectors */
    PetscInt nconv;
    EPSGetConverged(eps, &nconv);
    std::cout << std::endl << "nconv: " << nconv << std::endl;
    PetscScalar ev;
    Vec xr,yrA,yrB;
    /*! if A is m*n, left vector is m*1, right vector is n*1*/
    MatCreateVecs(A,NULL,&xr);
    MatCreateVecs(A,NULL,&yrA);
    MatCreateVecs(A,NULL,&yrB);
    for (int i=0; i<nconv; i++) {
	EPSGetEigenvalue(eps,i,&ev,NULL);
	std::cout << "eigenvalue " << i << ": " << ev << std::endl;
        EPSGetEigenvector(eps,i,xr,NULL);
        //VecView(xr,PETSC_VIEWER_STDOUT_WORLD);
	MatMult(A,xr,yrA);
	MatMult(B,xr,yrB);
	VecAXPY(yrA,-ev,yrB);
	PetscReal norm;
	VecNorm(yrA,NORM_1,&norm);
	std::cout << "   Ax-aBx: " << norm << std::endl;
    }

    EPSDestroy(&eps);
    MatDestroy(&A);
    MatDestroy(&B);
    VecDestroy(&xr);
    VecDestroy(&yrA);
    VecDestroy(&yrB);
    SlepcFinalize();
    return 0;
}
