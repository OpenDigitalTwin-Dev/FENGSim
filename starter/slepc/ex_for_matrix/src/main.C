#include <slepceps.h>
#include <iostream>

int main(int argc,char **argv) {
    Mat            A;
    PetscInt       m = 5, n = 5;
    PetscInt       Istart, Iend;
    PetscMPIInt    rank;
    
    PetscInitialize(&argc,&argv,NULL,NULL);
    
    // Compressed Sparse Row
    // [ 2, -1,  0,  0,  0]
    // [-1,  2, -1,  0,  0] 
    // [ 0, -1,  2, -1,  0]
    // [ 0,  0, -1,  2, -1]
    // [ 0,  0,  0, -1,  2]
    PetscInt row_ptr[6] = {0, 2, 5, 8, 11, 13};
    PetscInt col_ind[13] = {
        0, 1,          
        0, 1, 2,       
        1, 2, 3,       
        2, 3, 4,       
        3, 4           
    };
    PetscScalar values[13] = {
         2.0, -1.0,  
        -1.0,  2.0, -1.0,
        -1.0,  2.0, -1.0,
        -1.0,  2.0, -1.0,
        -1.0,  2.0       
    };
    
    MatCreate(PETSC_COMM_WORLD,&A);
    MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);
    MatSetType(A,MATAIJ); 
    MatSetFromOptions(A);  
    MatSetUp(A);
    
    PetscInt *d_nnz = NULL;
    PetscMalloc1(m,&d_nnz);
    for (PetscInt i=0; i<m; i++) {
        d_nnz[i] = row_ptr[i+1] - row_ptr[i];
    }
    MatMPIAIJSetPreallocation(A,0,d_nnz,0,NULL);
    //PetscFree(d_nnz);

    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    std::cout << Istart << " " << Iend << std::endl;
    for (PetscInt i=Istart; i<Iend; i++) {
        PetscInt ncols = row_ptr[i+1] - row_ptr[i];
        PetscInt *cols = &col_ind[row_ptr[i]];
        PetscScalar *vals = &values[row_ptr[i]];
        MatSetValues(A, 1, &i, ncols, cols, vals, INSERT_VALUES);
    }
    
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}
