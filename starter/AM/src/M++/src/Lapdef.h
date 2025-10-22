#ifndef _LAPDEF_H_
#define _LAPDEF_H_

#ifdef LAPACK
//getrf: LU-decomposition
//getri: inverse of LU-decomposition
//getrs: solving AX=B with A=LU
//gemm : matrix/matrix-operation: C = C+AB
//gemv : matrix/vector-operation: Y = Y+AX
//gees : eigenvalue+Schur-vectors
//gesvd: singular value decomposition A = U*S*V^t

extern "C" void dgetrf_(int *M, int *N, void *A, int *LDA, int *IPIV, int *INFO);
extern "C" void zgetrf_(int *M, int *N, void *A, int *LDA, int *IPIV, int *INFO);

extern "C" void dgetri_(int *N, void *A, int *LDA, int *IPIV, void *WORK, void *LWORK, int *INFO);
extern "C" void zgetri_(int *N, void *A, int *LDA, int *IPIV, void *WORK, void *LWORK, int *INFO);

extern "C" void dgetrs_(char* TRANS, int* N, int* NRHS, void* A, int* LDA, int* IPIV, void* B, int* LDB, int* INFO);
extern "C" void zgetrs_(char* TRANS, int* N, int* NRHS, void* A, int* LDA, int* IPIV, void* B, int* LDB, int* INFO);

extern "C" void dgemm_(char* TRANSA, char* TRANSB, int*M, int *N, int *K, double* alpha, void *A, int *LDA, void *B, int *LDB, double *beta, void *C, int *LDC);
extern "C" void zgemm_(char* TRANSA, char* TRANSB, int*M, int *N, int *K, double* alpha, void *A, int *LDA, void *B, int *LDB, double *beta, void *C, int *LDC);

extern "C" void dgemv_(char* TRANS, int* M, int* N, double* alpha, void* A, int* LDA, void* X, int* INCX, double* BETA, void* Y, int* INCY); 
// extern "C" void zgemv_(char* TRANS, int* M, int* N, double* alpha, void* A, int* LDA, void* X, int* INCX, double* BETA, void* Y, int* INCY); 
extern "C" void zgemv_(char* TRANS, int* M, int* N, void* alpha, void* A, int* LDA, void* X, int* INCX, void* BETA, void* Y, int* INCY); 

extern "C" void dgees_(char* JOBVS, char* SORT, bool* SELECT, int* N, void *A, int* LDA, int* SDIM, void* WR, void* WI, void* VS, int* LDVS, 
                       void* WORK, int* LWORK, bool* BWORK, int* INFO);

extern "C" void dgesvd_(char* JOBU, char* JOBVT, int* M, int* N, void* A, int* LDA, double* S, void* U, int* LDU, void* VT, int* LDVT, void* WORK, void* LWORK, int* INFO); 
extern "C" void zgesvd_(char* JOBU, char* JOBVT, int* M, int* N, void* A, int* LDA, double* S, void* U, int* LDU, void* VT, int* LDVT, void* WORK, void* LWORK, int* INFO); 


extern "C" void dpotrf_(char* UPLO, int* N, void* A, int* LDA, int* INFO);
extern "C" void zpotrf_(char* UPLO, int* N, void* A, int* LDA, int* INFO);

extern "C" void dpotrs_(char* UPLO, int* N, int* NRHS, void* A, int* LDA, void* B, int* LDB, int* INFO);
extern "C" void zpotrs_(char* UPLO, int* N, int* NRHS, void* A, int* LDA, void* B, int* LDB, int* INFO);

extern "C" void dpotri_(char* UPLO, int* N, void* A, int* LDA, int* INFO);
extern "C" void zpotri_(char* UPLO, int* N, void* A, int* LDA, int* INFO);

extern "C" void dsymm_(char* SIDE, char* UPLO, int* M, int* N, double* alpha, void* A, int* LDA, void* B, int* LDB, double* beta, void* C, int* LDC);
extern "C" void zsymm_(char* SIDE, char* UPLO, int* M, int* N, void* alpha, void* A, int* LDA, void* B, int* LDB, void* beta, void* C, int* LDC);

extern "C" void dtrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG, int* M, int* N, double* alpha, void* A, int* LDA, void* B, int* LDB);
extern "C" void ztrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG, int* M, int* N, void* alpha, void* A, int* LDA, void* B, int* LDB);

#else

void dgetrf_(int *M, int *N, void *A, int *LDA, int *IPIV, int *INFO) {
       mout << "LAPACK not defined --> no dgetrf_ available" << endl;exit(0);
}
void zgetrf_(int *M, int *N, void *A, int *LDA, int *IPIV, int *INFO) {
       mout << "LAPACK not defined --> no zgetrf_ available" << endl;exit(0);
}

void dgetri_(int *N, void *A, int *LDA, int *IPIV, void *WORK, void *LWORK, int *INFO) {
       mout << "LAPACK not defined --> no dgetri_ available" << endl;exit(0);
}
void zgetri_(int *N, void *A, int *LDA, int *IPIV, void *WORK, void *LWORK, int *INFO) {
       mout << "LAPACK not defined --> no zgetri_ available" << endl;exit(0);
}

void dgetrs_(char* TRANS, int* N, int* NRHS, void* A, int* LDA, int* IPIV, void* B, int* LDB, int* INFO) {
       mout << "LAPACK not defined --> no dgetrs_ available" << endl;exit(0);
}
void zgetrs_(char* TRANS, int* N, int* NRHS, void* A, int* LDA, int* IPIV, void* B, int* LDB, int* INFO) {
       mout << "LAPACK not defined --> no zgetrs_ available" << endl;exit(0);
}

void dgemm_(char* TRANSA, char* TRANSB, int*M, int *N, int *K, double* alpha, void *A, int *LDA, void *B, int *LDB, double *beta, void *C, int *LDC) {
       mout << "LAPACK not defined --> no dgemm_ available" << endl;exit(0);
}
void zgemm_(char* TRANSA, char* TRANSB, int*M, int *N, int *K, double* alpha, void *A, int *LDA, void *B, int *LDB, double *beta, void *C, int *LDC) {
       mout << "LAPACK not defined --> no zgemm_ available" << endl;exit(0);
}

void dgemv_(char* TRANS, int* M, int* N, double* alpha, void* A, int* LDA, void* X, int* INCX, double* BETA, void* Y, int* INCY) {
       mout << "LAPACK not defined --> no dgemv_ available" << endl;exit(0);
}
void zgemv_(char* TRANS, int* M, int* N, double* alpha, void* A, int* LDA, void* X, int* INCX, double* BETA, void* Y, int* INCY) {
       mout << "LAPACK not defined --> no zgemv_ available" << endl;exit(0);
}

void dgees_(char* JOBVS, char* SORT, bool* SELECT, int* N, void *A, int* LDA, int* SDIM, void* WR, void* WI, void* VS, int* LDVS, 
            void* WORK, int* LWORK, bool* BWORK, int* INFO) {
       mout << "LAPACK not defined --> no dgees_ available" << endl;exit(0);
}

void dgesvd_(char* JOBU, char* JOBVT, int* M, int* N, void* A, int* LDA, double* S, void* U, int* LDU, void* VT, int* LDVT, void* WORK, void* LWORK, int* INFO) {
       mout << "LAPACK not defined --> no dgesvd_ available" << endl;exit(0);
}
void zgesvd_(char* JOBU, char* JOBVT, int* M, int* N, void* A, int* LDA, double* S, void* U, int* LDU, void* VT, int* LDVT, void* WORK, void* LWORK, int* INFO) {
       mout << "LAPACK not defined --> no zgesvd_ available" << endl;exit(0);
}

void dtrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG, int* M, int* N, double* alpha, void* A, int* LDA, void* B, int* LDB) {
       mout << "LAPACK not defined --> no dtrsm_ available" << endl;exit(0);
}
void ztrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG, int* M, int* N, void* alpha, void* A, int* LDA, void* B, int* LDB) {
       mout << "LAPACK not defined --> no ztrsm_ available" << endl;exit(0);
}




#endif

#endif
