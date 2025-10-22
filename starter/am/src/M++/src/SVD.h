#ifndef _SVD_H_
#define _SVD_H_

#include "Small.h"
#include "Lapdef.h"

#include <cmath>
#include <algorithm>
#include <valarray>

const double min_EPS = 1.e-15;

class SVD {
    Scalar* VT;
    Scalar* U;
    double* SIG;
    int n;
    int m;
    int rank;
    double eps;

    void CreateSVD(Scalar* _A, double* _SIG, Scalar* _U, Scalar* _VT) {
        int info;
        char jobu = 'S';
        char jobvt = 'S';
        int _m = n;
        int _n = m;
        int lda = _m;
        int min_mn = (_m < _n) ? _m : _n;
        int ldu = _m;
        int ldvt = min_mn;

        int lwork = ((5*min_mn) > (3*min_mn+((_m>_n)?_m:_n))) ? 5*min_mn : (3*min_mn+((_m>_n)?_m:_n));
        double work[lwork];

        GESVD(&jobu, & jobvt, &_m, &_n, _A, &lda, _SIG, _U, &ldu, _VT, &ldvt, work, &lwork, &info);
    }

public:
    void SVD_C(Scalar* S, int N, int M, double EPS = min_EPS) {
        eps = EPS; n = N; m = M;
        rank = (n < m) ? n : m;

        int rank_old = rank;
        int ldu = n;
        int ldvt = rank;
        int _m = n;
        int _n = m;

        Scalar* _VT;
        Scalar* _U;
        double* _SIG;
        _VT = new Scalar[ldvt*_n];
        _U = new Scalar[ldu*rank];
        _SIG = new double[rank];

        CreateSVD(S,_SIG,_U,_VT);

        rank = 1; // fuer rank = 0 muessen noch einige sachen geaendert werden.
        for (int i=rank_old-1; i>=0; --i)
            if (_SIG[i] > eps) {
                rank = i+1;
                break;
            }

        U = _U; _U = 0;

        if (!realloc(U,n*rank*sizeof(Scalar))) mout << "rank 0 - matrix!\n";

        if (rank < rank_old) {
            VT = new Scalar[rank*m];
            SIG = new double[rank];
            for (int i=0; i<m; ++i) {
                memcpy(VT+i*rank,_VT+i*rank_old,rank*sizeof(Scalar));
            }
            memcpy(SIG,_SIG,rank*sizeof(double));
            delete[] _SIG;
            delete[] _VT;
        } else {
            VT = _VT; _VT = 0;
            SIG = _SIG; _SIG = 0;
        }

//         mout << "rank: " << rank << endl;
    }
    SVD(const SmallMatrix S, double EPS = min_EPS): eps(EPS) {
        Scalar* save = new Scalar[S.rows()*S.cols()];
        for (int i=0; i<S.rows(); ++i)
            for (int j=0; j<S.cols(); ++j)
                save[S.rows()*j+i] = S[i][j];
        SVD_C(save, S.rows(), S.cols(), eps);
        delete[] save;
    }
    SVD(const SmallMatrixTrans S, double EPS = min_EPS): eps(EPS) {
        Scalar* save = new Scalar[S.rows()*S.cols()];
        for (int i=0; i<S.rows()*S.cols(); ++i)
            save[i] = S[0][i];
        SVD_C(save, S.rows(), S.cols(), eps);
        delete[] save;
    }
    SVD(Scalar* S, int N, int M, double EPS = min_EPS) {
        SVD_C(S,N,M,EPS);  // S will be "lost"! (trotzdem noch speicher loeschen?)
    }
    SVD(const SVD& S): n(S.n),m(S.m),rank(S.rank),eps(S.eps) {
//         VT = new Scalar[
        mout << "HIER SOLLTE ICH NICHT SEIN!\n";
    }

    void Destruct() {
        if (VT) delete[] VT; VT = 0;
        if (U) delete[] U; U = 0;
        if (SIG) delete[] SIG; SIG = 0;
    }

    ~SVD() {
        Destruct();
    }

    void Solve(Scalar* B, int size = 1, bool transposed = false) {
        Scalar* C = new Scalar[rank*size];

        char trans = 'T';
        char nottrans = 'N';
        double one = 1;
        double zero = 0;
        if (!transposed) {
            GEMM(&trans,&nottrans,&rank,&size,&n, &one, U ,&n, B, &n, &zero, C, &rank);
    
            for (int i=0; i<rank; ++i) {
                if (SIG[i] > 0) {
                    for (int j=0; j<size; ++j)
                        C[j*rank+i] /= SIG[i];
                } else {mout << "error: SIG[" << i << "] = 0!\n"; exit(0);}
            }
    
            GEMM(&trans,&nottrans,&m,&size,&rank, &one, VT, &rank, C, &rank, &zero, B, &m);
         }

         delete[] C;
    }

    void Solve(SmallVector& b) {
        Solve(b.ref());
    }

    Scalar* GetVT() {return VT;}
    Scalar* GetU() {return U;}
    double* GetSIG() {return SIG;}
    double* GetSIG() const {return SIG;}

    int getrank() {return rank;}
    int rows() {return n;}
    int cols() {return m;}

    void multiply(SVD svd) {
        int n1 = n;
        int r1 = rank;
        int m1 = m;
        int r2 = svd.rank;
        int m2 = svd.m;
        Scalar* C = new Scalar[r1*r2];
        char nottrans = 'N';
        double one = 1;
//         GEMM(&nottrans, &nottrans,&r1,&r2,&m1, &one, VT, &m1
    }

    template <class S> void matausg(S* a, int _i, int _j) {
        for (int i=0; i<_i; ++i) {
            for (int j=0; j<_j; ++j)
                mout << a[j*_i+i] << " ";
            mout << endl;
            }
    }

    void add(const SVD& b) {
        if (m != b.m || n != b.n) {mout << "error in size\n"; exit(0);}
        double* SIG1 = GetSIG();
        double* SIG2 = b.GetSIG();
        int r1 = rank;
        int r2 = b.rank;
        Scalar* U1U2 = new Scalar[n*(r1+r2)];
        size_t size1 = n*r1*sizeof(Scalar);
        size_t size2 = n*r2*sizeof(Scalar);
        memcpy(U1U2,U,size1);
        memcpy(U1U2+n*r1,b.U,size2);
        Scalar* VT1VT2 = new Scalar[(r1+r2)*m];
        size1 = r1*m*sizeof(Scalar);
        size2 = r2*m*sizeof(Scalar);
        Scalar* ptr = VT1VT2; 
        for (int i=0; i<m; ++i) {
            memcpy(ptr,VT+i*r1,r1*sizeof(Scalar));
            ptr += r1+r2;
        }
        ptr = VT1VT2+r1;
        for (int i=0; i<m; ++i) {
            memcpy(ptr,b.VT+i*r2,r2*sizeof(Scalar));
            ptr += r1+r2;
        }
        ptr = 0;

        SVD U12(U1U2,n,(r1+r2));
        int r3 = U12.getrank();
        SVD VT12(VT1VT2,(r1+r2),m);
        int r4 = VT12.getrank();
        delete[] U1U2;
        delete[] VT1VT2;

        Scalar *U3 = U12.GetU();
        double *SIG3 = U12.GetSIG();
        Scalar *VT3 = U12.GetVT();
        Scalar *U4 = VT12.GetU();
        double *SIG4 = VT12.GetSIG();
        Scalar *VT4 = VT12.GetVT();

        // multiply A5 = SIG3 * VT3 * (SIG1 0 ; 0 SIG2) * U4 * SIG4

        for (int i=0; i<r3; ++i) {
            for (int j=0; j<r1; ++j)
                VT3[j*r3+i] *= SIG3[i]*SIG1[j];
            for (int j=r1; j<r1+r2;++j)
                VT3[j*r3+i] *= SIG3[i]*SIG2[j-r1];
        }

        SIG1 = 0;
        SIG2 = 0;

        for (int i=0; i<(r1+r2); ++i)
            for (int j=0; j<r4; ++j)
                U4[j*(r1+r2)+i] *= SIG4[j];

        Scalar *A5 = new Scalar[r3*r4];

        char nottrans = 'N';
        int _rows = r3;
        int _cols = r4;
        int _rm = r1+r2;
        int lda = r3;
        int ldb = r1+r2;
        double one = 1.0;
        double zero = 0.0;
        GEMM(&nottrans,&nottrans,&_rows,&_cols,&_rm, &one, VT3, &lda, U4, &ldb, &zero, A5, &_rows);

        // SVD of A5
        // U = U3 * U5;   SIG = SIG5;   VT = V4 * V5

        SVD SVD5(A5,r3,r4);

        Scalar* U5 = SVD5.GetU();
        Scalar* VT5 = SVD5.GetVT();
        double* SIG5 = SVD5.GetSIG();
        int r5 = SVD5.getrank();

        _rows = n; _cols = r5; _rm = r3; lda = n; ldb = r3;
        if (U) delete[] U; U = 0;
        U = new Scalar[n*r5];
        GEMM(&nottrans,&nottrans,&_rows,&_cols,&_rm, &one, U3, &lda, U5, &ldb, &zero, U, &_rows);

        if (SIG) delete[] SIG; SIG = 0;
        SIG = new double[r5];
        for (int i=0; i<r5; ++i) SIG[i] = SIG5[i];

        _rows = r5; _cols = m; _rm = r4; lda = r5; ldb = r4;
        if (VT) delete[] VT; VT = 0;
        VT = new Scalar[r5*m];
        GEMM(&nottrans,&nottrans,&_rows,&_cols,&_rm, &one, VT5, &lda, VT4, &ldb, &zero, VT, &_rows);

        rank = r5;
    }

    void ausgabe() {
        mout << "_______ U _______\n";
        matausg(U,n,rank);
        mout << endl;
        mout << "____ SIGMA ____\n";
        matausg(SIG,1,rank);
        mout << endl;
        mout << "______ VT _______\n";
        matausg(VT,rank,m);
    }

    void originaltest() {
        for (int i=0; i<n; ++i) {
            for (int j=0; j<rank; ++j)
                U[j*rank+i] *= SIG[j];
        }

        char nottrans = 'N';
        double one = 1;
        double zero = 0;
        Scalar* test = new Scalar[n*m];
        GEMM(&nottrans,&nottrans,&n,&m,&rank, &one, U, &n, VT, &rank, &zero, test, &n);
        mout << "Originalmatrix:\n";
        matausg(test,n,m);
    }

};

#endif

