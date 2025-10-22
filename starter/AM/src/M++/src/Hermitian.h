// file:    Hermitian.h
// author: Christian Wieners
// $Header: /public/M++/src/Hermitian.h,v 1.2 2009-10-15 06:51:39 wieners Exp $

#ifndef _HERMITIAN_H_
#define _HERMITIAN_H_

#include "Tensor.h"

#include <cmath>
#include <algorithm>
#include <valarray>

typedef complex<double> Complex;

inline double norm2 (const Complex& z){return real(z)*real(z)+imag(z)*imag(z);}

class HermitianMatrix {
    Complex* a;
    int n;
    int m;
public:
    HermitianMatrix (int N) : n(N), m(N) { a = new Complex [n*n]; }
    HermitianMatrix (const HermitianMatrix& H) : n(H.n), m(H.m){ 
	a = new Complex [n*n]; 
	for (int i=0; i<n*n; ++i) a[i] = H.a[i];
    }
    HermitianMatrix (int N, int M) : n(N), m(M) { 
        a = new Complex [n*m]; 
        for (int i=0; i<n*m; ++i)  {
	  //real(a[i]) = 0;
	  //imag(a[i]) = 0;
	    a[i].real(0);
	    a[i].imag(0);
	}
        if (n*m==0) {n = 0; m = 0;}
    }
    HermitianMatrix (const HermitianMatrix& H,
		     const HermitianMatrix& G) : n(H.n) { 
	a = new Complex [n*n]; 
	for (int i=0; i<n; ++i) 
	    for (int j=0; j<n; ++j) {
		Complex* sum = a+(i*n+j);
		*sum = 0;
		for (int k=0; k<n; ++k) 
		    *sum += H[i][k] * G[k][j];
	    }
    }
    ~HermitianMatrix () { delete[] a; }
    void Identity() { 
	for (int i=0; i<n; i++)
	    for (int j=0; j<n; j++)
		a[i*n+j] = (i==j);
    }
    void Conj () { for (int i=0; i<n*n; i++) a[i] = conj(a[i]); }
    void Adj () { 
	for (int i=0; i<n; i++) {
	    a[i*n+i] = conj(a[i*n+i]);
	    for (int j=0; j<i; j++) {
		Complex s = a[i*n+j];
		a[i*n+j] = conj(a[j*n+i]);
		a[j*n+i] = conj(s);
	    }
	}
    }
    HermitianMatrix& operator = (Complex x) { 
        for (int i=0; i<n*m; ++i)  a[i] = x;
	return *this;
    }
    const Complex* operator [] (int i) const { return a+m*i; } 
    Complex* operator [] (int i) { return a+m*i; } 
    int rows() const { return n; }
    int cols() const { return m; }
    void Cholesky () {
	Complex* mat = new Complex [n*n];
	for (int i=0; i<n*n; ++i) mat[i] = a[i];
	for (int i=0; i<n; i++) {
	    double sum = real(mat[i*n+i]);
	    if (abs(imag(mat[i*n+i]))>Eps)
		Exit("non-hermitian matrix in Cholesky");
	    for (int k=0; k<i; k++)
		sum -= norm2(a[i*n+k]);
	    double piv = sqrt(sum);
	    if (abs(piv)<Eps) Exit("singular matrix in Cholesky");
	    a[i*n+i] = piv = 1.0 / piv;
	    for (int j=i+1; j<n; j++) {
		Complex sum = conj(mat[i*n+j]);
		for (int k=0; k<i; k++)
		    sum -= a[j*n+k] * conj(a[i*n+k]);
		a[j*n+i] = piv * sum;
	    }
	}
	delete[] mat;
    }
};
inline ostream& operator << (ostream& s, const HermitianMatrix& M) {
    char buf[128];
    for (int i=0; i<M.rows(); ++i, s << endl) 
	for (int j=0; j<M.cols(); ++j) {
	    sprintf(buf,"(%9.5f|%9.5f)",real(M[i][j]),imag(M[i][j]));	
	    s << buf;
	}
    return s;
}

class ComplexVector;

inline constAB<HermitianMatrix,ComplexVector> 
operator * (const HermitianMatrix& S, const ComplexVector& v){
    return constAB<HermitianMatrix,ComplexVector>(S,v);
}

class ComplexVector : public valarray<Complex> {
public:
    ComplexVector (int N) : valarray<Complex>(N) {} 
    ComplexVector& operator -= 
	(const constAB<HermitianMatrix,ComplexVector>& Au) {
	for (int j=0; j<Au.first().rows(); ++j) {
	    Complex s = 0;
	    for (int k=0; k<Au.second().size(); ++k) 
		s += Au.first()[j][k] * Au.second()[k];
	    (*this)[j] -= s;
	}
	return *this;
    }
    ComplexVector& operator = (Complex x) { 
        for (int i=0; i<size(); ++i) {
	  //real((*this)[i]) = real(x); 
	  //imag((*this)[i]) = imag(x);
	  (*this)[i].real(x.real());
	  (*this)[i].imag(x.imag());
	}
	return *this;
    }
    double norm () const { 
	double s = 0;
	for (int k=0; k<size(); ++k) s += norm2((*this)[k]);
	return sqrt(s); 
    } 
    friend ostream& operator << (ostream& s, const ComplexVector& v) {
	char buf[128];
	for (int i=0; i<v.size(); ++i) { 
	    sprintf(buf,"(%9.5f|%9.5f)",real(v[i]),imag(v[i]));	
	    s << buf;
	}
	return s << endl;
    }
    friend Complex operator * (const ComplexVector& u,const ComplexVector& v){
	Complex s = 0;
	for (int i=0; i<v.size(); ++i) s += u[i] * conj(v[i]);
	return s;
    }
};

class DoubleVector : public valarray<double> {
public:
    DoubleVector (int N) : valarray<double>(N) {} 
    DoubleVector (double a, int N) : valarray<double>(a,N) {} 
    void Resize (int N) {
	valarray<double> v = *this;
	resize(N);
	for (int i=0; i<N; ++i) 
	    (*this)[i] = v[i];
    }
    friend ostream& operator << (ostream& s, const DoubleVector& v) {
	char buf[128];
	for (int i=0; i<v.size(); ++i) { 
	    sprintf(buf," %11.8f",v[i]);	
	    s << buf;
	}
	return s << endl;
    }
};



extern "C" void zhegv_(int *ITYPE, char *JOBZ, 
		       char *UPLO, int *N, void *A, int *LDA, void *B,
		       int *LDB, void *W, void *WORK, int *LWORK, 
		       void *RWORK, int *INFO, int ch1, int ch2);

extern "C" void zheev_(char *JOBZ, 
		       char *UPLO, int *N, void *A, int *LDA, 
		       void *W, void *WORK, int *LWORK, 
		       void *RWORK, int *INFO, int ch1, int ch2);

extern "C" void zgetrf_(int *M, int *N, void *A, int *LDA, int *IPIV, int *INFO);

extern "C" void zgetri_(int *N, void *A, int *LDA, int *IPIV, void *WORK, void *LWORK, int *INFO);

struct _dcomplex { double re, im; };
typedef struct _dcomplex dcomplex;
extern "C" void zgesvd_( char* jobu, char* jobvt, int* m, int* n, void* a,
                int* lda, double* s, void* u, int* ldu, void* vt, int* ldvt,
                void* work, int* lwork, double* rwork, int* info );

extern "C" void zgeev_( char* jobvl, char* jobvr, int* n, void* a,
                int* lda, void* w, void* vl, int* ldvl, void* vr, int* ldvr,
                void* work, int* lwork, double* rwork, int* info );

inline void EVcomplex (const HermitianMatrix &a, const HermitianMatrix &b,
		       DoubleVector &lambda, HermitianMatrix &e) {

    int R = lambda.size();

    char jobz='V', uplo='U';
    int itype=1, n=R, lwork=2*n, info;
    double w[n], work[2*lwork], rwork[3*n];
    complex<double> A[n*n], B[n*n];

    for(int i=0;i<n;i++)
	for(int j=0;j<n;j++) A[i*n+j]=a[i][j];
    for(int i=0;i<n;i++)
	for(int j=0;j<n;j++) B[i*n+j]=b[i][j];
    
    zhegv_(&itype, &jobz, &uplo, &n, A, 
	   &n, B, &n, w, work, &lwork, rwork, &info, 1, 1);
    
    if (info!=0) { 
       mout<<"Error in LAPACK/ZHEGV: info="<<info<<endl;
       Exit("Eigensolver failed. Program stoped.");
    }
    for(int i=0;i<n;i++) lambda[i]=w[i];
    
    for(int i=0;i<n;i++)
	for(int j=0;j<n;j++) e[j][i]= A[i*n+j];
}

inline void EVcomplexO (const HermitianMatrix &a,
		       DoubleVector &lambda, HermitianMatrix &e) {

    int R = lambda.size();

    char jobz='V', uplo='U';
    int n=R, lwork=2*n, info;
    double w[n], work[2*lwork], rwork[3*n];
    complex<double> A[n*n];

    for(int i=0;i<n;i++)
	for(int j=0;j<n;j++) A[i*n+j]=a[i][j];
    
    zheev_(&jobz, &uplo, &n, A, 
	   &n, w, work, &lwork, rwork, &info, 1, 1);
    
    if (info!=0) { 
       mout<<"Error in LAPACK/ZHEEV: info="<<info<<endl;
       Exit("Eigensolver failed. Program stoped.");
    }
    
    for(int i=0;i<n;i++) lambda[i]=w[i];
    
    for(int i=0;i<n;i++)
	for(int j=0;j<n;j++) e[j][i]= A[i*n+j];
}

inline void ComplexMatrixInverse (HermitianMatrix& a, HermitianMatrix& inva) {
    int n = a.rows();
    complex<double> *A = new complex<double> [n*n];
    int I[n];
    int info;
    int lwork = 2*n;
    double work[2*lwork];
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++) A[i*n+j] = a[i][j];

    zgetrf_(&n, &n, A, &n, I, &info);
    zgetri_(&n, A, &n, I, work, &lwork, &info); 
    
    for(int i=0;i<n;i++)
      for(int j=0;j<n;j++) inva[i][j] = A[i*n+j];
    
    if (info!=0) { 
       mout << "Error in LAPACK/CGESV: info=" << info << endl;
       Exit("ComplexMatrixsolver failed. Program stoped.");
    }
    delete [] A;
}

inline void ComplexMatrixSolver (HermitianMatrix& a, ComplexVector& b) {
    int n = a.rows();
    HermitianMatrix inva(n);
    ComplexVector tempb(n);

    ComplexMatrixInverse(a,inva);
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++) tempb[i] += inva[i][j]*b[j];
    for (int i=0; i<n; i++)
    b[i] = tempb[i];
}

inline void ComplexMatrixSVD (HermitianMatrix& A, HermitianMatrix& v, double* s, HermitianMatrix& w) {
    int M = A.rows();
    int N = A.cols();
    int LDA = M;
    int LDU = M;
    int LDVT = N;
    int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;

    //double s[min(M,N)];
    double rwork[5*min(M,N)];
    complex<double> *u = new complex<double> [LDU*M];
    complex<double> *vt = new complex<double> [LDVT*N];
    complex<double> *a = new complex<double>[LDA*N];

    lwork = 3*min(M,N)+max(M,N);
    complex<double> *work[2*lwork];

    for(int i=0;i<N;i++)
        for(int j=0;j<M;j++) a[i*M+j] = A[j][i];

    zgesvd_("All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
	    rwork, &info );
    
    for(int i=0;i<M;i++)
        for(int j=0;j<M;j++) v[i][j] = u[j*M+i];
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) w[i][j] = vt[j*N+i];

    delete [] a;
    delete [] u;
    delete [] vt;

    if( info > 0 ) {
        printf( "The algorithm computing SVD failed to converge.\n" );
	exit( 1 );
    }
}

inline void ComplexMatrixEV (HermitianMatrix& A, ComplexVector& ev) {
    int N = A.rows();
    int LDA = N;
    int LDVL = N;
    int LDVR = N;
    int n = N, lda = LDA, ldvl = LDVL, ldvr = LDVR, info, lwork;
    lwork = 2*N;
    complex<double> *work[2*lwork];
    double rwork[2*N];
    complex<double> *w = new complex<double> [N];
    complex<double> *vl = new complex<double> [LDVL*N];
    complex<double> *vr = new complex<double> [LDVR*N];
    complex<double> *a = new complex<double>[LDA*N];
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) a[i*n+j] = A[i][j];
    zgeev_( "Vectors", "Vectors", &n, a, &lda, w, vl, &ldvl, vr, &ldvr,
	   work, &lwork, rwork, &info );
    for (int i=0; i<N; i++)
        ev[i] = w[i];
    delete [] w;
    delete [] vl;
    delete [] vr;
    delete [] a;
}
#endif
