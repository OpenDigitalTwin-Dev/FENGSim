// file:   Small.C
// author: Christian Wieners, Antje Sydow
// $Header: /public/M++/src/Small.C,v 1.11 2009-10-28 15:18:56 maurer Exp $

#include "Small.h"
#include "Sparse.h"
#include "Schur.h"
#include "Lapdef.h"

SmallMatrix::SmallMatrix (Scalar x, int N, int M) : n(N), m(M) { 
    a = new Scalar [n*m]; 
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            a[i*m+j] = double(i==j) * x;
}

SmallMatrix::SmallMatrix (const SmallMatrix& A) : n(A.n), m(A.m) { 
    if (&A == this) return;
    a = new Scalar [n*m]; 
    for (int i=0; i<n*m; ++i) a[i] = A.a[i]; 
}

SmallMatrix::SmallMatrix (int N, int M, const SmallMatrix& A) : n(N), m(M) { 
    a = new Scalar [n*m]; 
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) 
            a[i*m+j] = A[i][j];
}

SmallMatrix::SmallMatrix (const SmallMatrix& A, const SmallMatrix& B) : 
    n(A.n),m(B.m) { 
    a = new Scalar [n*m]; 
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) {
            Scalar s = 0;
            for (int k=0; k<A.m; ++k) s += A[i][k] * B[k][j]; 
            a[i*m+j] = s;
        }
}

SmallMatrix::SmallMatrix (const SparseMatrix& M, const vector<int>& ind) :
    n(ind.size()), m(ind.size()) {
    a = new Scalar [n*m];
    for (int i=0; i<n; ++i)
	for (int j=0; j<m; ++j) {
	    int e = M.find(ind[i],ind[j]);
	    if (e == -1)
		a[i*m+j] = 0.0;
	    else
		a[i*m+j] = M.nzval(e);
	}
}

SmallMatrix::SmallMatrix (const SparseMatrix& M) : n(M.size()), m(M.size()) {
    a = new Scalar [n*m];
    for (int i=0; i<n; ++i)
	for (int j=0; j<m; ++j) {
	    int e = M.find(i,j);
	    if (e == -1)
		a[i*m+j] = 0.0;
	    else
		a[i*m+j] = M.nzval(e);
	}
}

SmallMatrix& SmallMatrix::operator = (Scalar x) { 
    for (int i=0; i<n*m; ++i) a[i] = x; 
    return *this;
}

SmallMatrix& SmallMatrix::operator -= (const SmallMatrix& A) { 
    for (int i=0; i<A.n; ++i)
        for (int j=0; j<A.m; ++j) 
            a[i*m+j] -= A[i][j];
    return *this;
}

SmallMatrix& SmallMatrix::operator += (const SmallMatrix& A) { 
    for (int i=0; i<A.n; ++i)
        for (int j=0; j<A.m; ++j) 
            a[i*m+j] += A[i][j];
    return *this;
}


SmallMatrix& SmallMatrix::operator *= (const SmallMatrix& A) { 
    Scalar* save;
    save = new Scalar [n*m]; 
    for (int i=0; i<n*m; ++i) save[i] = a[i];
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) {
            Scalar s = 0;
            for (int k=0; k<A.m; ++k) s += save[i*m+k] * A[k][j]; 
            a[i*m+j] = s;
        }
    delete[] save;
    return *this;
}

SmallMatrix& SmallMatrix::operator *= (double b) { 
    for (int i=0; i<n*m; ++i) a[i] *= b;
    return *this;
}

double SmallMatrix::norm () const { 
    double s = 0;
    for (int k=0; k<m*n; ++k) s += abs(a[k])*abs(a[k]); 
    return sqrt(s); 
}


#ifdef LAPACK
extern "C" void dgetrf_(int *M, int *N, void *A, int *LDA, int *IPIV, int *INFO);
extern "C" void dgetri_(int *N, void *A, int *LDA, int *IPIV, void *WORK, void *LWORK, int *INFO);

extern "C" void zgetrf_(int *M, int *N, void *A, int *LDA, int *IPIV, int *INFO);
extern "C" void zgetri_(int *N, void *A, int *LDA, int *IPIV, void *WORK, void *LWORK, int *INFO);

extern "C" void dtrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG, int *M, int *N, double* alpha, void *A, int *LDA, void *B, int *LDB);

#endif

void SmallMatrix::invert() { 

if (n != m) exit(0);

#ifdef LAPACK
 
#ifdef DCOMPLEX
   complex<double>* A = a;
#else
    double* A = a;
#endif

    int I[n];
    int info;
    int lwork = 2*n;
    double work[2*lwork];
    
#ifdef DCOMPLEX
    zgetrf_(&n, &n, A, &n, I, &info);
#else
    dgetrf_(&n, &n, A, &n, I, &info);
#endif
    if (info!=0) mout<<"Error in DGETRF: info="<<info<<endl;

#ifdef DCOMPLEX
    zgetri_(&n, A, &n, I, work, &lwork, &info); 
#else
    dgetri_(&n, A, &n, I, work, &lwork, &info);
#endif

    if (info!=0) mout<<"Error in DGETRI: info="<<info<<endl;

    A = 0;

#else
    int* ipv = new int [n];
    Scalar* Mat = new Scalar [n*n];
    Scalar* rhs = new Scalar [n];
    for (int i=0; i<n; ++i) ipv[i] = i;
    for (int i=0; i<n*n; ++i) Mat[i] = a[i];
    for (int i=0; i<n; ++i) {
        int k = i;
        double piv = abs(Mat[i*n+i]);
        for (int j=i+1; j<n; ++j) {
            double sum = abs(Mat[j*n+i]);
            if (sum > piv) {
                k = j;
                piv = sum;
            }
        }
        if (k != i) {
            swap(ipv[i],ipv[k]);
            for (int j=0; j<n; ++j) swap(Mat[k*n+j],Mat[i*n+j]);
        }
        Scalar dinv = Mat[i*n+i];
        if (abs(dinv)<1e-12) {
            cerr << "Error in SmallMatrix Inversion; file: Small.h\n";
            Assert(0);
        }	  
        dinv = Mat[i*n+i] = 1.0/dinv;
        for (int j=i+1; j<n; ++j) {
            Scalar piv = (Mat[j*n+i] *= dinv);
            for (int k=i+1; k<n; ++k)
                Mat[j*n+k] -= Mat[i*n+k] * piv;
        }
    }
    for (int k=0; k<n; ++k) {
        for (int i=0; i<n; ++i)
            rhs[i] = 0;
        rhs[k] = 1.0;
        for (int i=0; i<n; ++i) {
            Scalar sum = rhs[ipv[i]];
            for (int j=0; j<i; ++j)
                sum -= Mat[i*n+j] * a[j*n+k];
            a[i*n+k] = sum;	// Lii = 1 
        }
        for (int i=n-1; i>=0; i--) {
            Scalar sum = a[i*n+k];
            for (int j=i+1; j<n; ++j)
                sum -= Mat[i*n+j] * a[j*n+k];
            a[i*n+k] = sum * Mat[i*n+i];	// Uii = Inv(Mii) 
        }
    }	
    delete[] rhs;
    delete[] Mat;
    delete[] ipv;
#endif
}

void SmallMatrix::Accumulate() { PPM->Sum(a,n*m); }

int SmallMatrix::nz() const {
    int k = n;
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) 
            if (i!=j)
                if (abs(a[i*m+j]) != 0.0)
                    ++k;
    return k;
}

void SmallMatrix::copy (Scalar* nzval, int* rowind, int* colptr) const {
    int d = 0;
    for (int i=0; i<n; ++i) {
        nzval[d] = a[i*m+i];
        colptr[d] = i;
        rowind[i] = d;
        ++d;
        for (int j=0; j<m; ++j) 
            if (i!=j) 
                if (abs(a[i*m+j]) != 0.0) {
                    nzval[d] = a[i*m+j];
                    colptr[d] = j;
                    ++d;
                }
    }
    rowind[n] = d;
}

ostream& operator << (ostream& s, const SmallMatrix& M) {
    char buf[128];
    for (int i=0; i<M.rows(); ++i, s << endl) 
	for (int j=0; j<M.cols(); ++j) {
	    #ifdef NDOUBLE
	    s << " " << M[i][j];
	    #else
	    sprintf(buf,"%9.5f",M[i][j]);	
	    s << buf;
	    #endif
	}
    return s;
}

SmallMatrix& SmallMatrix::operator -= (const constAB<SmallMatrix,SmallMatrix>& AB) {
    int n = AB.first().rows();
    int m = AB.second().cols();
    int q = AB.first().cols();
    for (int i=0; i<n; ++i)
        for (int j=0; j<m; ++j) {
            Scalar s = 0;
            for (int k=0; k<q; ++k) s += AB.first()[i][k] * AB.second()[k][j]; 
            a[i*m+j] -= s;
        }
    return *this;
}

constAB<SmallMatrix,SmallVector> operator * (const SmallMatrix& S, 
						    const SmallVector& v){
    return constAB<SmallMatrix,SmallVector>(S,v);
}

constAB<DDProblem,SmallVector> operator * (const DDProblem& S, 
						    const SmallVector& v){
    return constAB<DDProblem,SmallVector>(S,v);
}


constAB<SmallMatrix,SmallMatrix> operator * (const SmallMatrix& A, 
				             const SmallMatrix& B){
    return constAB<SmallMatrix,SmallMatrix>(A,B);
}


// ----------------------------------------------------
//    SmallMatrixTrans
// ----------------------------------------------------

void SmallMatrixTrans::makeLU() {
    int n=rows();int m=cols();
    IPIV = new int[n];
    int info;
    Scalar* A; A = ref();
    char uplo = 'L';
//     symmetric = false;
    if (!symmetric)
        GETRF(&m, &m, A, &m, IPIV, &info);
    if (symmetric) {
        POTRF(&uplo, &m, A, &m, &info);
    }
    if (info!=0) mout <<"Error in GETRF (LU): info=" << info << " in file " << __FILE__ << " on line " << __LINE__ << endl;
    lu = true;
}

void SmallMatrixTrans::invert() {
    if (!lu) makeLU();
    int n=rows();int m=cols();
    Scalar* A = ref();
    int lwork = 2*n;
    double work[2*lwork];
    int info;
    char uplo = 'L';
    if (!symmetric)
        GETRI(&m, A, &m, IPIV, work, &lwork, &info);
    if (symmetric) {
        POTRI(&uplo, &m, A, &m, &info);
    }
    if (info!=0) mout <<"Error in GETRI (LU): info = " << info << " in file " << __FILE__ << " on line " << __LINE__ << endl;
    inverted = true;
}

void SmallMatrixTrans::Solve(Scalar* v, int size, bool trans) {
    int n=rows();int m=cols();
    if (!inverted) {
        char transa = 'N';
        int info = 0;
        Scalar* A = ref();
        if (!symmetric) {
            GETRS(&transa, &n, &size, A, &n, IPIV, v, &n, &info);
        }
        if (symmetric) {
            char side = 'L';
            char uplo = 'L';
            if (trans) transa = 'T';
            char diag = 'N';
            Scalar one = 1;
           TRSM(&side,&uplo,&transa,&diag,&n, &size, & one, A, &n, v, &n);
//             POTRS(&uplo, &n, &size, A, &n, v, &n, &info);
        }
        if (info!=0) mout <<"Error in GETRS (Solve(V)): info=" << info << " in file " << __FILE__ << " on line " << __LINE__ << endl;
    } else {
        char trans = 'N';
        double one = 1;
        double dnull = 0;
        int inull = 1;
        Scalar* A = ref();
        Scalar* B = new Scalar[m*size];
        for (int i=0; i<m*size; ++i) B[i] = v[i];
        if (!symmetric)
            GEMM(&trans, &trans, &n, &size, &m, &one, A, &n, B, &m, &dnull, v, &n);
        char side = 'L';
        char uplo = 'L';
        if (trans) uplo = 'U';
        if (symmetric)
            SYMM(&side, &uplo, &n, &size, &one, A, &n, B, &m, &dnull, v, &n);
        delete[] B;
    }
}

void SmallMatrixTrans::SetIPIV(Buffer& B) {
    B >> symmetric;
    B >> lu;
    B >> inverted;
    if ((symmetric) || inverted) return;
    int n = rows();
    if (!IPIV) IPIV = new int[n];
    size_t size = n*sizeof(int);
    B.read(*IPIV,size);
    B >> inverted;
}

void SmallMatrixTrans::SendIPIV(Buffer& B) {
    B << symmetric;
    B << lu;
    B << inverted;
    if ((symmetric) || (inverted)) return;
    int n = rows();
    size_t size = n*sizeof(int);
    B.fill(*IPIV, size);
    B << inverted;
}

size_t SmallMatrixTrans::GetSizeSendIPIV() {
    if ((symmetric) || (inverted)) return 3*sizeof(bool);
    return 3*sizeof(bool)+rows()*sizeof(int)+sizeof(bool);
}

ostream& operator << (ostream& s, const SmallMatrixTrans& M) {
    char buf[128];
    for (int i=0; i<M.rows(); ++i, s << endl) 
	for (int j=0; j<M.cols(); ++j) {
	    #ifdef NDOUBLE
	    s << " " << M(i,j);
	    #else
	    sprintf(buf,"%9.5f",M(i,j));	
	    s << buf;
	    #endif
	}
    return s;
}


// ----------------------------------------------------
//    SmallVector
// ----------------------------------------------------

SmallVector::SmallVector (const SmallMatrix& A, const SmallVector& u) 
	: valarray<Scalar>(A.rows()) {
    for (int j=0; j<size(); ++j) {
        Scalar s = 0;
        for (int k=0; k<u.size(); ++k) 
            s += A[j][k] * u[k];
        (*this)[j] = s;
    }
}

SmallVector::SmallVector (const SmallVector& u, const SmallMatrix& A) 
	: valarray<Scalar>(A.cols()) {
    for (int j=0; j<size(); ++j) {
        Scalar s = 0;
        for (int k=0; k<u.size(); ++k) 
            s += A[k][j] * u[k];
        (*this)[j] = s;
    }
}

SmallVector& 
SmallVector::operator = (const constAB<SmallMatrix,SmallVector>& Au) {
    SmallVector b(Au.second().size());
    for (int j=0; j<Au.first().rows(); ++j) {
        Scalar s = 0;
        for (int k=0; k<Au.second().size(); ++k) 
            s += Au.first()[j][k] * Au.second()[k];
	b[j] = s;
    }
    for (int i=0; i<Au.second().size(); ++i) (*this)[i] = b[i];
    return *this;
}

SmallVector& 
SmallVector::operator -= (const constAB<SmallMatrix,SmallVector>& Au) {
    for (int j=0; j<Au.first().rows(); ++j) {
        Scalar s = 0;
        for (int k=0; k<Au.second().size(); ++k) 
            s += Au.first()[j][k] * Au.second()[k];
        (*this)[j] -= s;
    }
}

SmallVector& 
SmallVector::operator -= (const constAB<DDProblem,SmallVector>& Au) {
    MultiplySubtract(Au.first(),Au.second(),*this);
    return *this;
}

SmallVector& SmallVector::operator = 
(const constAB<constAB<SmallOperator,Vector>,Vector> & Au) {
    (Au.first().first()).multiply(Au.first().second(), Au.second(), *this);
    return *this;
}

SmallVector& SmallVector::operator *= (const SmallMatrix& A) {
    Scalar* save;
    int n = A.rows();
    int m = A.cols();
    save = new Scalar [n];
    for (int i=0; i<n; ++i) save[i] = (*this)[i];
    (*this).resize(m); 

    for (int j=0; j<m; ++j) {
        Scalar s = 0;
        for (int k=0; k<n; ++k) 
             s += A[j][k] * save[k]; 
        (*this)[j] = s;
    }
    delete[] save;
    return *this;
}

SmallVector& SmallVector::operator *= (const int& a) {
    for (int i=0; i<size(); ++i) (*this)[i] *= a;
    return *this;
}
SmallVector& SmallVector::operator *= (const Scalar& a) {
    for (int i=0; i<size(); ++i) (*this)[i] *= a;
    return *this;
}

SmallVector& SmallVector::operator = (const SmallVector& v) {
    for (int i=0; i<v.size(); ++i) (*this)[i] = v[i];
    return *this;
}

SmallVector& SmallVector::operator = (const Scalar& a) {
    for (int i=0; i<size(); ++i) (*this)[i] = a;
    return *this;
}

SmallVector& SmallVector::operator += (const SmallVector& v) {
    for (int i=0; i<v.size(); ++i) (*this)[i] += v[i];
    return *this;
}

SmallVector& SmallVector::operator -= (const SmallVector& v) {
    for (int i=0; i<v.size(); ++i) (*this)[i] -= v[i];
    return *this;
}

double SmallVector::norm () const { 
    double s = 0;
    for (int k=0; k<size(); ++k) s += abs((*this)[k])*abs((*this)[k]);
    return sqrt(s); 
}

void SmallVector::Accumulate() { 
    int n = size();
    Scalar* a = new Scalar [n];
    for (int i=0; i<n; ++i) a[i] = (*this)[i];
    PPM->Sum(a,n); 
    for (int i=0; i<n; ++i) (*this)[i] = a[i];
    delete[] a;
}

ostream& operator << (ostream& s, const SmallVector& v) {
    char buf[128];
    for (int i=0; i<v.size(); ++i) { 
#ifdef NDOUBLE
	s << " " << v[i];
#else
	sprintf(buf,"%9.4f",v[i]);	
	s << buf;
#endif
    }
    return s << endl;
}

Scalar operator * (const SmallVector& u, const SmallVector& v) {
    Scalar s = 0;
    for (int i=0; i<v.size(); ++i) s += conj(u[i]) * v[i];
    return s;
}

SmallVector operator + (const SmallVector& u, const SmallVector& v) {
    SmallVector z(u); return z += v;
}

SmallVector operator * (const SmallVector& u, const SmallMatrix& v) {
    SmallVector z(u.size()); 
    for (int i=0; i<v.cols(); i++) {
        for (int j=0; j<v.rows(); j++) {
	    z[i] += u[j]*v[j][i];
	}
    }
    return z;
}

SmallVector operator - (const SmallVector& u, const SmallVector& v) {
    SmallVector z(u); return z -= v;
}

SmallVector operator * (Scalar d, const SmallVector& v) {
    SmallVector z(v); return z *= d;
}


// ----------------------------------------------------
//    SmallFunction
// ----------------------------------------------------

void SmallFunction::Jacobian (const SmallVector& x, SmallMatrix& J)  const { 
    // numerical computation of the Jacobian J
    int n = x.size();
    SmallVector d(n);
    SmallVector xh(x);
    SmallVector xhh(x);
    for (int j=0; j<n; ++j) {
	xh[j] += h;
	xhh[j] -= h;
        d = eval(xh) - eval(xhh);
        d *= 1/(2*h);
        for (int i=0; i<n; ++i) 
            J[i][j] = d[i];
	xh[j] -= h;
	xhh[j] += h;
    }
}

// ----------------------------------------------------
//    SmallNewtonMethod
// ----------------------------------------------------

void SmallNewtonMethod::operator () (SmallVector& x, SmallFunction& f) const { 
    double d, d1;
    int n = x.size();
    double eps = Eps + Red * x.norm();
    SmallVector r(n);
    r = f(x);
    d = r.norm();
    double d_previous = d;
    int LS_cnt = 0;
    SmallMatrix J(n,n);
    for (int iter = 0; iter < max_iter; ++iter) {
        vout(3) << "  SmallNewton: r(" << iter << ")= " << r << endl;
        if (d < eps) {
            Vout(3)<<"  SmallNewtonMethod:  d("<< iter <<") = "<< d <<endl;
            return;
        }
        vout(4) << "  SmallNewtonMethod:  d("<< iter <<") = " << d << endl;
	    
        // compute Jacobian J and determine correction c
        f.Jacobian(x,J);
        vout(9) << "  SmallNewtonMethod:  J("<< iter <<") =\n"<< J << endl;
        J.invert();
        SmallVector c(J,r);
        vout(6) << "  SmallNewtonMethod:  c("<< iter <<") = " << c << endl;
        vout(7) << "  SmallNewtonMethod: _x("<< iter <<") = " << x << endl;
        x -= c;
        vout(5) << "  SmallNewtonMethod: x_("<< iter <<") = " << x << endl;
	    
        // residual
        r = f(x);
        d = r.norm();
        if (d > d_previous) {
            for (int l=1; l<=LS_iter; ++l) {
                vout(2) << "  SmallNewtonMethod line search " << l 
                        << ": d("<< iter <<") = " << d << endl;
                c *= 0.5;
                x += c;
                r = f(x);
                d = r.norm();
            }
        }
        if (d > d_previous) {
            vout(5) << "  SmallNewtonMethod: line search unsuccessful\n"
                    << endl;
            ++LS_cnt;
            if (LS_cnt == 3) {
                Vout(1) << "  SmallNewtonMethod: " 
                        << "Too many line searches unsuccessful."
                        << endl;
                break;
            }
        }
        d_previous = d;
    }
    if (d > eps) {
        vout(0) << "  No convergence in SmallNewtonMethod;"
                << " Defect = " << d << "  epsilon = " << eps << endl; 
        vout(1) << " f(x) = " << f(x);
    }
}


class DirectSolver: public SmallSolver {
    int n;
    SmallMatrix* S;
public:
    DirectSolver(SmallMatrix& M) {
        n = M.rows();
        S = new SmallMatrix(M);
        (*S).invert();
    }
    void Solve (Scalar* x) {
        SmallVector* x_copy = new SmallVector(n);
        for (int i = 0; i<n; ++i) (*x_copy)[i] = x[i];
        SmallVector* loesung = new SmallVector(*S,*x_copy);
        for (int i = 0; i<n; ++i) x[i] = (*loesung)[i];
        delete x_copy;
        delete loesung;
    }
    ~DirectSolver() {
        delete S;
    }
};

SmallSolver* GetSmallSolver (SmallMatrix& S) {
    return new DirectSolver(S);
}

SmallSolver* GetSmallSolver (SmallMatrix& S, const string& name) {
    if (name == "DirectSolver") return new DirectSolver(S);
}

SmallMatrix SmallSolver::operator () (const SmallMatrix& m) {
    SmallMatrix S(m);
    S.invert();
    return S;
}

/*extern "C" void dgels_(char *trans, int *m, int *n, int *nrhs, double *a
			   , int *lda, double *b, int *ldb, double *work,
			   int *lwork, int *info);
*/
/*SmallVector SmallSolver::operator () (const SmallMatrix& a, const SmallVector& b) {
    int M = a.rows();
    int N = a.cols();
    int NRHS = 1;
    int LDA = M;
    int LDB = M;
    
    int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
    int lwork = 2*M;
    double work[lwork];

    double A[LDA*N];
    double B[LDB*NRHS];
    for (int j=0; j<N; j++) {
        for (int i=0; i<M; i++) {
	    A[j*M+i] = a[i][j];
	}
    }
    for (int i=0; i<M; i++) B[i] = b[i];

    dgels_( "No transpose", &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, &info );
    SmallVector r(M);
    for (int i=0; i<M; i++) r[i] =  B[i];
    return r;
    }*/
