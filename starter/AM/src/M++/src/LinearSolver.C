// file: LinearSolver.C
// author: Christian Wieners
// $Header: /public/M++/src/LinearSolver.C,v 1.22 2009-09-21 17:34:28 wieners Exp $

#include "LinearSolver.h"
#include <valarray>

class LS : public IterativeSolver {
public:
    LS (const string& prefix = "Linear") : IterativeSolver(prefix) {}
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	vout(11) << "u " << u;
	d = d_0 = r.norm();
	double eps = Eps + Red * d;
	Vector c(u);
	Vector p(u);
	Vector t(u);
	p = 0;
	Scalar rho_0 = 1;
	for (iter=0; iter<max_iter; ++iter) {
	    if (d < eps) break;
	    vout(1) << "  LS: d(" << iter << ")= " << d << endl; 
	    vout(10) << "r " << r;
	    c = B * r;
	    vout(10) << "c " << c;
	    u += c;
	    vout(10) << "u " << u;
	    r -= A * c;
	    d = r.norm();
	}
	Vout(0) << "  " << IterationEnd();
    }
    string Name () const { return "LS"; }
};

class CG : public IterativeSolver {
public:
    CG (const string& prefix = "Linear") : IterativeSolver(prefix) {}
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	d = d_0 = r.norm();
	double eps = Eps + Red * d;
	Vector c(u);
	Vector p(u);
	Vector t(u);
	p = 0;
	Scalar rho_0 = 1;
	for (iter=0; iter<max_iter; ++iter) {
	    if (d < eps) break;
	    vout(1) << "  PCG: d(" << iter << ")= " << d << endl;
	    vout(10) << "r " << r;


	   
	    
	    c = B * r;
	    vout(10) << "c " << c;
	    Scalar rho_1 = r * c;



	    
	    vout(2) << "rho " << r * c << " " << c * r << endl;


	   

	    p *= (rho_1 / rho_0);
	    p += c;
	    rho_0 = rho_1;
	    t = A * p;
	    Scalar alpha = rho_0 / (p * t);
	    vout(3) << "alpha " << p * t << " " << t * p << endl;


	   

	    
	    u += alpha * p;
	    vout(10) << "u " << u;
	    r -= alpha * t;
	    d = r.norm();


	    
	   

	}
	Vout(0) << "  " << IterationEnd();
    }
    string Name () const { return "PCG"; }
};

class BiCGStab : public IterativeSolver {
    int cnt;
    Scalar rho,alpha,omega;
    int restart;
 public:
    BiCGStab (const string& prefix = "Linear") : IterativeSolver(prefix),
	restart(100) {} 
    void Restart (Vector& a, Vector& b) {
	cnt = 0; rho = alpha = omega = 1; a = 0; b = 0;
    }
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	d = d_0 = norm(r);
	double epsilon = max(Eps,d*Red);
	Vector p(0.0,u), v(0.0,u),  y(0.0,u), t(0.0,u), r0(r);
	Restart(p,v);
	for (iter=0; iter<max_iter; ++iter, ++cnt) {
	    if (d < epsilon) break;
	    Vout(1) << "  BiCGStab: d(" << iter << ")= " << d << "\n";
	    Scalar rho_new = r0 *  r;
	    if ((abs(rho) < 1e-8 * abs(rho_new)) || 
		(abs(omega) < 1e-8 * abs(alpha))) {
		Vout(2) << "BiCGStab : breakdown occured (rho = "
			<< rho << " , omega = " << omega << ")" << "\n";
		Restart(p,v); continue;
	    }
	    else if (cnt >= restart) Restart(p,v);
	    Scalar beta = (rho_new/rho) * (alpha/omega);
	    p -= omega * v;
	    p *= beta;
	    p += r;
	    y = B * p;
	    v = A * y;
	    Scalar h = r0 * v;
	    if (abs(h) < 1e-8 * abs(rho_new)) {
		Vout(2) << "BiCGStab : breakdown occured (h=" <<h<< ")" <<"\n";
		Restart(p,v); continue;
	    }
	    alpha = rho_new / h;
	    u += alpha * y;
	    r -= alpha * v;
	    d = norm(r);
	    ++iter;
	    if (d < epsilon) break;
	    y = B * r;
	    t = A * y;
	    omega = (t * r) / (t * t);
	    u += omega * y;
	    r -= omega * t;
	    rho = rho_new;
	    d = norm(r);
	}
	
	Vout(0) << "  " << IterationEnd();
    }
    string Name () const { return "BiCGStab"; }
};

class BiCGStab2 : public IterativeSolver {
 public:
    BiCGStab2 (const string& prefix = "Linear") : IterativeSolver(prefix) {} 
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	d_0 = norm(r);
	d = d_0;
	double epsilon = max(Eps,d*Red);
	Vector y = r;
	Scalar delta = y * r;
	Vector s = r;
	for (iter = 0; iter<max_iter; ++iter) {
	    vout(2) << iter << " delta " << delta << endl;
	    if (abs(delta) < epsilon) {
		y = r;
		delta = y * r;
	    }
	    if (d < epsilon) break;
	    Vout(1) << "  BiCGStab: d(" << iter << ")= " << d << "\n";
	    Vector Bs = B * s;
	    Vector ABs = A * Bs;
	    Scalar phi = y * ABs / delta;
	    vout(2) << iter << " phi " << delta << endl;
	    if (abs(phi) < epsilon) break;
	    Scalar omega = 1.0 / phi;
	    u += omega * Bs;
	    Vector w = r;
	    w -= omega * ABs;
	    Vector Bw = B * w;
	    Vector ABw = A * Bw;
	    Scalar chi = (ABw * w) / (ABw * ABw);
	    vout(2) << iter << " chi " << chi << endl;
	    if (abs(chi) < epsilon) break;
	    r = w;
	    u += chi * Bw;
	    r -= chi * ABw;
	    s -= chi * ABs;
	    Scalar delta_new = y * r;
	    Scalar psi = - omega * delta_new / (delta * chi);
	    vout(2) << iter << " psi " << psi << endl;
	    if (abs(psi) < epsilon) break;
	    delta = delta_new;
	    s *= -psi;
	    s += r;
	    d = norm(r);
	}
	Vout(0) << "  " << IterationEnd();
    }
    string Name () const { return "BiCGStab"; }
};

class GMRES : public IterativeSolver {
 public:
    GMRES (const string& prefix = "Linear") : IterativeSolver(prefix) {}  
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	const int M = 100;
	d = d_0 = norm(r);
	double epsilon = max(Eps,Red * d);
	Scalar H[M+1][M], e[M+1], y[M], cs[M], sn[M];
	vector<Vector> v(M+1,u);
	Vector c(r), t(r);
	for (iter=0; iter < max_iter;) {
	    if (d < epsilon) break;
	    Vout(1) << "  GMRES: d(" << iter << ")= " << d << "\n";
	    v[0] = (1/d) * r;
	    e[0] = d; for (int i=1; i<M; ++i) e[i] = 0;
	    int k = 0;
	    for (; (k<M) && (iter<max_iter); ++k) {
		c = B * v[k];
		t = A * c;
		Vout(2) << "  GMRES: " << k 
			<< " c " << norm(c)
			<< " Ac " << norm(t) << "\n";
		v[k+1] = t;
		for (int i=0; i<=k; ++i) {
		    H[i][k] = t * v[i];
		    v[k+1] -= H[i][k] * v[i];
		}
		H[k+1][k] = norm(v[k+1]);
		Vout(3) << "  GMRES: " << k 
			<< " |v_k+1| " << H[k+1][k] << "\n";
		v[k+1] *= (1.0/H[k+1][k]);
		for (int i=0; i<k; ++i) {
		    Scalar giv_r = H[i][k];
		    Scalar giv_h = H[i+1][k];
		    H[i][k]   = sn[i]*giv_r + cs[i]*giv_h;
		    H[i+1][k] = cs[i]*giv_r - sn[i]*giv_h;
		}            
		Scalar giv_r = H[k][k];
		Scalar giv_h = H[k+1][k];
		Scalar co,si;
		d = sqrt(real(conj(giv_r)*giv_r+giv_h*giv_h));
		Vout(3) << "  GMRES: " << k << " d " << d << "\n";
		if (real(giv_r) > 0) {
		    co = giv_h / d;
		    si = giv_r / d;
		    H[k][k] = d;
		} else {
		    co = - giv_h / d;
		    si = - giv_r / d;
		    H[k][k] = -d;
		}            
		H[k+1][k] = 0;
		giv_r = e[k];
		e[k] = si * giv_r;
		e[k+1] = co * giv_r;
		cs[k] = co;
		sn[k] = si;
		++iter;
		if (abs(e[k+1]) < epsilon) {
		    k++;
		    break;
		}
	    }
	    for (int i=k-1; i>=0; --i) {
		Scalar s = e[i];
		for (int j=i+1; j<k; ++j) s -= H[i][j] * y[j];
		y[i] = s / H[i][i];
	    }
	    t = 0;
	    for (int i=0; i<k; ++i) t += y[i] * v[i];
	    c = B * t;
	    u += c;
	    r -= A * c;
	    d = norm(r);
	}
	if (d < epsilon) { Vout(0) << "  " << IterationEnd(); }
	else Vout(-1) << "  " << IterationEnd();
    }
    string Name () const { return "GMRES"; }
};

class ScalarMatrix {
    Scalar* a;
    int n;
    int m;
public:
    ScalarMatrix (int N) : n(N), m(N) { a = new Scalar [n*m]; }
    ScalarMatrix (int N, int M) : n(N), m(M) { a = new Scalar [n*m]; }
    ~ScalarMatrix () { delete[] a; }
    ScalarMatrix& operator = (Scalar x) { 
	for (int i=0; i<n*m; ++i) a[i] = x; 
	return *this;
    }
    Scalar* operator [] (int i) { return a+m*i; } 
    const Scalar* operator [] (int i) const { return a+m*i; } 
    int rows() const { return n; }
    int cols() const { return m; }
    void invert() { 
	int* ipv = new int [n];
	Scalar* Mat = new Scalar [n*n];
	Scalar* rhs = new Scalar [n];
	for (int i=0; i<n; ++i) ipv[i] = i;
	for (int i=0; i<n*n; ++i) Mat[i] = a[i];
	for (int i=0; i<n; ++i) {
	    int k = i;
	    Scalar piv = abs(Mat[i*n+i]);
	    for (int j=i+1; j<n; ++j) {
		Scalar sum = abs(Mat[j*n+i]);
		if (abs(sum) > abs(piv)) {
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
    }
    friend ostream& operator << (ostream& s, const ScalarMatrix& M) {
	for (int i=0; i<M.rows(); ++i, s << endl) 
	    for (int j=0; j<M.cols(); ++j) 
		s << M[i][j] << " ";
	return s;
    }
};

class ScalarVector : public valarray<Scalar> {
public:
    ScalarVector (Scalar x, int N) : valarray<Scalar>(x,N) {} 
    ScalarVector (int N) : valarray<Scalar>(N) {} 
};
inline constAB<ScalarMatrix,ScalarVector> operator * (const ScalarMatrix& S, 
						      const ScalarVector& v){
    return constAB<ScalarMatrix,ScalarVector>(S,v);
}

class MINRES : public IterativeSolver {
    int M;
 public:
    MINRES (const string& prefix = "Linear") : IterativeSolver(prefix) {
	M = 5;
    }  
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	d = d_0 = norm(r);
	double epsilon = max(Eps,Red * d);
	ScalarMatrix H(M+1,M);
	ScalarVector e(M);
	vector<Vector> v(M+1,u);
	Vector c(r), t(r), y(r);
	for (iter=0; iter < max_iter;) {
	    if (d < epsilon) break;
	    Vout(1) << "  MINRES: d(" << iter << ")= " << d << "\n";
	    v[0] = (1/d) * r;
	    int k = 0;
	    for (; k<M; ++k) {
		c = B * v[k];
		t = A * c;
		v[k+1] = t;
		for (int i=0; i<=k; ++i) {
		    H[i][k] = t * v[i];
		    v[k+1] -= H[i][k] * v[i];
		}
		H[k+1][k] = norm(v[k+1]);
		Vout(3) << "  GMRES: " << k 
			<< " |v_k+1| " << H[k+1][k] << "\n";
		v[k+1] *= (1.0/H[k+1][k]);
	    }
	    for (int i=k-1; i>=0; --i) {
		Scalar s = e[i];
		for (int j=i+1; j<k; ++j) s -= H[i][j] * y[j];
		y[i] = s / H[i][i];
	    }
	    t = 0;
	    for (int i=0; i<k; ++i) t += y[i] * v[i];
	    c = B * t;
	    u += c;
	    r -= A * c;
	    d = norm(r);
	}
	Vout(0) << "  " << IterationEnd();
    }
    string Name () const { return "MINRES"; }
};

extern "C" void zhegv_(int *ITYPE, char *JOBZ, 
		       char *UPLO, int *N, void *A, int *LDA, void *B,
		       int *LDB, void *W, void *WORK, int *LWORK, 
		       void *RWORK, int *INFO, int ch1, int ch2);

pair<double,double> SpectralBounds (int n, const ScalarMatrix &a) {
    if (n == 1) {
	if (a[0][0] == 0.0) return  pair<double,double>(1,1);
	return  pair<double,double>(double_of_Scalar(a[0][0]),double_of_Scalar(a[0][0]));
    }
    char jobz='V', uplo='U';
    int itype=1, lwork=2*n, info;
    double* w = new double [n];
    double* work = new double [2*lwork];
    double* rwork = new double [3*n];
    complex<double>* A = new complex<double> [n*n];
    complex<double>* B = new complex<double> [n*n];

    for (int i=0;i<n;i++)
	for (int j=0;j<n;j++) A[i*n+j]=a[i][j];
    for (int i=0;i<n;i++)
	for (int j=0;j<n;j++) B[i*n+j]=0;
    for (int i=0;i<n;i++)
	B[i*n+i]=1;
    
    zhegv_(&itype, &jobz, &uplo, &n, A, 
	   &n, B, &n, w, work, &lwork, rwork, &info, 1, 1);

    double w_min = w[0];
    double w_max = w[n-1];

    delete[] w;
    delete[] work;
    delete[] rwork;
    delete[] A;
    delete[] B;

    return pair<double,double>(w_min,w_max);
}

class CGX : public IterativeSolver {
public:
    CGX (const string& prefix = "Linear") : IterativeSolver(prefix) {}
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	d = d_0 = r.norm();
	double eps = Eps + Red * d;
	Vector c(u);
	Vector p(u);
	Vector t(u);
	p = 0;
	Scalar rho_0 = 1;
	ScalarMatrix H(max_iter+1);
	H = Scalar(0.0);
	vector<Vector> U(max_iter+1,u);
	vector<Vector> AU(max_iter+1,r);
	U[0] = B * r;
	AU[0] = A * U[0];
	Scalar h_10 = sqrt(U[0]*AU[0]);
	U[0] *= 1.0/h_10;
	AU[0] *= 1.0/h_10;
	pair<double,double> Kappa;
	for (iter=0; iter<max_iter; ++iter) {
	    Vector BAu = B * AU[iter];
	    U[iter+1] = BAu;
	    for (int i=0; i<=iter; ++i) {
		H[i][iter] = BAu * AU[i];
		U[iter+1] -= H[i][iter] * U[i];
	    }
	    AU[iter+1] = A * U[iter+1];
	    H[iter+1][iter] = sqrt(AU[iter+1]*U[iter+1]);
	    U[iter+1] *= 1.0/H[iter+1][iter];
	    AU[iter+1] *= 1.0/H[iter+1][iter];
	    Kappa = SpectralBounds(iter+1,H);
	    if (d < eps) break;
	    vout(1) << " PCG: d(" << iter << ")= " << d 
		    << " kappa=" << Kappa.second / Kappa.first
		    << " spec=[" <<  Kappa.first  
		    << "," << Kappa.second << "]" << endl; 
	    vout(10) << "r " << r;
	    c = B * r;
	    vout(10) << "c " << c;
	    Scalar rho_1 = r * c;
	    vout(2) << "rho " << r * c << " " << c * r << endl;
	    p *= (rho_1 / rho_0);
	    p += c;
	    rho_0 = rho_1;
	    t = A * p;
	    Scalar alpha = rho_0 / (p * t);
	    vout(3) << "alpha " << p * t << " " << t * p << endl;
	    u += alpha * p;
	    vout(10) << "u " << u;
	    r -= alpha * t;
	    d = r.norm();
	}
	vout(1) << "kappa = " << Kappa.second / Kappa.first 
		<< " spec=[" <<  Kappa.first  
		<< "," << Kappa.second << "]" << endl; 
	double s = 0;
	for (int i=0; i<iter; ++i) 
	    for (int j=0; j<i; ++j) 
		s += abs(H[i][j] - H[j][i]);
	vout(2) << "| skew H | = " << s << endl; 
	vout(2) << " H = " << endl << H << endl; 
	Vout(0) << "  " << IterationEnd();
    }
    string Name () const { return "PCGX"; }
};

class SolverX : public Solver {
    Solver S; 
    Matrix* A; 
    Vector* d;
    Vector* e;
    Scalar eAe;
    Vector* InftyVector () {
	Vector* c = new Vector(*A);
	if (c->Idx(Infty) == -1) Exit("no row Infty");
	row r_inf = c->find_row(Infty);
	for (row r=c->rows(); r!=c->rows_end(); ++r)
	    for (int i=0; i<r.n(); ++i) {
		(*c)(r,i) = (*A)(r,r_inf)[i];
		(*A)(r,r_inf)[i] = (*A)(r_inf,r)[i] = 0;
	    }
	(*c)(r_inf,0) = 0;
	(*A)(r_inf,r_inf)[0] = 1;
	Collect(*c);
	return c;
    }
    Operator& operator () (const Matrix& _A, int reassemble = 1) {
	if (A) delete A;
	if (d) delete d; 
	if (e) delete e; 
	A = new Matrix(_A);
	d = InftyVector();
	if (reassemble) S(*A); 
	e = new Vector(S * *d);
	eAe = *e * *d;
	if (abs(eAe) < Eps) Exit("eAe zero");
	return *this;
    }
    void multiply_plus (Vector& u, const Vector& b) const { 
	Scalar u_inf = (*e * b - b(Infty,0)) / eAe;
	PPM->Broadcast(u_inf);
	Vector r(b);
	r -= u_inf * *d;
	u(Infty,0) += u_inf;
	r(Infty,0) = 0;
	u += S * r;
    }
    friend ostream& operator << (ostream& s, const SolverX& S) {
	return s << " SolverX with " << S; 
    }
 public:
    SolverX (const MatrixGraphs& G, const Assemble& Ass, const string& name) 
	: Solver(0,0), S(G,Ass,name), A(0), d(0), e(0) {}
    SolverX (Preconditioner *pc, const string& name, const string& prefix) 
	: Solver(0,0), S(pc,name,prefix), A(0), d(0), e(0) {}
    ~SolverX () { 
	if (A) delete A; 
	if (d) delete d; 
	if (e) delete e; 
    }
};

string GetSolverName () {
    string sol = "pcg"; ReadConfig(Settings,"LinearSolver",sol);
    return sol;
}


class TestLS : public IterativeSolver {
public:
    TestLS (const string& prefix = "Linear") : IterativeSolver(prefix) {}
    void Solve (Vector& u, const Operator& A, const Operator& B, Vector& r) {
	Start = Date();
	vout(11) << "u " << u;
	d = d_0 = r.norm();
	double eps = Eps + Red * d;
	Vector c(u);
	Vector p(u);
	Vector t(u);
	p = 0;
	Scalar rho_0 = 1;
	for (iter=0; iter<max_iter; ++iter) {
	    if (d < eps) break;
	    vout(1) << "  LS: d(" << iter << ")= " << d << endl; 
	    vout(10) << "r " << r;
	    c = B * r;
	    vout(10) << "c " << c;
	    u += c;
	    vout(10) << "u " << u;
	    r -= A * c;
	    d = r.norm();
	}
	Vout(0) << "  " << IterationEnd();
    }
    string Name () const { return "TestLS"; }
};






IterativeSolver* GetIterativeSolver (const string& name, 
				     const string& prefix = "Linear") {

    if (name == "Test") return new TestLS(prefix);


    if (name == "LS") return new LS(prefix); 
    if (name == "CG") return new CG(prefix); 
    if (name == "CGX") return new CGX(prefix); 
    if (name == "pcg") return new CG(prefix); 
    if (name == "BiCGStab") return new BiCGStab(prefix); 
    if (name == "BiCGStab2") return new BiCGStab2(prefix); 
    if (name == "GMRES") return new GMRES(prefix); 
    if (name == "MINRES") return new MINRES(prefix); 
    if (name == "gmres") return new GMRES(prefix); 
    Exit("no solver " + name + "implemented");
}
IterativeSolver* GetIterativeSolver () { 
    return GetIterativeSolver(GetSolverName()); 
}
Solver::Solver () : PC(GetPC()), S(GetIterativeSolver()) {}
Solver::Solver (Preconditioner* pc) : PC(pc), S(GetIterativeSolver()) {} 
Solver::Solver (Preconditioner* pc, const string& name) : PC(pc) {
    S = GetIterativeSolver(name);
}
Solver::Solver (Preconditioner* pc, const string& name, const string& prefix) 
    : PC(pc) {
    S = GetIterativeSolver(name,prefix);
}
Solver::Solver (const MatrixGraphs& G, const Assemble& A) : PC(0), S(0) {
    string name = GetSolverName();
    PC = GetPC(G,A);
    S = GetIterativeSolver(name);
}
Solver::Solver (const MatrixGraphs& G, 
		const Assemble& A, const Transfer& Tr) : PC(0), S(0) {
    string name = GetSolverName();
    PC = GetPC(G,A,Tr);
    S = GetIterativeSolver(name);
}
Solver::Solver (const MatrixGraphs& G, const Assemble& A, const string& name) :
    PC(GetPC(G,A)), S(GetIterativeSolver(name)) {}
Solver* GetSolver () { return new Solver(); }
Solver* GetSolver (const string& name, const string& prefix = "Linear") {
    return new Solver(GetIterativeSolver(name,prefix),GetPC());
}
Solver* GetSolver (Preconditioner* pc,
		   const string& name, const string& prefix = "Linear") {
    if (name == "SolverX") {
	string key = prefix + "SubSolver";
	string sol = "pcg"; 
	ReadConfig(Settings,key.c_str(),sol);
	return new SolverX(pc,sol,prefix);
    }
    return new Solver(GetIterativeSolver(name,prefix),pc);
}
Solver* GetSolver (const MatrixGraphs& G, const Assemble& A,
		   const string& name, const string& prefix = "Linear") {
    if (name == "SolverX") {
	string key = prefix + "SubSolver";
	string sol = "pcg"; 
	ReadConfig(Settings,key.c_str(),sol);
	return new SolverX(G,A,sol);
    }
    return new Solver(GetIterativeSolver(name,prefix),GetPC(G,A));
}
Solver* GetSolver (const MatrixGraphs& G, const Assemble& A) {
    return GetSolver(G,A,GetSolverName());
}
