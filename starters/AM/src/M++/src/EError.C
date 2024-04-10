// file: EError.C
// author: Christian Wieners
// $Header: /public/M++/src/EError.C,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

//#include "iError.h"
#include "m++.h"

class Periodic {
public:
    Complex operator () (const Point& x) const { 
	if (x[0] < 0.5) return 8.0 * (1-4*x[0]); 
	                return 8.0 * (4*x[0]-3); 
    }
};

class Periodic2 {
public:
    double operator () (const Point& x) const { 
	if (x[0] < 0.5) return 1-(4*x[0]-1)*(4*x[0]-1); 
	                return (4*x[0]-3)*(4*x[0]-3)-1; 
    }
};

class bilinearx {
public:
    Complex operator () (const Point& x) const { return Complex(x[0],x[1]); }
};


class MaterialDensity {
public:
    virtual double operator () (const Point&) const = 0;
};

class DensityRho : public MaterialDensity {
    double r_max;
    double r_min;
public:
    DensityRho (double s = 1) : r_min(5-s*4), r_max(5) {}
    double operator () (const Point& z) const {
	double h = 1.0/16;
	if (z[0] > h)
	    if (z[0] < 1-h)
		if (z[1] > h)
		    if (z[1] < 1-h)
			return r_min; 
	return r_max; 
    }
    double max () { return r_max; }
    double min () { return r_min; }
};

class DensityOne : public MaterialDensity {
public:
    double operator () (const Point& z) const { return 1; }
};

class EErrorBound : public Assemble {
    Point k;
    Gradient ik;
    const MaterialDensity& rho;
    const Discretization& D; 
    const Discretization& D2; 
    double C;
    double _lambda;
    const Vector* _u;
public:
    EErrorBound (const Discretization& _D, const Discretization& _D2,
		 Vector& _Du, const Point& _k,
		 const MaterialDensity& _rho,
		 double c = 0.05) : 
	D(_D), D2(_D2), 
	k(_k), ik(iUnit*_k), 
	rho(_rho), C(c) {}
    void Set (const Point& _k) { k = _k; ik = Gradient(iUnit * k); }
    void Set (const Point& _k, double lambda, const Vector& u) { 
	k = _k; ik = Gradient(iUnit * k); 
	_lambda = lambda;
	_u = &u;
    }
    Gradient Gradient_k (const ScalarElement& E, 
			 int q, const Vector& u) const {
	return E.Derivative(q,u) + E.Value(q,u) * ik;
    }
    Gradient Gradient_k (const ScalarElement& E, int q, int i) const {
	return E.Derivative(q,i) + E.Value(q,i) * ik;
    }
    Scalar Divergence_k (const VectorFieldElement& E2, 
			 int q, const Vector& sigma) const {
	return E2.Divergence(q,sigma) - ik * E2.VectorValue(q,sigma);
    }
    Scalar Divergence_k (const VectorFieldElement& E2, int q, 
			 int j, int l) const {
	return E2.Divergence(q,j,l) - ik * E2.VectorValue(q,j,l);
    }
    Scalar a_k (const Vector& u, const Vector& v) const {
	Scalar s = 0;
	for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	    ScalarElement E(D,u,c);
	    for (int q=0; q<E.nQ(); ++q) 
		s += E.QWeight(q) * Gradient_k(E,q,u) * Gradient_k(E,q,v);
	}
	return PPM->Sum(s);
    }
    Scalar b_rho (const Vector& u, const Vector& v) const {
	Scalar s = 0;
	for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	    ScalarElement E(D,u,c);
	    for (int q=0; q<E.nQ(); ++q)
		s += E.QWeight(q) * rho(E.QPoint(q)) 
		    * conj(E.Value(q,u)) * E.Value(q,v);
	}
	return PPM->Sum(s);
    }
    void RitzBound (vector<Vector>& u, DoubleVector& lambda) const {
	int R = u.size();
	HermitianMatrix a(R);
	HermitianMatrix b(R);
	HermitianMatrix e(R);
	for (int r=0; r<R; ++r) 
	    for (int s=0; s<R; ++s) {
		a[s][r] = a_k(u[r],u[s]);
		b[s][r] = b_rho(u[r],u[s]);
	    }
	dout(3) << "a \n" << a << "b \n" << b;
	EVcomplex(a,b,lambda,e);
	dout(4) << "e \n" << e;
    }
    Scalar m_k (const Vector& u, const Vector& v, 
		double beta, double gamma) const {
	Scalar s = 0;
	for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	    ScalarElement E(D,u,c);
	    for (int q=0; q<E.nQ(); ++q) 
		s += E.QWeight(q) 
		    * (Gradient_k(E,q,u) * Gradient_k(E,q,v)
		       + (gamma-beta) * rho(E.QPoint(q)) 
		       * conj(E.Value(q,u)) * E.Value(q,v));
	}
	return PPM->Sum(s);
    }
    Scalar n_k (const Vector& u, const Vector& v, 
		const Vector& sigma, const Vector& tau,
		double beta, double gamma) const {
	Scalar s = 0;
	for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	    ScalarElement E(D,u,c);
	    VectorFieldElement E2(D2,sigma,c);
	    for (int q=0; q<E.nQ(); ++q) {
		double r = rho(E.QPoint(q)); 		
		s += E.QWeight(q) 
		    * (Gradient_k(E,q,u) * Gradient_k(E,q,v)
		       + (gamma-2*beta) * r * conj(E.Value(q,u)) * E.Value(q,v)
		       + beta*beta
		       *(E2.VectorValue(q,sigma)*E2.VectorValue(q,tau)
			 + r * conj(E.Value(q,u) + Divergence_k(E2,q,sigma)/r)
			 *(E.Value(q,v) + Divergence_k(E2,q,tau)/r) / gamma));
	    }
	}
	return PPM->Sum(s);
    }
    Scalar b0 (const Vector& sigma, const Vector& tau) const {
	Scalar s = 0;
	for (cell c=tau.cells(); c!=tau.cells_end(); ++c) {
	    VectorFieldElement E2(D2,sigma,c);
	    for (int q=0; q<E2.nQ(); ++q) {
		s += E2.QWeight(q) 
		    * E2.VectorValue(q,sigma) * E2.VectorValue(q,tau);
	    }
	}
	return PPM->Sum(s);
    }
    Scalar b1 (const Vector& u, const Vector& v, 
	       const Vector& sigma, const Vector& tau) const {
	Scalar s = 0;
	for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	    ScalarElement E(D,u,c);
	    VectorFieldElement E2(D2,sigma,c);
	    for (int q=0; q<E.nQ(); ++q) {
		double r = rho(E.QPoint(q)); 		
		s += E.QWeight(q) 
		    * r * conj(E.Value(q,u) + Divergence_k(E2,q,sigma)/r)
		    *(E.Value(q,v) + Divergence_k(E2,q,tau)/r);
	    }
	}
	return PPM->Sum(s);
    }
    void GoerischBound (vector<Vector>& u, vector<Vector>& sigma, 
			double beta, double gamma,
			DoubleVector& lambda) const {
	beta += gamma;
	int R = sigma.size();
	HermitianMatrix a(R);
	HermitianMatrix b(R);
	HermitianMatrix e(R);
	DoubleVector mu(R);
	for (int r=0; r<R; ++r) 
	    for (int s=0; s<R; ++s) {
		a[s][r] = m_k(u[r],u[s],beta,gamma);
		b[s][r] = n_k(u[r],u[s],sigma[r],sigma[s],beta,gamma);
	    }

	int r = 0;
	Scalar A_k = a_k(u[r],u[r]);
	Scalar B_rho = b_rho(u[r],u[r]);
	Scalar B0 = b0(sigma[r],sigma[r]); 
	Scalar B1 = b1(u[r],u[r],sigma[r],sigma[r]);

	dout(1) << "a_k   " << A_k << endl;
	dout(1) << "b_rho " << B_rho << endl;
	dout(1) << "b0    " << B0 << endl;
	dout(1) << "b0 * lambda " << B0 * A_k << endl;
	dout(1) << "b1    " << B1 << endl;

	dout(1) << "M = a_k+(gamma-beta)b_rho=" 
		<< A_k+(gamma-beta)*B_rho << endl;
	dout(1) << "N = A_k+(gamma-2beta)b_rho+beta*beta*(b0+b1/gamma)=" 
		<< A_k+(gamma-2*beta)*B_rho+beta*beta*(B0+B1/gamma)
		<< endl;

	dout(1) << "M \n" << a << "N \n" << b;
	EVcomplex(a,b,mu,e);
	dout(4) << "e \n" << e;
	mout << "Goerisch mu " << mu;
	mout << "beta " << beta << " gamma " << gamma
	     << " beta - gamma " << beta-gamma << endl;
	lambda = 0;
	for (int r=0; r<R; ++r) 
	    if (mu[r] < 0)
		lambda[R-1-r] = beta-gamma-beta/(1-mu[r]);

    }
    void Residual2 (double lambda, const Vector& u, 
		    const Vector& sigma, Vector& b) const {
	b = 0;
	for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	    ScalarElement E(D,u,c);
	    VectorFieldElement E2(D2,sigma,c);
	    RowValues b_c(b,E2);	    
	    for (int q=0; q<E2.nQ(); ++q) {
		double r = rho(E.QPoint(q)); 		
		double w = E2.QWeight(q); 
		Gradient G = Gradient_k(E,q,u);
		G /= lambda;
		G -= E2.VectorValue(q,sigma);
		Scalar d = E.Value(q,u);
		d += Divergence_k(E2,q,sigma) / r;
		for (int i=0; i<E2.size(); ++i)
		    for (int k=0; k<E2.n(i); ++k) 
			b_c(i,k) -= w 
			    * (E2.VectorValue(q,i,k) * G
			       - C * conj(Divergence_k(E2,q,i,k)) * d);
	    }
	}
	Collect(b);
    }
    void Jacobi2 (double lambda, const Vector& u, Matrix& A) const {
	A = 0;
	for (cell c=A.cells(); c!=A.cells_end(); ++c) {
	    VectorFieldElement E2(D2,A,c);
	    RowEntries A_c(A,E2);	    
	    for (int q=0; q<E2.nQ(); ++q) {
		double w = E2.QWeight(q); 
		double rr = 1.0 / rho(E2.QPoint(q)); 		
		for (int i=0; i<E2.size(); ++i) 
		    for (int k=0; k<E2.n(i); ++k) 
			for (int j=0; j<E2.size(); ++j) 
			    for (int l=0; l<E2.n(i); ++l)
				A_c(i,j,k,l) += w 
				    * (E2.VectorValue(q,i,k) 
				       * E2.VectorValue(q,j,l)
				       + C * rr 
				       * conj(Divergence_k(E2,q,i,k))
				       * Divergence_k(E2,q,j,l));
	    }
	}
    }
    double Residual (const Vector& sigma, Vector& res) const {
	Residual2 (_lambda,*_u,sigma,res); 
	return res.norm();
    }
    void Jacobi (const Vector& sigma, Matrix& J) const {
	Jacobi2 (_lambda,*_u,J); }
    pair<double,double> Error2 (double lambda, 
				const Vector& u, Vector& sigma) const {

	for (row r=u.rows(); r!=u.rows_end(); ++r)
	    dpout(2) << "r() " << r() 
		     << " u " << u(r,0)
		     << " s " << sigma(r,0) << " " << sigma(r,1)
		     << endl;

	double res0 = 0;
	double res1 = 0;
	for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	    ScalarElement E(D,u,c);
	    VectorFieldElement E2(D2,sigma,c);
	    for (int q=0; q<E2.nQ(); ++q) {
		double r = rho(E.QPoint(q)); 		
		double w = E2.QWeight(q); 
		Gradient G = Gradient_k(E,q,u);
		VectorField G2 = E2.VectorValue(q,sigma);
		G2 *= lambda;
		Scalar d = E.Value(q,u);
		Scalar d2 = Divergence_k(E2,q,sigma);

		dpout(2) << "z " << E.QPoint(q)
			 << " d " << d
			 << " d2 " << d2
			 << " G " << G
			 << " G2 " << G2
			 << " r " << r
			 << endl;

		G -= G2;
		d += d2/r;
		d *= lambda;
		res0 += w * r * real(conj(d) * d);
		res1 += w * real(G * G);
	    }
	}
	dpout(2) << "res0 " << res0 << endl;
	return pair<double,double>(sqrt(PPM->Sum(res0)),sqrt(PPM->Sum(res1)));
    }
    void MassMatrix (Matrix& B) {
	B = 0;
	for (cell c=B.cells(); c!=B.cells_end(); ++c) {
	    ScalarElement E(D,B,c);
	    RowEntries B_c(B,E);
	    for (int q=0; q<E.nQ(); ++q) {
		double w = rho(E.QPoint(q)) * E.QWeight(q); 
		for (int i=0; i<E.size(); ++i) 
		    for (int j=0; j<E.size(); ++j) 
			B_c(i,j) += w * E.Value(q,i) * E.Value(q,j);
	    }
	}
    }
    void StiffnessMatrix (Matrix& A) {
	A = 0;
	for (cell c=A.cells(); c!=A.cells_end(); ++c) {
	    ScalarElement E(D,A,c);
	    RowEntries A_c(A,E);
	    for (int q=0; q<E.nQ(); ++q) 
		for (int i=0; i<E.size(); ++i) {
		    Gradient Du_i = E.Derivative(q,i) 
			+ E.Value(q,i) * ik;
		    for (int j=0; j<E.size(); ++j) {
			Gradient Du_j = E.Derivative(q,j) 
			    + E.Value(q,j) * ik;
			A_c(i,j) += E.QWeight(q) * Du_i * Du_j;
		    }
		}
	}
    }
};

double Reference (const Point& k, int R, double rho = 1) {
    DoubleVector lambda_comp(infty,R+1);
    for (int k0=-R-2; k0<=R+2; ++k0) 
	for (int k1=-R-2; k1<=R+2; ++k1) {    
	    double s = (2*Pi*k0+k[0])*(2*Pi*k0+k[0])
		+(2*Pi*k1+k[1])*(2*Pi*k1+k[1]);
	    s /= rho;
	    for (int n=0; n<R+1; ++n)
		if (lambda_comp[n] > s) {
		    for (int i=R; i>n; --i)
			lambda_comp[i] = lambda_comp[i-1];
		    lambda_comp[n] = s;
		    break;
		}
	    dout(20) << "ref " << k << " " << lambda_comp;
	    dout(10) << "k0 " << k0 << " k1 " << k1 << " s " << s << endl;
	}
    mout << "ref     " << k << " " << lambda_comp;
    if (lambda_comp[0] == 0.0) return lambda_comp[1];
    return lambda_comp[0];
}

void Reference (const Point& k, int R, double rho,
		DoubleVector& Lambda) {
    DoubleVector lambda_comp(infty,R+1);
    for (int k0=-R-2; k0<=R+2; ++k0) 
	for (int k1=-R-2; k1<=R+2; ++k1) {    
	    double s = (2*Pi*k0+k[0])*(2*Pi*k0+k[0])
		+(2*Pi*k1+k[1])*(2*Pi*k1+k[1]);
	    s /= rho;
	    for (int n=0; n<R+1; ++n)
		if (lambda_comp[n] > s) {
		    for (int i=R; i>n; --i)
			lambda_comp[i] = lambda_comp[i-1];
		    lambda_comp[n] = s;
		    break;
		}
	    dout(20) << "ref " << k << " " << lambda_comp;
	    dout(10) << "k0 " << k0 << " k1 " << k1 << " s " << s << endl;
	}
    for (int i=0; i<Lambda.size(); ++i)
	Lambda[i] = lambda_comp[i];
}

Point reflect1 (const Point& z) { return Point(z[1],z[0]); }
Point reflect2 (const Point& z) { return Point(1-z[0],z[1]); }
Point reflect3 (const Point& z) { return Point(z[0],1-z[1]); }

double Symmetry1 (const Vector& u) {
    double s = 0;
    for (row r=u.rows(); r!=u.rows_end(); ++r)
	s += abs(u(r,0)-u(reflect1(r()),0));
    return s;
}
double Symmetry2 (const Vector& u) {
    double s = 0;
    for (row r=u.rows(); r!=u.rows_end(); ++r)
	s += abs(u(r,0)+u(reflect1(r()),0));
    return s;
}
double Symmetry3 (const Vector& u) {
    double s = 0;
    for (row r=u.rows(); r!=u.rows_end(); ++r)
	s += abs(u(r,0)+u(reflect2(r()),0));
    return s;
}
double Symmetry4 (const Vector& u) {
    double s = 0;
    for (row r=u.rows(); r!=u.rows_end(); ++r)
	s += abs(u(r,0)-u(reflect2(r()),0));
    return s;
}
double Symmetry5 (const Vector& u) {
    double s = 0;
    for (row r=u.rows(); r!=u.rows_end(); ++r)
	s += abs(u(r,0)+u(reflect3(r()),0));
    return s;
}
double Symmetry6 (const Vector& u) {
    double s = 0;
    for (row r=u.rows(); r!=u.rows_end(); ++r)
	s += abs(u(r,0)-u(reflect3(r()),0));
    return s;
}

void EError (const Discretization& D, const Discretization& D2,
	     double s, const Point& k,
	     vector<Vector>& U, 
	     DoubleVector& lambda,
	     DoubleVector& lambda_inf,
	     int R1, 
	     Vector& sigma,
	     Solver& S,
	     Newton& newton,
	     esolver& ES,
	     double gamma,
	     double lambda_gap,
	     double lb,
	     double C_0) {
    int R = U.size();
    DensityRho rho(s);
    EErrorBound EE(D,D2,sigma,k,rho,C_0);
    Matrix A(U[0]);
    Matrix B(U[0]);
    EE.StiffnessMatrix(A);
    EE.MassMatrix(B);
    S(A);
    if (lambda[0] == -1) 
	ES.Init(U,B,S);
    ES(U,lambda,A,B,S);
    mout << "gnuplot " << k << " " << lambda;
    if (R1>=R) return;
    vector<Vector> Sigma(R1,sigma);
    double beta = lambda[R1];
    for (int r=0; r<R1; ++r) {
	EE.Set(k,lambda[r],U[r]);
	newton(EE,Sigma[r]);
	Vector b(sigma);
	EE.Residual2(lambda[r],U[r],Sigma[r],b);
	mout << "lambda  " << lambda[r]
	     << " Residual2 " << b.norm();
	pair<double,double> 
	    res = EE.Error2(lambda[r],U[r],Sigma[r]);
	mout << " Residuals " << res.first << " " 
	     << res.second << " lb " << lb << endl;
	double err_eps = (1/sqrt(lb)) * res.first + res.second;
	double err = lambda[r]*err_eps/(sqrt(lambda[r])-err_eps);
	if (r == 0) {
	    lb = max(lb,lambda[r] - err);
	    err_eps = (1/sqrt(lb)) * res.first + res.second;
	    err = lambda[r]*err_eps / (sqrt(lambda[r]) - err_eps);
	    lb = max(lb,lambda[r] - err);
	}
	mout << "lambda[" << r << "] in (" 
	     << lambda[r] - err << "," 
	     << lambda[r] + 0.00005 << ")";
	mout << " eps " << err_eps << " err " << err << endl;
	beta = lambda[r] + err + 0.1;
    }
    EE.RitzBound (U,lambda);
    mout << "Ritz     " << lambda;
    if (gamma == 0) return;
    for (int r=0; r<R1; ++r) 
	Sigma[r] *= lambda[r]/(lambda[r] + gamma);
    EE.GoerischBound(U,Sigma,beta,gamma,lambda_inf);
    mout << "Goerisch " << lambda_inf;
    for (int r=0; r<R1; ++r)
	mout << "Lambda[" << r << "] in (" 
	     << lambda_inf[r] << "," 
	     << lambda[r]  << ")" << endl;
    mout << "Lambda[" << R1 << "] > " << beta << endl;
}

void EError2 (const Discretization& D, const Discretization& D2,
	      double s, const Point& k,
	      vector<Vector>& U, 
	      DoubleVector& lambda,
	      DoubleVector& lambda_inf,
	      Vector& sigma,
	      Solver& S,
	      Newton& newton,
	      const esolver& ES,
	      double gamma,
	      double lambda_gap,
	      double beta,
	      double C_0) {
    int R = U.size();
    DensityRho rho(s);
    EErrorBound EE(D,D2,sigma,k,rho,C_0);
    Matrix A(U[0]);
    Matrix B(U[0]);
    EE.StiffnessMatrix(A);
    EE.MassMatrix(B);
    S(A);
    if (lambda[0] == -1) 
	ES.Init(U,B,S);
    ES(U,lambda,A,B,S);
    mout << "k " << k 
	 << " s " << s + 0.0001
	 << " Lambda " << lambda;
    vector<Vector> Sigma(R,sigma);
    for (int r=0; r<R; ++r) {
	EE.Set(k,lambda[r],U[r]);
	newton(EE,Sigma[r]);
    }
    EE.RitzBound (U,lambda);
    mout << "Ritz     " << lambda;
    if (gamma == 0) return;
    for (int r=0; r<R; ++r) 
	Sigma[r] *= lambda[r]/(lambda[r] + gamma);
    if (beta < lambda[R-1]) mout << "Homotopy failed" << endl;
    EE.GoerischBound(U,Sigma,beta,gamma,lambda_inf);
    mout << "Goerisch lb " << lambda_inf;
}

void EError3 (const Discretization& D, const Discretization& D2,
	      const Point& k,
	      vector<Vector>& U, 
	      DoubleVector& lambda,
	      Vector& sigma,
	      Solver& S,
	      Newton& newton,
	      const esolver& ES,
	      double lambda_gap,
	      double C_0) {
    int R = U.size();
    DensityRho rho;
    EErrorBound EE(D,D2,sigma,k,rho,C_0);
    Matrix A(U[0]);
    Matrix B(U[0]);
    EE.StiffnessMatrix(A);
    EE.MassMatrix(B);
    S(A);
    if (lambda[0] == -1) 
	ES.Init(U,B,S);
    ES(U,lambda,A,B,S);
    mout << "k " << k 
	 << " Lambda " << lambda;
}

void EError4 (const Discretization& D, const Discretization& D2,
	      double s,
	      const Point& k,
	      vector<Vector>& U, 
	      DoubleVector& lambda,
	      Vector& sigma,
	      Solver& S,
	      Newton& newton,
	      const esolver& ES,
	      double C_0) {
    int R = U.size();
    DensityRho rho(s);
    EErrorBound EE(D,D2,sigma,k,rho,C_0);
    Matrix A(U[0]);
    Matrix B(U[0]);
    EE.StiffnessMatrix(A);
    EE.MassMatrix(B);
    S(A);
    if (lambda[0] == -1) 
	ES.Init(U,B,S);
    ES(U,lambda,A,B,S);
    return;
    mout << "k " << k 
	 << " s " << s 
	 << " Lambda " << lambda;
}

void ReadList (const char* list,
	       const Discretization& D, const Discretization& D2,
	       vector<Vector>& U, 
	       DoubleVector& lambda,
	       DoubleVector& lambda_inf,
	       Vector& sigma,
	       Solver& S,
	       Newton& newton,
	       esolver& ES,
	       double gamma,
	       double lambda_gap,
	       double C_0) {
    M_ifstream file(list);
    const int len = 128;
    char L[len];
    double k0,k1;
    while (!file.eof()) {
	file.getline(L,len);
	dout(99) << L << "\n";
	double s;
	int R;
	int m = sscanf(L,"hom k   %lf   %lf s %lf R %d",&k0,&k1,&s,&R); 
	int Rold = R;
	if (m != 4) return;
	Point k(k0,k1);
	dout(99) << "k " << k << "\n";
	dout(99) << s << " " << R << "\n";
	if (s == 0) {
	    DensityRho rho(1);
	    DoubleVector Lambda(R);
	    Reference(k,R,rho.max(),Lambda);
	    mout << "k " << k 
		 << " s " << s 
		 << " Lambda " << Lambda;

	    for (int r=1; r<R; ++r)
		lambda_inf[r] = lambda[r] = Lambda[r];
	}
	while (s != 1) {
	    file.getline(L,len);
	    m = sscanf(L,"hom k   %lf   %lf s %lf R %d",&k0,&k1,&s,&R); 
	    if (m != 4) return;
	    if (R == Rold) mout << "Homotopy failed" << endl;
	    dout(99) << s << " " << R << "\n";
	    if (R > U.size()) return;
	    vector<Vector> V(R,U[0]);
	    for (int i=0; i<R; ++i) V[i] = U[i];
	    DoubleVector Lambda(R);
	    Lambda[0] = lambda[0];
	    EError2(D,D2,s,k,V,Lambda,lambda_inf,
		    sigma,S,newton,ES,gamma,
		    lambda_gap,lambda_inf[R],C_0);
	    for (int i=0; i<R; ++i) U[i] = V[i];
	    for (int r=0; r<R; ++r)
		lambda[r] = Lambda[r];
	}
	for (int r=0; r<R; ++r) {
	    DensityRho rho(1);
	    if (lambda_gap < lambda_inf[r-1]) continue;
	    double delta = lambda_gap - lambda_inf[r-1];
	    if (lambda_gap > lambda[r]) continue;
	    delta = min(delta,lambda[r]-lambda_gap);
	    double h = rho.min()
		* (sqrt(lambda[r]+delta)-sqrt(lambda[r]));
	    mout << "k " << k << " delta " << delta << " h " << h << endl;
	}
    }
}

void ReadList0 (const char* list,
	       const Discretization& D, const Discretization& D2,
	       vector<Vector>& U, 
	       DoubleVector& lambda,
	       DoubleVector& lambda_inf,
	       Vector& sigma,
	       Solver& S,
	       Newton& newton,
	       esolver& ES,
	       double gamma,
	       double lambda_gap,
	       double C_0) {
    M_ifstream file(list);
    const int len = 128;
    char L[len];
    while (!file.eof()) {
	file.getline(L,len);
	dout(99) << L << "\n";
	double k0,k1;
	int m = sscanf(L,"%lf %lf",&k0,&k1); 
	if (m != 2) return;
	Point k(k0,k1);
	dout(99) << "k " << k << "\n";
	file.getline(L,len);
	double s;
	int R;
	m = sscanf(L,"%lf %d",&s,&R); 
	if (m != 2) return;
	dout(99) << s << " " << R << "\n";
	if (s == 0) {
	    DensityRho rho(1);
	    DoubleVector Lambda(R);
	    Reference(k,R,rho.max(),Lambda);
	    mout << "k " << k 
		 << " s " << s + 0.0001
		 << " Lambda " << Lambda;
	    for (int r=1; r<R; ++r)
		lambda_inf[r] = lambda[r] = Lambda[r];
	}
	while (s != 1) {
	    file.getline(L,len);
	    int m = sscanf(L,"%lf %d",&s,&R); 
	    if (m != 2) return;
	    dout(99) << s << " " << R << "\n";
	    if (R > U.size()) return;
	    vector<Vector> V(R,U[0]);
	    for (int i=0; i<R; ++i) V[i] = U[i];
	    DoubleVector Lambda(R);
	    Lambda[0] = lambda[0];
	    EError2(D,D2,s,k,V,Lambda,lambda_inf,
		    sigma,S,newton,ES,gamma,
		    lambda_gap,lambda_inf[R],C_0);
	    for (int i=0; i<R; ++i) U[i] = V[i];
	    for (int r=0; r<R; ++r)
		lambda[r] = Lambda[r];
	}
	for (int r=0; r<R; ++r) {
	    DensityRho rho(1);
	    if (lambda_gap < lambda_inf[r-1]) continue;
	    double delta = lambda_gap - lambda_inf[r-1];
	    if (lambda_gap > lambda[r]) continue;
	    delta = min(delta,lambda[r]-lambda_gap);
	    double h = rho.min()
		* (sqrt(lambda[r]+delta)-sqrt(lambda[r]));
	    mout << "k " << k << " delta " << delta << " h " << h << endl;
	}
    }
}

void WriteList (const Point& k,
		const Discretization& D, const Discretization& D2,
		vector<Vector>& U, 
		DoubleVector& lambda,
		Vector& sigma,
		Solver& S,
		Newton& newton,
		esolver& ES,
		int N,
		double lambda_gap,
		double C_0) {
    DensityRho rho;
    int R = U.size();
    vector<DoubleVector> Lambda(N+1,lambda);
    Reference(k,R,rho.max(),Lambda[0]);
    for (int n=1; n<=N; ++n) {
	double s = n * 1.0 / N;
	int RR = R-1;
	RR = min(5+N-n,RR);
	RR = R;
	vector<Vector> V(RR,U[0]);
	for (int i=0; i<RR; ++i) V[i] = U[i];
	Lambda[n][0] = lambda[0];
	EError4(D,D2,s,k,V,Lambda[n],
		sigma,S,newton,ES,C_0);
	for (int i=0; i<RR; ++i) U[i] = V[i];
	for (int i=R; i<RR; ++i) Lambda[i] = 0;
    }
    for (int n=0; n<=N; ++n) {
	double s = n * 1.0 / N;
	mout << "k " << k 
	     << " s " << int(1000*s)+1000
	     << " Lambda[n] " << Lambda[n];
    }
    int r = 0;
    for (; r<R; ++r) 
	if (Lambda[N][r] > lambda_gap) 
	    break;
    double bound = Lambda[N][r];
    double gap = 0.2;
    for (int n=N; n>0;) {
	bound = Lambda[n][r];
	while (bound+gap > Lambda[n-1][r+1]) {
	    ++r;
	    if (r>R) break;
	    bound = Lambda[n][r];
	}
	if (r>R) break;
	int m = 0;
	while (bound+gap > Lambda[m][r+1]) ++m;
	++r;
	if (r+1>R) break;
	while (n>m) Lambda[n--].Resize(r);
	double s = m * 1.0 / N;
	mout << "k " << k 
	     << " s " << int(1000*s)+1000
	     << " r " << r
	     << " n " << n
	     << " m " << m
	     << " bound " << bound 
	     << endl;
    }
    if (r<R) Lambda[0].Resize(r+1);
    if (r+1>R) {
	mout << "Homotopy failed\n";
	return;
    }
    for (int n=0; n<=N; ++n) {
	if (n>1111110) 
	    if (Lambda[n-1].size() == Lambda[n].size())
		continue;
	double s = n * 1.0 / N;
	mout << "s " << int(1000*s)+1000
	     << " L" << Lambda[n];
    }
    for (int n=0; n<=N; ++n) {
	if ((n>0) && (n<N)) 
	    if (Lambda[n+1].size() == Lambda[n].size())
		continue;
	double s = n * 1.0 / N;
	mout << "hom k " << k 
	     << " s " << s
	     << " R " << Lambda[n].size() 
	     << " bound " << Lambda[n][Lambda[n].size()-1]
	     << endl;
    }
}

void EError () {
    Meshes M;
    int R = 1;         ReadConfig(Settings,"R",R);
    int R1= 0;         ReadConfig(Settings,"R1",R1);
    Point k = 0;       ReadConfig(Settings,"k",k); 
    double C_0 = 0.05; ReadConfig(Settings,"C_0",C_0);
    int K = 0;         ReadConfig(Settings,"K",K);
    int K0 = 0;        ReadConfig(Settings,"K0",K0);
    int N = 0;         ReadConfig(Settings,"N",N);
    double gamma = 1;  ReadConfig(Settings,"gamma",gamma);
    double lambda_gap = 0; ReadConfig(Settings,"lambda_gap",lambda_gap);
    Discretization D(1);

    MatrixGraphs G(M,D);
    Plot P(M.fine(),1,1);
    P.gnu_mesh("mesh");
    P.dx_load();

    string disc2 = "linear"; ReadConfig(Settings,"Discretization2",disc2);
    Discretization D2(disc2.c_str(),D,2);
    MatrixGraphs G2(M,D2);
    esolver ES;
    Vector u(G.fine());
    u = 1;
    Vector sigma(G2.fine());
    sigma = 0;
    P.vertexdata(u);
    P.gnu_vertexdata("u");
    P.dx_vertexdata("u");
    vector<Vector> U(R,u);
    DoubleVector lambda(R);
    DoubleVector lambda_inf(infty,R);
    DoubleVector lambda_min(infty,R);
    DoubleVector lambda_max(-infty,R);
    Solver S;
    Newton newton(S);
    double h_min = infty;
    int h_cnt = 0;
    lambda[0] = -1;

    if (K==-1000) { 
	M_ifstream file("conf/List");
	const int len = 128;
	char L[len];
	while (!file.eof()) {
	    file.getline(L,len);
	    dout(99) << L << "\n";
	    double k0,k1;
	    int m = sscanf(L,"list k %lf %lf",&k0,&k1); 
	    if (m != 2) return;
	    Point k(k0,k1);
	    WriteList(k,D,D2,U,lambda,sigma,S,newton,ES,N,lambda_gap,C_0);
	}
    } else if (K==-100) { 
	const double frac = 1.0 / sqrt(2.0);
	const double h_min = 0.1;
	double k0 = h_min*frac;
	while (k0<Pi) {
	    double h_max = 10;
	    double k1 = h_min*frac;
	    while (k1<k0+h_min*frac) {
		if (k1 > k0) k1 = k0;
		Point k(k0,k1);
		mout << "list k " << k << endl;
		EError3(D,D2,k,U,lambda,
			sigma,S,newton,ES,
			lambda_gap,C_0);
		double h = 0.1;
		for (int r=0; r<R; ++r) {
		    DensityRho rho(1);
		    if (lambda_gap < lambda[r-1]) continue;
		    double delta = lambda_gap - lambda[r-1];
		    if (lambda_gap > lambda[r]-0.2) continue;
		    delta = min(delta,lambda[r]-lambda_gap-0.2);
		    h = rho.min() * (sqrt(lambda[r]+delta)-sqrt(lambda[r]));
		    mout << "k " << k << " delta " 
			 << delta << " h " << h << endl;
		}
		h_max = min(h,h_max);
		k1 += (h + h_min)*frac;
	    }
	    mout << "h_max " << h_max << endl;
	    k0 += (h_max + h_min)*frac;
	}
	return;
    } else if (K<0) 
	ReadList("conf/list",
		 D,D2,U,lambda,lambda_inf,
		 sigma,S,newton,ES,gamma,
		 lambda_gap,C_0);

    for (int k0=-K; k0<=K; ++k0) {
	mout << "gnuplot " << endl;
	for (int k1=-K; k1<=K; ++k1) {
	    if (K) k = Point(k0*Pi/K,k1*Pi/K);

//    for (int k0=0; k0<3*K; ++k0) {
//	    if (k0<K) k = Point(k0*Pi/K,k0*Pi/K);
//	    else if (k0<2*K) k = Point(Pi,(2*K-k0)*Pi/K);
//	    else k = Point((3*K-k0)*Pi/K,0.0);

	    DensityRho rho;
	    double lb = Reference(k,R,rho.max());
	    for (int i=1; i<=N; ++i) {    
		double s = i*1.0/N;
		if (N) mout << "scale " << s << endl;
		else s = 1.0;
		EError(D,D2,s,k,U,lambda,lambda_inf,
		       R1,sigma,S,newton,ES,gamma,
		       lambda_gap,lb,C_0);
	    }
	    h_cnt++;
	    for (int r=0; r<R; ++r) {
		lambda_max[r] = max(lambda_max[r],lambda[r]);
		lambda_min[r] = min(lambda_min[r],lambda[r]);
		if (r == 0) continue;
		if (lambda_gap < lambda_inf[r-1]) continue;
		double delta = lambda_gap - lambda[r-1];
		if (lambda_gap > lambda[r]) continue;
		delta = min(delta,lambda_inf[r]-lambda_gap);
		double h = rho.min()
		    * (sqrt(lambda[r]+delta)-sqrt(lambda[r]));
		mout << " delta " << delta << " h " << h << endl;
		h_min = min(h,h_min);
		h_cnt--;
	    }
	}
    }
    if (K>0) {
	mout << "lambda_min " << lambda_min;
	mout << "lambda_max " << lambda_max;
	for (int r=1; r<R; ++r) 
	    if (lambda_max[r-1] < lambda_min[r]) 
		mout << "gap(" << r << ") " 
		     << lambda_min[r] -lambda_max[r-1];
	if (h_cnt==0) 
	    mout << " h_min " << h_min 
		 << " h_max " << sqrt(2.0) * Pi / K;
	mout << endl;
    }
    for (int r=0; r<R; ++r) {
	P.vertexdata(U[r]);
	P.gnu_vertexdata(Number_Name("u",r));
	P.dx_vertexdata(Number_Name("u",r));
    }
}
