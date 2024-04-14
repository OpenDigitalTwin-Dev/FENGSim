// file: ELaplace.C
// author: Christian Wieners
// $Header: /public/M++/src/ELaplace.C,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#include "m++.h"

class MaterialDensity {
public:
    virtual double operator () (const Point&) const = 0;
};

class Density : public MaterialDensity {
public:
    double operator () (const Point& z) const {
	if (z[0] > 1.0/16)
	    if (z[0] < 15.0/16)
		if (z[1] > 1.0/8)
		    if (z[1] < 7.0/8)
			return 1; 
	return 5; 	
    }
};

class VectorDensity : public MaterialDensity {
    Vector& rho;
    double rho_min;
    double rho_max;
public:
    VectorDensity (Vector& r, double m = 1, double M = 11) : 
	rho(r), rho_min(m), rho_max(M) {}
    double operator () (const Point& z) const {
	return real(rho(z,0));
    }
    void Project () {
	for (row r = rho.rows(); r!=rho.rows_end(); ++r) {
	    rho(r,0) = max(real(rho(r,0)),rho_min);
	    rho(r,0) = min(real(rho(r,0)),rho_max);
	}
    }
    Vector& vec () { return rho; }
};

class ELaplaceProblem {
    const Discretization& D; 
    const MaterialDensity& rho;
    Gradient ik;
public:
    ELaplaceProblem (const Discretization& _D, const MaterialDensity& _rho,
		     const Point& _k) : D(_D), rho(_rho), ik(iUnit*_k) {}
    void MassMatrix (Matrix& B) {
	B = 0;
	for (cell c=B.cells(); c!=B.cells_end(); ++c) {
	    ScalarElement E(D,B,c);
	    RowEntries B_c(B,E);
	    for (int q=0; q<E.nQ(); ++q) {
		double w = rho(c()) * E.QWeight(q); 
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

void Evaluate (const Discretization& D,
	       const Vector& u,
	       Vector& u_cell) {
    for (cell c=u.cells(); c!=u.cells_end(); ++c) {
	ScalarElement E(D,u,c);
	u_cell(c(),0) = abs(E.Value(c.ReferenceCell()->Center(),u,0));
    }
}

class ELaplaceSeries {
    Discretization& D;
    int K;
    bool init;
    int R;
    vector<Vector>& U;
public:
    ELaplaceSeries (Discretization& _D, int _K, vector<Vector>& u) : 
	D(_D), K(_K), init(true), R(u.size()), U(u) {}
    double MinMax (Vector& rho_vector, double lambda_gap) {
	VectorDensity Rho(rho_vector);
	DoubleVector lambda(R);
	DoubleVector lambda_min(infty,R);
	DoubleVector lambda_max(-infty,R);
	for (int k0=0; k0<3*K; ++k0) {
	    Point k(k0*Pi/K,k0*Pi/K);
	    if (k0<2*K) k = Point(Pi,(2*K-k0)*Pi/K);
	    else k = Point((3*K-k0)*Pi/K,0.0);
	    ELaplaceProblem EP(D,Rho,k);
	    Matrix A(U[0]);
	    EP.StiffnessMatrix(A);
	    Matrix B(U[0]);
	    EP.MassMatrix(B);
	    Solver S(A);
	    DoubleVector lambda(R);
	    esolver ES;
	    if (init) ES.Init(U,B,S);
	    else init = false;
	    ES(U,lambda,A,B,S);
    	    for (int r=0; r<R; ++r) {
		lambda_max[r] = max(lambda_max[r],lambda[r]);
		lambda_min[r] = min(lambda_min[r],lambda[r]);
	    }
	}
	mout << "lambda_min " << lambda_min;
	mout << "lambda_max " << lambda_max;
	double gap = 0;
	for (int r=1; r<R; ++r) 
	    if (lambda_max[r-1] < lambda_min[r]) {
		mout << "gap(" << r << ") " 
		     << lambda_min[r] -lambda_max[r-1]
		     << endl;
		    if (lambda_gap > lambda_max[r-1])
			if (lambda_gap < lambda_max[r])
			    gap = lambda_min[r] -lambda_max[r-1];
		    
	    }
	return gap;
    }
};

void ELaplace () {
    Meshes M;
    int R = 1;             ReadConfig(Settings,"R",R);
    Point k = 0;           ReadConfig(Settings,"k",k); 
    int K = 0;             ReadConfig(Settings,"K",K);
    double lambda_gap = 0; ReadConfig(Settings,"lambda_gap",lambda_gap);
    Discretization D(1);
    Density rho;
    MatrixGraphs G(M,D);
    dof Dcell("cell",1);
    MatrixGraphs Gcell(M,Dcell);    
    Plot P(M.fine(),1,1);
    P.gnu_mesh("mesh");
    P.dx_load();
    Vector u(G.fine());
    u = 1;
    vector<Vector> U(R,u);
    Vector rho_vector(Gcell.fine());
    rho_vector << rho;
//    mout << "rho 0 " << rho_vector;


    if (K == 0) {
	VectorDensity Rho(rho_vector);
	ELaplaceProblem EP(D,Rho,k);
	Matrix A(u);
	EP.StiffnessMatrix(A);
	Matrix B(u);
	EP.MassMatrix(B);
	Vector Bu = B * u;
	Solver S(A);
	DoubleVector lambda(R);
	esolver ES;
	ES.Init(U,B,S);
	ES(U,lambda,A,B,S);
	Vector u_cell(rho_vector);    
	Evaluate(D,U[0],u_cell);
	rho_vector += u_cell;
	EP.StiffnessMatrix(A);
	EP.MassMatrix(B);
	S(A);
	ES(U,lambda,A,B,S);
    } else {
	ELaplaceSeries ELS(D,K,U); 
    
	ELS.MinMax(rho_vector,lambda_gap);

    }
    for (int r=0; r<R; ++r) {
	P.vertexdata(U[r]);
	P.gnu_vertexdata(Number_Name("u",r));
	P.dx_vertexdata(Number_Name("u",r));
    }
}
