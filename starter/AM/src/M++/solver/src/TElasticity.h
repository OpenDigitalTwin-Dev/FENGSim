// file: TElasticity.h
// author: Jiping Xin
// This module is to solve transient linear elasticity problem.

#include "m++.h"
#include "TElasticityProblems.h"

class TElasticityAssemble : public TElasticityProblems {
    Discretization disc;
    int dim;
    double dt;
public:
    TElasticityAssemble (int _dim, double t) : disc(Discretization(_dim)) {
		dt = t;
		dim = _dim;
    }
	void SetInitialCondition0 (Vector& x0); 
    void SetInitialCondition1 (Vector& x1);
    void SetDirichletBC (Vector& u, double t=0);
    void Jacobi (Matrix& A);
    double Residual (Vector& b, const Vector& u_d, Vector& x0, Vector& x1, double time);
    double L2Error (const Vector& x0, const Vector& x1, const Vector& x2, double time=0);
};

void TElasticityMain ();





