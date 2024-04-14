// file: Elasticity.h
// author: Dr. Jiping Xin

#include "m++.h"
#include "ElasticityProblems.h"

class ElasticityAssemble : public ElasticityProblems {
    Discretization disc;
    int dim;
public:
    ElasticityAssemble (int _dim) : disc(Discretization(_dim)) {
        dim = _dim;
    }
    void SetDirichletBC (Vector& u);
    void Jacobi (Matrix& A);
    double Residual (Vector& b, const Vector& u0);
    double L2Error (const Vector& x);
};

void ElasticityMain ();



