// file: Heat.C
// author: Jiping Xin

#include "m++.h"
#include "HeatProblems.h"

class HeatAssemble : public HeatProblems {
    Discretization disc;
    double dt;
public:
    HeatAssemble (double k) { dt = k; }
    void SetDirichletBC (Vector& u, double t);
    void SetInitialCondition (Vector& x1);
	void AssembleMatrix (Matrix& A) const;
    void AssembleVector (const Vector& x1, const Vector& u_d, Vector& b, double t);
    double L2Error (const Vector& x, double t);
};

void SetMaxMin (Vector x, double& max, double& min);
void HeatMain ();
