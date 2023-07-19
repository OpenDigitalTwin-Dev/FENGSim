#include "m++.h"
#include "PoissonProblems.h"

class PoissonAssemble : public PoissonProblems {
    Discretization disc;
  public:
    PoissonAssemble () {}
	/*!
	  set given dirichlet value
	*/
    void SetDirichletValues (Vector& u);
	/*!
	  assemble matrix
	*/
    void AssembleMatrix (Matrix& A) const;
    void AssembleVector (const Vector& u_d, Vector& b);
    double L2Error (const Vector& x);
    void vtk_derivative (const char* name, const Vector& x);
};

void PoissonMain ();
