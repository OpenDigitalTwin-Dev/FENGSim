// file: Solution.h
// author: Jiping Xin

#include "m++.h"

/*!
  for the prepost processing for the Poisson equation
*/
class PoissonProblems {
	int example_id;
public:
	PoissonProblems () {
		ReadConfig(Settings, "EXAMPLE", example_id);
	}
	//! set domain id for cell
	/*! 
	  \param M the mesh
	*/
	void SetDomain (Mesh& M);
	void SetBoundary (Mesh& M);
	bool IsDirichlet (int id);
    double alpha (Point p) const;
    double f (Point p) const;
    double g_D (Point p, int id) const;
    double g_N (Point p, int id) const;
    double u (Point p) const;
};
