// file: Solution.h
// author: Jiping Xin

#include "m++.h"

class ThermalElastoPlasticityProblems {
    int example_id;
  public:
    ThermalElastoPlasticityProblems () {
	ReadConfig(Settings, "EXAMPLE", example_id);
    }
    void SetSubDomain (Mesh& M);
    void SetBoundaryType (Mesh& M);
};

class ThermalElastoPlasticityProblems_T {
    int example_id;
  public:
    ThermalElastoPlasticityProblems_T () {
	ReadConfig(Settings, "EXAMPLE", example_id);
    }
    bool IsDirichlet (int id);
    // given data
    double f (int id, Point p, double time) const;
    double a (int id, Point p, double time);
    void g_D (int k, RowBndValues& u_c, int id, Point p, double time);
    double g_N (int id, Point p, double time) const;
    double u0 (const Point& p) const;
    double u (const Point& p, double time) const;
};

class ThermalElastoPlasticityProblems_D {
    int example_id;
  public:
    double mu = 1;
    double lambda = 1;
    double Young = 1;
    double PoissonRatio = 1;
  public:
    ThermalElastoPlasticityProblems_D () {
        ReadConfig(Settings, "EXAMPLE", example_id);
	ReadConfig(Settings, "Young", Young);
	ReadConfig(Settings, "PoissonRatio", PoissonRatio);
	// elasticity parameters
	mu = Young / 2.0 / (1 + PoissonRatio);
	lambda = Young * PoissonRatio / (1 + PoissonRatio) / (1 - 2 * PoissonRatio);
	mout << "mu: " << mu << " lambda: " << lambda << endl;
    }
    bool IsDirichlet (int id);
    // given data
    Point SourceValue (int id, Point p, double t);
    void DirichletValue (int k, RowBndValues& u_c, int id, Point p, double time);
    Point NeumannValue (int id, Point p, double t);
    Point Solution (Point p, double t);
};
