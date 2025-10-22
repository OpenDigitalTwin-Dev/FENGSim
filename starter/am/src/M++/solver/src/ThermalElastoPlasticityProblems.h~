// file: Solution.h
// author: Jiping Xin

#include "m++.h"

class ThermoElasticityGeoID {
    int example_id;
  public:
    ThermoElasticityGeoID () {
	ReadConfig(Settings, "EXAMPLE", example_id);
    }
    void SetSubDomain (Mesh& M);
    void SetBoundaryType (Mesh& M);
};

class ThermoElasticityBVP_T {
    int example_id;
  public:
    ThermoElasticityBVP_T () {
	ReadConfig(Settings, "EXAMPLE", example_id);
    }
    bool IsDirichlet (int id);
    // given data
    double SourceValue (int id, Point p, double t) const;
    double DiffusionCoe (int id, Point p, double t);
    double DirichletValue (int id, Point p, double t) const;
    double NeumannValue (int id, Point p, double t) const;
    double InitialValue (const Point& p) const;
    double Solution (const Point& p, double t) const;
};

class ThermoElasticityBVP_D {
    int example_id;
  public:
    double mu = 1;
    double lambda = 1;
    double Young = 1;
    double PoissonRatio = 1;
  public:
    ThermoElasticityBVP_D () {
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
    void DirichletValue (int k, RowBndValues& u_c, int id, Point p);
    Point NeumannValue (int id, Point p, double t);
    Point Solution (Point p, double t);
};
