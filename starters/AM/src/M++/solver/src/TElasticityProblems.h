// file: TransientLinearElasticityBVP.h
// author: Jiping Xin

#include "m++.h"

class TElasticityProblems {
  public:
    double mu = 1;
    double lambda = 1;
    double Young = 1;
    double PoissonRatio = 1;
    int example_id = 1;
	double time_k = 1;
  public:
    TElasticityProblems () {
		ReadConfig(Settings, "EXAMPLE", example_id);
		ReadConfig(Settings, "Young", Young);
		ReadConfig(Settings, "PoissonRatio", PoissonRatio);
		mu = Young/2.0/(1 + PoissonRatio);
		lambda = Young*PoissonRatio/(1 + PoissonRatio)/(1 - 2*PoissonRatio);
    }
    void SetSubDomain (Mesh& M);
    void SetBoundaryType (Mesh& M);
    bool IsDirichlet (int id);
    // given data
    void g_D (Point p, double time, int k, RowBndValues& u_c, int id);
    Point g_N (Point p, double t, int id);
    Point h0 (Point p);
    Point h1 (Point p, double dt);
    Point u (Point p, double t=0);
    Point f (Point p, double t=0);
};
