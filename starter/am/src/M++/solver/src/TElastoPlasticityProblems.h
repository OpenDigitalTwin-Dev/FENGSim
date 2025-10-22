// file: StaticLinearElasticityBVP.h
// author: Jiping Xin

#ifndef _TELASTOPLASTICITYPROBLEMS_H_
#define _TELASTOPLASTICITYPROBLEMS_H_

#include "m++.h"

class TElastoPlasticityProblems {
    int example_id;
public:
    double mu = 1;
    double lambda = 1;
    double Young = 1;
    double PoissonRatio = 1;
    double k_0 = 1;
    double h_0 = 0;
    double time_k = 0;
public:
    TElastoPlasticityProblems () {
	ReadConfig(Settings, "EXAMPLE", example_id);
	ReadConfig(Settings, "Young", Young);
	ReadConfig(Settings, "PoissonRatio", PoissonRatio);
	ReadConfig(Settings, "k_0", k_0);
	ReadConfig(Settings, "h_0", h_0);
		
	// calculate mu and lambda
	mu = Young/2.0/(1 + PoissonRatio);
	lambda = Young*PoissonRatio/(1 + PoissonRatio)/(1 - 2*PoissonRatio);
	mout << "    mu: " << mu << " lambda: " << lambda << endl;
    }
    void SetSubDomain (Mesh& M);
    void SetBoundaryType (Mesh& M, double time=0);
    bool IsDirichlet (int id);
    Point Solution (Point p, double t = 0);
    Point Source (Point p, double time = 0);
    void Dirichlet (Point p, double T, int k, RowBndValues& u_c, int id);
    Point Neumann (Point p, double t, int id);
    Point h0 (Point p);
    Point h1 (Point p, double dt);
    bool Contact (Point x, Point x1, Point x2, Point x3, Point x4);
    bool Contact (Point x, Point x1, Point x2);

    int ngeom = 0;
    void SetNGeom (int ng) { ngeom = ng; }

};


#endif
