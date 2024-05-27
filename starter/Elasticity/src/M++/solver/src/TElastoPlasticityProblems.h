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
    vector<vector<double>> bc;
public:
    void setbc () {
	if (example_id==0) {
	    ifstream is("./../../build-FENGSim-Desktop_Qt_5_12_12_GCC_64bit-Debug/BndConditions.txt");
	    const int len = 256;
	    char L[len];
	    is.getline(L,len);
	    while (strncasecmp("END",L,3)) {
		double z[4];
		int d = sscanf(L,"%lf %lf %lf %lf",z,z+1,z+2,z+3);
		vector<double> _bc;
		for (int i=0; i<4; i++)
		    _bc.push_back(z[i]);
		bc.push_back(_bc);
		is.getline(L,len);
	    }
	    for (int i=0; i<bc.size(); i++) {
		for (int j=0; j<bc[i].size(); j++) {
		    mout << bc[i][j] << " ";
		}
		mout << endl;
	    }
	}
    }
    TElastoPlasticityProblems () {
	ReadConfig(Settings, "EXAMPLE", example_id);
	setbc();
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
