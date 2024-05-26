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
    TElasticityProblems () {
		ReadConfig(Settings, "EXAMPLE", example_id);
		setbc();
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
