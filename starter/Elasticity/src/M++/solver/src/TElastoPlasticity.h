// file: TElastoPlasticity.C
// author: Jiping Xin
// This module is to solve elastoplasticity problem
// by using the Newton method and the return mapping algorithm.

#include "m++.h"
#include "TElastoPlasticityProblems.h"
#include "ReturnMappingAlgo.h"
#include <fstream>


class TElastoPlasticityAssemble : public TElastoPlasticityProblems {
    int dim;
    Discretization disc;
    double Mu = 1;
    double Lambda = 1;
    Tensor4 I4;
public:
    TElastoPlasticityAssemble (int _dim);
    void SetDirichlet (Vector& u, double t);
    void Update (const Tensor epsilon2, const Tensor epsilonp1, const double alpha1, const Tensor beta1,
		 Tensor& epsilonp2, double& alpha2, Tensor& beta2, Tensor4& C2, Tensor& sigma2);
    void Update (Vector& x, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1);
    // dynamic explict, reduction integration, hourglass
    void Jacobi (Matrix& A);
    double Residual (Vector& b, Vector& g_D, Vector& x0, Vector& x1, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1,
		     double time=0);
    // mass lumping
    void Jacobi2 (Matrix& A);
    double Residual2 (Vector& b, Vector& g_D, Vector& x0, Vector& x1, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1,
		      double time=0);
    // total lagrange, st. venant kirchhoss
    void Jacobi3 (Matrix& A);
    double Residual3 (Vector& b, Vector& g_D, Vector& x0, Vector& x1, Vector& EpsilonP1, Vector& Alpha1, Vector& Beta1,
		      double time=0);
    // update lagrange
    void Jacobi4 (Matrix& A, Vector& x1);
    double Residual4 (Vector& b, Vector& x1, Vector& _s1ud, Vector& _ep1ud, Vector& _st1, Vector& Alpha1, Vector& Beta1,
		      double time=0);
    double L2Error (const Vector& x);
    void SetH0 (Vector& x0);
    void SetH1 (Vector& x1, double dt);
	
    double* coords1 = NULL;
    int* ids1 = NULL;
    bool* fixed1 = NULL;
    int num_cells;
    int num_vertices;
    vector<Point> vertices1;
    vector<Point> disp0;
    vector<Point> disp1;
    vector<Point> cells1;
    vector<Tensor> strain1ud;
    vector<Tensor> epsilonp1ud;
    vector<Tensor> stress1;
    vector<Tensor> stress1ud;
    vector<double> rho1;
    vector<Point> bnds1;
    vector<int> bndsid1;
    double rho0;
    void Update4 (Vector& _x1, Vector& _s1ud, Vector& _ep1ud, Vector& _st1);
    void MeshExport (const Vector& x1);
    void MeshBndImport (Meshes& M2);
    void MeshSmoothing ();
    void DispExport (const Vector& x0, const Vector& x1, const Vector& x3);
    void DispImport (Vector& x0, Vector& x1);
    void PhysicsExport (const Vector& x1, const Vector& s1ud, const Vector& ep1ud, const Vector& st1);
    void PhysicsImport (Vector& s1ud, Vector& ep1ud, Vector& st1);
    void TESolver4 (Meshes& M2, int num, double dT);
	
    // logarithmic strain
    void Update5 (const Vector& _TDisp, Vector& _CStress);
    void TESolver5 (Meshes& M2, int num, double dT);
    void Jacobi5 (Matrix& A);
    double Residual5 (Vector& b, Vector& CS, Vector& x1, Vector& gd, double time=0);

    void PhysicsExport5 (const Vector& LStrain);
    void PhysicsImport5 (Vector& LStrain);
    void PhysicsExport6 (const Vector& _TDisp, const Vector& _RDisp);
    void PhysicsImport6 (Vector& _TDisp, Vector& _RDisp);

    void Update6 (const Vector& _RDisp, Vector& _LStrain, Vector& _CStress);
    void TESolver6 (Meshes& M2, int num, double dT);
    void Jacobi6 (Matrix& A);
    double Residual6 (Vector& b, Vector& CS, double time=0);

    void Update7 (const Vector& _RDisp, Vector& _LStrain, Vector& _CStress);
    void TESolver7 (Meshes& M2, int num, double dT);
    void Jacobi7 (Matrix& A);
    double Residual7 (Vector& b, Vector& CS, double time=0);
	
    vector<Point> TotalDisp;
    vector<Point> RefDisp;
    vector<Tensor> LogStrain;
    vector<Tensor> CauchyStress;
	
};

void LogStrain (const Tensor&F, Tensor&S);

// dynamic explict, reduction integration, hourglass
void TElastoPlasticityMain ();
// mass lumping
void TElastoPlasticity2Main ();
// total lagrange, st. venant kirchhoss
void TElastoPlasticity3Main ();
// updated lagrange
void TElastoPlasticity4Main ();
// updated lagrange logarithmic strain
void TElastoPlasticity5Main ();
// updated lagrange logarithmic strain
void TElastoPlasticity6Main ();
// updated lagrange objective stress rate
void TElastoPlasticity7Main ();

void stop ();

void LogStrainEigen3 (const Tensor&F, Tensor&S);
void ExpStrainEigen3 (const Tensor&F, Tensor&S);
void ExpStrainEigen4 (const Tensor&F, Tensor&S);
void ExpStrainEigen5 (const Tensor&F, Tensor&S);
