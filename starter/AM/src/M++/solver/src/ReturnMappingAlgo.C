#include "ReturnMappingAlgo.h"

Tensor4 DyadicProduct (const Tensor& t1, const Tensor& t2) {
    Tensor4 t;
    for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			t.set(i, j, t1[i][j] * t2);
		}
    }
    return t;
}

void SetTensorToVector (Tensor& T, Point p, Vector& q) {
    for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			q(p, i*3+j) = T[i][j];
}

Tensor GetTensorFromVector (Point p, const Vector& q) {
    double a[9];
    for (int i = 0; i < 9; i++)
		a[i] = q(p,i);
    return Tensor(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
}

void SetTensorToVector (Tensor& T, Point p, Vector& q, int k) {
    for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			q(p, i*3+j+k*9) = T[i][j];
}

Tensor GetTensorFromVector (Point p, Vector& q, int k) {
    double a[9];
    for (int i = 0; i < 9; i++)
		a[i] = q(p, i+k*9);
    return Tensor(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
}

Tensor ReturnMappingAlgo::Xi (Tensor epsilon2, Tensor epsilonp1, Tensor beta1) {
    Tensor sigma2t = 2.0 * Mu * (epsilon2 - epsilonp1) + Lambda * trace(epsilon2 - epsilonp1) * One;
    return dev(sigma2t) - beta1;
}

Tensor ReturnMappingAlgo::Normal (Tensor epsilon2, Tensor epsilonp1, Tensor beta1) {
    double t = norm(Xi(epsilon2,epsilonp1,beta1));
    if (t == 0) {
		return Zero;
    }
    Tensor S = Xi(epsilon2,epsilonp1,beta1);
    S *= (1/t);
    return  S;
}

double ReturnMappingAlgo::H (double alpha) {
    return 0.0;
}

double ReturnMappingAlgo::dH (double alpha) {
    return 0.0;
}

double ReturnMappingAlgo::K (double alpha) {
    // elasticity
    // return infty;
    // perfect plasticity
    return 0.24;
    // linear isotropic hardening
    return alpha + 0.2;
    // exponential isotropic hardening
    return alpha * alpha + 0.2;
}

double ReturnMappingAlgo::dK (double alpha) {
    // elasticity
    // return 0;
    // perfect plasticity
    return 0;
    // linear isotropic hardening
    return 1.0;
    // exponential isotropic hardening
    return 2.0 * alpha;
}

bool ReturnMappingAlgo::IsPlasticity (Tensor epsilon2, Tensor epsilonp1, double alpha1, Tensor beta1) {
    double s = norm(Xi(epsilon2,epsilonp1,beta1)) - sqrt(2.0/3.0) * K(alpha1);
    if (s > 0) {
		return true;
    }
    return false;
}

double ReturnMappingAlgo::g (double alpha1, double _gamma2, Tensor xi2) {
    double t = norm(xi2) - 2.0 * Mu * _gamma2
        - sqrt(2.0/3.0) * ( H( alpha1 + sqrt(2.0/3.0) * _gamma2 ) - H( alpha1 ) )
        - sqrt(2.0/3.0) * K( alpha1 + sqrt(2.0/3.0) * _gamma2 )
		;
    return t;
}

double ReturnMappingAlgo::dg (double alpha1, double _gamma2) {
    double t = -2.0 * Mu
		- 2.0/3.0 * dH( alpha1 + sqrt(2.0/3.0) * _gamma2 )
		- 2.0/3.0 * dK( alpha1 + sqrt(2.0/3.0) * _gamma2 )
		;
    return t;
}

double ReturnMappingAlgo::CalGamma (Tensor epsilon2, Tensor epsilonp1, double alpha1, Tensor beta1) {
    double residual = infty;
    double TOL = 1e-10;
    double gamma2 = 0;
    Tensor xi2 = Xi(epsilon2,epsilonp1,beta1);
    while (abs(residual) > TOL) {
		gamma2 += ( -1.0 * g(alpha1,gamma2,xi2) / dg(alpha1,gamma2) );
		residual = g(alpha1,gamma2,xi2);
    }
    
    return gamma2;
}

double ReturnMappingAlgo::UpdateAlpha (double alpha1, double gamma2) {
    return alpha1 + sqrt(2.0/3.0) * gamma2;
}

Tensor ReturnMappingAlgo::UpdateBeta (Tensor epsilon2, Tensor epsilonp1, double alpha1, double alpha2, Tensor beta1) {
    Tensor n2 = Normal(epsilon2,epsilonp1,beta1);
    return beta1 + sqrt(2.0/3.0) * ( H(alpha2) - H(alpha1) ) * n2;
}

Tensor ReturnMappingAlgo::UpdateEpsilonP (Tensor epsilon2, Tensor epsilonp1, Tensor beta1, double gamma2) {
    Tensor n2 = Normal(epsilon2, epsilonp1, beta1);
    return epsilonp1 + gamma2 * n2;
}

Tensor ReturnMappingAlgo::UpdateSigma (Tensor epsilon2, Tensor epsilonp1, Tensor beta1, double gamma2) {
    Tensor n2 = Normal(epsilon2, epsilonp1, beta1);
    Tensor sigma2;
    
    //Tensor sigma2 = 2.0 *Mu * (epsilon2 - epsilonp1) + Lambda * trace(epsilon2 - epsilonp1) * One - 2.0 * Mu * gamma2 * n2;
	
    Tensor4 C;
    //C = 2.0 * Mu * I4 + Lambda * DyadicProduct(One,One);

	
    
    sigma2 = C * (epsilon2 - epsilonp1 - gamma2 * n2);
    return sigma2;
}

Tensor4 ReturnMappingAlgo::UpdateTangent (Tensor epsilon2, Tensor epsilonp1, Tensor beta1, double alpha2, double gamma2) {
    Tensor xi2 = Xi(epsilon2, epsilonp1, beta1);
    // Tensor n2 = xi2 / norm(xi2);
    Tensor n2 = Normal(epsilon2, epsilonp1, beta1);
    Tensor4 t4;
    double theta1 = 1.0 - 2.0 * Mu * gamma2 / norm(xi2);
    double theta2 = 1.0 / (1.0 + ( dK(alpha2) + dH(alpha2) ) / 3.0 / Mu) - (1.0 - theta1);
    //t4 = (Lambda + 2.0 / 3.0 * Mu ) * DyadicProduct(One,One) + 2.0 * Mu * theta1 * (I4 - 1.0/3.0 * DyadicProduct(One,One) ) - 2.0 * Mu * theta2 * DyadicProduct(n2,n2);
    return t4;
}

Tensor4 ReturnMappingAlgo::UpdateElasticTangent () {
    Tensor4 t4;
    //t4 = 2.0 * Mu * I4 + Lambda * DyadicProduct(One,One);
    return t4;
}
