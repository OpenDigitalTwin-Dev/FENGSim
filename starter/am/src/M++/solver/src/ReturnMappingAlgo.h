#ifndef _RETURNMAPPING_H_
#define _RETURNMAPPING_H_

#include "m++.h"

class Tensor4 {
	Tensor t4[3][3];
public:
    Tensor4 () {
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t4[i][j] = Zero;
    }
    void set (int i, int j, const Tensor& t2) {
		t4[i][j] = t2;
    }
    const Tensor& operator () (int i, int j) const {
		return t4[i][j];
    }
    const Tensor4& operator = (const Tensor4& _t4) { 
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t4[i][j] = _t4(i, j);
		return *this; 
    }
    const Tensor4& operator += (const Tensor4& _t4) {
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t4[i][j] += _t4(i, j);
		return *this;
    }
    const Tensor4& operator -= (const Tensor4& _t4) {
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t4[i][j] -= _t4(i, j);
		return *this;
    }
    const Tensor4& operator *= (double t) {
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				t4[i][j] *= t;
		return *this;
    }
};

Tensor4 DyadicProduct (const Tensor& t1, const Tensor& t2);

inline Tensor operator * (const Tensor4& t4, const Tensor& t2) {
    Tensor s2;
    for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			s2[i][j] = Frobenius(t4(i,j), t2);
    return s2;
}
inline Tensor4 operator + (const Tensor4& t1, const Tensor4& t2) {
    Tensor4 s4 = t1;
    s4 += t2;
    return s4;
}
inline Tensor4 operator - (const Tensor4& t1, const Tensor4& t2) {
    Tensor4 s4 = t1;
    s4 -= t2;
    return s4;
}
inline Tensor4 operator * (const Tensor4& t4, double t) {
    Tensor4 s4 = t4;
    s4 *= t;
    return s4;
}
inline Tensor4 operator * (double t, const Tensor4& t4) {
    Tensor4 s4 = t4;
    s4 *= t;
    return s4;
}
inline Tensor4 operator / (const Tensor4& t4, double t) {
    Tensor4 s4 = t4;
    s4 *= (1.0 / t);
    return s4;
}
inline ostream& operator << (ostream& s, const Tensor4& t4) {
    for (int i=0; i<3; i++) {
		for (int k=0; k<3; k++) {
			for (int j=0; j<3; j++) {
				for (int l=0; l<3; l++)
					s << t4(i,j)[k][l] << " ";
			}
			s << endl;
		}
    }
    s << endl;
    return s;
}

void SetTensorToVector (Tensor& T, Point p, Vector& q);
Tensor GetTensorFromVector (Point p, const Vector& q);
void SetTensorToVector (Tensor& T, Point p, Vector& q, int k);
Tensor GetTensorFromVector (Point p, Vector& q, int k);

class ReturnMappingAlgo {
public:
    double Mu = 1;
    double Lambda = 1;
    Tensor4 I4;
public:
    ReturnMappingAlgo () {
      	for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				I4.set(i, j, Tensor(i,j));
			}
		}
    }
    ReturnMappingAlgo (double _Mu, double _Lambda) {
		Mu = _Mu;
		Lambda = _Lambda;
    }
    Tensor Xi (Tensor epsilon2, Tensor epsilonp1, Tensor beta1);
    Tensor Normal (Tensor epsilon2, Tensor epsilonp1, Tensor beta1);
    double H (double alpha);
    double K (double alpha);
    double dH (double alpha);
    double dK (double alpha);
    bool IsPlasticity (Tensor epsilon2, Tensor epsilonp1, double alpha1, Tensor beta1);
    double g (double alpha1, double gamma2, Tensor xi2);
    double dg (double alpha1, double gamma2);
    double CalGamma (Tensor epsilon2, Tensor epsilonp1, double alpha1, Tensor beta1);
    double UpdateAlpha (double alpha1, double gamma2);
    Tensor UpdateBeta (Tensor epsilon2, Tensor epsilonp1, double alpha1, double alpha2, Tensor beta1);
    Tensor UpdateEpsilonP (Tensor epsilon2, Tensor epsilonp1, Tensor beta1, double gamma2);
    Tensor UpdateSigma (Tensor epsilon2, Tensor epsilonp1, Tensor beta1, double gamma2);
    Tensor4 UpdateTangent (Tensor epsilon2, Tensor epsilonp1, Tensor beta1, double alpha2, double gamma2);
    Tensor4 UpdateElasticTangent ();
};


#endif
