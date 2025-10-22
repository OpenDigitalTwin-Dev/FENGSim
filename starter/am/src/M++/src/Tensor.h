 // file: Tensor.h
// author: Christian Wieners
// $Header: /public/M++/src/Tensor.h,v 1.23 2009-10-01 11:53:58 mueller Exp $

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "Debug.h"
#include "Point.h"
#include "IO.h"

template <class A, class B>
class constAB {
    const A& a;
    const B& b;
public:
    constAB (const A& _a, const B& _b) : a(_a), b(_b) {} 
    const A& first () const { return a; }
    const B& second () const { return b; }
};
template <class A, class B>
inline constAB<A,B> operator * (const A& a, const B& b) { 
    return constAB<A,B>(a,b);
}

class Gradient {
    Scalar z[3];
public:
    Gradient () { z[0] = z[1] = z[2] = 0.0; }
    Gradient (const Point& y) { z[0]=y[0]; z[1]=y[1]; z[2]=y[2]; }
    Gradient (const Gradient& y) { z[0]=y[0]; z[1]=y[1]; z[2]=y[2]; }
    Gradient (const constAB<Scalar,Point>& c) { 
		z[0] = c.first() * c.second()[0];  
		z[1] = c.first() * c.second()[1];  
		z[2] = c.first() * c.second()[2];  
    }
    Gradient (Scalar a, Scalar b, Scalar c=0) { z[0]=a; z[1]=b; z[2]=c; } 
    Gradient (double a) { z[0]=a; z[1]=a; z[2]=a; } 
    Gradient (Scalar* a) { z[0]=a[0]; z[1]=a[1]; z[2]=a[2]; } 
    Scalar operator [] (const unsigned int i) const { return z[i]; }
    Scalar& operator [] (const unsigned int i) { return z[i]; }
    const Scalar* operator () () const { return z; }
    Gradient& operator = (double a) { z[0] = z[1] = z[2] = a; return *this; }
    Gradient& operator = (const Gradient& y) { 
		z[0]=y[0]; z[1]=y[1]; z[2]=y[2];
		return *this; 
    }
    Gradient& operator += (const Gradient& y) {
		z[0] += y[0]; z[1] += y[1]; z[2] += y[2]; return *this; 
    }
    Gradient& operator += (const constAB<Scalar,Point>& c) { 
		z[0] += c.first() * c.second()[0];  
		z[1] += c.first() * c.second()[1];  
		z[2] += c.first() * c.second()[2];  
    }
    Gradient& operator -= (const Gradient& y) {
		z[0] -= y[0]; z[1] -= y[1]; z[2] -= y[2]; return *this; 
    }
    Gradient& operator = (const Point& y) { 
		z[0]=y[0]; z[1]=y[1]; z[2]=y[2];
		return *this; 
    }
    Gradient& operator += (const Point& y) {
		z[0] += y[0]; z[1] += y[1]; z[2] += y[2]; return *this; 
    }
    Gradient& operator -= (const Point& y) {
		z[0] -= y[0]; z[1] -= y[1]; z[2] -= y[2]; return *this; 
    }
    Gradient& operator *= (Scalar a) {
		z[0] *= a; z[1] *= a; z[2] *= a; return *this; 
    }
    Gradient& operator /= (Scalar a) {
		z[0] /= a; z[1] /= a; z[2] /= a; return *this; 
    }
    friend Gradient conj (const Gradient& x) {
		Gradient z;
		z.z[0] = conj(x.z[0]);
		z.z[1] = conj(x.z[1]);
		z.z[2] = conj(x.z[2]);
		return z;
    }
#ifdef NDOUBLE
    Gradient (const constAB<double,Point>& c) { 
		z[0] = c.first() * c.second()[0];  
		z[1] = c.first() * c.second()[1];  
		z[2] = c.first() * c.second()[2];  
    }
    Gradient& operator *= (double a) {
		z[0] *= a; z[1] *= a; z[2] *= a; return *this; 
    }
    Gradient& operator /= (double a) {
		z[0] /= a; z[1] /= a; z[2] /= a; return *this; 
    }
    Gradient (double a, double b, double c=0) { z[0]=a; z[1]=b; z[2]=c; } 
    Gradient (double* a) { z[0]=a[0]; z[1]=a[1]; z[2]=a[2]; } 
    friend Gradient operator * (double b, const Gradient& x) {
	Gradient z = x; return z *= b;
    }
    friend Gradient operator * (Scalar b, const Point& x) {
		Gradient z = x; return z *= b;
    }
#endif
};
typedef Gradient Displacement;
typedef Gradient Deformation;
typedef Gradient Velocity;
typedef Gradient VectorField;

inline Gradient operator + (const Gradient& x, const Point& y) {
    Gradient z = x; return z += y;
}
inline Gradient operator + (const Point& x, const Gradient& y) {
    Gradient z = x; return z += y;
}
inline Gradient operator + (const Gradient& x, const Gradient& y) {
    Gradient z = x; return z += y;
}
inline Gradient operator + (const constAB<Scalar,Point>& c) { 
    Gradient z = c.second()[0]; return z *= c.first();
}
inline Gradient operator - (const Gradient& x, const Point& y) {
    Gradient z = x; return z -= y;
}
inline Gradient operator - (const Point& x, const Gradient& y) {
    Gradient z = x; return z -= y;
}
inline Gradient operator - (const Gradient& x, const Gradient& y) {
    Gradient z = x; return z -= y;
}
inline Gradient operator ^ (const Gradient& x, const Point& y) {
    return Gradient(x[1]*y[2]-x[2]*y[1],
					x[2]*y[0]-x[0]*y[2],
					x[0]*y[1]-x[1]*y[0]);
}
inline Gradient operator ^ (const Point& x, const Gradient& y) {
    return Gradient(x[1]*y[2]-x[2]*y[1],
					x[2]*y[0]-x[0]*y[2],
					x[0]*y[1]-x[1]*y[0]);
}
inline Gradient operator ^ (const Gradient& x, const Gradient& y) {
    return Gradient(x[1]*y[2]-x[2]*y[1],
					x[2]*y[0]-x[0]*y[2],
					x[0]*y[1]-x[1]*y[0]);
}
inline Scalar operator * (const Gradient& x, const Point& y) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}
inline Scalar operator * (const Point& x, const Gradient& y) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}
inline Scalar operator * (const Gradient& x, const Gradient& y) {
    return conj(x[0])*y[0] + conj(x[1])*y[1] + conj(x[2])*y[2];
}
inline Gradient operator * (const Scalar& b, const Gradient& x) {
    Gradient z = x; return z *= b;
}
inline Gradient operator * (const Gradient& x, double b) {
    Gradient z = x; return z *= b;
}
inline Gradient operator * (int b, const Gradient& x) {
    Gradient z = x; return z *= double(b);
}
inline Gradient operator / (const Gradient& x, double b) {
    Gradient z = x; return z *= (1.0/b); 
}
inline double norm (const Gradient& x) { return sqrt(real(x*x)); }
inline ostream& operator << (ostream& s, const Gradient& Du) {
    return s << Du[0] << " " << Du[1] << " " << Du[2]; 
}

class Tensor {
protected:
    Gradient t[3];
public:
    Tensor () { t[0] = t[1] = t[2] = 0; }
    Tensor (double a) { t[0] = t[1] = t[2] = a; }
    Tensor (int i, int j) { 
		t[0] = t[1] = t[2] = zero; 
		t[i][j] = 1; 
    }
    Tensor (Scalar a00, Scalar a01, Scalar a02,
			Scalar a10, Scalar a11, Scalar a12,
			Scalar a20, Scalar a21, Scalar a22) {
		t[0][0] = a00; t[1][0] = a10; t[2][0] = a20;
		t[0][1] = a01; t[1][1] = a11; t[2][1] = a21;
		t[0][2] = a02; t[1][2] = a12; t[2][2] = a22;
    }
    Tensor (Scalar a00, Scalar a01,
			Scalar a10, Scalar a11) {
		t[0][0] = a00; t[1][0] = a10; t[2][0] = 0;
		t[0][1] = a01; t[1][1] = a11; t[2][1] = 0;
		t[0][2] = 0;   t[1][2] = 0;   t[2][2] = 0;
    }
    Tensor& operator = (const Tensor& S) {
		t[0] = S.t[0];
		t[1] = S.t[1];
		t[2] = S.t[2];
		return *this;
    }
    Tensor& operator += (const Tensor& S) {
		t[0] += S.t[0];
		t[1] += S.t[1];
		t[2] += S.t[2];
		return *this;
    }
    Tensor& operator -= (const Tensor& S) {
		t[0] -= S.t[0];
		t[1] -= S.t[1];
		t[2] -= S.t[2];
		return *this;
    }
    Tensor& operator *= (const Scalar& a) {
		t[0] *= a;
		t[1] *= a;
		t[2] *= a;
		return *this;
    }    
    const Gradient& operator [] (int k) const { return t[k]; }
    Gradient& operator [] (int k) { return t[k]; }
    friend Tensor sym (const Tensor& T) {
		Scalar t01 = 0.5 * (T[1][0] + T[0][1]);
		Scalar t02 = 0.5 * (T[2][0] + T[0][2]);
		Scalar t12 = 0.5 * (T[1][2] + T[2][1]);
		return Tensor(T[0][0],    t01,    t02,
					  t01,T[1][1],    t12,
					  t02,    t12,T[2][2]);
    }
    friend Tensor skew (const Tensor& T) {
		Scalar t01 = 0.5 * (T[0][1] - T[1][0]);
		Scalar t02 = 0.5 * (T[0][2] - T[2][0]);
		Scalar t12 = 0.5 * (T[1][2] - T[2][1]);
		return Tensor(      0,    t01,    t02,
							-t01,      0,    t12,
							-t02,   -t12,      0);
    }
    friend Tensor transpose (const Tensor& T) {
		return Tensor(T[0][0],T[1][0],T[2][0],
					  T[0][1],T[1][1],T[2][1],
					  T[0][2],T[1][2],T[2][2]);
    }
    friend Scalar trace (const Tensor& T) { return T[0][0]+T[1][1]+T[2][2]; }
    friend Scalar det (const Tensor& T) {
		return T[0][0]*T[1][1]*T[2][2]
			+ T[1][0]*T[2][1]*T[0][2]
			+ T[2][0]*T[0][1]*T[1][2]
			- T[2][0]*T[1][1]*T[0][2]
			- T[0][0]*T[2][1]*T[1][2]
			- T[1][0]*T[0][1]*T[2][2];
    }
    friend Scalar Frobenius (const Tensor& S, const Tensor& T) {
		return S[0]*T[0] + S[1]*T[1] + S[2]*T[2];
    }
    friend double norm (const Tensor& T) {
		return sqrt(abs(T[0]*T[0]+T[1]*T[1]+T[2]*T[2]));
    }
    friend double maxelem(const Tensor& T) {
		double m = 0;
		for (int i=0; i<3; ++i)
			for (int j=0; j<3; ++j)
				if (abs(T[i][j])>m) m = abs(T[i][j]);
		return m;
    }
    friend Tensor operator + (const Tensor& S, const Tensor& T) {
		Tensor ST = S; return ST += T;
    }
    friend Tensor operator - (const Tensor& S, const Tensor& T) {
		Tensor ST = S; return ST -= T;
    }
    friend Tensor operator * (const Scalar& b, const Tensor& T) {
		Tensor S = T; return S *= b;
    }
    friend Tensor operator / (const Tensor& T, const Scalar& b) {
		Tensor S = T; return S *= 1.0/b;
    }
    friend Tensor operator * (const Tensor& S, const Tensor& T) {
		return Tensor(S[0][0]*T[0][0]+S[0][1]*T[1][0]+S[0][2]*T[2][0],
					  S[0][0]*T[0][1]+S[0][1]*T[1][1]+S[0][2]*T[2][1],
					  S[0][0]*T[0][2]+S[0][1]*T[1][2]+S[0][2]*T[2][2],
					  S[1][0]*T[0][0]+S[1][1]*T[1][0]+S[1][2]*T[2][0],
					  S[1][0]*T[0][1]+S[1][1]*T[1][1]+S[1][2]*T[2][1],
					  S[1][0]*T[0][2]+S[1][1]*T[1][2]+S[1][2]*T[2][2],
					  S[2][0]*T[0][0]+S[2][1]*T[1][0]+S[2][2]*T[2][0],
					  S[2][0]*T[0][1]+S[2][1]*T[1][1]+S[2][2]*T[2][1],
					  S[2][0]*T[0][2]+S[2][1]*T[1][2]+S[2][2]*T[2][2]);
    }
#ifndef NDOUBLE 
    friend Point operator * (const Tensor& T, const Point& D) {
		return Point(T[0]*D,T[1]*D,T[2]*D);
    }
#endif
    friend Tensor Invert (const Tensor& S) {
		// T = S^{-1} = 1/det(S) * transpose(Cof(S))
		Scalar det = S[0][0]*S[1][1]*S[2][2]
			+ S[1][0]*S[2][1]*S[0][2]
			+ S[2][0]*S[0][1]*S[1][2]
			- S[2][0]*S[1][1]*S[0][2]
			- S[0][0]*S[2][1]*S[1][2]
			- S[1][0]*S[0][1]*S[2][2];
		if (abs(det)<Eps)   
			Exit("Error in Tensor Inversion: det<Eps; file: Tensor.h");
		Scalar invdet = 1.0/det;
		Tensor T = 0;
		T[0][0] = ( S[1][1]*S[2][2] - S[2][1]*S[1][2]) * invdet; 
		T[0][1] = (-S[0][1]*S[2][2] + S[2][1]*S[0][2]) * invdet;
		T[0][2] = ( S[0][1]*S[1][2] - S[1][1]*S[0][2]) * invdet;
		T[1][0] = (-S[1][0]*S[2][2] + S[2][0]*S[1][2]) * invdet;
		T[1][1] = ( S[0][0]*S[2][2] - S[2][0]*S[0][2]) * invdet; 
		T[1][2] = (-S[0][0]*S[1][2] + S[1][0]*S[0][2]) * invdet;
		T[2][0] = ( S[1][0]*S[2][1] - S[2][0]*S[1][1]) * invdet;
		T[2][1] = (-S[0][0]*S[2][1] + S[2][0]*S[0][1]) * invdet;
		T[2][2] = ( S[0][0]*S[1][1] - S[1][0]*S[0][1]) * invdet; 
		return T;
    }
    Tensor& operator += (const constAB<Scalar,Tensor>& cT) { 
		Tensor S = cT.second();
		S *= cT.first();
		*this += S; 
		return *this;
    }
    friend double maxnorm (const Tensor& T) {
		// maxnorm = max_i \sum_{j=1}^m |t_ij|
		double sum;
		double n = 0;
		for (int i = 0; i<3; ++i) {
	    sum = 0;
	    for (int j = 0; j<3; ++j) {
			sum += abs(T[i][j]);
	    }
	    n = max(n,sum);
		}
		return n;
    }
    friend Tensor MatrixGauss (const Tensor& D, const Tensor& N) {
		// solves DF = N bei Gaussian elimination for F
		Scalar c0 = D[1][0]/D[0][0];
		Scalar c1 = D[2][0]/D[0][0];
		Scalar c2 = (D[2][1] - c1*D[0][1])/(D[1][1]-c0*D[0][1]);
		
		Tensor F;
		for(int j = 0; j<3; ++j){
			F[2][j] = (N[2][j] - c1*N[0][j] - c2*(N[1][j]-c0*N[0][j]))/
				(D[2][2] - c1*D[0][2] - c2*(D[1][2]-c0*D[0][2]));
			F[1][j] = (N[1][j] - c0*N[0][j] - (D[1][2]-c0*D[0][2])*F[2][j])/
				(D[1][1] - c0*D[0][1]);
			F[0][j] = (N[0][j] - D[0][2]*F[2][j] - D[0][1]*F[1][j])/D[0][0];
		}
		return F;
    }
#ifndef NDOUBLE 
    friend Tensor exp (const Tensor& T) {
		// exp(T) by scaling and squaring algorithm using Pade approximation
		// see Golub: 'Matrix Computations' pg.573
		double nt = maxnorm(T);
		double j  = max(0.0,1.0+floor(log(nt)/log(2.0))); // log_2(|T|_infty)
		// scaling
		Tensor A  = (1.0/pow(2.0,j))*T;
		double delta = Eps + Eps * norm(T);
		
		int q = 1; // corresponds to delta >= 1/6
		if (delta >= 1.0/1440) {
			q = 2;  dout(10) << "q = 2: eps = " << 1.0/1440 << "\n"; }
		else if (delta >= 1.0/806400) {
			q = 3;  dout(10) << "q = 3: eps = " << 1.0/806400 << "\n"; }
		else if (delta >= 1.0/812851200) {
	    q = 4;  dout(10) << "q = 4: eps = " << 1.0/812851200 << "\n"; }
		else if (delta >= 7.76665066513e-13) { 
			q = 5;  dout(10) << "q = 5: eps = " << 7.76665066513e-13 << "\n"; }
		else if (delta >= 3.39451515084e-16) { 
			q = 6;  dout(10) << "q = 6: eps = " << 3.39451515084e-16 << "\n"; }
		else if (delta >= 1.08798562527e-19) { 
			q = 7;  dout(10) << "q = 6: eps = " << 1.08798562527e-19 << "\n"; }
		else Exit("exp: Selected accuracy unrealistic; file: Tensor.h\n");
		
		Tensor D (1,0,0,
				  0,1,0,
				  0,0,1);
		Tensor N (1,0,0,
				  0,1,0,
				  0,0,1);
		Tensor X (1,0,0,
				  0,1,0,
				  0,0,1);
		double c = 1;
		int altern_sign = 1;
		for (int k = 1; k<q; ++k) {
			c  = (q-k+1)*c/((2*q - k+1) * k);
			X  = A * X;
			altern_sign *= -1;
			D += altern_sign * c * X;
			N += c * X;
		}
		
		// solve DF = N bei Gaussian Elimination for F
		Tensor F = MatrixGauss(D,N);
		// squaring
		for (int k = 1; k<=int(j); ++k) F = F*F;
		
		return F;
    }
    friend int JacobiRotation (Tensor& S, Tensor& U, int i, int j) {
		const double EPS = 1e-10;
		double a_ij = S[i][j];
		if (abs(a_ij) < EPS) return 1;
		double a_ii = S[i][i];
		double a_jj = S[j][j];
		Tensor J (1,0,0,
				  0,1,0,
				  0,0,1);
		double t = (a_jj - a_ii) / (2.0 * a_ij);
		if (abs(t) < EPS) t = 1;
		else if (t > 0 )   t = 1 / (t + sqrt(t*t+1));
		else               t = 1 / (t - sqrt(t*t+1));
		J[i][i] = J[j][j] = 1 / sqrt(1+t*t);
		J[j][i] = t * J[j][j];  
		J[i][j] = - J[j][i];
		S = J * S * transpose(J);
		U = J * U;
		return 0;
    }
    friend Tensor diagonalize (Tensor& S) {
		Tensor S_old(S);
		const int ITER = 100;
		Tensor U (1,0,0,
				  0,1,0,
				  0,0,1);
		int i;	
		for (i=0; i<=ITER; ++i) {
			double a_10 = abs(S[1][0]);
			double a_20 = abs(S[2][0]);
			double a_21 = abs(S[2][1]);
			if (a_10 > a_20) {
				if (a_10 > a_21) {
					if (JacobiRotation(S,U,1,0)) return U;
				}
				else if (JacobiRotation(S,U,2,1)) return U;
			}
			else {
				if (a_20 > a_21) {
					if (JacobiRotation(S,U,2,0)) return U;
				}
				else if (JacobiRotation(S,U,2,1)) return U;
			}
		}
		cerr << "No convergence in diagonalize; file Tensor.h\n";
		mout << S_old << endl;
    }
    friend Tensor log (Tensor E) {
		Tensor U = diagonalize(E);
		E[0][0] = log(E[0][0]);
		E[1][1] = log(E[1][1]);
		E[2][2] = log(E[2][2]);
		return transpose(U)*E*U;
    }
    friend Tensor pow (const Tensor& T, double x) {
		// determines T^x=U^T*D^x*U for symmetric 3x3 tensors
		Tensor A = T;
		Tensor U = diagonalize(A);
		A[0][0]  = pow(A[0][0],x);
		A[1][1]  = pow(A[1][1],x);
		A[2][2]  = pow(A[2][2],x);
		return transpose(U)*A*U;
    }
    friend Tensor sqrt (Tensor T) {
		Tensor A = T;
		Tensor U = diagonalize(A);
		A[0][0] = sqrt(A[0][0]);
		A[1][1] = sqrt(A[1][1]);
		A[2][2] = sqrt(A[2][2]);
		return transpose(U)*A*U;
    }
#endif
};

const Tensor One(1,0,0,
				 0,1,0,
				 0,0,1);
const Tensor Zero(0,0,0,
				  0,0,0,
				  0,0,0);

inline ostream& operator << (ostream& s, const Tensor& T) {
    return s << T[0][0] << " " << T[0][1] << " " << T[0][2] << endl
			 << T[1][0] << " " << T[1][1] << " " << T[1][2] << endl
			 << T[2][0] << " " << T[2][1] << " " << T[2][2] << endl;
}
inline Tensor dev (const Tensor& T) { return T - trace(T)/3.0 * One; }

inline VectorField axl (const Tensor& T) {
    return VectorField(T[2][1],T[0][2],T[1][0]);
}

inline Tensor anti (const VectorField& V) {
    return Tensor(0    ,-V[2], V[1],
				  V[2] ,0    ,-V[0],
				  -V[1],V[0] ,    0);
}

inline Tensor anti (double a, double b, double c) {
    return anti(VectorField(a,b,c));
}

inline Scalar Euklid (const VectorField& x, const VectorField& y) {
    return conj(x[0])*y[0] + conj(x[1])*y[1] + conj(x[2])*y[2];
}

class VelocityGradient : public Tensor {
public:
    VelocityGradient () {}
    VelocityGradient (double a) { t[0] = t[1] = t[2] = a; }
    VelocityGradient (const Gradient& G, int j) { 
		t[0] = t[1] = t[2] = zero; t[j] = G; 
    }
};

#endif
