// file: Shape.h
// author: Christian Wieners, Wolfgang Mueller
// $Header: /public/M++/src/Shape.h,v 1.24 2009-11-24 09:46:35 wieners Exp $

#ifndef _SHAPE_H_
#define _SHAPE_H_

#include "IO.h" 
#include "Quadrature.h" 
 
typedef double ShapeValue[MaxQuadraturePoints][MaxNodalPoints];
typedef Point ShapeGradient[MaxQuadraturePoints][MaxNodalPoints];

class Shape {
    const Quadrature& Q;
    const Quadrature& FaceQ;
    ShapeValue value;
    ShapeGradient localgradient;
 public:
    virtual double operator () (const Point&, int) const {}
    virtual Point LocalGradient (const Point&, int) const {}
    virtual Point LocalVector (const Point&, int) const {}
    virtual Point LocalCurl (const Point&, int) const {}
    virtual double LocalDiv (const Point&, int) const {}
    virtual double LocalLaplace (const Point&, int) const {}
    virtual int size () const = 0;
    const Quadrature& GetQuad () const { return Q; }
    const Quadrature& GetFaceQuad () const { return FaceQ; }
    const ShapeValue& values () const { return value; }
    double operator () (int q, int i) const { return value[q][i]; }
    Point LocalGradient (int q, int i) const { return localgradient[q][i]; }
    Point Eval (const Point& z, const vector<Point>& x) const;
    void EvalGradient (vector<Point>&, const Point&, 
		       const vector<Point>&) const;
    void fill();
    void fill(int);
    Shape (const Quadrature& _Q, const Quadrature& _FaceQ) ;
    virtual ~Shape() {};
    friend ostream& operator << (ostream& s, const Shape& S);
};

// ------------------------------------------------------------
// 3-NODED TRIANGLE 
// ------------------------------------------------------------
class P1tri : public Shape {
public:
	double operator () (const Point& z, int i) const { 
		switch (i) {
		case 0: return (1-z[0]-z[1]);
		case 1: return z[0];
		case 2: return z[1];
		}
    }
    Point LocalGradient (const Point& z, int i) const { 
		switch (i) {
		case 0: return Point(-1.0,-1.0);
		case 1: return Point(1.0,0.0);
		case 2: return Point(0.0,1.0);
		}
    }
    int size () const { return 3; }
    P1tri (const Quadrature& Q, 
		   const Quadrature& FaceQ = GetQuadrature("Qint1")) 
		: Shape(Q,FaceQ) { fill(); }
};
const P1tri P1_tri(GetQuadrature("Qtri1"), GetQuadrature("Qint1"));

// ------------------------------------------------------------
// 3-NODED TRIANGLE FOR PPP
// ------------------------------------------------------------
class P1triPPP : public Shape {
public:
    double operator () (const Point& z, int i) const { 
        switch (i) {
		case 0: return (3.0 - 4.0 * z[0] - 4.0 * z[1]);
		case 1: return 4.0 * z[0] - 1.0;
		case 2: return 4.0 * z[1] - 1.0;
		}
    }
    Point LocalGradient (const Point& z, int i) const { 
		switch (i) {
		case 0: return Point(-4.0,-4.0);
		case 1: return Point(4.0,0.0);
		case 2: return Point(0.0,4.0);
		}
    }
    int size () const { return 3; }
    P1triPPP (const Quadrature& Q, 
			  const Quadrature& FaceQ = GetQuadrature("Qint1")) 
		: Shape(Q,FaceQ) { fill(); }
};
const P1triPPP P1_tri_ppp(GetQuadrature("Qtri1"), GetQuadrature("Qint1"));

// ------------------------------------------------------------
// 6-NODED TRIANGLE
// ------------------------------------------------------------
class P2tri : public Shape {
public:
    double operator () (const Point& z, int i) const { 
		switch (i) {
		case  0: return((1-z[0]-z[1])*(1-2*z[0]-2*z[1]));
		case  1: return(z[0]*(2*z[0]-1));
		case  2: return(z[1]*(2*z[1]-1));	
		case  3: return(4*z[0]*(1-z[0]-z[1]));
		case  4: return(4*z[0]*z[1]);
		case  5: return(4*z[1]*(1-z[0]-z[1]));
		}
    }
    Point LocalGradient (const Point& z, int i) const { 
		switch (i) {
		case 0: return Point(-3+ 4*z[0]+ 4*z[1],-3+ 4*z[1]+ 4*z[0]);
		case 1: return Point(4*z[0] - 1,0);
		case 2: return Point(0,4*z[1] - 1);
		case 3: return Point(4 - 8*z[0] - 4*z[1],-4*z[0]);
		case 4: return Point(4*z[1],4*z[0]);
		case 5: return Point(-4*z[1],4 - 8*z[1] - 4*z[0]);
		}
    }
    int size () const { return 6; }
    P2tri (const Quadrature& Q, 
	   const Quadrature& FaceQ = GetQuadrature("Qint1")) 
		: Shape(Q,FaceQ) { fill(); }
};
const P2tri P2_tri(GetQuadrature("Qtri7"), GetQuadrature("Qint1"));

// ------------------------------------------------------------
// 4-NODED QUADRILATERAL
// ------------------------------------------------------------
class P1quad : public Shape {
public:
    double operator () (const Point& z, int i) const { 
		switch (i) {
		case 0: return (1-z[0])*(1-z[1]);
		case 1: return z[0]*(1-z[1]);
		case 2: return z[0]*z[1];
		case 3: return (1-z[0])*z[1];
		}
    }
    Point LocalGradient (const Point& z, int i) const { 
		switch (i) {
		case 0: return Point(-1+z[1],-1+z[0]);
		case 1: return Point(1-z[1],-z[0]);
		case 2: return Point(z[1],z[0]);
		case 3: return Point(-z[1],1-z[0]);
		}
    }
    int size () const { return 4; }
    P1quad (const Quadrature& Q, 
			const Quadrature& FaceQ = GetQuadrature("Qint1")) 
		: Shape(Q,FaceQ) { fill(); }
};
const P1quad P1_quad(GetQuadrature("Qquad4"), GetQuadrature("Qint1"));

// ------------------------------------------------------------
// 8-NODED QUADRILATERAL
// ------------------------------------------------------------
class P2quadSerendipity : public Shape {
 public:
    double operator () (const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    int size () const { return 8; }
    P2quadSerendipity (const Quadrature& Q, 
		       const Quadrature& FaceQ = GetQuadrature("Qint1"));
};
const P2quadSerendipity P2_quadSerendipity(GetQuadrature("Qquad9"), 
					   GetQuadrature("Qint1"));

// ------------------------------------------------------------
// 9-NODED QUADRILATERAL
// ------------------------------------------------------------
class P2quad : public Shape {
    inline double  q01 (const double x) const;
    inline double  q00 (const double x) const;
    inline double  q11 (const double x) const;
    inline double Dq01 (const double x) const;
    inline double Dq00 (const double x) const;
    inline double Dq11 (const double x) const;
    inline double shapefct (int i, int k, const double x) const;
    inline double Dshapefct (int i, int k, const double x) const;
 public:
    double operator () (const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    int size () const { return 9; }
    P2quad (const Quadrature& Q, 
		       const Quadrature& FaceQ = GetQuadrature("Qint1"));
};
const P2quad P2_quad(GetQuadrature("Qquad9"), 
		     GetQuadrature("Qint1"));



// ------------------------------------------------------------
// 16-NODED QUADRILATERAL
// ------------------------------------------------------------
class P3quad : public Shape {
    inline double c0 (const double x) const;
    inline double c1 (const double x) const;
    inline double c2 (const double x) const;
    inline double c3 (const double x) const;
    inline double Dc0 (const double x) const;
    inline double Dc1 (const double x) const;
    inline double Dc2 (const double x) const;
    inline double Dc3 (const double x) const;
 public:
    double operator () (const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    int size () const { return 16; }
    P3quad (const Quadrature& Q, const Quadrature& FaceQ = GetQuadrature("Qint1"));
};

const P3quad P3_quad(GetQuadrature("Qquad16"), GetQuadrature("Qint1"));

// ------------------------------------------------------------
// 4-NODED TETRAHEDRON
// ------------------------------------------------------------
class P1tet : public Shape {
 public:
    double operator () (const Point& z, int i) const { 
	switch (i) {
	case 0: return (1-z[0]-z[1]-z[2]);
	case 1: return z[0];
	case 2: return z[1];
	case 3: return z[2];
	}
    }
    Point LocalGradient (const Point& z, int i) const { 
	switch (i) {
	case 0: return Point(-1.0,-1.0,-1.0);
	case 1: return Point(1.0,0.0,0.0);
	case 2: return Point(0.0,1.0,0.0);
	case 3: return Point(0.0,0.0,1.0);
	}
    }
    int size () const { return 4; }
    P1tet (const Quadrature& Q, 
	   const Quadrature& FaceQ = GetQuadrature("Qtri1")) 
	: Shape(Q,FaceQ) { fill(); }
};
const P1tet P1_tet(GetQuadrature("Qtet1"), GetQuadrature("Qtri1"));

// ------------------------------------------------------------
// 10-NODED THETRAHEDRON
// ------------------------------------------------------------
class P2tet : public Shape {
 public:
    double operator () (const Point& z, int i) const { 
	switch (i) {
	case 0: return (2*(1-z[0]-z[1]-z[2])-1)*(1-z[0]-z[1]-z[2]);
	case 1: return (2*z[0]-1)*z[0];
	case 2: return (2*z[1]-1)*z[1];
	case 3: return (2*z[2]-1)*z[2];
	case 4: return 4*(1-z[0]-z[1]-z[2])*z[0];
	case 5: return 4*z[0]*z[1];
	case 6: return 4*(1-z[0]-z[1]-z[2])*z[1];
	case 7: return 4*(1-z[0]-z[1]-z[2])*z[2];
	case 8: return 4*z[0]*z[2];
	case 9: return 4*z[1]*z[2];
	}
    }
    Point LocalGradient (const Point& z, int i) const { 
	switch (i) {
	case 0: return Point(-4.0*(1-z[0]-z[1]-z[2])+1.0,-4.0*(1-z[0]-z[1]-z[2])+1.0,-4.0*(1-z[0]-z[1]-z[2])+1.0);
	case 1: return Point(4.0*z[0]-1.0,0.0,0.0);
	case 2: return Point(0.0,4.0*z[1]-1.0,0.0);
	case 3: return Point(0.0,0.0,4.0*z[2]-1.0);
	case 4: return Point(4.0*(1-2.0*z[0]-z[1]-z[2]),-4.0*z[0],-4.0*z[0]);
	case 5: return Point(4.0*z[1],4.0*z[0],0.0);
	case 6: return Point(-4.0*z[1],4.0*(1-z[0]-2.0*z[1]-z[2]),-4.0*z[1]);
	case 7: return Point(-4.0*z[2],-4.0*z[2],4.0*(1-z[0]-z[1]-2.0*z[2]));
	case 8: return Point(4.0*z[2],0.0,4.0*z[0]);
	case 9: return Point(0.0,4.0*z[2],4.0*z[1]);
	}
    }
    int size () const { return 10; }
    P2tet (const Quadrature& Q, 
	   const Quadrature& FaceQ = GetQuadrature("Qtri7")) 
	: Shape(Q,FaceQ) { fill(); }
};
const P2tet P2_tet(GetQuadrature("Qtet11"), GetQuadrature("Qtri7"));

// ------------------------------------------------------------
// 8-NODED HEXAHEDRON
// ------------------------------------------------------------

class P1hex : public Shape {
 public:
    double operator () (const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    int size () const { return 8; }
    P1hex (const Quadrature& Q, 
	   const Quadrature& FaceQ = GetQuadrature("Qquad1")) 
	: Shape(Q,FaceQ) { fill(); }
};
const P1hex P1_hex(GetQuadrature("Qhex8"), GetQuadrature("Qquad1"));

// ------------------------------------------------------------
// 27-NODED HEXAHEDRON (CONFORMING WITH P2curl_sp_hex)
// ------------------------------------------------------------

class P2hex_sp : public Shape {
    double LocalValue (double x, double y, double z, int i) const;
    Point LocalGradient (double x, double y, double z, int i) const;
 public:
    double operator () (const Point& z, int i) const { return LocalValue(z[0],z[1],z[2],i); }
    Point LocalGradient (const Point& z, int i) const { return LocalGradient(z[0],z[1],z[2],i); }
    int size () const { return 27; }
    P2hex_sp (const Quadrature& Q, 
	   const Quadrature& FaceQ = GetQuadrature("Qquad4")) 
	: Shape(Q,FaceQ) { fill(); }
};
const P2hex_sp P2_hex_sp(GetQuadrature("Qhex27"), GetQuadrature("Qquad4"));

// ------------------------------------------------------------
// 20-NODED HEXAHEDRON
// ------------------------------------------------------------
class P2hexSerendipity : public Shape {
 public:
    double operator () (const Point& z, int i) const;
    
    Point LocalGradient (const Point& z, int i) const;
    
    int size () const { return 20; }
    P2hexSerendipity (const Quadrature& Q, 
		      const Quadrature& FaceQ = GetQuadrature("Qquad4"));
};

const P2hexSerendipity P2_hexSerendipity(GetQuadrature("Qhex27"), 
					 GetQuadrature("Qquad4"));

// ------------------------------------------------------------
// 27-NODED HEXAHEDRON
// ------------------------------------------------------------

class P2hex : public Shape {
    inline double  q01 (const double x) const;
    inline double  q00 (const double x) const;
    inline double  q11 (const double x) const;
    inline double Dq01 (const double x) const;
    inline double Dq00 (const double x) const;
    inline double Dq11 (const double x) const;
    inline double  q1 (const double x) const;
    inline double  q2 (const double x) const;
    inline double  q3 (const double x) const;
    inline double Dq1 (const double x) const;
    inline double Dq2 (const double x) const;
    inline double Dq3 (const double x) const;
    inline double shapefct (int i, int k, const double x) const;
    inline double Dshapefct (int i, int k, const double x) const;
 public:
    double operator () (const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    int size () const { return 27; }
    P2hex (const Quadrature& Q, 
	   const Quadrature& FaceQ = GetQuadrature("Qquad4"));
};

const P2hex P2_hex(GetQuadrature("Qhex27"), GetQuadrature("Qquad4"));

// ------------------------------------------------------------
// 64-NODED HEXAHEDRON
// ------------------------------------------------------------

class P3hex : public Shape {
 public:
    inline double  c1 (const double x) const;
    inline double  c2 (const double x) const;
    inline double  c3 (const double x) const;
    inline double  c4 (const double x) const;
    inline double Dc1 (const double x) const;
    inline double Dc2 (const double x) const;
    inline double Dc3 (const double x) const;
    inline double Dc4 (const double x) const;
    inline double shapefct (int i, int k, double x) const;
    inline double Dshapefct (int i, int k, double x) const;
    double operator () (const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    int size () const { return 64; }
    P3hex (const Quadrature& Q, 
	   const Quadrature& FaceQ = GetQuadrature("Qquad4"));
};

const P3hex P3_hex(GetQuadrature("Qhex27"), GetQuadrature("Qquad4"));

// ------------------------------------------------------------
// 12-NODED HEXAHEDRON FOR CURL
// ------------------------------------------------------------
class P1curl_hex : public Shape {
public:
    double operator () (const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    Point LocalVectorValue (double x, double y, double z, int i) const;
    Point LocalCurl (double x, double y, double z, int i) const;
 public:
    Point LocalVector (const Point& z, int i) const {
	return LocalVectorValue(z[0],z[1],z[2],i);
    }
    Point LocalCurl (const Point& z, int i) const {
	return LocalCurl(z[0],z[1],z[2],i);
    }
    int size () const { return 12; }
    P1curl_hex (const Quadrature& Q, 
		const Quadrature& FaceQ = GetQuadrature("Qquad1")) 
	: Shape(Q,FaceQ) { fill(); }
};
const P1curl_hex P1_curl_hex(GetQuadrature("Qhex8"), GetQuadrature("Qquad1"));

// ------------------------------------------------------------
// 54-NODED HEXAHEDRON FOR CURL
// ------------------------------------------------------------
class P2curl_sp_hex : public Shape {
public:
    double operator () (const Point& z, int i) const;
    double sig(const Point& z, int i) const;
    Point LocalGradient (const Point& z, int i) const;
    Point LocalVectorValue (double x, double y, double z, int i) const;
    Point LocalCurl (double x, double y, double z, int i) const;
 public:
    Point LocalVector (const Point& z, int i) const {
	return LocalVectorValue(z[0],z[1],z[2],i);
    }
    Point LocalCurl (const Point& z, int i) const {
	return LocalCurl(z[0],z[1],z[2],i);
    }
    int size () const { return 54; }
    P2curl_sp_hex (const Quadrature& Q, 
		const Quadrature& FaceQ = GetQuadrature("Qquad4")) 
	: Shape(Q,FaceQ) { fill(); }
};
const P2curl_sp_hex P2_curl_sp_hex(GetQuadrature("Qhex27"), GetQuadrature("Qquad4"));

// ------------------------------------------------------------
// 4-NODED TETRAHEDRON FOR CURL
// ------------------------------------------------------------
class P1curl_tet : public Shape {
    double shape (const Point& z, int i) const { 
	switch (i) {
	case 0: return (1-z[0]-z[1]-z[2]);
	case 1: return z[0];
	case 2: return z[1];
	case 3: return z[2];
	}
    }
    Point gradient (int i) const { 
	switch (i) {
	case 0: return Point(-1.0,-1.0,-1.0);
	case 1: return Point(1.0,0.0,0.0);
	case 2: return Point(0.0,1.0,0.0);
	case 3: return Point(0.0,0.0,1.0);
	}
    }
 public:
    Point LocalVector (const Point& z, int i) const {
	switch (i) {
	    case 0: return shape(z,0)*gradient(1)-shape(z,1)*gradient(0);
	    case 1: return shape(z,1)*gradient(2)-shape(z,2)*gradient(1);
	    case 2: return shape(z,0)*gradient(2)-shape(z,2)*gradient(0);
	    case 3: return shape(z,0)*gradient(3)-shape(z,3)*gradient(0);
	    case 4: return shape(z,1)*gradient(3)-shape(z,3)*gradient(1);
	    case 5: return shape(z,2)*gradient(3)-shape(z,3)*gradient(2);
	}
    }
    Point LocalCurl (const Point& z, int i) const {
	switch (i) {
	    case 0: return 2*curl(gradient(0),gradient(1));
	    case 1: return 2*curl(gradient(1),gradient(2));
	    case 2: return 2*curl(gradient(0),gradient(2));
	    case 3: return 2*curl(gradient(0),gradient(3));
	    case 4: return 2*curl(gradient(1),gradient(3));
	    case 5: return 2*curl(gradient(2),gradient(3));
	}
    }
    int size () const { return 6; }
    P1curl_tet (const Quadrature& Q, 
		const Quadrature& FaceQ = GetQuadrature("Qtri1")) 
	: Shape(Q,FaceQ) { fill(); }
};
const P1curl_tet P1_curl_tet(GetQuadrature("Qtet4"), GetQuadrature("Qtri1"));

// ------------------------------------------------------------
//  Crouzeix-Raviart Triangle
// ------------------------------------------------------------
class CR_P1tri : public Shape {
 public:
    double operator () (const Point& z, int i) const { 
	switch (i) {
	case 0: return 1.0-2*z[1];
	case 1: return 2*(z[0]+z[1]) -1.0;
	case 2: return 1.0-2*z[0];
	}
    }
    Point LocalGradient (const Point& z, int i) const { 
	switch (i) {
	case 0: return Point(0.0,-2.0);
	case 1: return Point(2.0,2.0);
	case 2: return Point(-2.0,0.0);
	}
    }
    int size () const { return 3; }
    CR_P1tri (const Quadrature& Q, 
	    const Quadrature& FaceQ = GetQuadrature("Qint1")) 
	: Shape(Q,FaceQ) { fill(); }
};
const CR_P1tri CR_P1_tri(GetQuadrature("Qtri3"), GetQuadrature("Qint1"));


/// ------------------------------------------------------------
//  Raviart-Thomas Triangle
// ------------------------------------------------------------
class RT0tri : public Shape {
 public:
    RT0tri (const Quadrature& Q, 
	      const Quadrature& FaceQ = GetQuadrature("Qint1")) 
	: Shape(Q,FaceQ) { 
	// fill(*this);
    }
    int size() const { return 3; }
    Point LocalVector(const Point& z, int i) const;
    double LocalDiv(const Point& z, int i) const;
};
const RT0tri RT0_tri(GetQuadrature("Qtri3"), GetQuadrature("Qint1"));

/// ------------------------------------------------------------
//  Raviart-Thomas Quadrilateral
// ------------------------------------------------------------
class RT0quad : public Shape {
 public:
    RT0quad (const Quadrature& Q, 
	     const Quadrature& FaceQ = GetQuadrature("Qint1")) 
	: Shape(Q,FaceQ) { 
	// fill(*this);
    }
    int size() const { return 4; }
    Point LocalVector(const Point& z, int i) const;
    double LocalDiv(const Point& z, int i) const;
};
const RT0quad RT0_quad(GetQuadrature("Qquad4"), GetQuadrature("Qint1"));
/// ------------------------------------------------------------
//  Raviart-Thomas Tetrahedron
// ------------------------------------------------------------
class RT0tet : public Shape {
 public:
    RT0tet (const Quadrature& Q, 
	    const Quadrature& FaceQ = GetQuadrature("Qint1")) 
	: Shape(Q,FaceQ) { 
	// fill(*this);
    }
    int size() const { return 4; }
    Point LocalVector(const Point& z, int i) const;
    double LocalDiv(const Point& z, int i) const;
};
const RT0tet RT0_tet(GetQuadrature("Qtet4"), GetQuadrature("Qtri1"));
/// ------------------------------------------------------------
//  Raviart-Thomas Hexahedron
// ------------------------------------------------------------
class RT0hex : public Shape {
 public:
    RT0hex (const Quadrature& Q, 
	    const Quadrature& FaceQ = GetQuadrature("Qint1")) 
	: Shape(Q,FaceQ) { 
	// fill(*this);
    }
    int size() const { return 6; }
    Point LocalVector(const Point& z, int i) const;
    double LocalDiv(const Point& z, int i) const;
};
const RT0hex RT0_hex(GetQuadrature("Qhex8"), GetQuadrature("Qquad1"));


/// ------------------------------------------------------------
//  P0 (the same on all elements)
// ------------------------------------------------------------
class P0 : public Shape {
 public:
    P0 (const Quadrature& Q, 
	const Quadrature& FaceQ = GetQuadrature("Qint1")) 
	: Shape(Q,FaceQ) { 
    }
    int size() const { return 1; }
    double operator() (const Point& z, int i) const { return 1.0; }
    Point LocalGradient (const Point& z, int i) const { 
	return zero;
    }
};


/* Note: in the following J is the Jacobian of the transformation 
 *       T: \hat K \rightarrow K from the reference element \hat K
 *       to K
 * in F[][] is the Jacobian, i.e. F = J
 * in T[][] is the transposed inverse of the jacobian, i.e. T=J^{-T}
 * 
 * T*z                 is  J^{-T} * z
 * z*T                 is  J^{-1} * z
 * operator() (P)      is  J^T*P
 * ApplyJ              is  J*P
 */
// ------------------------------------------------------------
//    Transformation
// ------------------------------------------------------------
class Transformation {
	double F[3][3];
	double T[3][3];
    double det;
public:
    Transformation () {}
    Transformation (const vector<Point>& z);
    
    double Det () const { return det; }
    const double* operator [] (int i) const { return T[i]; }
    const double* operator ()  (int i) const { return F[i]; }
    Point operator () (const Point& P) const;
    Point ApplyJ(const Point& P) const;
    friend Point operator * (const Transformation& T, const Point& z);
    friend Point operator * (const Point& z, const Transformation& T);
    friend ostream& operator << (ostream& s, const Transformation& T);
};

#endif




