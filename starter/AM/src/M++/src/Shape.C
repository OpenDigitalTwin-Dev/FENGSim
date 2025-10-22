// file: Shape.C
// author: Christian Wieners, Wolfgang Mueller
// $Header: /public/M++/src/Shape.C,v 1.8 2009-04-08 08:46:14 wieners Exp $

#include "Shape.h"

Point Shape::Eval (const Point& z, const vector<Point>& x) const {
    Point y(0.0,0.0,0.0);
    for (int i=0; i<x.size(); ++i) y += (*this)(z,i) * x[i];
    return y;
}

void Shape::EvalGradient (vector<Point>& y, 
						  const Point& z, const vector<Point>& x) const {
    for (int k=0; k<y.size(); ++k) y[k] = Point(0.0,0.0,0.0);
    for (int i=0; i<x.size(); ++i) {
        Point G = LocalGradient(z,i);
        for (int k=0; k<y.size(); ++k) y[k] += x[i][k] * G;
    }
}

void Shape::fill() {
    for (int q=0; q<Q.size(); ++q) 
        for (int i=0; i<size(); ++i) {
            value[q][i] = (*this)(Q.QPoint(q),i); 
            localgradient[q][i] = LocalGradient(Q.QPoint(q),i); 
        }
}

void Shape::fill(int n) {
    for (int q=0; q<Q.size(); ++q) 
        for (int i=0; i<n; ++i) {
            value[q][i] = (*this)(Q.QPoint(q),i); 
            localgradient[q][i] = LocalGradient(Q.QPoint(q),i); 
        }
}

Shape::Shape (const Quadrature& _Q, const Quadrature& _FaceQ) : Q(_Q), FaceQ(_FaceQ)
{}

ostream& operator << (ostream& s, const Shape& S) {
    for (int q=0; q<S.Q.size(); ++q) 
        for (int i=0; i<S.size(); ++i) {
            mout << q << " i " << i << " u " << S(q,i) 
                 << " Du " << S.LocalGradient(q,i) << endl; 
        }
    return s; 
}


// ------------------------------------------------------------
// 8-NODED QUADRILATERAL
// ------------------------------------------------------------

double P2quadSerendipity::operator () (const Point& z, int i) const { 
    switch (i) {
    case  0: return ((1-z[0])*(1-z[1])*(1-2*z[0]-2*z[1]));
    case  1: return (-z[0]*(1-z[1])*(1-2*z[0]+2*z[1]));
    case  2: return (-z[0]*z[1]*(3-2*z[0]-2*z[1]));
    case  3: return (-z[1]*(1-z[0])*(1+2*z[0]-2*z[1]));
    case  4: return (4*z[0]*(1-z[0])*(1-z[1]));
    case  5: return (4*z[0]*z[1]*(1-z[1]));
    case  6: return (4*z[0]*z[1]*(1-z[0]));
    case  7: return (4*z[1]*(1-z[0])*(1-z[1]));
    }
}
    
Point P2quadSerendipity::LocalGradient (const Point& z, int i) const { 
    switch (i) {
    case  0: return Point((1 - z[1]) * (-3 + 4*z[0] + 2*z[1]),
                          (1 - z[0]) * (-3 + 4*z[1] + 2*z[0]));
    case  1: return Point((1 - z[1]) * (-1 + 4*z[0] - 2*z[1]),
                          -z[0]*(1+2*z[0]-4*z[1]));
    case  2: return Point(- z[1] * (3 - 4*z[0] - 2*z[1]),
                          - z[0] * (3 - 4*z[1] - 2*z[0]));
    case  3: return Point(- z[1] * (1 - 4*z[0] + 2*z[1]),
                          (z[0]-1)*(1+2*z[0]-4*z[1]));
    case  4: return Point(4*(1-2*z[0])*(1-z[1]),-4*z[0]*(1-z[0]));
    case  5: return Point(4*z[1]*(1-z[1]),4*z[0]*(1-2*z[1]));
    case  6: return Point(4*z[1]*(1-2*z[0]),4*z[0]*(1-z[0]));
    case  7: return Point(-4*z[1]*(1-z[1]),4*(1-z[0])*(1-2*z[1]));
    }
}

P2quadSerendipity::P2quadSerendipity (const Quadrature& Q, const Quadrature& FaceQ)
	: Shape(Q,FaceQ) {
    fill();
}


// ------------------------------------------------------------
// 16-NODED QUADRILATERAL
// ------------------------------------------------------------

inline double P3quad::c0 (const double x) const { return -1/2  *(x-1)*(3*x-1)* (3*x-2); }
inline double P3quad::c1 (const double x) const { return  9/2*x*(x-1)        * (3*x-2); }
inline double P3quad::c2 (const double x) const { return -9/2*x*(x-1)*(3*x-1);          }
inline double P3quad::c3 (const double x) const { return  1/2*x      *(3*x-1)* (3*x-2); }
inline double P3quad::Dc0 (const double x) const { return -27/2*x*x + 18*x - 11/2*x; }
inline double P3quad::Dc1 (const double x) const { return  81/2*x*x - 45*x + 9;      }
inline double P3quad::Dc2 (const double x) const { return -81/2*x*x + 36*x - 9/2;    }
inline double P3quad::Dc3 (const double x) const { return  27/2*x*x -  9*x + 1;      }

double P3quad::operator () (const Point& z, int i) const { 
    switch (i) {
    case  0: return c0(z[0]) * c0(z[1]);
    case  1: return c1(z[0]) * c0(z[1]);
    case  2: return c2(z[0]) * c0(z[1]);
    case  3: return c3(z[0]) * c0(z[1]);
    case  4: return c0(z[0]) * c1(z[1]);
    case  5: return c1(z[0]) * c1(z[1]);
    case  6: return c2(z[0]) * c1(z[1]);
    case  7: return c3(z[0]) * c1(z[1]);
    case  8: return c0(z[0]) * c2(z[1]);
    case  9: return c1(z[0]) * c2(z[1]);
    case 10: return c2(z[0]) * c2(z[1]);
    case 11: return c3(z[0]) * c2(z[1]);
    case 12: return c0(z[0]) * c3(z[1]);
    case 13: return c1(z[0]) * c3(z[1]);
    case 14: return c2(z[0]) * c3(z[1]);
    case 15: return c3(z[0]) * c3(z[1]);
    }
}

Point P3quad::LocalGradient (const Point& z, int i) const { 
    switch (i) {
    case  0: return Point(Dc0(z[0]) * c0(z[1]),c0(z[0]) * Dc0(z[1]));
    case  1: return Point(Dc1(z[0]) * c0(z[1]),c1(z[0]) * Dc0(z[1]));
    case  2: return Point(Dc2(z[0]) * c0(z[1]),c2(z[0]) * Dc0(z[1]));
    case  3: return Point(Dc3(z[0]) * c0(z[1]),c3(z[0]) * Dc0(z[1]));
    case  4: return Point(Dc0(z[0]) * c1(z[1]),c0(z[0]) * Dc1(z[1]));
    case  5: return Point(Dc1(z[0]) * c1(z[1]),c1(z[0]) * Dc1(z[1]));
    case  6: return Point(Dc2(z[0]) * c1(z[1]),c2(z[0]) * Dc1(z[1]));
    case  7: return Point(Dc3(z[0]) * c1(z[1]),c3(z[0]) * Dc1(z[1]));
    case  8: return Point(Dc0(z[0]) * c2(z[1]),c0(z[0]) * Dc2(z[1]));
    case  9: return Point(Dc1(z[0]) * c2(z[1]),c1(z[0]) * Dc2(z[1]));
    case 10: return Point(Dc2(z[0]) * c2(z[1]),c2(z[0]) * Dc2(z[1]));
    case 11: return Point(Dc3(z[0]) * c2(z[1]),c3(z[0]) * Dc2(z[1]));
    case 12: return Point(Dc0(z[0]) * c3(z[1]),c0(z[0]) * Dc3(z[1]));
    case 13: return Point(Dc1(z[0]) * c3(z[1]),c1(z[0]) * Dc3(z[1]));
    case 14: return Point(Dc2(z[0]) * c3(z[1]),c2(z[0]) * Dc3(z[1]));
    case 15: return Point(Dc3(z[0]) * c3(z[1]),c3(z[0]) * Dc3(z[1]));
    }
}

P3quad::P3quad (const Quadrature& Q, const Quadrature& FaceQ) 
	: Shape(Q,FaceQ)
{ fill(); }



// ------------------------------------------------------------
// 20-NODED HEXAHEDRON
// ------------------------------------------------------------

double P2hexSerendipity::operator () (const Point& z, int i) const { 
    switch (i) {
    case 0: return((1.0-z[0])*(1.0-z[1])*(1.0-z[2])
                   *(1.0-2.0*z[0]-2.0*z[1]-2.0*z[2]));
    case 1: return((z[0])*(1.0-z[1])*(1.0-z[2])
                   *(2.0*(z[0])+2.0*(1.0-z[1])+2.0*(1.0-z[2])-5.0));
    case 2: return((z[0])*(z[1])*(1.0-z[2])
                   *(2.0*(z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0));
    case 3: return((1.0-z[0])*(z[1])*(1.0-z[2])
                   *(2.0*(1.0-z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0));
    case 4: return((1.0-z[0])*(1.0-z[1])*(z[2])
                   *(2.0*(1.0-z[0])+2.0*(1.0-z[1])+2.0*(z[2])-5.0));
    case 5: return((z[0])*(1.0-z[1])*(z[2])
                   *(2.0*(z[0])+2.0*(1.0-z[1])+2.0*(z[2])-5.0));
    case 6: return((z[0])*(z[1])*(z[2])
                   *(2.0*(z[0])+2.0*(z[1])+2.0*(z[2])-5.0));
    case 7: return((1.0-z[0])*(z[1])*(z[2])
                   *(2.0*(1.0-z[0])+2.0*(z[1])+2.0*(z[2])-5.0));
    case 8: return( 4.0*(1.0-z[0])*(1.0-z[1])*(1.0-z[2])*z[0]);
    case 9: return( 4.0*(z[0])    *(1.0-z[1])*(1.0-z[2])*z[1]);
    case 10: return(4.0*(1.0-z[0])*(z[1])    *(1.0-z[2])*z[0]);
    case 11: return(4.0*(1.0-z[0])*(1.0-z[1])*(1.0-z[2])*z[1]);
    case 12: return(4.0*(1.0-z[0])*(1.0-z[1])*(1.0-z[2])*z[2]);
    case 13: return(4.0*(z[0])    *(1.0-z[1])*(1.0-z[2])*z[2]);
    case 14: return(4.0*(z[0])    *(z[1])    *(1.0-z[2])*z[2]);
    case 15: return(4.0*(1.0-z[0])*(z[1])    *(1.0-z[2])*z[2]);
    case 16: return(4.0*(1.0-z[0])*(1.0-z[1])*(z[2])    *z[0]);
    case 17: return(4.0*(z[0])    *(1.0-z[1])*(z[2])    *z[1]);
    case 18: return(4.0*(1.0-z[0])*(z[1])    *(z[2])    *z[0]);
    case 19: return(4.0*(1.0-z[0])*(1.0-z[1])*(z[2])    *z[1]);
    }
}

Point P2hexSerendipity::LocalGradient (const Point& z, int i) const { 
    switch (i) {
    case 0:  return Point(-(1.0-z[1])*(1.0-z[2])
                          *(1.0-2.0*z[0]-2.0*z[1]-2.0*z[2])
                          -2.0*(1.0-z[0])*(1.0-z[1])*(1.0-z[2]),
                          -(1.0-z[0])*(1.0-z[2])
                          *(1.0-2.0*z[0]-2.0*z[1]-2.0*z[2])
                          -2.0*(1.0-z[0])*(1.0-z[1])*(1.0-z[2]),
                          -(1.0-z[0])*(1.0-z[1])
                          *(1.0-2.0*z[0]-2.0*z[1]-2.0*z[2])
                          -2.0*(1.0-z[0])*(1.0-z[1])*(1.0-z[2]));
    case 1:  return Point((1.0-z[1])*(1.0-z[2])
                          *(2.0*(z[0])+2.0*(1.0-z[1])+2.0*(1.0-z[2])-5.0)
                          +2.0*(z[0])*(1.0-z[1])*(1.0-z[2]),
                          -z[0]*(1.0-z[2])
                          *(2.0*(z[0])+2.0*(1.0-z[1])+2.0*(1.0-z[2])-5.0)
                          -2.0*(z[0])*(1.0-z[1])*(1.0-z[2]),
                          -z[0]*(1.0-z[1])
                          *(2.0*(z[0])+2.0*(1.0-z[1])+2.0*(1.0-z[2])-5.0)
                          -2.0*(z[0])*(1.0-z[1])*(1.0-z[2]));
    case 2:  return Point(z[1]*(1.0-z[2])
                          *(2.0*(z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0)
                          +2.0*z[0]*z[1]*(1.0-z[2]),
                          z[0]*(1.0-z[2])
                          *(2.0*(z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0)
                          +2.0*z[0]*z[1]*(1.0-z[2]),
                          -z[0]*z[1]
                          *(2.0*(z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0)
                          -2.0*z[0]*z[1]*(1.0-z[2]));
    case 3:  return Point(-z[1]*(1.0-z[2])
                          *(2.0*(1.0-z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0)
                          -2.0*(1.0-z[0])*z[1]*(1.0-z[2]),
                          (1.0-z[0])*(1.0-z[2])
                          *(2.0*(1.0-z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0)
                          +2.0*(1.0-z[0])*z[1]*(1.0-z[2]),
                          -(1.0-z[0])*z[1]
                          *(2.0*(1.0-z[0])+2.0*(z[1])+2.0*(1.0-z[2])-5.0)
                          -2.0*(1.0-z[0])*z[1]*(1.0-z[2]));
    case 4:  return Point(-(1.0-z[1])*z[2]
                          *(2.0*(1.0-z[0])+2.0*(1.0-z[1])+2.0*z[2]-5.0)
                          -2.0*(1.0-z[0])*(1.0-z[1])*z[2],
                          -(1.0-z[0])*z[2]
                          *(2.0*(1.0-z[0])+2.0*(1.0-z[1])+2.0*z[2]-5.0)
                          -2.0*(1.0-z[0])*(1.0-z[1])*z[2],
                          (1.0-z[0])*(1.0-z[1])
                          *(2.0*(1.0-z[0])+2.0*(1.0-z[1])+2.0*z[2]-5.0)
                          +2.0*(1.0-z[0])*(1.0-z[1])*z[2]);
    case 5:  return Point((1.0-z[1])*z[2]
                          *(2.0*z[0]+2.0*(1.0-z[1])+2.0*z[2]-5.0)
                          +2.0*z[0]*(1.0-z[1])*z[2],
                          -z[0]*z[2]
                          *(2.0*z[0]+2.0*(1.0-z[1])+2.0*z[2]-5.0)
                          -2.0*z[0]*(1.0-z[1])*z[2],
                          z[0]*(1.0-z[1])
                          *(2.0*z[0]+2.0*(1.0-z[1])+2.0*z[2]-5.0)
                          +2.0*z[0]*(1.0-z[1])*z[2]);
    case 6:  return Point(z[1]*z[2]
                          *(2.0*z[0]+2.0*z[1]+2.0*z[2]-5.0)
                          +2.0*z[0]*z[1]*z[2],
                          z[0]*z[2]
                          *(2.0*z[0]+2.0*z[1]+2.0*z[2]-5.0)
                          +2.0*z[0]*z[1]*z[2],
                          z[0]*z[1]
                          *(2.0*(z[0])+2.0*(z[1])+2.0*z[2]-5.0)
                          +2.0*z[0]*z[1]*z[2]);
    case 7:  return Point(-z[1]*z[2]
                          *(2.0*(1.0-z[0])+2.0*z[1]+2.0*z[2]-5.0) 
                          -2.0*(1.0-z[0])*z[1]*z[2],
                          (1.0-z[0])*z[2]
                          *(2.0*(1.0-z[0])+2.0*z[1]+2.0*z[2]-5.0)
                          +2.0*(1.0-z[0])*z[1]*z[2],
                          (1.0-z[0])*z[1]
                          *(2.0*(1.0-z[0])+2.0*z[1]+2.0*z[2]-5.0)
                          +2.0*(1.0-z[0])*z[1]*z[2]);
    case 8:  return Point(4.0*(1.0-2.0*z[0])*(1.0-z[1])*(1.0-z[2]),
                          -4.0*(1.0-z[0])*(1.0-z[2])*z[0],
                          -4.0*(1.0-z[0])*(1.0-z[1])*z[0]);
    case 9:  return Point(4.0*(1.0-z[1])*(1.0-z[2])*z[1],
                          4.0*(z[0])*(1.0-2.0*z[1])*(1.0-z[2]),
                          -4.0*(z[0])*(1.0-z[1])*z[1]);
    case 10: return Point(4.0*(1.0-2.0*z[0])*z[1]*(1.0-z[2]),
                          4.0*(1.0-z[0])*(1.0-z[2])*z[0],
                          -4.0*(1.0-z[0])*z[1]*z[0]);
    case 11: return Point(-4.0*(1.0-z[1])*(1.0-z[2])*z[1],
                          4.0*(1.0-z[0])*(1.0-2.0*z[1])*(1.0-z[2]),
                          -4.0*(1.0-z[0])*(1.0-z[1])*z[1]);
    case 12: return Point(-4.0*(1.0-z[1])*(1.0-z[2])*z[2],
                          -4.0*(1.0-z[0])*(1.0-z[2])*z[2],
                          4.0*(1.0-z[0])*(1.0-z[1])*(1.0-2.0*z[2]));
    case 13: return Point(4.0*(1.0-z[1])*(1.0-z[2])*z[2],
                          -4.0*(z[0])*(1.0-z[2])*z[2],
                          4.0*z[0]*(1.0-z[1])*(1.0-2.0*z[2]));
    case 14: return Point(4.0*z[1]*(1.0-z[2])*z[2],
                          4.0*z[0]*(1.0-z[2])*z[2],
                          4.0*z[0]*z[1]*(1.0-2.0*z[2]));
    case 15: return Point(-4.0*z[1]*(1.0-z[2])*z[2],
                          4.0*(1.0-z[0])*(1.0-z[2])*z[2],
                          4.0*(1.0-z[0])*z[1]*(1.0-2.0*z[2]));
    case 16: return Point(4.0*(1.0-2.0*z[0])*(1.0-z[1])*z[2],
                          -4.0*(1.0-z[0])*z[2]*z[0],
                          4.0*(1.0-z[0])*(1.0-z[1])*z[0]);
    case 17: return Point(4.0*(1.0-z[1])*z[2]*z[1],
                          4.0*z[0]*(1.0-2.0*z[1])*z[2],
                          4.0*z[0]*(1.0-z[1])*z[1]);
    case 18: return Point(4.0*(1.0-2.0*z[0])*z[1]*z[2],
                          4.0*(1.0-z[0])*z[2]*z[0],
                          4.0*(1.0-z[0])*z[1]*z[0]);
    case 19: return Point(-4.0*(1.0-z[1])*z[2]*z[1],
                          4.0*(1.0-z[0])*(1.0-2.0*z[1])*z[2],
                          4.0*(1.0-z[0])*(1.0-z[1])*z[1]);
    }
}

P2hexSerendipity::P2hexSerendipity (const Quadrature& Q, const Quadrature& FaceQ )
	: Shape(Q,FaceQ)
{ fill(); }

static const double c_27 [27][3] = { {0.0, 0.0, 0.0},
			      {1.0, 0.0, 0.0},
			      {1.0, 1.0, 0.0},
			      {0.0, 1.0, 0.0},
			      {0.0, 0.0, 1.0},
			      {1.0, 0.0, 1.0},
			      {1.0, 1.0, 1.0},
			      {0.0, 1.0, 1.0},
			      {0.5, 0.0, 0.0},
			      {1.0, 0.5, 0.0},
			      {0.5, 1.0, 0.0},
			      {0.0, 0.5, 0.0},
			      {0.0, 0.0, 0.5},
			      {1.0, 0.0, 0.5},
			      {1.0, 1.0, 0.5},
			      {0.0, 1.0, 0.5},
			      {0.5, 0.0, 1.0},
			      {1.0, 0.5, 1.0},
			      {0.5, 1.0, 1.0},
			      {0.0, 0.5, 1.0},
			      {0.5, 0.5, 0.0},
			      {0.5, 0.0, 0.5},
			      {1.0, 0.5, 0.5},
			      {0.5, 1.0, 0.5},
			      {0.0, 0.5, 0.5},
			      {0.5, 0.5, 1.0},
			      {0.5, 0.5, 0.5} };

inline double P2hex::q01 (const double x) const { return      x * (2*x-1); }
inline double P2hex::q00 (const double x) const { return   (x-1)* (2*x-1); }
inline double P2hex::q11 (const double x) const { return  4 * x * (1-x);   }
inline double P2hex::Dq01 (const double x) const { return  4*x-1; }
inline double P2hex::Dq00 (const double x) const { return  4*x-3; }
inline double P2hex::Dq11 (const double x) const { return -8*x+4; }
inline double P2hex::q1 (const double x) const { return      x * (2*x-1); }
inline double P2hex::q2 (const double x) const { return   (x-1)* (2*x-1); }
inline double P2hex::q3 (const double x) const { return  4 * x * (1-x);   }
inline double P2hex::Dq1 (const double x) const { return  4*x-1; }
inline double P2hex::Dq2 (const double x) const { return  4*x-3; }
inline double P2hex::Dq3 (const double x) const { return -8*x+4; }
inline double P2hex::shapefct (int i, int k, const double x) const {
    double s = c_27[i][k];
    if (s == 1.0) return  q1(x);
    if (s == 0.0) return  q2(x);
    if (s == 0.5) return  q3(x);
}

inline double P2hex::Dshapefct (int i, int k, const double x) const {
    double s = c_27[i][k];
    if (s == 1.0) return Dq1(x);
    if (s == 0.0) return Dq2(x);
    if (s == 0.5) return Dq3(x);
}


double P2hex::operator () (const Point& z, int i) const { 
    switch (i) {
    case 0: return q00(z[0]) * q00(z[1]) * q00(z[2]);
    case 1: return q01(z[0]) * q00(z[1]) * q00(z[2]);
    case 2: return q01(z[0]) * q01(z[1]) * q00(z[2]);
    case 3: return q00(z[0]) * q01(z[1]) * q00(z[2]);
    case 4: return q00(z[0]) * q00(z[1]) * q01(z[2]);
    case 5: return q01(z[0]) * q00(z[1]) * q01(z[2]);
    case 6: return q01(z[0]) * q01(z[1]) * q01(z[2]);
    case 7: return q00(z[0]) * q01(z[1]) * q01(z[2]);
    case 8: return q11(z[0]) * q00(z[1]) * q00(z[2]);
    case 9: return q01(z[0]) * q11(z[1]) * q00(z[2]);
    case 10: return q11(z[0]) * q01(z[1]) * q00(z[2]);
    case 11: return q00(z[0]) * q11(z[1]) * q00(z[2]);
    case 12: return q00(z[0]) * q00(z[1]) * q11(z[2]);
    case 13: return q01(z[0]) * q00(z[1]) * q11(z[2]);
    case 14: return q01(z[0]) * q01(z[1]) * q11(z[2]);
    case 15: return q00(z[0]) * q01(z[1]) * q11(z[2]);
    case 16: return q11(z[0]) * q00(z[1]) * q01(z[2]);
    case 17: return q01(z[0]) * q11(z[1]) * q01(z[2]);
    case 18: return q11(z[0]) * q01(z[1]) * q01(z[2]);
    case 19: return q00(z[0]) * q11(z[1]) * q01(z[2]);
    case 20: return q11(z[0]) * q11(z[1]) * q00(z[2]);
    case 21: return q11(z[0]) * q00(z[1]) * q11(z[2]);
    case 22: return q01(z[0]) * q11(z[1]) * q11(z[2]);
    case 23: return q11(z[0]) * q01(z[1]) * q11(z[2]);
    case 24: return q00(z[0]) * q11(z[1]) * q11(z[2]);
    case 25: return q11(z[0]) * q11(z[1]) * q01(z[2]);
    case 26: return q11(z[0]) * q11(z[1]) * q11(z[2]);
    }
}

Point P2hex::LocalGradient (const Point& z, int i) const { 
    return Point(Dshapefct(i,0,z[0])* shapefct(i,1,z[1])* shapefct(i,2,z[2]),
                 shapefct(i,0,z[0])*Dshapefct(i,1,z[1])* shapefct(i,2,z[2]),
                 shapefct(i,0,z[0])* shapefct(i,1,z[1])*Dshapefct(i,2,z[2]));
}

P2hex::P2hex (const Quadrature& Q, const Quadrature& FaceQ)
	: Shape(Q,FaceQ) { fill(); }

// ------------------------------------------------------------
// 8-NODED HEXAHEDRON
// ------------------------------------------------------------
    double P1hex::operator () (const Point& z, int i) const { 
	switch (i) {
	case 0: return (1-z[0])*(1-z[1])*(1-z[2]);
	case 1: return z[0]*(1-z[1])*(1-z[2]);
	case 2: return z[0]*z[1]*(1-z[2]);
	case 3: return (1-z[0])*z[1]*(1-z[2]);
	case 4: return (1-z[0])*(1-z[1])*z[2];
	case 5: return z[0]*(1-z[1])*z[2];
	case 6: return z[0]*z[1]*z[2];
	case 7: return (1-z[0])*z[1]*z[2];
	}
    }
    Point P1hex::LocalGradient (const Point& z, int i) const { 
	switch (i) {
	    case 0: return Point(-(1-z[1])*(1-z[2]),
				 -(1-z[0])*(1-z[2]),
				 -(1-z[0])*(1-z[1]));
	    case 1: return Point((1-z[1])*(1-z[2]),
				 -z[0]*(1-z[2]),
				 -z[0]*(1-z[1]));
	    case 2: return Point(z[1]*(1-z[2]),
				 z[0]*(1-z[2]),
				 -z[0]*z[1]);
	    case 3: return Point(-z[1]*(1-z[2]),
				 (1-z[0])*(1-z[2]),
				 -(1-z[0])*z[1]);
	    case 4: return Point(-(1-z[1])*z[2],
				 -(1-z[0])*z[2],
				 (1-z[0])*(1-z[1]));
	    case 5: return Point((1-z[1])*z[2],
				 -z[0]*z[2],
				 z[0]*(1-z[1]));
	    case 6: return Point(z[1]*z[2],
				 z[0]*z[2],
				 z[0]*z[1]);
	    case 7: return Point(-z[1]*z[2],
				 (1-z[0])*z[2],
				 (1-z[0])*z[1]);
	}
    }

// ------------------------------------------------------------
// 27-NODED HEXAHEDRON (CONFORMING WITH P2curl_sp_hex)
// ------------------------------------------------------------
    double P2hex_sp::LocalValue(double x, double y, double z, int i) const { 
	switch (i) {
        // the lowest order vertix-based
	case 0: return (1-x)*(1-y)*(1-z);
	case 1: return x*(1-y)*(1-z);
	case 2: return x*y*(1-z);
	case 3: return (1-x)*y*(1-z);
	case 4: return (1-x)*(1-y)*z;
	case 5: return x*(1-y)*z;
	case 6: return x*y*z;
	case 7: return (1-x)*y*z;
        // 2-nd order edge-based
        case 8: return 2*(x-1)*x*(y-1)*(z-1);
        case 9: return -2*x*(y-1)*y*(z-1);
        case 10: return -2*(x-1)*x*y*(z-1);
        case 11: return 2*(x-1)*(y-1)*y*(z-1);
        case 12: return 2*(x-1)*(y-1)*(z-1)*z;
        case 13: return -2*x*(y-1)*(z-1)*z;
        case 14: return 2*x*y*(z-1)*z;
        case 15: return -2*(x-1)*y*(z-1)*z;
        case 16: return -2*(x-1)*x*(y-1)*z;
        case 17: return 2*x*(y-1)*y*z;
        case 18: return 2*(x-1)*x*y*z;
        case 19: return -2*(x-1)*(y-1)*y*z;
        // 2-nd order face-based
        case 20: return -4*(x-1)*x*(y-1)*y*(z-1);
        case 21: return -4*(x-1)*x*(y-1)*(z-1)*z;
        case 22: return 4*x*(y-1)*y*(z-1)*z;
        case 23: return 4*(x-1)*x*y*(z-1)*z;
        case 24: return -4*(x-1)*(y-1)*y*(z-1)*z;
        case 25: return 4*(x-1)*x*(y-1)*y*z;
        // 2-nd order cell-based
        case 26: return 8*(x-1)*x*(y-1)*y*(z-1)*z;
	}
    }
    Point P2hex_sp::LocalGradient(double x, double y, double z, int i) const { 
	switch (i) {
            // the lowest order vertix-based
	    case 0: return Point(-(1-y)*(1-z),-(1-x)*(1-z),-(1-x)*(1-y));
	    case 1: return Point((1-y)*(1-z),-x*(1-z),-x*(1-y));
	    case 2: return Point(y*(1-z),x*(1-z),-x*y);
	    case 3: return Point(-y*(1-z),(1-x)*(1-z),-(1-x)*y);
	    case 4: return Point(-(1-y)*z,-(1-x)*z,(1-x)*(1-y));
	    case 5: return Point((1-y)*z,-x*z,x*(1-y));
	    case 6: return Point(y*z,x*z,x*y);
	    case 7: return Point(-y*z,(1-x)*z,(1-x)*y);
            // 2-nd order edge-based
            case 8: return Point(2*(2*x-1)*(y-1)*(z-1),2*(x-1)*x*(z-1),2*(x-1)*x*(y-1));
            case 9: return Point(-2*(y-1)*y*(z-1),-2*x*(2*y-1)*(z-1),-2*x*(y-1)*y);
            case 10: return Point(-2*(2*x-1)*y*(z-1),-2*(x-1)*x*(z-1),-2*(x-1)*x*y);
            case 11: return Point(2*(y-1)*y*(z-1),2*(x-1)*(2*y-1)*(z-1),2*(x-1)*(y-1)*y);
            case 12: return Point(2*(y-1)*(z-1)*z,2*(x-1)*(z-1)*z,2*(x-1)*(y-1)*(2*z-1));
            case 13: return Point(-2*(y-1)*(z-1)*z,-2*x*(z-1)*z,-2*x*(y-1)*(2*z-1));
            case 14: return Point(2*y*(z-1)*z,2*x*(z-1)*z,2*x*y*(2*z-1));
            case 15: return Point(-2*y*(z-1)*z,-2*(x-1)*(z-1)*z,-2*(x-1)*y*(2*z-1));
            case 16: return Point(-2*(2*x-1)*(y-1)*z,-2*(x-1)*x*z,-2*(x-1)*x*(y-1));
            case 17: return Point(2*(y-1)*y*z,2*x*(2*y-1)*z,2*x*(y-1)*y);
            case 18: return Point(2*(2*x-1)*y*z,2*(x-1)*x*z,2*(x-1)*x*y);
            case 19: return Point(-2*(y-1)*y*z,-2*(x-1)*(2*y-1)*z,-2*(x-1)*(y-1)*y);
            // 2-nd order face-based
            case 20: return Point(-4*(2*x-1)*(y-1)*y*(z-1),-4*(x-1)*x*(2*y-1)*(z-1),-4*(x-1)*x*(y-1)*y);
            case 21: return Point(-4*(2*x-1)*(y-1)*(z-1)*z,-4*(x-1)*x*(z-1)*z,-4*(x-1)*x*(y-1)*(2*z-1));
            case 22: return Point(4*(y-1)*y*(z-1)*z,4*x*(2*y-1)*(z-1)*z,4*x*(y-1)*y*(2*z-1));
            case 23: return Point(4*(2*x-1)*y*(z-1)*z,4*(x-1)*x*(z-1)*z,4*(x-1)*x*y*(2*z-1));
            case 24: return Point(-4*(y-1)*y*(z-1)*z,-4*(x-1)*(2*y-1)*(z-1)*z,-4*(x-1)*(y-1)*y*(2*z-1));
            case 25: return Point(4*(2*x-1)*(y-1)*y*z,4*(x-1)*x*(2*y-1)*z,4*(x-1)*x*(y-1)*y);
            // 2-nd order cell-based
            case 26: return Point(8*(2*x-1)*(y-1)*y*(z-1)*z,8*(x-1)*x*(2*y-1)*(z-1)*z,8*(x-1)*x*(y-1)*y*(2*z-1));
	}
    }

// ------------------------------------------------------------
// 12-NODED HEXAHEDRON FOR CURL
// ------------------------------------------------------------
    double P1curl_hex::operator () (const Point& z, int i) const { 
	switch (i) {
	case 0: return (1-z[0])*(1-z[1])*(1-z[2]);
	case 1: return z[0]*(1-z[1])*(1-z[2]);
	case 2: return z[0]*z[1]*(1-z[2]);
	case 3: return (1-z[0])*z[1]*(1-z[2]);
	case 4: return (1-z[0])*(1-z[1])*z[2];
	case 5: return z[0]*(1-z[1])*z[2];
	case 6: return z[0]*z[1]*z[2];
	case 7: return (1-z[0])*z[1]*z[2];
	}
    }
    Point P1curl_hex::LocalGradient (const Point& z, int i) const { 
	switch (i) {
	    case 0: return Point(-(1-z[1])*(1-z[2]),
				 -(1-z[0])*(1-z[2]),
				 -(1-z[0])*(1-z[1]));
	    case 1: return Point((1-z[1])*(1-z[2]),
				 -z[0]*(1-z[2]),
				 -z[0]*(1-z[1]));
	    case 2: return Point(z[1]*(1-z[2]),
				 z[0]*(1-z[2]),
				 -z[0]*z[1]);
	    case 3: return Point(-z[1]*(1-z[2]),
				 (1-z[0])*(1-z[2]),
				 -(1-z[0])*z[1]);
	    case 4: return Point(-(1-z[1])*z[2],
				 -(1-z[0])*z[2],
				 (1-z[0])*(1-z[1]));
	    case 5: return Point((1-z[1])*z[2],
				 -z[0]*z[2],
				 z[0]*(1-z[1]));
	    case 6: return Point(z[1]*z[2],
				 z[0]*z[2],
				 z[0]*z[1]);
	    case 7: return Point(-z[1]*z[2],
				 (1-z[0])*z[2],
				 (1-z[0])*z[1]);
	}
    }
    Point P1curl_hex::LocalVectorValue (double x, double y, double z, int i) const {
	switch (i) {
	case  0: return Point((1-y)*(1-z),         0.0,         0.0);
	case  1: return Point(        0.0,     x*(1-z),         0.0);
	case  2: return Point(   -y*(1-z),         0.0,         0.0); 
	case  3: return Point(        0.0, (x-1)*(1-z),         0.0); 
	case  4: return Point(        0.0,         0.0, (1-x)*(1-y));
	case  5: return Point(        0.0,         0.0,     x*(1-y));
	case  6: return Point(        0.0,         0.0,         x*y);
	case  7: return Point(        0.0,         0.0,     y*(1-x));
	case  8: return Point(    (1-y)*z,         0.0,         0.0);
	case  9: return Point(        0.0,         x*z,         0.0);
	case 10: return Point(       -y*z,         0.0,         0.0); 
	case 11: return Point(        0.0,     z*(x-1),         0.0); 
	}
    }
    Point P1curl_hex::LocalCurl (double x, double y, double z, int i) const {
	switch (i) {
	case  0: return Point(0.0, y-1, 1-z); 
	case  1: return Point(  x, 0.0, 1-z); 
	case  2: return Point(0.0,   y, 1-z); 
	case  3: return Point(x-1, 0.0, 1-z); 
	case  4: return Point(x-1, 1-y, 0.0); 
	case  5: return Point( -x, y-1, 0.0); 
	case  6: return Point(  x,  -y, 0.0); 
	case  7: return Point(1-x,   y, 0.0); 
	case  8: return Point(0.0, 1-y,   z); 
	case  9: return Point( -x, 0.0,   z); 
	case 10: return Point(0.0,  -y,   z); 
	case 11: return Point(1-x, 0.0,   z); 
	}
    }

// ------------------------------------------------------------
// 54-NODED HEXAHEDRON FOR CURL
// ------------------------------------------------------------

    double P2curl_sp_hex::operator () (const Point& z, int i) const { 
	switch (i) {
	case 0: return (1-z[0])*(1-z[1])*(1-z[2]);
	case 1: return z[0]*(1-z[1])*(1-z[2]);
	case 2: return z[0]*z[1]*(1-z[2]);
	case 3: return (1-z[0])*z[1]*(1-z[2]);
	case 4: return (1-z[0])*(1-z[1])*z[2];
	case 5: return z[0]*(1-z[1])*z[2];
	case 6: return z[0]*z[1]*z[2];
	case 7: return (1-z[0])*z[1]*z[2];
	}
    }
    Point P2curl_sp_hex::LocalGradient (const Point& z, int i) const { 
	switch (i) {
	    case 0: return Point(-(1-z[1])*(1-z[2]),
				 -(1-z[0])*(1-z[2]),
				 -(1-z[0])*(1-z[1]));
	    case 1: return Point((1-z[1])*(1-z[2]),
				 -z[0]*(1-z[2]),
				 -z[0]*(1-z[1]));
	    case 2: return Point(z[1]*(1-z[2]),
				 z[0]*(1-z[2]),
				 -z[0]*z[1]);
	    case 3: return Point(-z[1]*(1-z[2]),
				 (1-z[0])*(1-z[2]),
				 -(1-z[0])*z[1]);
	    case 4: return Point(-(1-z[1])*z[2],
				 -(1-z[0])*z[2],
				 (1-z[0])*(1-z[1]));
	    case 5: return Point((1-z[1])*z[2],
				 -z[0]*z[2],
				 z[0]*(1-z[1]));
	    case 6: return Point(z[1]*z[2],
				 z[0]*z[2],
				 z[0]*z[1]);
	    case 7: return Point(-z[1]*z[2],
				 (1-z[0])*z[2],
				 (1-z[0])*z[1]);
	}
    }
    Point P2curl_sp_hex::LocalVectorValue (double x, double y, double z, int i) const {
	switch (i) {
        // the lowest order edge-based
	case  0: return Point((1-y)*(1-z),         0.0,         0.0);
	case  1: return Point(        0.0,     x*(1-z),         0.0);
	case  2: return Point(   -y*(1-z),         0.0,         0.0);
	case  3: return Point(        0.0, (x-1)*(1-z),         0.0);
	case  4: return Point(        0.0,         0.0, (1-x)*(1-y));
	case  5: return Point(        0.0,         0.0,     x*(1-y));
	case  6: return Point(        0.0,         0.0,         x*y);
	case  7: return Point(        0.0,         0.0,     y*(1-x));
	case  8: return Point(    (1-y)*z,         0.0,         0.0);
	case  9: return Point(        0.0,         x*z,         0.0);
	case 10: return Point(       -y*z,         0.0,         0.0);
	case 11: return Point(        0.0,     z*(x-1),         0.0);
        // 2-nd order edge-based
        case 12: return Point(2*(-1 + 2*x)*(-1 + y)*(-1 + z),2*(-1 + x)*x*(-1 + z),2*(-1 + x)*x*(-1 + y));
        case 13: return Point(-2*(-1 + y)*y*(-1 + z),-2*x*(-1 + 2*y)*(-1 + z),-2*x*(-1 + y)*y);
        case 14: return Point(-2*(-1 + 2*x)*y*(-1 + z),-2*(-1 + x)*x*(-1 + z),-2*(-1 + x)*x*y);
        case 15: return Point(2*(-1 + y)*y*(-1 + z),2*(-1 + x)*(-1 + 2*y)*(-1 + z),2*(-1 + x)*(-1 + y)*y);
        case 16: return Point(2*(-1 + y)*(-1 + z)*z,2*(-1 + x)*(-1 + z)*z,2*(-1 + x)*(-1 + y)*(-1 + 2*z));
        case 17: return Point(-2*(-1 + y)*(-1 + z)*z,-2*x*(-1 + z)*z,-2*x*(-1 + y)*(-1 + 2*z));
        case 18: return Point(2*y*(-1 + z)*z,2*x*(-1 + z)*z,2*x*y*(-1 + 2*z));
        case 19: return Point(-2*y*(-1 + z)*z,-2*(-1 + x)*(-1 + z)*z,-2*(-1 + x)*y*(-1 + 2*z));
        case 20: return Point(-2*(-1 + 2*x)*(-1 + y)*z,-2*(-1 + x)*x*z,-2*(-1 + x)*x*(-1 + y));
        case 21: return Point(2*(-1 + y)*y*z,2*x*(-1 + 2*y)*z,2*x*(-1 + y)*y);
        case 22: return Point(2*(-1 + 2*x)*y*z,2*(-1 + x)*x*z,2*(-1 + x)*x*y);
        case 23: return Point(-2*(-1 + y)*y*z,-2*(-1 + x)*(-1 + 2*y)*z,-2*(-1 + x)*(-1 + y)*y);
        // 2-nd order face based
        // face (1, 0, 3, 2)
        case 24: return Point(4*(y-1)*y*(z-1),0,0);
        case 25: return Point(0,-4*(x-1)*x*(z-1),0);
        case 26: return Point(-4*(2*x-1)*(y-1)*y*(z-1),4*(x-1)*x*(2*y-1)*(z-1),0);
        case 27: return Point(-4*(-1 + 2*x)*(-1 + y)*y*(-1 + z),-4*(-1 + x)*x*(-1 + 2*y)*(-1 + z),-4*(-1 + x)*x*(-1 + y)*y);
        // face (0, 1, 5, 4)
        case 28: return Point(-4*(y-1)*(z-1)*z,0,0);
        case 29: return Point(0,0,-4*(x-1)*x*(y-1));
        case 30: return Point(-4*(2*x-1)*(y-1)*(z-1)*z,0,4*(x-1)*x*(y-1)*(2*z-1));
        case 31: return Point(-4*(-1 + 2*x)*(-1 + y)*(-1 + z)*z,-4*(-1 + x)*x*(-1 + z)*z,-4*(-1 + x)*x*(-1 + y)*(-1 + 2*z));
        // face (1, 2, 6, 5)
        case 32: return Point(0,4*x*(z-1)*z,0);
        case 33: return Point(0,0,4*x*(y-1)*y);
        case 34: return Point(0,4*x*(2*y-1)*(z-1)*z,-4*x*(y-1)*y*(2*z-1));
        case 35: return Point(4*(-1 + y)*y*(-1 + z)*z,4*x*(-1 + 2*y)*(-1 + z)*z,4*x*(-1 + y)*y*(-1 + 2*z));
        // face (2, 3, 7, 6)
        case 36: return Point(-4*y*(z-1)*z,0,0);
        case 37: return Point(0,0,4*(x-1)*x*y);
        case 38: return Point(4*(2*x-1)*y*(z-1)*z,0,-4*(x-1)*x*y*(2*z-1));
        case 39: return Point(4*(-1 + 2*x)*y*(-1 + z)*z,4*(-1 + x)*x*(-1 + z)*z,4*(-1 + x)*x*y*(-1 + 2*z));
        // face (3, 0, 4, 7)
        case 40: return Point(0,4*(x-1)*(z-1)*z,0);
        case 41: return Point(0,0,-4*(x-1)*(y-1)*y);
        case 42: return Point(0,-4*(x-1)*(2*y-1)*(z-1)*z,4*(x-1)*(y-1)*y*(2*z-1));
        case 43: return Point(-4*(-1 + y)*y*(-1 + z)*z,-4*(-1 + x)*(-1 + 2*y)*(-1 + z)*z,-4*(-1 + x)*(-1 + y)*y*(-1 + 2*z));
        // face (4, 5, 6, 7)
        case 44: return Point(4*(y-1)*y*z,0,0);
        case 45: return Point(0,4*(x-1)*x*z,0);
        case 46: return Point(4*(2*x-1)*(y-1)*y*z,-4*(x-1)*x*(2*y-1)*z,0);
        case 47: return Point(4*(-1 + 2*x)*(-1 + y)*y*z,4*(-1 + x)*x*(-1 + 2*y)*z,4*(-1 + x)*x*(-1 + y)*y);
        // 2-nd order cell-based
        case 48: return Point(4*y*(y-1)*z*(z-1),0,0);
        case 49: return Point(0,4*x*(x-1)*z*(z-1),0);
        case 50: return Point(0,0,4*x*(x-1)*y*(y-1));
        case 51: return Point(8*(2*x-1)*y*(y-1)*z*(z-1),-8*(2*y-1)*x*(x-1)*z*(z-1),8*(2*z-1)*x*(x-1)*y*(y-1));
        case 52: return Point(8*(2*x-1)*y*(y-1)*z*(z-1),8*(2*y-1)*x*(x-1)*z*(z-1),-8*(2*z-1)*x*(x-1)*y*(y-1));
        case 53: return Point(8*(-1 + 2*x)*(-1 + y)*y*(-1 + z)*z,8*(-1 + x)*x*(-1 + 2*y)*(-1 + z)*z,8*(-1 + x)*x*(-1 + y)*y*(-1 + 2*z));
	}
    }
    Point P2curl_sp_hex::LocalCurl (double x, double y, double z, int i) const {
	switch (i) {
        // the lowest order edge-based
	case  0: return Point(0.0, y-1, 1-z); 
	case  1: return Point(  x, 0.0, 1-z); 
	case  2: return Point(0.0,   y, 1-z); 
	case  3: return Point(x-1, 0.0, 1-z); 
	case  4: return Point(x-1, 1-y, 0.0); 
	case  5: return Point( -x, y-1, 0.0); 
	case  6: return Point(  x,  -y, 0.0); 
	case  7: return Point(1-x,   y, 0.0); 
	case  8: return Point(0.0, 1-y,   z); 
	case  9: return Point( -x, 0.0,   z); 
	case 10: return Point(0.0,  -y,   z); 
	case 11: return Point(1-x, 0.0,   z); 
        // 2-nd order edge-based
        case 12: return Point(0,0,0);
        case 13: return Point(0,0,0);
        case 14: return Point(0,0,0);
        case 15: return Point(0,0,0);
        case 16: return Point(0,0,0);
        case 17: return Point(0,0,0);
        case 18: return Point(0,0,0);
        case 19: return Point(0,0,0);
        case 20: return Point(0,0,0);
        case 21: return Point(0,0,0);
        case 22: return Point(0,0,0);
        case 23: return Point(0,0,0);
        // 2-nd order face based
        // face (1, 0, 3, 2)
        case 24: return Point(0,4*(y-1)*y,-4*(2*y-1)*(z-1));
        case 25: return Point(4*(x-1)*x,0,-4*(2*x-1)*(z-1));
        case 26: return Point(-4*(x-1)*x*(2*y-1),-4*(2*x-1)*(y-1)*y,8*(2*x-1)*(2*y-1)*(z-1));
        case 27: return Point(0,0,0);
        // face (0, 1, 5, 4)
        case 28: return Point(0,-4*(y-1)*(2*z-1),4*(z-1)*z);
        case 29: return Point(-4*(x-1)*x,4*(2*x-1)*(y-1),0);
        case 30: return Point(4*(x-1)*x*(2*z-1),-8*(2*x-1)*(y-1)*(2*z-1),4*(2*x-1)*(z-1)*z);
        case 31: return Point(0,0,0);
        // face (1, 2, 6, 5)
        case 32: return Point(-4*x*(2*z-1),0,4*(z-1)*z);
        case 33: return Point(4*x*(2*y-1),-4*(y-1)*y,0);
        case 34: return Point(-8*x*(2*y-1)*(2*z-1),4*(y-1)*y*(2*z-1),4*(2*y-1)*(z-1)*z);
        case 35: return Point(0,0,0);
        // face (2, 3, 7, 6)
        case 36: return Point(0,-4*y*(2*z-1),4*(z-1)*z);
        case 37: return Point(4*(x-1)*x,-4*(2*x-1)*y,0);
        case 38: return Point(-4*(x-1)*x*(2*z-1),8*(2*x-1)*y*(2*z-1),-4*(2*x-1)*(z-1)*z);
        case 39: return Point(0,0,0);
        // face (3, 0, 4, 7)
        case 40: return Point(-4*(x-1)*(2*z-1),0,4*(z-1)*z);
        case 41: return Point(-4*(x-1)*(2*y-1),4*(y-1)*y,0);
        case 42: return Point(8*(x-1)*(2*y-1)*(2*z-1),-4*(y-1)*y*(2*z-1),-4*(2*y-1)*(z-1)*z);
        case 43: return Point(0,0,0);
        // face (4, 5, 6, 7)
        case 44: return Point(0,4*(y-1)*y,-4*(2*y-1)*z);
        case 45: return Point(-4*(x-1)*x,0,4*(2*x-1)*z);
        case 46: return Point(4*(x-1)*x*(2*y-1),4*(2*x-1)*(y-1)*y,-8*(2*x-1)*(2*y-1)*z);
        case 47: return Point(0,0,0);
        // 2-nd order cell-based
        case 48: return Point(0,4*(y-1)*y*(2*z-1),-4*(2*y-1)*(z-1)*z);
        case 49: return Point(-4*(x-1)*x*(2*z-1),0,4*(2*x-1)*(z-1)*z);
        case 50: return Point(4*(x-1)*x*(2*y-1),-4*(2*x-1)*(y-1)*y,0);
        case 51: return Point(16*(x-1)*x*(2*y-1)*(2*z-1),0,-16*(2*x-1)*(2*y-1)*(z-1)*z);
        case 52: return Point(-16*(x-1)*x*(2*y-1)*(2*z-1),16*(2*x-1)*(y-1)*y*(2*z-1),0);
        case 53: return Point(0,0,0);
	}
    }

// ------------------------------------------------------------
// 64-NODED HEXAHEDRON
// ------------------------------------------------------------
static const double c_64 [64][3] = { {0.0, 0.0, 0.0},
                                     {1.0, 0.0, 0.0},
                                     {1.0, 1.0, 0.0},
			      {0.0, 1.0, 0.0},
			      {0.0, 0.0, 1.0},
			      {1.0, 0.0, 1.0},
			      {1.0, 1.0, 1.0},
			      {0.0, 1.0, 1.0},
			      {1/3, 0.0, 0.0},
			      {2/3, 0.0, 0.0},
			      {1.0, 1/3, 0.0},
			      {1.0, 2/3, 0.0},
			      {2/3, 1.0, 0.0},
			      {1/3, 1.0, 0.0},
			      {0.0, 2/3, 0.0},
			      {0.0, 1/3, 0.0},
			      {0.0, 0.0, 1/3},
			      {1.0, 0.0, 1/3},
			      {1.0, 1.0, 1/3},
			      {0.0, 1.0, 1/3},
			      {0.0, 0.0, 2/3},
			      {1.0, 0.0, 2/3},
			      {1.0, 1.0, 2/3},
			      {0.0, 1.0, 2/3},
			      {1/3, 0.0, 1.0},
			      {2/3, 0.0, 1.0},
			      {1.0, 1/3, 1.0},
			      {1.0, 2/3, 1.0},
			      {2/3, 1.0, 1.0},
			      {1/3, 1.0, 1.0},
			      {0.0, 2/3, 1.0},
			      {0.0, 1/3, 1.0},
			      {1/3, 1/3, 0.0},
			      {2/3, 1/3, 0.0},
			      {2/3, 2/3, 0.0},
			      {1/3, 2/3, 0.0},
			      {1/3, 0.0, 1/3},
			      {2/3, 0.0, 1/3},
			      {1.0, 1/3, 1/3},
			      {1.0, 2/3, 1/3},
			      {2/3, 1.0, 1/3},
			      {1/3, 1.0, 1/3},
			      {0.0, 2/3, 1/3},
			      {0.0, 1/3, 1/3},
			      {1/3, 0.0, 2/3},
			      {2/3, 0.0, 2/3},
			      {1.0, 1/3, 2/3},
			      {1.0, 2/3, 2/3},
			      {2/3, 1.0, 2/3},
			      {1/3, 1.0, 2/3},
			      {0.0, 2/3, 2/3},
			      {0.0, 1/3, 2/3},
			      {1/3, 1/3, 1.0},
			      {2/3, 1/3, 1.0},
			      {2/3, 2/3, 1.0},
			      {1/3, 2/3, 1.0},
			      {1/3, 1/3, 1/3},
			      {2/3, 1/3, 1/3},
			      {2/3, 2/3, 1/3},
			      {1/3, 2/3, 1/3},
			      {1/3, 1/3, 2/3},
			      {2/3, 1/3, 2/3},
			      {2/3, 2/3, 2/3},
			      {1/3, 2/3, 2/3} };

inline double  P3hex::c1 (const double x) const { return  -1/2   * (x-1)* (3*x-1)* (3*x-2); }
inline double  P3hex::c2 (const double x) const { return   1/2*x        * (3*x-1)* (3*x-2); }
inline double  P3hex::c3 (const double x) const { return   9/2*x * (x-1)         * (3*x-2); }
inline double  P3hex::c4 (const double x) const { return  -9/2*x * (x-1)* (3*x-1);          }
inline double P3hex::Dc1 (const double x) const { return -27/2*x * x + 18*x - 11/2*x; }
inline double P3hex::Dc2 (const double x) const { return  27/2*x * x -  9*x + 1;      }
inline double P3hex::Dc3 (const double x) const { return  81/2*x * x - 45*x + 9;      }
inline double P3hex::Dc4 (const double x) const { return -81/2*x * x + 36*x - 9/2;    }
inline double P3hex::shapefct (int i, int k, double x) const {
    double s = c_64[i][k];
    if (s == 0.0) return  c1(x);
    if (s == 1.0) return  c2(x);
    if (s == 1/3) return  c3(x);
    if (s == 2/3) return  c4(x);
}

inline double P3hex::Dshapefct (int i, int k, double x) const {
    double s = c_64[i][k];
    if (s == 0.0) return Dc1(x);
    if (s == 1.0) return Dc2(x);
    if (s == 1/3) return Dc3(x);
    if (s == 2/3) return Dc4(x);
}

inline double P3hex::operator () (const Point& z, int i) const { 
    return  shapefct(i,0,z[0]) * shapefct(i,1,z[1]) * shapefct(i,2,z[2]) ;
}

Point P3hex::LocalGradient (const Point& z, int i) const { 
    return Point (Dshapefct(i,0,z[0])* shapefct(i,1,z[1])* shapefct(i,2,z[2]),
                  shapefct(i,0,z[0])*Dshapefct(i,1,z[1])* shapefct(i,2,z[2]),
                  shapefct(i,0,z[0])* shapefct(i,1,z[1])*Dshapefct(i,2,z[2]) );
}

P3hex::P3hex (const Quadrature& Q, const Quadrature& FaceQ)
	: Shape(Q,FaceQ) { fill(); }

// ------------------------------------------------------------
// 9-NODED QUADRILATERAL
// ------------------------------------------------------------
inline double P2quad::q01 (const double x) const { return      x * (2*x-1); }
inline double P2quad::q00 (const double x) const { return   (x-1)* (2*x-1); }
inline double P2quad::q11 (const double x) const { return  4 * x * (1-x);   }
inline double P2quad::Dq01 (const double x) const { return  4*x-1; }
inline double P2quad::Dq00 (const double x) const { return  4*x-3; }
inline double P2quad::Dq11 (const double x) const { return -8*x+4; }

double P2quad::operator () (const Point& z, int i) const { 
    switch (i) {
    case 0: return q00(z[0])*q00(z[1]);
    case 1: return q01(z[0])*q00(z[1]);
    case 2: return q01(z[0])*q01(z[1]);
    case 3: return q00(z[0])*q01(z[1]);
    case 4: return q11(z[0])*q00(z[1]);
    case 5: return q01(z[0])*q11(z[1]);
    case 6: return q11(z[0])*q01(z[1]);
    case 7: return q00(z[0])*q11(z[1]);
    case 8: return q11(z[0])*q11(z[1]);
    }
}

Point P2quad::LocalGradient (const Point& z, int i) const { 
    switch(i) {
    case 0: return Point (  Dq00(z[0])*q00(z[1]), 
			    q00(z[0])*Dq00(z[1]) );
    case 1: return Point (  Dq01(z[0])*q00(z[1]),
			    q01(z[0])*Dq00(z[1]) );
    case 2: return Point (  Dq01(z[0])*q01(z[1]),
			    q01(z[0])*Dq01(z[1]) );
    case 3: return Point (  Dq00(z[0])*q01(z[1]), 
			    q00(z[0])*Dq01(z[1]) );
    case 4: return Point (  Dq11(z[0])*q00(z[1]),
			    q11(z[0])*Dq00(z[1]) );
    case 5: return Point (  Dq01(z[0])*q11(z[1]),
			    q01(z[0])*Dq11(z[1]) );
    case 6: return Point (  Dq11(z[0])*q01(z[1]), 
			    q11(z[0])*Dq01(z[1]) );
    case 7: return Point (  Dq00(z[0])*q11(z[1]),
			    q00(z[0])*Dq11(z[1]) );
    case 8: return Point (  Dq11(z[0])*q11(z[1]),
			    q11(z[0])*Dq11(z[1]) );
    default: cerr<<"Problem in P2quad::LocalGradient ... "<<endl;
    }
    return zero;
}
P2quad::P2quad (const Quadrature& Q, const Quadrature& FaceQ)
	: Shape(Q,FaceQ) { fill(); }

// ------------------------------------------------------------
// Raviart Thomas TRIANGLE
// ------------------------------------------------------------
Point RT0tri::LocalVector(const Point& z, int i) const {
    switch(i) {
    case 0: return Point(z[0], z[1]-1.0);
    case 1: return Point(z[0],z[1]);
    case 2: return Point(z[0]-1.0, z[1]);
    }
}
double RT0tri::LocalDiv(const Point& z, int i) const {
    return 2.0;
}
// ------------------------------------------------------------
// Raviart Thomas Quadrilateral
// ------------------------------------------------------------
Point RT0quad::LocalVector(const Point& z, int i) const {
    switch(i) {
    case 0: return Point(0.0, z[1]-1.0);
    case 1: return Point(z[0],0.0);
    case 2: return Point(0.0, z[1]);
    case 3: return Point(z[0]-1.0, 0.0);
    }
}
double RT0quad::LocalDiv(const Point& z, int i) const {
    return 1.0;
}
// ------------------------------------------------------------
// Raviart Thomas Tetrahedron (not yet implemented)
// ------------------------------------------------------------
Point RT0tet::LocalVector(const Point& z, int i) const {
    return zero;

}
double RT0tet::LocalDiv(const Point& z, int i) const {
    return 0.0;
}
// ------------------------------------------------------------
// Raviart Thomas Hexahedron
// ------------------------------------------------------------
Point RT0hex::LocalVector(const Point& z, int i) const {
    switch(i) {
    case 0: return Point(0.0, 0.0, z[2]-1.0);
    case 1: return Point(0.0, z[1]-1.0, 0.0);
    case 2: return Point(z[0], 0.0, 0.0);
    case 3: return Point(0.0, z[1], 0.0);
    case 4: return Point(z[0]-1.0, 0.0, 0.0);
    case 5: return Point(0.0, 0.0, z[2]);
    }
}
double RT0hex::LocalDiv(const Point& z, int i) const {
    return 1.0;
}

// ------------------------------------------------------------
//    Transformation
// ------------------------------------------------------------

Transformation::Transformation (const vector<Point>& z) { 
    for (int i=0; i<z.size(); ++i) { 
        for (int j=0; j<3; ++j) {
			F[i][j] = z[i][j];
		} 
    }
    for (int i=z.size(); i<3; ++i) 
        for (int j=0; j<3; ++j)
            F[i][j] = 0;
    if (z.size() == 2) {
        det = z[0][0]*z[1][1] - z[0][1]*z[1][0];
        double invdet = 1.0 / det;
        T[0][0] =  z[1][1] * invdet;
        T[0][1] = -z[1][0] * invdet;
        T[1][0] = -z[0][1] * invdet;
        T[1][1] =  z[0][0] * invdet;
        T[2][2] = 1;
        T[2][0] = T[2][1] = T[0][2] = T[1][2] = 0;
        return;
    }
    det = z[0][0]*z[1][1]*z[2][2]
        + z[1][0]*z[2][1]*z[0][2]
        + z[2][0]*z[0][1]*z[1][2]
        - z[2][0]*z[1][1]*z[0][2]
        - z[0][0]*z[2][1]*z[1][2]
        - z[1][0]*z[0][1]*z[2][2];
    if (abs(det)<Eps) { Exit("singular Transformation"); }
    double invdet = 1.0/det;
    T[0][0] = ( z[1][1]*z[2][2] - z[2][1]*z[1][2]) * invdet;
    T[1][0] = (-z[0][1]*z[2][2] + z[2][1]*z[0][2]) * invdet;
    T[2][0] = ( z[0][1]*z[1][2] - z[1][1]*z[0][2]) * invdet;
    T[0][1] = (-z[1][0]*z[2][2] + z[2][0]*z[1][2]) * invdet;
    T[1][1] = ( z[0][0]*z[2][2] - z[2][0]*z[0][2]) * invdet;
    T[2][1] = (-z[0][0]*z[1][2] + z[1][0]*z[0][2]) * invdet;
    T[0][2] = ( z[1][0]*z[2][1] - z[2][0]*z[1][1]) * invdet;
    T[1][2] = (-z[0][0]*z[2][1] + z[2][0]*z[0][1]) * invdet;
    T[2][2] = ( z[0][0]*z[1][1] - z[1][0]*z[0][1]) * invdet;
}

Point Transformation::operator () (const Point& P) const {
    return Point(F[0][0]*P[0]+F[1][0]*P[1]+F[2][0]*P[2],
                 F[0][1]*P[0]+F[1][1]*P[1]+F[2][1]*P[2],
                 F[0][2]*P[0]+F[1][2]*P[1]+F[2][2]*P[2]);
}
Point Transformation::ApplyJ(const Point& P) const {
    return Point(F[0][0]*P[0]+F[0][1]*P[1]+F[0][2]*P[2],
                 F[1][0]*P[0]+F[1][1]*P[1]+F[1][2]*P[2],
                 F[2][0]*P[0]+F[2][1]*P[1]+F[2][2]*P[2]);
};

Point operator * (const Transformation& T, const Point& z) {
    return Point(T.T[0][0]*z[0]+T.T[0][1]*z[1]+T.T[0][2]*z[2],
                 T.T[1][0]*z[0]+T.T[1][1]*z[1]+T.T[1][2]*z[2],
                 T.T[2][0]*z[0]+T.T[2][1]*z[1]+T.T[2][2]*z[2]);
}
Point operator * (const Point& z, const Transformation& T) {
    return Point(T.T[0][0]*z[0]+T.T[1][0]*z[1]+T.T[2][0]*z[2],
                 T.T[0][1]*z[0]+T.T[1][1]*z[1]+T.T[2][1]*z[2],
                 T.T[0][2]*z[0]+T.T[1][2]*z[1]+T.T[2][2]*z[2]);
}
ostream& operator << (ostream& s, const Transformation& T) {
    return s << T.T[0][0] << " " << T.T[0][1] << " " << T.T[0][2] << endl
             << T.T[1][0] << " " << T.T[1][1] << " " << T.T[1][2] << endl
             << T.T[2][0] << " " << T.T[2][1] << " " << T.T[2][2] << endl
	     <<"--------------------"<<endl
	     << T.F[0][0] << " " << T.F[0][1] << " " << T.F[0][2] << endl
             << T.F[1][0] << " " << T.F[1][1] << " " << T.F[1][2] << endl
             << T.F[2][0] << " " << T.F[2][1] << " " << T.F[2][2] << endl
	     <<"DET= " <<T.Det()<<endl<<endl;
;
}





