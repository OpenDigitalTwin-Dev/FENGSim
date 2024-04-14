// file: Point.h
// author: Christian Wieners
// $Header: /public/M++/src/Point.h,v 1.4 2009-06-09 14:45:14 wieners Exp $

#ifndef _POINT_H_
#define _POINT_H_

#include "Compiler.h"
#include "Constants.h"
#include <stdio.h>
#include <iostream>

class Point {
    double z[3];
public:
    Point () { z[0] = z[1] = z[2] = 0.0; }
    Point (const Point& y) { z[0]=y.z[0]; z[1]=y.z[1]; z[2]=y.z[2]; }
    Point (double a, double b, double c=0) { z[0]=a; z[1]=b; z[2]=c; } 
    Point (const double* a) { z[0]=a[0]; z[1]=a[1]; z[2]=a[2]; } 
//    Point (double a) { z[0]=a; z[1]=a; z[2]=a; } 
//    Point (int a) { z[0]=double(a); z[1]=double(a); z[2]=double(a); } 
    template <class C> Point (const C& c) { z[0]=c[0]; z[1]=c[1]; z[2]=c[2]; } 
    double operator [] (const unsigned int i) const { return z[i]; }
    double& operator [] (const unsigned int i) { return z[i]; }
    const double* operator () () const { return z; }
//    Point& operator = (double a) { z[0] = z[1] = z[2] = a; return *this; }
    Point& operator = (const Point& y) { 
        if (&y != this) memcpy(z,y.z,3*sizeof(double)); 
	return *this; 
    }
    Point& operator += (const Point& y) {
	z[0] += y.z[0]; z[1] += y.z[1]; z[2] += y.z[2]; return *this; 
    }
    Point& operator -= (const Point& y) {
	z[0] -= y.z[0]; z[1] -= y.z[1]; z[2] -= y.z[2]; return *this; 
    }
    Point& operator *= (double a) {
	z[0] *= a; z[1] *= a; z[2] *= a; return *this; 
    }
    Point& operator /= (double a) {
	z[0] /= a; z[1] /= a; z[2] /= a; return *this; 
    }
    bool operator == (const Point& y) const {
	if (abs(z[0] - y.z[0]) > GeometricTolerance) return false;
	if (abs(z[1] - y.z[1]) > GeometricTolerance) return false;
	if (abs(z[2] - y.z[2]) > GeometricTolerance) return false;
	return true;
    }
    bool operator != (const Point& y) const { return !(*this == y); }
    bool operator < (const Point& y) const {
	if (z[0] < y.z[0] - GeometricTolerance) return true;
	if (z[0] > y.z[0] + GeometricTolerance) return false;
	if (z[1] < y.z[1] - GeometricTolerance) return true;
	if (z[1] > y.z[1] + GeometricTolerance) return false;
	if (z[2] < y.z[2] - GeometricTolerance) return true;
	return false;
    }
    bool operator > (const Point& y) const {
	if (z[0] > y.z[0] + GeometricTolerance) return true;
	if (z[0] < y.z[0] - GeometricTolerance) return false;
	if (z[1] > y.z[1] + GeometricTolerance) return true;
	if (z[1] < y.z[1] - GeometricTolerance) return false;
	if (z[2] > y.z[2] + GeometricTolerance) return true;
	return false;
    }
    size_t Hash () const {
	return size_t(115421.1*z[0] + 124323.3*z[1] + 221313.7*z[2]
		      +345453*GeometricTolerance);
    }
};
const Point zero(0.0,0.0,0.0);
const Point Infty(infty,infty,infty);
class Hash { 
 public: 
    size_t operator () (const Point& x) const { return x.Hash(); }
};
inline Point operator + (const Point& x, const Point& y) {
    Point z = x; return z += y;
}
inline Point operator - (const Point& x, const Point& y) {
    Point z = x; return z -= y;
}
inline Point operator ^ (const Point& x, const Point& y) {
    return Point(x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]);
}
inline Point curl (const Point& x, const Point& y) {
    return Point(x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]);
}
inline double operator * (const Point& x, const Point& y) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}
inline Point operator * (double b, const Point& x) {
    Point z = x; return z *= b;
}
inline Point operator * (const Point& x, double b) {
    Point z = x; return z *= b;
}
inline Point operator * (int b, const Point& x) {
    Point z = x; return z *= double(b);
}
inline Point operator / (const Point& x, double b) {
    Point z = x; return z *= (1/b); 
}
inline double norm (const Point& x) { return sqrt(x*x); }
inline double dist (const Point& x, const Point& y) { return norm(x-y); }
inline double det (const Point& x, const Point& y, const Point& z) {
    return x * curl(y,z);

}
inline double arg (const Point& x) {
    return atan2(x[1],x[0]); 
    const double pi = 4 * atan(1.0);
    if (abs(x[1]) <= Eps) return x[1];
    if (abs(x[1]) < x[0]) return atan(x[1]/x[0]);
    if ((x[1] > 0) && (-x[0] > x[1])) return pi - atan(-x[1]/x[0]);
    if ((x[1] < 0) && (-x[0] > -x[1])) return pi - atan(x[1]/x[0]);
    if (abs(x[0]) < x[1]) return 0.5*pi - atan(x[0]/x[1]);
    return -0.5*pi - atan(x[0]/x[1]);
}

void Point_2d();
void Point_3d();
ostream& operator << (ostream&, const Point&);
void SetPointPrecision (int, int);

class Points : public vector<Point> {
 public:
    Points (int n) : vector<Point>(n) {}
    Points (int n, const Point* x) : vector<Point>(n) {
	for (int i=0; i<n; ++i) (*this)[i] = x[i];
    }
    Points (int n1, const Point* x1, 
	    int n2, const Point* x2) : vector<Point>(n1+n2){
	for (int i=0; i<n1; ++i) (*this)[i]    = x1[i];
	for (int i=0; i<n2; ++i) (*this)[n1+i] = x2[i];
    }
    Points (int n1, const Point* x1, 
	    int n2, const Point* x2,
	    int n3, const Point* x3,
	    int n4, const Point* x4) : vector<Point>(n1+n2+n3+n4){
	for (int i=0; i<n1; ++i) (*this)[i]          = x1[i];
	for (int i=0; i<n2; ++i) (*this)[n1+i]       = x2[i];
	for (int i=0; i<n3; ++i) (*this)[n1+n2+i]    = x3[i];
	for (int i=0; i<n4; ++i) (*this)[n1+n2+n3+i] = x4[i];
    }
};

#endif
