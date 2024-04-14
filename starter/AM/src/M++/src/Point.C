// file: Point.C
// author: Christian Wieners
// $Header: /public/M++/src/Point.C,v 1.3 2009-06-09 14:45:14 wieners Exp $

#include "Point.h"

int PointDim = 3;

void Point_2d () { PointDim = 2; } 
void Point_3d () { PointDim = 3; } 

const char* fmt_1d = "%8.4f";
const char* fmt_2d = "%8.4f %8.4f";
const char* fmt_3d = "%8.4f %8.4f %8.4f";
string fmt1d;
string fmt2d;
string fmt3d;

void SetPointPrecision (int p, int e) {
    char fmt[64];
    sprintf(fmt,"%d.%df",p+e,p);
    fmt1d = string("\%") + string(fmt);
    fmt2d = string("\%") + string(fmt) + string( " \%") + string(fmt); 
    fmt3d = string("\%") + string(fmt) 
             + string( " \%") + string(fmt) 
             + string( " \%") + string(fmt); 
    fmt_1d = fmt1d.c_str();
    fmt_2d = fmt2d.c_str();
    fmt_3d = fmt3d.c_str();
}

ostream& operator << (ostream& s, const Point& z) {
    if (norm(z-Infty) < 1.0) {
	if (PointDim == 2) return s << "           Infty";
	if (PointDim == 3) return s << "                   Infty";
    }
    double x[3];
    if (abs(z[0]) < GeometricTolerance) x[0] = 0; else x[0] = z[0]; 
    if (abs(z[1]) < GeometricTolerance) x[1] = 0; else x[1] = z[1]; 
    if (abs(z[2]) < GeometricTolerance) x[2] = 0; else x[2] = z[2]; 
    char buf[128];
    if (PointDim == 2) sprintf(buf,fmt_2d,x[0],x[1]);
    if (PointDim == 3) sprintf(buf,fmt_3d,x[0],x[1],x[2]);
    return s << buf;
}
