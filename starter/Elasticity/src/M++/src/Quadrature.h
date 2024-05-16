// file:     Quadrature.h
// author:   Christian Wieners
// $Header: /public/M++/src/Quadrature.h,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#ifndef _QUADRATURE_H_
#define _QUADRATURE_H_

#include "Point.h"

class Quadrature {
    int n;
    const Point *z;
    const double *w;
 public:
    Quadrature (int N, const Point* Z, const double* W) : n(N), z(Z), w(W) {}
    int size () const { return n; }
    const Point& QPoint (int i) const { return z[i]; }
    double Weight (int i) const { return w[i]; }

    const Point* QPoint () const { return z; }
    const double* Weight () const { return w; }
};
inline ostream& operator << (ostream& s, const Quadrature& Q) {
    for (int i=0; i<Q.size(); ++i)
	s << "z[" << i << "] = " << Q.QPoint(i)
	  << " w[" << i << "] = " << Q.Weight(i)
	  << endl;
    return s;
}

const Quadrature& GetQuadrature (const string&);

#endif
