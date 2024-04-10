// file: Discretization.h
// author: Christian Wieners
// $Header: /public/M++/src/Discretization.h,v 1.11 2008-10-07 22:01:10 mueller Exp $

#ifndef _DISCRETIZATION_H_
#define _DISCRETIZATION_H_

#include "DoF.h"
#include "Shape.h"

class Discretization  : public dof {
protected:
    const Shape* S[5][3];
    string name;
    string GetDiscName () {
		string s = "linear"; ReadConfig(Settings,"Discretization",s);
		return s;
    }
private:
    void clean () {
		for (int i=0; i<5; ++i) 
			for (int k=0; k<2; ++k) 
				S[i][k] = 0;
    }
    void fill (int, int);
    void fill (const Discretization&, int, int);
public:
    Discretization (const string& disc, int m = 1, int dim = -1, int n = 0) 
		: dof(GetDoF(disc,m,dim,n)), name(disc) { 
		fill(m,dim);
    } 
    Discretization (const string& disc, const Discretization& Disc, 
					int m = 1, int dim = -1) 
		: dof(GetDoF(disc,m,dim,Disc.n_infty())), name(disc) { 
		fill(Disc,m,dim); 
    } 
    Discretization (int m = 1, int dim = -1, int n = 0) 
		: dof(GetDoF(GetDiscName(),m,dim,n)), name(GetDiscName()) { 
		fill(m,dim); 
    } 
    Discretization (DoF* D, const string n = "linear") 
		: dof(D), name(n) { 
		clean(); 
    }
    ~Discretization () { 
		delete ptr(); 
		for (int i=0; i<5; ++i) 
			for (int k=0; k<2; ++k) 
				if (S[i][k]) delete S[i][k];
    }
    const Shape& GetShape (const cell& c, int k = 0) const {
		switch (c.ReferenceType()) {
		case TRIANGLE:      return *S[0][k]; 
		case QUADRILATERAL: return *S[1][k]; 
		case TETRAHEDRON:   return *S[2][k]; 
		case PRISM:         return *S[3][k]; 
		case HEXAHEDRON:    return *S[4][k]; 
		}
    }
    const Quadrature& GetQuad (const cell& c) const {
		return GetShape(c).GetQuad(); 
    }
    const Quadrature& GetFaceQuad (const cell& c, int face) const {
		return GetShape(c).GetFaceQuad(); 
    }
    friend ostream& operator << (ostream& s, const Discretization& D) {
		return s << D.name; }
	
    string DiscName () const     { return name; }
};

#endif
