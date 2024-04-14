// file: DoF.h
// author: Christian Wieners
// $Header: /public/M++/src/DoF.C,v 1.22 2009-11-24 09:46:35 wieners Exp $

#include "DoF.h"

class VertexDoF : public DoF {
	int m;
public:
	VertexDoF (int M = 1, int n = 0, bool bnd = false) : DoF(n,bnd), m(M) {}
	int NodalPoints (const cell& c) const { return c.Corners(); }
    void NodalPoints (const cell& c, vector<Point>& z) const {
		z.resize(NodalPoints(c));
		for (int i=0; i<z.size(); ++i) z[i] = c[i];
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
		z.resize(NodalPoints(c));
		for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
		return c.FaceCorners(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
		return c.facecorner(i,k); 
    }
    int TypeDoF (int tp) const { 
		if (NODETYPE(tp) == VERTEX) return m;
		return 0;
    }
    string Name () const { return "VertexDoF"; }
};

class VertexEdgeDoF : public DoF {
    int m;
public:
    VertexEdgeDoF (int M = 1, int n = 0, bool bnd = false) : DoF(n,bnd), m(M) {}
    int NodalPoints (const cell& c) const { return c.Corners() + c.Edges(); }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i)+c.FaceEdges(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	int fc = c.FaceCorners(i);
	if (k < fc) return c.facecorner(i,k); 
	return c.Corners() + c.faceedge(i,k-fc); 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return m;
	if (NODETYPE(tp) == EDGE)   return m;
	return 0;
    }
    string Name () const { return "VertexEdgeDoF"; }
};

class VertexEdgeCellDoF : public DoF {
    int m;
public:
    VertexEdgeCellDoF (int M = 1, int n = 0, bool bnd = false) : DoF(n,bnd), m(M) {}
    int NodalPoints (const cell& c) const { 
	return c.Corners() + c.Edges() + 1; 
    }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
	z[n++] = c();
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i)+c.FaceEdges(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	int fc = c.FaceCorners(i);
	if (k < fc) return c.facecorner(i,k); 
	return c.Corners() + c.faceedge(i,k-fc); 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return m;
	if (NODETYPE(tp) == EDGE)   return m;
	if (NODETYPE(tp) == CELL)   return m;
	return 0;
    }
    string Name () const { return "VertexEdgeCellDoF"; }
};


class EdgeDoF : public DoF {
    int m;
public:
    EdgeDoF (int M = 1, int n = 0, bool bnd = false) : DoF(n,bnd), m(M) {}
    int NodalPoints (const cell& c) const { return c.Edges(); }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<c.Edges(); ++i) z[i] = c.Edge(i);
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceEdges(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	return c.faceedge(i,k); 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == EDGE) return m;
	return 0;
    }
    string Name () const { return "EdgeDoF"; }
};

class FaceDoF : public DoF {
    int m;
public:
    FaceDoF (int M = 1, int n = 0, bool bnd = false) : DoF(n,bnd), m(M) {}
    int NodalPoints (const cell& c) const { return c.Faces(); }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<c.Faces(); ++i) z[i] = c.Face(i);
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { return 1; }
    int NodalPointOnFace (const cell& c, int i, int k) const { return i; }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == FACE) return m;
	return 0;
    }
    string Name () const { return "FaceDoF"; }
};

class CellDoF : public DoF {
    int m;
public:
    CellDoF (int M = 1) : m(M) {}
    int NodalPoints (const cell& c) const { return 1; }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	z[0] = c();
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { return 0; }
    int NodalPointOnFace (const cell& c, int i, int k) const { return -1; }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == CELL) return m;
	return 0;
    }
    string Name () const { return "CellDoF"; }
};

class VertexEdgeFaceCellDoF : public DoF {
    int m;
public:
    VertexEdgeFaceCellDoF (int M = 1) : m(M) {}
    int NodalPoints (const cell& c) const { 
	return c.Corners() + c.Edges() + c.Faces() + 1; 
    }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i)   z[n++] = c.Edge(i);
	for (int i=0; i<c.Faces(); ++i)   z[n++] = c.Face(i);
	z[n++] = c();
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i) + c.FaceEdges(i) + 1;
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	int fc = c.FaceCorners(i);
	int fe = fc + c.FaceEdges(i);
	if (k < fc) return c.facecorner(i,k); 
	if (k < fe) return c.Corners() + c.faceedge(i,k-fc); 
	return c.Corners() + c.Edges() + i; 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return m;
	if (NODETYPE(tp) == EDGE)   return m;
	if (NODETYPE(tp) == FACE)   return m;
	if (NODETYPE(tp) == CELL)   return m;
	return 0;
    }
    string Name () const { return "VertexEdgeFaceCellDoF"; }
};

class Curl2DoF : public DoF {
    int m;
    // remapping of face corners local numbering, need to coincide the DOFs
    Point FaceCor(const cell& c, int i, int j) const {
       if (i == 0)
          switch(j) {
	    case 0: return c[1];
	    case 1: return c[0];
	    case 2: return c[3];
	    case 3: return c[2];
          }
       else return c.FaceCorner(i,j);
    }
public:
    Curl2DoF (int M = 1) : m(M) {}
    int NodalPoints (const cell& c) const { return 54; }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
        for (int i=0; i<c.Edges(); ++i) z[n++] = 0.5*(c.Edge(i)+c.EdgeCorner(i,(c.EdgeCorner(i,0)<c.EdgeCorner(i,1))));
	for (int i=0; i<c.Faces(); ++i) {
           Point fm = c.Face(i);
           Point f0 = FaceCor(c,i,0);
           Point f1 = FaceCor(c,i,1);
           z[n++] = fm + 0.25 * (2 * (f1 > f0) - 1) * (f1 - f0);
           z[n++] = fm + 0.25 * (FaceCor(c,i,3) - f0);
           z[n++] = fm - 0.25 * (FaceCor(c,i,3) - f0);
           z[n++] = fm;
        }
	for (int i=0; i<5; ++i) z[n++] = 0.5 * (c() + c.Face(i));
        z[n++] = c();
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
        z=vector<short>(NodalPoints(c),m);
    }
    int NodalPointsOnFace (const cell& c, int i) const { return 12; }
    int NodalPointsOnEdge (const cell& c, int i) const { return 2; }
    int NodalPointOnFace (const cell& c, int i, int k) const {
        int nfe = c.FaceEdges(i);
	if (k < nfe) return c.faceedge(i,k);
        k -= nfe;
        if (k < nfe) return c.Edges() + c.faceedge(i,k);
	return 2*c.Edges() + 4*i + k - nfe;
    }
    int NodalPointOnEdge (const cell& c, int i, int k) const {
       return i + 12*k;
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return 0;
	if (NODETYPE(tp) == EDGE)   return m;
	if (NODETYPE(tp) == FACE)   return m;
	if (NODETYPE(tp) == CELL)   return m;
	return 0;
    }
    string Name () const { return "Curl2DoF"; }
};

class CubicQuadDoF : public DoF {
// needs debugging
    int m;
public:
    CubicQuadDoF (int M = 1) : m(M) {}
    int NodalPoints (const cell& c) const { return 4*c.Corners(); }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	const double h = 1.0 / 3.0;
	for (int i=0; i<4; ++i)
	    for (int j=0; j<4; ++j)
		z[i+4*j] = c[Point(i*h,j*h)];
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return 4;
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	switch (i) {
	case  0: return k;
	case  1: return 3 + 4*k;
	case  2: return 12 + k;
	case  3: return 4*k;
	}
    }
    int TypeDoF (int tp) const { 
	return 0;
    }
    string Name () const { return "CubicQuadDoF"; }
};

class CubicCubeDoF : public DoF {
// needs debugging
    int m;
public:
    CubicCubeDoF (int M = 1) : m(M) {}
    int NodalPoints (const cell& c) const {
	return c.Corners() + 2*c.Edges() + 4*c.Faces() + 8; 
    }
    void NodalPoints (const cell& c, vector<Point>& z) const {
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) { z[n++] = c[i]; }
	for (int i=0; i<c.Edges(); ++i) {
	    z[n++] = 1/3*c.EdgeCorner(i,0) + 2/3*c.Edge(i);
	    z[n++] = 1/3*c.EdgeCorner(i,1) + 2/3*c.Edge(i);
	}
	for (int i=0; i<c.Faces(); ++i) {
	    z[n++] = 1/3*c.FaceCorner(i,0) + 2/3*c.Face(i);
	    z[n++] = 1/3*c.FaceCorner(i,1) + 2/3*c.Face(i);
	    z[n++] = 1/3*c.FaceCorner(i,2) + 2/3*c.Face(i);
	    z[n++] = 1/3*c.FaceCorner(i,3) + 2/3*c.Face(i);
	}
	for (int i=0; i<c.Corners(); ++i) {
	    z[n++] = 1/3*c.Corner(i) + 2/3*c();
	}
    }
    void NodalDoFs (const cell& c, vector<short>& z) const {
	z.resize(NodalPoints(c));
	for (int i=0; i<z.size(); ++i) z[i] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const {
	return c.FaceCorners(i) + 2*c.FaceEdges(i) + 4;
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
    // needs debugging here
	int fc = c.FaceCorners(i);
	int fe = fc + 2*c.FaceEdges(i);
	if (k < fc) return c.facecorner(i,k); 
	if (k < fe) {
	    int edgeno = (k-fc)/2;
	    //return c.Corners() + 2*c.faceedge(i,edgeno) + 0 oder 1;
	    return c.Corners() + 2*c.faceedge(i,edgeno);
	}
	return c.Corners() + 2*c.Edges() + 4*i + (k-fe); 
    }
    int TypeDoF (int tp) const {
	if (NODETYPE(tp) == VERTEX) return m;
	if (NODETYPE(tp) == EDGE)   return m;
	if (NODETYPE(tp) == FACE)   return m;
	if (NODETYPE(tp) == CELL)   return m;
	return 0;
    }
    string Name () const { return "CubicCubeDoF"; }
};

class TaylorHoodSerendipityDoF : public DoF {
    int m;  // actually is the dimension
public:
    TaylorHoodSerendipityDoF (int M = 2) : m(M) {}
    int NodalPoints (const cell& c) const { return c.Corners() + c.Edges(); }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Corners(); ++i) z[n++]=m+1;
	for (int i=0; i<c.Edges(); ++i) z[n++] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i)+c.FaceEdges(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	int mm = c.FaceCorners(i);
	if (k < mm) return c.facecorner(i,k); 
	return c.Corners() + c.faceedge(i,k-mm); 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return m+1;
	if (NODETYPE(tp) == EDGE) return m;
	return 0;
    }
    string Name () const { return "TaylorHoodSerendipityDoF"; }
};
class TaylorHoodQuadraticDoF : public DoF {
    int m;
public:
    TaylorHoodQuadraticDoF (int M = 2) : m(M) { }
    int NodalPoints (const cell& c) const { 
	if (m==2) return c.Corners() + c.Edges() + 1; 
	return  c.Corners() + c.Faces() +c.Edges() + 1; 
    }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
	if (m==3) { for (int i=0; i<c.Faces(); ++i)   z[n++] = c.Face(i); }
	z[n++] = c();
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Corners(); ++i) z[n++]=m+1;
	for (int i=0; i<c.Edges(); ++i) z[n++] = m;
	if (m==3) { for (int i=0; i<c.Faces(); ++i)   z[n++] = m; }
	z[n++] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	if (m==2) return c.FaceCorners(i)+c.FaceEdges(i);
	return c.FaceCorners(i)+c.FaceEdges(i) +1;
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	if (m==2) {
	    int mm = c.FaceCorners(i);
	    if (k < mm) return c.facecorner(i,k); 
	    return c.Corners() + c.faceedge(i,k-mm); 
	}
	int fc = c.FaceCorners(i);
	int fe = fc + c.FaceEdges(i);
	if (k < fc) return c.facecorner(i,k); 
	if (k < fe) return c.Corners() + c.faceedge(i,k-fc); 
	return c.Corners() + c.Edges() + i; 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return m+1;
	if (NODETYPE(tp) == EDGE) return m;
	if (m==3) { if (NODETYPE(tp) == FACE) return m; }
	if (NODETYPE(tp) == CELL) return m;
	return 0;
    }
    string Name () const { return "TaylorHoodQuadraticDoF"; }
};


class DoubleVertexDoF : public DoF {
    int dim;
    int dof1;
    int dof2;
public:
    DoubleVertexDoF (int _dim, int _dof1, int _dof2,
		     int n = 0, bool bnd = false)
	: DoF(n,bnd), 
	  dim(_dim), dof1(_dof1), dof2(_dof2) {}
    int NodalPoints (const cell& c) const { return c.Corners(); }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = dof1+dof2;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	return c.facecorner(i,k); 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return dof1+dof2;
	return 0;
    }
    string Name () const { return "DoubleVertexDoF"; }
};

class THserendipityDoF : public DoF {
    int dim;
    int dof1;
    int dof2;
public:
    THserendipityDoF (int _dim, int _dof1, int _dof2,
		      int n = 0, bool bnd = false)
	: DoF(n,bnd), 
	  dim(_dim), dof1(_dof1), dof2(_dof2) {}
    int NodalPoints (const cell& c) const { return c.Corners() + c.Edges(); }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = dof1+dof2;
	for (int i=0; i<c.Edges(); ++i) z[n++] = dof1;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i)+c.FaceEdges(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	int mm = c.FaceCorners(i);
	if (k < mm) return c.facecorner(i,k); 
	return c.Corners() + c.faceedge(i,k-mm); 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return dof1+dof2;
	if (NODETYPE(tp) == EDGE) return dof1;
	return 0;
    }
    string Name () const { return "THserendipityDoF"; }
};

class DoubleVertexEdgeDoF : public DoF {
    int dim;
    int dof1;
    int dof2;
public:
    DoubleVertexEdgeDoF (int _dim, int _dof1, int _dof2,
			 int n = 0, bool bnd = false)
	: DoF(n,bnd), 
	  dim(_dim), dof1(_dof1), dof2(_dof2) {}
    int NodalPoints (const cell& c) const { return c.Corners() + c.Edges(); }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = dof1+dof2;
	for (int i=0; i<c.Edges(); ++i) z[n++] = dof1+dof2;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i)+c.FaceEdges(i);
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	int mm = c.FaceCorners(i);
	if (k < mm) return c.facecorner(i,k); 
	return c.Corners() + c.faceedge(i,k-mm); 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return dof1+dof2;
	if (NODETYPE(tp) == EDGE) return dof1+dof2;
	return 0;
    }
    string Name () const { return "DoubleVertexEdgeDoF"; }
};

class THquadraticDoF : public DoF {
    int dim;
    int dof1;
    int dof2;
public:
    THquadraticDoF (int _dim, int _dof1, int _dof2) : 
        dim(_dim), dof1(_dof1), dof2(_dof2) {}
    int NodalPoints (const cell& c) const { 
	if (dim==2) return c.Corners() + c.Edges() + 1; 
	return c.Corners() + c.Faces() +c.Edges() + 1; 
    }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Edges(); ++i) z[n++] = c.Edge(i);
	if (dim==3) for (int i=0; i<c.Faces(); ++i) z[n++] = c.Face(i);
	z[n++] = c();
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = dof1+dof2;
	for (int i=0; i<c.Edges(); ++i) z[n++] = dof1;
	if (dim==3) 
            for (int i=0; i<c.Faces(); ++i)
                z[n++] = dof1;
	z[n++] = dof1;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	if (dim==3) return c.FaceCorners(i)+c.FaceEdges(i);
	return c.FaceCorners(i)+c.FaceEdges(i)+1;
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	if (dim==2) {
	    int mm = c.FaceCorners(i);
	    if (k < mm) return c.facecorner(i,k); 
	    return c.Corners() + c.faceedge(i,k-mm); 
	}
	int fc = c.FaceCorners(i);
	int fe = fc + c.FaceEdges(i);
	if (k < fc) return c.facecorner(i,k); 
	if (k < fe) return c.Corners() + c.faceedge(i,k-fc); 
	return c.Corners() + c.Edges() + i; 
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return dof1+dof2;
	if (NODETYPE(tp) == EDGE) return dof1;
	if (dim==3) 
            if (NODETYPE(tp) == FACE) return dof1;
	if (NODETYPE(tp) == CELL) return dof1;
	Exit("not implemented!");
    }
    string Name () const { return "THquadraticDoF"; }
};

class RT0_P1DoF : public DoF {
    int m;
public:
    RT0_P1DoF (int M=1) : m(M) { 
	mout << "in RT0DoF with  m = "<<m<<endl;
    }
    int NodalPoints (const cell& c) const { 
	return c.Corners() + c.Faces();
    }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = c[i];
	for (int i=0; i<c.Faces(); ++i)   z[n++] = c.Face(i); 
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Corners(); ++i) z[n++] = m;
	for (int i=0; i<c.Faces(); ++i)   z[n++] = m; 
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return c.FaceCorners(i) + 1;
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	if (k< c.FaceCorners(i)) return c.facecorner(i,k); 
	return c.Corners() + i;
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == VERTEX) return m;
	if (NODETYPE(tp) == FACE) return m;
	return 0;
    }
    string Name () const { return "RT0DoF"; }
};


class RT0_P0DoF : public DoF {
    int m;
public:
    RT0_P0DoF (int M=1) : m(M) { 
	mout << "in RT0DoF with  m = "<<m<<endl;
    }
    int NodalPoints (const cell& c) const { 
	return c.Faces() + 1;
    }
    void NodalPoints (const cell& c, vector<Point>& z) const { 
	z.resize(NodalPoints(c));
	int n = 0;
	for (int i=0; i<c.Faces(); ++i)   z[n++] = c.Face(i); 
	z[n++] = c();
    }
    void NodalDoFs (const cell& c, vector<short>& z) const { 
	z.resize(NodalPoints(c));
	int n=0;
	for (int i=0; i<c.Faces(); ++i)   z[n++] = m; 
	z[n++] = m;
    }
    int NodalPointsOnFace (const cell& c, int i) const { 
	return 1;
    }
    int NodalPointOnFace (const cell& c, int i, int k) const { 
	return i;
    }
    int TypeDoF (int tp) const { 
	if (NODETYPE(tp) == CELL) return m;
	if (NODETYPE(tp) == FACE) return m;
	return 0;
    }
    string Name () const { return "RT0DoF"; }
};


DoF* GetDoF (const string& disc, int m, int dim, int n = 0) {
    if (disc == "linear")      return new VertexDoF(m,n);
    if (disc == "CR_linear")   return new FaceDoF(m);
    if (disc == "cell")        return new CellDoF(m);
    if (disc == "dGscalar")    return new CellDoF(m);
    if (disc == "dGscalar_P2") return new CellDoF(m);
    if (disc == "dGvector")    return new CellDoF(m*dim);
    if (disc == "quadsp")      return new VertexEdgeFaceCellDoF(m);
    if (disc == "curl")        return new EdgeDoF(m);
    if (disc == "curl2")       return new Curl2DoF(m);
    if (disc == "div")         return new FaceDoF(m);
    if (disc == "TaylorHoodSerendipity_P2P1")
        return new TaylorHoodSerendipityDoF(dim);
    if (disc == "TaylorHoodQuadratic_P2P1")
        return new TaylorHoodQuadraticDoF(dim);
    if (dim == 2) {
	if (disc == "serendipity") return new VertexEdgeDoF(m,n);
	if (disc == "quadratic") return new VertexEdgeCellDoF(m);
    }
    if (dim == 3) {
	if (disc == "serendipity") return new VertexEdgeDoF(m,n);
	if (disc == "quadratic")   return new VertexEdgeFaceCellDoF(m);
    }

    if (disc == "EqualOrderP1P1")  return new VertexDoF(m,n);
    if (disc == "EqualOrderP1P1PPP")  return new VertexDoF(m,n);
    if (disc == "EqualOrderP2P2")  return new VertexEdgeDoF(m);

    if (disc == "DG") return new CellDoF(m);
    if (disc == "CosseratP1")
        return new DoubleVertexDoF(dim,dim,2*dim-3,n);
    if (disc == "CosseratP2")
        return new DoubleVertexEdgeDoF(dim,dim,2*dim-3,n);
    if (disc == "CosseratP2P1")
        return new THserendipityDoF(dim,dim,2*dim-3,n);
    if (disc == "THquadratic")
        return new THquadraticDoF(dim,dim,2*dim-3);

    if (disc == "RT0_P0") return new RT0_P0DoF(m);

    if (disc == "QuasiMagnetoStaticsMixed") return new VertexEdgeDoF(m,n);

    Exit(disc + " not implemented; file DoF.C\n");
}

DoF* GetBndDoF (int k, int n) { 
    if (k==-1) return new EdgeDoF(1,n,true); 
    if (k==0) return new FaceDoF(1,n,true); 
    if (k==1) return new VertexDoF(1,n,true); 
    Exit("not implemented; file DoF.C\n");
}
