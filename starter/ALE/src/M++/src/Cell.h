// file: Cell.h
// author: Christian Wieners
// $Header: /public/M++/src/Cell.h,v 1.11 2008-10-06 18:53:19 mueller Exp $

#ifndef _CELL_H_
#define _CELL_H_

#include "Point.h" 
#include "Shape.h" 
 
#include <vector>

enum CELLTYPE { NONE = 0, 
		INTERVAL = 122,
		TRIANGLE = 233, 
		QUADRILATERAL = 244, QUADRILATERAL2 = 248, 
		TETRAHEDRON = 344, 
		PYRAMID = 355, 
		PRISM = 366, 
		HEXAHEDRON = 388, HEXAHEDRON20 = 3820, HEXAHEDRON27 = 3827};

class Rule {
    CELLTYPE tp;
 public:
    vector<short> node;
    vector<short> face;
    Rule (CELLTYPE TP, int n, const short* C) : tp(TP) {
	node.resize(n);
	for (int j=0; j<n; ++j) node[j] = C[j];
    }
    Rule (const Rule& R) : tp(R.tp), node(R.node), face(R.face) {}
    Rule () : tp(NONE) {}
    CELLTYPE type() const { return tp; }
    void operator () (const vector<Point>& z, vector<Point>& x) const {
	x.resize(node.size());
	for (int j=0; j<node.size(); ++j) x[j] = z[node[j]];
    }
    friend ostream& operator << (ostream& s, const Rule& R) {
	s << "z "; for (int i=0; i<R.node.size(); ++i) s << R.node[i] << " ";
	s << "f "; for (int i=0; i<R.face.size(); ++i) s << R.face[i] << " ";
	return s << "tp " << R.type();
    }
};
class Rules : public vector<Rule> {
 public:
    Rules (int n, const Rule* R) : vector<Rule>(n) {
	for (int i=0; i<n; ++i) (*this)[i] = R[i];
    }
};

class Cell : public vector<Point> {
    short subdomain;
 protected:
    Cell (const vector<Point>& x, short sd) : 
	vector<Point>(x), subdomain(sd) {}
    const Point& z (int i) const {return (*this).vector<Point>::operator[](i);}
 public:
    virtual Point operator () () const = 0;
    Point Center () const { return (*this)(); }
    virtual CELLTYPE Type () const = 0;
    virtual CELLTYPE ReferenceType () const = 0;
    virtual const Cell* ReferenceCell () const = 0;
    short Subdomain () const { return subdomain; }
    void SetSubdomain (short sd) { subdomain = sd; } // jiping changed
    virtual int Corners () const = 0;
    const Point& operator [] (int i) const { 
	return (*this).vector<Point>::operator[](i); }
    const Point& Corner (int i) const { return (*this)[i]; }
    virtual const Point& LocalCorner (int) const = 0;
    virtual const Point& LocalFaceNormal (int) const = 0;
    virtual Point LocalToGlobal (const Point&) const = 0;
    virtual Transformation GetTransformation (const Point&) const = 0;
    virtual int Edges () const = 0;
    virtual Point Edge (int) const = 0;
    virtual const Point& LocalEdge (int) const = 0;
    virtual const Point& LocalFace (int) const = 0;
    virtual const Point& LocalCenter () const = 0;
    virtual const Point& EdgeCorner (int, int) const = 0;
    virtual short edgecorner (int, int) const = 0;
    virtual int Faces () const = 0;
    virtual Point Face (int) const = 0;
    virtual short FarFace (int) const { return -1; }
    virtual short MiddleEdge (int i,int j) const { return -1; }
    virtual bool faceedgecircuit (int i) const { return false; }
    virtual short FaceCorners (int) const = 0;
    virtual const Point& FaceCorner (int, int) const = 0;
    virtual short facecorner (int, int) const = 0;
    virtual short FaceEdges (int) const = 0;
    virtual short faceedge (int, int) const = 0;
    Point FaceEdge (int i, int j) const { return Edge(faceedge(i,j)); }
    short faceedgecorner (int i, int j, int k) const {
	return edgecorner(faceedge(i,j),k); }
    const Point& FaceEdgeCorner (int i, int j, int k) const {
	return EdgeCorner(faceedge(i,j),k); }
    virtual double LocalFaceArea (int face) const = 0;
    virtual const Rules& Refine (vector<Point>&) const = 0;
    virtual int dim () const = 0;
    virtual bool plane () const = 0;
    virtual ~Cell () {}
};
inline ostream& operator << (ostream& s, const Cell& C) {
    for (int i=0; i<C.size(); ++i) s << C[i] << "|";
    return s << " tp " << int(C.Type()) << " sd " << C.Subdomain();
}
inline ostream& operator << (ostream& s, const Cell* C) { return s << *C; }
class cell : public hash_map<Point,Cell*,Hash>::const_iterator {
    typedef hash_map<Point,Cell*,Hash>::const_iterator Iterator;
 public:
    cell () {}
    cell (Iterator c) : Iterator(c) {}
    int size () const { return (*this)->second->size(); }
    const Point& operator () () const { return (*this)->first; }
    const Point& Center () const { return (*this)->first; }
    const Cell& operator * () const { return *((*this)->second); }
    CELLTYPE Type () const { return (*this)->second->Type(); }
    CELLTYPE ReferenceType () const { return (*this)->second->ReferenceType();}
    const Cell* ReferenceCell () { return (*this)->second->ReferenceCell(); }
    short Subdomain () const { return (*this)->second->Subdomain(); }
    int Corners () const { return (*this)->second->Corners(); }
    const Point& Corner (int i) const { return (*this)->second->Corner(i); }
    const Point& LocalCorner (int i) const { 
	return (*this)->second->LocalCorner(i); }
    const Point& LocalEdge (int i) const {
	return (*this)->second->LocalEdge(i); }
    const Point& LocalFace (int i) const {
	return (*this)->second->LocalFace(i); }
    const Point& LocalCenter () const { return (*this)->second->LocalCenter();}
    const Point& LocalFaceNormal (int i) const { 
	return (*this)->second->LocalFaceNormal(i); }
    Point LocalToGlobal (const Point& z) const {
		return (*this)->second->LocalToGlobal(z); }
    Point operator [] (const Point& z) const { 
		return (*this)->second->LocalToGlobal(z); }
    Transformation GetTransformation (const Point& z) const { 
		return (*this)->second->GetTransformation(z); }
    const Point& operator [] (int i) const {return (*this)->second->Corner(i);}
    int Edges () const { return (*this)->second->Edges(); }
    Point Edge (int i) const { return (*this)->second->Edge(i); }
    const Point& EdgeCorner (int i, int j) const { 
	return (*this)->second->EdgeCorner(i,j); }
    short edgecorner (int i, int j) const { 
	return (*this)->second->edgecorner(i,j); }
    int Faces () const { return (*this)->second->Faces(); }
    Point Face (int i) const { return (*this)->second->Face(i); }
    short FarFace (int i) const { return (*this)->second->FarFace(i); }
    short MiddleEdge (int i,int j) const { return (*this)->second->MiddleEdge(i,j); }
    bool faceedgecircuit (int i) const { return (*this)->second->faceedgecircuit(i); }
    short FaceCorners (int i) const { return (*this)->second->FaceCorners(i); }
    const Point& FaceCorner (int i, int j) const {
	return (*this)->second->FaceCorner(i,j); }
    short facecorner (int i, int j) const {
	return (*this)->second->facecorner(i,j); }
    short FaceEdges (int i) const { return (*this)->second->FaceEdges(i); }
    const Point& FaceEdgeCorner (int i, int j, int k) const {
	return (*this)->second->FaceEdgeCorner(i,j,k); }
    Point FaceEdge (int i, int j) const {
	return (*this)->second->FaceEdge(i,j); }
    short faceedge (int i, int j) const {
	return (*this)->second->faceedge(i,j); }
    short faceedgecorner (int i, int j, int k) const {
	return (*this)->second->faceedgecorner(i,j,k); }
    short facecorner (const Point& z) const {
	for (int i=0; i<Faces(); ++i)
	    if (z == Face(i)) return i;
	return -1;
    }
    int FaceId (const Point& z) const { 
	for (int i=0; i<Faces(); ++i) if (Face(i) == z) return i; 
    }
    double LocalFaceArea (int face) const { 
	return (*this)->second->LocalFaceArea(face); }
    const Rules& Refine (vector<Point>& z) const {
	return (*this)->second->Refine(z); }
    int dim () const { return (*this)->second->dim(); }
    bool plane () const { return (*this)->second->plane(); }
};
inline ostream& operator << (ostream& s, const cell& c) {
    return s << c->first << " : " << *(c->second);
}
class Cells : public hash_map<Point,Cell*,Hash> {
public:
    cell cells () const { return cell(begin()); }
    cell cells_end () const { return cell(end()); }
    cell find_cell (const Point& z) const { return cell(find(z)); }
    int psize () const { return PPM->Sum(int(size())); }
    cell Insert (Cell* c) { 
		Point z = c->Center();
		(*this)[z] = c; 
        return find_cell(z);
    }
    CELLTYPE Type (int, const vector<Point>&);
    cell Insert (CELLTYPE, int, const vector<Point>&);
    void Remove (const Point& x) { 
		hash_map<Point,Cell*,Hash>::iterator c = find(x);
		delete c->second;
		erase(c);
    }
    const Cells& ref() const { return *this; }
    Cells& ref() { return *this; }
};

Cell* CreateCell (CELLTYPE, int, const vector<Point>&);

#endif
