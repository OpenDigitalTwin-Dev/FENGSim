// file: Face.h
// author: Christian Wieners
// $Header: /public/M++/src/Face.h,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#ifndef _FACE_H_
#define _FACE_H_

#include "Point.h" 

class Face {
    Point left;
    Point right;
public:
    Face () : left(Infty), right(Infty) {}
    Face (const Point& l) : left(l), right(Infty) {}
    Face (const Point& l, const Point& r) {
	if (l < r) { left = l; right = r; }
	else       { left = r; right = l; }
    }
    const Point& Left () const { return left; }
    const Point& Right () const { return right; }
    friend ostream& operator << (ostream& s, const Face& F) {
	return s << "(" << F.left << "," << F.right << ")";
    }
};
class face : public hash_map<Point,Face,Hash>::const_iterator {
    typedef hash_map<Point,Face,Hash>::const_iterator Iterator;
 public:
    face () {}
    face (Iterator f) : Iterator(f) {}
    const Point& operator () () const { return (*this)->first; }
    const Face& operator * () const { return (*this)->second; }
    const Point& Left () const { return (*this)->second.Left(); }
    const Point& Right () const { return (*this)->second.Right(); }
    friend ostream& operator << (ostream& s, const face& F) {
	return s << F->first << " : " << F->second << endl;
    }
};
class Faces : public hash_map<Point,Face,Hash> {
 public:
    face faces () const { return face(begin()); }
    face faces_end () const { return face(end()); }
    face find_face (const Point& z) const { return face(find(z)); }
    void Insert (const Point& z, const Point& c) {
	face f = find_face(z);
	if (f == faces_end()) (*this)[z] = Face(c);
	else if (f.Right() == Infty)
	    if (f.Left() != c)
		(*this)[z] = Face(f.Left(),c);
    }
    void Insert (const Point& f, const Face& F) { (*this)[f] = F; }
    void Remove (const Point& x) { erase(x); }
    const Faces& ref() const { return *this; }
};

class BoundaryFace {
    short part;
public:
    BoundaryFace (int p = 0) { part = p; }
    int Part () const { return part; }
    void SetPart (short p) { part = p; }
    friend ostream& operator << (ostream& s, const BoundaryFace& F) {
	return s << F.part;
    }
};
class bnd_face : public hash_map<Point,BoundaryFace,Hash>::const_iterator {
    typedef hash_map<Point,BoundaryFace,Hash>::const_iterator Iterator;
 public:
    bnd_face () {}
    bnd_face (Iterator b) : Iterator(b) {}
    const Point& operator () () const { return (*this)->first; }
    const BoundaryFace& operator * () const { return (*this)->second; }
    int Part () const { return (*this)->second.Part(); }
};
inline ostream& operator << (ostream& s, const bnd_face& b) {
    return s << b() << " : " << *b << endl;
}
class BoundaryFaces : public hash_map<Point,BoundaryFace,Hash> {
 public:
    bnd_face bnd_faces () const { return bnd_face(begin()); }
    bnd_face bnd_faces_end () const { return bnd_face(end()); }
    bnd_face find_bnd_face (const Point& z) const { 
	return bnd_face(find(z)); 
    }
    int Part (const Point& z) const { 
	bnd_face b = find_bnd_face(z);
	if (b == bnd_faces_end()) return -1;
	return b.Part();
    }
    void Insert (const Point& z, int part) {(*this)[z] = BoundaryFace(part); }
    void Insert (const Point& z, const BoundaryFace& B) { (*this)[z] = B; }
    void Remove (const Point& x) { erase(x); }
    const BoundaryFaces& ref() const { return *this; }
    BoundaryFaces& ref() { return *this; }
};

#endif
