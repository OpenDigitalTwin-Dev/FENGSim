// file: Vertex.h
// author: Christian Wieners
// $Header: /public/M++/src/Vertex.h,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#ifndef _VERTEX_H_
#define _VERTEX_H_

#include "Point.h" 

class Vertex {
    short part;
    short cnt;
public:
    Vertex () { part = 0; cnt = 0; }
    Vertex (int p) { part = p; cnt = 1; }
    Vertex (const Vertex& v) { part = v.part; cnt = v.cnt; }
    int n () const { return cnt; }
    void inc () { ++cnt; }
    short dec () { return --cnt; }
    void SetPart (short p) { part = p; }
    friend ostream& operator << (ostream& s, const Vertex& V) {
	return s<< V.part << " [" << V.cnt << "]"; 
    }
};
class vertex : public hash_map<Point,Vertex,Hash>::const_iterator {
    typedef hash_map<Point,Vertex,Hash>::const_iterator Iterator;
 public:
    vertex (Iterator v) : Iterator(v) {}
    const Point& operator () () const { return (*this)->first; }
    double operator [] (int i) const { return (*this)->first[i]; }
};
inline ostream& operator << (ostream& s, const vertex& v) {
    return s << v->first << " : " << v->second << endl;
}

class Vertices : public hash_map<Point,Vertex,Hash> {
public:
    vertex vertices () const { return vertex(begin()); }
    vertex vertices_end () const { return vertex(end()); }
    vertex find_vertex (const Point& x) const { return vertex(find(x)); }
    void Insert (const Point& x) {
	hash_map<Point,Vertex,Hash>::iterator v = find(x);
	if (v == end()) 
	    (*this)[x] = Vertex(0);
	else
	    v->second.inc();
    }
    bool Remove (const Point& x) { 
	hash_map<Point,Vertex,Hash>::iterator v = find(x);
	if (v->second.dec() > 0) return false;
	erase(v);
	return true;
    }
    const Vertices& ref() const { return *this; }
};

#endif
