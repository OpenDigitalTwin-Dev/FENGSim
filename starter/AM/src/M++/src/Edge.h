// file: Edge.h
// author: Christian Wieners
// $Header: /public/M++/src/Edge.h,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#ifndef _EDGE_H_
#define _EDGE_H_

#include "Point.h" 
 
class Edge {
    Point left;
    Point right;
    short cnt;
public:
    Edge () : left(Infty), right(Infty), cnt(0) {}
    Edge (const Point& l, const Point& r) : cnt(1) {
	if (l < r) { left = l; right = r; }
	else       { left = r; right = l; }
    }
    int n () const { return cnt; }
    void inc () { ++cnt; }
    short dec () { return --cnt; }
    const Point& Left () const { return left; }
    const Point& Right () const { return right; }
    Point operator () () const { return 0.5 * (left + right); };
    friend ostream& operator << (ostream& s, const Edge& E) {
	return s << "(" << E.left << "," << E.right << ") [" << E.cnt << "]";
    }
};
class edge : public hash_map<Point,Edge,Hash>::const_iterator {
    typedef hash_map<Point,Edge,Hash>::const_iterator Iterator;
 public:
    edge (Iterator e) : Iterator(e) {}
    const Point& operator () () const { return (*this)->first; }
    const Point& Center () const { return (*this)->first; }
    const Point& Left () const { return (*this)->second.Left(); }
    const Point& Right () const { return (*this)->second.Right(); }
    double length () const { return dist(Left(),Right()); }
    friend ostream& operator << (ostream& s, const edge& E) {
	return s << E->first << " : " << E->second << endl;
    }
};
class Edges : public hash_map<Point,Edge,Hash> {
 public:
    edge edges () const { return edge(begin()); }
    edge edges_end () const { return edge(end()); }
    edge find_edge (const Point& P) const { return edge(find(P)); }
    template <class C>
    edge find_edge (const C& c) const { return edge(find(c())); }
    void Insert (const Point& l, const Point& r) {
	Point m = 0.5 * (l + r);
	hash_map<Point,Edge,Hash>::iterator e = find(m);
	if (e == end()) (*this)[m] = Edge(l,r);
	else e->second.inc();
    }
    bool Remove (const Point& x) { 
	hash_map<Point,Edge,Hash>::iterator e = find(x);
	if (e->second.dec() > 0) return false;
	erase(e);
	return true;
    }
    const Edges& ref() const { return *this; }
};

#endif
