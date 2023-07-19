// file: Identify.h
// author: Christian Wieners
// $Header: /public/M++/src/Identify.h,v 1.5 2009-08-12 17:48:38 wieners Exp $

#ifndef _IDENTIY_H_
#define _IDENTIY_H_

#include "Point.h" 

class IdentifySet : public vector<Point> {
 public:
    IdentifySet () {};
    void Add (const Point& x);
    friend ostream& operator << (ostream& s, const IdentifySet& I);
};
class identifyset : public hash_map<Point,IdentifySet,Hash>::const_iterator {
    typedef hash_map<Point,IdentifySet,Hash>::const_iterator Iterator;
 public:
    identifyset (const Iterator& I) : Iterator(I) {}
    const Point& operator () () const { return (*this)->first; }
    const IdentifySet& operator * () const { return (*this)->second; }
    int size () const { return (*this)->second.size(); }
    Point operator [] (int i) const { return (*this)->second[i]; }
    bool master() const { return ((*this)->first < (*this)->second[0]); }
    friend ostream& operator << (ostream& s, const identifyset& I);
};
class IdentifySets : public hash_map<Point,IdentifySet,Hash> {
 public:
    identifyset identifysets () const { return identifyset(begin()); }
    identifyset identifysets_end () const { return identifyset(end()); }
    identifyset find_identifyset (const Point& z) const { 
	return identifyset(find(z)); 
    }
    void Insert (const Point& x, const Point& y);
    void Insert2 (const Point& x, const Point& y);
    void Insert (const identifyset&);
    void Append (const Point&, const identifyset&);
    void Identify (const Point& y, int mode);
    void Identify2 (const Point& y, int mode);
    const IdentifySets& ref() const { return *this; }
};

#endif
