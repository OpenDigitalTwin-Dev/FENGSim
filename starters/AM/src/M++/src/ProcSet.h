// file:    ProcSet.h
// author:  Christian Wieners
// $Header: /public/M++/src/ProcSet.h,v 1.7 2009-09-16 13:36:45 maurer Exp $

#ifndef _PROCSET_H_
#define _PROCSET_H_

#include "Parallel.h" 

class ProcSet : public vector<short> {
 public:
    ProcSet () {};
    ProcSet (int q) : vector<short>(1) { (*this)[0] = q; }
    ProcSet (int p, int q) : vector<short>(2) { (*this)[0]=p; (*this)[1]=q; }
    void Add (short q) { 
	int m = size();
	if (m == 0) {
	    resize(1); 
	    (*this)[0] = q;
	}
	for (short i=0; i<m; ++i) 
	    if ((*this)[i] == q) return;
	resize(m+1); 
	if (q < (*this)[0]) {
	    (*this)[m] = (*this)[0];
	    (*this)[0] = q;
	}
	else (*this)[m] = q; 
    }
    ProcSet (const ProcSet& P) : vector<short>(0) {
        for (int i=0; i<P.size(); ++i) Add(P[i]);
    }
    void Append (short q) { 
	int m = size();
	for (short i=0; i<m; ++i) 
	    if ((*this)[i] == q) return;
	resize(m+1); 
	(*this)[m] = q; 
    }
    void Append (const ProcSet& PS) { 
	for (int i=0; i<PS.size(); ++i) 
	    Append(PS[i]);
    }
    void Add(const ProcSet& PS) {
        for (int i=0; i<PS.size(); ++i)
            Add(PS[i]);
    }
    bool Erase () {
    	if ((size() == 1) && ((*this)[0] == PPM->proc())) return true;
	for (int i=0; i<size(); ++i)
	    if ((*this)[i] == PPM->proc()) return false;
	return false;
    }
    void erase (short q) { 
	int m = size();
	short i=0;
	for (;i<m; ++i) 
	    if ((*this)[i] == q) break;
	if (i == m) return;
	for (++i;i<m; ++i) 
	    (*this)[i-1] = (*this)[i];
	resize(m-1); 
    }
    short master () const { return (*this)[0]; }
    void SetMaster (int q) { 
	if ((*this)[0] == q) return;
	for (short i=1; i<size(); ++i) 
	    if ((*this)[i] == q) {
		(*this)[i] = (*this)[0];
		(*this)[0] = q;
		return;
	    }
	Exit("not valid proc id");
    }

    bool equalProcSet(const ProcSet& P) {
        if (P.size() != (*this).size()) return false;
        for (int i=0; i<(*this).size(); ++i) {
            bool equal = false;
            for (int j=0; j<P.size(); ++j)
                if ((*this)[i] == P[j]) {
                   equal = true; 
                   break;
                }
            if (!equal) return false;
        }
        return true;
    }

    bool existselementof(const ProcSet& Q) {
        for (int i=0; i<(*this).size(); ++i)
            for (int j=0; j<Q.size(); ++j)
                if ((*this)[i] == Q[j]) return true;
        return false;
    }

    bool subset(const ProcSet& Q) {
        for (int i=0; i<(*this).size(); ++i) {
            bool goon = false;
            for (int j=0; j<Q.size(); ++j)
                if ((*this)[i] == Q[j]) {goon = true;break;}
            if (goon) continue;
            return false;
        }
        return true;
    }

};
inline ostream& operator << (ostream& s, const ProcSet& P) {
    for (short i=0; i<P.size(); ++i) s << " " << P[i];
    return s;
} 

class procset : public hash_map<Point,ProcSet,Hash>::const_iterator {
    typedef hash_map<Point,ProcSet,Hash>::const_iterator Iterator;
 public:
    procset (Iterator p) : Iterator(p) {}
    const Point& operator () () const { return (*this)->first; }
    const ProcSet& operator * () const { return (*this)->second; }
    int size () const { return (*this)->second.size(); }
    short operator [] (int i) const { return (*this)->second[i]; }
    short master () const { return (*this)->second[0]; }
    bool in (int q) const { 
	for (int i=0; i<size(); ++i) 
	    if ((*this)->second[i] == q) 
		return true;
	return false;
    }
};
inline bool master (const procset& p) { return (p.master() == PPM->proc()); }
inline ostream& operator << (ostream& s, const procset& p) {
    return s << p() << " : " << *p;
}
class ProcSets : public hash_map<Point,ProcSet,Hash> {
 public:
    procset procsets () const { return procset(begin()); }
    procset procsets_end () const { return procset(end()); }
    procset find_procset (const Point& z) const { return procset(find(z)); }
    template <class C> 
	procset find_procset (const C& c) const { return procset(find(c())); }
    void Copy (const procset& p, const Point& z) { (*this)[z] = *p; }
    void Copy (const procset& p) { (*this)[p()] = *p; }
    void Add (const Point& z, int q) {
	hash_map<Point,ProcSet,Hash>::iterator p = find(z);
	if (p == end()) (*this)[z] = ProcSet(q);
	else            p->second.Add(q);
    }
    void Add (const Point& z, const procset& q) { 
	hash_map<Point,ProcSet,Hash>::iterator p = find(z);
	if (p == end()) (*this)[z] = *q;
	else  
	    for (int i=0; i<q.size(); ++i) 
		p->second.Add(q[i]); 
    }
    void AddInfty () {
	hash_map<Point,ProcSet,Hash>::iterator p = find(Infty);
	if (p == end()) (*this)[Infty] = ProcSet(0);
	for (int q=1; q<PPM->size(); ++q) 
	    p->second.Add(q);
    }
    void Append (const Point& z, int q) {
	hash_map<Point,ProcSet,Hash>::iterator p = find(z);
	if (p == end()) (*this)[z] = ProcSet(PPM->proc(),q);
	else            p->second.Append(q);
    }
    void Append (const Point& z, const procset& q) { 
	hash_map<Point,ProcSet,Hash>::iterator p = find(z);
	if (p == end()) (*this)[z] = *q;
	else  
	    for (int i=0; i<q.size(); ++i) 
		p->second.Append(q[i]); 
    }
    void Insert (const Point& z, const ProcSet& PS) { 
	hash_map<Point,ProcSet,Hash>::iterator p = find(z);
	if (p == end()) (*this)[z] = PS; 
	else            p->second.Append(PS);
    }
    void Remove (const Point& z) { erase(z); }
    bool on (hash_map<Point,ProcSet,Hash>::iterator p, int q) {
	for (int i=0; i<p->second.size(); ++i) 
	    if (p->second[i] == q) 
		return true;
	return false;
    }
    void Remove (const Point& z, int q) {
	hash_map<Point,ProcSet,Hash>::iterator p = find(z);
	if (p == end()) return;
	if (!on(p,q)) return;
	p->second.erase(q);
    }
    void RemoveSingle () {
	for (hash_map<Point,ProcSet,Hash>::iterator p = begin(); p!=end();) {
	    hash_map<Point,ProcSet,Hash>::iterator q = p++;
	    if (q->second.size() == 1) erase(q);
	}
    }
    void Clean () {
	for (hash_map<Point,ProcSet,Hash>::iterator p = begin(); p!=end();) {
	    hash_map<Point,ProcSet,Hash>::iterator q = p++;
	    if (!on(q,PPM->proc())) erase(q);
	}
    }
    bool master (const Point& z) const { 
	procset p = find_procset(z);
	if (p == procsets_end()) return true;
	return (p.master() == PPM->proc()); 
    }
    const ProcSets& ref() const { return *this; }
    ProcSets& ref() { return *this; }
};

#endif
