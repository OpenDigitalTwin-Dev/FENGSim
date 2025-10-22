// file: Overlap.h
// author: Christian Wieners
// $Header: /public/M++/src/Overlap.h,v 1.1 2008-06-06 09:53:26 wieners Exp $

#ifndef _OVERLAP_H_
#define _OVERLAP_H_

#include "Cell.h" 

class OverlapCells : public hash_map<Point,Cell*,Hash> {
 public:
    cell overlap () const { return cell(begin()); }
    cell overlap_end () const { return cell(end()); }
    cell find_overlap_cell (const Point& z) const { return cell(find(z)); }
    int psize () const { return PPM->Sum(int(size())); }
    cell Insert (Cell* c) { 
	Point z = c->Center();
	(*this)[z] = c; 
        return find_overlap_cell(z);
    }
    cell InsertOverlap (CELLTYPE type, int sd, const vector<Point>& z) {
	Cell* C = CreateCell (type,sd,z);
	return Insert(C);
    }
    void Remove (const Point& x) { 
	hash_map<Point,Cell*,Hash>::iterator c = find(x);
	delete c->second;
	erase(c);
    }
    const OverlapCells& ref() const { return *this; }
    OverlapCells& ref() { return *this; }
};

#endif
