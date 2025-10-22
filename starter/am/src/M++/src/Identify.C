// file: Identify.C
// author: Christian Wieners
// $Header: /public/M++/src/Identify.C,v 1.3 2009-08-12 17:48:38 wieners Exp $

#include "Identify.h" 
#include "Debug.h" 
#include "Parallel.h" 

void IdentifySet::Add (const Point& x) { 
    int m = size();
    if (m==0) {
        resize(1); 
        (*this)[0] = x;
        return;
    }
    for (short i=0; i<m; ++i) 
        if ((*this)[i] == x) return;
    resize(m+1); 
    if (x < (*this)[0]) {
        (*this)[m] = (*this)[0];
        (*this)[0] = x;
    }
    else (*this)[m] = x; 
}

ostream& operator << (ostream& s, const IdentifySet& I) {
    for (short i=0; i<I.size(); ++i) s << " " << I[i];
    return s;
}


ostream& operator << (ostream& s, const identifyset& I) {
    return s << I() << " : " << *I << endl;
}

void IdentifySets::Insert (const Point& x, const Point& y) {
    (*this)[x];
    hash_map<Point,IdentifySet,Hash>::iterator i = find(x);
    if (i != end())
	i->second.Add(y);
//    (*this)[y];
//    i = find(y);
//    i->second.Add(x);
}

void IdentifySets::Insert2 (const Point& x, const Point& y) {

    pout << " x " << x << endl;


    hash_map<Point,IdentifySet,Hash>::iterator j = find(x);
    if (j != end())
	IdentifySet IS = (*this)[x];

    IdentifySet IS;

    int n = size();

    if (j == end())
	this->insert(make_pair(x,IS));

    pout << "size " << n << " " << size() << endl; 

    return;



    hash_map<Point,IdentifySet,Hash>::iterator i = find(x);
    if (i != end())
	i->second.Add(y);
//    (*this)[y];
//    i = find(y);
//    i->second.Add(x);
}

void IdentifySets::Insert (const identifyset& is) { Append(is(),is); }

void IdentifySets::Append (const Point& x, const identifyset& is) {
    (*this)[x];
    hash_map<Point,IdentifySet,Hash>::iterator i = find(x);
    if (is() != x) 
	i->second.Add(is());
    for (int j=0; j<is.size(); ++j)
	if (is[j] != x)
	    i->second.Add(is[j]);
}

void IdentifySets::Identify (const Point& y, int mode) {
    if (find(y) != end()) return;
    if (mode == 999) {
        Point X1 (1,0);
        Point X2 (0,1);
        if (y[0]          <GeometricTolerance) Insert(y,y+X1);
        if (abs(y[0] - 1) <GeometricTolerance) Insert(y,y-X1);
        if (y[1]          <GeometricTolerance) Insert(y,y+X2);
        if (abs(y[1] - 1) <GeometricTolerance) Insert(y,y-X2);
        if (y[0]          <GeometricTolerance
            && y[1]       <GeometricTolerance) Insert(y,y+X1+X2);
        if (abs(y[0] - 1) <GeometricTolerance 
            && y[1]       <GeometricTolerance) Insert(y,y-X1+X2);
        if (y[0]          <GeometricTolerance 
            && abs(y[1]-1)<GeometricTolerance) Insert(y,y+X1-X2);
        if (abs(y[0] - 1) <GeometricTolerance 
            && abs(y[1]-1)<GeometricTolerance) Insert(y,y-X1-X2);
    } else if (mode == 9999) {
        Point s[3];
        int f[3]={0,0,0};
	    
        s[0]=Point(1,0,0); s[1]=Point(0,1,0); s[2]=Point(0,0,1);
        for (int c=0;c<3;++c)
            for (int t=0;t<2;++t)
                if (abs(y[c]-t)<GeometricTolerance) {
                    f[c]=(t==0)?1:-1;
                    Insert(y,y+f[c]*s[c]);
                };
        for (int c1=1;c1<3;++c1)
            for (int c2=0;c2<c1;++c2)
                if ((f[c1]!=0)&&(f[c2]!=0)) 
                    Insert(y,y+f[c1]*s[c1]+f[c2]*s[c2]);
        if ((f[0]!=0)&&(f[1]!=0)&&(f[2]!=0)) 
            Insert(y,y+f[0]*s[0]+f[1]*s[1]+f[2]*s[2]);
    }
}

void IdentifySets::Identify2 (const Point& y, int mode) {
    if (find(y) != end()) return;

    pout << " y " << y << endl;

        Point s[3];
        int f[3]={0,0,0};
	    
        s[0]=Point(1,0,0); s[1]=Point(0,1,0); s[2]=Point(0,0,1);
        for (int c=0;c<3;++c)
            for (int t=0;t<2;++t)
                if (abs(y[c]-t)<GeometricTolerance) {
                    f[c]=(t==0)?1:-1;
                    pout  << " ident2 " << y+f[c]*s[c] << endl;

                    Insert2(y,y+f[c]*s[c]);


                };

	return;


        for (int c1=1;c1<3;++c1)
            for (int c2=0;c2<c1;++c2)
                if ((f[c1]!=0)&&(f[c2]!=0)) 
                    Insert(y,y+f[c1]*s[c1]+f[c2]*s[c2]);
        if ((f[0]!=0)&&(f[1]!=0)&&(f[2]!=0)) 
            Insert(y,y+f[0]*s[0]+f[1]*s[1]+f[2]*s[2]);

}
