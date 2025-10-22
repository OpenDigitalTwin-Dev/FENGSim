// file:    IO.C
// author:  Christian Wieners
// $Header: /public/M++/src/Time.C,v 1.7 2009-09-18 08:43:32 maurer Exp $

#include "Time.h"
#include "Compiler.h"
#include "Parallel.h"

#include <ctime>
#include <climits>
#include <cmath>
#include <iostream>

bool nodate = false;

void NoDate () { nodate = true; }

ostream& operator << (ostream& s, const Date& d) { 
    if (nodate) return s << endl;
    return s << ctime(&d.t);
}

ostream& operator << (ostream& s, const Time& t) { 
    int M = t.Minutes();
    double S = t.Seconds();
    char c[64];
    sprintf(c,"%5.2f",S + 0.005);
    if (M == 0) return s << c << " seconds";
    int H = t.Hours(); 
    S -= 60 * M;
    sprintf(c,"%d:%05.2f",M,S);
    if (H == 0) return s << c << " minutes";
    M -= 60 * H;
    sprintf(c,"%d:%02d:%05.2f",H,M,S);
    return s << c << " hours"; 
}

Time operator - (const Date& d2, const Date& d1) {
    const double maxclock = INT_MAX * 0.9999 / CLOCKS_PER_SEC;
    Time t;
    t.t = difftime(d2.t, d1.t);
    if (t.t > maxclock) return t;
    double d = double(d2.c - d1.c) / CLOCKS_PER_SEC;
    if (d < 0.0) d = double(INT_MAX + d2.c - d1.c) / CLOCKS_PER_SEC;
    if (abs(t.t-d) > 2.0) return t;
    t.t = d;
    return t;
}

bool operator < (const Time& t1, const Time& t2) {
    if (t1.Seconds() < t2.Seconds()) return true;
    return false;
}

void DateTime::SetMax () {elapsed = PPM->Max(elapsed);}
double DateTime::GetMax () { return elapsed = PPM->Max(elapsed); }
