// file:    Time.h
// author:  Christian Wieners
// $Header: /public/M++/src/Time.h,v 1.6 2009-09-18 08:43:32 maurer Exp $

#ifndef _TIME_H_
#define _TIME_H_

#include "Compiler.h"

#include <ctime>
#include <climits>
#include <cmath>
#include <iostream>

class Time;

class Date {
    clock_t c;
    time_t t;
 public:
    Date() { c = clock(); t = time(0); }
    friend Time operator - (const Date&, const Date&);
    friend ostream& operator << (ostream&, const Date&);
};

ostream& operator << (ostream& s, const Date& d);

class Time {
    double t;
 public:
    double Seconds () const { return t; }
    int Minutes () const { return int(t / 60.0); }
    int Hours () const { return int(t / 3600.0); }
    Time& operator += (const Time& T) { t += T.t; return *this;}
    Time& operator -= (const Time& T) { t -= T.t; return *this;}
    friend Time operator - (const Date&, const Date&);
};

ostream& operator<< (ostream& s, const Time& t);
Time operator- (const Date& d2, const Date& d1);
bool operator < (const Time&, const Time&);
void NoDate();

class DateTime {
    Date Start;
    double elapsed;
    string name;
 public:
    DateTime () {}
    DateTime (string N) : name(N) {}
    DateTime (const DateTime& DT) : 
	Start(DT.Start), elapsed(DT.elapsed),name(DT.name) {}
    void SetDate () { Start = Date(); }
    void AddTime() { elapsed += (Date() - Start).Seconds(); }
    double GetTime () const { return elapsed; }
    void SetName (string N) { name = N; }
    void ResetTime () { elapsed = 0; }
    string GetName () const { return name; }
    void SetMax();
    double GetMax ();
};

#endif
