// file:    Debug.h
// author:  Christian Wieners
// $Header: /public/M++/src/Debug.h,v 1.1.1.1 2007-02-19 15:55:20 wieners Exp $

#ifndef _DEBUG_H_
#define _DEBUG_H_

#include "Compiler.h"

#include <iostream>
#include <string>
#include <typeinfo>
#include <cassert>

template <class C> 
inline const char* Type (const C& c) { return typeid(c).name(); }
template <class C> 
inline const char* Type (const C* c) { return typeid(*c).name(); }

class Assertion {
 public:
    Assertion () {}
    void operator () (bool b, const char* s, const char* f, int L) const { 
	if (b) return;
	cerr << "assert: " << s 
	     << " failed in "  << f << " on line " << L << endl;
	assert(0);
    }
    void operator () (bool b, string s, const char* f, int L) const { 
	if (b) return;
	cerr << "assert: " << s 
	     << " failed in "  << f << " on line " << L << endl;
	assert(0);
    }
    void operator () (bool b, const char* s, int n,
		      const char* f, int L) const { 
	if (b) return;
	cerr << "assert: " << s << n 
	     << " failed in "  << f << " on line " << L << endl;
	assert(0);
    }
};
const Assertion assertion;
inline void Exit (const char* s ="") { assertion(false,s,__FILE__,__LINE__); }
inline void Exit (string s) { assertion(false,s,__FILE__,__LINE__); }
inline void Exit (const char* s,int n){assertion(false,s,n,__FILE__,__LINE__);}
extern int DebugLevel; 

#define pout        cout << PPM->proc() << ": "
#define lout        *logging 
//#define lout        if (me == 0) cout
#define mout        lout
#define Vout(i)     if (verbose>=i) lout
#define tout(i)     if (TimeLevel>i) lout

#ifdef NDEBUG

#define Assert(s)   
#define CheckMem(a) 
#define dout(i)      if (0) cout 
#define dpout(i)      if (0) cout 
//#define tout(i)      if (0) cout 
#define vout(i)     if (0) cout 
#define SOUT(s)
#define MOUT(s)
#define POUT(s)
#define dSynchronize if (0) PPM->Synchronize  

#else

#define Assert(s)   assertion(s,__STRING(s),__FILE__,__LINE__)
#define CheckMem(a) if (!(a))assertion(false,"out of memory",__FILE__,__LINE__)
#define dout(i)     if (DebugLevel>=i) lout 
#define dpout(i)    if (DebugLevel>=i) pout
#define vout(i)     if (verbose>=i) lout
#define SOUT(s)     lout << __STRING(s) << " = " << s << "\n"
#define MOUT(s)     mout << __STRING(s) << " = " << s << "\n"
#define POUT(s)     pout << __STRING(s) << " = " << s << "\n"; cout.flush()
#define dSynchronize PPM->Synchronize  

#endif

#endif
