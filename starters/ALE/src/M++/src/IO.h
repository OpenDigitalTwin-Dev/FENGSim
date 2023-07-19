// file:    IO.h
// author:  Christian Wieners
// $Header: /public/M++/src/IO.h,v 1.5 2007-09-21 12:12:39 neuss Exp $

#ifndef _IO_H_
#define _IO_H_

#include "Parallel.h"
#include "Time.h"
#include "ctools.h"

#include <fstream>
#include <list> 
#include <map> 
#include <string> 

class M_ifstream : public ifstream {
 public:
    M_ifstream (const char* name, bool test = true) : ifstream(name) {
	Assert(PPM->master());
	if (test)
	    if (!this->is_open()) {
	        cerr << "M_ERROR cannot open file " << name << endl; 
		exit(1);
	    }
    }
};
class M_ofstream : public ofstream {
 public:
    M_ofstream () {}
    M_ofstream (const char* name) : ofstream(name) {
	Assert(PPM->master());
	if (!this->is_open()) {
	    cerr << "M_ERROR cannot open file " << name << endl; 
	    exit(1);
	}
    }
    M_ofstream (const char*, int);
    M_ofstream (const char*, int, const char*);
    M_ofstream (const char*, const char*);
    void open (const char*, const char*);
    void open (const char*);
    void open_dx (const char*);
    void open_gmv (const char*);
    void popen (const char*);
    void popen (const char*, int);
    void popen (const char*, const char*);
    void popen (const char*, int, const char*);
};
bool FileExists (const char*);
inline bool FileExists (const string& s) { return FileExists(s.c_str()); };

inline char* Number_Name (const char* name, int i) {
    static char NameBuffer[256];
    return NumberName(name,NameBuffer,i);
}

class Logging {
    Date Start;
    M_ofstream out;
 public:
    Logging ();
    ~Logging ();
    template <class S> Logging& operator<< (const S& s);
    template <class S> Logging& operator, (const S& s);
    void flush () { if (!PPM->master()) return; cout.flush(); out.flush(); }
};

template <class S> Logging& Logging::operator<< (const S& s) {
    if (!PPM->master()) return *this;
    cout << s;
    out << s;
#ifndef NDEBUG
    out.flush();
#endif
    return *this;
}

template <class S> Logging& Logging::operator, (const S& s) {
    if (!PPM->master()) return *this;
    cout << s;
    out << s;
#ifndef NDEBUG
    out.flush();
#endif
    return *this;
}

#define endl "\n"

extern Logging* logging;
extern int TimeLevel; 

template <class C> ostream& operator << (ostream& s, const list<C>& P) {
    for (typename list<C>::const_iterator p = P.begin(); p!=P.end(); ++p)
	s << *p << endl;
    return s;
}
template <class C> ostream& operator << (ostream& s,const vector<C>& P){
    for (typename vector<C>::const_iterator p = P.begin(); p!=P.end(); ++p)
	s << *p << endl;
    return s;
}
template <class P, class C, class H>
ostream& operator << (ostream& s, const map<P,C,H>& M) {
    for (typename map<P,C,H>::const_iterator m=M.begin(); m!=M.end(); ++m)
	s << m->first << ": " << m->second << endl;
    return s;
}
template <class P, class C, class H>
ostream& operator << (ostream& s, const hash_map<P,C,H>& M) {
    for (typename hash_map<P,C,H>::const_iterator m=M.begin(); m!=M.end(); ++m)
	s << m->first << ": " << m->second << endl;
    return s;
}

const int buffer_length = 512;
bool _ReadConfig (const char*, const char*, string&);
bool ReadConfig (const char* name, const char* key, 
			string& S, bool mute = false);
bool _ReadConfig (const char*, const char*, double&);
bool ReadConfig (const char* name, const char* key, 
			int& a, bool mute = false);
bool ReadConfig (const char* name, string key, 
			int& a, bool mute = false);

bool ReadConfig (const char* name, const char* key, 
			double& a, bool mute = false);

bool ReadConfig (const char* name, string key, 
			double& a, bool mute = false);


bool _ReadConfig (const char*, const char*, vector<int>&);
bool ReadConfig (const char* name, const char* key, vector<int>& a);

bool _ReadConfig (const char*, const char*, vector<double>&, int);
bool ReadConfig (const char* name, const char* key, 
			vector<double>& a, int size = -1);

bool ReadConfig (const char* name, const char* key, Point& z);

bool ReadConfig (const char* name, const char* key, bool& b);


#endif
