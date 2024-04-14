// file:    IO.C
// author:  Christian Wieners
// $Header: /public/M++/src/IO.C,v 1.7 2009-08-20 18:11:18 wieners Exp $

#include "IO.h"

int TimeLevel = 0; 
int DebugLevel = 0; 
Logging* logging;

#include <iostream>

const string MError("M_ERROR cannot open file ");
char namebuffer[64];
char NameBuffer[64];
M_ofstream::M_ofstream (const char* name, int i) : 
    ofstream(NumberName(name,namebuffer,i)) {
    Assert(PPM->master());
    if (!this->is_open()) Exit(MError + name);
}
M_ofstream::M_ofstream (const char* name, int i, const char* ext) : 
    ofstream(NumberName(name,namebuffer,i,ext)) {
    Assert(PPM->master());
    if (!this->is_open()) Exit(MError + name);
}
bool FileExists (const char* name) {
    Assert(PPM->master());
    ifstream file(name);
    if (!file.is_open()) return false;
    return true;
}
const char* CheckMode (const char* name, const char* mode) {
    if (strcmp(mode,"rename") == 0)
	if (FileExists(name))
	    Rename(name);
    return name;
}
M_ofstream::M_ofstream (const char* name, const char* mode) : 
    ofstream(CheckMode(name,mode)) {
    Assert(PPM->master());
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::open (const char* name, const char* mode) {
    this->ofstream::open(CheckMode(name,mode));
    Assert(PPM->master());
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::open (const char* name) {
    this->ofstream::open(name);
    Assert(PPM->master());
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::open_dx (const char* name) {
    Assert(PPM->master());
    string Name(name);
    Name += ".dx"; 
    this->ofstream::open(Name.c_str());
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::open_gmv (const char* name) {
    Assert(PPM->master());
    string Name(name);
    Name += ".gmv"; 
    this->ofstream::open(Name.c_str());
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::popen (const char* name) {
    this->ofstream::open(pNumberName(name,namebuffer));
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::popen (const char* name, int i) {
    this->ofstream::open(pNumberName(name,namebuffer,i));
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::popen (const char* name, const char* ext) {
    this->ofstream::open(pNumberName(name,namebuffer,ext));
    if (!this->is_open()) Exit(MError + name);
}
void M_ofstream::popen (const char* name, int i, const char* ext) {
    this->ofstream::open(pNumberName(name,namebuffer,i,ext));
    if (!this->is_open()) Exit(MError + name);
}

Logging::Logging () { 
    int p = cout.precision(); 
    int e = 4;
    if (PPM->master()) {
	string name = "log/log"; 
	ReadConfig(Settings,"logfile",name,true);   
	out.open(name.c_str(),"rename"); 
	out  << "start program on " << PPM->size() << " procs at " <<Start;
	cout << "start program on " << PPM->size() << " procs at " <<Start;
	ReadConfig(Settings,"precision",p,true);   
	if (p != cout.precision())
	    ReadConfig(Settings,"precision_ext",e,true);   
    }
    PPM->Broadcast(p);
    PPM->Broadcast(e);
    cout.precision(p); 
    out.precision(p);
    SetPointPrecision(p,e);
}
Logging::~Logging () { 
    if (!PPM->master()) return; 
    Date End;
    out  << "end program after " << End - Start 
	 << " on " << PPM->size() << " procs at " << End;
    cout << "end program after " << End - Start 
	 << " on " << PPM->size() << " procs at " << End;
}

inline void clear (char* c) {
    for (int i=0; i<buffer_length; ++i) {
	if (c[i] == 0) break;
	if (c[i] == ';') {
	    c[i] = 0;
	    return;
	}
    }
}

bool CheckLoadConf (const char* L, char* c) {
    if (L[0] == '#') return false;
    string s0 = "loadconf =%s";
    if (sscanf(L,s0.c_str(),c) == 1) {
	clear(c);
	return true;
    }
    s0 = "Loadconf =%s";
    if (sscanf(L,s0.c_str(),c) == 1) {
	clear(c);
	return true;
    }
    s0 = "Loadconf=%s";
    if (sscanf(L,s0.c_str(),c) == 1) {
	clear(c);
	return true;
    }
    s0 = "loadconf=%s";
    if (sscanf(L,s0.c_str(),c) == 1) {
	clear(c);
	return true;
    }
}

bool _ReadConfig (const char* name, const char* key, string& S) {
    if (!FileExists(name)) return false;
    M_ifstream is(name);
    char L[buffer_length];
    while (is.getline(L,buffer_length)) {
	if (L[0] == '#') continue;
	string s(key);
	char c[buffer_length];
	if (CheckLoadConf(L,c))
	    if (_ReadConfig(c,key,S)) 
		return true;
        string s0 = s + " =%s";
	if (sscanf(L,s0.c_str(),c) == 1) {
	    clear(c);
	    S = string(c);
	    return true;
	}
	string s1 = s + "=%s";
	if (sscanf(L,s0.c_str(),c) == 1) {
	    clear(c);
	    S = string(c);
	    return true;
	}
    }
    return false;
}
bool _ReadConfig (const char* name, const char* key, double& a) {
    if (!FileExists(name)) return false;
    M_ifstream is(name);
    string s(key);
    const int buffer_length = 128;
    char L[buffer_length];
    while (is.getline(L,buffer_length)) {
	if (L[0] == '#') continue;
	char c[buffer_length];
	if (CheckLoadConf(L,c))
	    if (_ReadConfig(c,key,a)) 
		return true;
	string s0 = s + " =%lf";
	if (sscanf(L,s0.c_str(),&a) == 1) return true;
	string s1 = s + "=%lf";
	if (sscanf(L,s1.c_str(),&a) == 1) return true;
    }
    return false;
}
bool _ReadConfig (const char* name, const char* key, vector<int>& a) {
    if (!FileExists(name)) return false;
    M_ifstream is(name);
    char L[buffer_length];
    while (is.getline(L,buffer_length)) {
	if (L[0] == '#') continue;
 	string s(key);
	char c[buffer_length];
	if (CheckLoadConf(L,c))
	    if (_ReadConfig(c,key,a)) 
		return true;
        string s0 = s + " =%d";
	if (sscanf(L,s0.c_str(),&a[0]) == 1) {
	  s += " =";
	  for(int i=1; i<a.size(); ++i) {
	    s += "%*d";
	    s0 = s + "%d";
	    sscanf(L,s0.c_str(),&a[i]);
	  }
	  return true;
	}
	string s1 = s + "=%d";
	if (sscanf(L,s1.c_str(),&a[0]) == 1) {
	  s += "=";
	  for(int i=1; i<a.size(); ++i) {
	    s += "%*d";
	    s1 = s + "%d";
	    sscanf(L,s1.c_str(),&a[i]);
	  }
	  return true;
	}
    }
    return false;
}
bool _ReadConfig (const char* name, const char* key, 
		  vector<double>& a, int size) {
    if (!FileExists(name)) return false;
    M_ifstream is(name);
    char L[buffer_length];
    while (is.getline(L,buffer_length)) {
	if (L[0] == '#') continue;
 	string s(key);
	char c[buffer_length];
	if (CheckLoadConf(L,c))
	    if (_ReadConfig(c,key,a,size)) 
		return true;
	if (size == -1) {
	    string s0 = s + " =%d";
	    int n;
	    if (sscanf(L,s0.c_str(),&n) == 1) {
		a.resize(n);
		s += " =%*d";
		for(int i=0; i<a.size(); ++i) {
		    s0 = s + "%lf";
		    sscanf(L,s0.c_str(),&a[i]);
		    s += "%*lf";
		}
		return true;
	    }
	    string s1 = s + "=%d";
	    if (sscanf(L,s1.c_str(),&n) == 1) {
		s += "=%*d";
		a.resize(n);
		for(int i=0; i<a.size(); ++i) {
		    s1 = s + "%lf";
		    sscanf(L,s1.c_str(),&a[i]);
		    s += "%*lf";
		}
		return true;
	    }
	}
	else {
	    a.resize(size);
	    string s0 = s + " =%lf";
	    if (sscanf(L,s0.c_str(),&a[0]) == 1) {
		s += " =";
		for(int i=1; i<a.size(); ++i) {
		    s += "%*lf";
		    s0 = s + "%lf";
		    sscanf(L,s0.c_str(),&a[i]);
		}
		return true;
	    }
	    string s1 = s + "=%lf";
	    if (sscanf(L,s1.c_str(),&a[0]) == 1) {
		s += "=";
		for(int i=1; i<a.size(); ++i) {
		    s += "%*lf";
		    s1 = s + "%lf";
		    sscanf(L,s1.c_str(),&a[i]);
		}
		return true;
	    }
	}
    }
    return false;
}


bool ReadConfig (const char* name, const char* key, 
                 string& S, bool mute) {
    bool ok;
    if (PPM->master()) {
        ok = _ReadConfig(name,key,S);
    }
    if (mute) return ok;
    PPM->Broadcast(ok);
    if (ok) { 
	char c[buffer_length];
	int n;
	if (PPM->master()) {
	    strcpy(c,S.c_str());
	    for (int i=0; i<S.length(); ++i) {
		if (S[i] == ' ') c[i] = 0;
		if (S[i] == ';') c[i] = 0;
	    }
	    n = strlen(c);
	} 
	PPM->Broadcast(n);
	PPM->Broadcast(c,n+1);
	S = string(c);
    }
    if (ok) mout << "  reading ... " << key << " = " << S << "\n";
    else    mout << "  default ... " << key << " = " << S << "\n";
    logging->flush();
    return ok;
}

bool ReadConfig (const char* name, const char* key, 
		 int& a, bool mute) {
    bool ok;
    if (PPM->master()) {
	double b;
	ok = _ReadConfig(name,key,b);
	if (ok) a = int(b);
    }
    if (mute) return ok;
    PPM->Broadcast(ok);
    if (ok) PPM->Broadcast(a);
    if (ok) mout << "  reading ... " << key << " = " << a << "\n";
    else    mout << "  default ... " << key << " = " << a << "\n";
    logging->flush();
    return ok;
}
bool ReadConfig (const char* name, string key, 
			int& a, bool mute) {
    return ReadConfig(name,key.c_str(),a,mute);
}
bool ReadConfig (const char* name, const char* key, 
			double& a, bool mute) {
    bool ok;
    if (PPM->master()) ok = _ReadConfig(name,key,a);
    if (mute) return ok;
    PPM->Broadcast(ok);
    if (ok) PPM->Broadcast(a);
    if (ok) mout << "  reading ... " << key << " = " << a << "\n";
    else    mout << "  default ... " << key << " = " << a << "\n";
    logging->flush();
    return ok;
}
bool ReadConfig (const char* name, string key, 
			double& a, bool mute) {
    return ReadConfig(name,key.c_str(),a,mute);
}

bool ReadConfig (const char* name, const char* key, vector<int>& a) {
    bool ok;
    if (PPM->master()) ok = _ReadConfig(name,key,a);
    PPM->Broadcast(ok);
    if (ok) {
      for(int i=0; i<a.size(); ++i)
	PPM->Broadcast(a[i]);
      mout << "  reading ... " << key << " =";
      char buf[128];
      for(int i=0; i<a.size(); ++i) {
	sprintf(buf,"%4d",a[i]);
	string s(buf);
	mout  << s;
      }
      mout << "\n";
    }
    logging->flush();
    return ok;
}
bool ReadConfig (const char* name, const char* key, 
			vector<double>& a, int size) {
    bool ok;
    if (PPM->master()) ok = _ReadConfig(name,key,a,size);
    PPM->Broadcast(ok);
    if (ok) {
	int n = a.size();
	PPM->Broadcast(n);
	if(!PPM->master())
	    a.resize(n);
	for (int i=0; i<a.size(); ++i)
	    PPM->Broadcast(a[i]);
	mout << "  reading ... " << key << " =";
	char buf[128];
	for (int i=0; i<a.size(); ++i) {
	    sprintf(buf,"%8.5f",a[i]);
	    string s(buf);
	    mout  << s;
	}
	mout << "\n";
    } else if (a.size() >= 1) {
	mout << "  default ... " << key << " =";
	char buf[128];
	for(int i=0; i<a.size(); ++i) {
	    sprintf(buf,"%8.5f",a[i]);
	    string s(buf);
	    mout  << s;
	}
	mout << "\n";
    }
    logging->flush();
    return ok;
}
bool ReadConfig (const char* name, const char* key, Point& z) {
    vector<double> v(3); 
    bool ok = ReadConfig(name, key, v, 3);
    z[0] = v[0]; 
    z[1] = v[1]; 
    z[2] = v[2];
    return ok;
}
bool ReadConfig (const char* name, const char* key, bool& b) {
    int i;
    i = (b) ? 1 : 0;
    ReadConfig(name, key, i);
    if (i == 0) {
	b = false;
    }
    else if (i == 1) {
	b = true;
    }
    else {
	Exit("Enter 0 or 1 for boolean variable; file: IO.h \n"); 
    }
    return 1;
}
