// file:    Parallel.h
// author:  Christian Wieners
// $Header: /public/M++/src/Parallel.h,v 1.5 2009-07-09 13:29:25 maurer Exp $

#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "Point.h" 
#include "Debug.h" 
#include <complex>

#include "Time.h"

class ExchangeBuffer;

class ParallelProgrammingModel {
    int p; 
    int N;
 public: 
    ParallelProgrammingModel (int*, char ***);
    ~ParallelProgrammingModel ();
    short proc () const { return p; }
    short size () const { return N; }
    bool master () const {     
      return (p == 0);
    }
    void Broadcast (void*, int);
    void Sum (int*, size_t);
    void Sum (double*, size_t);
    void Sum (complex<double>*, size_t);
    // template <class C> void Sum (vector<C>& v);
    template <class C> void Sum (vector<C>& v) {
        int n = v.size();
        C* a = new C [n];
        for (int i=0; i<n; ++i) a[i] = v[i]; 
        Sum(a,n);
        for (int i=0; i<n; ++i) v[i] = a[i]; 
        delete[] a;
    }
    double Min (double);
    double Max (double);
    bool Boolean (bool b) {
	double a = 0;
	if (b) a = 1;
	a = Max(a);
	if (a == 1) return true;
	return false;
    }
    //template <class C> void Broadcast(C& c);
    //template <class C> void Broadcast(const C& c);
    template <class C> void Broadcast(C& c) { Broadcast(&c,sizeof(c));}

    template <class C> void Broadcast(const C& c) { 
        Assert(master()); C d = c; Broadcast(&d,sizeof(c));
    }
    int Sum (int a) { Sum(&a,1); return a; }
    int Max (int a) { return int(Max(double(a))); }
    double Sum (double a) { Sum(&a,1); return a; }
    complex<double> Sum (complex<double> z) { 
	double a[2];
	a[0] = real(z);
	a[1] = imag(z);
	Sum(a,2); 
	return complex<double>(a[0],a[1]); 
    }
    void BroadcastDouble (double d);
    double BroadcastDouble ();
    void BroadcastInt (int i);
    int BroadcastInt ();
    void Synchronize (const char * c = 0);
    void Communicate (ExchangeBuffer&);
};
extern ParallelProgrammingModel* PPM;

class Exchange {
    vector<int> n;
    vector<int> d;
    vector<int> s;
    int n_send;
    int n_recv;
    bool init;
 public:
    Exchange ();
    void SendSize (int m, short q) { s[q] = m; }
    size_t SendSize (short q) const;
    size_t SendSize () const;
    size_t ReceiveSize (short q) const;
    int SendMessages () const { return n_send; }
    int RecvMessages () const { return n_recv; }
    int Messages (int k) const { return n[k]; }
    int MessageDest (int k) const { return d[k]; }
    int MessageSize (int k) const { return s[k]; }
    void CommunicateSize ();
    bool Initialized () const { return init; }
    friend ostream& operator << (ostream& s, const Exchange& E);
};

class Buffer {
    char* b;
    char* p;
    size_t n;
 public:
    Buffer (size_t m = 0);
    size_t size () const { return size_t(p-b); }
    size_t Size () const { return n; }
    ~Buffer ();
    void rewind ();
    char* operator () () { return b; }
    void resize (size_t m);

//    template <class C> Buffer& fill (const C& c, size_t m) {

//     Buffer& fill (const char& c, size_t m) {
// //    Buffer& fill (const char* c, size_t m) {
//         dpout(100) << " template " << m << " << " << &c << endl; 
//         while (size()+m>n) resize(n+BufferSize);
// //        memcpy(p,c,m);
//         memcpy(p,&c,m); 
//         p += m; 
//         dpout(100) << " s " << size() << " S " << Size() 
//                    << " b " << long(p)
//                    << " p " << long(p) << endl;
//         return *this; 
//     }

    template <class C> Buffer& fill (const C& c, size_t m) {
//    Buffer& fill (const char* c, size_t m) {
        dpout(100) << " template " << m << " << " << &c << endl; 
        if (size()+m > n) cout << "resized Buffer " << size() << "  " << m << "  " << n << endl;
        while (size()+m>n) resize(n+BufferSize);
//        memcpy(p,c,m);
        memcpy(p,&c,m); 
        p += m; 
        dpout(100) << " s " << size() << " S " << Size() 
                   << " b " << long(p)
                   << " p " << long(p) << endl;
        return *this; 
    }

//     Buffer& read (char* c, size_t m) {
//     Buffer& read (char& c, size_t m) {
//         dpout(100) << " template " << m << " >> " << c << endl;
//         memcpy(&c,p,m); 
//         p += m; 
//         dpout(100) << " S " << Size() 
//                    << " p " << long(p) << endl;
//         return *this; 
//     }

    template <class C> Buffer& read (C& c, size_t m) {
        dpout(100) << " template " << m << " >> " << c << endl;
        memcpy(&c,p,m); 
        p += m; 
        dpout(100) << " S " << Size() 
                   << " p " << long(p) << endl;
        return *this; 
    }

    template <class C> Buffer& operator << (const C& c) {
        size_t m = sizeof(c);
        dpout(100) << " template " << m << " << " << c << endl; 
        while (size()+m>n) resize(n+BufferSize);
        memcpy(p,&c,m); 
        p += m; 
        dpout(100) << " s " << size() << " S " << Size() 
                   << " b " << long(p)
                   << " p " << long(p) << endl;
        return *this; 
    }
    template <class C> Buffer& operator >> (C& c) {
        size_t m = sizeof(c); 
        dpout(100) << " template " << m << " >> " << c << endl;
        memcpy(&c,p,m); 
        p += m; 
        dpout(100) << " S " << Size() 
                   << " p " << long(p) << endl;
        return *this; 
    }
    //template <class C> Buffer& operator << (const C& c);
    //template <class C> Buffer& operator >> (C& c);
};

class ExchangeBuffer : public Exchange {
    vector<Buffer> SendBuffers;
    vector<Buffer> ReceiveBuffers;
    void CommunicateSizes ();
 public:
    ExchangeBuffer();
    Buffer& Send (short q);
    Buffer& Receive (short q);
    void Communicate ();
    void Rewind ();
};

#endif
