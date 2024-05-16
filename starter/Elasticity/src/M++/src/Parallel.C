// file:    Parallel.C
// author:  Christian Wieners
// $Header: /public/M++/src/Parallel.C,v 1.6 2009-07-09 13:29:45 maurer Exp $

#include "Parallel.h"
#include "Time.h"

///////////////////////////////////////////////////////////////////////////
/// from Parallel.h
///////////////////////////////////////////////////////////////////////////

/* template <class C> void ParallelProgrammingModel::Sum (vector<C>& v) {
    int n = v.size();
    C* a = new C [n];
    for (int i=0; i<n; ++i) a[i] = v[i]; 
    Sum(a,n);
    for (int i=0; i<n; ++i) v[i] = a[i]; 
    delete[] a;
}

template <class C> void ParallelProgrammingModel<C>::Broadcast(C& c) { Broadcast(&c,sizeof(c));}

template <class C> void ParallelProgrammingModel<C>::Broadcast(const C& c) { 
    Assert(master()); C d = c; Broadcast(&d,sizeof(c));
}
*/

void ParallelProgrammingModel::BroadcastDouble (double d) { Assert(master()); Broadcast(d); }

double ParallelProgrammingModel::BroadcastDouble () { 
    Assert(!master()); double a; Broadcast(a); return a;
}

void ParallelProgrammingModel::BroadcastInt (int i) { Assert(master()); Broadcast(i); }

int ParallelProgrammingModel::BroadcastInt () { 
    Assert(!master()); int i; Broadcast(i); return i;
}

void ParallelProgrammingModel::Synchronize (const char * c) {
    cout.flush();
    Sum(1.0);
    if (c==0) return;
    cout << p << "|"; 
    Sum(1.0);
    cout.flush();
    if (master()) cout << c << endl;
    cout.flush();
    Sum(1.0); 
    cout.flush();
}

Exchange::Exchange () : n(PPM->size()+1), s(PPM->size(),0), 
	n_send(0), n_recv(0), init(false) {}

size_t Exchange::SendSize (short q) const { 
    for (int k=n[PPM->proc()]; k<n[PPM->proc()+1]; ++k) 
        if (d[k] == q) return s[k];
    return 0;
}

size_t Exchange::SendSize () const {
    int b = 0;
    for (int k=n[0]; k<n[PPM->size()]; ++k) 
        b +=s[k];
    return b;
}

size_t Exchange::ReceiveSize (short q) const {
    for (int k=n[q]; k<n[q+1]; ++k) 
        if (d[k] == PPM->proc()) return s[k];
    return 0;
}

void Exchange::CommunicateSize () {
    for (int q=0; q<PPM->size(); ++q) n[q] = 0;
    for (int q=0; q<PPM->size(); ++q) if (s[q]) ++(n[PPM->proc()]);
    PPM->Sum(n);
    for (int q=PPM->size(); q>0; --q) n[q] = n[q-1];
    n[0] = 0;
    for (int q=0; q<PPM->size(); ++q) n[q+1] += n[q];
    d.resize(n[PPM->size()],0); 
    vector<int> tmp(n[PPM->size()],0); 
    int k = n[PPM->proc()];
    for (int q=0; q<PPM->size(); ++q) 
        if (s[q]) { 
            tmp[k] = s[q];
            d[k++] = q;
        }
    s = tmp;
    PPM->Sum(s);
    PPM->Sum(d);
    n_send = n[PPM->proc()+1] - n[PPM->proc()];
    n_recv = 0;
    for (int k=0; k<n[PPM->size()]; ++k) if (d[k] == PPM->proc()) ++n_recv;
    init = true;
}

ostream& operator << (ostream& s, const Exchange& E) {
    s << " on " << PPM->proc() << " :" 
      << "send " << E.SendMessages() << " : ";
    for (int k=E.Messages(PPM->proc()); k<E.Messages(PPM->proc()+1); ++k)
        s << E.MessageDest(k) << "|" << E.MessageSize(k) << " ";
    s << "recv " << E.RecvMessages() << " : "; 
    for (int q=0; q<PPM->size(); ++q)
        for (int k=E.Messages(q); k<E.Messages(q+1); ++k) 
            if (E.MessageDest(k) == PPM->proc()) 
                s << q << "|" << E.MessageSize(k) << " ";
    return s << endl;
}


Buffer::Buffer (size_t m) : n(m) { if (m) b = new char [m]; else b=0; p=b; }

Buffer::~Buffer () {
    dpout(100) << " delete " << " s " << size() << endl;
    delete[] b;
    dpout(100) << " S " << Size() 
               << " b " << long(p)
               << " p " << long(p) << endl;
}

void Buffer::rewind () {
    dpout(100) << " rewind " << " s " << size() << endl;
    p = b;
    dpout(100) << " S " << Size() 
               << " b " << long(p)
               << " p " << long(p) << endl;
}

void Buffer::resize (size_t m) {
    dpout(100) << " resize " << " s " << size() << endl;
    char* tmp = new char [m]; 
    int s = size();
    if (s) {
        memcpy(tmp,b,s);
        delete[] b;
    }
    p = tmp + s;
    b = tmp;
    n = m;
    dpout(100) << " S " << Size() 
               << " b " << long(p)
               << " p " << long(p) << endl;
}

/*
 template <class C> Buffer& Buffer::operator << (const C& c) {
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

template <class C> Buffer& Buffer<C>::operator >> (C& c) {
    size_t m = sizeof(c); 
    dpout(100) << " template " << m << " >> " << c << endl;
    memcpy(&c,p,m); 
    p += m; 
    dpout(100) << " S " << Size() 
               << " p " << long(p) << endl;
    return *this; 
}
*/
void ExchangeBuffer::CommunicateSizes () { 
    for (short q=0; q<PPM->size(); ++q) SendSize(Send(q).size(),q);
    CommunicateSize();
    for (short q=0; q<PPM->size(); ++q) Receive(q).resize(ReceiveSize(q));
}

ExchangeBuffer::ExchangeBuffer() : SendBuffers(PPM->size()), ReceiveBuffers(PPM->size()) {}

Buffer& ExchangeBuffer::Send (short q) { return SendBuffers[q]; } 

Buffer& ExchangeBuffer::Receive (short q) { return ReceiveBuffers[q]; } 

void ExchangeBuffer::Communicate () { 
    if (!Initialized()) CommunicateSizes();
    PPM->Communicate(*this);
}

void ExchangeBuffer::Rewind () { 
    for (short q=0; q<PPM->size(); ++q) {
        Send(q).rewind(); 
        Receive(q).rewind(); 
    }
}

///////////////////////////////////////////////////////////////////////////


#ifdef NPARALLEL

const int MPI_COMM_WORLD = 0;
const int MPI_BYTE = 1;
const int MPI_SUM = 2;
const int MPI_DOUBLE = 3;
const int MPI_INT = 4;
const int MPI_MIN = 5;
const int MPI_MAX = 6;
const int MPI_SUCCESS = 1;
typedef int MPI_Status;
typedef int MPI_Request;
inline int MPI_Init (int* argc, char*** argv) { return MPI_SUCCESS; }
inline int MPI_Finalize () {}
inline int MPI_Comm_size (int i, int* p) { *p = 1; return MPI_SUCCESS; }
inline int MPI_Comm_rank (int i, int* p) { *p = 0; return MPI_SUCCESS; }
inline int MPI_Bcast (void* a, int b, int c, int d, int e) { 
    return MPI_SUCCESS; }
inline int MPI_Allreduce (void* a, void* b, int c, int d, int e, int f) {
    if (d == MPI_DOUBLE) memcpy(b,a,sizeof(double)*c);
    if (d == MPI_INT) memcpy(b,a,sizeof(int)*c);
    return MPI_SUCCESS;
}
inline int MPI_Wait (int* a, int* b) { return MPI_SUCCESS; }
inline int MPI_Irecv (void* a, int b, int c, int d, int e, int f, int* g) {
    return MPI_SUCCESS; }
inline int MPI_Isend(void* a, int b, int c, int d, int e, int f, int* g) {
    return MPI_SUCCESS; }

#else
#include "mpi.h"
#endif

ParallelProgrammingModel* PPM = 0;

ParallelProgrammingModel::ParallelProgrammingModel (int* argc, char ***argv) {
    MPI_Init(argc,argv);
    MPI_Comm_size(MPI_COMM_WORLD,&N);
    MPI_Comm_rank(MPI_COMM_WORLD,&p);
}
ParallelProgrammingModel::~ParallelProgrammingModel () {  MPI_Finalize(); }
void ParallelProgrammingModel::Broadcast (void* data, int size) {
    char* Data = (char*) data;
    int Size = size;
    int pos = 0;
    while (Size > MaxBroadcastSize) {
	MPI_Bcast (Data+pos,MaxBroadcastSize,MPI_BYTE,0,MPI_COMM_WORLD);
	pos += MaxBroadcastSize;
	Size -= MaxBroadcastSize;
    }
    if (Size > 0) 
	MPI_Bcast (Data+pos,Size,MPI_BYTE,0,MPI_COMM_WORLD);
}
void ParallelProgrammingModel::Sum (int* a, size_t n) {
    int* b = new int [n];
    MPI_Allreduce(a,b,n,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    memcpy(a,b,sizeof(int)*n);
    delete[] b;
}
void ParallelProgrammingModel::Sum (double* a, size_t n) {
    double* b = new double [n];
    MPI_Allreduce(a,b,n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    memcpy(a,b,sizeof(double)*n);
    delete[] b;
}
void ParallelProgrammingModel::Sum (complex<double>* z, size_t n) {
    double* a = new double [2*n];
    double* b = new double [2*n];
    for (int i=0; i<n; ++i) {
	a[i] = real(z[i]);
	a[i+n] = imag(z[i]);
    }
    MPI_Allreduce(a,b,n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    for (int i=0; i<n; ++i) 
	z[i] = complex<double>(b[i],b[i+n]);
    delete[] a;
    delete[] b;
}
double ParallelProgrammingModel::Min (double a) {
    double b;
    MPI_Allreduce(&a,&b,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    return b;
}
double ParallelProgrammingModel::Max (double a) {
    double b;
    MPI_Allreduce(&a,&b,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    return b;
}
void ParallelProgrammingModel::Communicate (ExchangeBuffer& E) {
#ifdef NPARALLEL
    for (int k=E.Messages(0); k<E.Messages(1); ++k) 
	memcpy(E.Receive(0)(),E.Send(0)(),E.MessageSize(k));
    return;
#endif

//     cout << "Exchange " << E << endl;

    const int tag = 27;
    MPI_Request* req_send = new MPI_Request [E.SendMessages()];
    MPI_Request* req_recv = new MPI_Request [E.RecvMessages()];
    MPI_Request* r = req_recv;
    for (int q=0; q<PPM->size(); ++q)
	for (int k=E.Messages(q); k<E.Messages(q+1); ++k) 
	    if (E.MessageDest(k) == PPM->proc()) 
		MPI_Irecv(E.Receive(q)(),E.MessageSize(k),MPI_BYTE,
			  q,tag,MPI_COMM_WORLD,r++);
    r = req_send; 
    for (int k=E.Messages(PPM->proc()); k<E.Messages(PPM->proc()+1); ++k)
	MPI_Isend(E.Send(E.MessageDest(k))(),E.MessageSize(k),MPI_BYTE,
		  E.MessageDest(k),tag,MPI_COMM_WORLD,r++);
#ifdef DEBUG_PARALLEL
    for (int q=0; q<PPM->size(); ++q)
	for (int k=E.Messages(q); k<E.Messages(q+1); ++k) 
	    if (E.MessageDest(k) == PPM->proc()) 
		pout << "R " << E.MessageSize(k) << " q " << q << endl;
    for (int k=E.Messages(PPM->proc()); k<E.Messages(PPM->proc()+1); ++k)
        pout << "S " << E.MessageSize(k) << " q " << E.MessageDest(k) <<endl;
#endif
    MPI_Status st;
    r = req_send; for (int k=0; k<E.SendMessages(); ++k) MPI_Wait(r++,&st);
    r = req_recv; for (int k=0; k<E.RecvMessages(); ++k) MPI_Wait(r++,&st);
    delete[] req_send;
    delete[] req_recv;
}



