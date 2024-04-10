// file: Compiler.h
// author: Christian Wieners
// $Header: /public/M++/src/CompilerDOUBLE.h,v 1.5 2009-11-24 09:47:38 wieners Exp $

#ifndef _COMPILER_H_
#define _COMPILER_H_

#define SUPERLU
//#define ILU_MULTILEVEL
//#define UMFSOLVER
//#define NPARALLEL
//#define NINTERVAL
//#define DCOMPLEX
#define RIB
#define LAPACK

using namespace std;

#include <ext/hash_map>
using __gnu_cxx::hash_map;

#include <ext/numeric>
using __gnu_cxx::power; 

#include <complex>
typedef complex<double> Complex;

#include <cstring>

#ifdef DCOMPLEX

#define NDOUBLE 
typedef complex<double> Scalar;
inline double double_of_Scalar (const Scalar& z) { return real(z); } 
inline Scalar eval (const Complex& z) { return z; }
const Scalar iUnit(0,1);

#define GEMM zgemm_
#define GETRS zgetrs_
#define GEMV zgemv_
#define GETRI zgetri_
#define GETRF zgetrf_
#define GESVD zgesvd_
#define POTRF zpotrf_
#define POTRS zpotrs_
#define POTRI zpotri_
#define SYMM zsymm_
#define TRSM ztrsm_

#else

#define GEMM dgemm_
#define GETRS dgetrs_
//#define GEMV dgemv_
#define GETRI dgetri_
#define GETRF dgetrf_
#define GESVD dgesvd_
#define POTRF dpotrf_
#define POTRS dpotrs_
#define POTRI dpotri_
#define SYMM dsymm_
#define TRSM dtrsm_

typedef double Scalar;
inline double double_of_Scalar (const Scalar& z) { return z; } 
inline double conj (double x) { return x; }
inline double real (double x) { return x; }
inline void set_real (Scalar&z, double x) { z = x; }
inline double eval (const Complex& z) { return real(z); }
const Scalar iUnit(-1);

#endif

#endif
