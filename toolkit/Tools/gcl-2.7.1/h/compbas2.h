#ifndef COMP_BAS_2
#define COMP_BAS_2
/* if already mp.h has been included skip */
#define save_avma 
#define restore_avma 


EXTER object MVloc[10];
EXTER int Rset;

#ifndef U8_DEFINED


typedef int8_t  i8 ;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64; 

typedef uint8_t  n8 ;
typedef uint16_t n16;
typedef uint32_t n32;
typedef uint64_t n64;

typedef float       f32;
typedef double      f64;
typedef long double f128;

typedef fcomplex c64;
typedef dcomplex c128;

typedef object o32;

typedef union {int8_t i;uint8_t u;n8 n;} u8;
typedef union {int16_t i;uint16_t u;n16 n;}  __attribute__((__packed__)) u16;
typedef union {
  int32_t i;
#if SIZEOF_LONG!=4
  uint32_t u;
  n32 n;
#else
  object o;
#endif
  float f;}  __attribute__((__packed__)) u32;
typedef union {
#if SIZEOF_LONG!=4
  int64_t i;
  object o;
#endif
  double f;
  fcomplex c;}  __attribute__((__packed__)) u64;
typedef union {dcomplex c;}  __attribute__((__packed__)) u128;

#define U8_DEFINED
#endif
#endif
