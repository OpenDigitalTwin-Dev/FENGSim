  /* Copyright William F. Schelter 1991
     Copyright 2024 Camm Maguire
   Bignum routines.


   
num_arith.c: add_int_big
num_arith.c: big_minus
num_arith.c: big_plus
num_arith.c: big_quotient_remainder
num_arith.c: big_sign
num_arith.c: big_times
num_arith.c: complement_big
num_arith.c: copy_big
num_arith.c: div_int_big
num_arith.c: mul_int_big
num_arith.c: normalize_big
num_arith.c: normalize_big_to_object
num_arith.c: stretch_big
num_arith.c: sub_int_big
num_comp.c: big_compare
num_comp.c: big_sign
num_log.c: big_sign
num_log.c: copy_to_big
num_log.c: normalize_big
num_log.c: normalize_big_to_object
num_log.c: stretch_big
num_pred.c: big_sign
number.c: big_to_double
predicate.c: big_compare
typespec.c: big_sign
print.d: big_minus
print.d: big_sign
print.d: big_zerop
print.d: copy_big
print.d: div_int_big
read.d: add_int_big
read.d: big_to_double
read.d: complement_big
read.d: mul_int_big
read.d: normalize_big
read.d: normalize_big_to_object

 */

#define remainder gclremainder
#define NEED_MP_H
#include "include.h"
#include "num_include.h"

#ifdef STATIC_FUNCTION_POINTERS
static void* alloc_relblock_static (size_t n) {return alloc_relblock (n);}
static void* alloc_contblock_static(size_t n) {return alloc_contblock(n);}
#endif

void* (*gcl_gmp_allocfun)(size_t)=FFN(alloc_relblock);
int gmp_relocatable=1;

DEFUN("INTEGER-QUOTIENT-REMAINDER_1",object,fSinteger_quotient_remainder_1,SI,4,4,NONE,OO,OO,IO,OO,(object r,object x,object y,fixnum d),"") {

  integer_quotient_remainder_1(x,y,&r->c.c_car,&r->c.c_cdr,d);

  RETURN1(r);

}



DEFUN("MBIGNUM2",object,fSbignum2,SI,2,2,NONE,OI,IO,OO,OO,(fixnum h,fixnum l),"") {

  object x = new_bignum();

  mpz_set_si(MP(x),h);
  mpz_mul_2exp(MP(x),MP(x),8*sizeof(x));
  mpz_add_ui(MP(x),MP(x),l);

  RETURN1(normalize_big(x));

}


DEFUN("SET-GMP-ALLOCATE-RELOCATABLE",object,fSset_gmp_allocate_relocatable,SI,1,1,NONE,OO,OO,OO,OO,
      (object flag),"Set the allocation to be relocatble ")
{
  if (flag == Ct) {
    gcl_gmp_allocfun = FFN(alloc_relblock);
    gmp_relocatable=1;
  } else {
    gcl_gmp_allocfun = FFN(alloc_contblock);
    gmp_relocatable=0;
  }
  RETURN1(flag);
}

#ifdef GMP
#include "gmp_big.c"
#else
#include "pari_big.c"
#endif



int big_sign(object x)
{
  return BIG_SIGN(x);
}

void set_big_sign(object x, int sign)
{
  SET_BIG_SIGN(x,sign);
}

void zero_big(object x)
{
  ZERO_BIG(x);
}


#ifndef HAVE_MP_COERCE_TO_STRING

double digitsPerBit[37]={ 0,0,
1.0, /* 2 */
0.6309297535714574, /* 3 */
0.5, /* 4 */
0.4306765580733931, /* 5 */
0.3868528072345416, /* 6 */
0.3562071871080222, /* 7 */
0.3333333333333334, /* 8 */
0.3154648767857287, /* 9 */
0.3010299956639811, /* 10 */
0.2890648263178878, /* 11 */
0.2789429456511298, /* 12 */
0.2702381544273197, /* 13 */
0.2626495350371936, /* 14 */
0.2559580248098155, /* 15 */
0.25, /* 16 */
0.244650542118226, /* 17 */
0.2398124665681315, /* 18 */
0.2354089133666382, /* 19 */
0.2313782131597592, /* 20 */
0.227670248696953, /* 21 */
0.2242438242175754, /* 22 */
0.2210647294575037, /* 23 */
0.2181042919855316, /* 24 */
0.2153382790366965, /* 25 */
0.2127460535533632, /* 26 */
0.2103099178571525, /* 27 */
0.2080145976765095, /* 28 */
0.2058468324604345, /* 29 */
0.2037950470905062, /* 30 */
0.2018490865820999, /* 31 */
0.2, /* 32 */
0.1982398631705605, /* 33 */
0.1965616322328226, /* 34 */
0.1949590218937863, /* 35 */
0.1934264036172708, /* 36 */
};

object
coerce_big_to_string(x,printbase)
     int printbase;
     object x;
{ int i;
 int sign=big_sign(x);
 object b;
 int size = (int)((ceil(MP_SIZE_IN_BASE2(MP(x))* digitsPerBit[printbase]))+.01);
 char *q,*p = ZALLOCA(size+5);
 q=p;
 if(sign<=0) {
   *q++ = '-';
   b=big_minus(x);
 } else {
   b=copy_big(x);
 }
 while (!big_zerop(b))
   *q++=digit_weight(div_int_big(printbase, b),printbase);
 *q++=0;
  object ans = alloc_simple_string(q-p);
  ans->ust.ust_self=alloc_relblock(ans->ust.ust_dim);
  bcopy(ans->ust.ust_self,p,ans->ust.ust_dim);
  return ans;
}

#endif
