/* Copyright (C) 2024 Camm Maguire */
#define NEED_MP_H
#ifndef FIRSTWORD
#include "include.h"
#endif
#include "num_include.h"

/*  #include "arith.h"   */



/* I believe the instructions used here are ok for 68010.. */

#ifdef MC68K
#define MC68020
#endif
  
/* static for gnuwin95 the save routine is not saving statics... */

object *gclModulus;
#define FIXNUMP(x) (type_of(x)==t_fixnum)

/* Note: the gclModulus is guaranteed > 0 */

#define FIX_MOD(X,MOD) {			\
  register fixnum MOD_2;			\
  if (X > (MOD_2=(MOD>>1)))			\
    X=X-MOD;					\
  else						\
    if (X < -MOD_2)				\
      X=X+MOD;					\
    else					\
      if (X == -MOD_2 && (MOD&0x1)==0)		\
	X=X+MOD;				\
  }


object ctimes(object a, object b),cplus(object a, object b),cdifference(object a, object b),cmod(object x);
	  
object make_integer(__mpz_struct *u);  
 	  
#define our_minus(a,b) ((FIXNUMP(a)&&FIXNUMP(b))?fixnum_sub(fix(a),fix(b)): \
			number_minus(a,b))
#define our_plus(a,b) ((FIXNUMP(a)&&FIXNUMP(b))?fixnum_add(fix(a),fix(b)): \
			number_plus(a,b))
#define our_times(a,b) number_times(a,b)

/* fix (and check) this on 64 bit machines, where long is the long long */
#ifdef HAVE_LONG_LONG
static int
dblrem(int a, int b, int mod)
{
  return  (int)(((long long int)a*(long long int)b)%(long long int) mod);
}
#else

static int
dblrem(a,b,mod)
int a,b,mod;
{int h,sign;
 if (a<0) 
   {a= -a; sign= (b<0)? (b= -b,1) :-1;}
 else { sign= (b<0) ? (b= -b,-1) : 1;}
 { mp_limb_t ar[2],q[2],aa;
 aa = a;
  ar[1]=mpn_mul_1(ar,&aa,1,b);
  h = mpn_divrem_1(q,0,ar,2,mod);
 return ((sign<0) ? -h :h);
 }
}
#endif

/* #if sizeof(fixnum) != sizeof(mp_limb_t) */
/* #error fixnum mp_limb_t size mismatch */
/* #endif */

static fixnum
fdblrem(fixnum a,fixnum b,fixnum mod) {

  fixnum h,sign;
  mp_limb_t ar[2],q[2],aa;

  if (a<0) {
    a= -a; 
    sign= (b<0) ? (b= -b,1) : -1;
  } else
    sign= (b<0) ? (b= -b,-1) : 1;

  aa = a;
  ar[1]=mpn_mul_1(ar,&aa,1,b);
  h = mpn_divrem_1(q,0,ar,2,mod);

  return ((sign<0) ? -h :h);

}

object	  
cmod(object x) {

  register object mod = *gclModulus;

  if (mod==Cnil) 
    return(x);

  else if ((type_of(mod)==t_fixnum && type_of(x)==t_fixnum)) {

    register fixnum xx,mm=fix(mod);
    
    if (mm==2) 
      return small_fixnum((fix(x)&1));

    xx=(fix(x)%mm);
    FIX_MOD(xx,mm);
    return make_fixnum(xx);

  } else {

    object rp,mod2;
    int compare;

    integer_quotient_remainder_1(x,mod,NULL,&rp,0);/*FIXME*/
    mod2=integer_fix_shift(mod,-1);
    compare = number_compare(rp,small_fixnum(0));
    if (compare >= 0) {

      compare=number_compare(rp,mod2);
      if (compare > 0) rp=number_minus(rp,mod);

    } else if (number_compare(number_negate(mod2), rp) > 0)
      rp = number_plus(rp,mod);

    return rp;

  }

}


object
ctimes(object a, object b) {

  object mod = *gclModulus;

  if (FIXNUMP(mod)) {
    
    register fixnum res, m=fix(mod);

    if (sizeof(fixnum)==sizeof(int) || (m>>(sizeof(int)*8)==(m>>(sizeof(fixnum)*8-1))))

      res=dblrem(fix(a),fix(b),m);

    else
      
      res=fdblrem(fix(a),fix(b),m);

    FIX_MOD(res,m);
    return make_fixnum(res);

  } else if (mod==Cnil)
    return(our_times(a,b));

  return cmod(number_times(a,b));

}


#define SMALL_MODULUS_P(mod) (FIXNUMP(mod) && (fix(mod) < (MOST_POSITIVE_FIX)/2))

object
cdifference(object a, object b) {

  object mod = *gclModulus;

  if (SMALL_MODULUS_P(mod)) {
    
    register fixnum res,m;

    res=((fix(a)-fix(b))%(m=fix(mod)));
    FIX_MOD(res,m);
    return make_fixnum(res);

  } else if (mod==Cnil)
    return (our_minus(a,b));

 else return(cmod(number_minus(a,b)));

}

object
cplus(object a, object b) {

  object mod = *gclModulus;

 if (SMALL_MODULUS_P(mod)) {

   register fixnum res,m;

   res=((fix(a)+fix(b))%(m=fix(mod)));
   FIX_MOD(res,m);
   return make_fixnum(res);

 } else if (mod==Cnil)
   return (our_plus(a,b));
 
 return(cmod(number_plus(a,b)));

}

DEFUN("CMOD",object,fScmod,SI,1,1,NONE,OO,OO,OO,OO,(object num),"") {
  num=cmod(num);
  RETURN1(num);
}


DEFUN("CPLUS",object,fScplus,SI,2,2,NONE,OO,OO,OO,OO,(object x0,object x1),"") {
  x0 = cplus(x0,x1);
  RETURN1( x0 );
}

DEFUN("CTIMES",object,fSctimes,SI,2,2,NONE,OO,OO,OO,OO,(object x0,object x1),"") {
 x0=ctimes(x0,x1);
 RETURN1(x0);
}

DEFUN("CDIFFERENCE",object,fScdifference,SI,2,2,NONE,OO,OO,OO,OO,(object x0,object x1),"") {
  x0=cdifference(x0,x1);
  RETURN1(x0);
}


void     
gcl_init_cmac(void) {

  gclModulus = (&((make_si_special("MODULUS",Cnil))->s.s_dbind));

}
