/*
 Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
 Copyright (C) 2024 Camm Maguire

This file is part of GNU Common Lisp, herein referred to as GCL

GCL is free software; you can redistribute it and/or modify it under
the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GCL is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
License for more details.

You should have received a copy of the GNU Library General Public License 
along with GCL; see the file COPYING.  If not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

*/

/*
	num_co.c
	IMPLEMENTATION-DEPENDENT

	This file contains those functions
	that know the representation of floating-point numbers.
*/	
#define IN_NUM_CO

#define NEED_MP_H
#define NEED_ISFINITE

#include "include.h"
#include "num_include.h"

object plus_half, minus_half;
#ifdef CONVEX
#define VAX
#endif

/*   A number is normal when:
   * it is finite,
   * it is not zero, and
   * its exponent is non-zero.
*/

#ifndef IEEEFLOAT
#error this file needs IEEEFLOAT
#endif

int 
gcl_isnormal_double(double d) {

  union {double d;int i[2];} u;
  
  if (!ISFINITE(d) || !d)
    return 0;

  u.d = d;
  return (u.i[HIND] & 0x7ff00000) != 0;

}

int
gcl_isnormal_float(float f) {

  union {float f;int i;} u;

  if (!ISFINITE(f) || !f)
    return 0;

  u.f = f;
  return (u.i & 0x7f800000) != 0;

}

static inline int
gcl_isnan_double(double d) {

  if (ISFINITE(d))
    return 0;
  if (d==d)
    return 0;
  return 1;

}

static inline int
gcl_isnan_float(float f) {

  if (ISFINITE(f))
    return 0;
  if (f==f)
    return 0;
  return 1;

}

int
gcl_isnan(object x) {

  switch(type_of(x)) {
  case t_shortfloat:
    return gcl_isnan_float(sf(x));
  case t_longfloat:
    return gcl_isnan_double(lf(x));
  default:
    return 0;
  }

}

int
gcl_is_not_finite(object x)  {

  switch(type_of(x)) {
  case t_shortfloat:
    return !ISFINITE(sf(x));
  case t_longfloat:
    return !ISFINITE(lf(x));
  default:
    return 0;
  }
}

DEFUN("ISFINITE",object,fSisfinite,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  switch (type_of(x)) {
  case t_longfloat:
    return lf(x)==0.0 || ISFINITE(lf(x)) ? Ct : Cnil;
    break;
  case t_shortfloat:
    return sf(x)==0.0 || ISFINITE(sf(x)) ? Ct : Cnil;
    break;
  default:
    return Cnil;
    break;
  }

  return Cnil;

}

DEFUN("ISNORMAL",object,fSisnormal,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  switch (type_of(x)) {
  case t_longfloat:
    return ISNORMAL(lf(x)) ? Ct : Cnil;
  case t_shortfloat:
    return ISNORMAL(sf(x)) ? Ct : Cnil;
  default:
    return Cnil;
  }

}

static void
integer_decode_double(double d, int *hp, int *lp, int *ep, int *sp)
{
	int h, l;
	union {double d;int i[2];} u;

	if (d == 0.0) {
		*hp = *lp = 0;
		*ep = 0;
		*sp = 1;
		return;
	}
	u.d=d;
	h=u.i[HIND];
	l=u.i[LIND];
	if (ISNORMAL(d)) {
	  *ep = ((h & 0x7ff00000) >> 20) - 1022 - 53;
	  h = ((h & 0x000fffff) | 0x00100000);
	} else {
	  *ep = ((h & 0x7fe00000) >> 20) - 1022 - 53 + 1;
	  h = (h & 0x001fffff);
	}
	if (32-BIG_RADIX)
	  /* shift for making bignum */
	  { h = h << (32-BIG_RADIX) ; 
	    h |= ((l & (-1 << (32-BIG_RADIX))) >> (32-BIG_RADIX));
	    l &=  ~(-1 << (32-BIG_RADIX));
	  }
	*hp = h;
	*lp = l;
	*sp = (d > 0.0 ? 1 : -1);
}

object
double_to_rational(double d) {

  object x;
  int h,l,e,s;

  integer_decode_double(d,&h,&l,&e,&s);
  x=number_times((h!=0 || l<0) ? bignum2(h,l) : make_fixnum(l),
		 number_expt(make_fixnum(2),make_fixnum(e)));
  if (s<0) x=number_negate(x);
  return x;

}

static void
integer_decode_float(float f, int *mp, int *ep, int *sp)
{
	int m;
	union {float f;int i;} u;

	if (f == 0.0) {
		*mp = 0;
		*ep = 0;
		*sp = 1;
		return;
	}
	u.f=f;
	m=u.i;
/* 	m = *(int *)(&f); */
	if (ISNORMAL(f)) {
	  *ep = ((m & 0x7f800000) >> 23) - 126 - 24;
	  *mp = (m & 0x007fffff) | 0x00800000;
	} else {
	  *ep = ((m & 0x7f000000) >> 23) - 126 - 24 + 1;
	  *mp = m & 0x00ffffff;
	}
	*sp = (f > 0.0 ? 1 : -1);
}

static int
double_exponent(double d)
{
	union {double d;int i[2];} u;

	if (d == 0.0)
		return(0);
	u.d=d;
	return (((u.i[HIND] & 0x7ff00000) >> 20) - 1022);
}

static double
set_exponent(double d, int e)
{
	union {double d;int i[2];} u;

	if (d == 0.0)
		return(0.0);
	  
	u.d=d;
	u.i[HIND]= (u.i[HIND] & 0x800fffff) | (((e + 1022) << 20) & 0x7ff00000);
	return(u.d);
}


object
double_to_integer(double d) {

  int h, l, e, s;
  object x;
  vs_mark;
  
  if (d == 0.0)
    return(small_fixnum(0));
  integer_decode_double(d, &h, &l, &e, &s);

  if (e <= -BIG_RADIX) {
    e = (-e) - BIG_RADIX;
    if (e >= BIG_RADIX)
      return(small_fixnum(0));
    h >>= e;
    return(make_fixnum(s*h));
  }
  if (h != 0 || l<0)
    x = bignum2(h, l);
  else
    x = make_fixnum(l);
  vs_push(x);
  x = integer_fix_shift(x, e);
  if (s < 0) {
    vs_push(x);
    x = number_negate(x);
  }
  vs_reset;
  return(x);
}

static object
num_remainder(object x, object y, object q)
{
	object z;

	z = number_times(q, y);
	vs_push(z);
	z = number_minus(x, z);
	vs_popp;
	return(z);
}

inline void
intdivrem(object x,object y,fixnum d,object *q,object *r) {

  enum type tx=type_of(x),ty=type_of(y);
  object z,q2,q1;

  if (number_zerop(y)==TRUE)
    DIVISION_BY_ZERO(sLtruncate,list(2,x,y));

  switch(tx) {
  case t_fixnum:
  case t_bignum:
    switch (ty) {
    case t_fixnum:
    case t_bignum:
      integer_quotient_remainder_1(x,y,q,r,d);
      return;
    case t_ratio:
      z=integer_divide1(number_times(y->rat.rat_den,x),y->rat.rat_num,d);
      if (q) *q=z;
      if (r) *r=num_remainder(x,y,z);
      return;
    default:
      break;
    }
    break;
  case t_ratio:
    switch (ty) {
    case t_fixnum:
    case t_bignum:
      z=integer_divide1(x->rat.rat_num,number_times(x->rat.rat_den,y),d);
      if (q) *q=z;
      if (r) *r=num_remainder(x,y,z);
      return;
    case t_ratio:
      z=integer_divide1(number_times(x->rat.rat_num,y->rat.rat_den),number_times(x->rat.rat_den,y->rat.rat_num),d);
      if (q) *q=z;
      if (r) *r=num_remainder(x,y,z);
      return;
    default:
      break;
    }
    break;
  default:
    break;
  }

  q2=number_divide(x,y);
  q1=double_to_integer(number_to_double(q2));
  if (d && (d<0 ? number_minusp(q2) : number_plusp(q2)) && number_compare(q2, q1))
    q1 = d<0 ? one_minus(q1) : one_plus(q1);
  if (q) *q=q1;
  if (r) *r=num_remainder(x,y,q1);
  return;
  
}

DEFUN("INTDIVREM",object,fSintdivrem,SI,3,3,NONE,OO,OI,OO,OO,(object x,object y,fixnum d),"") {

  intdivrem(x,y,d,&x,&y);

  RETURN1(MMcons(x,y));

}


object
number_ldb(object x,object y) {
  object (*foo)(object,object)=(void *)sLldb->s.s_gfdef->fun.fun_self;
  return foo(x,y);
}

object
number_ldbt(object x,object y) {
  object (*foo)(object,object)=(void *)sLldb_test->s.s_gfdef->fun.fun_self;
  return foo(x,y);
}

object
number_dpb(object x,object y,object z) {
  object (*foo)(object,object,object)=(void *)sLdpb->s.s_gfdef->fun.fun_self;
  return foo(x,y,z);
}

object
number_dpf(object x,object y,object z) {
  object (*foo)(object,object,object)=(void *)sLdeposit_field->s.s_gfdef->fun.fun_self;
  return foo(x,y,z);
}

DEFUNM("FLOOR",object,fLfloor,LISP,1,2,NONE,OO,OO,OO,OO,(object x,...),"") {

  fixnum nargs=INIT_NARGS(1);
  object f=OBJNULL,l=Cnil,y;
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;
  va_list ap;

  va_start(ap,x);
  y=NEXT_ARG(nargs,ap,l,f,make_fixnum(1));
  va_end(ap);

  intdivrem(x,y,-1,&x,&y);

  RETURN2(x,y);

}

DEFUNM("CEILING",object,fLceiling,LISP,1,2,NONE,OO,OO,OO,OO,(object x,...),"") {

  fixnum nargs=INIT_NARGS(1);
  object f=OBJNULL,l=Cnil,y;
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;
  va_list ap;

  va_start(ap,x);
  y=NEXT_ARG(nargs,ap,l,f,make_fixnum(1));
  va_end(ap);

  intdivrem(x,y,1,&x,&y);

  RETURN2(x,y);

}

DEFUNM("TRUNCATE",object,fLtruncate,LISP,1,2,NONE,OO,OO,OO,OO,(object x,...),"") {

  fixnum nargs=INIT_NARGS(1);
  object f=OBJNULL,l=Cnil,y;
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;
  va_list ap;

  va_start(ap,x);
  y=NEXT_ARG(nargs,ap,l,f,make_fixnum(1));
  va_end(ap);

  intdivrem(x,y,0,&x,&y);

  RETURN2(x,y);

}

DEFUNM("ROUND",object,fLround,LISP,1,2,NONE,OO,OO,OO,OO,(object x,...),"") {

  fixnum nargs=INIT_NARGS(1);
  object f=OBJNULL,l=Cnil,y,q,q1,r;
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;
  double d;
  int c;
  enum type tp;
  va_list ap;

  va_start(ap,x);
  y=NEXT_ARG(nargs,ap,l,f,make_fixnum(1));
  va_end(ap);

  check_type_or_rational_float(&x);
  check_type_or_rational_float(&y);

  q = eql(y,small_fixnum(1)) ? x : number_divide(x, y);

  switch ((tp=type_of(q))) {

  case t_fixnum:
  case t_bignum:
    RETURN2(q,small_fixnum(0));

  case t_ratio:
    q1 = integer_divide1(q->rat.rat_num, q->rat.rat_den,0);/*FIXME*/
    r = number_minus(q, q1);
    if ((c = number_compare(r, plus_half)) > 0 ||
	(c == 0 && number_oddp(q1)))
      q1 = one_plus(q1);
    if ((c = number_compare(r, minus_half)) < 0 ||
	(c == 0 && number_oddp(q1)))
      q1 = one_minus(q1);
    RETURN2(q1,num_remainder(x, y, q1));

  case t_shortfloat:
  case t_longfloat:
    d = number_to_double(q);
    q1 = double_to_integer(d + (d >= 0.0 ? 0.5 : -0.5));
    d -= number_to_double(q1);
    if (d == 0.5 && number_oddp(q1)) {
      q1 = one_plus(q1);
      d=-0.5;
    }
    if (d == -0.5 && number_oddp(q1)) {
      q1 = one_minus(q1);
      d=+0.5;
    }
    RETURN2(q1,tp==t_shortfloat ? make_shortfloat((shortfloat)d) : make_longfloat(d));
  default:
    TYPE_ERROR(q,sLreal);
  }
}

DEFUN("MOD",object,fLmod,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  intdivrem(x,y,-1,NULL,&y);
  RETURN1(y);
}

DEFUN("REM",object,fLrem,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  intdivrem(x,y,0,NULL,&y);
  RETURN1(y);
}

DEFUNM("DECODE-FLOAT",object,fLdecode_float,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  int e,s;
  fixnum vals=(fixnum)fcall.valp;
  double d;
  object *base=vs_top;

  check_type_float(&x);

  if (type_of(x) == t_shortfloat)
    d = sf(x);
  else
    d = lf(x);
  if (d >= 0.0)
    s = 1;
  else {
    d = -d;
    s = -1;
  }
  e=0;
  if (!ISNORMAL(d)) {
    int hp,lp,sp;

    integer_decode_double(d,&hp,&lp,&e,&sp);
    if (hp!=0 || lp<0)
      d=number_to_double(bignum2(hp, lp));
    else
      d=lp;
  }
  e += double_exponent(d);
  d = set_exponent(d, 0);

  RETURN3(type_of(x) == t_shortfloat ? make_shortfloat((shortfloat)d) : make_longfloat(d),
	  make_fixnum(e),
	  type_of(x) == t_shortfloat ? make_shortfloat((shortfloat)s) : make_longfloat((double)s));

}

DEFUN("SCALE-FLOAT",object,fLscale_float,LISP,2,2,NONE,OO,IO,OO,OO,(object x,fixnum k),"") {

  double d;
  int e;

  if (type_of(x) == t_shortfloat)
    d = sf(x);
  else
    d = lf(x);

  e = double_exponent(d) + k;

  /* Upper bound not needed, handled by floating point overflow */
  /* this checks if we're in the denormalized range */
  if (!ISNORMAL(d) || (type_of(x) == t_shortfloat && e <= -126/*  || e >= 130 */) ||
      (type_of(x) == t_longfloat && (e <= -1022 /* || e >= 1026 */))) {
    for (;k>0;d*=2.0,k--);
    for (;k<0;d*=0.5,k++);
  } else
    d = set_exponent(d, e);

  RETURN1(type_of(x) == t_shortfloat ? make_shortfloat((shortfloat)d) : make_longfloat(d));

}

DEFUNM("INTEGER-DECODE-FLOAT",object,fLinteger_decode_float,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  int h,l,e,s;
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;
  
  check_type_float(&x);
  h=0;
  if (type_of(x) == t_longfloat)
    integer_decode_double(lf(x), &h, &l, &e, &s);
  else
    integer_decode_float(sf(x), &l, &e, &s);
  RETURN3((h || l<0) ? bignum2(h, l) : make_fixnum(l),make_fixnum(e),make_fixnum(s));

}

void
gcl_init_num_co(void) {

  float smallest_float, smallest_norm_float, biggest_float;
  double smallest_double, smallest_norm_double, biggest_double;
  float float_epsilon, float_negative_epsilon;
  double double_epsilon, double_negative_epsilon;
  union {double d;int i[2];} u;
  union {float f;int i;} uf;

  uf.i=1;
  u.i[HIND]=0;
  u.i[LIND]=1;
  smallest_float=uf.f;
  smallest_double=u.d;

  uf.i=0x7f7fffff;
  u.i[HIND]=0x7fefffff;
  u.i[LIND]=0xffffffff;
  biggest_float=uf.f;
  biggest_double=u.d;

  biggest_double = DBL_MAX;
  smallest_norm_double = DBL_MIN;
  smallest_norm_float = FLT_MIN;
  biggest_float = FLT_MAX;

  {

    volatile double rd,dd,td,td1;
    volatile float  rf,df,tf,tf1;
    int i,j;
#define MAX 500

    for (rf=1.0f,df=0.5f,i=j=0;i<MAX && j<MAX && df!=1.0f;i++,df=1.0f-(0.5f*(1.0f-df)))
      for (tf=rf,tf1=tf+1.0f,j=0;j<MAX && tf1!=1.0f;j++,rf=tf,tf*=df,tf1=tf+1.0f);
    if (i==MAX||j==MAX)
      printf("WARNING, cannot calculate float_epsilon: %d %d %f   %f %f %f\n",i,j,rf,df,tf,tf1);
    float_epsilon=rf;

    for (rf=1.0f,df=0.5f,i=j=0;i<MAX && j<MAX && df!=1.0f;i++,df=1.0f-(0.5f*(1.0f-df)))
      for (tf=rf,tf1=1.0f-tf,j=0;j<MAX && tf1!=1.0f;j++,rf=tf,tf*=df,tf1=1.0f-tf);
    if (i==MAX||j==MAX)
      printf("WARNING, cannot calculate float_negative_epsilon: %d %d %f   %f %f %f\n",i,j,rf,df,tf,tf1);
    float_negative_epsilon=rf;

    for (rd=1.0,dd=0.5,i=j=0;i<MAX && j<MAX && dd!=1.0;i++,dd=1.0-(0.5*(1.0-dd)))
      for (td=rd,td1=td+1.0,j=0;j<MAX && td1!=1.0;j++,rd=td,td*=dd,td1=td+1.0);
    if (i==MAX||j==MAX)
      printf("WARNING, cannot calculate double_epsilon: %d %d %f   %f %f %f\n",i,j,rd,dd,td,td1);
    double_epsilon=rd;

    for (rd=1.0,dd=0.5,i=j=0;i<MAX && j<MAX && dd!=1.0;i++,dd=1.0-(0.5*(1.0-dd)))
      for (td=rd,td1=1.0-td,j=0;j<MAX && td1!=1.0;j++,rd=td,td*=dd,td1=1.0-td);
    if (i==MAX||j==MAX)
      printf("WARNING, cannot calculate double_negative_epsilon: %d %d %f   %f %f %f\n",i,j,rd,dd,td,td1);
    double_negative_epsilon=rd;

  }



  make_si_constant("+INF",make_longfloat(INFINITY));
  make_si_constant("-INF",make_longfloat(-INFINITY));
  make_si_constant("NAN",make_longfloat(NAN));

  make_si_constant("+SINF",make_shortfloat(INFINITY));
  make_si_constant("-SINF",make_shortfloat(-INFINITY));
  make_si_constant("SNAN",make_shortfloat(NAN));

  make_constant("MOST-POSITIVE-SHORT-FLOAT",
		make_shortfloat(biggest_float));
  make_constant("LEAST-POSITIVE-SHORT-FLOAT",
		make_shortfloat(smallest_float));
  make_constant("LEAST-NEGATIVE-SHORT-FLOAT",
		make_shortfloat(-smallest_float));
  make_constant("MOST-NEGATIVE-SHORT-FLOAT",
		make_shortfloat(-biggest_float));

  make_constant("MOST-POSITIVE-SINGLE-FLOAT",
		make_longfloat(biggest_double));
  make_constant("LEAST-POSITIVE-SINGLE-FLOAT",
		make_longfloat(smallest_double));
  make_constant("LEAST-NEGATIVE-SINGLE-FLOAT",
		make_longfloat(-smallest_double));
  make_constant("MOST-NEGATIVE-SINGLE-FLOAT",
		make_longfloat(-biggest_double));

  make_constant("MOST-POSITIVE-DOUBLE-FLOAT",
		make_longfloat(biggest_double));
  make_constant("LEAST-POSITIVE-DOUBLE-FLOAT",
		make_longfloat(smallest_double));
  make_constant("LEAST-NEGATIVE-DOUBLE-FLOAT",
		make_longfloat(-smallest_double));
  make_constant("MOST-NEGATIVE-DOUBLE-FLOAT",
		make_longfloat(-biggest_double));

  make_constant("MOST-POSITIVE-LONG-FLOAT",
		make_longfloat(biggest_double));
  make_constant("LEAST-POSITIVE-LONG-FLOAT",
		make_longfloat(smallest_double));
  make_constant("LEAST-NEGATIVE-LONG-FLOAT",
		make_longfloat(-smallest_double));
  make_constant("MOST-NEGATIVE-LONG-FLOAT",
		make_longfloat(-biggest_double));

  make_constant("SHORT-FLOAT-EPSILON",
		make_shortfloat(float_epsilon));
  make_constant("SINGLE-FLOAT-EPSILON",
		make_longfloat(double_epsilon));
  make_constant("DOUBLE-FLOAT-EPSILON",
		make_longfloat(double_epsilon));
  make_constant("LONG-FLOAT-EPSILON",
		make_longfloat(double_epsilon));

  make_constant("SHORT-FLOAT-NEGATIVE-EPSILON",
		make_shortfloat(float_negative_epsilon));
  make_constant("SINGLE-FLOAT-NEGATIVE-EPSILON",
		make_longfloat(double_negative_epsilon));
  make_constant("DOUBLE-FLOAT-NEGATIVE-EPSILON",
		make_longfloat(double_negative_epsilon));
  make_constant("LONG-FLOAT-NEGATIVE-EPSILON",
		make_longfloat(double_negative_epsilon));

  /* Normalized constants added, CM */
  make_constant("LEAST-POSITIVE-NORMALIZED-SHORT-FLOAT",
		make_shortfloat(smallest_norm_float));
  make_constant("LEAST-NEGATIVE-NORMALIZED-SHORT-FLOAT",
		make_shortfloat(-smallest_norm_float));
  make_constant("LEAST-POSITIVE-NORMALIZED-SINGLE-FLOAT",
		make_longfloat(smallest_norm_double));
  make_constant("LEAST-NEGATIVE-NORMALIZED-SINGLE-FLOAT",
		make_longfloat(-smallest_norm_double));
  make_constant("LEAST-POSITIVE-NORMALIZED-DOUBLE-FLOAT",
		make_longfloat(smallest_norm_double));
  make_constant("LEAST-NEGATIVE-NORMALIZED-DOUBLE-FLOAT",
		make_longfloat(-smallest_norm_double));
  make_constant("LEAST-POSITIVE-NORMALIZED-LONG-FLOAT",
		make_longfloat(smallest_norm_double));
  make_constant("LEAST-NEGATIVE-NORMALIZED-LONG-FLOAT",
		make_longfloat(-smallest_norm_double));

  plus_half = make_ratio(small_fixnum(1), small_fixnum(2),1);
  enter_mark_origin(&plus_half);

  minus_half = make_ratio(small_fixnum(-1), small_fixnum(2),1);
  enter_mark_origin(&minus_half);

}
