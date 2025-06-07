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

#define IN_NUM_CO
#define NEED_ISFINITE

#include "include.h"
#include "num_include.h"

object imag_unit, minus_imag_unit, imag_two;

fixnum
fixnum_expt(fixnum x, fixnum y)
{
	fixnum z;

 	z = 1;
	while (y > 0)
		if (y%2 == 0) {
			x *= x;
			y /= 2;
		} else {
			z *= x;
			--y;
		}
	return(z);
}

static object number_sin(object);
static object number_cos(object);
static object number_exp(object);
static object number_nlog(object);
static object number_atan2(object,object);


static double
pexp(double y,object z,int s) {

  double x=exp(y);
  if (s) x=(float)x;
  /* if (!x) */
  /*   FLOATING_POINT_UNDERFLOW(sLexp,z); */
  /* if (!ISFINITE(x) && ISFINITE(y)) */
  /*   FLOATING_POINT_OVERFLOW(sLexp,z); */
  return x;

}

static object
number_exp(object x)
{
	double exp(double);

	switch (type_of(x)) {

	case t_fixnum:
	case t_bignum:
	case t_ratio:
		return(make_longfloat((longfloat)pexp(number_to_double(x),x,0)));

	case t_shortfloat:
		return(make_shortfloat((shortfloat)pexp((double)sf(x),x,1)));

	case t_longfloat:
		return(make_longfloat(pexp(lf(x),x,0)));

	case t_complex:
	{
		object y, y1;
	        vs_mark;
	
		y = x->cmp.cmp_imag;
		x = x->cmp.cmp_real;
		x = number_exp(x);
		vs_push(x);
		y1 = number_cos(y);
		vs_push(y1);
		y = number_sin(y);
		vs_push(y);
		y = make_complex(y1, y);
		vs_push(y);
		x = number_times(x, y);
		vs_reset;
		return(x);
	}

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

static inline object
number_fix_iexpt(object x,fixnum y,fixnum ly,fixnum j) {
  object z;
  
  if (j+1==ly) return x;
  z=number_fix_iexpt(number_times(x,x),y,ly,j+1);
  return fixnum_bitp(j,y) ? number_times(x,z) : z;
}

static inline object
number_big_iexpt(object x,object y,fixnum ly,fixnum j) {
  object z;
  
  if (j+1==ly) return x;
  z=number_big_iexpt(number_times(x,x),y,ly,j+1);
  return mpz_tstbit(MP(y),j) ? number_times(x,z) : z;

}

static inline fixnum
number_contagion_index(object x) {

  switch(type_of(x)) {
  case t_fixnum:
  case t_bignum:
  case t_ratio:
    return 0;
  case t_shortfloat:
    return 1;
  case t_longfloat:
    return 2;
  case t_complex:
    return 3+number_contagion_index(x->cmp.cmp_real);
  }
  return 0;
}

static inline object
number_zero_expt(object x,fixnum cy) {

  enum type cx=number_contagion_index(x);

  if (gcl_is_not_finite(x))/*FIXME, better place?*/
    return number_exp(number_times(number_nlog(x),small_fixnum(0)));

  switch (cx<cy ? cy : cx) {
  case 3:case 0:
    return make_fixnum(1);
  case 1:
    return make_shortfloat(1.0);
  case 2:
    return make_longfloat(1.0);
  case 4:
    return make_complex(make_shortfloat(1.0),make_fixnum(0));
  case 5:
    return make_complex(make_longfloat(1.0),make_fixnum(0));
  default:
    FEwrong_type_argument(sLnumber,x);
    return Cnil;
  }

}


static inline object
number_ui_expt(object x,fixnum fy) {

  switch (type_of(x)) {
  case t_fixnum:
    { 
      fixnum fx=fix(x);
      object z;
      MPOP(z=,mpz_ui_pow_ui,labs(fx),fy);
      if (fx<0&&(fy&0x1)) return number_negate(z); else return z;
    }
  case t_bignum:
    MPOP(return,mpz_pow_ui,MP(x),fy);
  case t_ratio:
    return make_ratio(number_ui_expt(x->rat.rat_num,fy),number_ui_expt(x->rat.rat_den,fy),1);

  case t_shortfloat:
  case t_longfloat:
  case t_complex:
    {
      fixnum ly=fixnum_length(fy);

      return ly ? number_fix_iexpt(x,fy,ly,0) : number_zero_expt(x,0);

    }
	
  default:
    FEwrong_type_argument(sLnumber,x);
    return Cnil;
  }
    
}

static inline object
number_ump_expt(object x,object y) {
  return number_big_iexpt(x,y,fix(integer_length(y)),0);
}

static inline object
number_log_expt(object x,object y) {
  return number_zerop(y) ? number_zero_expt(x,number_contagion_index(y)) : number_exp(number_times(number_nlog(x),y));
}

static inline object
number_invert(object x,object y,object z) {

  switch (type_of(z)) {
  case t_shortfloat:
    if (!ISNORMAL(sf(z))) return number_log_expt(x,y);
    break;
  case t_longfloat:
    if (!ISNORMAL(lf(z))) return number_log_expt(x,y);
    break;
  }
  return number_divide(small_fixnum(1),z);
}
    

static inline object
number_si_expt(object x,object y) {
  switch (type_of(y)) {
  case t_fixnum:
    { 
      fixnum fy=fix(y);
      if (fy>=0)
	return number_ui_expt(x,fy);
      if (fy==MOST_NEGATIVE_FIX)
	return number_invert(x,y,number_ump_expt(x,number_negate(y)));
      return number_invert(x,y,number_ui_expt(x,-fy));
    }
  case t_bignum:
    return big_sign(y)<0 ? number_invert(x,y,number_ump_expt(x,number_negate(y))) : number_ump_expt(x,y);
  case t_ratio:
  case t_shortfloat:
  case t_longfloat:
  case t_complex:
    return number_log_expt(x,y);
  default:
    FEwrong_type_argument(sLnumber,y);
    return Cnil;
  }
}

object
number_expt(object x, object y) {

  if (number_zerop(x)&&y!=small_fixnum(0)) {
    if (!number_plusp(type_of(y)==t_complex?y->cmp.cmp_real:y))
      FEerror("Cannot raise zero to the power ~S.", 1, y);
    return(number_times(x, y));
  }

  return number_si_expt(x,y);

}

static object
number_nlog(object x)
{
	double log(double);
	object r=Cnil, i=Cnil, a, p;
	vs_mark;

	if (type_of(x) == t_complex) {
		r = x->cmp.cmp_real;
		i = x->cmp.cmp_imag;
		goto COMPLEX;
	}
	if (number_zerop(x))
		FEerror("Zero is the logarithmic singularity.", 0);
	if (number_minusp(x)) {
		r = x;
		i = small_fixnum(0);
		goto COMPLEX;
	}
	switch (type_of(x)) {
	case t_fixnum:
	case t_bignum:
	case t_ratio:
		return(make_longfloat(log(number_to_double(x))));

	case t_shortfloat:
		return(make_shortfloat((shortfloat)log((double)(sf(x)))));

	case t_longfloat:
		return(make_longfloat(log(lf(x))));

	default:
		FEwrong_type_argument(sLnumber, x);
	}

COMPLEX:
	a = number_times(r, r);
	vs_push(a);
	p = number_times(i, i);
	vs_push(p);
	a = number_plus(a, p);
	vs_push(a);
	a = number_nlog(a);
	vs_push(a);
	a = number_divide(a, small_fixnum(2));
	vs_push(a);
	p = number_atan2(i, r);
	vs_push(p);
	x = make_complex(a, p);
	vs_reset;
	return(x);
}

static object
number_log(object x, object y)
{
	object z;
	vs_mark;

	if (number_zerop(y))
		FEerror("Zero is the logarithmic singularity.", 0);
	if (number_zerop(x))
		return(number_times(x, y));
	x = number_nlog(x);
	vs_push(x);
	y = number_nlog(y);
	vs_push(y);
	z = number_divide(y, x);
	vs_reset;
	return(z);
}

static object
number_sqrt(object x)
{
	object z;
	vs_mark;

	if (type_of(x) == t_complex)
		goto COMPLEX;
	if (number_minusp(x))
		goto COMPLEX;
	switch (type_of(x)) {
	case t_fixnum:
	case t_bignum:
	case t_ratio:
		return(make_longfloat(
			(longfloat)sqrt(number_to_double(x))));

	case t_shortfloat:
		return(make_shortfloat((shortfloat)sqrtf((double)(sf(x)))));

	case t_longfloat:
		return(make_longfloat(sqrt(lf(x))));

	default:
		FEwrong_type_argument(sLnumber, x);
	}

COMPLEX:
	{extern object plus_half;
	z = number_expt(x, plus_half);}
	vs_reset;
	return(z);
}

object
number_abs(object x) {

  object r,i,z;

  switch(type_of(x)) {

  case t_complex:
    if (number_zerop(x)) return x->cmp.cmp_real;
    r=number_abs(x->cmp.cmp_real);
    i=number_abs(x->cmp.cmp_imag);
    if (number_compare(r,i)<0) {
      object z=i;
      i=r;
      r=z;
    }
    z=number_divide(i,r);
    return number_times(r,number_sqrt(one_plus(number_times(z,z))));

  case t_fixnum:
    {fixnum fx=fix(x);return fx==MOST_NEGATIVE_FIX ? fixnum_add(1,MOST_POSITIVE_FIX) : (fx<0 ? make_fixnum(-fx) : x);}

  case t_bignum:
    return big_sign(x)<0 ? big_minus(x) : x;

  case t_ratio:
    {object n=number_abs(x->rat.rat_num);return n==x ? x : make_ratio(n,x->rat.rat_den,1);}

  case t_shortfloat:
    return sf(x)<0.0 ? make_shortfloat(-sf(x)) : x;

  case t_longfloat:
    return lf(x)<0.0 ? make_longfloat(-lf(x)) : x;

  default:
    FEwrong_type_argument(sLnumber,x);
    return(Cnil);
  }
}

object
number_signum(object x) {

  switch (type_of(x)) {

  case t_fixnum:
    {fixnum fx=fix(x);return make_fixnum(fx<0 ? -1 : (fx==0 ? 0 : 1));}

  case t_bignum:
    return make_fixnum(big_sign(x)<0 ? -1 : 1);

  case t_ratio:
    return number_signum(x->rat.rat_num);

  case t_shortfloat:
    return make_shortfloat(sf(x)<0.0 ? -1.0 : (sf(x)==0.0 ? 0.0 : 1.0));

  case t_longfloat:
    return make_longfloat(lf(x)<0.0 ? -1.0 : (lf(x)==0.0 ? 0.0 : 1.0));

  case t_complex:
    return number_zerop(x) ? x : number_divide(x,number_abs(x));

  default:
    FEwrong_type_argument(sLnumber,x);
    return(Cnil);

  }

}

static object
number_atan2(object y, object x)
{
	object z;
	double atan(double), dy, dx, dz=0.0;

	dy = number_to_double(y);
	dx = number_to_double(x);
	if (dx > 0.0)
		if (dy > 0.0)
			dz = atan(dy / dx);
		else if (dy == 0.0)
			dz = 0.0;
		else
			dz = -atan(-dy / dx);
	else if (dx == 0.0)
		if (dy > 0.0)
			dz = PI / 2.0;
		else if (dy == 0.0)
		        dz = 0.0;
		else
			dz = -PI / 2.0;
	else
		if (dy > 0.0)
			dz = PI - atan(dy / -dx);
		else if (dy == 0.0)
			dz = PI;
		else
			dz = -PI + atan(-dy / -dx);
	if (type_of(x) == t_shortfloat)
	  z = make_shortfloat((shortfloat)dz);
	else
	  z = make_longfloat(dz);
	return(z);
}

static object
number_atan(object y)
{
	object z, z1;
        vs_mark;

	if (type_of(y) == t_complex) {
		z = number_times(imag_unit, y);
		vs_push(z);
		z = one_plus(z);
		vs_push(z);
		z1 = number_times(y, y);
		vs_push(z1);
		z1 = one_plus(z1);
		vs_push(z1);
		z1 = number_sqrt(z1);
		vs_push(z1);
		z = number_divide(z, z1);
		vs_push(z);
		z = number_nlog(z);
		vs_push(z);
		z = number_times(minus_imag_unit, z);
		vs_reset;
		return(z);
	}
	return(number_atan2(y, small_fixnum(1)));
}

static object
number_sin(object x)
{
	double sin(double);

	switch (type_of(x)) {

	case t_fixnum:
	case t_bignum:
	case t_ratio:
		return(make_longfloat((longfloat)sin(number_to_double(x))));

	case t_shortfloat:
		return(make_shortfloat((shortfloat)sin((double)(sf(x)))));

	case t_longfloat:
		return(make_longfloat(sin(lf(x))));

	case t_complex:
	{
		object	r;
		object	x0, x1, x2;
		vs_mark;

		x0 = number_times(imag_unit, x);
		vs_push(x0);
		x0 = number_exp(x0);
		vs_push(x0);
		x1 = number_times(minus_imag_unit, x);
		vs_push(x1);
		x1 = number_exp(x1);
		vs_push(x1);
		x2 = number_minus(x0, x1);
		vs_push(x2);
		r = number_divide(x2, imag_two);

		vs_reset;
		return(r);
	}

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);

	}
}

static object
number_cos(object x)
{
	double cos(double);

	switch (type_of(x)) {

	case t_fixnum:
	case t_bignum:
	case t_ratio:
		return(make_longfloat((longfloat)cos(number_to_double(x))));

	case t_shortfloat:
		return(make_shortfloat((shortfloat)cos((double)(sf(x)))));

	case t_longfloat:
		return(make_longfloat(cos(lf(x))));

	case t_complex:
	{
		object r;
		object x0, x1, x2;
		vs_mark;

		x0 = number_times(imag_unit, x);
		vs_push(x0);
		x0 = number_exp(x0);
		vs_push(x0);
		x1 = number_times(minus_imag_unit, x);
		vs_push(x1);
		x1 = number_exp(x1);
		vs_push(x1);
		x2 = number_plus(x0, x1);
		vs_push(x2);
		r = number_divide(x2, small_fixnum(2));

		vs_reset;
		return(r);
	}

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);

	}
}

static object
number_tan1(object x)
{
	double cos(double);

	switch (type_of(x)) {

	case t_fixnum:
	case t_bignum:
	case t_ratio:
		return(make_longfloat((longfloat)tan(number_to_double(x))));

	case t_shortfloat:
		return(make_shortfloat((shortfloat)tan((double)(sf(x)))));

	case t_longfloat:
		return(make_longfloat(tan(lf(x))));

	case t_complex:
	{
		object r;
		object x0, x1, x2;
		vs_mark;

		x0 = number_times(imag_two, x);
		vs_push(x0);
		x0 = number_exp(x0);
		vs_push(x0);
		x1 = number_minus(x0,small_fixnum(1));
		vs_push(x1);
		x2 = number_plus(x0,small_fixnum(1));
		vs_push(x2);
		x2 = number_times(x2,imag_unit);
		vs_push(x2);
		r = number_divide(x1, x2);

		vs_reset;
		return(r);
	}

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);

	}
}

static object
number_tan(object x)
{
	object r, c;
	vs_mark;

	c = number_cos(x);
	vs_push(c);
	if (number_zerop(c) == TRUE)
		FEerror("Cannot compute the tangent of ~S.", 1, x);
	r = number_tan1(x);
	vs_reset;
	return(r);
}

LFD(Lexp)(void)
{
	check_arg(1);
	check_type_number(&vs_base[0]);
	vs_base[0] = number_exp(vs_base[0]);
}

DEFUN("EXPT",object,fLexpt,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  check_type_number(&vs_base[0]);
  check_type_number(&vs_base[1]);
  RETURN1(number_expt(x,y));

}

LFD(Llog)(void)
{
	int narg;
	
	narg = vs_top - vs_base;
	if (narg < 1)
		too_few_arguments();
	else if (narg == 1) {
		check_type_number(&vs_base[0]);
		vs_base[0] = number_nlog(vs_base[0]);
	} else if (narg == 2) {
		check_type_number(&vs_base[0]);
		check_type_number(&vs_base[1]);
		vs_base[0] = number_log(vs_base[1], vs_base[0]);
		vs_popp;
	} else
		too_many_arguments();
}

LFD(Lsqrt)(void)
{
	check_arg(1);
	check_type_number(&vs_base[0]);
	vs_base[0] = number_sqrt(vs_base[0]);
}

LFD(Lsin)(void)
{
	check_arg(1);
	check_type_number(&vs_base[0]);
	vs_base[0] = number_sin(vs_base[0]);
}

LFD(Lcos)(void)
{
	check_arg(1);
	check_type_number(&vs_base[0]);
	vs_base[0] = number_cos(vs_base[0]);
}

LFD(Ltan)(void)
{
	check_arg(1);
	check_type_number(&vs_base[0]);
	vs_base[0] = number_tan(vs_base[0]);
}

LFD(Latan)(void)
{
	int narg;

	narg = vs_top - vs_base;
	if (narg < 1)
		too_few_arguments();
	if (narg == 1) {
		check_type_number(&vs_base[0]);
		vs_base[0] = number_atan(vs_base[0]);
	} else if (narg == 2) {
		check_type_or_rational_float(&vs_base[0]);
		check_type_or_rational_float(&vs_base[1]);
		vs_base[0] = number_atan2(vs_base[0], vs_base[1]);
		vs_popp;
	} else
		too_many_arguments();
}

static void
FFN(siLmodf)(void)
{
  
  object x;
  double d,ip;

  check_arg(1);
  check_type_float(&vs_base[0]);
  x=vs_base[0];
  vs_base=vs_top;
  d=type_of(x) == t_longfloat ? lf(x) : (double)sf(x);
  d=modf(d,&ip);
  vs_push(make_fixnum((int)ip));
  vs_push(type_of(x) == t_longfloat ? make_longfloat(d) : make_shortfloat((shortfloat)d));

}

DEFUN("ISNAN",object,fSisnan,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  switch (type_of(x)) {
  case t_longfloat:
    return isnan(lf(x)) ? Ct : Cnil;
    break;
  case t_shortfloat:
    return isnan(sf(x)) ? Ct : Cnil;
    break;
  default:
    return Cnil;
    break;
  }

  return Cnil;

}


DEFUN("ISINF",object,fSisinf,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  switch (type_of(x)) {
  case t_longfloat:
    return isinf(lf(x)) ? Ct : Cnil;
    break;
  case t_shortfloat:
    return  isinf(sf(x)) ? Ct : Cnil;
    break;
  default:
    return Cnil;
    break;
  }

  return Cnil;

}


void
gcl_init_num_sfun(void)
{
	imag_unit
	= make_complex(make_longfloat((longfloat)0.0),
		       make_longfloat((longfloat)1.0));
	enter_mark_origin(&imag_unit);
	minus_imag_unit
	= make_complex(make_longfloat((longfloat)0.0),
		       make_longfloat((longfloat)-1.0));
	enter_mark_origin(&minus_imag_unit);
	imag_two
	= make_complex(make_longfloat((longfloat)0.0),
		       make_longfloat((longfloat)2.0));
	enter_mark_origin(&imag_two);

	make_constant("PI", make_longfloat(PI));

	make_function("EXP", Lexp);
	make_function("LOG", Llog);
	make_function("SQRT", Lsqrt);
	make_function("SIN", Lsin);
	make_function("COS", Lcos);
	make_function("TAN", Ltan);
	make_function("ATAN", Latan);
	make_si_function("MODF", siLmodf);
}
