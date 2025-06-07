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
	Arithmetic operations
*/
#define NEED_MP_H
#define NEED_ISFINITE
#include "include.h"

#include "num_include.h"

object fixnum_add(fixnum i, fixnum j)
{

  if (i>=0)
   { if (j<= (MOST_POSITIVE_FIX-i))
      { return make_fixnum(i+j);
      }
   MPOP(return,addss,i,j);
   } else { /* i < 0 */
     if ((MOST_NEGATIVE_FIX -i) <= j) {
       return make_fixnum(i+j);
     }
   MPOP(return,addss,i,j);
   }
}
/* return i - j */
object fixnum_sub(fixnum i, fixnum j)
{  

  if (i>=0)
   { if (j >= (i - MOST_POSITIVE_FIX))
      { return make_fixnum(i-j);
      }
   MPOP(return,subss,i,j);
   } else { /* i < 0 */
     if (j <= (i-MOST_NEGATIVE_FIX)) {
       return make_fixnum(i-j);
     }
   MPOP(return,subss,i,j);
   }
}

inline object 
fixnum_times(fixnum i, fixnum j) {

#ifdef HAVE_CLZL
  if (i!=MOST_NEGATIVE_FIX && j!=MOST_NEGATIVE_FIX && fixnum_mul_safe(i,j))
#else
  if (i>=0 ? (j>=0 ? (!i || j<= (MOST_POSITIVE_FIX/i)) : (j==-1 || i<= (MOST_NEGATIVE_FIX/j))) :
      (j>=0 ? (i==-1 || j<= (MOST_NEGATIVE_FIX/i)) : (i>MOST_NEGATIVE_FIX && -i<= (MOST_POSITIVE_FIX/-j))))
#endif
      return make_fixnum(i*j);
  else
    MPOP(return,mulss,i,j);
}


static object
number_to_complex(object x)
{
	object z;

	switch (type_of(x)) {

	case t_fixnum:
	case t_bignum:
	case t_ratio:
	case t_shortfloat:
	case t_longfloat:
		z = alloc_object(t_complex);
		z->cmp.cmp_real = x;
		z->cmp.cmp_imag = small_fixnum(0);
		return(z);

	case t_complex:
		return(x);

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

static object
integer_exact_quotient(object r,object x,object y) {

  if (y==small_fixnum(1) || x==small_fixnum(0))
    return x;

  if (type_of(x)==t_fixnum)  /* no in_place for fixnums as could be small */
    return make_fixnum((type_of(y)==t_fixnum ? fix(x)/fix(y) : -1));
  /* Only big dividing a fix is most-negative-fix/abs(most-negative-fix)*/

  if (type_of(y)==t_fixnum)
    mpz_divexact_ui(MP(r),MP(x),fix(y));
  else
    mpz_divexact(MP(r),MP(x),MP(y));

  return normalize_big(r);

}

static object
fixnum_abs(object x) {

  if (type_of(x)==t_fixnum) {

    fixnum f=fix(x);

    return f==MOST_NEGATIVE_FIX ? sSPminus_most_negative_fixnumP->s.s_dbind : (f<0 ? make_fixnum(-f) : x);

  }

  return x;

}




static object
get_gcd_r_abs(object r,object x,object y) {

  if (x==small_fixnum(1) || y==small_fixnum(1))
    return small_fixnum(1);

  switch(type_of(x)) {
  case t_fixnum:
    switch(type_of(y)) {
    case t_fixnum:
      return make_fixnum(fixnum_gcd(fix(x),fix(y)));
    default:
      mpz_gcd_ui(MP(r),MP(y),fix(x));
      return normalize_big(r);
    }
  default:
    switch(type_of(y)) {
    case t_fixnum:
      mpz_gcd_ui(MP(r),MP(x),fix(y));
      return normalize_big(r);
    default:
      mpz_gcd(MP(r),MP(x),MP(y));
      return normalize_big(r);

    }
  }
}

static object
get_gcd_r(object r,object x,object y) {

  return get_gcd_r_abs(r,fixnum_abs(x),fixnum_abs(y));

}


object
get_gcd(object x,object y) {

  x=get_gcd_r(big_fixnum1,x,y);
  return x==big_fixnum1 ? replace_big(x) : x;

}

static object
get_gcd_abs(object x,object y) {

  x=get_gcd_r_abs(big_fixnum1,x,y);
  return x==big_fixnum1 ? replace_big(x) : x;

}

static object
integer_times(object r,object a,object b) {

  if (a==small_fixnum(1))
    return b;

  if (b==small_fixnum(1))
    return a;

  if (type_of(a)==t_fixnum)
    if (type_of(b)==t_fixnum)
      return fixnum_times(fix(a),fix(b));
    else {
      mpz_mul_si(MP(r),MP(b),fix(a));
      return normalize_big(r);
    }
  else
    if (type_of(b)==t_fixnum) {
      mpz_mul_si(MP(r),MP(a),fix(b));
      return normalize_big(r);
    } else {
      mpz_mul(MP(r),MP(a),MP(b));
      return normalize_big(r);
    }
}

#define mneg(a_) ((a_)==MOST_NEGATIVE_FIX ? (ufixnum)(a_) : (ufixnum)(-(a_)))
#define mpz_add_si(a_,b_,c_) ((c_)<0 ? mpz_sub_ui(a_,b_,mneg(c_)) : mpz_add_ui(a_,b_,c_))
#define mpz_sub_si(a_,b_,c_) ((c_)<0 ? mpz_add_ui(a_,b_,mneg(c_)) : mpz_sub_ui(a_,b_,c_))


static object
integer_add(object r,object a,object b) {

  if (a==small_fixnum(0))
    return b;

  if (b==small_fixnum(0))
    return a;

  if (type_of(a)==t_fixnum)
    if (type_of(b)==t_fixnum)
      return fixnum_add(fix(a),fix(b));
    else {
      mpz_add_si(MP(r),MP(b),fix(a));
      return normalize_big(r);
    }
  else
    if (type_of(b)==t_fixnum) {
      mpz_add_si(MP(r),MP(a),fix(b));
      return normalize_big(r);
    } else {
      mpz_add(MP(r),MP(a),MP(b));
      return normalize_big(r);
    }
}

static object
integer_sub(object r,object a,object b) {

  /* if (a==small_fixnum(0)) */
  /*   return b; */

  if (b==small_fixnum(0))
    return a;

  if (type_of(a)==t_fixnum)
    if (type_of(b)==t_fixnum)
      return fixnum_sub(fix(a),fix(b));
    else {
      mpz_sub_si(MP(r),MP(b),fix(a));
      mpz_neg(MP(r),MP(r));
      return normalize_big(r);
    }
  else
    if (type_of(b)==t_fixnum) {
      mpz_sub_si(MP(r),MP(a),fix(b));
      return normalize_big(r);
    } else {
      mpz_sub(MP(r),MP(a),MP(b));
      return normalize_big(r);
    }
}

static object
ratio_mult_with_cancellation(object a,object b,object c,object d) {

  object gad,gbc;

  gad=get_gcd_r(big_fixnum2,a,d);
  gbc=get_gcd_r(big_fixnum5,b,c);

  a=integer_exact_quotient(big_fixnum3,a,gad);
  c=integer_exact_quotient(big_fixnum4,c,gbc);
  a=integer_times(big_fixnum3,a,c);/*integer_times can clobber big_fixnum1*/
  if (a==big_fixnum3 || a==big_fixnum4)
    a=replace_big(a);

  b=integer_exact_quotient(big_fixnum3,b,gbc);
  d=integer_exact_quotient(big_fixnum4,d,gad);
  b=integer_times(big_fixnum3,b,d);/*integer_times can clobber big_fixnum1*/
  if (b==big_fixnum3 || b==big_fixnum4)
    b=replace_big(b);

  return make_ratio(a,b,1);

}

static object
ratio_op_with_cancellation(object a,object b,object c,object d,object (*op)(object,object,object)) {

  object b0,d0,g,t,g1;

  b0=b;
  d0=d;

  g=get_gcd_r(big_fixnum2,b,d);

  b=integer_exact_quotient(big_fixnum3,b,g);
  d=integer_exact_quotient(big_fixnum4,d,g);

  c=integer_times(big_fixnum3,b,c);/*integer_times can clobber big_fixnum1*/
  a=integer_times(big_fixnum5,a,d);/*integer_times can clobber big_fixnum1*/
  t=op(big_fixnum3,a,c);

  g1=get_gcd_r(big_fixnum2,t,g);

  t=integer_exact_quotient(big_fixnum3,t,g1);
  if (t==big_fixnum3 || t==big_fixnum4 || t==big_fixnum5)
    t=replace_big(t);

  b=integer_exact_quotient(big_fixnum2,b0,g1);
  b=integer_times(big_fixnum2,b,d);/*integer_times can clobber big_fixnum1*/
  if (b==big_fixnum2 || b==big_fixnum4)
    b=replace_big(b);

  return make_ratio(t,b,1);

}



object
number_plus(object x, object y)
{
	double dx, dy;
	object z;
	switch (type_of(x)) {
	case t_fixnum:
		switch(type_of(y)) {
		case t_fixnum:
		  return fixnum_add(fix(x),fix(y));
		case t_bignum:
		  MPOP(return, addsi,fix(x),MP(y));
		case t_ratio:
		  return ratio_op_with_cancellation(x,small_fixnum(1),
						    y->rat.rat_num,y->rat.rat_den,
						    integer_add);
		case t_shortfloat:
			dx = (double)(fix(x));
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = (double)(fix(x));
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_bignum:
		switch (type_of(y)) {
		case t_fixnum:
		  MPOP(return,addsi,fix(y),MP(x)); 
		case t_bignum:
		  MPOP(return,addii,MP(y),MP(x)); 
		case t_ratio:
		  return ratio_op_with_cancellation(x,small_fixnum(1),
						    y->rat.rat_num,y->rat.rat_den,
						    integer_add);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_ratio:
		switch (type_of(y)) {
		case t_fixnum:
		case t_bignum:
		  return ratio_op_with_cancellation(x->rat.rat_num,x->rat.rat_den,
						    y,small_fixnum(1),
						    integer_add);
		case t_ratio:
   		  return ratio_op_with_cancellation(x->rat.rat_num,x->rat.rat_den,
						    y->rat.rat_num,y->rat.rat_den,
						    integer_add);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_shortfloat:
		switch (type_of(y)) {
		case t_fixnum:
			dx = (double)(sf(x));
			dy = (double)(fix(y));
			goto SHORTFLOAT;
		case t_shortfloat:
			dx = (double)(sf(x));
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = (double)(sf(x));
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dx = (double)(sf(x));
			dy = number_to_double(y);
			goto SHORTFLOAT;
		}
	SHORTFLOAT:
		z = alloc_object(t_shortfloat);
		sf(z) = (shortfloat)(dx + dy);/*FPE*/
		return(z);

	case t_longfloat:
		dx = lf(x);
		switch (type_of(y)) {
		case t_fixnum:
			dy = (double)(fix(y));
			goto LONGFLOAT;
		case t_shortfloat:
			dy = (double)(sf(y));
			goto LONGFLOAT;
		case t_longfloat:
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dy = number_to_double(y);
			goto LONGFLOAT;
		}
	LONGFLOAT:
		z = alloc_object(t_longfloat);
		lf(z) = dx + dy;
		return(z);

	case t_complex:
	COMPLEX:
		x = number_to_complex(x);
		y = number_to_complex(y);
		z = make_complex(number_plus(x->cmp.cmp_real, y->cmp.cmp_real),
				 number_plus(x->cmp.cmp_imag, y->cmp.cmp_imag));
		return(z);

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

object
one_plus(object x)
{
	double dx;
	object z;

	
	switch (type_of(x)) {

	case t_fixnum:
	  return fixnum_add(fix(x),1);
	case t_bignum:
	  MPOP(return,addsi,1,MP(x));
	case t_ratio:
	  return ratio_op_with_cancellation(x->rat.rat_num,x->rat.rat_den,
					    small_fixnum(1),small_fixnum(1),
					    integer_add);
	case t_shortfloat:
		dx = (double)(sf(x));
		z = alloc_object(t_shortfloat);
		sf(z) = (shortfloat)(dx + 1.0);
		return(z);

	case t_longfloat:
		dx = lf(x);
		z = alloc_object(t_longfloat);
		lf(z) = dx + 1.0;
		return(z);

	case t_complex:
		z = make_complex(one_plus(x->cmp.cmp_real), x->cmp.cmp_imag);
		return(z);

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

object
number_minus(object x, object y)
{
	double dx, dy;
	object z;

	
	switch (type_of(x)) {

	case t_fixnum:
		switch(type_of(y)) {
		case t_fixnum:
		  return fixnum_sub(fix(x),fix(y));
/* 		  MPOP(return,subss,fix(x),fix(y)); */
		case t_bignum:
		  MPOP(return, subsi,fix(x),MP(y));
		case t_ratio:
		  return ratio_op_with_cancellation(x,small_fixnum(1),
						    y->rat.rat_num,y->rat.rat_den,
						    integer_sub);
		case t_shortfloat:
			dx = (double)(fix(x));
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = (double)(fix(x));
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_bignum:
		switch (type_of(y)) {
		case t_fixnum:
		  MPOP(return,subis,MP(x),fix(y));
		case t_bignum:
		  MPOP(return,subii,MP(x),MP(y));
		case t_ratio:
		  return ratio_op_with_cancellation(x,small_fixnum(1),
						    y->rat.rat_num,y->rat.rat_den,
						    integer_sub);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_ratio:
		switch (type_of(y)) {
		case t_fixnum:
		case t_bignum:
		  return ratio_op_with_cancellation(x->rat.rat_num,x->rat.rat_den,
						    y,small_fixnum(1),
						    integer_sub);
		case t_ratio:
   		  return ratio_op_with_cancellation(x->rat.rat_num,x->rat.rat_den,
						    y->rat.rat_num,y->rat.rat_den,
						    integer_sub);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_shortfloat:
		switch (type_of(y)) {
		case t_fixnum:
			dx = (double)(sf(x));
			dy = (double)(fix(y));
			goto SHORTFLOAT;
		case t_shortfloat:
			dx = (double)(sf(x));
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = (double)(sf(x));
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dx = (double)(sf(x));
			dy = number_to_double(y);
			goto SHORTFLOAT;
		}
	SHORTFLOAT:
		z = alloc_object(t_shortfloat);
		sf(z) = (shortfloat)(dx - dy);/*FPE*/
		return(z);

	case t_longfloat:
		dx = lf(x);
		switch (type_of(y)) {
		case t_fixnum:
			dy = (double)(fix(y));
			goto LONGFLOAT;
		case t_shortfloat:
			dy = (double)(sf(y));
			goto LONGFLOAT;
		case t_longfloat:
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dy = number_to_double(y);
		}
	LONGFLOAT:
		z = alloc_object(t_longfloat);
		lf(z) = dx - dy;
		return(z);

	case t_complex:
	COMPLEX:
		x = number_to_complex(x);
		y = number_to_complex(y);
		z = make_complex(number_minus(x->cmp.cmp_real, y->cmp.cmp_real),
				 number_minus(x->cmp.cmp_imag, y->cmp.cmp_imag));
		return(z);

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

object
one_minus(object x)
{
	double dx;
	object z;
	switch (type_of(x)) {

	case t_fixnum:
	  return fixnum_sub(fix(x),1);
	case t_bignum:
	  MPOP(return,addsi,-1,MP(x));
	case t_ratio:
	  return ratio_op_with_cancellation(x->rat.rat_num,x->rat.rat_den,
					    small_fixnum(1),small_fixnum(1),
					    integer_sub);
	case t_shortfloat:
		dx = (double)(sf(x));
		z = alloc_object(t_shortfloat);
		sf(z) = (shortfloat)(dx - 1.0);
		return(z);

	case t_longfloat:
		dx = lf(x);
		z = alloc_object(t_longfloat);
		lf(z) = dx - 1.0;
		return(z);

	case t_complex:
		z = make_complex(one_minus(x->cmp.cmp_real), x->cmp.cmp_imag);
		return(z);

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

object
number_negate(object x)
{
	object	z;

	switch (type_of(x)) {

	case t_fixnum:
		if(fix(x) == MOST_NEGATIVE_FIX)
		  return sSPminus_most_negative_fixnumP->s.s_dbind; /* fixnum_add(1,MOST_POSITIVE_FIX); */
		else
		  return(make_fixnum(-fix(x)));
	case t_bignum:
		return big_minus(x);
	case t_ratio:
	  return make_ratio(number_negate(x->rat.rat_num),x->rat.rat_den,1);

	case t_shortfloat:
		z = alloc_object(t_shortfloat);
		sf(z) = -sf(x);
		return(z);

	case t_longfloat:
		z = alloc_object(t_longfloat);
		lf(z) = -lf(x);
		return(z);

	case t_complex:
		z = make_complex(number_negate(x->cmp.cmp_real),
				 number_negate(x->cmp.cmp_imag));
		return(z);

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

object
number_times(object x, object y)
{  
	object z;
	double dx, dy;

	switch (type_of(x)) {

	case t_fixnum:
		switch (type_of(y)) {
		case t_fixnum:
		  return fixnum_times(fix(x),fix(y));
		case t_bignum:
		  MPOP(return,mulsi,fix(x),MP(y));
		case t_ratio:
		  return ratio_mult_with_cancellation(x,small_fixnum(1),
						      y->rat.rat_num,y->rat.rat_den);
		case t_shortfloat:
			dx = (double)(fix(x));
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = (double)(fix(x));
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_bignum:
		switch (type_of(y)) {
		case t_fixnum:
 		  MPOP(return,mulsi,fix(y),MP(x));
		case t_bignum:
		  MPOP(return,mulii,MP(y),MP(x));
		case t_ratio:
		  return ratio_mult_with_cancellation(x,small_fixnum(1),
						      y->rat.rat_num,y->rat.rat_den);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_ratio:
		switch (type_of(y)) {
		case t_fixnum:
		case t_bignum:
		  return ratio_mult_with_cancellation(x->rat.rat_num,x->rat.rat_den,
						      y,small_fixnum(1));
		case t_ratio:
		  return ratio_mult_with_cancellation(x->rat.rat_num,x->rat.rat_den,
						      y->rat.rat_num,y->rat.rat_den);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_shortfloat:
		switch (type_of(y)) {
		case t_fixnum:
			dx = (double)(sf(x));
			dy = (double)(fix(y));
			goto SHORTFLOAT;
		case t_shortfloat:
			dx = (double)(sf(x));
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = (double)(sf(x));
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dx = (double)(sf(x));
			dy = number_to_double(y);
			break;
		}
	SHORTFLOAT:
		z = alloc_object(t_shortfloat);
		sf(z) = (shortfloat)(dx * dy);/*FPE*/
		/* if (number_zerop(z) && dx && dy) FLOATING_POINT_UNDERFLOW(sLA,list(2,x,y)); */
		/* if (!ISFINITE(sf(z)) && ISFINITE(dx) && ISFINITE(dy)) FLOATING_POINT_OVERFLOW(sLA,list(2,x,y)); */
		return(z);

	case t_longfloat:
		dx = lf(x);
		switch (type_of(y)) {
		case t_fixnum:
			dy = (double)(fix(y));
			goto LONGFLOAT;
		case t_shortfloat:
			dy = (double)(sf(y));
			goto LONGFLOAT;
		case t_longfloat:
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dy = number_to_double(y);
		}
	LONGFLOAT:
		z = alloc_object(t_longfloat);
		lf(z) = dx * dy;/*FPE*/
		/* if (number_zerop(z) && dx && dy) FLOATING_POINT_UNDERFLOW(sLA,list(2,x,y)); */
		/* if (!ISFINITE(lf(z)) && ISFINITE(dx) && ISFINITE(dy)) FLOATING_POINT_OVERFLOW(sLA,list(2, x,y));*/
		return(z);

	case t_complex:
	COMPLEX:
	{
		object z1, z2, z11, z12, z21, z22;

		x = number_to_complex(x);
		y = number_to_complex(y);
		z11 = number_times(x->cmp.cmp_real, y->cmp.cmp_real);
		z12 = number_times(x->cmp.cmp_imag, y->cmp.cmp_imag);
		z21 = number_times(x->cmp.cmp_imag, y->cmp.cmp_real);
		z22 = number_times(x->cmp.cmp_real, y->cmp.cmp_imag);
		z1 =  number_minus(z11, z12);
		z2 =  number_plus(z21, z22);
		z = make_complex(z1, z2);
		return(z);
	}

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

object
number_divide(object x, object y)
{
	object z;
	double dx, dy;

	switch (type_of(x)) {

	case t_fixnum:
	case t_bignum:
		switch (type_of(y)) {
		case t_fixnum:
		case t_bignum:
/* 			if(number_zerop(y) == TRUE) */
/* 				zero_divisor(); */
/* 			if (number_minusp(y) == TRUE) { */
/* 				x = number_negate(x); */
/* 				y = number_negate(y); */
/* 			} */
/* 			z = make_ratio(x, y, 0); */
			return(make_ratio(x, y, 0));
		case t_ratio:
		  /* if(number_zerop(y->rat.rat_num)) DIVISION_BY_ZERO(sLD,list(2,x,y)); */
		  return ratio_mult_with_cancellation(x,small_fixnum(1),y->rat.rat_den,y->rat.rat_num);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_ratio:
		switch (type_of(y)) {
		case t_fixnum:
		case t_bignum:
		  /* if (number_zerop(y)) DIVISION_BY_ZERO(sLD,list(2,x,y)); */
			return ratio_mult_with_cancellation(x->rat.rat_num,x->rat.rat_den,
							    small_fixnum(1),y);
		case t_ratio:
		  return ratio_mult_with_cancellation(x->rat.rat_num,x->rat.rat_den,
						      y->rat.rat_den,y->rat.rat_num);
		case t_shortfloat:
			dx = number_to_double(x);
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = number_to_double(x);
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			FEwrong_type_argument(sLnumber, y);
		}

	case t_shortfloat:
		switch (type_of(y)) {
		case t_fixnum:
			dx = (double)(sf(x));
			dy = (double)(fix(y));
			goto SHORTFLOAT;
		case t_shortfloat:
			dx = (double)(sf(x));
			dy = (double)(sf(y));
			goto SHORTFLOAT;
		case t_longfloat:
			dx = (double)(sf(x));
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dx = (double)(sf(x));
			dy = number_to_double(y);
			goto LONGFLOAT;
		}
	SHORTFLOAT:
		z = alloc_object(t_shortfloat);
		/* if (dy == 0.0) DIVISION_BY_ZERO(sLD,list(2,x,y)); */
		sf(z) = (shortfloat)(dx / dy);/*FPE ?*/
		return(z);


	case t_longfloat:
		dx = lf(x);
		switch (type_of(y)) {
		case t_fixnum:
			dy = (double)(fix(y));
			goto LONGFLOAT;
		case t_shortfloat:
			dy = (double)(sf(y));
			goto LONGFLOAT;
		case t_longfloat:
			dy = lf(y);
			goto LONGFLOAT;
		case t_complex:
			goto COMPLEX;
		default:
			dy = number_to_double(y);
		}
	LONGFLOAT:
		z = alloc_object(t_longfloat);
		/* if (dy == 0.0) DIVISION_BY_ZERO(sLD,list(2,x,y)); */
		lf(z) = dx / dy;
		return(z);

	case t_complex:
	COMPLEX:
	{
		object z1, z2, z3;

		x = number_to_complex(x);
		y = number_to_complex(y);
		z1 = number_times(y->cmp.cmp_real, y->cmp.cmp_real);
		z2 = number_times(y->cmp.cmp_imag, y->cmp.cmp_imag);
		z3 = number_plus(z1, z2);
		/* if (number_zerop(z3 = number_plus(z1, z2))) DIVISION_BY_ZERO(sLD,list(2,x,y)); */
		z1 = number_times(x->cmp.cmp_real, y->cmp.cmp_real);
		z2 = number_times(x->cmp.cmp_imag, y->cmp.cmp_imag);
		z1 = number_plus(z1, z2);
		z = number_times(x->cmp.cmp_imag, y->cmp.cmp_real);
		z2 = number_times(x->cmp.cmp_real, y->cmp.cmp_imag);
		z2 = number_minus(z, z2);
		z1 = number_divide(z1, z3);
		z2 = number_divide(z2, z3);
		z = make_complex(z1, z2);
		return(z);
	}

	default:
		FEwrong_type_argument(sLnumber, x);
		return(Cnil);
	}
}

object
number_recip(object x) {

  switch (type_of(x)) {
  case t_fixnum:
  case t_bignum:
    return(make_ratio(small_fixnum(1), x, 1));
  case t_ratio:
    return(make_ratio(x->rat.rat_den,x->rat.rat_num, 1));
  case t_shortfloat:
    return make_shortfloat(1.0/sf(x));
  case t_longfloat:
    return make_longfloat(1.0/lf(x));
  case t_complex:
    return number_divide(make_complex(x->cmp.cmp_real,number_negate(x->cmp.cmp_imag)),
			 number_plus(number_times(x->cmp.cmp_real,x->cmp.cmp_real),
				     number_times(x->cmp.cmp_imag,x->cmp.cmp_imag)));
  default:
    FEwrong_type_argument(sLnumber, x);
    return(Cnil);
  }

}

object
integer_divide1(object x, object y,fixnum d) {
  object q;

  integer_quotient_remainder_1(x, y, &q, NULL,d);
  return(q);

}

object
integer_divide2(object x, object y,fixnum d,object *r) {
  object q;

  integer_quotient_remainder_1(x, y, &q, r,d);
  return(q);

}

DEFUN("P2",object,fSp2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_plus(x,y));

}

DEFUN("M2",object,fSs2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_minus(x,y));

}

DEFUN("*2",object,fSt2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_times(x,y));

}

DEFUN("/2",object,fSd2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_divide(x,y));

}


DEFUN("NUMBER-PLUS",object,fSnumber_plus,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_plus(x,y));

}

DEFUN("NUMBER-MINUS",object,fSnumber_minus,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_minus(x,y));

}

DEFUN("NUMBER-NEGATE",object,fSnumber_negate,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  RETURN1(number_negate(x));

}

DEFUN("NUMBER-TIMES",object,fSnumber_times,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_times(x,y));

}

DEFUN("NUMBER-DIVIDE",object,fSnumber_divide,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  RETURN1(number_divide(x,y));

}

DEFUN("NUMBER-RECIP",object,fSnumber_recip,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  RETURN1(number_recip(x));

}


LFD(Lplus)(void)
{
        fixnum i, j;
	
	j = vs_top - vs_base;
	if (j == 0) {
		vs_push(small_fixnum(0));
		return;
	}
	for (i = 0;  i < j;  i++)
		check_type_number(&vs_base[i]);
	for (i = 1;  i < j;  i++)
		vs_base[0] = number_plus(vs_base[0], vs_base[i]);
	vs_top = vs_base+1;
}

LFD(Lminus)(void)
{
	fixnum i, j;

	j = vs_top - vs_base;
	if (j == 0)
		too_few_arguments();
	for (i = 0; i < j ; i++)
		check_type_number(&vs_base[i]);
	if (j == 1) {
		vs_base[0] = number_negate(vs_base[0]);
		return;
	}
	for (i = 1;  i < j;  i++)
		vs_base[0] = number_minus(vs_base[0], vs_base[i]);
	vs_top = vs_base+1;
}

LFD(Ltimes)(void)
{
	fixnum i, j;

	j = vs_top - vs_base;
	if (j == 0) {
		vs_push(small_fixnum(1));
		return;
	}
	for (i = 0;  i < j;  i++)
		check_type_number(&vs_base[i]);
	for (i = 1;  i < j;  i++)
		vs_base[0] = number_times(vs_base[0], vs_base[i]);
	vs_top = vs_base+1;
}

LFD(Ldivide)(void)
{
	fixnum i, j;

	j = vs_top - vs_base;
	if (j == 0)
		too_few_arguments();
	for(i = 0;  i < j;  i++)
		check_type_number(&vs_base[i]);
	if (j == 1) {
		vs_base[0] = number_divide(small_fixnum(1), vs_base[0]);
		vs_top = vs_base+1;
		return;
	}
	for (i = 1; i < j; i++)
		vs_base[0] = number_divide(vs_base[0], vs_base[i]);
	vs_top = vs_base+1;
}

LFD(Lone_plus)(void)
{
	
	check_arg(1);
	check_type_number(&vs_base[0]);
	vs_base[0] = one_plus(vs_base[0]);
}

LFD(Lone_minus)(void)
{
	
	check_arg(1);
	check_type_number(&vs_base[0]);
	vs_base[0] = one_minus(vs_base[0]);
}

LFD(Lconjugate)(void)
{
	object	c, i;

	check_arg(1);
	check_type_number(&vs_base[0]);
	c = vs_base[0];
	if (type_of(c) == t_complex) {
		i = number_negate(c->cmp.cmp_imag);
		vs_push(i);
		vs_base[0] = make_complex(c->cmp.cmp_real, i);
		vs_popp;
	}
}

LFD(Lgcd)(void) {

  fixnum i, narg=vs_top-vs_base;
  
  if (narg == 0) {
    vs_push(small_fixnum(0));
    return;
  }

  for (i = 0;  i < narg;  i++)
    check_type_integer(&vs_base[i]);

  vs_top=vs_base;
  vs_push(number_abs(vs_base[0]));
  
  for (i = 1;  i < narg;  i++)
    vs_base[0] = get_gcd_abs(vs_base[0],number_abs(vs_base[i]));

}

object
get_lcm_abs(object x,object y) {

  object g=get_gcd_abs(x,y);

  return number_zerop(g) ? g : number_times(x,integer_divide1(y,g,0));

}

object
get_lcm(object x,object y) {

  return get_lcm_abs(number_abs(x),number_abs(y));

}

LFD(Llcm)(void) {

  fixnum i, narg;
  
  narg = vs_top - vs_base;

  if (narg == 0)
    too_few_arguments();

  for (i = 0;  i < narg;  i++)
    check_type_integer(&vs_base[i]);

  vs_top=vs_base;
  vs_push(number_abs(vs_base[0]));

  for (i=1;i<narg && !number_zerop(vs_base[0]);i++)
    vs_base[0]=get_lcm_abs(vs_base[0],number_abs(vs_base[i]));

}

DEFUN("FACTORIAL",object,fSfactorial,SI,1,1,NONE,OI,OO,OO,OO,(fixnum x),"") {

  object r;

  if (x<0) {
    object y=make_fixnum(x);
    TYPE_ERROR(y,sSnon_negative_fixnum);
    x=fix(y);
  }
  r=new_bignum();
  mpz_fac_ui(MP(r),x);
  RETURN1(normalize_big(r));

}
  
void
gcl_init_num_arith(void)
{
	make_function("+", Lplus);
	make_function("-", Lminus);
	make_function("*", Ltimes);
	make_function("/", Ldivide);
	make_function("1+", Lone_plus);
	make_function("1-", Lone_minus);
	make_function("CONJUGATE", Lconjugate);
	make_function("GCD", Lgcd);
	make_function("LCM", Llcm);
}
