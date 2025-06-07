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
	number.c
	IMPLEMENTATION-DEPENDENT

	This file creates some implementation dependent constants.
*/

#define IN_NUM_CO

#include "include.h"
#include "num_include.h"


long
fixint(object x)
{
	if (type_of(x) != t_fixnum)
		FEwrong_type_argument(sLfixnum, x);
	return(fix(x));
}

int
fixnnint(object x)
{
	if (type_of(x) != t_fixnum || fix(x) < 0)
		FEerror("~S is not a non-negative fixnum.", 1, x);
	return(fix(x));
}
#if 0
object small_fixnum ( int i ) {
#include <assert.h>    
    assert ( ( -SMALL_FIXNUM_LIMIT <= i ) && ( i < SMALL_FIXNUM_LIMIT ) ); 
    (object) small_fixnum_table + SMALL_FIXNUM_LIMIT + i;
}
#endif


/*FIXME, make these immutable and of type immfix*/
#define BIGGER_FIXNUM_RANGE

#ifdef BIGGER_FIXNUM_RANGE
struct {int min,max;} bigger_fixnums;

struct fixnum_struct *bigger_fixnum_table=NULL,*bigger_fixnum_table_end=NULL;
#if !defined(IM_FIX_BASE) || defined(USE_SAFE_CDR)
#define STATIC_BIGGER_FIXNUM_TABLE_BITS 10
static struct fixnum_struct bigger_fixnum_table1[1<<(STATIC_BIGGER_FIXNUM_TABLE_BITS+1)] OBJ_ALIGN;
#endif

DEFUN("ALLOCATE-BIGGER-FIXNUM-RANGE",object,fSallocate_bigger_fixnum_range,SI,2,2,NONE,OI,IO,OO,OO,(fixnum min,fixnum max),"")  {

  int j; 

  if (min > max) FEerror("Need Min <= Max",0);

#if !defined(IM_FIX_BASE) || defined(USE_SAFE_CDR)
  if (min==-(1<<STATIC_BIGGER_FIXNUM_TABLE_BITS) && max==(1<<STATIC_BIGGER_FIXNUM_TABLE_BITS)) {
    bigger_fixnum_table=bigger_fixnum_table1;
    bigger_fixnum_table_end=(void *)bigger_fixnum_table+sizeof(bigger_fixnum_table1);
  } else
#endif
    {
      bigger_fixnum_table=(void *)malloc(sizeof(struct fixnum_struct)*(max - min));
      bigger_fixnum_table_end=(void *)bigger_fixnum_table+sizeof(struct fixnum_struct)*(max - min);
    }
  
  for (j=min ; j < max ; j=j+1) { 		
    object x=(object)(bigger_fixnum_table+j-min);
    x->fw=0;
    set_type_of(x,t_fixnum);
    x->FIX.FIXVAL=j;
  }
  bigger_fixnums.min=min;
  bigger_fixnums.max=max;
  
  return Ct;
}
#endif

int
is_bigger_fixnum(void *v)  {
  return v>=(void *)bigger_fixnum_table && v<(void *)bigger_fixnum_table_end ? 1 : 0;
}

object
make_fixnum1(long i)
{
	object x;

	/* In a macro now */
/* 	if (-SMALL_FIXNUM_LIMIT <= i && i < SMALL_FIXNUM_LIMIT) */
/* 		return(small_fixnum(i)); */
#ifdef BIGGER_FIXNUM_RANGE
	if (bigger_fixnum_table)
	  { if (i >= bigger_fixnums.min
		&& i < bigger_fixnums.max)
	      return (object)(bigger_fixnum_table +(i -bigger_fixnums.min));
	  }
#endif	
	      
	x = alloc_object(t_fixnum);	    
	set_fix(x,i);
	return(x);
}

object
make_ratio(object num, object den,int pre_cancelled)
{
	object g, r, get_gcd(object x, object y);
	vs_mark;

	if (den==small_fixnum(0) /* number_zerop(den) */)
	  DIVISION_BY_ZERO(sLD,list(2,num,den));
	if (num==small_fixnum(0)/* number_zerop(num) */)
		return(num);
	if (number_minusp(den)) {
		num = number_negate(num);
		vs_push(num);
		den = number_negate(den);
		vs_push(den);
	}
	if (den==small_fixnum(1)/* type_of(den) == t_fixnum && fix(den) == 1 */)
		return(num);
	if (!pre_cancelled) {
	  g = get_gcd(num, den);
	  num = integer_divide1(num, g,0); /*FIXME exact division here*/
	  den = integer_divide1(den, g,0);
	  if(den==small_fixnum(1)/* type_of(den) == t_fixnum && fix(den) == 1 */) {
	    return(num);
	  }
	}
	r = alloc_object(t_ratio);
	r->rat.rat_num = num;
	r->rat.rat_den = den;
	vs_reset;
	return(r);
}

DEFUN("MAKE-RATIO",object,fSmake_ratio,SI,3,3,NONE,OO,OI,OO,OO,(object num,object den,fixnum pre_canceled),"")  {

  RETURN1(make_ratio(num,den,pre_canceled));

}

DEFUN("MAKE-COMPLEX",object,fSmake_complex,SI,3,3,NONE,OI,OO,OO,OO,(fixnum tt,object r,object i),"")  {
  object x=alloc_object(t_complex);
  massert(tt>=0 && tt<=5);
  x->d.tt=tt;
  x->cmp.cmp_real=r;
  x->cmp.cmp_imag=i;
  RETURN1(x);
}

object
make_shortfloat(float f)
{
	object x;

	if (f == (shortfloat)0.0)
		return(shortfloat_zero);
	x = alloc_object(t_shortfloat);
	sf(x) = (shortfloat)f;
	return(x);
}

object
make_longfloat(longfloat f)
{
	object x;

	if (f == (longfloat)0.0)
		return(longfloat_zero);
	x = alloc_object(t_longfloat);
	lf(x) = f;
	return(x);
}

object
make_complex(object r, object i)
{
	object c;
	vs_mark;

	switch (type_of(r)) {
	case t_fixnum:
	case t_bignum:
	case t_ratio:
		switch (type_of(i)) {
		case t_fixnum:
			if (fix(i) == 0)
				return(r);
			break;
		case t_shortfloat:
			r = make_shortfloat((shortfloat)number_to_double(r));
			vs_push(r);
			break;
		case t_longfloat:
			r = make_longfloat(number_to_double(r));
			vs_push(r);
			break;
		default:
		  break;
		}
		break;
	case t_shortfloat:
		switch (type_of(i)) {
		case t_fixnum:
		case t_bignum:
		case t_ratio:
			i = make_shortfloat((shortfloat)number_to_double(i));
			vs_push(i);
			break;
		case t_longfloat:
			r = make_longfloat((double)(sf(r)));
			vs_push(r);
			break;
		default:
		  break;
		}
		break;
	case t_longfloat:
		switch (type_of(i)) {
		case t_fixnum:
		case t_bignum:
		case t_ratio:
		case t_shortfloat:
			i = make_longfloat(number_to_double(i));
			vs_push(i);
			break;
		default:
		  break;
		}
		break;
	default:
	  break;
	}			
	c = alloc_object(t_complex);
	{enum type tp=type_of(r);
	  c->cmp.tt= tp==t_longfloat ? 5 :
	    (tp==t_shortfloat ? 4 :
	     (tp==t_ratio && type_of(i)==t_ratio ?  3 :
	      (tp==t_ratio ? 2 :
	       (type_of(i)==t_ratio ? 1 : 0))));
	}
	c->cmp.cmp_real = r;
	c->cmp.cmp_imag = i;
	vs_reset;
	return(c);
}

double
number_to_double(object x)
{
	switch(type_of(x)) {
	case t_fixnum:
		return((double)(fix(x)));

	case t_bignum:
		return(big_to_double(/*  (struct bignum *) */x));

	case t_ratio:
	  
	  {
	    double dx,dy;
	    object xx,yy;
	    
	    for (xx=x->rat.rat_num,yy=x->rat.rat_den,dx=number_to_double(xx),dy=number_to_double(yy);
		 dx && dy && (!ISNORMAL(dx) || !ISNORMAL(dy));) {

	      if (ISNORMAL(dx))
		dx*=0.5;
	      else {
		xx=integer_divide1(xx,small_fixnum(2),0);
		dx=number_to_double(xx);
	      }

	      if (ISNORMAL(dy))
		dy*=0.5;
	      else {
		yy=integer_divide1(yy,small_fixnum(2),0);
		dy=number_to_double(yy);
	      }

	    }

	    return dx/dy;
	  }

	case t_shortfloat:
		return((double)(sf(x)));

	case t_longfloat:
		return(lf(x));

	default:
		wrong_type_argument(TSor_rational_float, x);
		return(0.0);
	}
}

void
gcl_init_number(void) {

#if !defined(IM_FIX_BASE) || defined(USE_SAFE_CDR)
  FFN(fSallocate_bigger_fixnum_range)(-1024,1024);
#endif

  shortfloat_zero = alloc_object(t_shortfloat);
  sf(shortfloat_zero) = (shortfloat)0.0;
  longfloat_zero = alloc_object(t_longfloat);
  lf(longfloat_zero) = (longfloat)0.0;
  enter_mark_origin(&shortfloat_zero);
  enter_mark_origin(&longfloat_zero);
  
  make_constant("MOST-POSITIVE-FIXNUM",make_fixnum(MOST_POSITIVE_FIX));
  make_constant("MOST-NEGATIVE-FIXNUM",make_fixnum(MOST_NEGATIVE_FIX));

  gcl_init_big();
  gcl_init_num_pred();
  gcl_init_num_comp();
  gcl_init_num_arith();
  gcl_init_num_co();
  gcl_init_num_log();
  gcl_init_num_sfun();
  gcl_init_num_rand();

}
