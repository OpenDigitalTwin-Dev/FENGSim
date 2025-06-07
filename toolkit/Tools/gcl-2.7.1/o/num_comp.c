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
	Comparisons on numbers
*/
#define NEED_MP_H
#include "include.h"
#include "num_include.h"

/*
	The value of number_compare(x, y) is

		-1	if	x < y
		0	if	x = y
		1	if	x > y.

	If x or y is complex, 0 or 1 is returned.
*/

int
number_compare(object x, object y) {

  double dx;
  static double dy;
  object q;
  enum type tx,ty;

  tx=type_of(x);
  ty=type_of(y);
  
  switch (tx) {
    
  case t_fixnum:

    switch (ty) {

    case t_fixnum:

      {
	fixnum fx=fix(x),fy=fix(y);
	return fx<fy ? -1 : (fx==fy ? 0 : 1);
      }

    case t_bignum:
      return big_sign(y) < 0 ? 1 : -1;

    case t_ratio:
      x = number_times(x, y->rat.rat_den);
      y = y->rat.rat_num;
      return(number_compare(x, y));

    case t_shortfloat:
      {
	volatile float fx=fix(x);
	dx = fx;
	dy = sf(y);
      }
      goto LONGFLOAT;

    case t_longfloat:
      dx = fix(x);
      dy = lf(y);
      goto LONGFLOAT;

    case t_complex:
      goto Y_COMPLEX;

    default:
      wrong_type_argument(sLnumber, y);

    }
    
  case t_bignum:

    switch (ty) {

    case t_fixnum:
      return big_sign(x) < 0 ? -1 : 1;

    case t_bignum:
      return cmpii(MP(x),MP(y));

    case t_ratio:
      x = number_times(x, y->rat.rat_den);
      y = y->rat.rat_num;
      return(number_compare(x, y));

    case t_shortfloat:

      if ((float)number_to_double((q=double_to_integer((double)sf(y))))==sf(y))
	return(number_compare(x,q));

      dx=number_to_double(x);
      dy=sf(y);
      goto LONGFLOAT;

    case t_longfloat:
      if (number_to_double((q=double_to_integer(lf(y))))==lf(y))
	return(number_compare(x,q));

      dx=number_to_double(x);
      dy=lf(y);
      goto LONGFLOAT;

    case t_complex:
      goto Y_COMPLEX;

    default:
      wrong_type_argument(sLnumber, y);

    }
    
  case t_ratio:

    switch (ty) {
    case t_fixnum:
    case t_bignum:

      y = number_times(y, x->rat.rat_den);
      x = x->rat.rat_num;
      return(number_compare(x, y));

    case t_ratio:
      {
	object x1,y1;
	x1=number_times(x->rat.rat_num,y->rat.rat_den);
	y1=number_times(y->rat.rat_num,x->rat.rat_den);
	return(number_compare(x1,y1));
      }

    case t_shortfloat:
      return(number_compare(x,double_to_rational(sf(y))));

    case t_longfloat:
      return(number_compare(x,double_to_rational(lf(y))));

    case t_complex:
      goto Y_COMPLEX;

    default:
      wrong_type_argument(sLnumber, y);

    }
    
  case t_shortfloat:

    dx = sf(x);
    goto LONGFLOAT0;
    
  case t_longfloat:
    dx = lf(x);

  LONGFLOAT0:

    switch (ty) {

    case t_fixnum:

      if (tx==t_shortfloat) {
	volatile float fy=fix(y);
	dy=fy;
      } else
	dy=fix(y);
      goto LONGFLOAT;

    case t_bignum:

      if (number_to_double((q=double_to_integer(dx)))==dx)
	return(number_compare(q,y));
      dy=number_to_double(y);
      goto LONGFLOAT;

    case t_ratio:
      return(number_compare(double_to_rational(dx),y));

    case t_shortfloat:
      dy = sf(y);
      goto LONGFLOAT;

    case t_longfloat:
      dy = lf(y);
      goto LONGFLOAT;

    case t_complex:
      goto Y_COMPLEX;

    default:
      break;
    }

  LONGFLOAT:

    return(dx < dy ? -1 : (dx == dy) ? 0 : 1);
    
  Y_COMPLEX:

    if (number_zerop(y->cmp.cmp_imag))
      return(number_compare(x, y->cmp.cmp_real) ? 1 : 0);
    else
      return(1);
    
  case t_complex:

    if (ty != t_complex) {
      if (number_zerop(x->cmp.cmp_imag))
	return(number_compare(x->cmp.cmp_real, y) ? 1 : 0);
      else
	return(1);
    }

    if (number_compare(x->cmp.cmp_real, y->cmp.cmp_real) == 0 &&
	number_compare(x->cmp.cmp_imag, y->cmp.cmp_imag) == 0 )
      return(0);
    else
      return(1);
    
  default:
    FEwrong_type_argument(sLnumber, x);
    return(0);

  }

}

LFD(Lall_the_same)(void)
{
	int narg, i;

	narg = vs_top - vs_base;
	if (narg == 0)
		too_few_arguments();
	for (i = 0; i < narg; i++)
		check_type_number(&vs_base[i]);
	for (i = 1; i < narg; i++)
		if (number_compare(vs_base[i-1],vs_base[i]) != 0) {
			vs_top = vs_base+1;
			vs_base[0] = Cnil;
			return;
		}
	vs_top = vs_base+1;
	vs_base[0] = Ct;
}

LFD(Lall_different)(void)
{
	int narg, i, j;

	narg = vs_top - vs_base;
	if (narg == 0)
		too_few_arguments();
	else if (narg == 1) {
		vs_base[0] = Ct;
		return;
	}
	for (i = 0; i < narg; i++)
		check_type_number(&vs_base[i]);
	for(i = 1; i < narg; i++)
		for(j = 0; j < i; j++)
			if (number_compare(vs_base[j], vs_base[i]) == 0) {
				vs_top = vs_base+1;
				vs_base[0] = Cnil;
				return;
			}
	vs_top = vs_base+1;
	vs_base[0] = Ct;
}

DEFUN("<2",object,fSl2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  RETURN1(!gcl_isnan(x) && !gcl_isnan(y) && number_compare(x,y)<0 ? Ct : Cnil);

}


DEFUN("<=2",object,fSle2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  RETURN1(!gcl_isnan(x) && !gcl_isnan(y) && number_compare(x,y)<1 ? Ct : Cnil);

}


DEFUN(">2",object,fSg2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  RETURN1(!gcl_isnan(x) && !gcl_isnan(y) && number_compare(x,y)>0 ? Ct : Cnil);

}


DEFUN(">=2",object,fSge2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  RETURN1(!gcl_isnan(x) && !gcl_isnan(y) && number_compare(x,y)>-1 ? Ct : Cnil);

}


DEFUN("=2",object,fSe2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  RETURN1(!gcl_isnan(x) && !gcl_isnan(y) && number_compare(x,y)==0 ? Ct : Cnil);

}


DEFUN("/=2",object,fSne2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  RETURN1(gcl_isnan(x) || gcl_isnan(y) || number_compare(x,y)!=0 ? Ct : Cnil);

}


DEFUN("MAX2",object,fSx2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  object z=fixnum_float_contagion(x,y);
  y=fixnum_float_contagion(y,z);

  RETURN1(number_compare(z,y)<0 ? y : z);

}


DEFUN("MIN2",object,fSm2,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  
  object z=fixnum_float_contagion(x,y);
  y=fixnum_float_contagion(y,z);

  RETURN1(number_compare(z,y)>0 ? y : z);

}




static void
Lnumber_compare(int s, int t)
{
	int narg, i;

	narg = vs_top - vs_base;
	if (narg == 0)
		too_few_arguments();
	for (i = 0; i < narg; i++) {
		check_type_or_rational_float(&vs_base[i]);
		if (gcl_isnan(vs_base[i])) {
		  vs_top = vs_base+1;
		  vs_base[0] = Cnil;
		  return;
		}
	}
	for (i = 1; i < narg; i++)
		if (s*number_compare(vs_base[i], vs_base[i-1]) < t) {
			vs_top = vs_base+1;
			vs_base[0] = Cnil;
			return;
		}
	vs_top = vs_base+1;
	vs_base[0] = Ct;
}

LFD(Lmonotonically_increasing)(void) { Lnumber_compare( 1, 1); }
LFD(Lmonotonically_decreasing)(void) { Lnumber_compare(-1, 1); }
LFD(Lmonotonically_nondecreasing)(void) { Lnumber_compare( 1, 0); }
LFD(Lmonotonically_nonincreasing)(void) { Lnumber_compare(-1, 0); }

LFD(Lmax)(void)
{
	object max;
	int narg, i;
	
	narg = vs_top - vs_base;
	if (narg == 0)
		too_few_arguments();
	for (i = 0;  i < narg;  i++)
		check_type_or_rational_float(&vs_base[i]);
	for (i = 1, max = vs_base[0];  i < narg;  i++) {
	  object x=fixnum_float_contagion(vs_base[i],max);
	  max=fixnum_float_contagion(max,vs_base[i]);
	  max = number_compare(max,x) < 0 ? x : max;
	}
	vs_top = vs_base+1;
	vs_base[0] = max;
}

LFD(Lmin)(void)
{
	object min;
	int narg, i;
	
	narg = vs_top - vs_base;
	if (narg == 0)
		too_few_arguments();
	for (i = 0;  i < narg;  i++)
		check_type_or_rational_float(&vs_base[i]);
	for (i = 1, min = vs_base[0];  i < narg;  i++) {
	  object x=fixnum_float_contagion(vs_base[i],min);
	  min=fixnum_float_contagion(min,vs_base[i]);
	  min = number_compare(min,x) > 0 ? x : min;
	}
	vs_top = vs_base+1;
	vs_base[0] = min;
}

void
gcl_init_num_comp(void)
{
	make_function("=", Lall_the_same);
	make_function("/=", Lall_different);
	make_function("<", Lmonotonically_increasing);
	make_function(">", Lmonotonically_decreasing);
	make_function("<=", Lmonotonically_nondecreasing);
	make_function(">=", Lmonotonically_nonincreasing);
	make_function("MAX", Lmax);
	make_function("MIN", Lmin);
}
