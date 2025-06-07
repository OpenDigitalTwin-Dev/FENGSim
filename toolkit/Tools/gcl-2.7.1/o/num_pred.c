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
	Predicates on numbers
*/
#define NEED_MP_H
#include "include.h"
#include "num_include.h"


int
number_zerop(object x)
{
	switch (type_of(x)) {

	case t_fixnum:
		if (fix(x) == 0)
			return(1);
		else
			return(0);

	case t_bignum:
	case t_ratio:
		return(0);

	case t_shortfloat:
		if (sf(x) == 0.0)
			return(1);
		else
			return(0);

	case t_longfloat:
		if (lf(x) == 0.0)
			return(1);
		else
			return(0);

	case t_complex:
		return(number_zerop(x->cmp.cmp_real) &&
		       number_zerop(x->cmp.cmp_imag));

	default:
		FEwrong_type_argument(sLnumber, x);
		return(0);
	}
}

int
number_plusp(object x)
{
	switch (type_of(x)) {

	case t_fixnum:
		if (fix(x) > 0)
			return(1);
		else
			return(0);

	case t_bignum:
		if (big_sign(x) > 0)
			return(1);
		else
			return(0);

	case t_ratio:
		if (number_plusp(x->rat.rat_num))
			return(1);
		else
			return(0);

	case t_shortfloat:
		if (sf(x) > 0.0)
			return(1);
		else
			return(0);

	case t_longfloat:
		if (lf(x) > 0.0)
			return(1);
		else
			return(0);

	default:
		FEwrong_type_argument(TSor_rational_float,x);
		return(0);
	}
}

int
number_minusp(object x)
{
	switch (type_of(x)) {

	case t_fixnum:
		if (fix(x) < 0)
			return(1);
		else
			return(0);

	case t_bignum:
		if (big_sign(x) < 0)
			return(1);
		else
			return(0);

	case t_ratio:
		if (number_minusp(x->rat.rat_num))
			return(1);
		else
			return(0);

	case t_shortfloat:
		if (sf(x) < 0.0)
			return(1);
		else
			return(0);

	case t_longfloat:
		if (lf(x) < 0.0)
			return(1);
		else
			return(0);

	default:
		FEwrong_type_argument(TSor_rational_float,x);
		return(0);
	}
}

int
number_oddp(object x)
{
	int i=0;

	if (type_of(x) == t_fixnum)
		i = fix(x);
	else if (type_of(x) == t_bignum)
	   i = MP_LOW(MP(x),lgef(MP(x)));
	else
		FEwrong_type_argument(sLinteger, x);
	return(i & 1);
}

int
number_evenp(object x)
{
	int i=0;

	if (type_of(x) == t_fixnum)
		i = fix(x);
	else if (type_of(x) == t_bignum)
	  i = MP_LOW(MP(x),lgef(MP(x)));
	else
		FEwrong_type_argument(sLinteger, x);
	return(~i & 1);
}

LFD(Lzerop)(void)
{
	check_arg(1);
	check_type_number(&vs_base[0]);
	if (number_zerop(vs_base[0]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

LFD(Lplusp)(void)
{
	check_arg(1);
	check_type_or_rational_float(&vs_base[0]);
	if (number_plusp(vs_base[0]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

LFD(Lminusp)(void)
{
	check_arg(1);
	check_type_or_rational_float(&vs_base[0]);
	if (number_minusp(vs_base[0]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

LFD(Loddp)(void)
{
	check_arg(1);
	check_type_integer(&vs_base[0]);
	if (number_oddp(vs_base[0]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

LFD(Levenp)(void)
{
	check_arg(1);
	check_type_integer(&vs_base[0]);
	if (number_evenp(vs_base[0]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

/* this is just to force things into memory in num_co.c */
/* static void  _assure_in_memory (void *p) */
/* { */
/* ; */
/* } */

/* static int */
/* lf_eqlp(double *p, double *q) */
/* { */
/*   return *p == *q; */
/* } */


void
gcl_init_num_pred(void)
{
#ifndef GMP
        big_register_1 = new_bignum();
	ZERO_BIG(big_register_1);
 	enter_mark_origin(&big_register_1);
#endif
	make_function("ZEROP", Lzerop);
	make_function("PLUSP", Lplusp);
	make_function("MINUSP", Lminusp);
	make_function("ODDP", Loddp);
	make_function("EVENP", Levenp);
}
