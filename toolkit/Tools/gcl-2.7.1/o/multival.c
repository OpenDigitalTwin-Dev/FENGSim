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

	multival.c

	Multiple Values
*/

#include "include.h"

LFD(Lvalues)(void)
{
	if (vs_base == vs_top) vs_base[0] = Cnil;
	if (vs_top-vs_base > MULTIPLE_VALUES_LIMIT)
	  FEerror("Too many function call values", 0);
}

LFD(Lvalues_list)(void)
{

	object x;

	check_arg(1);
	x = vs_base[0];
	vs_top = vs_base;
	while (!endp(x)) {	
		vs_push(MMcar(x));
		x = MMcdr(x);
	}
	if (vs_top == vs_base) vs_base[0] = Cnil;
	if (vs_top-vs_base > MULTIPLE_VALUES_LIMIT)
	  FEerror("Too many function call values", 0);

}

static void
FFN(Fmultiple_value_list)(object form)
{

	object *top = vs_top;

	if (endp(form))
		FEtoo_few_argumentsF(form);
	if (!endp(MMcdr(form)))
		FEtoo_many_argumentsF(form);
	vs_push(Cnil);
	eval(MMcar(form));
	while (vs_base < vs_top) {	
		top[0] = MMcons(vs_top[-1],top[0]);
		vs_top--;
	}
	vs_base = top;
	vs_top = top+1;
}

static void
FFN(Fmultiple_value_call)(object form)
{

	object *top = vs_top;
	object *top1;
	object *top2;

	if (endp(form))
		FEtoo_few_argumentsF(form);
	eval(MMcar(form));
	vs_top = top;
	vs_push(vs_base[0]);
	form = MMcdr(form);
	while (!endp(form)) {
		top1 = vs_top;
		eval(MMcar(form));
		top2 = vs_top;
		vs_top = top1;
		while (vs_base < top2) {
			vs_push(vs_base[0]);
			vs_base++;
		}
		form = MMcdr(form);
	}
	vs_base = top+1;
	super_funcall(top[0]);
}

static void
FFN(Fmultiple_value_prog1)(object forms)
{

	object *top;
	object *base = vs_top;

	if (endp(forms))
		FEtoo_few_argumentsF(forms);
	eval(MMcar(forms));
	top = vs_top;
	vs_top=base;
	while (vs_base < top) {	
		vs_push(vs_base[0]);
		vs_base++;
	}
	top = vs_top;
	forms = MMcdr(forms);
	while (!endp(forms)) {	
		eval(MMcar(forms));
		vs_top = top;
		forms = MMcdr(forms);
	}
	vs_base = base;
	vs_top = top;
	if (vs_base == vs_top) vs_base[0] = Cnil;
}

	
void
gcl_init_multival(void)
{
	make_constant("MULTIPLE-VALUES-LIMIT",make_fixnum(1+MULTIPLE_VALUES_LIMIT));
	make_function("VALUES",Lvalues);
	make_function("VALUES-LIST",Lvalues_list);
	make_special_form("MULTIPLE-VALUE-CALL",Fmultiple_value_call);
	make_special_form("MULTIPLE-VALUE-PROG1",
			  Fmultiple_value_prog1);
	make_special_form("MULTIPLE-VALUE-LIST",Fmultiple_value_list);
}
