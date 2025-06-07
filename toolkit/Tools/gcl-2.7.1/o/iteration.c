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

	iteration.c

*/

#include "include.h"

static void
FFN(Floop)(object form)
{

	object x;
	object *oldlex = lex_env;
	object *top;

	make_nil_block();

	if (nlj_active) {
		nlj_active = FALSE;
		frs_pop();
		lex_env = oldlex;
		return;
	}

	top = vs_top;

	for(x = form; !endp(x); x = MMcdr(x)) {
		vs_top = top;
		eval(MMcar(x));
	}
LOOP:
	/*  Just !endp(x) is replaced by x != Cnil.  */
	for(x = form;  x != Cnil;  x = MMcdr(x)) {
		vs_top = top;
		eval(MMcar(x));
	}
	goto LOOP;
}

/*
	use of VS in Fdo and FdoA:
			|	|
	     lex_env ->	| lex1	|
			| lex2	|
			| lex3	|
	     start ->	|-------|	where each bt is a bind_temp:
			|  bt1	|
			|-------|	|  var	| -- name of DO variable
			    :		|  spp	| -- T if special
			|-------|	| init	|
			|  btn	|	|  aux	| -- step-form or var (if no
			|-------|		     step-form is given)
	     end ->	| body	|
	     old_top->	|-------|	If 'spp' != T, it is NIL during
					initialization, and is the pointer to
					(var value) in lexical environment
					during the main loop.
*/

static void
do_var_list(object var_list)
{

	object is, x, y;

	for (is = var_list;  !endp(is);  is = MMcdr(is)) {
		x = MMcar(is);
           if (type_of(x)==t_symbol)
               {vs_push(x);vs_push(Cnil);vs_push(Cnil);vs_push(x);
	        continue;}
   

          


		if (!consp(x))
			FEinvalid_form("The index, ~S, is illegal.", x);
		y = MMcar(x);
		check_var(y);
		vs_push(y);
		vs_push(Cnil);
		if (endp(MMcdr(x))) {
			vs_push(Cnil);
			vs_push(y);
		} else {
			x = MMcdr(x);
			vs_push(MMcar(x));
			if (endp(MMcdr(x)))
				vs_push(y);
			else {
				x = MMcdr(x);
				vs_push(MMcar(x));
				if (!endp(MMcdr(x)))
				    FEerror("Too many forms to the index ~S.",
					    1, y);
			}
		}
	}
}

static void
FFN(Fdo)(VOL object arg)
{

	object *oldlex = lex_env;
	object *old_top;
	struct bind_temp *start, *end, *bt;
	object end_test, body;
	VOL object result;
	bds_ptr old_bds_top = bds_top;

	if (endp(arg) || endp(MMcdr(arg)))
		FEtoo_few_argumentsF(arg);
	if (endp(MMcadr(arg)))
		FEinvalid_form("The DO end-test, ~S, is illegal.",
				MMcadr(arg));

	end_test = MMcaadr(arg);
	result = MMcdadr(arg);

	make_nil_block();

	if (nlj_active) {
		nlj_active = FALSE;
		goto END;
	}

	start = (struct bind_temp *) vs_top;

	do_var_list(MMcar(arg));
	end = (struct bind_temp *)vs_top;
	body = let_bind(MMcddr(arg), start, end);
	vs_push(body);

	for (bt = start;  bt < end;  bt++)
		if ((enum stype)bt->bt_var->s.s_stype != stp_ordinary)
			bt->bt_spp = Ct;
		else if (bt->bt_spp == Cnil)
			bt->bt_spp = assoc_eq(bt->bt_var, lex_env[0]);

	old_top = vs_top;

LOOP:	/* the main loop */
	vs_top = old_top;
	eval(end_test);
	if (vs_base[0] != Cnil) {
		/* RESULT evaluation */
		if (endp(result)) {
			vs_base = vs_top = old_top;
			vs_push(Cnil);
		} else
			do {
				vs_top = old_top;
				eval(MMcar(result));
				result = MMcdr(result);
			} while (!endp(result));
		goto END;
	}

	vs_top = old_top;

	Ftagbody(body);

	/* next step */
	for (bt = start;  bt<end;  bt++) {
		if (bt->bt_aux != bt->bt_var) {
			eval_assign(bt->bt_init, bt->bt_aux);
		}
	}
	for (bt = start;  bt<end;  bt++) {
	  if (bt->bt_aux != bt->bt_var) {
	    if (bt->bt_spp == Ct)
	      bt->bt_var->s.s_dbind = bt->bt_init;
	    else
	      MMcadr(bt->bt_spp) = bt->bt_init;
	  }
	}
	goto LOOP;

END:
	bds_unwind(old_bds_top);
	frs_pop();
	lex_env = oldlex;
}

static void
FFN(FdoA)(VOL object arg)
{

	object *oldlex = lex_env;
	object *old_top;
	struct bind_temp *start, *end, *bt;
	object end_test, body;
	VOL object result;
	bds_ptr old_bds_top = bds_top;

	if (endp(arg) || endp(MMcdr(arg)))
		FEtoo_few_argumentsF(arg);
	if (endp(MMcadr(arg)))
		FEinvalid_form("The DO* end-test, ~S, is illegal.",
				MMcadr(arg));

	end_test = MMcaadr(arg);
	result = MMcdadr(arg);

	make_nil_block();

	if (nlj_active) {
		nlj_active = FALSE;
		goto END;
	}

	start = (struct bind_temp *)vs_top;
	do_var_list(MMcar(arg));
	end = (struct bind_temp *)vs_top;
	body = letA_bind(MMcddr(arg), start, end);
	vs_push(body);

	for (bt = start;  bt < end;  bt++)
		if ((enum stype)bt->bt_var->s.s_stype != stp_ordinary)
			bt->bt_spp = Ct;
		else if (bt->bt_spp == Cnil)
			bt->bt_spp = assoc_eq(bt->bt_var, lex_env[0]);

	old_top = vs_top;

LOOP:	/* the main loop */
	eval(end_test);
	if (vs_base[0] != Cnil) {
		/* RESULT evaluation */
		if (endp(result)) {
			vs_base = vs_top = old_top;
			vs_push(Cnil);
		} else
			do {
				vs_top = old_top;
				eval(MMcar(result));
				result = MMcdr(result);
			} while (!endp(result));
		goto END;
	}

	vs_top = old_top;

	Ftagbody(body);

	/* next step */
	for (bt = start;  bt < end;  bt++)
		if (bt->bt_aux != bt->bt_var) {
			if (bt->bt_spp == Ct) {
			    eval_assign(bt->bt_var->s.s_dbind, bt->bt_aux);
			} else {
			    eval_assign(MMcadr(bt->bt_spp), bt->bt_aux);
			}
		}
	goto LOOP;

END:
	bds_unwind(old_bds_top);
	frs_pop();
	lex_env = oldlex;
}

static void
FFN(Fdolist)(VOL object arg)
{

	object *oldlex = lex_env;
	object *old_top;
	struct bind_temp *start;
	object x, listform, body;
	VOL object result;
	bds_ptr old_bds_top = bds_top;

	if (endp(arg))
		FEtoo_few_argumentsF(arg);

	x = MMcar(arg);
	if (endp(x))
		FEerror("No variable.", 0);
	start = (struct bind_temp *)vs_top;
	vs_push(MMcar(x));
	vs_push(Cnil);
	vs_push(Cnil);
	vs_push(Cnil);
	x = MMcdr(x);
	if (endp(x))
		FEerror("No listform.", 0);
	listform = MMcar(x);
	x = MMcdr(x);
	if (endp(x))
		result = Cnil;
	else {
		result = MMcar(x);
		if (!endp(MMcdr(x)))
			FEerror("Too many resultforms.", 0);
	}

	make_nil_block();

	if (nlj_active) {
		nlj_active = FALSE;
		goto END;
	}

	eval_assign(start->bt_init, listform);
	body = find_special(MMcdr(arg), start, start+1,NULL); /*?*/
	vs_push(body);
	bind_var(start->bt_var, Cnil, start->bt_spp);
	if ((enum stype)start->bt_var->s.s_stype != stp_ordinary)
		start->bt_spp = Ct;
	else if (start->bt_spp == Cnil)
		start->bt_spp = assoc_eq(start->bt_var, lex_env[0]);

	old_top = vs_top;

LOOP:	/* the main loop */
	if (endp(start->bt_init)) {
		if (start->bt_spp == Ct)
			start->bt_var->s.s_dbind = Cnil;
		else
			MMcadr(start->bt_spp) = Cnil;
		eval(result);
		goto END;
	}

	if (start->bt_spp == Ct)
		start->bt_var->s.s_dbind = MMcar(start->bt_init);
	else
		MMcadr(start->bt_spp) = MMcar(start->bt_init);
	start->bt_init = MMcdr(start->bt_init);

	vs_top = old_top;

	Ftagbody(body);

	goto LOOP;

END:
	bds_unwind(old_bds_top);
	frs_pop();
	lex_env = oldlex;
}

static void
FFN(Fdotimes)(VOL object arg)
{

	object *oldlex = lex_env;
	object *old_top;
	struct bind_temp *start;
	object x, countform, body;
	VOL object result;
	bds_ptr old_bds_top = bds_top;

	if (endp(arg))
		FEtoo_few_argumentsF(arg);

	x = MMcar(arg);
	if (endp(x))
		FEerror("No variable.", 0);
	start = (struct bind_temp *)vs_top;
	vs_push(MMcar(x));
	vs_push(Cnil);
	vs_push(Cnil);
	vs_push(Cnil);
	x = MMcdr(x);
	if (endp(x))
		FEerror("No countform.", 0);
	countform = MMcar(x);
	x = MMcdr(x);
	if (endp(x))
		result = Cnil;
	else {
		result = MMcar(x);
		if (!endp(MMcdr(x)))
			FEerror("Too many resultforms.", 0);
	}

	make_nil_block();

	if (nlj_active) {
		nlj_active = FALSE;
		goto END;
	}

	eval_assign(start->bt_init, countform);
	if (type_of(start->bt_init) != t_fixnum &&
	    type_of(start->bt_init) != t_bignum)
		FEwrong_type_argument(sLinteger, start->bt_init);
	body = find_special(MMcdr(arg), start, start+1,NULL); /*?*/
	vs_push(body);
	bind_var(start->bt_var, make_fixnum(0), start->bt_spp);
	if ((enum stype)start->bt_var->s.s_stype != stp_ordinary) {
		start->bt_spp = Ct;
		x = start->bt_var->s.s_dbind;
	} else if (start->bt_spp == Cnil) {
		start->bt_spp = assoc_eq(start->bt_var, lex_env[0]);
		x = MMcadr(start->bt_spp);
	} else
		x = start->bt_var->s.s_dbind;

	old_top = vs_top;

LOOP:	/* the main loop */
	if (number_compare(x, start->bt_init) >= 0) {
		eval(result);
		goto END;
	}

	vs_top = old_top;

	Ftagbody(body);

	if (start->bt_spp == Ct)
		x = start->bt_var->s.s_dbind = one_plus(x);
	else
		x = MMcadr(start->bt_spp) = one_plus(x);

	goto LOOP;

END:
	bds_unwind(old_bds_top);
	frs_pop();
	lex_env = oldlex;
}

void
gcl_init_iteration(void)
{
	make_special_form("LOOP", Floop);
	make_special_form("DO", Fdo);
	make_special_form("DO*", FdoA);
	make_special_form("DOLIST", Fdolist);
	make_special_form("DOTIMES", Fdotimes);
}
