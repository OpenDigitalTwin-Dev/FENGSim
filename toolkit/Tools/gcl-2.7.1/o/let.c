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
	let.c
*/

#include "include.h"

void
let_var_list(object var_list)
{

	object x, y;

	for (x = var_list;  !endp(x);  x = x->c.c_cdr) {
		y = x->c.c_car;
		if (type_of(y) == t_symbol) {
			check_var(y);
			vs_push(y);
			vs_push(Cnil);
			vs_push(Cnil);
			vs_push(Cnil);
		} else {
			endp(y);
			check_var(y->c.c_car);
			vs_push(y->c.c_car);
			vs_push(Cnil);
			y = y->c.c_cdr;
			if (endp(y)) /*
				FEerror("No initial form to the variable ~S.",
					1, vs_top[-2]) */ ;
			else if (!endp(y->c.c_cdr))
			 FEerror("Too many initial forms to the variable ~S.",
				 1, vs_top[-2]);
			vs_push(y->c.c_car);
			vs_push(Cnil);
		}
	}
}

static void
FFN(Flet)(object form)
{

	object body;
	struct bind_temp *start;
	object *old_lex;
	bds_ptr old_bds_top;
	
	if (endp(form))
		FEerror("No argument to LET.", 0);

	old_lex = lex_env;
	lex_copy();
	old_bds_top = bds_top;

	start = (struct bind_temp *)vs_top;
	let_var_list(form->c.c_car);
	body = let_bind(form->c.c_cdr, start, (struct bind_temp *)vs_top);
	vs_top = (object *)start;
	vs_push(body);

	Fprogn(body);

	lex_env = old_lex;
	bds_unwind(old_bds_top);
}

static void
FFN(FletA)(object form)
{

	object body;
	struct bind_temp *start;
	object *old_lex;
	bds_ptr old_bds_top;
	
	if (endp(form))
		FEerror("No argument to LET*.", 0);

	old_lex = lex_env;
	lex_copy();
	old_bds_top = bds_top;

	start = (struct bind_temp *)vs_top;
	let_var_list(form->c.c_car);
	body = letA_bind(form->c.c_cdr, start, (struct bind_temp *)vs_top);
	vs_top = (object *)start;
	vs_push(body);

	Fprogn(body);

	lex_env = old_lex;
	bds_unwind(old_bds_top);
}

static void
FFN(Fmultiple_value_bind)(object form)
{

	object body, values_form, x, y;
        int n, m, i;
	object *base;
	object *old_lex;
	bds_ptr old_bds_top;
	struct bind_temp *start;
	
	if (endp(form))
		FEerror("No argument to MULTIPLE-VALUE-BIND.", 0);
	body = form->c.c_cdr;
	if (endp(body))
		FEerror("No values-form to MULTIPLE-VALUE-BIND.", 0);
	values_form = body->c.c_car;
	body = body->c.c_cdr;

	old_lex = lex_env;
	lex_copy();
	old_bds_top = bds_top;

	eval(values_form);
	base = vs_base;
	m = vs_top - vs_base;

	start = (struct bind_temp *)vs_top;
	for (n = 0, x = form->c.c_car;  !endp(x);  n++, x = x->c.c_cdr) {
		y = x->c.c_car;
		check_var(y);
		vs_push(y);
		vs_push(Cnil);
		vs_push(Cnil);
		vs_push(Cnil);
	}
	{
	 object *vt = vs_top;
	 vs_push(find_special(body, start, (struct bind_temp *)vt,NULL)); /*?*/
	}
	for (i = 0;  i < n;  i++)
		bind_var(start[i].bt_var,
			 (i < m ? base[i] : Cnil),
			 start[i].bt_spp);
	body = vs_pop;

	vs_top = vs_base = base;

	vs_push(body);
	Fprogn(body);
	lex_env = old_lex;
	bds_unwind(old_bds_top);
}

static void
FFN(Fcompiler_let)(object form)
{

	object body;
	object *old_lex;
	bds_ptr old_bds_top;
	struct bind_temp *start, *end, *bt;
	
	if (endp(form))
		FEerror("No argument to COMPILER-LET.", 0);

	body = form->c.c_cdr;

	old_lex = lex_env;
	lex_copy();
	old_bds_top = bds_top;

	start = (struct bind_temp *)vs_top;
	let_var_list(form->c.c_car);
	end = (struct bind_temp *)vs_top;
	for (bt = start;  bt < end;  bt++) {
		eval_assign(bt->bt_init, bt->bt_init);
	}
	for (bt = start;  bt < end;  bt++)
		bind_var(bt->bt_var, bt->bt_init, Ct);

	vs_top = (object *)start;

	Fprogn(body);

	lex_env = old_lex;
	bds_unwind(old_bds_top);
}

static void
FFN(Fflet)(object args)
{

	object def_list;
	object def;
	object *lex = lex_env;
	object *top = vs_top;

	vs_push(Cnil);			/*  space for each closure  */
	if (endp(args))
		FEtoo_few_argumentsF(args);
	def_list = MMcar(args);
	lex_copy();
	while (!endp(def_list)) {
	  object x;
	  def = MMcar(def_list);
	  x=MMcar(def);
	  if (type_of(x)!=t_symbol) {
	    x=ifuncall1(sSfunid_to_sym,x);
	    def=MMcons(x,MMcdr(def));
	  }
		if (endp(def) || endp(MMcdr(def)) ||
		    type_of(MMcar(def)) != t_symbol)
			FEerror("~S~%\
is an illegal function definition in FLET.",
				1, def);
		top[0] = MMcons(lex[2], def);
		top[0] = MMcons(lex[1], top[0]);
		top[0] = MMcons(lex[0], top[0]);
		top[0] = MMcons(sSlambda_block_closure, top[0]);
		{
		  top[0]=fSfset_in(Cnil,top[0],MMcar(def));
		}
		/* { */
		/*   object x=alloc_object(t_ifun); */
		/*   x->ifn.ifn_self=top[0]; */
		/*   x->ifn.ifn_name=x->ifn.ifn_call=Cnil; */
		/*   top[0]=x; */
		/* } */
		lex_fun_bind(MMcar(def), top[0]);
		def_list = MMcdr(def_list);
	}
	vs_push(find_special(MMcdr(args), NULL, NULL,NULL));
	Fprogn(vs_head);
	lex_env = lex;
}

DEF_ORDINARY("FUNID-TO-SYM",sSfunid_to_sym,SI,"");

static void
FFN(Flabels)(object args)
{

	object def_list;
	object def;
	object closure_list;
	object *lex = lex_env;
	object *top = vs_top;

        vs_push(Cnil);			/*  space for each closure  */
	vs_push(Cnil);			/*  space for closure-list  */
	if (endp(args))
		FEtoo_few_argumentsF(args);
	def_list = MMcar(args);
	lex_copy();
	while (!endp(def_list)) {
	  object x;
	  def = MMcar(def_list);
	  x=MMcar(def);
	  if (type_of(x)!=t_symbol) {
	    x=ifuncall1(sSfunid_to_sym,x);
	    def=MMcons(x,MMcdr(def));
	  }

	  if (endp(def) || endp(MMcdr(def)) ||
	      type_of(MMcar(def)) != t_symbol)
	    FEerror("~S~%\
is an illegal function definition in LABELS.",1, def);
	  top[0] = MMcons(lex[2], def); 
	  top[0] = MMcons(Cnil, top[0]);
	  top[1] = MMcons(top[0], top[1]);
	  top[0] = MMcons(lex[0], top[0]);
	  top[0] = MMcons(sSlambda_block_closure, top[0]);
	  {
	    top[0]=fSfset_in(Cnil,top[0],MMcar(def));
	  }
	  /* { */
	  /*   object x=alloc_object(t_ifun); */
	  /*   x->ifn.ifn_self=top[0]; */
	  /*   x->ifn.ifn_name=x->ifn.ifn_call=Cnil; */
	  /*   top[0]=x; */
	  /* } */
	  lex_fun_bind(MMcar(def), top[0]);
	  def_list = MMcdr(def_list);
	}
	closure_list = top[1];
	while (!endp(closure_list)) {
		MMcaar(closure_list) = lex_env[1];
		closure_list = MMcdr(closure_list);
	}
	vs_push(find_special(MMcdr(args), NULL, NULL,NULL));
	Fprogn(vs_head);
	lex_env = lex;
}

static void
FFN(Fmacrolet)(object args)
{

	object def_list;
	object def;
	object *lex = lex_env;
	object *top = vs_top;

	vs_push(Cnil);			/*  space for each macrodef  */
	if (endp(args))
		FEtoo_few_argumentsF(args);
	def_list = MMcar(args);
	lex_copy();
	while (!endp(def_list)) {
	  object x;
	  def = MMcar(def_list);
	  x=MMcar(def);
	  if (type_of(x)!=t_symbol) {
	    x=ifuncall1(sSfunid_to_sym,x);
	    def=MMcons(x,MMcdr(def));
	  }
		if (endp(def) || endp(MMcdr(def)) ||
		    type_of(MMcar(def)) != t_symbol)
			FEerror("~S~%\
is an illegal macro definition in MACROLET.",
				1, def);
		top[0] = ifuncall3(sSdefmacro_lambda,
				   MMcar(def),
				   MMcadr(def),
				   MMcddr(def));
		{
		  top[0]=fSfset_in(Cnil,top[0],MMcar(def));
		}
		/* { */
		/*   object x=alloc_object(t_ifun); */
		/*   x->ifn.ifn_self=top[0]; */
		/*   x->ifn.ifn_name=x->ifn.ifn_call=Cnil; */
		/*   top[0]=x; */
		/* } */
		lex_macro_bind(MMcar(def), top[0]);
		def_list = MMcdr(def_list);
	}
	vs_push(find_special(MMcdr(args), NULL, NULL,NULL));
	Fprogn(vs_head);
	lex_env = lex;
}

void
gcl_init_let(void)
{
	make_special_form("LET", Flet);
	make_special_form("LET*", FletA);
	make_special_form("MULTIPLE-VALUE-BIND", Fmultiple_value_bind);
	make_si_special_form("COMPILER-LET", Fcompiler_let);
	make_special_form("FLET",Fflet);
	make_special_form("LABELS",Flabels);
	make_special_form("MACROLET",Fmacrolet);
}
