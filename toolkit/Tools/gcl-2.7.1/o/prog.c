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

	prog.c
*/

#include "include.h"

/*
	use of VS in tagbody:

	     old_top ->	|  id	|
			| lex0	|
			| lex1	|
			| lex2	|
	   tinf_base ->	| tag1	|	where 'bodyi' is the part of tag-body
			| body1	|	that follows the tag 'tagi'.
			|   :	|
			    :		i.e.
			|   :	|	tag-body
			| tagn	|	= (...tag1..........tagn.............)
			| bodyn	|		  |		|<- bodyn ->|
	     new_top ->	|	|		  |			    |
			   VS			  |<-------- body1 -------->|
*/

FFD(Ftagbody)(object body)
{

	object *old_top = vs_top;
	object * VOL new_top;
	object *tinf;
	object * VOL tinf_base;
	object *env = lex_env;
	object id = alloc_frame_id();
	VOL object bodysv = body;
	object label;
	enum type item_type;

	vs_push(id);
	lex_copy();
	tinf_base = vs_top;
	while (!endp(body)) {
		label = MMcar(body);
		item_type = type_of(label);
		if (item_type == t_symbol || item_type == t_fixnum ||
	            item_type == t_bignum) {
			lex_tag_bind(label, id);
			vs_push(label);
			vs_push(MMcdr(body));
		}
		body = MMcdr(body);
	}

	new_top = vs_top;

	frs_push(FRS_CATCH, id);
	body = bodysv;
	if (nlj_active) {
		label = cdr(nlj_tag);
		nlj_active = FALSE;
		for(tinf = tinf_base;
		    tinf < new_top && !eql(tinf[0],label);
		    tinf += 2)
			;
		if (tinf >= new_top)
			FEerror("Someone tried to RETURN-FROM a TAGBODY.",0);
		body = tinf[1];
	}
	while (body != Cnil) {
		vs_top = new_top;
		item_type = type_of(MMcar(body));
		if (item_type != t_symbol && item_type != t_fixnum &&
		    item_type != t_bignum)
			eval(MMcar(body));
		body = MMcdr(body);
	}
	frs_pop();
	lex_env = env;
	vs_base = old_top;
	vs_top = old_top+1;
	vs_base[0] = Cnil;
}

static void
FFN(Fprog)(VOL object arg)
{

	object *oldlex = lex_env;
	struct bind_temp *start;
	object body;
	bds_ptr old_bds_top = bds_top;

	if (endp(arg))
		FEtoo_few_argumentsF(arg);

	make_nil_block();

	if (nlj_active) {
		nlj_active = FALSE;
		goto END;
	}

	start = (struct bind_temp *)vs_top;
	let_var_list(arg->c.c_car);
	body = let_bind(arg->c.c_cdr, start, (struct bind_temp *)vs_top);
	vs_top = (object *)start;
	vs_push(body);

	Ftagbody(body);

END:
	bds_unwind(old_bds_top);
	frs_pop();
	lex_env = oldlex;
}

static void
FFN(FprogA)(VOL object arg)
{

	object *oldlex = lex_env;
	struct bind_temp *start;
	object body;
	bds_ptr old_bds_top = bds_top;

	if (endp(arg))
		FEtoo_few_argumentsF(arg);

	make_nil_block();

	if (nlj_active) {
		nlj_active = FALSE;
		goto END;
	}

	start = (struct bind_temp *) vs_top;
	let_var_list(arg->c.c_car);
	body = letA_bind(arg->c.c_cdr, start, (struct bind_temp *)vs_top);
	vs_top = (object *)start;
	vs_push(body);

	Ftagbody(body);

END:
	bds_unwind(old_bds_top);
	frs_pop();
	lex_env = oldlex;
}

static void
FFN(Fgo)(object args)
{

	object lex_tag;
	frame_ptr fr;
	if (endp(args))
		FEtoo_few_argumentsF(args);
	if (!endp(MMcdr(args)))
		FEtoo_many_argumentsF(args);
	lex_tag = lex_tag_sch(MMcar(args));
	if (MMnull(lex_tag))
/* 		FEerror("~S is an undefined tag.", 1, MMcar(args)); */
		PROGRAM_ERROR("~S is an undefined tag.", MMcar(args));
	fr = frs_sch(MMcaddr(lex_tag));
	if (fr == NULL)
		FEerror("The tag ~S is missing.", 1, MMcar(args));
	vs_push(MMcons(MMcaddr(lex_tag), MMcar(lex_tag)));
	vs_base = vs_top;
	unwind(fr,vs_top[-1]);
	/*  never reached  */
}

static void
FFN(Fprogv)(object args)
{

	object *top;
	object symbols;
	object values;
	bds_ptr old_bds_top;
	object var;

	if (endp(args) || endp(MMcdr(args)))
 		FEtoo_few_argumentsF(args);

	old_bds_top=bds_top;

	top=vs_top;
	eval(MMcar(args));
	vs_top=top;
	symbols=vs_base[0];
	vs_push(symbols);
	eval(MMcadr(args));
	vs_top=top+1;
	values=vs_base[0];
	vs_push(values);
	while (!endp(symbols)) {
		var = MMcar(symbols);

		if (type_of(var)!=t_symbol) not_a_symbol(var);
		if ((enum stype)var->s.s_stype == stp_constant)
			FEerror("Cannot bind the constant ~S.", 1, var);

		if (endp(values)) {
			bds_bind(var, OBJNULL);
		} else {
			bds_bind(var, MMcar(values));
			values=MMcdr(values);
		}
		symbols=MMcdr(symbols);
	}

	Fprogn(MMcddr(args));

	bds_unwind(old_bds_top);
}

FFD(Fprogn)(object body)
{

	if(endp(body)) {
		vs_base=vs_top;
		vs_push(Cnil);
	} else {
		object *top=vs_top;
		do {
			vs_top=top;
			eval(MMcar(body));
			body=MMcdr(body);
		} while (!endp(body));
	}
}

static void
FFN(Fprog1)(object arg)
{

	object *top = vs_top;

	if(endp(arg))
		FEtoo_few_argumentsF(arg);
	eval(MMcar(arg));
	vs_top = top;
	vs_push(vs_base[0]);
	for(arg = MMcdr(arg);  !endp(arg);  vs_top = top+1, arg = MMcdr(arg))
		eval(MMcar(arg));
	vs_base = top;
	vs_top = top + 1;
}

static void
FFN(Fprog2)(object arg)
{

	object *top = vs_top;

	if(endp(arg) || endp(MMcdr(arg)))
		FEtoo_few_argumentsF(arg);
	eval(MMcar(arg));
	vs_top = top;
	arg = MMcdr(arg);
	eval(MMcar(arg));
	vs_top = top;
	vs_push(vs_base[0]);
	for(arg = MMcdr(arg);  !endp(arg);  vs_top = top+1, arg = MMcdr(arg))
		eval(MMcar(arg));
	vs_base = top;
	vs_top = top+1;
}

void
gcl_init_prog(void)
{
	make_special_form("TAGBODY", Ftagbody);
	make_special_form("PROG", Fprog);
	make_special_form("PROG*", FprogA);
	make_special_form("GO", Fgo);

	make_special_form("PROGV", Fprogv);

	sLprogn=make_special_form("PROGN",Fprogn);
	make_special_form("PROG1",Fprog1);
	make_special_form("PROG2",Fprog2);
}
