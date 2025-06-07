/*
 Copyright (C) 1994 M. Hagiya, W. sLchelter, T. Yuasa
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

	block.c

	blocks and exits
*/

#include "include.h"

static void
FFN(Fblock)(VOL object args)
{
	object *oldlex = lex_env;
	object id;
	object body;
	object *top;

	if(endp(args))
		FEtoo_few_argumentsF(args);
	lex_copy();
	id = alloc_frame_id();
	vs_push(id);
	lex_block_bind(MMcar(args), id);
	vs_popp;
	frs_push(FRS_CATCH, id);
	if (nlj_active)
		nlj_active = FALSE;
	else {
		body = MMcdr(args);
		if (endp(body)) {
			vs_base = vs_top;
			vs_push(Cnil);
		} else {
			top = vs_top;
			do {
				vs_top = top;
				eval(MMcar(body));
				body = MMcdr(body);
			} while (!endp(body));
		}
	}
	frs_pop();
	lex_env = oldlex;
}

static void
FFN(Freturn_from)(object args)
{
	object lex_block;
	frame_ptr fr;

	if (endp(args))
		FEtoo_few_argumentsF(args);
	if (!endp(MMcdr(args)) && !endp(MMcddr(args)))
		FEtoo_many_argumentsF(args);
	lex_block = lex_block_sch(MMcar(args));
	if (MMnull(lex_block))
		FEerror("The block name ~S is undefined.", 1, MMcar(args));
	fr = frs_sch(MMcaddr(lex_block));
	if(fr == NULL)
		FEerror("The block ~S is missing.", 1, MMcar(args));
	if(endp(MMcdr(args))) {
		vs_base = vs_top;
		vs_push(Cnil);
	}
	else
		eval(MMcadr(args));
	unwind(fr, MMcaddr(lex_block));
	/*  never reached  */
}

static void
FFN(Freturn)(object args)
{
	object lex_block;
	frame_ptr fr;

	if(!endp(args) && !endp(MMcdr(args)))
		FEtoo_many_argumentsF(args);
	lex_block = lex_block_sch(Cnil);
	if (MMnull(lex_block))
 		FEerror("The block name ~S is undefined.", 1, Cnil);
	fr = frs_sch(MMcaddr(lex_block));
	if (fr == NULL)
		FEerror("The block ~S is missing.", 1, Cnil);
	if(endp(args)) {
		vs_base = vs_top;
		vs_push(Cnil);
	} else
		eval(MMcar(args));
	unwind(fr, MMcaddr(lex_block));
	/*  never reached  */
}

void
gcl_init_block(void)
{
	sLblock = make_special_form("BLOCK", Fblock);
	enter_mark_origin(&sLblock);
	make_special_form("RETURN-FROM", Freturn_from);
	make_special_form("RETURN", Freturn);
}
