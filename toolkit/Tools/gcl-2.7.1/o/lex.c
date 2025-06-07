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

	lex.c

	lexical environment
*/

#include "include.h"


object
assoc_eq(object key, object alist)
{

	while (!endp(alist)) {
		if (MMcaar(alist) == key)
			return(MMcar(alist));
		alist = MMcdr(alist);
	}
	return(Cnil);
}

void
lex_fun_bind(object name, object fun)
{
	object *top = vs_top;

	vs_push(make_cons(fun, Cnil));
	top[0] = make_cons(sLfunction, top[0]);
	top[0] = make_cons(name, top[0]);
	lex_env[1] = make_cons(top[0],lex_env[1]);
	vs_top = top;
}

void
lex_macro_bind(object name, object exp_fun)
{
	object *top = vs_top;
	vs_push(make_cons(exp_fun, Cnil));
	top[0] = make_cons(sSmacro, top[0]);
	top[0] = make_cons(name, top[0]);
	lex_env[1]=make_cons(top[0], lex_env[1]);			  
	vs_top = top;
}

void
lex_tag_bind(object tag, object id)
{
	object *top = vs_top;

	vs_push(make_cons(id, Cnil));
	top[0] = make_cons(sStag, top[0]);
	top[0] = make_cons(tag, top[0]);
	lex_env[2] =make_cons(top[0], lex_env[2]);
	vs_top = top;
}

void
lex_block_bind(object name, object id)
{
	object *top = vs_top;

	vs_push(make_cons(id, Cnil));
	top[0] = make_cons(sLblock, top[0]);
	top[0] = make_cons(name, top[0]);
	lex_env[2]= make_cons(top[0], lex_env[2]);
	vs_top = top;
}

object
lex_tag_sch(object tag)
{

	object alist = lex_env[2];

	while (!endp(alist)) {
		if (eql(MMcaar(alist), tag) && MMcadar(alist) == sStag)
			return(MMcar(alist));
		alist = MMcdr(alist);
	}
	return(Cnil);
}

object lex_block_sch(object name)
{

	object alist = lex_env[2];

	while (!endp(alist)) {
		if (MMcaar(alist) == name && MMcadar(alist) == sLblock)
			return(MMcar(alist));
		alist = MMcdr(alist);
	}
	return(Cnil);
}

void
gcl_init_lex(void)
{
/* 	sLfunction = make_ordinary("FUNCTION"); */
/* 	enter_mark_origin(&sLfunction); */
	sSmacro = make_si_ordinary("MACRO");
	enter_mark_origin(&sSmacro);
	sStag = make_si_ordinary("TAG");
	enter_mark_origin(&sStag);
	sLblock =  make_ordinary("BLOCK");
	enter_mark_origin(&sLblock);
}
