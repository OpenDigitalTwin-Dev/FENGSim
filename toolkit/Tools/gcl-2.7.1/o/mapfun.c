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

	mapfun.c

	Mapping
*/

#include "include.h"

/*

Use of VS in mapfunctions:

		|	|
		|-------|
	base ->	|  fun	|
		| list1	|
		|   :	|
		|   :	|
		| listn	|
	top ->	| value	| -----	the list which should be returned
		| arg1	| --|
		|   :	|   |--	arguments to FUN.
		|   :	|   |	On call to FUN, vs_base = top+1
		| argn	| --|			vs_top  = top+n+1
		|-------|
		|	|
		   VS
*/

LFD(Lmapcar)(void)
{

	object *top = vs_top;
	object *base = vs_base;
	object x, handy;
	int n = vs_top-vs_base-1;
	int i;

	if (n <= 0)
		too_few_arguments();
	vs_push(Cnil);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			base[0] = Cnil;
			vs_top = base+1;
			vs_base = base;
			return;
		}
		vs_push(MMcar(x));
		base[i] = MMcdr(x);
	}
	handy = top[0] = MMcons(Cnil,Cnil);
LOOP:
	vs_base = top+1;
	super_funcall(base[0]);
	MMcar(handy) = vs_base[0];
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_base = top;
			vs_top = top+1;
			return;
		}
		top[i] = MMcar(x);
		base[i] = MMcdr(x);
	}
	vs_top = top+n+1;
	handy = MMcdr(handy) = MMcons(Cnil,Cnil);
	goto LOOP;
}

LFD(Lmaplist)(void)
{

	object *top = vs_top;
	object *base = vs_base;
	object x, handy;
	int n = vs_top-vs_base-1;
	int i;

	if (n <= 0)
		too_few_arguments();
	vs_push(Cnil);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			base[0] = Cnil;
			vs_top = base+1;
			vs_base = base;
			return;
		}
		vs_push(x);
		base[i] = MMcdr(x);
	}
	handy = top[0] = MMcons(Cnil,Cnil);
LOOP:
	vs_base = top+1;
	super_funcall(base[0]);
	MMcar(handy) = vs_base[0];
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_base = top;
			vs_top = top+1;
			return;
		}
		top[i] = x;
		base[i] = MMcdr(x);
	}
	vs_top = top+n+1;
	handy = MMcdr(handy) = MMcons(Cnil,Cnil);
	goto LOOP;
}

LFD(Lmapc)(void)
{

	object *top = vs_top;
	object *base = vs_base;
	object x;
	int n = vs_top-vs_base-1;
	int i;

	if (n <= 0)
		too_few_arguments();
	vs_push(base[1]);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_top = top+1;
			vs_base = top;
			return;
		}
		vs_push(MMcar(x));
		base[i] = MMcdr(x);
	}
LOOP:
	vs_base = top+1;
	super_funcall(base[0]);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_base = top;
			vs_top = top+1;
			return;
		}
		top[i] = MMcar(x);
		base[i] = MMcdr(x);
	}
	vs_top = top+n+1;
	goto LOOP;
}

LFD(Lmapl)(void)
{

	object *top = vs_top;
	object *base = vs_base;
	object x;
	int n = vs_top-vs_base-1;
	int i;

	if (n <= 0)
		too_few_arguments();
	vs_push(base[1]);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_top = top+1;
			vs_base = top;
			return;
		}
		vs_push(x);
		base[i] = MMcdr(x);
	}
LOOP:
	vs_base = top+1;
	super_funcall(base[0]);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_base = top;
			vs_top = top+1;
			return;
		}
		top[i] = x;
		base[i] = MMcdr(x);
	}
	vs_top = top+n+1;
	goto LOOP;
}

LFD(Lmapcan)(void)
{

	object *top = vs_top;
	object *base = vs_base;
	object x, handy;
	int n = vs_top-vs_base-1;
	int i;

	if (n <= 0)
		too_few_arguments();
	vs_push(Cnil);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			base[0] = Cnil;
			vs_top = base+1;
			vs_base = base;
			return;
		}
		vs_push(MMcar(x));
		base[i] = MMcdr(x);
	}
	handy = Cnil;
LOOP:
	vs_base = top+1;
	super_funcall(base[0]);
	if (endp(handy)) handy = top[0] = vs_base[0];
	else {
		x = MMcdr(handy);
		while(!endp(x)) {
			handy = x;
			x = MMcdr(x);
		}
		MMcdr(handy) = vs_base[0];
		}
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_base = top;
			vs_top = top+1;
			return;
		}
		top[i] = MMcar(x);
		base[i] = MMcdr(x);
	}
	vs_top = top+n+1;
	goto LOOP;
}

LFD(Lmapcon)(void)
{

	object *top = vs_top;
	object *base = vs_base;
	object x, handy;
	int n = vs_top-vs_base-1;
	int i;

	if (n <= 0)
		too_few_arguments();
	vs_push(Cnil);
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			base[0] = Cnil;
			vs_top = base+1;
			vs_base = base;
			return;
		}
		vs_push(x);
		base[i] = MMcdr(x);
	}
	handy = Cnil;
LOOP:
	vs_base = top+1;
	super_funcall(base[0]);
	if (endp(handy))
		handy = top[0] = vs_base[0];
	else {
		x = MMcdr(handy);
		while(!endp(x)) {
			handy = x;
			x = MMcdr(x);
		}
		MMcdr(handy) = vs_base[0];
	}
	for (i = 1;  i <= n;  i++) {
		x = base[i];
		if (endp(x)) {
			vs_base = top;
			vs_top = top+1;
			return;
		}
		top[i] = x;
		base[i] = MMcdr(x);
	}
	vs_top = top+n+1;
	goto LOOP;
}

void
gcl_init_mapfun(void)
{
	make_function("MAPCAR", Lmapcar);
	make_function("MAPLIST", Lmaplist);
	make_function("MAPC", Lmapc);
	make_function("MAPL", Lmapl);
	make_function("MAPCAN", Lmapcan);
	make_function("MAPCON", Lmapcon);
}
