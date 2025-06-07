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

#include "include.h"

#define	attach(x)	(vs_head = make_cons(x, vs_head))
#define	make_list	(vs_popp,vs_head=list(2,vs_head,*vs_top))


#define	QUOTE	1
#define	EVAL	2
#define	LIST	3
#define	LISTA	4
#define	APPEND	5
#define	NCONC	6

#define	siScomma_at sSYB
#define siScomma_dot sSYZ
object sSXB;
object sSYB;
object sSYZ;

static void
kwote_cdr(void)
{
	object x;

	x = vs_head;
	if (type_of(x) == t_symbol) {
		if ((enum stype)x->s.s_stype == stp_constant &&
		    x->s.s_dbind == x)
			return;
		goto KWOTE;
	} else if (consp(x) || TS_MEMBER(type_of(x),TS(t_vector)|TS(t_simple_vector)))
		goto KWOTE;
	return;

KWOTE:
	vs_head = make_cons(vs_head, Cnil);
	vs_head = make_cons(sLquote, vs_head);
}

static void
kwote_car(void)
{
	object x;

	x = vs_top[-2];
	if (type_of(x) == t_symbol) {
		if ((enum stype)x->s.s_stype == stp_constant &&
		    x->s.s_dbind == x)
			return;
		goto KWOTE;
	} else if (consp(x) || TS_MEMBER(type_of(x),TS(t_vector)|TS(t_simple_vector)))
		goto KWOTE;
	return;

KWOTE:
	vs_top[-2] = make_cons(vs_top[-2], Cnil);
	vs_top[-2] = make_cons(sLquote, vs_top[-2]);
}

/*
	Backq_cdr(x) pushes a form on vs and returns one of

		QUOTE		the form should be quoted
		EVAL		the form should be evaluated
		LIST		the form should be applied to LIST
		LISTA		the form should be applied to LIST*
		APPEND		the form should be applied to APPEND
		NCONC		the form should be applied to NCONC
*/
static int
backq_cdr(object x)
{
	int a, d;

	cs_check(x);

	if (!consp(x)) {
		vs_push(x);
		return(QUOTE);
	}
	if (x->c.c_car == siScomma) {
		vs_push(x->c.c_cdr);
		return(EVAL);
	}
	if (x->c.c_car == siScomma_at || x->c.c_car == siScomma_dot)
		FEerror(",@ or ,. has appeared in an illegal position.", 0);
	a = backq_car(x->c.c_car);
	d = backq_cdr(x->c.c_cdr);
	if (d == QUOTE)
		switch (a) {
		case QUOTE:
			vs_popp;
			vs_head = x;
			return(QUOTE);

		case EVAL:
			if (vs_head == Cnil) {
				stack_cons();
				return(LIST);
			}
			if (consp(vs_head) &&
			    vs_head->c.c_cdr == Cnil) {
				vs_head = vs_head->c.c_car;
				kwote_cdr();
				make_list;
				return(LIST);
			}
			kwote_cdr();
			make_list;
			return(LISTA);

		case APPEND:
			if (vs_head == Cnil) {
			  vs_popp;
			  if (!consp(vs_head) ||
			      (vs_head->c.c_car!=siScomma_at &&
			       vs_head->c.c_car!=siScomma_dot))
			    return(EVAL);
			  vs_push(Cnil);
			}
			kwote_cdr();
			make_list;
			return(APPEND);

		case NCONC:
			if (vs_head == Cnil) {
			  vs_popp;
			  if (!consp(vs_head) ||
			      (vs_head->c.c_car!=siScomma_at &&
			       vs_head->c.c_car!=siScomma_dot))
			    return(EVAL);
			  vs_push(Cnil);
			}
			kwote_cdr();
			make_list;
			return(NCONC);

		default:
			error("backquote botch");
		}
	if (d == EVAL)
		switch (a) {
		case QUOTE:
			kwote_car();
			make_list;
			return(LISTA);

		case EVAL:
			make_list;
			return(LISTA);

		case APPEND:
			make_list;
			return(APPEND);

		case NCONC:
			make_list;
			return(NCONC);

		default:
			error("backquote botch");
		}
	if (a == d) {
		stack_cons();
		return(d);
	}
	switch (d) {
	case LIST:
		if (a == QUOTE) {
			kwote_car();
			stack_cons();
			return(d);
		}
		if (a == EVAL) {
			stack_cons();
			return(d);
		}
		attach(sLlist);
		break;

	case LISTA:
		if (a == QUOTE) {
			kwote_car();
			stack_cons();
			return(d);
		}
		if (a == EVAL) {
			stack_cons();
			return(d);
		}
		attach(sLlistA);
		break;

	case APPEND:
		attach(sLappend);
		break;

	case NCONC:
		attach(sLnconc);
		break;

	default:
		error("backquote botch");
	}
	switch (a) {
	case QUOTE:
		kwote_car();
		make_list;
		return(LISTA);

	case EVAL:
		make_list;
		return(LISTA);

	case APPEND:
		make_list;
		return(APPEND);

	case NCONC:
		make_list;
		return(NCONC);

	default:
		error("backquote botch");
		return(0);
	}
}

/*
	Backq_car(x) pushes a form on vs and returns one of

		QUOTE		the form should be quoted
		EVAL		the form should be evaluated
		APPEND		the form should be appended
				into the outer form
		NCONC		the form should be nconc'ed
				into the outer form
*/
int
backq_car(object x)
{
	int d;

	cs_check(x);

	if (!consp(x)) {
		vs_push(x);
		return(QUOTE);
	}
	if (x->c.c_car == siScomma) {
		vs_push(x->c.c_cdr);
		return(EVAL);
	}
	if (x->c.c_car == siScomma_at) {
		vs_push(x->c.c_cdr);
		return(APPEND);
	}
	if (x->c.c_car == siScomma_dot) {
		vs_push(x->c.c_cdr);
		return(NCONC);
	}
	d = backq_cdr(x);
	switch (d) {
	case QUOTE:
		return(QUOTE);

	case EVAL:
		return(EVAL);

	case LIST:
		attach(sLlist);
		break;

	case LISTA:
		attach(sLlistA);
		break;

	case APPEND:
		attach(sLappend);
		break;

	case NCONC:
		attach(sLnconc);
		break;

	default:
		error("backquote botch");
        }
	return(EVAL);
}

static object
backq(object x)
{
	int a;

	a = backq_car(x);
	if (a == APPEND || a == NCONC)
		FEerror(",@ or ,. has appeared in an illegal position.", 0);
	if (a == QUOTE)
		kwote_cdr();
	return(vs_pop);
}

static object fLcomma_reader(object x0, object x1)
{ object w;
	object in, c;

	/* 2 args */

	in = x0;
	if (backq_level <= 0)
		READER_ERROR(in,"A comma has appeared out of a backquote.");
	c = peek_char(FALSE, in);
	if (c == code_char('@')) {
		w = siScomma_at;
		read_char(in);
	} else if (c == code_char('.')) {
		w=siScomma_dot;
		read_char(in);
	} else
		w=siScomma;
	--backq_level;
	x0 = make_cons(w,read_object(in));
	backq_level++;
	RETURN1(x0);
}

static object fLbackquote_reader(object x0, object x1)
{
	object in;

	/* 2 args */
	in = x0;
	backq_level++;
	x0 = read_object(in);
	--backq_level;
	x0 = backq(x0);
	RETURN1(x0);
}

#define	make_cf(f)	make_cfun((f), Cnil, Cnil, NULL, 0);
/* #define MAKE_AFUN(addr,n) MakeAfun(addr,F_ARGD(n,n,ONE_VAL,ARGTYPES(OO,OO,OO,OO)),0); */
#define MAKE_AFUN(addr,n) fSinit_function(Cnil,(object)addr,Cnil,Cnil,-1,0,2|(2<<6))


/* DEF_ORDINARY("Y",sSY,SI,""); */
DEF_ORDINARY("XB",sSXB,SI,"");
DEF_ORDINARY("YB",sSYB,SI,"");
DEF_ORDINARY("YZ",sSYZ,SI,"");
DEF_ORDINARY("LIST*",sLlistA,LISP,"");

DEF_ORDINARY("APPEND",sLappend,LISP,"");
DEF_ORDINARY("NCONC",sLnconc,LISP,"");
DEF_ORDINARY("APPLY",sLapply,LISP,"");
DEF_ORDINARY("VECTOR",sLvector,LISP,"");


void
gcl_init_backq(void)
{
	object r;


	r = standard_readtable;
	r->rt.rt_self['`'].rte_chattrib = cat_terminating;
	r->rt.rt_self['`'].rte_macro = MAKE_AFUN(fLbackquote_reader,2);
	r->rt.rt_self[','].rte_chattrib = cat_terminating;
	r->rt.rt_self[','].rte_macro = MAKE_AFUN(fLcomma_reader,2);

	backq_level = 0;
}
