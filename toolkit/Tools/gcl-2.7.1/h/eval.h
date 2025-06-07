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
	eval.h
*/


/*  C control stack  */

/* #define	CSSIZE		20000 */
#define	CSGETA		4000

#ifdef __ia64__
EXTER int *cs_base2;
EXTER int *cs_org2;
#endif

EXTER int *cs_base;
EXTER int *cs_org;
EXTER int *cs_limit;


/* we catch the segmentation fault and check to warn of c stack overflow */
#ifdef AV
#ifndef cs_check
#define	cs_check(something) \
	if ((int *)(&something) < cs_limit) \
		cs_overflow()
#endif
#endif
#ifdef MV



#endif

/*  bind template  */

struct bind_temp {
	object	bt_var;
	object	bt_spp;
	object	bt_init;
	object	bt_aux;
};


#define check_symbol(x) \
	if (type_of(x) != t_symbol) \
		not_a_symbol(x)

#define	check_var(x) \
	if (type_of(x) != t_symbol || \
	    (enum stype)(x)->s.s_stype == stp_constant) \
		not_a_variable(x)


#define	eval_assign(to, form)  \
{  \
	object *old_top = vs_top;  \
  \
	eval(form);  \
	to = vs_base[0];  \
	vs_top = old_top;  \
}


#define MMcall(x) ({extern int Rset;int rset=Rset;if (!rset) {ihs_check;ihs_push(x);}(*(x)->cf.cf_self)();if (!rset) ihs_pop();})
#define MMccall(x,env_top) ({extern int Rset;int rset=Rset;if (!rset) {ihs_check;ihs_push(x);}(*(x)->cc.cc_self)(env_top);if (!rset) ihs_pop();})

#define MMcons(a,d)	make_cons((a),(d))

#define MMcar(x)	(x)->c.c_car
#define MMcdr(x)	(x)->c.c_cdr
#define MMcaar(x)	(x)->c.c_car->c.c_car
#define MMcadr(x)	(x)->c.c_cdr->c.c_car
#define MMcdar(x)	(x)->c.c_car->c.c_cdr
#define MMcddr(x)	(x)->c.c_cdr->c.c_cdr
#define MMcaaar(x)	(x)->c.c_car->c.c_car->c.c_car
#define MMcaadr(x)	(x)->c.c_cdr->c.c_car->c.c_car
#define MMcadar(x)	(x)->c.c_car->c.c_cdr->c.c_car
#define MMcaddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_car
#define MMcdaar(x)	(x)->c.c_car->c.c_car->c.c_cdr
#define MMcdadr(x)	(x)->c.c_cdr->c.c_car->c.c_cdr
#define MMcddar(x)	(x)->c.c_car->c.c_cdr->c.c_cdr
#define MMcdddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_cdr
#define MMcaaaar(x)	(x)->c.c_car->c.c_car->c.c_car->c.c_car
#define MMcaaadr(x)	(x)->c.c_cdr->c.c_car->c.c_car->c.c_car
#define MMcaadar(x)	(x)->c.c_car->c.c_cdr->c.c_car->c.c_car
#define MMcaaddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_car->c.c_car
#define MMcadaar(x)	(x)->c.c_car->c.c_car->c.c_cdr->c.c_car
#define MMcadadr(x)	(x)->c.c_cdr->c.c_car->c.c_cdr->c.c_car
#define MMcaddar(x)	(x)->c.c_car->c.c_cdr->c.c_cdr->c.c_car
#define MMcadddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_cdr->c.c_car
#define MMcdaaar(x)	(x)->c.c_car->c.c_car->c.c_car->c.c_cdr
#define MMcdaadr(x)	(x)->c.c_cdr->c.c_car->c.c_car->c.c_cdr
#define MMcdadar(x)	(x)->c.c_car->c.c_cdr->c.c_car->c.c_cdr
#define MMcdaddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_car->c.c_cdr
#define MMcddaar(x)	(x)->c.c_car->c.c_car->c.c_cdr->c.c_cdr
#define MMcddadr(x)	(x)->c.c_cdr->c.c_car->c.c_cdr->c.c_cdr
#define MMcdddar(x)	(x)->c.c_car->c.c_cdr->c.c_cdr->c.c_cdr
#define MMcddddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_cdr->c.c_cdr


#define MMnull(x)	((x)==Cnil)

