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
	bind.c
*/

#include <string.h>

#include "include.h"

static void
illegal_lambda(void);


struct nil3 { object nil3_self[3]; } three_nils;
struct nil6 { object nil6_self[6]; } six_nils;

struct required {
	object	req_var;
	object	req_spp;
};

struct optional {
	object	opt_var;
	object	opt_spp;
	object	opt_init;
	object	opt_svar;
	object	opt_svar_spp;
};

struct rest {
	object	rest_var;
	object	rest_spp;
};

struct keyword {
	object	key_word;
	object	key_var;
	object	key_spp;
	object	key_init;
	object	key_svar;
	object	key_svar_spp;
	object	key_val;
	object	key_svar_val;
};

struct aux {
	object	aux_var;
	object	aux_spp;
	object	aux_init;
};





#define	isdeclare(x)	((x) == sLdeclare)

void
lambda_bind(object *arg_top)
{
  
        object temporary;
	object lambda, lambda_list, body, form=Cnil, x, ds, vs, v;
	int narg, i, j;
	object *base = vs_base;
	struct required *required;
	int nreq;
	struct optional *optional=NULL;
	int nopt;
	struct rest *rest=NULL;
	bool rest_flag;
	struct keyword *keyword=NULL;
	bool key_flag;
	bool allow_other_keys_flag, other_keys_appeared;
	int nkey;
	struct aux *aux=NULL;
	int naux;
	bool special_processed;
	object s[1],ss;
	vs_mark;

	bds_check;
	lambda = vs_head;
	if (!consp(lambda))
		FEerror("No lambda list.", 0);
	lambda_list = lambda->c.c_car;
	body = lambda->c.c_cdr;

	required = (struct required *)vs_top;
	nreq = 0;
	s[0]=Cnil;
	for (;;) {
		if (endp(lambda_list))
			goto REQUIRED_ONLY;
		x = lambda_list->c.c_car;
		lambda_list = lambda_list->c.c_cdr;
		check_symbol(x);
		if (x == ANDallow_other_keys)
			illegal_lambda();
		if (x == ANDoptional) {
			nopt = nkey = naux = 0;
			rest_flag = key_flag = allow_other_keys_flag
			= FALSE;
			goto OPTIONAL;
		}
		if (x == ANDrest) {
			nopt = nkey = naux = 0;
			key_flag = allow_other_keys_flag
			= FALSE;
			goto REST;
		}
		if (x == ANDkey) {
			nopt = nkey = naux = 0;
			rest_flag = allow_other_keys_flag
			= FALSE;
			goto KEYWORD;
		}
		if (x == ANDaux) {
			nopt = nkey = naux = 0;
			rest_flag = key_flag = allow_other_keys_flag
			= FALSE;
			goto AUX_L;
		}
		if ((enum stype)x->s.s_stype == stp_constant)
			FEerror("~S is not a variable.", 1, x);
		vs_push(x);
		vs_push(Cnil);
		nreq++;
	}

OPTIONAL:
	optional = (struct optional *)vs_top;
	for (;;  nopt++) {
		if (endp(lambda_list))
			goto SEARCH_DECLARE;
		x = lambda_list->c.c_car;
		lambda_list = lambda_list->c.c_cdr;
		if (consp(x)) {
			check_symbol(x->c.c_car);
			check_var(x->c.c_car);
			vs_push(x->c.c_car);
			x = x->c.c_cdr;
			vs_push(Cnil);
			if (endp(x)) {
				*(struct nil3 *)vs_top = three_nils;
				vs_top += 3;
				continue;
			}
			vs_push(x->c.c_car);
			x = x->c.c_cdr;
			if (endp(x)) {
				vs_push(Cnil);
				vs_push(Cnil);
				continue;
			}
			check_symbol(x->c.c_car);
			check_var(x->c.c_car);
			vs_push(x->c.c_car);
			vs_push(Cnil);
			if (!endp(x->c.c_cdr))
				illegal_lambda();
		} else {
			check_symbol(x);
			if (x == ANDoptional ||
			    x == ANDallow_other_keys)
				illegal_lambda();
			if (x == ANDrest)
				goto REST;
			if (x == ANDkey)
				goto KEYWORD;
			if (x == ANDaux)
				goto AUX_L;
			check_var(x);
			vs_push(x);
			*(struct nil6 *)vs_top = six_nils;
			vs_top += 4;
		}
	}

REST:
	rest = (struct rest *)vs_top;
	if (endp(lambda_list))
		illegal_lambda();
	check_symbol(lambda_list->c.c_car);
	check_var(lambda_list->c.c_car);
	rest_flag = TRUE;
	vs_push(lambda_list->c.c_car);
	vs_push(Cnil);
	lambda_list = lambda_list->c.c_cdr;
	if (endp(lambda_list))
		goto SEARCH_DECLARE;
	x = lambda_list->c.c_car;
	lambda_list = lambda_list->c.c_cdr;
	check_symbol(x);
	if (x == ANDoptional || x == ANDrest ||
	    x == ANDallow_other_keys)
		illegal_lambda();
	if (x == ANDkey)
		goto KEYWORD;
	if (x == ANDaux)
		goto AUX_L;
	illegal_lambda();

KEYWORD:
	keyword = (struct keyword *)vs_top;
	key_flag = TRUE;
	for (;;  nkey++) {
		if (endp(lambda_list))
			goto SEARCH_DECLARE;
		x = lambda_list->c.c_car;
		lambda_list = lambda_list->c.c_cdr;
		if (consp(x)) {
			if (consp(x->c.c_car)) {
				if (type_of(x->c.c_car->c.c_car)!=t_symbol)
				  /* FIXME better message */
					FEunexpected_keyword(x->c.c_car->c.c_car);
				vs_push(x->c.c_car->c.c_car);
				if (endp(x->c.c_car->c.c_cdr))
					illegal_lambda();
				check_symbol(x->c.c_car
					      ->c.c_cdr->c.c_car);
				vs_push(x->c.c_car->c.c_cdr->c.c_car);
				if (!endp(x->c.c_car->c.c_cdr->c.c_cdr))
					illegal_lambda();
			} else {
				check_symbol(x->c.c_car);
				check_var(x->c.c_car);
				vs_push(intern(x->c.c_car, keyword_package));
				vs_push(x->c.c_car);
			}
			vs_push(Cnil);
			x = x->c.c_cdr;
			if (endp(x)) {
				*(struct nil6 *)vs_top = six_nils;
				vs_top += 5;
				continue;
			}
			vs_push(x->c.c_car);
			x = x->c.c_cdr;
			if (endp(x)) {
				*(struct nil6 *)vs_top = six_nils;
				vs_top += 4;
				continue;
			}
			check_symbol(x->c.c_car);
			check_var(x->c.c_car);
			vs_push(x->c.c_car);
			vs_push(Cnil);
			if (!endp(x->c.c_cdr))
				illegal_lambda();
			vs_push(Cnil);
			vs_push(Cnil);
		} else {
			check_symbol(x);
			if (x == ANDallow_other_keys) {
				allow_other_keys_flag = TRUE;
				if (endp(lambda_list))
					goto SEARCH_DECLARE;
				x = lambda_list->c.c_car;
				lambda_list = lambda_list->c.c_cdr;
			}
			if (x == ANDoptional || x == ANDrest ||
			    x == ANDkey || x == ANDallow_other_keys)
				illegal_lambda();
			if (x == ANDaux)
				goto AUX_L;
			check_var(x);
			vs_push(intern(x, keyword_package));
			vs_push(x);
			*(struct nil6 *)vs_top = six_nils;
			vs_top += 6;
		}
	}

AUX_L:
	aux = (struct aux *)vs_top;
	for (;;  naux++) {
		if (endp(lambda_list))
			goto SEARCH_DECLARE;
		x = lambda_list->c.c_car;
		lambda_list = lambda_list->c.c_cdr;
		if (consp(x)) {
			check_symbol(x->c.c_car);
			check_var(x->c.c_car);
			vs_push(x->c.c_car);
			vs_push(Cnil);
			x = x->c.c_cdr;
			if (endp(x)) {
				vs_push(Cnil);
				continue;
			}
			vs_push(x->c.c_car);
			if (!endp(x->c.c_cdr))
				illegal_lambda();
		} else {
			check_symbol(x);
			if (x == ANDoptional || x == ANDrest ||
			    x == ANDkey || x == ANDallow_other_keys ||
	    		    x == ANDaux)
				illegal_lambda();
			check_var(x);
			vs_push(x);
			vs_push(Cnil);
			vs_push(Cnil);
		}
	}

SEARCH_DECLARE:
	vs_push(Cnil);
	for (;  !endp(body);  body = body->c.c_cdr) {
		form = body->c.c_car;

		/*  MACRO EXPANSION  */
		form = macro_expand(form);
		vs_head = form;

		if (stringp(form)) {
			if (endp(body->c.c_cdr))
				break;
			continue;
		}
		if (!consp(form) || !isdeclare(form->c.c_car))
			break;
		for (ds = form->c.c_cdr; !endp(ds); ds = ds->c.c_cdr) {
			if (!consp(ds->c.c_car))
				illegal_declare(form);
			if (ds->c.c_car->c.c_car == sLspecial) {
				vs = ds->c.c_car->c.c_cdr;
				for (;  !endp(vs);  vs = vs->c.c_cdr) {
					v = vs->c.c_car;
					check_symbol(v);
/**/

	special_processed = FALSE;
	for (i = 0;  i < nreq;  i++)
		if (required[i].req_var == v) {
			required[i].req_spp = Ct;
			special_processed = TRUE;
		}
	for (i = 0;  i < nopt;  i++)
		if (optional[i].opt_var == v) {
			optional[i].opt_spp = Ct;
			special_processed = TRUE;
		} else if (optional[i].opt_svar == v) {
			optional[i].opt_svar_spp = Ct;
			special_processed = TRUE;
		} /* else if (optional[i].opt_init == v) */
/* 		  special_processed = TRUE; */
		
	if (rest_flag && rest->rest_var == v) {
		rest->rest_spp = Ct;
		special_processed = TRUE;
	}
	for (i = 0;  i < nkey;  i++)
		if (keyword[i].key_var == v) {
			keyword[i].key_spp = Ct;
			special_processed = TRUE;
		} else if (keyword[i].key_svar == v) {
			keyword[i].key_svar_spp = Ct;
			special_processed = TRUE;
		} /* else if (keyword[i].key_init == v) */
/* 		  special_processed = TRUE; */
	for (i = 0;  i < naux;  i++)
		if (aux[i].aux_var == v) {
			aux[i].aux_spp = Ct;
			special_processed = TRUE;
		} /* else if (aux[i].aux_init == v) */
/* 		  special_processed = TRUE; */
	if (special_processed)
		continue;
	s[0] = MMcons(MMcons(v, Cnil), s[0]);

/**/
				}
			}
		}
	}

	narg = arg_top - base;
	if (narg < nreq) {
		if (nopt == 0 && !rest_flag && !key_flag) {
			vs_base = base;
			vs_top = arg_top;
			check_arg_failed(nreq);
		}
		FEtoo_few_arguments(base, arg_top);
	}
	if (!rest_flag && !key_flag && narg > nreq+nopt) {
		if (nopt == 0) {
			vs_base = base;
			vs_top = arg_top;
			check_arg_failed(nreq);
		}
		FEtoo_many_arguments(base, arg_top);
	}
	for (i = 0;  i < nreq;  i++)
		bind_var(required[i].req_var,
			 base[i],
			 required[i].req_spp);
	for (i = 0;  i < nopt;  i++)
		if (nreq+i < narg) {
			bind_var(optional[i].opt_var,
				 base[nreq+i],
				 optional[i].opt_spp);
			if (optional[i].opt_svar != Cnil)
				bind_var(optional[i].opt_svar,
					 Ct,
					 optional[i].opt_svar_spp);
		} else {
			eval_assign(temporary, optional[i].opt_init);
			bind_var(optional[i].opt_var,
				 temporary,
				 optional[i].opt_spp);
			if (optional[i].opt_svar != Cnil)
				bind_var(optional[i].opt_svar,
					 Cnil,
					 optional[i].opt_svar_spp);
		}
	if (rest_flag) {
	  object *l=vs_top++;
	  for (i=nreq+nopt;i<narg;i++)
	    collect(l,make_cons(base[i],Cnil));
	  *l=Cnil;
	  bind_var(rest->rest_var, vs_head, rest->rest_spp);
	}
	if (key_flag) {
                int allow_other_keys_found=0;
		i = narg - nreq - nopt;
		if (i >= 0 && i%2 != 0)
		  /* FIXME better message */
		  FEunexpected_keyword(Cnil);
		other_keys_appeared = FALSE;
		for (i = nreq + nopt;  i < narg;  i += 2) {
			if (type_of(base[i])!=t_symbol)
				FEunexpected_keyword(base[i]);
			if (base[i] == sKallow_other_keys && !allow_other_keys_found) {
			    allow_other_keys_found=1;
			    if (base[i+1] != Cnil)
				allow_other_keys_flag = TRUE;
                        }
			for (j = 0;  j < nkey;  j++) {
				if (keyword[j].key_word == base[i]) {
					if (keyword[j].key_svar_val
					    != Cnil)
						goto NEXT_ARG;
					keyword[j].key_val
					= base[i+1];
					keyword[j].key_svar_val
					= Ct;
					goto NEXT_ARG;
				}
			}
                        if (base[i] != sKallow_other_keys)
			  other_keys_appeared = TRUE;

		NEXT_ARG:
			continue;
		}
		if (other_keys_appeared && !allow_other_keys_flag)
		  /* FIXME better message */
		  FEunexpected_keyword(Ct);
	}
	for (i = 0;  i < nkey;  i++)
		if (keyword[i].key_svar_val != Cnil) {
			bind_var(keyword[i].key_var,
				 keyword[i].key_val,
				 keyword[i].key_spp);
			if (keyword[i].key_svar != Cnil)
				bind_var(keyword[i].key_svar,
					 keyword[i].key_svar_val,
					 keyword[i].key_svar_spp);
		} else {
			eval_assign(temporary, keyword[i].key_init);
			bind_var(keyword[i].key_var,
				 temporary,
				 keyword[i].key_spp);
			if (keyword[i].key_svar != Cnil)
				bind_var(keyword[i].key_svar,
					 keyword[i].key_svar_val,
					 keyword[i].key_svar_spp);
		}
	for (i = 0;  i < naux;  i++) {
		eval_assign(temporary, aux[i].aux_init);
		bind_var(aux[i].aux_var, temporary, aux[i].aux_spp);
	}
	if (!consp(body) || body->c.c_car == form) {
		vs_reset;
		vs_head = body;
	} else {
		body = make_cons(form, body->c.c_cdr);
		vs_reset;
		vs_head = body;
	}

	if (s[0]!=Cnil) {
	  for (ss=s[0];ss->c.c_cdr!=Cnil;ss=ss->c.c_cdr);
	  ss->c.c_cdr=lex_env[0];
	  lex_env[0]=s[0];
	}

	return;

REQUIRED_ONLY:
	vs_push(Cnil);
	for (;  !endp(body);  body = body->c.c_cdr) {
		form = body->c.c_car;

		/*  MACRO EXPANSION  */
		vs_head = form = macro_expand(form);

		if (stringp(form)) {
			if (endp(body->c.c_cdr))
				break;
			continue;
		}
		if (!consp(form) || !isdeclare(form->c.c_car))
			break;
		for (ds = form->c.c_cdr; !endp(ds); ds = ds->c.c_cdr) {
			if (!consp(ds->c.c_car))
				illegal_declare(form);
			if (ds->c.c_car->c.c_car == sLspecial) {
				vs = ds->c.c_car->c.c_cdr;
				for (;  !endp(vs);  vs = vs->c.c_cdr) {
					v = vs->c.c_car;
					check_symbol(v);
/**/

	special_processed = FALSE;
	for (i = 0;  i < nreq;  i++)
		if (required[i].req_var == v) {
			required[i].req_spp = Ct;
			special_processed = TRUE;
		}
	if (special_processed)
		continue;
	/*  lex_special_bind(v);  */
	temporary = MMcons(v, Cnil);
	s[0] = MMcons(temporary, s[0]);

/**/
				}
			}
		}
	}

	narg = arg_top - base;
	if (narg != nreq) {
		vs_base = base;
		vs_top = arg_top;
		check_arg_failed(nreq);
	}
	for (i = 0;  i < nreq;  i++)
		bind_var(required[i].req_var,
			 base[i],
			 required[i].req_spp);
	if (!consp(body) || body->c.c_car == form) {
		vs_reset;
		vs_head = body;
	} else {
		body = make_cons(form, body->c.c_cdr);
		vs_reset;
		vs_head = body;
	}

	if (s[0]!=Cnil) {
	  for (ss=s[0];ss->c.c_cdr!=Cnil;ss=ss->c.c_cdr);
	  ss->c.c_cdr=lex_env[0];
	  lex_env[0]=s[0];
	}

}

void
bind_var(object var, object val, object spp)
{ 
        object temporary;
	vs_mark;

	switch (var->s.s_stype) {
	case stp_constant:
		FEerror("Cannot bind the constant ~S.", 1, var);

	case stp_special:
		bds_bind(var, val);
		break;

	default:
		if (spp != Cnil) {
			/*  lex_special_bind(var);  */
			temporary = MMcons(var, Cnil);
			lex_env[0] = MMcons(temporary, lex_env[0]);
			bds_bind(var, val);
		} else {
			/*  lex_local_bind(var, val);  */
			temporary = MMcons(val, Cnil);
			temporary = MMcons(var, temporary);
			lex_env[0] = MMcons(temporary, lex_env[0]);
		}
		break;
	}
	vs_reset;
}

static void
illegal_lambda(void)
{
	FEerror("Illegal lambda expression.", 0);
}

/*
struct bind_temp {
	object	bt_var;
	object	bt_spp;
	object	bt_init;
	object	bt_aux;
};
*/

object
find_special(object body, struct bind_temp *start, struct bind_temp *end,object *s)
{ 
        object temporary;
	object form=Cnil;
	object ds, vs, v;
	struct bind_temp *bt;
	bool special_processed;
	vs_mark;

	vs_push(Cnil);
	s=s ? s : lex_env;
	for (;  !endp(body);  body = body->c.c_cdr) {
		form = body->c.c_car;

		/*  MACRO EXPANSION  */
		form = macro_expand(form);
		vs_head = form;

		if (stringp(form)) {
			if (endp(body->c.c_cdr))
				break;
			continue;
		}
		if (!consp(form) || !isdeclare(form->c.c_car))
			break;
		for (ds = form->c.c_cdr; !endp(ds); ds = ds->c.c_cdr) {
			if (!consp(ds->c.c_car))
				illegal_declare(form);
			if (ds->c.c_car->c.c_car == sLspecial) {
				vs = ds->c.c_car->c.c_cdr;
				for (;  !endp(vs);  vs = vs->c.c_cdr) {
					v = vs->c.c_car;
					check_symbol(v);
/**/
	special_processed = FALSE;
	for (bt = start;  bt < end;  bt++)
		if (bt->bt_var == v) {
			bt->bt_spp = Ct;
			special_processed = TRUE;
		}
	if (special_processed)
		continue;
	/*  lex_special_bind(v);  */
	temporary = MMcons(v, Cnil);
	s[0] = MMcons(temporary, s[0]);
/**/
				}
			}
		}
	}

	if (body != Cnil && body->c.c_car != form && type_of(form)==t_cons && isdeclare(form->c.c_car))/*FIXME*/
		body = make_cons(form, body->c.c_cdr);
	vs_reset;
	return(body);
}

object
let_bind(object body, struct bind_temp *start, struct bind_temp *end)
{
	struct bind_temp *bt;

	bds_check;
	for (bt = start;  bt < end;  bt++) {
		eval_assign(bt->bt_init, bt->bt_init);
	}
	vs_push(find_special(body, start, end,NULL));
	for (bt = start;  bt < end;  bt++) {
		bind_var(bt->bt_var, bt->bt_init, bt->bt_spp);
	}
	return(vs_pop);
}

object
letA_bind(object body, struct bind_temp *start, struct bind_temp *end)
{
	struct bind_temp *bt;
	object s[1],ss;

	bds_check;
	s[0]=Cnil;
	vs_push(find_special(body, start, end,s));
	for (bt = start;  bt < end;  bt++) {
		eval_assign(bt->bt_init, bt->bt_init);
		bind_var(bt->bt_var, bt->bt_init, bt->bt_spp);
	}
	if (s[0]!=Cnil) {
	  for (ss=s[0];ss->c.c_cdr!=Cnil;ss=ss->c.c_cdr);
	  ss->c.c_cdr=lex_env[0];
	  lex_env[0]=s[0];
	}
	return(vs_pop);
}


#ifdef MV

#endif

#define	NOT_YET		stp_ordinary
#define	FOUND		stp_special
#define	NOT_KEYWORD	1

void
parse_key(object *base, bool rest, bool allow_other_keys, int n, ...)
{ 
        object temporary;
	va_list ap;
	object other_key = OBJNULL;
	int narg, error_flag = 0, allow_other_keys_found=0;
	object *v, k, *top;
	register int i;

	narg = vs_top - base;
	if (narg <= 0) {
		if (rest) {
			base[0] = Cnil;
			base++;
		}
		top = base + n;
		for (i = 0;  i < n;  i++) {
			base[i] = Cnil;
			top[i] = Cnil;
		}
		return;
	}
	if (narg%2 != 0)
	  /* FIXME better message */
	  FEunexpected_keyword(Cnil);
	if (narg == 2) {
		k = base[0];
		if (type_of(k)!=t_symbol)
		  FEunexpected_keyword(k);
		if (k == sKallow_other_keys && ! allow_other_keys_found) {
		  allow_other_keys_found=1;
		  if (base[1]!=Cnil)
		    allow_other_keys=TRUE;
		}
		temporary = base[1];
		if (rest)
			base++;
		top = base + n;
		other_key = k == sKallow_other_keys ? OBJNULL : k;
		va_start(ap,n);
		for (i = 0;  i < n;  i++) {
		    
			if (va_arg(ap,object) == k) {
				base[i] = temporary;
				top[i] = Ct;
				other_key = OBJNULL;
			} else {
				base[i] = Cnil;
				top[i] = Cnil;
			}
		}
		va_end(ap);
		if (rest) {
			temporary = make_cons(temporary, Cnil);
			base[-1] = make_cons(k, temporary);
		}
		if (other_key != OBJNULL && !allow_other_keys)
			FEunexpected_keyword(other_key);
		return;
	}
	va_start(ap,n);
	for (i = 0;  i < n;  i++) {
		k = va_arg(ap,object);
		k->s.s_stype = NOT_YET;
		k->s.s_dbind = Cnil;
	}
	va_end(ap);
	for (v = base;  v < vs_top;  v += 2) {
		k = v[0];
		if (type_of(k)!=t_symbol) {
			error_flag = NOT_KEYWORD;
			other_key = k;
			continue;
		}
		if (k->s.s_stype == NOT_YET) {
			k->s.s_dbind = v[1];
			k->s.s_stype = FOUND;
		} else if (k->s.s_stype == FOUND) {
			;
		} else if (other_key == OBJNULL && k!=sKallow_other_keys)
			other_key = k;
		if (k == sKallow_other_keys && !allow_other_keys_found) {
		  allow_other_keys_found=1;
		  if (v[1] != Cnil)
		    allow_other_keys = TRUE;
		}
	}
	if (rest) {
	  object *a,*l;
	  for (l=a=base;a<vs_top;a++)
	    collect(l,make_cons(*a,Cnil));
	  *l=Cnil;
	  base++;
	}
	top = base + n;
	va_start(ap,n);
	for (i = 0;  i < n;  i++) {
		k = va_arg(ap,object);
		base[i] = k->s.s_dbind;
		top[i] = k->s.s_stype == FOUND ? Ct : Cnil;
		k->s.s_dbind = k;
		k->s.s_stype = (short)stp_constant;
	}
	va_end(ap);
	if (error_flag == NOT_KEYWORD)
	  FEunexpected_keyword(other_key);
	if (other_key != OBJNULL && !allow_other_keys)
	  FEunexpected_keyword(other_key);
}

void
check_other_key(object l, int n, ...)
{
	va_list ap;
	object other_key = OBJNULL;
	object k;
	int i;
	bool allow_other_keys = FALSE;
	int allow_other_keys_found=0;

	for (;  !endp(l);  l = l->c.c_cdr->c.c_cdr) {
		k = l->c.c_car;
		if (type_of(k)!=t_symbol)
		  FEunexpected_keyword(k);
		if (endp(l->c.c_cdr))
		  /* FIXME better message */
		  FEunexpected_keyword(Cnil);
		if (k == sKallow_other_keys && !allow_other_keys_found) {
		  allow_other_keys_found=1;
		  if (l->c.c_cdr->c.c_car != Cnil)
		    allow_other_keys = TRUE;
		} else {
		  char buf [100];
		  bzero(buf,n);
		  va_start(ap,n);
		  for (i = 0;  i < n;  i++)
		    { if (va_arg(ap,object) == k &&
			  buf[i] ==0) {buf[i]=1; break;}}
		  va_end(ap);
		  if (i >= n) other_key = k;
		}
	}
	if (other_key != OBJNULL && !allow_other_keys)
	  FEunexpected_keyword(other_key);
}


/*  struct key {short n,allow_other_keys; */
/*  	    iobject *defaults; */
/*  	    iobject keys[1]; */
/*  	   }; */


object Cstd_key_defaults[15]={Cnil,Cnil,Cnil,Cnil,Cnil,Cnil,Cnil,
				Cnil,Cnil,Cnil,Cnil,Cnil,Cnil,Cnil,Cnil};

/* FIXME rewrite this */
/* static int */
/* parse_key_new(int n, object *base, struct key *keys, va_list ap) */
/* {object *new; */
/*  COERCE_VA_LIST(new,ap,n); */

/*  new = new + n ; */
/*   {int j=keys->n; */
/*    object *p= (object *)(keys->defaults); */
/*    while (--j >=0) base[j]=p[j]; */
/*  } */
/*  {if (n==0){ return 0;} */
/*  {int allow = keys->allow_other_keys; */
/*   object k; */

/*   if (!allow) { */
/*     int i; */
/*     for (i=n;i>0 && new[-i]!=sKallow_other_keys;i-=2); */
/*     if (i>0 && new[-i+1]!=Cnil) */
/*       allow=1; */
/*   } */

/*  top: */
/*   while (n>=2) */
/*     {int i= keys->n; */
/*      iobject *ke=keys->keys ; */
/*      new = new -2; */
/*      k = *new; */
/*      while(--i >= 0) */
/*        {if ((*(ke++)).o == k) */
/* 	  {base[i]= new[1]; */
/* 	   n=n-2; */
/* 	   goto top; */
/* 	 }} */
     /* the key is a new one */
/*      if (allow || k==sKallow_other_keys)  */
/*        n=n-2; */
/*      else */
/*        goto error; */
/*    } */
  /* FIXME better message */
/*   if (n!=0) FEunexpected_keyword(Cnil); */
/*   return 0; */
/*  error: */
/*   FEunexpected_keyword(k); */
/*   return -1; */
/* }}} */

int
parse_key_new_new(int n, object *base, struct key *keys, object first, va_list ap)
{object *new;
 COERCE_VA_LIST_KR_NEW(new,first,ap,n);

 /* from here down identical to parse_key_rest */
 new = new + n ;
  {int j=keys->n;
   object **p= (object **)(keys->defaults);
   while (--j >=0) base[j]=*(p[j]);
 }
 {if (n==0){ return 0;}
 {int allow = keys->allow_other_keys;
  object k;

  if (!allow) {
    int i;
    for (i=n;i>0 && new[-i]!=sKallow_other_keys;i-=2);
    if (i>0 && new[-i+1]!=Cnil)
      allow=1;
  }

 top:
  while (n>=2)
    {int i= keys->n;
     iobject *ke=keys->keys ;
     new = new -2;
     k = *new;
     while(--i >= 0)
       {if ((*(ke++)).o == k)
	  {base[i]= new[1];
	   n=n-2;
	   goto top;
	 }}
     /* the key is a new one */
     if (allow || k==sKallow_other_keys) 
       n=n-2;
     else
       goto error;
   }
  /* FIXME better message */
  if (n!=0) FEunexpected_keyword(Cnil);
  return 0;
 error:
  FEunexpected_keyword(k);
  return -1;
}}}

/* static int */
/* parse_key_rest(object rest, int n, object *base, struct key *keys, va_list ap) */
/* {object *new; */
/*  COERCE_VA_LIST(new,ap,n); */

 /* copy the rest arg */
/*  {object *p = new; */
/*   int m = n; */
/*   while (--m >= 0) */
/*     {rest->c.c_car = *p++; */
/*      rest = rest->c.c_cdr;}} */
    
/*  new = new + n ; */
/*   {int j=keys->n; */
/*    object *p= (object *)(keys->defaults); */
/*    while (--j >=0) base[j]=p[j]; */
/*  } */
/*  {if (n==0){ return 0;} */
/*  {int allow = keys->allow_other_keys; */
/*   object k; */

/*   if (!allow) { */
/*     int i; */
/*     for (i=n;i>0 && new[-i]!=sKallow_other_keys;i-=2); */
/*     if (i>0 && new[-i+1]!=Cnil) */
/*       allow=1; */
/*   } */

/*  top: */
/*   while (n>=2) */
/*     {int i= keys->n; */
/*      iobject *ke=keys->keys ; */
/*      new = new -2; */
/*      k = *new; */
/*      while(--i >= 0) */
/*        {if ((*(ke++)).o == k) */
/* 	  {base[i]= new[1]; */
/* 	   n=n-2; */
/* 	   goto top; */
/* 	 }} */
     /* the key is a new one */
/*      if (allow || k==sKallow_other_keys)  */
/*        n=n-2; */
/*      else */
/*        goto error; */
/*    } */
  /* FIXME better message */
/*   if (n!=0) FEunexpected_keyword(Cnil); */
/*   return 0; */
/*  error: */
/*   FEunexpected_keyword(k); */
/*   return -1; */
/* }}} */

int
parse_key_rest_new(object rest, int n, object *base, struct key *keys, object first,va_list ap)
{object *new;
 COERCE_VA_LIST_KR_NEW(new,first,ap,n);

 /* copy the rest arg */
 {object *p = new;
  int m = n;
  while (--m >= 0)
    {rest->c.c_car = *p++;
     rest = rest->c.c_cdr;}}
    
 new = new + n ;
  {int j=keys->n;
   object *p= (object *)(keys->defaults);
   while (--j >=0) base[j]=p[j];
 }
 {if (n==0){ return 0;}
 {int allow = keys->allow_other_keys;
  object k;

  if (!allow) {
    int i;
    for (i=n;i>0 && new[-i]!=sKallow_other_keys;i-=2);
    if (i>0 && new[-i+1]!=Cnil)
      allow=1;
  }

 top:
  while (n>=2)
    {int i= keys->n;
     iobject *ke=keys->keys ;
     new = new -2;
     k = *new;
     while(--i >= 0)
       {if ((*(ke++)).o == k)
	  {base[i]= new[1];
	   n=n-2;
	   goto top;
	 }}
     /* the key is a new one */
     if (allow || k==sKallow_other_keys) 
       n=n-2;
     else
       goto error;
   }
  /* FIXME better message */
  if (n!=0) FEunexpected_keyword(Cnil);
  return 0;
 error:
  FEunexpected_keyword(k);
  return -1;
}}}

void
set_key_struct(struct key *ks, object data)
{int i=ks->n;
 while (--i >=0)
   {ks->keys[i].o =   data->cfd.cfd_self[ ks->keys[i].i ];
    if (ks->defaults != (void *)Cstd_key_defaults)
      {fixnum m=ks->defaults[i].i;
        ks->defaults[i].o=
	  (m==-2 ? Cnil :
	   m==-1 ? OBJNULL :
	   data->cfd.cfd_self[m]);}
}}

#undef AUX

DEF_ORDINARY("ALLOW-OTHER-KEYS",sKallow_other_keys,KEYWORD,"");


void
gcl_init_bind(void)
{
	ANDoptional = make_ordinary("&OPTIONAL");
	enter_mark_origin(&ANDoptional);
	ANDrest = make_ordinary("&REST");
	enter_mark_origin(&ANDrest);
	ANDkey = make_ordinary("&KEY");
	enter_mark_origin(&ANDkey);
	ANDallow_other_keys = make_ordinary("&ALLOW-OTHER-KEYS");
	enter_mark_origin(&ANDallow_other_keys);
	ANDaux = make_ordinary("&AUX");
	enter_mark_origin(&ANDaux);

	make_constant("LAMBDA-LIST-KEYWORDS",
	make_cons(ANDoptional,
	make_cons(ANDrest,
	make_cons(ANDkey,
	make_cons(ANDallow_other_keys,
	make_cons(ANDaux,
	make_cons(make_ordinary("&WHOLE"),
	make_cons(make_ordinary("&ENVIRONMENT"),
	make_cons(make_ordinary("&BODY"), Cnil)))))))));

	make_constant("LAMBDA-PARAMETERS-LIMIT",
		      make_fixnum(MAX_ARGS+1));



	three_nils.nil3_self[0] = Cnil;
	three_nils.nil3_self[1] = Cnil;
	three_nils.nil3_self[2] = Cnil;

	six_nils.nil6_self[0] = Cnil;
	six_nils.nil6_self[1] = Cnil;
	six_nils.nil6_self[2] = Cnil;
	six_nils.nil6_self[3] = Cnil;
	six_nils.nil6_self[4] = Cnil;
	six_nils.nil6_self[5] = Cnil;
}
