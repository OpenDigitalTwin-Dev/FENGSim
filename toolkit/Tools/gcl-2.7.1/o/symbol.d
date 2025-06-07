/* -*-C-*- */
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
	symbol.d
*/

#include <ctype.h>
#include "include.h"

/*FIXME this symbol is needed my maxima MAKE_SPECIAL*/
void
check_type_symbol(object *x) {
  check_type_sym(x);
}

static void
odd_plist(object);


object siSpname;

object
make_symbol(st)
object st;
{
	object x;

	{BEGIN_NO_INTERRUPT;	
	x = alloc_object(t_symbol);
	x->s.s_dbind = OBJNULL;
	x->s.s_sfdef = NOT_SPECIAL;
	x->s.s_name=copy_simple_string(st);/*FIXME*/
	x->s.s_gfdef = OBJNULL;
	x->s.s_plist = Cnil;
	x->s.s_hpack = Cnil;
	x->s.s_stype = (short)stp_ordinary;
	x->s.s_mflag = FALSE;
	vs_push(x);
	x->s.s_hash = ihash_equal1(x,0);
	END_NO_INTERRUPT;}	
	return(vs_pop);
}

/*
	Make_ordinary(s) makes an ordinary symbol from C string s
	and interns it in lisp package as an external symbol.
*/

#define P_EXTERNAL(x,j) ((x)->p.p_external[(j) % (x)->p.p_external_size])


object
make_ordinary(s)
char *s;
{
	int j;
	object x, l, *ep;
	vs_mark;

	j = pack_hash(str(s));
	ep = &P_EXTERNAL(lisp_package,j);
	for (l = *ep;  consp(l);  l = l->c.c_cdr)
	  if (string_eq(l->c.c_car->s.s_name, str(s)))
	    return(l->c.c_car);
	x = make_symbol(str(s));
	vs_push(x);
	x->s.s_hpack = lisp_package;
	*ep = make_cons(x, *ep);
	lisp_package->p.p_external_fp ++;
	vs_reset;
	return(x);
}

/*
	Make_special(s, v) makes a special variable from C string s
	with initial value v in lisp package.
*/
object
make_special(s, v)
char *s;
object v;
{
	object x;

	x = make_ordinary(s);
	x->s.s_stype = (short)stp_special;
	x->s.s_dbind = v;
	return(x);
}

/*
	Make_constant(s, v) makes a constant from C string s
	with constant value v in lisp package.
*/
object
make_constant(s, v)
char *s;
object v;
{
	object x;

	x = make_ordinary(s);
	x->s.s_stype = (short)stp_constant;
	x->s.s_dbind = v;
	return(x);
}

/*
	Make_si_ordinary(s) makes an ordinary symbol from C string s
	and interns it in system package as an external symbol.
	It assumes that the (only) package used by system is lisp.
*/



object
make_si_ordinary(s)
char *s;
{
	int j;
	object x, l, *ep;
	vs_mark;

	j = pack_hash(str(s));
	ep = & P_EXTERNAL(system_package,j);
	for (l = *ep;  consp(l);  l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, str(s)))
			return(l->c.c_car);
	for (l =  P_EXTERNAL(lisp_package,j);
	     consp(l);
	     l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, str(s)))
		    error("name conflict --- can't make_si_ordinary");
	x = make_symbol(str(s));
	vs_push(x);
	x->s.s_hpack = system_package;
	system_package->p.p_external_fp ++;
	*ep = make_cons(x, *ep);
	vs_reset;
	return(x);
}

object
make_gmp_ordinary(s)
char *s;
{
        int i,j;
	object x, l, *ep;
	vs_mark;
	char *ss=alloca(strlen(s)+1);

	for (i=0;s[i];i++)
	  ss[i]=toupper(s[i]);
	ss[i]=0;
	j = pack_hash(str(ss));
	ep = & P_EXTERNAL(gmp_package,j);
	for (l = *ep;  consp(l);  l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, str(ss)))
			return(l->c.c_car);
	for (l =  P_EXTERNAL(lisp_package,j);
	     consp(l);
	     l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, str(ss)))
		    error("name conflict --- can't make_si_ordinary");
	x = make_symbol(str(ss));
	vs_push(x);
	x->s.s_hpack = gmp_package;
	gmp_package->p.p_external_fp ++;
	*ep = make_cons(x, *ep);
	vs_reset;
	return(x);
}

/*
	Make_si_special(s, v) makes a special variable from C string s
	with initial value v in system package.
*/
object
make_si_special(s, v)
char *s;
object v;
{
	object x;

	x = make_si_ordinary(s);
	x->s.s_stype = (short)stp_special;
	x->s.s_dbind = v;
	return(x);
}

/*
	Make_si_constant(s, v) makes a constant from C string s
	with constant value v in system package.
*/
object
make_si_constant(s, v)
char *s;
object v;
{
	object x;

	x = make_si_ordinary(s);
	x->s.s_stype = (short)stp_constant;
	x->s.s_dbind = v;
	return(x);
}

/*
	Make_keyword(s) makes a keyword from C string s.
*/
object
make_keyword(s)
char *s;
{
	int j;
	object x, l, *ep;
	vs_mark;

	j = pack_hash(str(s));
	ep = &P_EXTERNAL(keyword_package,j);
	for (l = *ep;  consp(l);  l = l->c.c_cdr)
		if (string_eq(l->c.c_car->s.s_name, str(s)))
			return(l->c.c_car);
	x = make_symbol(str(s));
	vs_push(x);
	x->s.s_hpack = keyword_package;
	x->s.tt=2;
	x->s.s_stype = (short)stp_constant;
	x->s.s_dbind = x;
	*ep = make_cons(x, *ep);
	keyword_package->p.p_external_fp ++;
	vs_reset;
	return(x);
}

object
symbol_value(s)
object s;
{
/*
	if (type_of(s) != t_symbol)
		FEinvalid_variable("~S is not a symbol.", s);
*/
  if (s->s.s_dbind == OBJNULL)
    FEunbound_variable(s);
  return(s->s.s_dbind);
}

object
getf(place, indicator, deflt)
object place, indicator, deflt;
{

	object l;
#define cendp(obj) ((!consp(obj)))
	for (l = place;  !cendp(l);  l = l->c.c_cdr->c.c_cdr) {
		if (cendp(l->c.c_cdr))
			break;
		if (l->c.c_car == indicator)
			return(l->c.c_cdr->c.c_car);
	}
	if(l==Cnil) return deflt;
	FEinvalid_form("Bad plist ~a",place);	
	return Cnil;
}

object
get(s, p, d)
object s, p, d;
{
	if (type_of(s) != t_symbol)
		not_a_symbol(s);
	return(getf(s->s.s_plist, p, d));
}

/*
	Putf(p, v, i) puts value v for property i to property list p
	and returns the resulting property list.
*/
object
putf(p, v, i)
object p, v, i;
{
	object l;

	for (l = p;  !cendp(l);  l = l->c.c_cdr->c.c_cdr) {
		if (cendp(l->c.c_cdr))
			break;
		if (l->c.c_car == i) {
			l->c.c_cdr->c.c_car = v;
			return(p);
		}
	}
        if(l!=Cnil) FEerror("Bad plist ~a",1,p);
	return listA(3,i,v,p);
}

object
putprop(s, v, p)
object s, v, p;
{
	if (type_of(s) != t_symbol)
		not_a_symbol(s);
	s->s.s_plist = putf(s->s.s_plist, v, p);
	return(v);
}

DEFUN("SPUTPROP",object,fSsputprop,SI,3,3,NONE,OO,OO,OO,OO,(object s,object p,object v),"") {

  if (type_of(s) != t_symbol)
    not_a_symbol(s);
  s->s.s_plist = putf(s->s.s_plist, v, p);
  return(v);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSsputprop(object x,object y,object z) {
  return FFN(fSsputprop)(x,y,z);
}
#endif

/*
	Remf(p, i) removes property i
	from the property list pointed by p,
	which is a pointer to an object.
	The returned value of remf(p, i) is:

		TRUE	if the property existed
		FALSE	otherwise.
*/
bool
remf(p, i)
object *p, i;
{
	object l0 = *p;

	for(;  !endp(*p);  p = &(*p)->c.c_cdr->c.c_cdr) {
		if (endp((*p)->c.c_cdr))
			odd_plist(l0);
		if ((*p)->c.c_car == i) {
			*p = (*p)->c.c_cdr->c.c_cdr;
			return(TRUE);
		}
	}
	return(FALSE);
}

object
remprop(s, p)
object s, p;
{
	if (type_of(s) != t_symbol)
		not_a_symbol(s);
	if (remf(&s->s.s_plist, p))
		return(Ct);
	else
		return(Cnil);
}

bool
keywordp(s)
object s;
{
	return(type_of(s) == t_symbol && s->s.s_hpack == keyword_package);
/*
	if (type_of(s) != t_symbol) {
		vs_push(s);
		check_type_sym(&vs_head);
		vs_pop;
	}
	if (s->s.s_hpack == OBJNULL)
		return(FALSE);
	return(s->s.s_hpack == keyword_package);
*/
}

@(defun get (sym indicator &optional deflt)
@
	check_type_sym(&sym);
	@(return `getf(sym->s.s_plist, indicator, deflt)`)
@)

LFD(Lremprop)()
{
	check_arg(2);

	check_type_sym(&vs_base[0]);
	if (remf(&vs_base[0]->s.s_plist, vs_base[1]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
	vs_popp;
}

DEFUN("SYMBOL-PLIST",object,fLsymbol_plist,LISP,1,1,NONE,OO,OO,OO,OO,(object sym),"") {
  check_type_sym(&sym);
  RETURN1(sym->s.s_plist);
}

@(defun getf (place indicator &optional deflt)
@
	@(return `getf(place, indicator, deflt)`)
@)

@(defun get_properties (place indicator_list)
	object l, m;

@
	for (l = place;  !endp(l);  l = l->c.c_cdr->c.c_cdr) {
		if (endp(l->c.c_cdr))
			odd_plist(place);
		for (m = indicator_list;  !endp(m);  m = m->c.c_cdr)
			if (l->c.c_car == m->c.c_car)
				@(return `l->c.c_car`
					 `l->c.c_cdr->c.c_car`
					 l)
	}
	@(return Cnil Cnil Cnil)
@)

DEFUN("SYMBOL-STRING",object,fSsymbol_string,SI,1,1,NONE,OO,OO,OO,OO,(object sym),"") {
  RETURN1(sym->s.s_name);
}


object
symbol_name(x)
object x;
{
 if (type_of(x)!=t_symbol) FEwrong_type_argument(sLsymbol,x);
 return(x->s.s_name);
}

DEFUN("SYMBOL-NAME",object,fLsymbol_name,LISP,1,1,NONE,OO,OO,OO,OO,(object sym),"") {
/* LFD(Lsymbol_name)() */
  RETURN1(symbol_name(sym));
}

DEFUN("MAKE-SYMBOL",object,fLmake_symbol,LISP,1,1,NONE,OO,OO,OO,OO,(object name),"") {
/* LFD(Lmake_symbol)() */
  check_type_string(&name);
  RETURN1(make_symbol(name));
}

@(defun copy_symbol (sym &optional cp &aux x)
@
	check_type_sym(&sym);
	x = make_symbol(sym->s.s_name);
	if (cp == Cnil)
		@(return x)
	x->s.s_stype = sym->s.s_stype;
	x->s.s_dbind = sym->s.s_dbind;
	x->s.s_mflag = sym->s.s_mflag;
	x->s.s_gfdef = sym->s.s_gfdef;
	x->s.s_plist = copy_list(sym->s.s_plist);
	@(return x)
@)

DEFVAR("*GENSYM-COUNTER*",sLgensym_counter,LISP,make_fixnum(0),"");

static object
gensym_int(object this_gensym_prefix,object this_gensym_counter) {

  int i, j, sign, size;
  fixnum f;
  char *q=NULL,*p=NULL;
  object big;
  object sym;

  switch (type_of(this_gensym_counter)) {
  case t_bignum:
    big=this_gensym_counter;
    sign=BIG_SIGN(big);
    size = mpz_sizeinbase(MP(big),10)+(BIG_SIGN(big)<0? 1 : 0)+1;
    massert(p=alloca(size));
    massert(p=mpz_get_str(p,10,MP(big)));
    q=p+strlen(p);
    break;
  case t_fixnum:
    for (size=1,f=fix(this_gensym_counter);f;f/=10,size++);
    q=p=ZALLOCA(size+5);
    if ((j=snprintf(p,size+5,"%d",(int)fix(this_gensym_counter)))<=0)
      FEerror("Cannot write gensym counter",0);
    q=p+j;
    break;
  default:
    TYPE_ERROR(this_gensym_counter,sLinteger);
    break;
  }
  
  i = (q-p)+(this_gensym_prefix==OBJNULL ? 1 : VLEN(this_gensym_prefix));
  sym = make_symbol(str(""));
  {
    BEGIN_NO_INTERRUPT;
    sym->s.s_name=alloc_simple_string(i);
    sym->s.s_name->st.st_self = alloc_relblock(i);
    /* sym->s.s_fillp = i; */
    i=this_gensym_prefix==OBJNULL ? 1 : VLEN(this_gensym_prefix);
    for (j = 0;  j < i;  j++)
      sym->s.s_name->st.st_self[j] = this_gensym_prefix==OBJNULL ? 'G' : this_gensym_prefix->st.st_self[j];
    for (;j<VLEN(sym->s.s_name);j++)
      sym->s.s_name->st.st_self[j] = p[j-i];
    END_NO_INTERRUPT;
  }
  
  RETURN1(sym);

}

DEFUN("GENSYM0",object,fSgensym0,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {

  object x;

  x=sLgensym_counter->s.s_dbind;
  sLgensym_counter->s.s_dbind=number_plus(sLgensym_counter->s.s_dbind,small_fixnum(1));
  RETURN1(gensym_int(OBJNULL,x));

}

DEFUN("GENSYM1S",object,fSgensym1s,SI,1,1,NONE,OO,OO,OO,OO,(object g),"") {

  object x;

  x=sLgensym_counter->s.s_dbind;
  sLgensym_counter->s.s_dbind=number_plus(sLgensym_counter->s.s_dbind,small_fixnum(1));
  RETURN1(gensym_int(g,x));

}

DEFUN("GENSYM1IG",object,fSgensym1ig,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  check_type_non_negative_integer(&x);
  RETURN1(gensym_int(OBJNULL,x));

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSgensym0(void) {
  return FFN(fSgensym0)();
}
object
fSgensym1s(object x) {
  return FFN(fSgensym1s)(x);
}
object
fSgensym1ig(object x) {
  return FFN(fSgensym1ig)(x);
}
#endif

@(defun gentemp (&optional (prefix gentemp_prefix)
			   (pack `current_package()`)
		 &aux smbl)
	int i, j;
@
	check_type_string(&prefix);
        if (type_of(pack)!=t_package) 
           {object tem; 
            tem=find_package(pack);
            if (tem==Cnil) FEerror("No package named ~a exists",1,pack); 
            pack=tem;}
/* 	check_type_package(&pack); */
/*
	gentemp_counter = 0;
*/
ONCE_MORE:
	for (j = gentemp_counter, i = 0;  j > 0;  j /= 10)
		i++;
	if (i == 0)
		i++;
        i += VLEN(prefix);
	{BEGIN_NO_INTERRUPT;	
	string_register->st.st_dim = i;
	string_register->st.st_self = alloc_relblock(i);
	for (j = 0;  j < VLEN(prefix);  j++)
		string_register->st.st_self[j] = prefix->st.st_self[j];
	if ((j = gentemp_counter) == 0)
		string_register->st.st_self[--i] = '0';
	else
		for (;  j > 0;  j /= 10)
			string_register->st.st_self[--i] = j%10 + '0';
	gentemp_counter++;
	smbl = intern(string_register, pack);
	if (intern_flag != 0)
		goto ONCE_MORE;
	END_NO_INTERRUPT;}	
	@(return smbl)
@)

DEFUN("SYMBOL-PACKAGE",object,fLsymbol_package,LISP,1,1,NONE,OO,OO,OO,OO,(object sym),"") {
  check_type_sym(&sym);
  RETURN1(sym->s.s_hpack);
}

DEFUN("KEYWORDP",object,fLkeywordp,LISP,1,1,NONE,OO,OO,OO,OO,(object sym),"") {
  RETURN1(type_of(sym) == t_symbol && keywordp(sym) ? Ct : Cnil);
}

/*
	(SI:PUT-F plist value indicator)
	returns the new property list with value for property indicator.
	It will be used in SETF for GETF.
*/
LFD(siLput_f)()
{
	check_arg(3);

	vs_base[0] = putf(vs_base[0], vs_base[1], vs_base[2]);
	vs_top = vs_base+1;
}

/*
	(SI:REM-F plist indicator) returns two values:

		* the new property list
		  in which property indcator is removed

		* T	if really removed
		  NIL	otherwise.

	It will be used for macro REMF.
*/
LFD(siLrem_f)()
{
	check_arg(2);

	if (remf(&vs_base[0], vs_base[1]))
		vs_base[1] = Ct;
	else
		vs_base[1] = Cnil;
}

LFD(siLset_symbol_plist)(void)
{
	check_arg(2);

	check_type_sym(&vs_base[0]);
	vs_base[0]->s.s_plist = vs_base[1];
	vs_base[0] = vs_base[1];
	vs_popp;
}

LFD(siLputprop)()
{
	check_arg(3);

	check_type_sym(&vs_base[0]);
	vs_base[0]->s.s_plist
	= putf(vs_base[0]->s.s_plist, vs_base[1], vs_base[2]);
	vs_base[0] = vs_base[1];
	vs_top = vs_base+1;
}


static void
odd_plist(place)
object place;
{
	FEerror("The length of the property-list ~S is odd.", 1, place);
}


void
gcl_init_symbol()
{
	string_register = alloc_simple_string(0);
/* 	gensym_prefix = make_simple_string("G"); */
/* 	gensym_counter = 0; */
	gentemp_prefix = make_simple_string("T");
	gentemp_counter = 0;
	token = alloc_string(INITIAL_TOKEN_LENGTH);
	token->st.st_fillp = 0;
	token->st.st_self = alloc_contblock(INITIAL_TOKEN_LENGTH);
	token->st.st_hasfillp = TRUE;
	token->st.st_adjustable = TRUE;

	enter_mark_origin(&string_register);
/* 	enter_mark_origin(&gensym_prefix); */
	enter_mark_origin(&gentemp_prefix);
	enter_mark_origin(&token);
}

void
gcl_init_symbol_function()
{
	make_function("GET", Lget);
	make_function("REMPROP", Lremprop);
/* 	make_function("SYMBOL-PLIST", Lsymbol_plist); */
	make_function("GETF", Lgetf);
	make_function("GET-PROPERTIES", Lget_properties);
/* 	make_function("SYMBOL-NAME", Lsymbol_name); */
/* 	make_function("MAKE-SYMBOL", Lmake_symbol); */
	make_function("COPY-SYMBOL", Lcopy_symbol);
/* 	make_function("GENSYM", Lgensym); */
	make_function("GENTEMP", Lgentemp);
/* 	make_function("SYMBOL-PACKAGE", Lsymbol_package); */
/* 	make_function("KEYWORDP", Lkeywordp); */

	make_si_function("PUT-F", siLput_f);
	make_si_function("REM-F", siLrem_f);
	make_si_function("SET-SYMBOL-PLIST", siLset_symbol_plist);

	make_si_function("PUTPROP", siLputprop);
/* 	make_si_sfun("SPUTPROP",sputprop,3); */


	siSpname = make_si_ordinary("PNAME");
	enter_mark_origin(&siSpname);
/* 	enter_mark_origin(&sLgensym_counter); */
}
