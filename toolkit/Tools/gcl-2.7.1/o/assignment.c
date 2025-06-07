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

	assignment.c

	Assignment
*/

#include "include.h"

static object
setf(object,object);

object sLsetf;

object sLget;
object sLgetf;
object sLaref;
object sLsvref;
object sLelt;
object sLchar;
object sLschar;
object sLfill_pointer;
object sLgethash;
object sLcar;
object sLcdr;

object sLpush;
object sLpop;
object sLincf;
object sLdecf;

object sSstructure_access;
object sSsetf_lambda;



object sSclear_compiler_properties;

object sLwarn;

object sSAinhibit_macro_specialA;

void
setq(object sym, object val)
{
	object vd;
	enum stype type;

	if(type_of(sym) != t_symbol)
		not_a_symbol(sym);
	type = (enum stype)sym->s.s_stype;
	if(type == stp_special)
		sym->s.s_dbind = val;
	else
	if (type == stp_constant)
		FEinvalid_variable("Cannot assign to the constant ~S.", sym);
	else {
		vd = lex_var_sch(sym);
		if(MMnull(vd) || endp(MMcdr(vd)))
			sym->s.s_dbind = val;
		else
			MMcadr(vd) = val;
	}
}

static void
FFN(Fsetq)(object form)
{
	object ans;
	if (endp(form)) {
		vs_base = vs_top;
		vs_push(Cnil);
	} else {
		object *top = vs_top;
		do {
			vs_top = top;
			if (endp(MMcdr(form)))
			FEinvalid_form("No value for ~S.", form->c.c_car);
			setq(MMcar(form),ans=Ieval1(MMcadr(form)));
			form = MMcddr(form);
		} while (!endp(form));
		top[0]=ans;
		vs_base=top;
		vs_top= top+1;
	}
}

static void
FFN(Fpsetq)(object arg)
{
	object *old_top = vs_top;
	object *top;
	object argsv = arg;
	for (top = old_top;  !endp(arg);  arg = MMcddr(arg), top++) {
		if(endp(MMcdr(arg)))
			FEinvalid_form("No value for ~S.", arg->c.c_car);
		
		top[0] = Ieval1(MMcadr(arg));
		vs_top = top + 1;
	}
	for (arg = argsv, top = old_top; !endp(arg); arg = MMcddr(arg), top++)
		setq(MMcar(arg),top[0]);
	vs_base = vs_top = old_top;
	vs_push(Cnil);
}

DEFUN("SET",object,fLset,LISP,2,2,NONE,OO,OO,OO,OO,(object symbol,object value),"") {

  /* 2 args */
  if (type_of(symbol) != t_symbol)
    not_a_symbol(symbol);
  if ((enum stype)symbol->s.s_stype == stp_constant)
    FEinvalid_variable("Cannot assign to the constant ~S.",
		       symbol);
  symbol->s.s_dbind = value;
  RETURN1(value);

}

DEFUN("FUNCTION-NAME",object,fSfunction_name,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  switch(type_of(x)) {
  case t_function: 
    x=Cnil;
    break;
  default:
    TYPE_ERROR(x,sLfunction);
    x=Cnil;
    break;
  }

  return x;

}

	


DEFUN("FSET",object,fSfset,SI,2,2,NONE,OO,OO,OO,OO,(object sym,object function),"") {

  object x;

  if (type_of(sym)!=t_symbol)
    sym=ifuncall1(sSfunid_to_sym,sym);
  
  if (sym->s.s_sfdef != NOT_SPECIAL) {
    if (sym->s.s_mflag) {
      if (symbol_value(sSAinhibit_macro_specialA) != Cnil)
	sym->s.s_sfdef = NOT_SPECIAL;
    } else if (symbol_value(sSAinhibit_macro_specialA) != Cnil)
      FEerror("~S, a special form, cannot be redefined.",
	      1, sym);
  }
  sym = clear_compiler_properties(sym,function);
  if (type_of(function) == t_function) {
    sym->s.s_gfdef = function;
    sym->s.s_mflag = FALSE;
  } else if (car(function) == sLspecial)
    FEerror("Cannot define a special form.", 0);
  else if (function->c.c_car == sSmacro) {
    function=function->c.c_cdr;
    sym->s.s_gfdef = function;
    sym->s.s_mflag = TRUE;
  } else {
    sym->s.s_gfdef = function;
    sym->s.s_mflag = FALSE;
  }
  
  sym->s.s_sfdef=NOT_SPECIAL;/*FIXME?*/
  if (function->fun.fun_plist!=Cnil) {
    function->fun.fun_plist->c.c_cdr->c.c_cdr->c.c_cdr->c.c_cdr->c.c_cdr->c.c_car=sym;/*FIXME*/
    x=function->fun.fun_plist->c.c_cdr->c.c_cdr->c.c_cdr->c.c_car;
    function->fun.fun_plist->c.c_cdr->c.c_cdr->c.c_cdr->c.c_car=x==Cnil ? sLAload_truenameA->s.s_dbind : x;
  }
  RETURN1(function);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSfset(object sym,object function) {
  return FFN(fSfset)(sym,function);
}
#endif

static void
FFN(Fmultiple_value_setq)(object form) {

  object vars,*vals;
  int n, i;
  
  if (endp(form) || endp(form->c.c_cdr) ||
      !endp(form->c.c_cdr->c.c_cdr))
    FEinvalid_form("~S is an illegal argument to MULTIPLE-VALUE-SETQ",form);

  vars = form->c.c_car;
  vals=ZALLOCA(MULTIPLE_VALUES_LIMIT*sizeof(*vals));

  vals[0]=Ievaln(form->c.c_cdr->c.c_car,vals+1);
  for (i=0,n=vs_top-vals;!endp(vars);i++,vars=vars->c.c_cdr)
    setq(vars->c.c_car,i<n ? vals[i] : Cnil);

  vs_base[0]=vals[0];
  vs_top=vs_base+1;

}

DEFUN("MAKUNBOUND",object,fLmakunbound,LISP,1,1,NONE,OO,OO,OO,OO,(object sym),"") {

  if (type_of(sym) != t_symbol)
    not_a_symbol(sym);
  if ((enum stype)sym->s.s_stype == stp_constant)
    FEinvalid_variable("Cannot unbind the constant ~S.",
		       sym);
  sym->s.s_dbind = OBJNULL;
  RETURN1(sym);

}

object sStraced;

DEFUN("FMAKUNBOUND",object,fLfmakunbound,LISP,1,1,NONE,OO,OO,OO,OO,(object sym),"") {

  object rsym;

  rsym=type_of(sym)==t_symbol ? sym : ifuncall1(sSfunid_to_sym,sym);

  if (rsym->s.s_sfdef != NOT_SPECIAL) {
    if (rsym->s.s_mflag) {
      if (symbol_value(sSAinhibit_macro_specialA) != Cnil)
	rsym->s.s_sfdef = NOT_SPECIAL;
    } else if (symbol_value(sSAinhibit_macro_specialA) != Cnil)
      FEerror("~S, a special form, cannot be redefined.", 1, rsym);
  }
  remf(&(rsym->s.s_plist),sStraced);
  clear_compiler_properties(rsym,Cnil);

  rsym->s.s_gfdef = OBJNULL;
  rsym->s.s_mflag = FALSE;
  RETURN1(sym);

}

static void
FFN(Fsetf)(object form) {

  object result=Cnil,*top=vs_top;

  for (;!endp(form);form=MMcddr(form)) {
    vs_top = top;
    if (endp(MMcdr(form)))
      FEinvalid_form("No value for ~S.", form->c.c_car);
    result=setf(MMcar(form), MMcadr(form));
  }
  vs_base=top;
  vs_base[0]=result;
  vs_top=vs_base+1;

}
    
/*   if (endp(form)) { */
/*     vs_base = vs_top; */
/*     vs_push(Cnil); */
/*   } else { */
/*     object *top = vs_top; */
/*     do { */
/*       vs_top = top; */
/*       if (endp(MMcdr(form))) */
/* 	FEinvalid_form("No value for ~S.", form->c.c_car); */
/*       result = setf(MMcar(form), MMcadr(form)); */
/*       form = MMcddr(form); */
/*     } while (!endp(form)); */
/*     vs_top = vs_base = top; */
/*     vs_base[0]=result; */
/*     vs_top=vs_base+1; */
    
/*   } */
/* } */

#define	eval_push(form)  \
{  \
	object *old_top = vs_top;  \
  \
	*old_top = Ieval1(form);  \
	vs_top = old_top + 1;  \
}

static object
setf(object place, object form)
{
	object fun;
	object args;
	object x,result,y;
	int i;

	if (!consp(place)) {
	  setq(place, result=Ieval1(form));
	  return result;
	}
	
	fun = place->c.c_car;
	if (type_of(fun) != t_symbol)
		goto OTHERWISE;
	args = place->c.c_cdr;

	{
	  object p=lisp_package;
	  char *s;

	  if (fun->s.s_hpack==p && fun->s.s_name->st.st_self[0]=='C' && fun->s.s_name->st.st_self[VLEN(fun->s.s_name)-1]=='R' && VLEN(fun->s.s_name)!=3) {

	    s=alloca(VLEN(fun->s.s_name));
	    s[0]='C';
	    memcpy(s+1,fun->s.s_name->st.st_self+2,VLEN(fun->s.s_name)-2);
	    s[VLEN(fun->s.s_name)-1]=0;
	    
	    fun=fun->s.s_name->st.st_self[1]=='A' ? sLcar : sLcdr;
	    args=MMcons(MMcons(find_symbol(make_simple_string(s),p),MMcons(args->c.c_car,Cnil)),Cnil);

	  }
	} /*FIXME*/
	  
	if (fun == sLget) {
            object sym,val,key,deflt1;
	  sym = Ieval1(car(args));
	  key = Ieval1(car(Mcdr(args)));
          deflt1 = Mcddr(args);
          if (consp(deflt1))
	    Ieval1(car(deflt1));
	  val = Ieval1(form);
	  return putprop(sym,val,key); 
	}

	if (fun == find_symbol(str("SYMBOL-FUNCTION"),lisp_package))
	  return Ieval1(MMcons(find_symbol(str("FSET"),system_package),MMcons(MMcar(args),MMcons(form,Cnil))));
	if (fun == sLsbit)
	  return Ieval1(MMcons(find_symbol(str("ASET"),system_package),MMcons(form,args)));
	if (fun == sLaref) 
	  return Ieval1(MMcons(find_symbol(str("ASET"),system_package),MMcons(form,args)));
	if (fun == sLsvref)
	  return Ieval1(MMcons(find_symbol(str("SVSET"),system_package),append(args,MMcons(form,Cnil))));
	if (fun == sLelt)
	  return Ieval1(MMcons(find_symbol(str("ELT-SET"),system_package),append(args,MMcons(form,Cnil))));
	if (fun == sLchar)
	  return Ieval1(MMcons(find_symbol(str("CHAR-SET"),system_package),append(args,MMcons(form,Cnil))));
	if (fun == sLschar)
	  return Ieval1(MMcons(find_symbol(str("SCHAR-SET"),system_package),append(args,MMcons(form,Cnil))));
	if (fun == sLfill_pointer) 
	  return Ieval1(MMcons(find_symbol(str("FILL-POINTER-SET"),system_package),append(args,MMcons(form,Cnil))));
	if (fun == sLgethash) 
	  return Ieval1(MMcons(find_symbol(str("HASH-SET"),system_package),append(args,MMcons(form,Cnil))));
	if (fun == sLcar) {
		x = Ieval1(Mcar(args));
		result = Ieval1(form);
		if (!consp(x))
			FEerror("~S is not a cons.", 1, x);
		Mcar(x) = result;
		return result;
	}
	if (fun == sLcdr) {
		x = Ieval1(Mcar(args));
		result = Ieval1(form);
		if (!consp(x))
			FEerror("~S is not a cons.", 1, x);
		Mcdr(x) = result;
		return result;
	}

	x = getf(fun->s.s_plist, sSstructure_access, Cnil);
	if (x == Cnil || !consp(x))
		goto OTHERWISE;
	if (getf(fun->s.s_plist, sSsetf_lambda, Cnil) == Cnil)
		goto OTHERWISE;
	if (type_of(x->c.c_cdr) != t_fixnum)
		goto OTHERWISE;
	i = fix(x->c.c_cdr);
	x = x->c.c_car;
	y = Ieval1(Mcar(args));
	result = Ieval1(form);
	if (x == sLvector) {
	  if (!TS_MEMBER(type_of(y),TS(t_vector)|TS(t_simple_vector)) || i >= VLEN(y))/*FIXME*/
	    goto OTHERWISE;
	  y->v.v_self[i] = result;
	} else if (x == sLlist) {
		for (x = y;  i > 0;  --i)
			x = cdr(x);
		if (!consp(x))
			goto OTHERWISE;
		x->c.c_car = result;
	} else {
		structure_set(y, x, i, result);
	}
	return result;


OTHERWISE:
	vs_base = vs_top;
	vs_push(list(3,sLsetf,place,result=form));
/***/
#define VS_PUSH_ENV \
	if(lex_env[1]){ \
	  vs_push(list(3,lex_env[0],lex_env[1],lex_env[2]));} \
	else {vs_push(Cnil);}
        VS_PUSH_ENV ;
/***/
	if (!sLsetf->s.s_mflag || sLsetf->s.s_gfdef == OBJNULL)
		FEerror("Where is SETF?", 0);
	funcall(sLsetf->s.s_gfdef);
	return Ieval1(vs_base[0]);
}

static void
FFN(Fpush)(object form)
{
	object var;
	
	if (endp(form) || endp(MMcdr(form)))
		FEtoo_few_argumentsF(form);
	if (!endp(MMcddr(form)))
		FEtoo_many_argumentsF(form);
	var = MMcadr(form);
	if (!consp(var)) {
		eval(MMcar(form));
		form = vs_base[0];
		eval(var);
		vs_base[0] = MMcons(form, vs_base[0]);
		setq(var, vs_base[0]);
		return;
	}
	vs_base = vs_top;
	vs_push(make_cons(sLpush,form));
/***/
         VS_PUSH_ENV ;
/***/
	if (!sLpush->s.s_mflag || sLpush->s.s_gfdef == OBJNULL)
		FEerror("Where is PUSH?", 0);
	funcall(sLpush->s.s_gfdef);
	eval(vs_base[0]);
}

static void
FFN(Fpop)(object form)
{
	object var;

	if (endp(form))
		FEtoo_few_argumentsF(form);
	if (!endp(MMcdr(form)))
		FEtoo_many_argumentsF(form);
	var = MMcar(form);
	if (!consp(var)) {
		eval(var);
		setq(var, cdr(vs_base[0]));
		vs_base[0] = car(vs_base[0]);
		return;
	}
	vs_base = vs_top;
	vs_push(make_cons(sLpop,form));
/***/
	VS_PUSH_ENV ;
/***/
	if (!sLpop->s.s_mflag || sLpop->s.s_gfdef == OBJNULL)
		FEerror("Where is POP?", 0);
	funcall(sLpop->s.s_gfdef);
	eval(vs_base[0]);
}

static void
FFN(Fincf)(object form)
{
	object var;
	object one_plus(object x), number_plus(object x, object y);

	if (endp(form))
		FEtoo_few_argumentsF(form);
	if (!endp(MMcdr(form)) && !endp(MMcddr(form)))
		FEtoo_many_argumentsF(form);
	var = MMcar(form);
	if (!consp(var)) {
		if (endp(MMcdr(form))) {
			eval(var);
			vs_base[0] = one_plus(vs_base[0]);
			setq(var, vs_base[0]);
			return;
		}
		eval(MMcadr(form));
		form = vs_base[0];
		eval(var);
		vs_base[0] = number_plus(vs_base[0], form);
		setq(var, vs_base[0]);
		return;
	}
	vs_base = vs_top;
	vs_push(make_cons(sLincf,form));
/***/
	VS_PUSH_ENV ;
/***/
	if (!sLincf->s.s_mflag || sLincf->s.s_gfdef == OBJNULL)
		FEerror("Where is INCF?", 0);
	funcall(sLincf->s.s_gfdef);
	eval(vs_base[0]);
}

static void
FFN(Fdecf)(object form)
{
	object var;
	object one_minus(object x), number_minus(object x, object y);

	if (endp(form))
		FEtoo_few_argumentsF(form);
	if (!endp(MMcdr(form)) && !endp(MMcddr(form)))
		FEtoo_many_argumentsF(form);
	var = MMcar(form);
	if (!consp(var)) {
		if (endp(MMcdr(form))) {
			eval(var);
			vs_base[0] = one_minus(vs_base[0]);
			setq(var, vs_base[0]);
			return;
		}
		eval(MMcadr(form));
		form = vs_base[0];
		eval(var);
		vs_base[0] = number_minus(vs_base[0], form);
		setq(var, vs_base[0]);
		return;
	}
	vs_base = vs_top;
	vs_push(make_cons(sLdecf,form));
/***/
	VS_PUSH_ENV ;
/***/
	if (!sLdecf->s.s_mflag || sLdecf->s.s_gfdef == OBJNULL)
		FEerror("Where is DECF?", 0);
	funcall(sLdecf->s.s_gfdef);
	eval(vs_base[0]);
}


DEF_ORDINARY("CLEAR-COMPILER-PROPERTIES",sSclear_compiler_properties,SI,"");

DEFUN("CLEAR-COMPILER-PROPERTIES",object,fSclear_compiler_properties,SI
   ,2,2,NONE,OO,OO,OO,OO,(object x0,object x1),"")

{
	/* 2 args */
  RETURN1(Cnil);
}

DEFUN("EMERGENCY-FSET",object,fSemergency_fset,SI
   ,2,2,NONE,OO,OO,OO,OO,(object sym,object function),"")

{

  if (type_of(sym)!=t_symbol || sym->s.s_sfdef!=NOT_SPECIAL || consp(function)) {
    printf("Emergency fset: skipping %-.*s\n",(int)VLEN(sym->s.s_name),sym->s.s_name->st.st_self);
    RETURN1(Cnil);
  }
  sym->s.s_gfdef=function;
  sym->s.s_mflag=FALSE;
  RETURN1(Ct);
}

DEF_ORDINARY("AREF",sLaref,LISP,"");
DEF_ORDINARY("CAR",sLcar,LISP,"");
DEF_ORDINARY("CDR",sLcdr,LISP,"");
DEF_ORDINARY("CHAR",sLchar,LISP,"");
DEF_ORDINARY("DECF",sLdecf,LISP,"");
DEF_ORDINARY("ELT",sLelt,LISP,"");
DEF_ORDINARY("FILL-POINTER",sLfill_pointer,LISP,"");
DEF_ORDINARY("GET",sLget,LISP,"");
DEF_ORDINARY("GETF",sLgetf,LISP,"");
DEF_ORDINARY("GETHASH",sLgethash,LISP,"");
DEF_ORDINARY("INCF",sLincf,LISP,"");
DEF_ORDINARY("POP",sLpop,LISP,"");
DEF_ORDINARY("PUSH",sLpush,LISP,"");
DEF_ORDINARY("SCHAR",sLschar,LISP,"");
DEF_ORDINARY("SETF",sLsetf,LISP,"");
DEF_ORDINARY("SETF-LAMBDA",sSsetf_lambda,SI,"");
DEF_ORDINARY("STRUCTURE-ACCESS",sSstructure_access,SI,"");
DEF_ORDINARY("SVREF",sLsvref,LISP,"");
DEF_ORDINARY("TRACED",sStraced,SI,"");
DEF_ORDINARY("VECTOR",sLvector,LISP,"");

void
gcl_init_assignment(void)
{
	make_special_form("SETQ", Fsetq);
	make_special_form("PSETQ", Fpsetq);
	make_special_form("MULTIPLE-VALUE-SETQ", Fmultiple_value_setq);
	sLsetf=make_special_form("SETF", Fsetf);
	sLpush=make_special_form("PUSH", Fpush);
	sLpop=make_special_form("POP", Fpop);
	sLincf=make_special_form("INCF", Fincf);
	sLdecf=make_special_form("DECF", Fdecf);

}
