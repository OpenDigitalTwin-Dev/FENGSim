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

	toplevel.c

	Top-Level Forms and Declarations
*/

#include "include.h"

object sLcompile, sLload, sLeval, sKcompile_toplevel, sKload_toplevel, sKexecute;
object sLprogn;


object sLwarn;

object sSAinhibit_macro_specialA;

object sLtypep;

static void
FFN(Fdefun)(object args)
{

	object name,oname;
	object body, form;

	if (endp(args) || endp(MMcdr(args)))
		FEtoo_few_argumentsF(args);
	if (MMcadr(args) != Cnil && !consp(MMcadr(args)))
		FEerror("~S is an illegal lambda-list.", 1, MMcadr(args));
	oname=name = MMcar(args);
	
	if (type_of(name) != t_symbol)
	  name=ifuncall1(sSfunid_to_sym,name);

	if (name->s.s_sfdef != NOT_SPECIAL) {
		if (name->s.s_mflag) {
			if (symbol_value(sSAinhibit_macro_specialA) != Cnil)
				name->s.s_sfdef = NOT_SPECIAL;
		} else if (symbol_value(sSAinhibit_macro_specialA) != Cnil)
		 FEerror("~S, a special form, cannot be redefined.", 1, name);
	}
	vs_base = vs_top;
	if (lex_env[0] == Cnil && lex_env[1] == Cnil && lex_env[2] == Cnil) {
	  vs_push(MMcons(sSlambda_block, args));
	} else {
	  vs_push(MMcons(lex_env[2], args));
	  vs_base[0] = MMcons(lex_env[1], vs_base[0]);
	  vs_base[0] = MMcons(lex_env[0], vs_base[0]);
	  vs_base[0] = MMcons(sSlambda_block_closure, vs_base[0]);
	}
	{/* object fname; */
	  vs_base[0]=fSfset_in(name,vs_base[0],name);/*FIXME ?*/
	/* object x=alloc_object(t_ifun); */
	/* x->ifn.ifn_self=vs_base[0]; */
	/* x->ifn.ifn_name=name; */
	/* x->ifn.ifn_call=Cnil; */
	/* vs_base[0]=x; */
	/* fname =  clear_compiler_properties(name,vs_base[0]); */
	/* fname->s.s_gfdef = vs_base[0]; */
	/* fname->s.s_mflag = FALSE; */
	}
	vs_base[0] = oname;
	for (body = MMcddr(args);  !endp(body);  body = body->c.c_cdr) {
	  form = macro_expand(body->c.c_car);
	  if (stringp(form)) {
	    if (endp(body->c.c_cdr))
	      break;
	    vs_push(form);
	    name->s.s_plist =
	      putf(name->s.s_plist,
		   form,
		   sSfunction_documentation);
	    vs_popp;
	    break;
	  }
	  if (!consp(form) || form->c.c_car != sLdeclare)
	    break;
	}
}
	
static void
FFN(siLAmake_special)(void)
{
	check_arg(1);
	check_type_sym(&vs_base[0]);
	if ((enum stype)vs_base[0]->s.s_stype == stp_constant)
		FEerror("~S is a constant.", 1, vs_base[0]);
	vs_base[0]->s.s_stype = (short)stp_special;
}

DEFUN("OBJNULL",object,fSobjnull,SI,0,0,NONE,IO,OO,OO,OO,(void),"") {return OBJNULL;}

DEFUN("*MAKE-CONSTANT",object,fSAmake_constant,SI,2,2,NONE,OO,OO,OO,OO, \
	  (object s,object v),"") { 

  check_type_sym(&s);
  switch(s->s.s_stype) {
  case stp_special:
    FEerror("The argument ~S to defconstant is a special variable.", 1, s);
    break;
  case stp_constant:
    break;
  default:
    s->s.s_dbind=v;
    break;
  }

  s->s.s_stype=stp_constant;

  RETURN1(s);

}

/* static void */
/* FFN(siLAmake_constant)(void) */
/* { */
/* 	check_arg(2); */
/* 	check_type_sym(&vs_base[0]); */
/* 	if ((enum stype)vs_base[0]->s.s_stype == stp_special) */
/* 		FEerror( */
/* 		 "The argument ~S to DEFCONSTANT is a special variable.", */
/* 		 1, vs_base[0]); */
/* 	vs_base[0]->s.s_stype = (short)stp_constant; */
/* 	vs_base[0]->s.s_dbind = vs_base[1]; */
/* 	vs_popp; */
/* } */

static void
FFN(Feval_when)(object arg)
{

	object *base = vs_top;
	object ss;
	bool flag = FALSE;

	if(endp(arg))
		FEtoo_few_argumentsF(arg);
	for (ss = MMcar(arg);  !endp(ss);  ss = MMcdr(ss))
		if(MMcar(ss) == sLeval || (MMcar(ss) == sKexecute) )
			flag = TRUE;
		else if(MMcar(ss) != sLload && MMcar(ss) != sLcompile &&
                          MMcar(ss) != sKload_toplevel && MMcar(ss) != sKcompile_toplevel )
		 FEinvalid_form("~S is an undefined situation for EVAL-WHEN.",
				MMcar(ss));
	if(flag) {
		vs_push(make_cons(sLprogn, MMcdr(arg)));
		eval(vs_head);
	} else {
		vs_base = base;
		vs_top = base+1;
		vs_base[0] = Cnil;
	}
}

static void
FFN(Fload_time_value)(object arg)
{

	if(endp(arg))
		FEtoo_few_argumentsF(arg);
	if(!endp(MMcdr(arg)) && !endp(MMcddr(arg)))
		FEtoo_many_argumentsF(arg);
	vs_push(MMcar(arg));
	eval(vs_head);

}

static void
FFN(Fdeclare)(object arg)
{
	FEerror("DECLARE appeared in an invalid position.", 0);
}

static void
FFN(Flocally)(object body)
{
	object *oldlex = lex_env;

	lex_copy();
	body = find_special(body, NULL, NULL,NULL);
	vs_push(body);
	Fprogn(body);
	lex_env = oldlex;
}

static void
FFN(Fthe)(object args)
{

	object *vs;

	if(endp(args) || endp(MMcdr(args)))
		FEtoo_few_argumentsF(args);
	if(!endp(MMcddr(args)))
		FEtoo_many_argumentsF(args);
	eval(MMcadr(args));
	args = MMcar(args);
	if (consp(args) && MMcar(args) == sLvalues) {
	  vs = vs_base;
	  for (args=MMcdr(args); !endp(args) && vs<vs_top; args=MMcdr(args), vs++) {
	    if (MMcar(args)==ANDrest) {
	      for (args=MMcdr(args);vs<vs_top;vs++)
		if (ifuncall2(sLtypep, *vs, MMcar(args)) == Cnil)
		  FEwrong_type_argument(MMcar(args), *vs);
	    } else if (MMcar(args)==ANDoptional)
	      vs--;
	    else if (ifuncall2(sLtypep, *vs, MMcar(args)) == Cnil)
	      FEwrong_type_argument(MMcar(args), *vs);
	  }
	  /*}
		if (vs < vs_top)
			FEerror("Too few return values.", 0);*/
	  for (; !endp(args) && MMcar(args)!=ANDrest && MMcar(args)!=ANDoptional; args=MMcdr(args))
	    if (ifuncall2(sLtypep, Cnil, MMcar(args)) == Cnil)
	      FEwrong_type_argument(MMcar(args), Cnil);
	  
	} else {
	  if (sLtypep->s.s_gfdef!=OBJNULL && ifuncall2(sLtypep, vs_base[0], args) == Cnil)
			FEwrong_type_argument(args, vs_base[0]);
	}
}

DEF_ORDINARY("WILD-PATHNAME-P",sLwild_pathname_p,LISP,"");
DEF_ORDINARY("LDB",sLldb,LISP,"");
DEF_ORDINARY("LDB-TEST",sLldb_test,LISP,"");
DEF_ORDINARY("DPB",sLdpb,LISP,"");
DEF_ORDINARY("DEPOSIT-FIELD",sLdeposit_field,LISP,"");
DEF_ORDINARY("COMPILE",sLcompile,LISP,"");
DEF_ORDINARY("COMPILE-TOPLEVEL",sKcompile_toplevel,KEYWORD,"");
DEF_ORDINARY("DECLARE",sLdeclare,LISP,"");
DEF_ORDINARY("EVAL",sLeval,LISP,"");
DEF_ORDINARY("EXECUTE",sKexecute,KEYWORD,"");
DEF_ORDINARY("FUNCTION-DOCUMENTATION",sSfunction_documentation,SI,"");
DEF_ORDINARY("LOAD",sLload,LISP,"");
DEF_ORDINARY("LOAD-TOPLEVEL",sKload_toplevel,KEYWORD,"");
DEF_ORDINARY("PROGN",sLprogn,LISP,"");
DEF_ORDINARY("TYPEP",sLtypep,LISP,"");
DEF_ORDINARY("VALUES",sLvalues,LISP,"");
DEF_ORDINARY("VARIABLE-DOCUMENTATION",sSvariable_documentation,SI,"");
DEF_ORDINARY("WARN",sLwarn,LISP,"");

void
gcl_init_toplevel(void)
{
	make_special_form("DEFUN",Fdefun);
	make_si_function("*MAKE-SPECIAL", siLAmake_special);
	/* make_si_function("*MAKE-CONSTANT", siLAmake_constant); */
	make_special_form("EVAL-WHEN", Feval_when);
	make_special_form("LOAD-TIME-VALUE", Fload_time_value);
	make_special_form("THE", Fthe);
	sLdeclare=make_function("DECLARE",Fdeclare);
	make_special_form("LOCALLY",Flocally);


}
