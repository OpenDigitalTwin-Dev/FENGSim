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

	error.c

	Errors
*/

#include "include.h"
object siSuniversal_error_handler;

object sSterminal_interrupt;

void
assert_error(const char *a,unsigned l,const char *f,const char *n) {

  if (!raw_image && core_end && core_end==sbrk(0) && errno!=ENOMEM && ihs_top>=ihs_org)
    FEerror("The assertion ~a on line ~a of ~a in function ~a failed: ~a",5,
	    make_simple_string(a),make_fixnum(l),
	    make_simple_string(f),make_simple_string(n),make_simple_string(strerror(errno)));
  else {
    emsg("The assertion %s on line %d of %s in function %s failed: %s",a,l,f,n,strerror(errno));
    gcl_abort();
  }

}

void
terminal_interrupt(int correctable)
{
    signals_allowed = sig_normal; 
    ifuncall1(sSterminal_interrupt, correctable?Ct:Cnil);
}

static object
ihs_function_name(object x)
{
	object y;

	switch (type_of(x)) {
	case t_symbol:
		return(x);

	case t_cons:
		y = x->c.c_car;
		if (y == sLlambda)
			return(sLlambda);
		if (y == sSlambda_closure)
			return(sSlambda_closure);
		if (y == sSlambda_block || y == sSlambda_block_expanded) {
			x = x->c.c_cdr;
			if (!consp(x))
				return(sSlambda_block);
			return(x->c.c_car);
		}
		if (y == sSlambda_block_closure) {
			x = x->c.c_cdr;
			if (!consp(x))
				return(sSlambda_block_closure);
			x = x->c.c_cdr;
			if (!consp(x))
				return(sSlambda_block_closure);
			x = x->c.c_cdr;
			if (!consp(x))
				return(sSlambda_block_closure);
			x = x->c.c_cdr;
			if (!consp(x))
				return(sSlambda_block_closure);
			return(x->c.c_car);
		}
		/* a general special form */
		if (y->s.s_sfdef != NOT_SPECIAL)
		  return y;
		return(Cnil);

	/* case t_cfun: */

	/* 	return(x->cf.cf_name); */

        case t_function:

		return(Cnil);

	default:
		return(Cnil);
	}
}

object
ihs_top_function_name(ihs_ptr h)
{
	object x;


	while (h >= ihs_org) {
		x = ihs_function_name(h->ihs_function);
		if (x != Cnil)
			return(x);
		h--;
	}
	return(Cnil);
}

static object
Icall_gen_error_handler_ap(object ci,object cs,object en,object es,ufixnum n,va_list ap) {

  object *b;
  ufixnum i;

  n+=5;
  b=alloca(n*sizeof(*b));
  b[0]= en;
  b[1]= ci; 
  b[2] = ihs_top_function_name(ihs_top);
  b[3] = cs;
  b[4] = es;
   
  for (i=5;i<n;i++)
    b[i]= va_arg(ap,object);

  return funcall_vec(coerce_funcall_object_to_function(sSuniversal_error_handler),n,b);

  return Cnil;

}

object
Icall_gen_error_handler(object ci,object cs,object en,object es,ufixnum n,...) {

  object x;
  va_list ap;

  va_start(ap,n);

  x=Icall_gen_error_handler_ap(ci,cs,en,es,n,ap);

  va_end(ap);

  return x;

}

object
Icall_gen_error_handler_noreturn(object ci,object cs,object en,object es,ufixnum n,...) {

  va_list ap;

  va_start(ap,n);

  Icall_gen_error_handler_ap(ci,cs,en,es,n,ap);

  va_end(ap);

  while (1);

}

/*
	Lisp interface to IHS
*/

static ihs_ptr get_ihs_ptr(object x) {

  ihs_ptr p;

  if (type_of(x) != t_fixnum)
    goto ILLEGAL;
  p = ihs_org + fix(x);
  p=p<ihs_org ? ihs_org : p;
  p=p>ihs_top ? ihs_top : p;
  return p;
 ILLEGAL:
  FEerror("~S is an illegal ihs index.", 1, x);
  return(NULL);

}

DEFUN("IHS-TOP",object,fSihs_top,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  fixnum i=ihs_top-ihs_org;
  for (;i>0 && type_of(ihs_org[i].ihs_function)==t_function && 
	 ihs_org[i].ihs_function->fun.fun_self==FFN(fSihs_top);i--);
  RETURN1(make_fixnum(i));
}

DEFUN("IHS-FUN",object,fSihs_fun,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0 = get_ihs_ptr(x0)->ihs_function;
  RETURN1(x0);
}

DEFUN("IHS-VS",object,fSihs_vs,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0 = make_fixnum(get_ihs_ptr(x0)->ihs_base - vs_org);
  RETURN1(x0);
}

static frame_ptr get_frame_ptr(object x) {

  frame_ptr p;
  
  if (type_of(x) != t_fixnum)
    goto ILLEGAL;
  p = frs_org + fix(x);
  if (fix(x)==0) return p;
  if (frs_org <= p && p <= frs_top)
    return(p);
 ILLEGAL:
  FEerror("~S is an illegal frs index.", 1, x);
  return NULL;

}

DEFUN("FRS-TOP",object,fSfrs_top,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  RETURN1((make_fixnum(frs_top - frs_org)));
}

DEFUN("FRS-VS",object,fSfrs_vs,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0 = make_fixnum(get_frame_ptr(x0)->frs_lex - vs_org);
  RETURN1(x0);
}

DEFUN("FRS-BDS",object,fSfrs_bds,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0=make_fixnum(get_frame_ptr(x0)->frs_bds_top - bds_org);
  RETURN1(x0);
}

DEFUN("FRS-CLASS",object,fSfrs_class,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {

  enum fr_class c;

  c = get_frame_ptr(x0)->frs_class;
  if (c == FRS_CATCH) x0 = sKcatch;
  else if (c == FRS_PROTECT) x0 = sKprotect;
  else if (c == FRS_CATCHALL) x0 = sKcatchall;
  else FEerror("Unknown frs class was detected.", 0);
  RETURN1(x0);

}

DEFUN("FRS-TAG",object,fSfrs_tag,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0 = get_frame_ptr(x0)->frs_val;
  RETURN1(x0);
}

DEFUN("FRS-IHS",object,fSfrs_ihs,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0=make_fixnum(get_frame_ptr(x0)->frs_ihs - ihs_org);
  RETURN1(x0);
}

static bds_ptr get_bds_ptr(object x) {

  bds_ptr p;
  
  if (type_of(x) != t_fixnum)
    goto ILLEGAL;
  p = bds_org + fix(x);
  if (0 == fix(x)) return p;
  if (bds_org <= p && p <= bds_top)
    return(p);
 ILLEGAL:
  FEerror("~S is an illegal bds index.", 1, x);
  return NULL;

}

DEFUN("BDS-TOP",object,fSbds_top,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  RETURN1((make_fixnum(bds_top - bds_org)));
}

DEFUN("BDS-VAR",object,fSbds_var,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0 = get_bds_ptr(x0)->bds_sym;
  RETURN1(x0);
}

DEFUN("BDS-VAL",object,fSbds_val,SI,1,1,NONE,IO,OO,OO,OO,(object x0),"") {
  RETURN1(get_bds_ptr(x0)->bds_val);
}

static object *get_vs_ptr(object x) {

  object *p;

  if (type_of(x) != t_fixnum)
    goto ILLEGAL;
  p = vs_org + fix(x);
  if (vs_org <= p && p < vs_top)
    return(p);
 ILLEGAL:
  FEerror("~S is an illegal vs index.", 1, x);
  return NULL;

}

DEFUN("VS-TOP",object,fSvs_top,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  object x;
  x = (make_fixnum(vs_top - vs_org));
  RETURN1(x);
}

DEFUN("VS",object,fSvs,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {
  x0 = *get_vs_ptr(x0);
  RETURN1(x0);
}

DEFUN("SCH-FRS-BASE",object,fSsch_frs_base,SI,2,2,NONE,OO,OO,OO,OO,(object x0,object x1),"") {

  frame_ptr x;
  ihs_ptr y;
  
  /* 2 args */
  if (type_of(x1)!=t_fixnum || fix(x1)>ihs_top-ihs_org || fix(x1)<0) RETURN1(Cnil);
  y = get_ihs_ptr(x1);
  for (x = get_frame_ptr(x0);
       x <= frs_top && x->frs_ihs < y; 
       x++);
  if (x > frs_top) x0 = Cnil;
  else x0 = make_fixnum(x - frs_org);
  
  RETURN1(x0);

}

DEFUNM("INTERNAL-SUPER-GO",object,fSinternal_super_go,SI,3,3,NONE,OO,OO,OO,OO,(object tag,object x1,object x2),"") {
  frame_ptr fr;
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;
  
  fr = frs_sch(tag);
  if (fr == NULL)
    FEerror("The tag ~S is missing.", 1, tag);
  if (x2 == Cnil)
    tag = x1;
  else
    tag = MMcons(tag, x1);
  unwind(fr,tag);
  RETURN0;

}

DEF_ORDINARY("UNIVERSAL-ERROR-HANDLER",sSuniversal_error_handler,SI
	     ,"Redefined in lisp, this is the function called by the \
               internal error handling mechanism. \
               Args:  (error-name correctable function-name \
                       continue-format-string error-format-string \
                       &rest args)");
DEFUN("UNIVERSAL-ERROR-HANDLER",object,fSuniversal_error_handler,SI
	   ,1,F_ARG_LIMIT,NONE,OO,OO,OO,OO,
	  (object x0,object x1,object x2,object x3,object x4,
	   object error_fmt_string,...),"") {

  va_list ap;
  object z,l,f;
  ufixnum n=INIT_NARGS(6);

  /* 5 args */
  princ(x0,Cnil);
  putchar(' ');
  princ(x1,Cnil);
  putchar(' ');
  princ(x2,Cnil);
  putchar(' ');
  princ(x3,Cnil);
  putchar(' ');
  princ(x4,Cnil);
  putchar(' ');
  princ(error_fmt_string,Cnil);
  putchar(' ');
  va_start(ap,error_fmt_string);
  for (l=Cnil,f=OBJNULL;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL;)
    princ(z,Cnil);
  flush_stream(symbol_value(sLAstandard_outputA));
  va_end(ap);
  printf("\nLisp initialization failed.\n");
  
  exit(0);
  RETURN1(x0);

}

void
check_arg_failed(int n)
{
  if (n<vs_top-vs_base)
    FEtoo_few_arguments(vs_base,vs_top);
  else
    FEtoo_many_arguments(vs_base,vs_top);
}

void
too_few_arguments(void)
{
	FEtoo_few_arguments(vs_base, vs_top);
}

void
too_many_arguments(void)
{
	FEtoo_many_arguments(vs_base, vs_top);
}

void
ck_larg_at_least(int n, object x) {
	for(; n > 0; n--, x = x->c.c_cdr)
		if(endp(x))
		  FEerror("APPLY sent too few arguments to LAMBDA.", 0);
}

void
ck_larg_exactly(int n, object x) {
	for(; n > 0; n--, x = x->c.c_cdr)
		if(endp(x))
		  FEerror("APPLY sent too few arguments to LAMBDA.", 0);
	if(!endp(x)) FEerror("APPLY sent too many arguments to LAMBDA.", 0);
}

void
invalid_macro_call(void)
{
	FEinvalid_macro_call();
}

object
wrong_type_argument(object typ, object obj)
{
	FEwrong_type_argument(typ, obj);
	/*  no return  */
	return(Cnil);
}

void
illegal_declare(object form)
{
	FEinvalid_form("~S is an illegal declaration form.", form);
}

void
not_a_string_or_symbol(object x)
{
  FEerror("~S is not a string or symbol.", 1, x);
}

void
not_a_symbol(object obj)
{
/* 	FEinvalid_variable("~S is not a symbol.", obj); */
	FEwrong_type_argument(sLsymbol,obj);
}

int
not_a_variable(object obj)
{
	FEinvalid_variable("~S is not a variable.", obj);
	return -1;
}

void
illegal_index(object x, object i)
{
	FEerror("~S is an illegal index to ~S.", 2, i, x);
}

void
check_socket(object x)
{

 if (type_of(x) != t_stream || x->sm.sm_mode != smm_socket) 
   FEwrong_type_argument(sSsocket,x);
  
}

void
check_stream(object strm)
{
if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
}


void
check_arg_range(fixnum nargs,int n, int m) {  
  
  if (nargs < n)
    FEwrong_no_args("Need at least ~D argument(s).",make_fixnum(n));
  else if (nargs > m)
    FEwrong_no_args("Need no more than ~D argument(s).",make_fixnum(m));

 }
			 
     
DEF_ORDINARY("TERMINAL-INTERRUPT",sSterminal_interrupt,SI,"");
DEF_ORDINARY("WRONG-TYPE-ARGUMENT",sKwrong_type_argument,KEYWORD,"");
DEF_ORDINARY("CONTROL-ERROR",sKcontrol_error,KEYWORD,"");
DEF_ORDINARY("PROGRAM-ERROR",sKprogram_error,KEYWORD,"");
DEF_ORDINARY("TOO-FEW-ARGUMENTS",sKtoo_few_arguments,KEYWORD,"");
DEF_ORDINARY("TOO-MANY-ARGUMENTS",sKtoo_many_arguments,KEYWORD,"");
DEF_ORDINARY("UNEXPECTED-KEYWORD",sKunexpected_keyword,KEYWORD,"");
DEF_ORDINARY("INVALID-FORM",sKinvalid_form,KEYWORD,"");
DEF_ORDINARY("UNBOUND-VARIABLE",sKunbound_variable,KEYWORD,"");
DEF_ORDINARY("INVALID-VARIABLE",sKinvalid_variable,KEYWORD,"");
DEF_ORDINARY("UNDEFINED-FUNCTION",sKundefined_function,KEYWORD,"");
DEF_ORDINARY("INVALID-FUNCTION",sKinvalid_function,KEYWORD,"");
DEF_ORDINARY("PACKAGE-ERROR",sKpackage_error,KEYWORD,"");
DEF_ORDINARY("PARSE-ERROR",sKparse_error,KEYWORD,"");
DEF_ORDINARY("READER-ERROR",sKreader_error,KEYWORD,"");
DEF_ORDINARY("CATCH",sKcatch,KEYWORD,"");
DEF_ORDINARY("PROTECT",sKprotect,KEYWORD,"");
DEF_ORDINARY("CATCHALL",sKcatchall,KEYWORD,"");


DEF_ORDINARY("CONDITION",sLcondition,LISP,"");
DEF_ORDINARY("SERIOUS-CONDITION",sLserious_condition,LISP,"");
DEF_ORDINARY("SIMPLE-CONDITION",sLsimple_condition,LISP,"");

DEF_ORDINARY("ERROR",sLerror,LISP,"");
DEF_ORDINARY("SIMPLE-ERROR",sLsimple_error,LISP,"");
DEF_ORDINARY("FORMAT-CONTROL",sKformat_control,KEYWORD,"");
DEF_ORDINARY("FORMAT-ARGUMENTS",sKformat_arguments,KEYWORD,"");

DEF_ORDINARY("TYPE-ERROR",sLtype_error,LISP,"");
DEF_ORDINARY("DATUM",sKdatum,KEYWORD,"");
DEF_ORDINARY("EXPECTED-TYPE",sKexpected_type,KEYWORD,"");
DEF_ORDINARY("SIMPLE-TYPE-ERROR",sLsimple_type_error,LISP,"");

DEF_ORDINARY("PROGRAM-ERROR",sLprogram_error,LISP,"");
DEF_ORDINARY("CONTROL-ERROR",sLcontrol_error,LISP,"");
DEF_ORDINARY("PACKAGE-ERROR",sLpackage_error,LISP,"");
DEF_ORDINARY("PACKAGE",sKpackage,KEYWORD,"");

DEF_ORDINARY("STREAM-ERROR",sLstream_error,LISP,"");
DEF_ORDINARY("STREAM",sKstream,KEYWORD,"");
DEF_ORDINARY("END-OF-FILE",sLend_of_file,LISP,"");

DEF_ORDINARY("FILE-ERROR",sLfile_error,LISP,"");
DEF_ORDINARY("PATHNAME",sKpathname,KEYWORD,"");

DEF_ORDINARY("CELL-ERROR",sLcell_error,LISP,"");
DEF_ORDINARY("NAME",sKname,KEYWORD,"");
DEF_ORDINARY("UNBOUND-SLOT",sLunbound_slot,LISP,"");
DEF_ORDINARY("UNBOUND-VARIABLE",sLunbound_variable,LISP,"");
DEF_ORDINARY("UNDEFINED-FUNCTION",sLundefined_function,LISP,"");

DEF_ORDINARY("ARITHMETIC-ERROR",sLarithmetic_error,LISP,"");
DEF_ORDINARY("OPERATION",sKoperation,KEYWORD,"");
DEF_ORDINARY("OPERANDS",sKoperands,KEYWORD,"");
DEF_ORDINARY("DIVISION-BY-ZERO",sLdivision_by_zero,LISP,"");
DEF_ORDINARY("FLOATING-POINT-OVERFLOW",sLfloating_point_overflow,LISP,"");
DEF_ORDINARY("FLOATING-POINT-UNDERFLOW",sLfloating_point_underflow,LISP,"");
DEF_ORDINARY("FLOATING-POINT-INEXACT",sLfloating_point_inexact,LISP,"");
DEF_ORDINARY("FLOATING-POINT-INVALID-OPERATION",sLfloating_point_invalid_operation,LISP,"");

DEF_ORDINARY("PARSE-ERROR",sLparse_error,LISP,"");

DEF_ORDINARY("PRINT-NOT-READABLE",sLprint_not_readable,LISP,"");

DEF_ORDINARY("READER-ERROR",sLreader_error,LISP,"");
DEF_ORDINARY("PATHNAME-ERROR",sLpathname_error,SI,"");

DEF_ORDINARY("STORAGE-CONDITION",sLstorage_condition,LISP,"");

DEF_ORDINARY("WARNING",sLwarning,LISP,"");
DEF_ORDINARY("SIMPLE-WARNING",sLsimple_warning,LISP,"");
DEF_ORDINARY("STYLE-WARNING",sLstyle_warning,LISP,"");

void
gcl_init_error(void) {
  null_string = make_simple_string("");
  enter_mark_origin(&null_string);
}
