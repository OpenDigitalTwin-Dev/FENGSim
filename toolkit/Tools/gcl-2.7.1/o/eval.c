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
	eval.c
*/

#include "include.h"
#include "sfun_argd.h"

static void
call_applyhook(object);


struct nil3 { object nil3_self[3]; } three_nils;

#ifdef DEBUG_AVMA
#undef DEBUG_AVMA
unsigned long avma,bot;
#define DEBUG_AVMA unsigned long saved_avma =  avma;
warn_avma()
{ 
  print(list(2,make_simple_string("avma changed"),ihs_top_function_name(ihs_top)),
	sLAstandard_outputA->s.s_dbind);
}
#define CHECK_AVMA if(avma!= saved_avma) warn_avma();
#define DEBUGGING_AVMA  
#else
#define DEBUG_AVMA
#define CHECK_AVMA
#endif



/*  object c_apply_n(long int (*fn)(), int n, object *x); */

object sSAbreak_pointsA;
object sSAbreak_stepA;

/* for t_sfun,t_gfun with args on vs stack */

#define POP_BITS(x_,y_) ({ufixnum _t=x_&((1<<y_)-1);x_>>=y_;_t;})
#define COERCE_ARG(a,type)  \
  ({enum ftype _t=type;\
    _t==f_object ? a : (_t==f_fixnum ? (object)(fixint(a)) : (object)otoi(a));})
#define UNCOERCE_ARG(a,type)  \
  ({enum ftype _t=type;\
     _t==f_object ? a : (_t==f_fixnum ? make_fixnum((fixnum)a) : make_integer((GEN)a));})

#include "apply_n.h"

static object
quick_call_function_vec(object fun,ufixnum n,object *b) {

  return c_apply_n_fun(fun,n,b);

}

static object
quick_call_function_vec_coerce(object fun,ufixnum n,object *b) {

  register object res;
  ufixnum argd,j;
  enum ftype restype;
  object *tmp;

  argd=fun->fun.fun_argd;
  restype = POP_BITS(argd,2);

  if (argd) {
    static object q[MAX_ARGS+1];
    for (tmp=q,j=0;j<n;j++)
      tmp[j]=COERCE_ARG(b[j],POP_BITS(argd,2));
  } else
    tmp=b;

  res=quick_call_function_vec(fun,n,tmp);

  return UNCOERCE_ARG(res,restype);

}

static object
unwind_vals(object *vals,object *base) {
  
  ufixnum n;
  object o;

  o=vs_base[0];
  n=vs_top-vs_base;
  if (vals) {
    if (n>1) memmove(vals,vs_base+1,(n-1)*sizeof(*vals));
    vs_top=(vals-1)+n;
  } else
    vs_top=base;
  
  return n>0 ? o : Cnil;
  
}

object
funcall_vec(object fun,fixnum n,object *b) {

  ufixnum m=labs(n),l=m;
  object x;
  
  if (n<0)
    for (l=m-1,x=b[l];x!=Cnil;l++,x=x->c.c_cdr);

  fcall.argd=n;
  fcall.fun=fun;

  if (l<fun->fun.fun_minarg) {
    FEtoo_few_arguments(b,b+l);
    return Cnil;
  }

  if (l>fun->fun.fun_maxarg) {
    FEtoo_many_arguments(b,b+l);
    return Cnil;
  }
    
  return quick_call_function_vec_coerce(fun,m,b);

}


static object
funcall_ap(object fun,fixnum n,va_list ap) {

  static object b[MAX_ARGS+1];
  object *t=b;
  ufixnum j=labs(n),i;

  for (i=j;i--;)
    *t++=va_arg(ap,object);
  if (n<0 && fun->fun.fun_minarg>(j-1)) {
    object x=*--t;
    for (i=fun->fun.fun_minarg-(j-1);i--;*t++=x->c.c_car,x=x->c.c_cdr,n--)
      if (x==Cnil)
	FEtoo_few_arguments(b,t);
    *t++=x;
  }

  return funcall_vec(fun,n,b);

}
    

static void
quick_call_function(object fun) {

  ufixnum n;
  object *base;

  base=vs_base;
  n=vs_top-vs_base;
  if (n<fun->fun.fun_minarg) {
    FEtoo_few_arguments(base,vs_top); 
    return;
  }
  
  if (n>fun->fun.fun_maxarg) {
    FEtoo_many_arguments(base,vs_top); 
    return;
  }

  fcall.argd=n;
  fcall.valp=(fixnum)(base+1);
  fcall.fun=fun;

  base[0]=quick_call_function_vec_coerce(fun,n,vs_base);

  vs_base=base;
  if (!fun->fun.fun_neval && !fun->fun.fun_vv)
    vs_top=base+1;

  return;

}


void
Iinvoke_c_function_from_value_stack(object (*f)(), ufixnum argd) { 
  
  static union lispunion fun;
  extern void quick_call_function(object);/*FIXME*/

  set_type_of(&fun,t_function);
  fun.fun.fun_self=f;
  fun.fun.fun_data=Cnil;
  fun.fun.fun_plist=Cnil;
  fun.fun.fun_argd=F_TYPES(argd);
  fun.fun.fun_minarg=F_MIN_ARGS(argd);
  fun.fun.fun_maxarg=F_MAX_ARGS(argd);;
  if (!(argd&ONE_VAL)) {
    fun.fun.fun_neval=31;
    fun.fun.fun_vv=1;
  }

  quick_call_function((object)&fun);

} 

static object
kar(object x) {
  if (consp(x))
    return(x->c.c_car);
  FEwrong_type_argument(sLcons, x);
  return(Cnil);
}

void
funcall(object fun) { 
/*         object VOL sfirst=NULL; */
/*         wipe_stack(&sfirst); */
/* 	{ */
  object temporary=OBJNULL;
  object x=OBJNULL;
  object * VOL top=NULL;
  object *lex=NULL;
  bds_ptr old_bds_top=NULL;
  VOL bool b=0;
  bool c=0;
  DEBUG_AVMA
    TOP:
  if (fun == OBJNULL)
    FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   MMcall(fun); */
  /*   CHECK_AVMA; return; */
    
  case t_function:
    {int i=Rset;
      if (!i) {ihs_check;ihs_push(fun);}
      quick_call_function(fun);
      if (!i) ihs_pop();
    }
    return;
    
  case t_symbol:
    {
      object x = fun->s.s_gfdef;
      if (x!=OBJNULL) { fun = x; goto TOP;}
      else
	FEundefined_function(fun);
    }
    
  /* case t_ifun: */
  /*   { */
  /*     object x = fun->ifn.ifn_self; */
  /*     if (x) { fun = x;  /\* ihs_check;ihs_push(fun); *\/break;} */
  /*     else */
  /* 	FEundefined_function(fun); */
  /*   } */
    
  case t_cons:
    if (fun->c.c_car!=sLlambda &&
	fun->c.c_car!=sSlambda_closure &&
	fun->c.c_car!=sSlambda_block &&
	fun->c.c_car!=sSlambda_block_expanded &&
	fun->c.c_car!=sSlambda_block_closure)
      FEinvalid_function(fun);
    break;
    
  default:
    FEinvalid_function(fun);
  }
  
  /*
    This part is the same as that of funcall_no_event.
  */
  
  /* we may have pushed the calling form if this is called invoked from 
     eval.   A lambda call requires vs_push's, so we can tell
     if we pushed by vs_base being the same.
  */
  { VOL int not_pushed = 0;
    if (vs_base != ihs_top->ihs_base){
      ihs_check;
      ihs_push(fun);
    }
    else
      not_pushed = 1;
    
    ihs_top->ihs_base = lex_env;
    x = MMcar(fun);
    top = vs_top;
    lex = lex_env;
    old_bds_top = bds_top;
    
    /* maybe digest this lambda expression
       (lambda-block-expand name ..) has already been
       expanded.    The value of lambda-block-expand may
       be a compiled function in which case we say expand
       with it)
    */
    
    if (x == sSlambda_block_expanded) {
      
      b = TRUE;
      c = FALSE;
      fun = fun->c.c_cdr;
      
    } else if (x == sSlambda_block) {
      b = TRUE;
      c = FALSE;
      if(sSlambda_block_expanded->s.s_dbind!=OBJNULL)
	fun = ifuncall1(sSlambda_block_expanded->s.s_dbind,fun);
      
      fun = fun->c.c_cdr;
      
      
      
    } else if (x == sSlambda_closure) {
      b = FALSE;
      c = TRUE;
      fun = fun->c.c_cdr;
    } else if (x == sLlambda) {
      b = c = FALSE;
      fun = fun->c.c_cdr;
    } else if (x == sSlambda_block_closure) {
      b = c = TRUE;
      fun = fun->c.c_cdr;
    } else
      b = c = TRUE;
    if (c) {
      vs_push(kar(fun));
      fun = fun->c.c_cdr;
      vs_push(kar(fun));
      fun = fun->c.c_cdr;
      vs_push(kar(fun));
      fun = fun->c.c_cdr;
    } else {
      *(struct nil3 *)vs_top = three_nils;
      vs_top += 3;
    }
    if (b) {
      x = kar(fun);  /* block name */
      fun = fun->c.c_cdr;
    }
    lex_env = top;
    vs_push(fun);
    lambda_bind(top);
    ihs_top->ihs_base = lex_env;
    if (b) {
      fun = temporary = alloc_frame_id();
      /*  lex_block_bind(x, temporary);  */
      temporary = MMcons(temporary, Cnil);
      temporary = MMcons(sLblock, temporary);
      temporary = MMcons(x, temporary);
      lex_env[2] = MMcons(temporary, lex_env[2]);
      frs_push(FRS_CATCH, fun);
      if (nlj_active) {
	nlj_active = FALSE;
	goto END;
      }
    }
    x = top[3];  /* body */
    if(endp(x)) {
      vs_base = vs_top;
      vs_push(Cnil);
    } else {
      top = vs_top;
      for (;;) {
	eval(MMcar(x));
	x = MMcdr(x);
	if (endp(x))
	  break;
	vs_top = top;
      }
    }
  END:
    if (b)
      frs_pop();
    bds_unwind(old_bds_top);
    lex_env = lex;
    if (not_pushed == 0) {ihs_pop();}
    CHECK_AVMA;
  }
}

void
funcall_no_event(object fun) {
  DEBUG_AVMA
    if (fun == OBJNULL)
      FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   (*fun->cf.cf_self)(); */
  /*   break; */
    
  case t_function:
    quick_call_function(fun); return;
  default:
    funcall(fun);
    
  }
}

void
lispcall(object *funp, int narg) {

  DEBUG_AVMA
    object fun = *funp;
  
  vs_base = funp + 1;
  vs_top = vs_base + narg;
  
  if (fun == OBJNULL)
    FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   MMcall(fun); */
  /*   break; */
    
  default:
    funcall(fun);
    
  }
  CHECK_AVMA;
}

void
lispcall_no_event(object *funp, int narg) {

  DEBUG_AVMA
    object fun = *funp;
  
  vs_base = funp + 1;
  vs_top = vs_base + narg;
  
  if (fun == OBJNULL)
    FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   (*fun->cf.cf_self)(); */
  /*   break; */
    
  default:
    funcall(fun);
    
  }
  CHECK_AVMA;
}

void
symlispcall(object sym, object *base, int narg) {
  DEBUG_AVMA
    object fun = symbol_function(sym);
  
  vs_base = base;
  vs_top = vs_base + narg;

  if (fun == OBJNULL)
    FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   MMcall(fun); */
  /*   break; */
    
  default:
    funcall(fun);
  }
  CHECK_AVMA;
}

void
symlispcall_no_event(object sym, object *base, int narg) {

  DEBUG_AVMA
    object fun = symbol_function(sym);
  
  vs_base = base;
  vs_top = vs_base + narg;
  
  if (fun == OBJNULL)
    FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   (*fun->cf.cf_self)(); */
  /*   break; */
    
  default:
    funcall(fun);
    
  }
  CHECK_AVMA;

}

object
simple_lispcall(object *funp, int narg) {

  DEBUG_AVMA
    object fun = *funp;
  object *sup = vs_top;
  
  vs_base = funp + 1;
  vs_top = vs_base + narg;
  
  if (fun == OBJNULL)
    FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   MMcall(fun); */
  /*   break; */
    
  default:
    funcall(fun);
  }
  vs_top = sup;
  CHECK_AVMA;
  return(vs_base[0]);
  
}

object
simple_symlispcall(object sym, object *base, int narg) {

  DEBUG_AVMA
    object fun = symbol_function(sym);
  object *sup = vs_top;
  
  vs_base = base;
  vs_top = vs_base + narg;
  
  if (fun == OBJNULL)
    FEerror("Undefined function.", 0);
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   MMcall(fun); */
  /*   break; */
    
  default:
    funcall(fun);
    
  }
  vs_top = sup;
  CHECK_AVMA;
  return(vs_base[0]);
}

void
super_funcall(object fun) {

  if (type_of(fun) == t_symbol) {
    if (fun->s.s_sfdef != NOT_SPECIAL || fun->s.s_mflag)
      FEinvalid_function(fun);
    if (fun->s.s_gfdef == OBJNULL)
      FEundefined_function(fun);
    fun = fun->s.s_gfdef;
  }
  funcall(fun);
}

void
super_funcall_no_event(object fun) {
#ifdef DEBUGGING_AVMA
  funcall_no_event(fun); return;
#endif 
 TOP:
  switch (type_of(fun)) {
  /* case t_cfun: */
  /*   (*fun->cf.cf_self)();return;break; */
  case t_function:
    quick_call_function(fun); return;break;
  case t_symbol:
    if (fun->s.s_sfdef != NOT_SPECIAL || fun->s.s_mflag)
      FEinvalid_function(fun);
    if (fun->s.s_gfdef == OBJNULL)
      FEundefined_function(fun);
    fun = fun->s.s_gfdef;
    goto TOP;
  }
  funcall_no_event(fun);
}

object
Ievaln(object form,object *vals) { 

  object *base=vs_top;

  eval(form);
  return unwind_vals(vals,base);

}
  
void
eval(object form)
{ 
        object temporary;
        DEBUG_AVMA
	object fun, x;
	object *top;
	object *base;

	cs_check(form);

EVAL:

	vs_check;

	if (siVevalhook->s.s_dbind != Cnil && eval1 == 0)
	{
		bds_ptr old_bds_top = bds_top;
		object hookfun = symbol_value(siVevalhook);
		/*  check if siVevalhook is unbound  */

		bds_bind(siVevalhook, Cnil);
		vs_base = vs_top;
		vs_push(form);
		vs_push(list(3,lex_env[0],lex_env[1],lex_env[2]));
		super_funcall(hookfun);
		bds_unwind(old_bds_top);
		return;
	} else
		eval1 = 0;

	if (consp(form))
		goto APPLICATION;

	if (type_of(form) != t_symbol) {
		vs_base = vs_top;
		vs_push(form);
		return;
	}

	switch (form->s.s_stype) {
	case stp_constant:
		vs_base = vs_top;
		vs_push(form->s.s_dbind);
		return;

	case stp_special:
		if(form->s.s_dbind == OBJNULL)
			FEunbound_variable(form);
		vs_base = vs_top;
		vs_push(form->s.s_dbind);
		return;

	default:
		/*  x = lex_var_sch(form);  */
		for (x = lex_env[0];  consp(x);  x = x->c.c_cdr)
			if (x->c.c_car->c.c_car == form) {
				x = x->c.c_car->c.c_cdr;
				if (endp(x))
					break;
				vs_base = vs_top;
				vs_push(x->c.c_car);
				return;
			}
		if(form->s.s_dbind == OBJNULL)
			FEunbound_variable(form);
		vs_base = vs_top;
		vs_push(form->s.s_dbind);
		return;
	}

APPLICATION:
	/* Hook for possibly stopping at forms in the break point
	   list.  Also for stepping.  We only want to check
	   one form each time round, so we do *breakpoints*
	   */
	if (sSAbreak_pointsA->s.s_dbind != Cnil)
	  { if (sSAbreak_stepA->s.s_dbind == Cnil ||
		ifuncall2(sSAbreak_stepA->s.s_dbind,form,
			  list(3,lex_env[0],lex_env[1],lex_env[2])) == Cnil)
	      {object* bpts = sSAbreak_pointsA->s.s_dbind->v.v_self;
		int i = VLEN(sSAbreak_pointsA->s.s_dbind);
	       while (--i >= 0)
		 { if((*bpts)->c.c_car == form)
		     {ifuncall2(sSAbreak_pointsA->s.s_gfdef,form,
				list(3,lex_env[0],lex_env[1],lex_env[2]));

		      break;}
		   bpts++;}
	     }}
	
	fun = MMcar(form);
	if (type_of(fun) != t_symbol)
		goto LAMBDA;
	if (fun->s.s_sfdef != NOT_SPECIAL) {
		ihs_check;
		ihs_push(form);
		ihs_top->ihs_base = lex_env;
		((void (*)())fun->s.s_sfdef)(MMcdr(form));
		CHECK_AVMA;
		ihs_pop();
		return;
	}
	/*  x = lex_fd_sch(fun);  */
	for (x = lex_env[1];  consp(x);  x = x->c.c_cdr)
		if (x->c.c_car->c.c_car == fun) {
			x = x->c.c_car;
			if (MMcadr(x) == sSmacro) {
				x = MMcaddr(x);
				goto EVAL_MACRO;
			}
			x = MMcaddr(x);
			goto EVAL_ARGS;
		}

	if ((x = fun->s.s_gfdef) == OBJNULL)
		FEundefined_function(fun);

	if (fun->s.s_mflag) {
	EVAL_MACRO:
		top = vs_top;
		form=Imacro_expand1(x, form);
		vs_top = top;
		vs_push(form);
		goto EVAL;
	}

	  
	
EVAL_ARGS:
	vs_push(x);
	ihs_check;
	ihs_push(form);
	ihs_top->ihs_base = lex_env;
	form = form->c.c_cdr;
	base = vs_top;
	top = vs_top;
	while(!endp(form)) {
		eval(MMcar(form));
		top[0] = vs_base[0];
		vs_top = ++top;
		form = MMcdr(form);
	}
	vs_base = base;
	if (siVapplyhook->s.s_dbind != Cnil) {
		call_applyhook(fun);
		return;
	}
	ihs_top->ihs_function = x;
	ihs_top->ihs_base = vs_base;
	/* if (type_of(x) == t_cfun)  */
	/*   (*(x)->cf.cf_self)(); */
	/* else */
	  funcall_no_event(x);
	CHECK_AVMA;
	ihs_pop();
	return;

LAMBDA:
	if (consp(fun) && MMcar(fun) == sLlambda) {
		temporary = make_cons(lex_env[2], fun->c.c_cdr);
		temporary = make_cons(lex_env[1], temporary);
		temporary = make_cons(lex_env[0], temporary);
		x = make_cons(sSlambda_closure, temporary);
		vs_push(x);
		goto EVAL_ARGS;
	}
	if (consp(fun) && (MMcar(fun) == sSlambda_closure || MMcar(fun) == sSlambda_block || MMcar(fun) == sSlambda_block_closure)) {
		vs_push(x=fun);
		goto EVAL_ARGS;
	}
	FEinvalid_function(fun);
}	

static void
call_applyhook(object fun)
{
	object ah;

	ah = symbol_value(siVapplyhook);
	stack_list();
	vs_push(vs_base[0]);
	vs_base[0] = fun;
	vs_push(list(3,lex_env[0],lex_env[1],lex_env[2]));
	super_funcall(ah);
}

object
coerce_funcall_object_to_function(object fun) {

  switch (type_of(fun)) {
  case t_function: break;
  case t_symbol:
    if (fun->s.s_mflag || fun->s.s_sfdef!=NOT_SPECIAL ||
	(fun=fun->s.s_gfdef)==OBJNULL)
    UNDEFINED_FUNCTION(fun);
    break;
  default:
    TYPE_ERROR(fun,list(3,sLor,sLsymbol,sLfunction));
    break;
  }

  return fun;

}


static object
funcall_apply(object fun,fixnum nargs,va_list ap) {

  object res,*vals=(object *)fcall.valp;

  fun=coerce_funcall_object_to_function(fun);

  res=funcall_ap(fun,nargs,ap);

  if (type_of(fun)==t_function && !fun->fun.fun_neval && !fun->fun.fun_vv && vals)
    vs_top=vals;

  return res;

}

DEFUNM("FUNCALL",object,fLfuncall,LISP,1,MAX_ARGS,NONE,OO,OO,OO,OO,(object fun,...),"") { 

  va_list ap;
  object res;

  va_start(ap,fun);
  
  res=funcall_apply(fun,(abs(VFUN_NARGS)-1)*(VFUN_NARGS/abs(VFUN_NARGS)),ap);
  va_end(ap);

  return res;

}

DEFUNM("APPLY",object,fLapply,LISP,1,MAX_ARGS,NONE,OO,OO,OO,OO,(object fun,...),"") {	
  
  va_list ap;
  object res;

  va_start(ap,fun);
  res=funcall_apply(fun,1-VFUN_NARGS,ap);
  va_end(ap);

  return res;

}

object
apply_format_function(object x,object y,object z,object a,object b,object c) {
  return FFN(fLapply)(x,y,z,a,b,c);
}


DEFUNM("EVAL",object,fLeval,LISP,1,1,NONE,OO,OO,OO,OO,(object x0),"") {

  object *lex=lex_env,*base=vs_top;
  object *vals=(object *)fcall.valp;
  
  lex_new();
  eval(x0);
  lex_env=lex;
  
  return unwind_vals(vals,base);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fLeval(object x) {
  RETURN1(FFN(fLeval)(x));
}
#endif

/* DEFUN("EVAL-SRC",object,fSeval_src,SI,0,63,NONE,OO,OO,OO,OO,(object first,...),"") { */

/*   object fun=fcall.fun,f,*base=vs_top,*vals=(object *)fcall.valp; */
/*   struct cons *p,*p1,*pp,*q,*qq; */
/*   fixnum j,narg=VFUN_NARGS; */
/*   va_list ap; */

/*   f=fun->fun.fun_plist->c.c_cdr->c.c_cdr->c.c_car; */
/*   princ(f->c.c_car,Cnil); */
/*   if (f->c.c_car==sLlambda_block) {printf(" ");princ(f->c.c_cdr->c.c_car,Cnil);} */
/*   printf("\n"); */
/*   flush_stream(symbol_value(sLAstandard_outputA)); */
/*   j=abs(narg)+1; */
/*   if (narg < 0) j--; */
/*   p=alloca((j+1)*sizeof(*p)); */
/*   p1=p=(void *)p+((unsigned)p%sizeof(*p)); */
/*   p->c_car=f; */
/*   va_start(ap,first); */
/*   for (;j--;first=NULL) { */
/*     object x=(j || (narg < 0)) ? (first ? first : va_arg(ap,object)) : Cnil; */
/*     if (j) { */
/*       pp=p++; */
/*       pp->c_cdr=(void *)p; */
/*       p->c_car=x;/\* MMcons(sLquote,MMcons(x,Cnil)); *\/ */
/*     } else p->c_cdr=x; */
/*   } */
/*   va_end(ap); */
/*   for (j=0,p=p1;p!=(void *)Cnil;j++,p=(void *)p->c_cdr); */
/*   q=alloca((2*j+1)*sizeof(*q)); */
/*   q=(void *)q+((unsigned)q%sizeof(*q)); */
/*   for (p=(void *)p1->c_cdr;p!=(void *)Cnil;p=(void *)p->c_cdr) { */
/*     object x=p->c_car; */
/*     p->c_car=(void *)(qq=q); */
/*     qq->c_car=sLquote; */
/*     qq->c_cdr=(void *)++q; */
/*     qq=q++; */
/*     qq->c_car=x; */
/*     qq->c_cdr=Cnil; */
/*   } */
/*   eval((void *)p1); */
/*   return unwind_vals(vals,base); */

/* } */

/* DEFUNM("EVAL-CFUN",object,fSeval_cfun,SI,0,63,NONE,OO,OO,OO,OO,(object first,...),"") { */

/*   object fun=fcall.fun,*base=vs_top,*vals=(object *)fcall.valp; */
/*   void (*f)(); */
/*   fixnum i,j,narg=VFUN_NARGS; */
/*   va_list ap; */

/*   f=(void *)fix(fun->fun.fun_plist->c.c_cdr->c.c_cdr->c.c_car); */
/*   j=abs(narg)+((narg < 0) ? 0 : 1); */
/*   va_start(ap,first); */
/*   vs_base=vs_top; */
/*   for (i=1;j--;) { */
/*     object x=(j || (narg < 0)) ? (i ? (i=0,first) : va_arg(ap,object)) : Cnil; */
/*     if (j)  */
/*       vs_push(x); */
/*     else for (;x!=Cnil;x=x->c.c_cdr) vs_push(x->c.c_car); */
/*   } */
/*   va_end(ap); */
/*   f(); */
/*   return unwind_vals(vals,base); */

/* } */

DEFUNM("EVAL-SRC",object,fSeval_src,SI,0,63,NONE,OO,OO,OO,OO,(object first,...),"") {

  object fun=fcall.fun,f,*base=vs_top,*vals=(object *)fcall.valp;
  fixnum i,j,narg=VFUN_NARGS;
  va_list ap;

  f=fun->fun.fun_plist->c.c_cdr->c.c_cdr->c.c_car;
  j=labs(narg)+((narg < 0) ? 0 : 1);
  va_start(ap,first);
  vs_base=vs_top;
  for (i=1;j--;) {
    object x=(j || (narg < 0)) ? (i ? (i=0,first) : va_arg(ap,object)) : Cnil;
    if (j) 
      vs_push(x);
    else for (;x!=Cnil;x=x->c.c_cdr) vs_push(x->c.c_car);
  }
  va_end(ap);
  if (type_of(f)==t_fixnum) ((void (*)())(fix(f)))(); else funcall(f);
  return unwind_vals(vals,base);

}

void *feval_src=(void *)FFN(fSeval_src);

DEFUN("FSET-IN",object,fSfset_in,SI,3,3,NONE,OO,OO,OO,OO,(object sym,object src,object name),"") {
  
  object x;

  x=fSinit_function(list(6,Cnil,Cnil,src,Cnil,make_fixnum(0),name),(void *)FFN(fSeval_src),Cnil,Cnil,-1,0,(((1<<6)-1)<<6)|(((1<<5)-1)<<12)|(1<<17));
  x->fun.fun_env=src_env;
  if (sym!=Cnil) fSfset(sym,x);
  RETURN1(x);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSfset_in(object sym,object src,object name) {
  RETURN1(FFN(fSfset_in)(sym,src,name));
}
#endif

LFD(siLevalhook)(void)
{
	object env;
	bds_ptr old_bds_top = bds_top;
	object *lex = lex_env;
	int n = vs_top - vs_base;

	lex_env = vs_top;
	if (n < 3)
		too_few_arguments();
	else if (n == 3) {
		*(struct nil3 *)vs_top = three_nils;
		vs_top += 3;
	} else if (n == 4) {
		env = vs_base[3];
		vs_push(car(env));
		env = cdr(env);
		vs_push(car(env));
		env = cdr(env);
		vs_push(car(env));
	} else
		too_many_arguments();
	bds_bind(siVevalhook, vs_base[1]);
	bds_bind(siVapplyhook, vs_base[2]);
	eval1 = 1;
	eval(vs_base[0]);
	lex_env = lex;
	bds_unwind(old_bds_top);
}

LFD(siLapplyhook)(void)
{

	object env;
	bds_ptr old_bds_top = bds_top;
	object *lex = lex_env;
	int n = vs_top - vs_base;
	object l, *z;

	lex_env = vs_top;
	if (n < 4)
		too_few_arguments();
	else if (n == 4) {
		*(struct nil3 *)vs_top = three_nils;
		vs_top += 3;
	} else if (n == 5) {
		env = vs_base[4];
		vs_push(car(env));
		env = cdr(env);
		vs_push(car(env));
		env = cdr(env);
		vs_push(car(env));
	} else
		too_many_arguments();
	bds_bind(siVevalhook, vs_base[2]);
	bds_bind(siVapplyhook, vs_base[3]);
	z = vs_top;
	for (l = vs_base[1];  !endp(l);  l = l->c.c_cdr)
		vs_push(l->c.c_car);
	l = vs_base[0];
	vs_base = z;
	super_funcall(l);
	lex_env = lex;
	bds_unwind(old_bds_top);
}

DEFUN("CONSTANTP",object,fLconstantp,LISP,1,2,NONE,OO,OO,OO,OO,(object x0,...),"") {

  enum type tp=type_of(x0);

  RETURN1((tp==t_cons && x0->c.c_car!=sLquote)||
	  (tp==t_symbol && x0->s.s_stype!=stp_constant) ? Cnil : Ct);

}

object
ieval(object x) {

  object *old_vs_base;
  object *old_vs_top;
  
  old_vs_base = vs_base;
  old_vs_top = vs_top;
  eval(x);
  x = vs_base[0];
  vs_base = old_vs_base;
  vs_top = old_vs_top;
  return(x);

}

object
ifuncall1(object fun, object arg1) {

  object *old_vs_base;
  object *old_vs_top;
  object x;
  
  old_vs_base = vs_base;
  old_vs_top = vs_top;
  vs_base = vs_top;
  vs_push(arg1);
  super_funcall(fun);
  x = vs_base[0];
  vs_top = old_vs_top;
  vs_base = old_vs_base;
  return(x);

}

object
ifuncall2(object fun, object arg1, object arg2) {

  object *old_vs_base;
  object *old_vs_top;
  object x;
  
  old_vs_base = vs_base;
  old_vs_top = vs_top;
  vs_base = vs_top;
  vs_push(arg1);
  vs_push(arg2);
  super_funcall(fun);
  x = vs_base[0];
  vs_top = old_vs_top;
  vs_base = old_vs_base;
  return(x);

}

object
ifuncall3(object fun, object arg1, object arg2, object arg3) {

  object *old_vs_base;
  object *old_vs_top;
  object x;
  
  old_vs_base = vs_base;
  old_vs_top = vs_top;
  vs_base = vs_top;
  vs_push(arg1);
  vs_push(arg2);
  vs_push(arg3);
  super_funcall(fun);
  x = vs_base[0];
  vs_top = old_vs_top;
  vs_base = old_vs_base;
  return(x);

}

object
ifuncall4(object fun, object arg1, object arg2, object arg3,object arg4) {

  object *old_vs_base;
  object *old_vs_top;
  object x;

  old_vs_base = vs_base;
  old_vs_top = vs_top;
  vs_base = vs_top;
  vs_push(arg1);
  vs_push(arg2);
  vs_push(arg3);
  vs_push(arg4);
  super_funcall(fun);
  x = vs_base[0];
  vs_top = old_vs_top;
  vs_base = old_vs_base;
  return(x);

}

void
funcall_with_catcher(object fname, object fun) {

  int n = vs_top - vs_base;
  if (n > MAX_ARGS+1) 
    FEerror("Call argument linit exceeded",0);
  frs_push(FRS_CATCH, make_cons(fname, make_fixnum(n)));
  if (nlj_active)
    nlj_active = FALSE;
  else
    funcall(fun);
  frs_pop();

}

static object 
fcalln_general(object first,va_list ap) {
  int i=fcall.argd,n= SFUN_NARGS(i);
  object *old_vs_top=vs_top;
  object x;
  enum ftype typ,restype=SFUN_RETURN_TYPE(i);

  vs_top =  vs_base = old_vs_top;
  SFUN_START_ARG_TYPES(i);
  if (i==0) {
    int jj=0;
    while (n-- > 0) {
      typ= SFUN_NEXT_TYPE(i);
      x =
	(typ==f_object ?	(jj ? va_arg(ap,object) : first) :
	 (typ==f_fixnum ? make_fixnum((jj ? va_arg(ap,fixnum) : (fixnum)first)) :
	   (object) (FEerror("bad type",0),Cnil)));
      *(vs_top++) = x;
      jj++;
    }
  } else {
    object *base=vs_top;
    *(base++)=first;
    n--;
    while (n-- > 0) 
      *(base++) = va_arg(ap,object);
    vs_top=base;
  }
  funcall(fcall.fun);
  x= vs_base[0];
  vs_top=old_vs_top;
  /* vs_base=old_vs_base; */
  return (restype== f_object ? x :
	  (restype== f_fixnum ? (object) (fix(x)) :
	   (object) (FEerror("bad type",0),Cnil)));
}

object 
fcalln1(object first,...) {  

  va_list ap;
  object fun=fcall.fun;
  enum type tp;
  DEBUG_AVMA
  va_start(ap,first);
  tp=fun==OBJNULL ? -1 : type_of(fun);
  /* if(tp==t_cfun) */
  /*    {object *base=vs_top,*old_base=base; */
  /*     int i=fcall.argd; */
  /*     vs_base=base; */
  /*     if (i) { */
  /* 	*(base++)=first; */
  /* 	i--; */
  /*     } */
  /*     switch(i){ */
  /*     case 10: *(base++)=va_arg(ap,object); */
  /*     case 9: *(base++)=va_arg(ap,object); */
  /*     case 8: *(base++)=va_arg(ap,object); */
  /*     case 7: *(base++)=va_arg(ap,object); */
  /*     case 6: *(base++)=va_arg(ap,object); */
  /*     case 5: *(base++)=va_arg(ap,object); */
  /*     case 4: *(base++)=va_arg(ap,object); */
  /*     case 3: *(base++)=va_arg(ap,object); */
  /*     case 2: *(base++)=va_arg(ap,object); */
  /*     case 1: *(base++)=va_arg(ap,object); */
  /*     case 0: break; */
  /*     default: */
  /* 	FEerror("bad args",0); */
  /*     }  vs_top=base; */
  /*     base=old_base; */
  /*     (*fcall.fun->cf.cf_self)(); */
  /*     vs_top=base; */
  /*     CHECK_AVMA; */
  /*     return(vs_base[0]); */
  /*   } */
   return(fcalln_general(first,ap));
  va_end(ap);
 }

/* call a cfun eg funcall_cfun(Lmake_hash_table,2,sKtest,sLeq) */
/*  typedef void (*funcvoid)(); */

object
funcall_cfun(funcvoid fn,int n,...)
{object *old_top = vs_top;
 object *old_base= vs_base;
 object result;
 va_list ap;
 DEBUG_AVMA
 vs_base=vs_top;
 va_start(ap,n);
 while(n-->0) vs_push(va_arg(ap,object));
 va_end(ap);
 (*fn)();
 if(vs_top>vs_base) result=vs_base[0];
 else result=Cnil;
 vs_top=old_top;
 vs_base=old_base;
 CHECK_AVMA;
 return result;}
 
DEF_ORDINARY("LAMBDA-BLOCK-EXPANDED",sSlambda_block_expanded,SI,"");
DEFVAR("*BREAK-POINTS*",sSAbreak_pointsA,SI,Cnil,"");
DEFVAR("*BREAK-STEP*",sSAbreak_stepA,SI,Cnil,"");

void
gcl_init_eval(void)
{




        make_constant("CALL-ARGUMENTS-LIMIT", make_fixnum(MAX_ARGS+1));


	siVevalhook = make_si_special("*EVALHOOK*", Cnil);
	siVapplyhook = make_si_special("*APPLYHOOK*", Cnil);


	three_nils.nil3_self[0] = Cnil;
	three_nils.nil3_self[1] = Cnil;
	three_nils.nil3_self[2] = Cnil;

	make_si_function("EVALHOOK", siLevalhook);
	make_si_function("APPLYHOOK", siLapplyhook);

}
