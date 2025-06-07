/* Copyright (C) 2024 Camm Maguire */
#include "include.h"
#include "funlink.h"
#include "page.h"

DEFUN("SET-FUNCTION-ENVIRONMENT",object,fSset_function_environment,SI,2,2,NONE,OO,OO,OO,OO, \
	  (object f,object env),"") { 

  ufixnum n;
  object x,*p;

  if (type_of(f)!=t_function)
    TYPE_ERROR(f,sLcompiled_function);
  
  for (n=0,x=env;x!=Cnil;x=x->c.c_cdr,n++);

  if (n++) {

    {
      BEGIN_NO_INTERRUPT; 
      p=(object *)alloc_relblock(n*sizeof(object));
      END_NO_INTERRUPT;
    }

    *p++=(object)n;
    f->fun.fun_env=p;
    
    for (;env!=Cnil;env=env->c.c_cdr)
      *p++=env;

  }

  RETURN1(f);

}

#define PADDR(i) ((void *)(((fixnum *)sSPinit->s.s_dbind->a.a_self)[Mfix(i)]))

#define POP_BITS(x_,y_) ({ufixnum _t=x_&((1<<y_)-1);x_>>=y_;_t;})
static object
make_fun(void *addr,object data,object call,object env,ufixnum argd,ufixnum sizes) {
  
  object x;

  x=alloc_object(t_function);
  x->fun.fun_self=addr;
  x->fun.fun_data=data;
  x->fun.fun_argd=argd;
  x->fun.fun_plist=call;/*FIXME*/
  x->fun.fun_minarg=POP_BITS(sizes,6);
  x->fun.fun_maxarg=POP_BITS(sizes,6);
  x->fun.fun_neval =POP_BITS(sizes,5);
  x->fun.fun_vv    =POP_BITS(sizes,1);
  x->fun.fun_env=def_env;

  if ((void *)x->fun.fun_self==feval_src)
    x->d.tt=2;

  FFN(fSset_function_environment)(x,env);

  return x;

}

#define GET_DATA(d_,a_) ((d_)!=Cnil ? (d_) : ((a_) && (a_)->s.s_dbind!=OBJNULL && type_of((a_)->s.s_dbind)==t_cfdata ? (a_)->s.s_dbind : 0))

DEFUN("FUNCTION-ENVIRONMENT",object,fSfunction_environment,SI,1,1,NONE,OO,OO,OO,OO,(object f),"") {

  RETURN1(f->fun.fun_env[0]);

}

DEFUN("INIT-FUNCTION",object,fSinit_function,SI,7,7,NONE,OO,OO,OI,II, \
	  (object sc,object addr,object data,object env,\
	   fixnum key,fixnum argd,fixnum sizes),\
	  "Store a compiled function on SYMBOL whose body is in the VV array at \
           INDEX, and whose argd descriptor is ARGD.  If more arguments IND1, IND2,.. \
           are supplied these are indices in the VV array for the environment of this \
           closure.") { 

  object s,d,m,i,fun,c;
  fixnum z;

  m=sSPmemory;
  m=m ? m->s.s_dbind : m;
  m=m && m!=OBJNULL && type_of(m)==t_cfdata ? m : 0;
  d=data!=Cnil ? data : m;
  i=sSPinit;
  i=i ? i->s.s_dbind : i;
  if (is_text_addr(addr)||(get_pageinfo(addr)&&!is_bigger_fixnum(addr))||!i||i==OBJNULL)
    s=addr;
  else {
    massert(type_of(addr)==t_fixnum);
    s=i->v.v_self[fix(addr)];
  }
  z=type_of(sc)==t_cons && sc->c.c_car==sSmacro; /*FIXME limited no. of args.*/
  sc=z ? sc->c.c_cdr : sc;
  sc=type_of(sc)==t_function ? sc->fun.fun_plist : sc;
  c=type_of(sc)==t_symbol ? Cnil : sc;

  fun=make_fun(s,d,c,env,argd,sizes);

  if (i && key>=0 && d)
    set_key_struct((void *)i->v.v_self[key],d);

  if (sc!=c) {
    fSfset(sc,fun);
    if (z) sc->s.s_mflag=TRUE;
  }
  
  return fun;

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSinit_function(object x,object y,object z,object w,fixnum a,fixnum b,fixnum c) {
  return FFN(fSinit_function)(x,y,z,w,a,b,c);
}
#endif

DEFUN("SET-KEY-STRUCT",object,fSset_key_struct,SI,1,1,NONE,OO,OO,OO,OO,(object key_struct_ind),
      "Called inside the loader.  The keystruct is set up in the file with \
   indexes rather than the actual entries.  We change these indices to \
   the objects")
{ set_key_struct(PADDR(key_struct_ind),sSPmemory->s.s_dbind);
  return Cnil;
}
     
#define mcollect(top_,next_,val_) ({object _x=MMcons(val_,Cnil);\
                                   if (top_==Cnil) top_=next_=_x; \
                                   else next_=next_->c.c_cdr=_x;})


static void
put_fn_procls(object sym,fixnum argd,fixnum oneval,object def,object rdef) {

  unsigned int atypes=F_TYPES(argd) >> F_TYPE_WIDTH;
  unsigned int minargs=F_MIN_ARGS(argd);
  unsigned int maxargs=F_MAX_ARGS(argd);
  unsigned int rettype=F_RESULT_TYPE(argd);
  unsigned int i;
  object ta=Cnil,na=Cnil;

  for (i=0;i<minargs;i++,atypes >>=F_TYPE_WIDTH) 
    switch(maxargs!=minargs ? F_object : atypes & MASK_RANGE(0,F_TYPE_WIDTH)) {
    case F_object:
      mcollect(ta,na,def);
      break;
    case F_int:
      mcollect(ta,na,sLfixnum);
      break;
    case F_shortfloat:
      mcollect(ta,na,sLshort_float);
      break;
    case F_double_ptr:
      mcollect(ta,na,sLlong_float);
      break;
    default:
      FEerror("Bad sfn declaration",0);
      break;
    }
  if (maxargs!=minargs)
    mcollect(ta,na,sLA);
  putprop(sym,ta,sSproclaimed_arg_types);
  ta=na=Cnil;
  if (oneval) 
    switch(rettype) {
    case F_object:
      ta=rdef;
      break;
    case F_int:
      ta=sLfixnum;
      break;
    case F_shortfloat:
      ta=sLshort_float;
      break;
    case F_double_ptr:
      ta=sLlong_float;
      break;
    default:
      FEerror("Bad sfn declaration",0);
      break;
    }
  else
/*     ta=MMcons(sLA,Cnil); */
    ta=sLA;
  putprop(sym,ta,sSproclaimed_return_type);
  /* if (oneval) */
    putprop(sym,Ct,sSproclaimed_function);

}  


void
SI_makefun(char *strg, object (*fn) (/* ??? */), unsigned int argd) { 

  object sym = make_si_ordinary(strg);
  ufixnum at=F_TYPES(argd)>>F_TYPE_WIDTH;
  ufixnum ma=F_MIN_ARGS(argd);
  ufixnum xa=F_MAX_ARGS(argd);
  ufixnum rt=F_RESULT_TYPE(argd);

  fSinit_function(sym,(void *)fn,Cnil,Cnil,-1,
		  rt | (at<<F_TYPE_WIDTH),ma|(xa<<6)|(0<<12)|(0<<17)|((xa>ma? 1 : 0)<<18));
/*   fSfset(sym, fSmakefun(sym,fn,argd)); */
  put_fn_procls(sym,argd,1,Ct,Ct);

}

void
LISP_makefun(char *strg, object (*fn) (/* ??? */), unsigned int argd) { 

  object  sym = make_ordinary(strg);
  ufixnum at=F_TYPES(argd)>>F_TYPE_WIDTH;
  ufixnum ma=F_MIN_ARGS(argd);
  ufixnum xa=F_MAX_ARGS(argd);
  ufixnum rt=F_RESULT_TYPE(argd);
  
  fSinit_function(sym,(void *)fn,Cnil,Cnil,-1,
		  rt | (at<<F_TYPE_WIDTH),ma|(xa<<6)|(0<<12)|(0<<17)|((xa>ma? 1 : 0)<<18));
  put_fn_procls(sym,argd,1,Ct,Ct);

}

void
GMP_makefunb(char *strg, object (*fn)(),unsigned int argd,object p) { 

  object sym = make_gmp_ordinary(strg);
  ufixnum at=F_TYPES(argd)>>F_TYPE_WIDTH;
  ufixnum ma=F_MIN_ARGS(argd);
  ufixnum xa=F_MAX_ARGS(argd);
  ufixnum rt=F_RESULT_TYPE(argd);

  fSinit_function(sym,(void *)fn,Cnil,Cnil,-1,
		  rt | (at<<F_TYPE_WIDTH),ma|(xa<<6)|((type_of(p)==t_symbol ? 0 : 1)<<12)|
		  (0<<17)|((xa>ma? 1 : 0)<<18));
  put_fn_procls(sym,argd,1,sLinteger,p);

}

void
SI_makefunm(char *strg, object (*fn) (/* ??? */), unsigned int argd) { 

  object sym = make_si_ordinary(strg);
  ufixnum at=F_TYPES(argd)>>F_TYPE_WIDTH;
  ufixnum ma=F_MIN_ARGS(argd);
  ufixnum xa=F_MAX_ARGS(argd);
  ufixnum rt=F_RESULT_TYPE(argd);
  
  fSinit_function(sym,(void *)fn,Cnil,Cnil,-1,
		  rt | (at<<F_TYPE_WIDTH),ma|(xa<<6)|(31<<12)|(1<<17)|((xa>ma? 1 : 0)<<18));
  
  /*  fSfset(sym, fSmakefun(sym,fn,argd)); */
  put_fn_procls(sym,argd,0,Ct,Ct);
  
}

void
LISP_makefunm(char *strg, object (*fn) (/* ??? */), unsigned int argd) { 

  object sym = make_ordinary(strg);
  ufixnum at=F_TYPES(argd)>>F_TYPE_WIDTH;
  ufixnum ma=F_MIN_ARGS(argd);
  ufixnum xa=F_MAX_ARGS(argd);
  ufixnum rt=F_RESULT_TYPE(argd);
  
  fSinit_function(sym,(void *)fn,Cnil,Cnil,-1,
		  rt | (at<<F_TYPE_WIDTH),ma|(xa<<6)|(31<<12)|(1<<17)|((xa>ma? 1 : 0)<<18));
  
  /*  fSfset(sym, fSmakefun(sym,fn,argd)); */
  put_fn_procls(sym,argd,0,Ct,Ct);
  
}
      
DEFUN("INVOKE",object,fSinvoke,SI,1,ARG_LIMIT,NONE,OO,OO,OO,OO,(object x),
      "Invoke a C function whose body is at INDEX in the VV array")
{ int (*fn)();
  fn = (void *) PADDR(x);
  (*fn)();
  return Cnil;
}
  
