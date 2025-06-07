/*  Copyright William Schelter. All rights reserved.
    Copyright 2024 Camm Maguire
Fast linking method for kcl by W. Schelter University of Texas
   Note there are also changes to 
 cmpcall.lsp and cmptop.lsp */


#include "include.h"
#include "sfun_argd.h"
#include "page.h"

#if 0
#define DO_FUNLINK_DEBUG
#endif

#ifdef DO_FUNLINK_DEBUG
void print_lisp_string ( char *boilerplate, object s )
{
  if ( s && VLEN(s) && s->st.st_self ) {
    int last = VLEN(s);
        int i;
        fprintf ( stderr, "%s", boilerplate ); 
        for (i = 0;  (i < last) && (i < 30);  i++) {
            fputc ( s->st.st_self[i], stderr );
        }
        fputc ( '\n', stderr );
    } else {
        fprintf ( stderr, "Object %x not a string or empty\n", s );
    }
}
#endif

static int
clean_link_array(object *,object *);

object sScdefn;
typedef object (*object_func)();

static int     
vpush_extend(void *,object);

int Rset = 0;

/* for pushing item into an array, where item is an address if array-type = t
or a fixnum if array-type = fixnum */

#define SET_ITEM(ar,ind,val) (*((object *)(&((ar)->ust.ust_self[ind]))))= val
static int     
vpush_extend(void *item, object ar) { 
  register int ind;
#ifdef DO_FUNLINK_DEBUG
  fprintf ( stderr, "vpush_extend: item %x, ar %x\n", item, ar );
#endif 
  ind = ar->ust.ust_fillp;  
 AGAIN:
  if (ind < ar->ust.ust_dim) {
    SET_ITEM(ar,ind,item);
    ind += sizeof(void *); 
    return(ar->v.v_fillp = ind);
  } else { 
    int newdim= CEI((2 + (int) (1.3 * ind)),PTR_ALIGN);
    unsigned char *newself;
    newself = (void *)alloc_relblock(newdim);
    bcopy(ar->ust.ust_self,newself,ind);
    ar->ust.ust_dim=newdim;
    ar->ust.ust_self=newself;
    goto AGAIN;
  }
#ifdef DO_FUNLINK_DEBUG_1
  fprintf ( stderr, "vpush_extend: item %x, ar %x END\n", item, ar );
#endif 
}


/* if we unlink a bunch of functions, this will mean there are some
   holes in the link array, and we should probably go through it and
   push them back  */
static int number_unlinked=0;

static void
delete_link(void *address, object link_ar) {
  object *ar,*ar_end,*p;
#ifdef DO_FUNLINK_DEBUG
  fprintf ( stderr, "delete_link: address %x, link_ar %x START\n", address, link_ar );
#endif 
  p=0;
  ar = link_ar->v.v_self;
  ar_end = (object *)&(link_ar->ust.ust_self[link_ar->v.v_fillp]);
  while (ar < ar_end) {
    if (*ar && *((void **)*ar)==address) {
      p = (object *) *ar;
      *ar=0;
      *p = *(ar+1);
      number_unlinked++;
    }
    ar=ar+2;
  }
  if (number_unlinked > 40)
    link_ar->v.v_fillp=clean_link_array(link_ar->v.v_self,ar_end);
#ifdef DO_FUNLINK_DEBUG
  fprintf ( stderr, "delete_link: address %x, link_ar %x END\n", address, link_ar );
#endif 
}


DEFUN("USE-FAST-LINKS",object,fSuse_fast_links,SI,1,2,NONE,OO,OO,OO,OO,(object flag,...),
      "Usage: (use-fast-links {nil,t} &optional fun) turns on or off \
the fast linking depending on FLAG, so that things will either go \
faster, or turns it off so that stack information is kept.  If SYMBOL \
is supplied and FLAG is nil, then this function is deleted from the fast links") {
 object sym;
 va_list ap;
 object *p,*ar,*ar_end;
 object link_ar;
 object fun=Cnil,l=Cnil,f=OBJNULL;
 fixnum n=INIT_NARGS(1);

 va_start(ap,flag);
 sym=NEXT_ARG(n,ap,l,f,Cnil);
 
 if (sSAlink_arrayA==0)
   RETURN1(Cnil);

 link_ar=sSAlink_arrayA->s.s_dbind;
 if (link_ar==Cnil && flag==Cnil) 
   RETURN1(Cnil);

 check_type_array(&link_ar);
 if (!stringp(link_ar))
   FEerror("*LINK-ARRAY* must be a string",0);

 ar=link_ar->v.v_self;
 ar_end=(object *)&(link_ar->ust.ust_self[link_ar->v.v_fillp]);

 if (sym==Cnil) {

   if (flag==Cnil) {
     Rset=0;
     while (ar<ar_end) {
      /* set the link variables back to initial state */
       p=(object *)*ar;
       if (p) *p=*++ar; else ar++;
       ar++;
     }
     link_ar->v.v_fillp=0;
   } else
     Rset=1;

 } else {

   if ((type_of(sym)==t_symbol))
     fun=sym->s.s_gfdef;
   else 
     fun=sym;
     /* FEerror("Second arg: ~a must be symbol or closure",0,sym); */

   if (Rset) {

     if (fun==OBJNULL) 
       RETURN1(Cnil);

     switch(type_of(fun)) {
     /* case t_cfun: */
     /*   if (flag==Cnil) */
     /* 	 delete_link(fun->cf.cf_self,link_ar); */
     /*   break; */
     case t_function:	
       if (flag==Cnil)
	 delete_link(fun->fun.fun_self,link_ar);
       break;
     default: 
       break;	
     }

   }

 }

 RETURN1(Cnil);

}

object
fSuse_fast_links_2(object flag,object res) {
  VFUN_NARGS=2;
  return FFN(fSuse_fast_links)(flag,res);
}

object
clear_compiler_properties(object sym, object code) { 
  object tem;
  extern object sSclear_compiler_properties;  
  
  if (sSclear_compiler_properties && sSclear_compiler_properties->s.s_gfdef!=OBJNULL)
    if ((sSAinhibit_macro_specialA && sSAinhibit_macro_specialA->s.s_dbind != Cnil) ||
	sym->s.s_sfdef == NOT_SPECIAL)
      (void)ifuncall2(sSclear_compiler_properties,sym,code);
  tem = getf(sym->s.s_plist,sStraced,Cnil);

  VFUN_NARGS=2;
  FFN(fSuse_fast_links)(Cnil,sym);
  return tem!=Cnil ? tem : sym;
  
}


static int
clean_link_array(object *ar, object *ar_end) {
  int i=0;
  object *orig;
#ifdef DO_FUNLINK_DEBUG
  fprintf ( stderr, "clean_link_array: ar %x, ar_end %x START\n", ar, ar_end );
#endif 
  orig=ar;
  number_unlinked=0;
  while(ar<ar_end)
    if(*ar) {
      orig[i++]= *ar++ ;
      orig[i++]= *ar++;
    }
    else ar=ar+2;       
#ifdef DO_FUNLINK_DEBUG
  fprintf ( stderr, "clean_link_array: ar %x, ar_end %x END\n", ar, ar_end );
#endif 
  return(i*sizeof(object *));
}

#include "apply_n.h"    

DEFVAR("*FAST-LINK-WARNINGS*",sSAfast_link_warningsA,SI,Cnil,"");

#include "pbits.h"

#ifdef WORDS_BIGENDIAN
typedef struct {ufixnum pad:LM(21),nf:1,pu:1,va:1,vv:1,nv:5,xa:6,ma:6;} fw;
#else
typedef struct {ufixnum ma:6,xa:6,nv:5,vv:1,va:1,pu:1,nf:1,pad:LM(21);} fw;
#endif

typedef union {
  ufixnum i;
  fw f;
} fu;

object
call_proc_new(object sym,ufixnum clp,ufixnum vld,void **link,ufixnum argd,object first,va_list ll) {

  object fun;
  enum type tp;
  ufixnum margs,nargs,fas,do_link,varg,pushed=0,nfargs;
  fixnum vald;
  object *tmp,*x/* ,*p */;
  int i;
  fu u;

  if (type_of(sym)==t_symbol) {
    if ((fun=sym->s.s_gfdef)==OBJNULL)
      FEundefined_function(sym);
  } else
    fun=sym;
  check_type_function(&fun);
  tp=type_of(fun);

  u.i=vld;
  
  /* p=0; */
  if (u.f.pu) {
    u.f.ma=vs_top-vs_base;
    u.f.va=u.f.nv=u.f.vv=0;
    /* p=vs_base; */
    pushed=1;
  }
  
  margs=u.f.ma;
  varg=u.f.va;
  nargs=u.f.va ? abs(VFUN_NARGS) : margs;
  nfargs=u.f.va && VFUN_NARGS<0 ? nargs-1 : nargs;
  vald=!u.f.vv ? -(fixnum)u.f.nv : u.f.nv;
  
  x=tmp=(u.f.pu && !fun->fun.fun_argd && VFUN_NARGS>=fun->fun.fun_minarg) ? 
    vs_base : ZALLOCA(nargs*sizeof(object));
  
  if (tmp!=vs_base) {
    if (u.f.pu) 
      memcpy(tmp,vs_base,nargs*sizeof(*tmp));
    else for (i=0;i<nargs;i++)
	   *x++=(i || u.f.nf) ? va_arg(ll,object) : first;
  }

  /*FIXME: Problem here relying on VFUN_NARGS or fcall.fun or FUN_VALP might foil sharing these links in different contexts*/
  /*links currently shared by rt at clp apnarg, so VFUN_NARGS<0 is safe*/
  /*abs(VFUN_NARGS) above is dangerous*/

  fas=do_link=Rset;
  switch(tp) {
  case t_function:
    {
      fixnum neval=fun->fun.fun_neval/* ,nvald=vald */;
      neval=fun->fun.fun_vv ? neval : -neval;
      /* nvald=FUN_VALP ? vald : 0; */
      if (pushed)
	fas=0;
      else if (margs!=fun->fun.fun_minarg) /*margs < fun->fun.fun_minarg*/
      	fas=0;
      else if (u.f.va &&(nfargs<fun->fun.fun_minarg || nfargs>fun->fun.fun_maxarg))/*u.f.va -> varg, xxx*/
	fas=0;
      else if (u.f.va && VFUN_NARGS<0 && fun->fun.fun_minarg==fun->fun.fun_maxarg)/*runtime apply #arg checking omitted in reg fns*/
	fas=0;
      /* else if (u.f.va && VFUN_NARGS<0 && */
      /* 	       (nargs-1<fun->fun.fun_minarg || nargs-1>fun->fun.fun_maxarg))/\*u.f.va -> varg, xxx*\/ */
      /* 	fas=0; */
      /* FIXME: below should be removed?*/
      else if (!varg && (fun->fun.fun_minarg!=fun->fun.fun_maxarg))/*and maybe inverse for error checking*/
	fas=0;
      else if (vald!=neval && (vald<=0 || !neval || neval>vald))/*margs funvalp aggregate across file*//*FIXME check valp*/
	fas=0;
      else if (fun->fun.fun_env!=def_env && !clp)
	fas=0;
      else if (fun->fun.fun_argd!=argd)
	fas=0;
    }
    break;
  default:
    fas=0;
  }

  if (fas!=Rset && sSAfast_link_warningsA->s.s_dbind==Ct) {
    if (tp==t_function) {
      fprintf(stderr,"Warning: arg/val mismatch in call to %-.*s (%p) prevents fast linking:\n %ld %ld/%ld %d(%d)  %ld %d  %ld %d  %ld, recompile caller\n",
	      (int)(type_of(sym)==t_symbol ? VLEN(sym->s.s_name) : 0),sym->s.s_name->st.st_self,sym,
	      argd,(long)fun->fun.fun_argd,
	      vald,fun->fun.fun_neval,fun->fun.fun_vv,
	      margs,fun->fun.fun_minarg,nargs,fun->fun.fun_maxarg,pushed);
      fflush(stderr);
    }

    /* if (tp==t_cfun) */
    /*   fprintf(stderr,"Warning: arg/val mismatch in call to %-.*s (%p) prevents fast linking:is cfun\n", */
    /* 	      (int)(type_of(sym)==t_symbol ? sym->s.s_fillp : 0),sym->s.s_self,sym); */

  }

  if (sSAprofilingA->s.s_dbind!=Cnil)
    sSin_call->s.s_gfdef->fun.fun_self(sym);

  if (fas) {

    if (do_link && link) {
      (void) vpush_extend(link,sSAlink_arrayA->s.s_dbind);
      (void) vpush_extend(*link,sSAlink_arrayA->s.s_dbind);
      *link = (void *)fun->fun.fun_self;
    }

    if (sSAprofilingA->s.s_dbind!=Cnil)
      sSout_call->s.s_gfdef->fun.fun_self(fSgettimeofday());
    
    return(c_apply_n_fun(fun,x-tmp,tmp));

  } else {
    
    object res;
    register object *base,*old_top;
    enum ftype result_type;
    fixnum larg=0,i;

#define POP_BITS(x_,y_) ({ufixnum _t=x_&((1<<y_)-1);x_>>=y_;_t;})

    result_type=POP_BITS(argd,2);

    if (vald || u.f.vv) larg=(fixnum)fcall.valp;

    if (!pushed) {
      
      object y;
      
      vs_base=vs_top; /*???*/

      for (i=0;i<nargs;i++) {
	
	enum ftype typ;

	y=tmp[i];
	
	switch((typ=POP_BITS(argd,2))) {
	case f_fixnum:
	  y=make_fixnum((fixnum)y);
	  break;
	default:
	  break;
	}
	
	vs_push(y);
	
      }

      if (u.f.va && VFUN_NARGS<0)
	for (y=*--vs_top;y!=Cnil;y=y->c.c_cdr)
	  vs_push(y->c.c_car);
      
      vs_check;

    }

    base=vs_base;
    old_top=vs_top;
    funcall(fun);
    
    res=vs_base[0];
    if (larg) {
      object *tmp=vs_base+1,*tl=(void *)larg,*tle=tl+labs(vald);/*FIXME avoid if pushed*/
      for (;tl<tle && tmp<vs_top;)
	*tl++=*tmp++;
      if (vald<0)
	for (;tl<tle;)
	  *tl++=Cnil;
      vs_top=tmp>vs_top ? tl-1 : tl;
    } else
      vs_top=base;

    for (;--old_top>=vs_top && vs_top>=vs_org;) *old_top=Cnil;
    
    switch(result_type) {
    case f_fixnum:
      res=(object)fix(res);
      break;
    default:
      break;
    }
    
    if (sSAprofilingA->s.s_dbind!=Cnil)
      sSout_call->s.s_gfdef->fun.fun_self(fSgettimeofday());

    return res;
    
  }

}
object
call_proc_new_nval(object sym,ufixnum clp,ufixnum vld,void **link,ufixnum argd,object first,...) {
  object x;
  va_list b;
  va_start(b,first);
  x=call_proc_new(sym,clp,vld,link,argd,first,b);
  va_end(b);
  return x;
}

object
call_proc_cs1(object fun,...) {
  register object res;
  ufixnum vald;
  va_list ap;
  va_start(ap,fun);
  vald=((31<<12)|(1<<17)|(1<<18)|(1<<20));
  res=call_proc_new(fun,1,vald,0,0,0,ap);
  va_end(ap);
  return res;
}


object
call_proc_cs2(object first,...) {
  register object res;
  ufixnum vald;
  va_list ap;
  object fun=fcall.fun;
  va_start(ap,first);
  vald=((31<<12)|(1<<17)|(1<<18));
  res=call_proc_new(fun,1,vald,0,0,first,ap);
  va_end(ap);
  return res;
}


object
ifuncall(object sym,int n,...)
{ va_list ap;
  int i;
  object *old_vs_base;
  object *old_vs_top;
  object x;
  old_vs_base = vs_base;
  old_vs_top = vs_top;
  vs_base = old_vs_top;
  vs_top=old_vs_top+n;
  vs_check;
  va_start(ap,n);
  for(i=0;i<n;i++)
    old_vs_top[i]= va_arg(ap,object);
  va_end(ap);
  /* if (type_of(sym->s.s_gfdef)==t_cfun) */
  /*   (*(sym->s.s_gfdef)->cf.cf_self)(); */
  /* else   */super_funcall(sym);
  x = vs_base[0];
  vs_top = old_vs_top;
  vs_base = old_vs_base;
  return(x);
}

/* go from beg+1 below limit setting entries equal to 0 until you
   come to FRESH 0's . */

#define FRESH 40

int
clear_stack(object *beg, object *limit) {
  int i=0;
  while (++beg < limit) {
    if (*beg==0) i++;
    if (i > FRESH) return 0;
    *beg=0;
  } 
  return 0;
}

DEFUN("SET-MV",object,fSset_mv,SI,2,2,NONE,OI,OO,OO,OO,(ufixnum i, object val),"") {
  if (i >= (sizeof(MVloc)/sizeof(object)))
    FEerror("Bad mv index",0);
  return(MVloc[i]=val);

}


DEFUN("MV-REF",object,fSmv_ref,SI,1,1,NONE,OI,OO,OO,OO,(ufixnum i),"") {
  object x;
  if (i >= (sizeof(MVloc)/sizeof(object)))
    FEerror("Bad mv index",0);
  x = MVloc[i];
  return x;
}


#include "xdrfuns.c"

DEF_ORDINARY("CDEFN",sScdefn,SI,"");
DEFVAR("*LINK-ARRAY*",sSAlink_arrayA,SI,Cnil,"");

void
gcl_init_links(void) {	
  gcl_init_xdrfuns();
}

