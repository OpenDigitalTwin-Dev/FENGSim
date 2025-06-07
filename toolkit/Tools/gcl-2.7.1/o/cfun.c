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
	cfun.c
*/

#define _GNU_SOURCE 1
#include "include.h"
#include <dlfcn.h>
#include "page.h"

#define dcheck_vs do{if (vs_base < vs_org || vs_top < vs_org) error("bad vs");} while (0)
#define dcheck_type(a,b) check_type(a,b) ; dcheck_vs 

#define PADDR(i) ((void *)(long)(sSPinit->s.s_dbind->v.v_self[fix(i)]))
object sSPinit,sSPmemory;


object
make_cfun(void (*self)(), object name, object data, char *start, int size) {

   if (data && type_of(data)==t_cfdata) { 
     data->cfd.cfd_start=start;  
     data->cfd.cfd_size=size; 
   } else if (size) FEerror("Bad call to make_cfun",0); 

   return fSinit_function(list(6,Cnil,Cnil,make_fixnum((fixnum)self),Cnil,make_fixnum(0),name),
			  feval_src,data,Cnil,-1,0,(((1<<6)-1)<<6)|(((1<<5)-1)<<12)|(1<<17)); 

}

DEFUN("CFDL",object,fScfdl,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {

  struct typemanager *tm=tm_of(t_cfdata);
  int j;
  object x;
  void *p;
  struct pageinfo *v;

  for (v=cell_list_head;v;v=v->next) {
    if (tm!=tm_of(v->type))
      continue;
    for (p=pagetochar(page(v)),j=tm->tm_nppage;j>0;--j,p+=tm->tm_size) {
      x=(object)p;
      if (type_of(x)!=t_cfdata || is_marked_or_free(x))
	continue;
      for (x=x->cfd.cfd_dlist;x!=Cnil;x=x->c.c_cdr) {
	fixnum j=fix(x->c.c_car->c.c_cdr),k=fix(x->c.c_car->c.c_car->s.s_dbind);
	if (*(fixnum *)j!=k)
	  *(fixnum *)j=k;
      }
    }
  }
  RETURN1(Cnil);
}
    
DEFUN("DLSYM",object,fSdlsym,SI,2,2,NONE,OI,OO,OO,OO,(fixnum h,object name),"") {

  void *ad;

  dlerror();
  name=coerce_to_string(name);
  massert(snprintf(FN1,sizeof(FN1),"%-.*s",VLEN(name),name->st.st_self)>0);
#ifndef __CYGWIN__
  ad=dlsym(h ? (void *)h : RTLD_DEFAULT,FN1);
  ad=ad ? ad : dlsym(RTLD_DEFAULT,FN1);
  ad=is_text_addr(ad) ? dlsym(RTLD_NEXT,FN1) : ad;
#else
  ad=0;
  if (h) ad=dlsym((void *)h,FN1);
  {
    static void *n,*u,*c;
    n=n ? n : dlopen("ntdll.dll",RTLD_LAZY|RTLD_GLOBAL);
    u=u ? u : dlopen("ucrtbase.dll",RTLD_LAZY|RTLD_GLOBAL);
    c=c ? c : dlopen("cygwin1.dll",RTLD_LAZY|RTLD_GLOBAL);
    ad=ad ? ad : dlsym(n,FN1);
    ad=ad ? ad : dlsym(u,FN1);
    ad=ad ? ad : dlsym(c,FN1);
    ad=ad ? ad : dlsym(RTLD_DEFAULT,FN1);
  }
#endif
  if (!ad) {
    char *er=dlerror();
    FEerror("dlsym lookup failure on ~s: ~s",2,name,make_simple_string(er ? er : ""));
  }
  RETURN1(make_fixnum((fixnum)ad));

}

DEFUN("DLADDR",object,fSdladdr,SI,2,2,NONE,OI,OO,OO,OO,(fixnum ad,object n),"") {

  Dl_info info;
  unsigned long u;
  const char *c;
  char *d,*de;

  dlerror();
  dladdr((void *)ad,&info);
  if (dlerror())
    FEerror("dladdr lookup failure on ~s",1,make_fixnum(ad));
  u=(unsigned long)info.dli_fbase;
  c=info.dli_fname;
  if (n!=Cnil) {
    d=alloca(strlen(c)+1);
    strcpy(d,c);
    for (de=d+strlen(d);de>d && de[-1]!='/';de--)
      if (*de=='.') *de=0;
    c=de;
  }
  if (u>=(ufixnum)data_start && u<(unsigned long)core_end)
    c="";
  
  RETURN1(make_simple_string(c));

}

DEFUN("DLOPEN",object,fSdlopen,SI,1,1,NONE,OO,OO,OO,OO,(object name),"") {

  char *err;
  void *v;

  dlerror();
  name=coerce_to_string(name);
  if (!strncmp("libc.so",name->st.st_self,VLEN(name)) || !strncmp("libm.so",name->st.st_self,VLEN(name)))
    v=dlopen(0,RTLD_LAZY|RTLD_GLOBAL);
  else {
    massert(snprintf(FN1,sizeof(FN1),"%-.*s",VLEN(name),name->st.st_self)>0);
    v=dlopen(FN1,RTLD_LAZY|RTLD_GLOBAL);
  }
  if ((err=dlerror()))
    FEerror("dlopen failure on ~s: ~s",2,name,make_simple_string(err));

  update_real_maxpage();
  
  RETURN1(make_fixnum((fixnum)v));

}

DEFUN("DLADDR-SET",object,fSdladdr_set,SI,2,2,NONE,OI,IO,OO,OO,(fixnum adp,fixnum ad),"") {

  *(void **)adp=(void *)ad;
  RETURN1(Cnil);

}

DEFUN("DLLIST-PUSH",object,fSdllist_push,SI,3,3,NONE,OO,OI,OO,OO,(object cfd,object sym,fixnum adp),"") {

  cfd->cfd.cfd_dlist=MMcons(MMcons(sym,make_fixnum(adp)),cfd->cfd.cfd_dlist);
  RETURN1(Cnil);

}


/* 		sym->s.s_sfdef = NOT_SPECIAL; */

  

object
make_function_internal(char *s, void (*f)())
{
	object x;
	vs_mark;

	x = make_ordinary(s);
	if (x->s.s_gfdef!=OBJNULL) {
	  printf("Skipping redefinition of %-.*s\n",(int)VLEN(x->s.s_name),x->s.s_name->st.st_self);
	  return(x);
	}
	vs_push(x);
	x->s.s_gfdef = make_cfun(f, x, Cnil, NULL, 0);
	x->s.s_mflag = FALSE;
	vs_reset;
	return(x);
}


object
make_si_function_internal(char *s, void (*f)())
{
	object x;
	vs_mark;

	x = make_si_ordinary(s);
	if (x->s.s_gfdef!=OBJNULL) {
	  printf("Skipping redefinition of %-.*s\n",(int)VLEN(x->s.s_name),x->s.s_name->st.st_self);
	  return(x);
	}
	vs_push(x);
	x->s.s_gfdef = make_cfun(f, x, Cnil, NULL, 0);
	x->s.s_mflag = FALSE;
	vs_reset;
	return(x);
}




object
make_special_form_internal(char *s,void *f)
{
	object x;
	x = make_ordinary(s);
	x->s.s_sfdef = (fixnum)f;
	return(x);
}

object
make_si_special_form_internal(char *s,void *f)
{
	object x;
	x = make_si_ordinary(s);
	x->s.s_sfdef = (fixnum)f;
	return(x);
}

object
make_macro_internal(char *s, void (*f)())
{
	object x;
	x = make_ordinary(s);
	x->s.s_gfdef = make_cfun(f, x, Cnil, NULL, 0);
	x->s.s_mflag=TRUE;
	return(x);
}

DEFUN("COMPILED-FUNCTION-NAME",object,fScompiled_function_name,SI
   ,1,1,NONE,OO,OO,OO,OO,(object fun),"")

{
	/* 1 args */
	switch(type_of(fun)) {
	case t_function:
	  fun=Cnil;
	  break;
	/* case t_cfun: */
	/*   fun = fun->cf.cf_name; */
	/*   break; */
	default:
	  FEerror("~S is not a compiled-function.", 1, fun);
	}RETURN1(fun);
}

void
gcl_init_cfun(void) {
}
