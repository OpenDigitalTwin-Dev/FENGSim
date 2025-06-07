/*
 Copyright (C) 1994  W. Schelter
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

*/

#ifdef HAVE_XDR

#ifdef DARWIN
#undef __LP64__ /*Apple header declaration bug workaround for xdr_long*/
#endif

#ifdef AIX3
#include <sys/select.h>
#endif
#ifdef __CYGWIN__
#include <rpc/xdr.h>
#else  /* __CYGWIN__ */
#include <rpc/rpc.h>
#endif /* __CYGWIN__ */

extern aet_type_struct aet_types[];

DEFUN("XDR-OPEN",object,fSxdr_open,SI,1,1,NONE,OO,OO,OO,OO,(object f),"") {

  XDR *xdrs;
  object ar= alloc_string(sizeof(XDR));
  array_allocself(ar,1,OBJNULL);
  xdrs= (XDR *) ar->a.a_self;
  if (f->sm.sm_fp == 0) FEerror("stream not ok for xdr io",0);
  xdrstdio_create(xdrs, f->sm.sm_fp,
		  (f->sm.sm_mode == smm_input ?  XDR_DECODE :
		   f->sm.sm_mode == smm_output ?  XDR_ENCODE :
		   (FEerror("stream not input or output",0),XDR_ENCODE)))
		   ;
  return ar;
}

DEFUN("XDR-WRITE",object,fSxdr_write,SI,2,2,NONE,OO,OO,OO,OO,(object str,object elt),"") {

  XDR *xdrp= (XDR *) str->ust.ust_self;
  xdrproc_t e;

  switch (type_of(elt)) {
  case t_fixnum:
    {
      fixnum e=fix(elt);
      if(xdr_long(xdrp,(long *)&e)) goto error;
    }
    break;
  case t_longfloat:
    if(xdr_double(xdrp,&lf(elt))) goto error;
    break;
  case t_shortfloat:
    if(xdr_float(xdrp,&sf(elt))) goto error;
    break;
  case t_simple_vector:
  case t_vector:
    
    switch(elt->v.v_elttype) {
    case aet_lf:
      e=(xdrproc_t)xdr_double;
      break;
    case aet_sf:
      e=(xdrproc_t)xdr_float;
      break;
    case aet_fix:
      e=(xdrproc_t)xdr_long;
      break;
    case aet_short:
      e=(xdrproc_t)xdr_short;
      break;
    default:
      FEerror("unsupported xdr size",0);
      goto error;
      break;
    }
    {
      u_int tmp=VLEN(elt);
      if (tmp!=VLEN(elt))
	goto error;
      if(xdr_array(xdrp,(void *)&elt->v.v_self,
		    &tmp,
		    elt->v.v_dim,
		    aet_types[elt->v.v_elttype].size,
		    e))
	goto error;
    }
    break;
  default:
    FEerror("unsupported xdr ~a",1,elt);
    break;
  }
  return elt;
 error:
  FEerror("bad xdr write",0);
  return elt;
}

DEFUN("XDR-READ",object,fSxdr_read,SI,2,2,NONE,OO,OO,OO,OO,(object str,object elt),"") {

  XDR *xdrp= (XDR *) str->ust.ust_self;
  xdrproc_t e;

  switch (type_of(elt)) { 
  case t_fixnum:
    {fixnum l;
      if(xdr_long(xdrp,(long *)&l)) goto error;
    return make_fixnum(l);}
    break;
  case t_longfloat:
    {double x;
    if(xdr_double(xdrp,&x)) goto error;
    return make_longfloat(x);}
  case t_shortfloat:
    {float x;
    if(xdr_float(xdrp,&x)) goto error;
    return make_shortfloat(x);}
  case t_simple_vector:
  case t_vector:
    switch(elt->v.v_elttype) {
    case aet_lf:
      e=(xdrproc_t)xdr_double;
      break;
    case aet_sf:
      e=(xdrproc_t)xdr_float;
      break;
    case aet_fix:
      e=(xdrproc_t)xdr_long;
      break;
    case aet_short:
      e=(xdrproc_t)xdr_short;
      break;
    default:
      FEerror("unsupported xdr size",0);
      goto error;
      break;
    }

    {
      u_int tmp=VLEN(elt);
      if (tmp!=VLEN(elt))
	goto error;
      if(xdr_array(xdrp,(void *)&elt->v.v_self,
		    &tmp,
		    elt->v.v_dim,
		    aet_types[elt->v.v_elttype].size,
		    e))
	goto error;
    }
    return elt;
    break;
  default:
    FEerror("unsupported xdr ~a",1,elt);
    return elt;
    break;
  }
 error:
  FEerror("bad xdr read",0);
  return elt;
}
static void
gcl_init_xdrfuns()
{/*  make_si_sfun("XDR-WRITE",siGxdr_write, */
/* 	       ARGTYPE2(f_object,f_object)|RESTYPE(f_object)); */

/*   make_si_sfun("XDR-READ",siGxdr_read, */
/* 	       ARGTYPE2(f_object,f_object)|RESTYPE(f_object)); */
/*   make_si_sfun("XDR-OPEN",siGxdr_open, */
/* 	       ARGTYPE1(f_object)|RESTYPE(f_object)); */
  
}
#else
static void gcl_init_xdrfuns(void) {;}
#endif     
