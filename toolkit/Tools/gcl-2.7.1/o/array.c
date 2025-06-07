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

You should have received a copy of the GNU Library General Public License 
along with GCL; see the file COPYING.  If not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

*/

#include "include.h"
#include "page.h"

static object Iname_t=Ct;
static char zero[4*SIZEOF_LONG];/*FIXME*/

aet_type_struct aet_types[] = {
  {" ",&sLcharacter,sizeof(char)},
  {zero,&sLbit,sizeof(char)},
  {zero,&sSnon_negative_char,sizeof(char)},
  {zero,&sSunsigned_char,sizeof(char)},
  {zero,&sSsigned_char,sizeof(char)},
  {zero,&sSnon_negative_short,sizeof(short)},
  {zero,&sSunsigned_short,sizeof(short)},
  {zero,&sSsigned_short,sizeof(short)},
  {zero,&sLshort_float,sizeof(float)},
#if SIZEOF_LONG != SIZEOF_INT
  {zero,&sSnon_negative_int,sizeof(int)},
  {zero,&sSunsigned_int,sizeof(int)},
  {zero,&sSsigned_int,sizeof(int)},
#endif
  {zero,&sLlong_float,sizeof(double)},
  {Cnil,&Iname_t,sizeof(object)},
  {zero,&sSnon_negative_fixnum,sizeof(fixnum)},
  {zero,&sLfixnum,sizeof(fixnum)}
#if SIZEOF_LONG == SIZEOF_INT
  ,{zero,&sSnon_negative_int,sizeof(int)},
  {zero,&sSunsigned_int,sizeof(int)},
  {zero,&sSsigned_int,sizeof(int)}
#endif
};

static void
displace(object, object, int);

static enum aelttype
Iarray_element_type(object);


/*  #define ARRAY_DIMENSION_LIMIT MOST_POSITIVE_FIXNUM */

DEFCONST("ARRAY-RANK-LIMIT",sLarray_rank_limit,LISP,make_fixnum(ARRAY_RANK_LIMIT),"");
DEFCONST("ARRAY-DIMENSION-LIMIT", sLarray_dimension_limit,LISP,make_fixnum(ARRAY_DIMENSION_LIMIT),"");
DEFCONST("ARRAY-TOTAL-SIZE-LIMIT", sLarray_total_size_limit,LISP,make_fixnum(ARRAY_DIMENSION_LIMIT),"");

DEF_ORDINARY("BIT",sLbit,LISP,"");
DEF_ORDINARY("SBIT",sLsbit,LISP,"");

#define ARRAY_BODY_PTR(ar,n) \
  (void *)(ar->ust.ust_self + aet_types[Iarray_element_type(ar)].size*n)

#define N_FIXNUM_ARGS 6

/*FIXME*/
DEFUN("AREF",object,fLaref,LISP,1,MAX_ARGS,ONE_VAL,OO,II,II,II,(object x,...),"") {

  va_list ap;
  fixnum k,n=INIT_NARGS(1);
  object l=Cnil,f=OBJNULL;
  ufixnum i1,m,rank=type_of(x)==t_array ? x->a.a_rank : 1;

  va_start(ap,x);
  for (m=i1=0;(k=(fixnum)NEXT_ARG(n,ap,l,f,(object)-1))!=-1 && m<rank;m++) {
    if (m>=N_FIXNUM_ARGS) {
      object x=(object)k;
      check_type(x,t_fixnum);
      k=Mfix(x);
    }
    if (k>=(rank>1 ? x->a.a_dims[m] : x->v.v_dim)||k<0)
      FEerror("Index ~a to array is out of bounds",1,make_fixnum(m));
    i1*=rank>1 ? x->a.a_dims[m] : 1;
    i1+=k;
  }
  va_end(ap);
  if (m!=rank || k!=-1)
    FEerror("Array rank/index number mismatch on ~a",1,x);
    
  RETURN1(fLrow_major_aref(x,i1));

}

static void
fScheck_bounds_bounds(object x, fixnum i)
{
    if ( ( i >= x->a.a_dim ) || ( i < 0 ) ) {
        FEerror("Array index ~a out of bounds for ~a", 2,  make_fixnum(i),x);
    }
}

DEFUN("SVREF",object,fLsvref,LISP,2,2,ONE_VAL,OO,IO,OO,OO,(object x,ufixnum i),"For array X and index I it returns (aref x i) ") {

  if (TS_MEMBER(type_of(x),TS(t_vector)|TS(t_simple_vector)) && (enum aelttype)x->v.v_elttype == aet_object) {/*FIXME*/
     if (x->v.v_dim > i)
       RETURN1(x->v.v_self[i]);
     else
       TYPE_ERROR(make_fixnum(i),list(3,sLinteger,make_fixnum(0),make_fixnum(x->v.v_dim)));
 } else
   TYPE_ERROR(x,sLsimple_vector);
 return(Cnil);

}
    
DEFUN("ROW-MAJOR-AREF",object,fLrow_major_aref,LISP,2,2,NONE,OO,IO,OO,OO,(object x,fixnum i),
      "For array X and index I it returns (aref x i) as if x were \
1 dimensional, even though its rank may be bigger than 1") {

  switch (type_of(x)) {
  case t_array:
  case t_simple_vector:
  case t_simple_bitvector:
  case t_vector:
  case t_bitvector:
    fScheck_bounds_bounds(x, i);
    switch (x->v.v_elttype) {
    case aet_object:
      return x->v.v_self[i];
    case aet_ch:
      return code_char(x->st.st_self[i]);
    case aet_bit:
      i += BV_OFFSET(x);
      return make_fixnum(BITREF(x, i));
    case aet_fix:
    case aet_nnfix:
      return make_fixnum(((fixnum *)x->a.a_self)[i]);
    case aet_sf:
      return make_shortfloat(((float *)x->a.a_self)[i]);
    case aet_lf:
      return make_longfloat(((double *)x->a.a_self)[i]);
    case aet_char:
    case aet_nnchar:
      return small_fixnum(x->st.st_self[i]);
    case aet_uchar:
      return small_fixnum(x->ust.ust_self[i]);
    case aet_short:
    case aet_nnshort:
      return make_fixnum(SHORT_GCL(x, i));
    case aet_ushort:
      return make_fixnum(USHORT_GCL(x, i));
    case aet_int:
    case aet_nnint:
      return make_fixnum(INT_GCL(x, i));
    case aet_uint:
      return make_fixnum(UINT_GCL(x, i));
    default:
      FEerror("unknown array type",0);
    }
  case t_simple_string:
  case t_string:
    fScheck_bounds_bounds(x, i);
    return code_char(x->st.st_self[i]);
  default:
    FEwrong_type_argument(sLarray,x);
    return(Cnil);
  }
}
#ifdef STATIC_FUNCTION_POINTERS
object
fLrow_major_aref(object x,fixnum i) {
  return FFN(fLrow_major_aref)(x,i);
}
#endif

object
aset1(object x,fixnum i,object val) {
  return fSaset1(x,i,val);
}

DEFUN("ASET1", object, fSaset1, SI, 3, 3, NONE, OO, IO, OO,OO,(object x, fixnum i,object val),"") {

  switch (type_of(x)) {
  case t_array:
  case t_simple_vector:
  case t_simple_bitvector:
  case t_vector:
  case t_bitvector:
    fScheck_bounds_bounds(x, i);
    switch (x->v.v_elttype) {
    case aet_object:
      x->v.v_self[i] = val;
      break;
    case aet_ch:
      ASSURE_TYPE(val,t_character);
      x->st.st_self[i] = char_code(val);
      break;
    case aet_bit:
      i +=  BV_OFFSET(x);
      ASSURE_TYPE(val,t_fixnum);
      switch (Mfix(val)) {
      case 0:
	CLEAR_BITREF(x,i);
	break;
      case 1:
	SET_BITREF(x,i);
	break;
      default:
	TYPE_ERROR(val,sLbit);
      }
      break;
    case aet_fix:
    case aet_nnfix:
      ASSURE_TYPE(val,t_fixnum);
      (((fixnum *)x->a.a_self)[i]) = Mfix(val);
      break;
    case aet_sf:
      ASSURE_TYPE(val,t_shortfloat);
      (((float *)x->a.a_self)[i]) = Msf(val);
      break;
    case aet_lf:
      ASSURE_TYPE(val,t_longfloat);
      (((double *)x->a.a_self)[i]) = Mlf(val);
      break;
    case aet_char:
    case aet_nnchar:
      ASSURE_TYPE(val,t_fixnum);
      x->st.st_self[i] = Mfix(val);
      break;
    case aet_uchar:
      ASSURE_TYPE(val,t_fixnum);
      (x->ust.ust_self[i])= Mfix(val);
      break;
    case aet_short:
    case aet_nnshort:
      ASSURE_TYPE(val,t_fixnum);
      SHORT_GCL(x, i) = Mfix(val);
      break;
    case aet_ushort:
      ASSURE_TYPE(val,t_fixnum);
      USHORT_GCL(x, i) = Mfix(val);
      break;
    case aet_int:
    case aet_nnint:
      ASSURE_TYPE(val,t_fixnum);
      INT_GCL(x, i) = Mfix(val);
      break;
    case aet_uint:
      ASSURE_TYPE(val,t_fixnum);
      UINT_GCL(x, i) = Mfix(val);
      break;
    default:
      FEerror("unknown array type",0);
    }
    break;
  case t_simple_string:
  case t_string:
    fScheck_bounds_bounds(x, i);
    ASSURE_TYPE(val,t_character);
    x->st.st_self[i] = char_code(val);
    break;
  default:
    FEwrong_type_argument(sLarray,x);
  }
  return val;
}
#ifdef STATIC_FUNCTION_POINTERS
object
fSaset1(object x, fixnum i,object val) {
  return FFN(fSaset1)(x,i,val);
}
#endif

DEFUN("ASET",object,fSaset,SI,2,ARG_LIMIT,NONE,OO,OO,OO,OO,(object y,object x,...),"") { 

  va_list ap;
  fixnum k,n=INIT_NARGS(2);
  ufixnum m,i1,rank=type_of(x)==t_array ? x->a.a_rank : 1;
  object z,l=Cnil,f=OBJNULL;

  va_start(ap,x);
  for (i1=m=0;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL && m<rank;m++) {
    check_type(z,t_fixnum);
    k=Mfix(z);
    if (k>=(rank>1 ? x->a.a_dims[m] : x->v.v_dim)||k<0)
      FEerror("Index ~a to array is out of bounds",1,make_fixnum(m));
    i1*=rank>1 ? x->a.a_dims[m] : 1;
    i1+=k;
  }
  va_end(ap);
  if (m!=rank || z!=OBJNULL)
    FEerror("Array rank/index number mismatch on ~a",1,x);

  RETURN1(fSaset1(x,i1,y));
   
}


DEFUN("SVSET",object,fSsvset,SI,3,3,NONE,OO,IO,OO,OO,(object x,fixnum i,object val),"") {
  if (!TS_MEMBER(type_of(x),TS(t_vector)|TS(t_simple_vector)) || DISPLACED_TO(x) != Cnil)/*FIXME*/
    TYPE_ERROR(x,sLsimple_vector);
  /*     Wrong_type_error("simple array",0); */
  if (i > x->v.v_dim)
    FEerror("out of bounds",0);
  return x->v.v_self[i] = val;
}

void
set_array_elttype(object x,enum aelttype tp) {
  x->a.a_elttype=tp;
  x->a.a_eltsize=fixnum_length(elt_size(tp));
  x->a.a_eltmode=elt_mode(tp);
}

fixnum
elt_size(fixnum elt_type) {
  switch (elt_type) {
  case aet_bit:         /*  bit  */
    return 0;
  case aet_ch:          /*  character  */
  case aet_nnchar:      /*  non-neg char */
  case aet_char:        /*  signed char */
  case aet_uchar:       /*  unsigned char */
    return sizeof(char);
  case aet_nnshort:     /*  non-neg short   */
  case aet_short:       /*  signed short */
  case aet_ushort:      /*  unsigned short   */
    return sizeof(short);
    break;
  case aet_nnint:       /*  non-neg int   */
  case aet_int:         /*  signed int */
  case aet_uint:        /*  unsigned int   */
    return sizeof(int);
    break;
  case aet_nnfix:       /*  non-neg fixnum  */
  case aet_fix:         /*  fixnum  */
  case aet_object:      /*  t  */
    return sizeof(fixnum);
  case aet_sf:          /*  short-float  */
    return sizeof(float);
  case aet_lf:          /*  plong-float  */
    return sizeof(double);
  default:
    FEerror("Bad elt type",0);
    return -1;
  }
}

fixnum
elt_mode(fixnum elt_type) {
  switch (elt_type) {
  case aet_bit:         /*  bit  */
  case aet_uchar:       /*  unsigned char */
  case aet_ushort:      /*  unsigned short   */
  case aet_uint:        /*  unsigned int   */
    return aem_unsigned;
  case aet_ch:          /*  character  */
    return aem_character;
  case aet_nnchar:      /*  non-neg char */
  case aet_char:        /*  signed char */
  case aet_nnshort:     /*  non-neg short   */
  case aet_short:       /*  signed short */
  case aet_nnint:       /*  non-neg int   */
  case aet_int:         /*  signed int */
  case aet_nnfix:       /*  non-neg fixnum  */  /*FIXME*/
  case aet_fix:         /*  fixnum  */
    return aem_signed;
  case aet_object:      /*  t  */
    return aem_t;
  case aet_sf:          /*  short-float  */
  case aet_lf:          /*  plong-float  */
    return aem_float;
  default:
    FEerror("Bad elt type",0);
    return -1;
  }
}

DEFUN("MAKE-VECTOR",object,fSmake_vector,SI,8,8,NONE,OO,IO,OO,IO,
	  (object etp,fixnum n,object adjp,object fp,object displaced_to,fixnum V9,object staticp,object initial_element),"") {

  object x;
  fixnum elt_type=type_of(etp)==t_symbol ? fix(fSget_aelttype(etp)) : fix(etp);
  fixnum fillp=fp==Cnil ? -1 : (fp==Ct ? n : Mfix(fp));

  BEGIN_NO_INTERRUPT;

  switch(elt_type) {
  case aet_ch:
    x = adjp==Cnil && fp==Cnil && displaced_to == Cnil ? alloc_simple_string(n) : alloc_string(n);
    break;
  case aet_bit:
    x = adjp==Cnil && fp==Cnil && displaced_to == Cnil ? alloc_simple_bitvector(n) : alloc_bitvector(n);
    break;
  case aet_object:
    x = adjp==Cnil && fp==Cnil && displaced_to == Cnil ? alloc_simple_vector(n) : alloc_vector(n,aet_object);
    break;
  default:
    x = alloc_vector(n,elt_type);
  }

  if (fillp<0)
    x->a.a_hasfillp=0;
  else if (fillp>n)
    FEerror("bad fillp",0);
  VFILLP_SET(x,fillp);
  
  if (displaced_to==Cnil)
    array_allocself(x,staticp!=Cnil,initial_element);
  else 
    displace(x,displaced_to,V9);
  
  END_NO_INTERRUPT;
  
  return x;

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSmake_vector(object etp,fixnum n,object adjp,object fp,object displaced_to,fixnum V9,object staticp,object initial_element) {
  return FFN(fSmake_vector)(etp,n,adjp,fp,displaced_to,V9,staticp,initial_element);
}
#endif


object
aelttype_list(void) {

  aet_type_struct *p,*pe;
  object f=Cnil,x,y=OBJNULL;

  for (p=aet_types,pe=p+aet_fix;p<=pe;p++) {
    x=MMcons(*p->namep,Cnil);
    y=y!=OBJNULL ? (y->c.c_cdr=x) : (f=x);
  }
  
  return f;

}
DEFCONST("+ARRAY-TYPES+",sSParray_typesP,SI,aelttype_list(),"");
  

DEFUN("GET-AELTTYPE",object,fSget_aelttype,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  int i;

  for (i=0 ; i <   aet_last ; i++)
    if (x == * aet_types[i].namep)
      return make_fixnum((enum aelttype) i);
  if (x == sLlong_float || x == sLsingle_float || x == sLdouble_float)
    return make_fixnum(aet_lf);
  if (x==sSnegative_char)
    return make_fixnum(aet_char);
  if (x==sSnegative_short)
    return make_fixnum(aet_short);
  if (x==sSnegative_int)
#if SIZEOF_LONG != SIZEOF_INT
    return make_fixnum(aet_int);
#else
    return make_fixnum(aet_fix);
#endif
  if (x==sSnegative_fixnum || x==sSsigned_fixnum)
    return make_fixnum(aet_fix);
  return make_fixnum(aet_object);
}
#ifdef STATIC_FUNCTION_POINTERS
object
fSget_aelttype(object x) {
  return FFN(fSget_aelttype)(x);
}
#endif

DEFUN("MAKE-ARRAY1",object,fSmake_array1,SI,7,7,NONE,OO,OO,OI,OO,
	  (object x0,object staticp,object initial_element,object displaced_to,fixnum displaced_index_offset,
	   object dimensions,object adjp),"") {   

  int rank = length(dimensions);
  fixnum elt_type=fix(fSget_aelttype(x0));
  if (rank >= ARRAY_RANK_LIMIT)
    FEerror("Array rank limit exceeded.",0);
  { object x,v;
    char *tmp_alloc;
    int dim =1,i; 
    BEGIN_NO_INTERRUPT;
    x = alloc_object(t_array);
    set_array_elttype(x,elt_type);
    x->a.a_self = 0;
    x->a.a_hasfillp = 0;
    x->a.a_rank = rank;
    x->a.a_dims = AR_ALLOC(alloc_relblock,rank,ufixnum);
    i = 0;
    v = dimensions;
    while (i < rank)
      { x->a.a_dims[i] = FIX_CHECK(Mcar(v));
	if (x->a.a_dims[i] < 0)
	  { FEerror("Dimension must be non negative",0);}
	if (dim && x->a.a_dims[i]>((1UL<<(sizeof(dim)*8-1))-1)/dim)
	  FEerror("Total dimension overflow on dimensions ~s",1,dimensions);
	dim *= x->a.a_dims[i++];
	v = Mcdr(v);}
    x->a.a_dim = dim;
    x->a.a_adjustable = TRUE;/* adjp!=Cnil; */
    SET_ADISP(x,Cnil);
    { if (displaced_to == Cnil)
	array_allocself(x,staticp!=Cnil,initial_element);
    else { displace(x,displaced_to,displaced_index_offset);}
      END_NO_INTERRUPT;
	return x;
      }
 }}
#ifdef STATIC_FUNCTION_POINTERS
object
fSmake_array1(object elt_type,object staticp,object initial_element,object displaced_to,
	      fixnum displaced_index_offset,object dimensions,object adjustable) {
  return FFN(fSmake_array1)(elt_type,staticp,initial_element,
			    displaced_to,displaced_index_offset,dimensions,adjustable);
}
#endif


/*
(proclaim '(ftype (function (object t  *)) array-displacement1))
(defun array-displacement1 ( array )
*/

/*  DEFUNO_NEW("ARRAY-DISPLACEMENT1",object,fSarray_displacement,SI,1,1, */
/*        NONE,OO,OO,OO,OO,void,siLarray_displacement,"") */
/*       (object array) { */

/*    object a; */
/*    int s,n; */

/*    BEGIN_NO_INTERRUPT; */
/*    if (type_of(array)!=t_array && type_of(array)!=t_vector) */
/*      FEerror("Argument is not an array",0); */
/*    a=array->a.a_displaced->c.c_car; */
/*    if (a==Cnil) { */
/*      END_NO_INTERRUPT; */
/*      return make_cons(Cnil,make_fixnum(0)); */
/*    } */
/*    s=aet_sizes[Iarray_element_type(a)]; */
/*    n=(void *)array->a.a_self-(void *)a->a.a_self; */
/*    if (n%s) */
/*      FEerror("Array is displaced by fractional elements",0); */
/*    END_NO_INTERRUPT; */
/*    return make_cons(a,make_fixnum(n/s)); */

/*  } */

static void
FFN(Larray_displacement)(void) {

  object array,a;
  int s,n;

  BEGIN_NO_INTERRUPT;

  n = vs_top - vs_base;
  if (n < 1)
    FEtoo_few_arguments(vs_base,vs_top);
  if (n > 1)
    FEtoo_many_arguments(vs_base,vs_top);
  array = vs_base[0];
  vs_base=vs_top;

/*   if (type_of(array)!=t_array && type_of(array)!=t_vector && */
/*       type_of(array)!=t_bitvector && type_of(array)!=t_string) */
/*     FEwrong_type_argument(sLarray,array); */
  IisArray(array);
  a=ADISP(array)->c.c_car;

  if (a==Cnil) {

    vs_push(Cnil);
    vs_push(make_fixnum(0));
    END_NO_INTERRUPT;

    return;

  }

  s=aet_types[Iarray_element_type(a)].size;
  n=(void *)array->a.a_self-(void *)a->a.a_self;
  if (Iarray_element_type(a)==aet_bit)
    n=n*CHAR_SIZE+BV_OFFSET(array)-BV_OFFSET(a);
  if (n%s)
    FEerror("Array is displaced by fractional elements",0);

  vs_push(a);
  vs_push(make_fixnum(n/s));
  END_NO_INTERRUPT;

  return;

}

/*
  For the X->a.a_displaced field, the CAR is an array which X
  's body is displaced to (ie body of X is part of Another array)
  and the (CDR) is the LIST of arrays whose bodies are displaced
  to X
 (setq a (make-array 2 :displaced-to (setq b (make-array 4 ))))
                ;{  A->displ = (B), B->displ=(nil A)}
(setq w (make-array 3))   ;; w->displaced= (nil y u) 
(setq y (make-array 2 :displaced-to  w))  ;; y->displaced=(w z z2)
(setq u (make-array 2 :displaced-to w))   ;; u->displaced = (w)
(setq z (make-array 2 :displaced-to y))   ;; z->displaced = (y)
(setq z2 (make-array 2 :displaced-to y))  ;; z2->displaced= (y)
*/

void
set_displaced_body_ptr(object from_array) {

  object displaced=ADISP(from_array)->c.c_car;

  if (displaced!=Cnil) {

    enum aelttype typ =Iarray_element_type(from_array);
    object dest_array=displaced->c.c_car;
    int offset=fix(Scdr(displaced));

    if (typ == aet_bit) {
      if (Iarray_element_type(dest_array)==aet_bit)
	offset += BV_OFFSET(dest_array);
      from_array->bv.bv_self = (void *)dest_array->bv.bv_self + offset/CHAR_SIZE;
      SET_BV_OFFSET(from_array,offset % CHAR_SIZE);
    } else
      from_array->a.a_self = ARRAY_BODY_PTR(dest_array,offset);

  }

}

static void
displace(object from_array, object dest_array, int offset)
{
  enum aelttype typ;
  IisArray(from_array);
  IisArray(dest_array);
  typ =Iarray_element_type(from_array);

  if (offset<0) FEerror("Negative offset",0);

  if (typ!=aet_bit) {
    void *v1;
    ufixnum n=0;
    v1=((void *)dest_array->a.a_self)+
      (Iarray_element_type(dest_array)!=aet_bit ? elt_size(dest_array->a.a_elttype)*offset :
       FLR((n=offset+BV_OFFSET(dest_array)),CHAR_SIZE)/CHAR_SIZE);
    if (((unsigned long)v1)%elt_size(from_array->a.a_elttype) || n%CHAR_SIZE)
      FEerror("Offset produces illegal array alignment.",0);
  }

#define BIT_SIZE(a_,b_) ((a_)*((b_) ? (b_)*CHAR_SIZE : 1))
  if (BIT_SIZE(from_array->a.a_dim,elt_size(from_array->a.a_elttype))>
      BIT_SIZE(((fixnum)dest_array->a.a_dim)-offset,elt_size(dest_array->a.a_elttype)))
    FEerror("Destination array too small",0);

  /* ensure that we have a cons */
  if (ADISP(dest_array) == Cnil)
    SET_ADISP(dest_array,list(2,Cnil,from_array));
  else
    Mcdr(ADISP(dest_array)) = make_cons(from_array,Mcdr(ADISP(dest_array)));
  SET_ADISP(from_array,make_cons(make_cons(dest_array,make_fixnum(offset)),Cnil));

  /* now set the actual body of from_array to be the address
    of body in dest_array.  If it is a bit array, this cannot carry the
    offset information, since the body is only recorded as multiples of
    BV_BITS
  */
    
  set_displaced_body_ptr(from_array);

}
    


static enum aelttype
Iarray_element_type(object x)
{enum aelttype t=aet_last;
  switch(TYPE_OF(x))
    { case t_array:
	 t = (enum aelttype) x->a.a_elttype;
	 break;
       case t_simple_vector:
       case t_vector:
	 t = (enum aelttype) x->v.v_elttype;
	 break;
       case t_simple_bitvector:
       case t_bitvector:
	 t = aet_bit;
	 break;
       case t_simple_string:
       case t_string:
	 t = aet_ch;
	 break;
       default:
	 FEwrong_type_argument(sLarray,x);
       }
  return t;
}


void
adjust_displaced(object x) {

  set_displaced_body_ptr(x);
  for (x = ADISP(x)->c.c_cdr;  x != Cnil;  x = x->c.c_cdr)
    adjust_displaced(x->c.c_car);
  
}




   /* RAW_AET_PTR returns a pointer to something of raw type obtained from X
      suitable for using GSET for an array of elt type TYP.
      If x is the null pointer, return a default for that array element
      type.
      */

static char *
raw_aet_ptr(object x, short int typ)
{  /* doubles are the largest raw type */

  static union{
    object o;char c;int i;unsigned int ui;
    fixnum f;shortfloat sf;longfloat d;
    unsigned char uc;short s;unsigned short us;} u;

  if (x==Cnil) 
    return aet_types[typ].dflt;

  switch (typ){
/* #define STORE_TYPED(pl,type,val) *((type *) pl) = (type) val; break; */
  case aet_object: 
    /* STORE_TYPED(&u,object,x); */
    u.o=x;
    break;
  case aet_ch:     
    /* STORE_TYPED(&u,char, char_code(x)); */
    u.c=char_code(x);
    break;
  case aet_bit:    
    /* STORE_TYPED(&u,fixnum, -Mfix(x)); */
    u.f=-Mfix(x);
    break;
  case aet_fix:    
  case aet_nnfix:    
    /* STORE_TYPED(&u,fixnum, Mfix(x)); */
    u.f=Mfix(x);
    break;
  case aet_sf:     
    /* STORE_TYPED(&u,shortfloat, Msf(x)); */
    u.sf=Msf(x);
    break;
  case aet_lf:     
    /* STORE_TYPED(&u,longfloat, Mlf(x)); */
    u.d=Mlf(x);
    break;
  case aet_char:   
  case aet_nnchar:   
    /* STORE_TYPED(&u, char, Mfix(x)); */
    u.c=(char)Mfix(x);
    break;
  case aet_uchar:  
    /* STORE_TYPED(&u, unsigned char, Mfix(x)); */
    u.uc=(unsigned char)Mfix(x);
    break;
  case aet_short:  
  case aet_nnshort:  
    /* STORE_TYPED(&u, short, Mfix(x)); */
    u.s=(short)Mfix(x);
    break;
  case aet_ushort: 
    /* STORE_TYPED(&u,unsigned short,Mfix(x)); */
    u.us=(unsigned short)Mfix(x);
    break;
  case aet_int:  
  case aet_nnint:  
    /* STORE_TYPED(&u, int, Mfix(x)); */
    u.i=(int)Mfix(x);
    break;
  case aet_uint: 
    /* STORE_TYPED(&u,unsigned int,Mfix(x)); */
    u.ui=(unsigned int)Mfix(x);
    break;
  default: 
    FEerror("bad elttype",0);
    break;
  }
  return (char *)&u;
}


     /* GSET copies into array ptr P1, the value
	pointed to by the ptr VAL into the next N slots.  The
	array type is typ.  If VAL is the null ptr, use
	the default for that element type
	NOTE: for type aet_bit n is the number of Words
	ie (nbits +WSIZE-1)/WSIZE and the words are set.
	*/     

void
gset(void *p1, void *val, fixnum n, int typ) {

  if (val==0)
    val = aet_types[typ].dflt;

  switch (typ){

#define GSET(p,n,typ,val) {typ x = *((typ *) val); GSET1(p,n,typ,x)}
#define GSET1(p,n,typ,val) while (n-- > 0)	\
      { *((typ *) p) = val; \
	p = p + sizeof(typ);			\
      } break;

    case aet_object: GSET(p1,n,object,val);
    case aet_ch:     GSET(p1,n,char,val);
      /* Note n is number of fixnum WORDS for bit */
    case aet_bit:    GSET(p1,n,fixnum,val);
    case aet_fix:case aet_nnfix:    GSET(p1,n,fixnum,val);
    case aet_sf:     GSET(p1,n,shortfloat,val);
    case aet_lf:     GSET(p1,n,longfloat,val);
    case aet_char:case aet_nnchar:   GSET(p1,n,char,val);
    case aet_uchar:  GSET(p1,n,unsigned char,val);
    case aet_short:case aet_nnshort:  GSET(p1,n,short,val);
    case aet_ushort: GSET(p1,n,unsigned short,val);
    case aet_int:case aet_nnint:  GSET(p1,n,int,val);
    case aet_uint: GSET(p1,n,unsigned int,val);
    default:         FEerror("bad elttype",0);
    }
  }


DEFUN("COPY-ARRAY-PORTION",object,fScopy_array_portion,SI,4,
	  5,NONE,OO,OO,OO,OO,(object x,object y,object o1,object o2,...),
   "Copy elements from X to Y starting at x[i1] to x[i2] and doing N1 \
elements if N1 is supplied otherwise, doing the length of X - I1 \
elements.  If the types of the arrays are not the same, this has \
implementation dependent results.") { 

  enum aelttype typ1=Iarray_element_type(x);
  enum aelttype typ2=Iarray_element_type(y);
  fixnum i1=fix(o1),i2=fix(o2);
  int n1,nc;
  fixnum n=INIT_NARGS(4);
  object z,l=Cnil,f=OBJNULL;
  va_list ap;

  va_start(ap,o2);
  z=NEXT_ARG(n,ap,l,f,OBJNULL);
  n1=z==OBJNULL ? x->v.v_dim-i1 : fix(z);
  va_end(ap);

  if (typ1==aet_bit) {
    if (i1 % CHAR_SIZE)
    badcopy:
      FEerror("Bit copies only if aligned",0);
    else {
      int rest=n1%CHAR_SIZE;
      if (rest!=0) {
	if (typ2!=aet_bit)
	  goto badcopy;
	while(rest> 0) {
	  FFN(fSaset1)(y,i2+n1-rest,(FFN(fLrow_major_aref)(x,i1+n1-rest)));
	  rest--;
	}
      }
      i1=i1/CHAR_SIZE;
      n1=n1/CHAR_SIZE;
      typ1=aet_char;
    }
  }

  if (typ2==aet_bit) {
    if (i2 % CHAR_SIZE)
      goto badcopy;
    i2=i2/CHAR_SIZE ;
  }

  if ((typ1 ==aet_object || typ2  ==aet_object) && typ1 != typ2)
    FEerror("Can't copy between different array types",0);
  nc=n1*aet_types[(int)typ1].size;
  if (i1+n1 > x->a.a_dim || ((y->a.a_dim - i2) *aet_types[(int)typ2].size) < nc)
    FEerror("Copy  out of bounds",0);
  bcopy(x->ust.ust_self + (i1*aet_types[(int)typ1].size),
	y->ust.ust_self + (i2*aet_types[(int)typ2].size),
	nc);

  return x;

}

/* X is the header of an array.  This supplies the body which
   will not be relocatable if STATICP.  If DFLT is 0, do not
   initialize (the caller promises to reset these before the
   next gc!).   If DFLT == Cnil then initialize to default type
   for this array type.   Otherwise DFLT is an object and its
   value is used to init the array */
   
void
array_allocself(object x, int staticp, object dflt)
{
	int n;
	void *(*fun)(size_t),*tmp_alloc;
	enum aelttype typ;
	fun = (staticp ? alloc_contblock : alloc_relblock);
	{  /* this must be called from within no interrupt code */
	n = x->a.a_dim;
	typ = Iarray_element_type(x);
	switch (typ) {
	case aet_object:
		x->a.a_self = AR_ALLOC(*fun,n,object);
		break;
	case aet_ch:
	case aet_char:
	case aet_nnchar:
        case aet_uchar:
		x->st.st_self = AR_ALLOC(*fun,n,char);
		break;
        case aet_short:
        case aet_nnshort:
        case aet_ushort:
		x->ust.ust_self = (unsigned char *) AR_ALLOC(*fun,n,short);
		break;
        case aet_int:
        case aet_nnint:
        case aet_uint:
		x->ust.ust_self = (unsigned char *) AR_ALLOC(*fun,n,int);
		break;
	case aet_bit:
	  n=ceil(n,BV_ALLOC);
	  n++;/*allow for arrays displaced to end BV_ALLOC access*/
	  SET_BV_OFFSET(x,0);
	case aet_fix:
	case aet_nnfix:
	  x->a.a_self = (void *)AR_ALLOC(*fun,n,fixnum);
	  break;
	case aet_sf:
	  x->a.a_self = (void *)AR_ALLOC(*fun,n,shortfloat);
	  break;
	case aet_lf:
	  x->a.a_self = (void *)AR_ALLOC(*fun,n,longfloat);
	  break;
	default:
	  break;
	}
	if(dflt!=OBJNULL) gset(x->st.st_self,raw_aet_ptr(dflt,typ),n,typ);
      }
	
}

DEFUN("FILL-POINTER-SET",object,fSfill_pointer_set,SI,2,2,NONE,OO,IO,OO,OO,(object x,fixnum i),"") {

    if (!(TS_MEMBER(type_of(x),TS(t_vector)|TS(t_bitvector)|TS(t_string))))
      goto no_fillp;
    if (x->v.v_hasfillp == 0)
      goto no_fillp;
    if (i < 0 || i > x->a.a_dim)
      FEerror("~a is not suitable for a fill pointer for ~a",2,make_fixnum(i),x);
    x->v.v_fillp = i;
    return make_fixnum(i);
  
  no_fillp:
	FEerror("~a does not have a fill pointer",1,x);

  return make_fixnum(0);
}

/* DEFUN("FILL-POINTER-INTERNAL",fixnum,fSfill_pointer_internal,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") { */
/*   RETURN1(x->v.v_fillp); */
/* } */

DEFUN("ARRAY-HAS-FILL-POINTER-P",object,fLarray_has_fill_pointer_p,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  if (TS_MEMBER(type_of(x),TS(t_vector)|TS(t_bitvector)|TS(t_string)))
    return (x->v.v_hasfillp == 0 ? Cnil : sLt);
  else if (TYPE_OF(x) == t_array)
    return Cnil;
  else IisArray(x);
  return Cnil;
}


	
/* DEFUN("MAKE-ARRAY-INTERNAL",object,fSmake_array_internal,SI,0,0,NONE,OO,OO,OO,OO)
 (element_type,adjustable,displaced_to,displaced_index_offset,static,initial_element,dimensions)
  object element_type,adjustable,displaced_to,displaced_index_offset,static,initial_element,dimensions;
     
*/

DEFUN("ARRAY-ELEMENT-TYPE",object,fLarray_element_type,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") { 
  enum aelttype t;
  t = Iarray_element_type(x);
  return * aet_types[(int)t].namep;
}


DEFUN("REF",object,fSref,SI,5,5,NONE,OI,II,IO,OO,(fixnum addr,fixnum s,fixnum u,fixnum z,object v),"") { 

#define el(s_,e_) ((Mjoin(u,s_) *)addr)->e_
#define nw(s_,e_,v_) ({if (z) el(s_,e_)=v_(v); el(s_,e_);})

  switch (s) {
  case 1:
    switch (u) {
    case aem_character: RETURN1(code_char(nw(8,u,char_code)));
    case aem_unsigned:  RETURN1(make_fixnum(nw(8,u,fix)));
    case aem_signed:    RETURN1(make_fixnum(nw(8,i,fix)));
    default: FEerror("Bad mode",0); RETURN1(Cnil);
    }
  case 2:
    switch (u) {
    case aem_unsigned:  RETURN1(make_fixnum(nw(16,u,fix)));
    case aem_signed:    RETURN1(make_fixnum(nw(16,i,fix)));
    default: FEerror("Bad mode",0); RETURN1(Cnil);
    }
  case 4:
    switch (u) {
    case aem_signed:    RETURN1(make_fixnum(nw(32,i,fix)));
    case aem_float:     RETURN1(make_shortfloat(nw(32,f,sf)));
#if SIZEOF_LONG!=4
    case aem_unsigned:  RETURN1(make_fixnum(nw(32,u,fix)));
#else
    case aem_t:         RETURN1(nw(32,o,));
#endif
    default: FEerror("Bad mode",0); RETURN1(Cnil);
    }
  case 8:
    switch (u) {
#if SIZEOF_LONG!=4
    case aem_t:         RETURN1(nw(64,o,));
    case aem_signed:    RETURN1(make_fixnum(nw(64,i,fix)));
#endif
    case aem_float:     RETURN1(make_longfloat(nw(64,f,lf)));
    case aem_complex:   RETURN1(make_fcomplex(nw(64,c,sfc)));
    default: FEerror("Bad mode",0); RETURN1(Cnil);
    }
  case 16:
    switch (u) {
    case aem_complex:   RETURN1(make_dcomplex(nw(64,c,lfc)));
    default: FEerror("Bad mode",0); RETURN1(Cnil);
    }
  default:
    FEerror("Bad size", 0);
    RETURN1(Cnil);
  }
}

DEFUN("CREF",object,fScref,SI,5,5,NONE,OI,II,IO,OO,(fixnum addr,fixnum s,fixnum u,fixnum z,object v),"") { 
  RETURN1(FFN(fSref)(addr,s,u,z,v));
}


DEFUN("RREF",object,fSrref,SI,4,5,NONE,OO,II,IO,OO,(object x,fixnum i,fixnum s,fixnum u,...),"") { 
  fixnum n=INIT_NARGS(4);
  object l=Cnil,f=OBJNULL,v;
  va_list ap;

  va_start(ap,u);
  v=NEXT_ARG(n,ap,l,f,OBJNULL);
  va_end(ap);

  RETURN1(FFN(fSref)((long)((char *)x->a.a_self+i*elt_size(x->a.a_elttype)),s,u,v!=OBJNULL,v));

}

DEFUN("ARRAY-ELTSIZE",object,fSarray_eltsize,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  RETURN1((object)elt_size(x->a.a_elttype));
}

DEFUN("ARRAY-DIMS",object,fSarray_dims,SI,2,2,NONE,IO,IO,OO,OO,(object x,fixnum i),"") {
  RETURN1((object)x->a.a_dims[i]);
}

DEFUN("ARRAY-MODE",object,fSarray_mode,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  RETURN1((object)elt_mode(x->a.a_elttype));
}

DEFUN("ARRAY-HASFILLP",object,fSarray_hasfillp,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  RETURN1((object)(fixnum)x->a.a_hasfillp);
}

DEFUN("VECTOR-DIM",object,fSvector_dim,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  RETURN1((object)(fixnum)x->v.v_dim);
}

DEFUN("ARRAY-ELTTYPE",object,fSarray_elttype,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  RETURN1((object)(fixnum)x->a.a_elttype);
}

DEFUN("ADJUSTABLE-ARRAY-P",object,fLadjustable_array_p,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") { 
  IisArray(x);
  switch (type_of(x)) {
  case t_array:
    x=x->a.a_adjustable ? Ct : Cnil; 
    break;
  case t_string:
    x=x->st.st_adjustable ? Ct : Cnil; 
    break;
  case t_vector:
    x=x->v.v_adjustable ? Ct : Cnil; 
    break;
  case t_bitvector:
    x=x->bv.bv_adjustable ? Ct : Cnil; 
    break;
  default:
    FEerror("Bad array type",0);
    break;
  }
  return x;
}

DEFUN("DISPLACED-ARRAY-P",object,fSdisplaced_array_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") { 
  IisArray(x);
  return (ADISP(x) == Cnil ? Cnil : sLt);
}

DEFUN("ARRAY-RANK",object,fLarray_rank,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") { 
  if (type_of(x) == t_array)
    RETURN1(make_fixnum(x->a.a_rank));
  IisArray(x);
  RETURN1(make_fixnum(1));
}

DEFUN("ARRAY-DIMENSION",object,fLarray_dimension,LISP,2,2,NONE,OO,IO,OO,OO,(object x,fixnum i),"") { 

  if (type_of(x) == t_array) {  
    if ((unsigned int)i >= x->a.a_rank)
      TYPE_ERROR(make_fixnum(i),list(3,sLinteger,make_fixnum(0),make_fixnum(x->a.a_rank)));
    else { 
      RETURN1(make_fixnum(x->a.a_dims[i]));
    }
  }
  IisArray(x);
  RETURN1(make_fixnum(x->v.v_dim));
}
#ifdef STATIC_FUNCTION_POINTERS
object
fLarray_dimension(object x,fixnum i) {
  return FFN(fLarray_dimension)(x,i);
}
#endif

static void
Icheck_displaced(object displaced_list, object ar, int dim)
{ 
  while (displaced_list!=Cnil)
    { object u = Mcar(displaced_list);
      displaced_list = Mcdr(displaced_list);
      if (u->a.a_self == NULL) continue;
      if ((Iarray_element_type(u) == aet_bit &&
	   (u->bv.bv_self - ar->bv.bv_self)*BV_BITS +u->bv.bv_dim -dim
	    + BV_OFFSET(u) - BV_OFFSET(ar) > 0)
	  || (ARRAY_BODY_PTR(u,u->a.a_dim) > ARRAY_BODY_PTR(ar,dim)))
	FEerror("Bad displacement",0);
      Icheck_displaced(DISPLACED_FROM(u),ar,dim);
    }
}

DEFUN("MEMCPY",object,fSmemcpy,SI,3,3,NONE,II,II,OO,OO,(fixnum x,fixnum y,fixnum z),"") {

  RETURN1((object)(fixnum)memcpy((void *)x,(void *)y,z));

}

DEFUN("MEMMOVE",object,fSmemmove,SI,3,3,NONE,II,II,OO,OO,(fixnum x,fixnum y,fixnum z),"") {

  RETURN1((object)(fixnum)memmove((void *)x,(void *)y,z));

}


DEFUN("REPLACE-ARRAY",object,fSreplace_array,SI,2,2,NONE,OO,OO,OO,OO,(object old,object new),"") { 
  
  struct dummy fw;
  int offset;
  object displaced;
  enum type otp=type_of(old),ntp=type_of(new);;
    
  fw = old->d;
  old = IisArray(old);
  
  if (otp != ntp || (otp == t_array && old->a.a_rank != new->a.a_rank))
    FEerror("Cannot do array replacement ~a by ~a",2,old,new);

  offset = new->ust.ust_self  - old->ust.ust_self;
  displaced = make_cons(DISPLACED_TO(new),DISPLACED_FROM(old));
  Icheck_displaced(DISPLACED_FROM(old),old,new->a.a_dim);

  switch (otp) {
  case t_array:
    old->a=new->a;
    break;
  case t_bitvector:
    old->bv=new->bv;
    break;
  case t_vector:
    old->v=new->v;
    break;
  case t_string:
    old->st=new->st;
    break;
  default:
    FEwrong_type_argument(sLarray,old);
    break;
  }
    
  /* prevent having two arrays with the same body--which are not related
     that would cause the gc to try to copy both arrays and there might
     not be enough space. */
  new->a.a_dim = 0;
  new->a.a_self = 0;

  SET_ADISP(old,displaced);
  adjust_displaced(old);

  return old;

}

DEFUN("ARRAY-TOTAL-SIZE",object,fLarray_total_size,LISP,1,1,NONE,IO,OO,OO,OO,(object x),"") {
  x = IisArray(x);
  RETURN1((object)(fixnum)x->a.a_dim);
}

DEFUN("ASET-BY-CURSOR",object,fSaset_by_cursor,SI,3,3,NONE,OO,OO,OO,OO,(object array,object val,object cursor),"") {

  object x=(VFUN_NARGS=-3,FFN(fSaset)(val,array,cursor));
  RETURN1(x);

}

void
gcl_init_array_function(void) {
  make_function("ARRAY-DISPLACEMENT", Larray_displacement);

}
     



