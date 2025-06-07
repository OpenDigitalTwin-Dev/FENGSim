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
	list.d

	list manipulating routines
*/

#include "include.h"
#include "num_include.h"
#include "page.h"


object
car(x)
object x;
{
	if (x == Cnil)
		return(x);
	if (consp(x))
		return(x->c.c_car);
	FEwrong_type_argument(sLlist, x);
	return(Cnil);
}

object
cdr(x)
object x;
{
	if (x == Cnil)
		return(x);
	if (consp(x))
		return(x->c.c_cdr);
	FEwrong_type_argument(sLlist, x);
	return(Cnil);
}


void
stack_cons(void)
{
	object d=vs_pop,a=vs_pop;
	*vs_top++ = make_cons(a,d);
}

object on_stack_list_vector_new(fixnum n,object first,va_list ap)
{object res=(object) alloca_val;
 struct cons *p;
 object x;
 int jj=0;
 p=(struct cons *) res;
 if (n<=0) return Cnil;
 TOP:
#ifdef WIDE_CONS
 set_type_of(p,t_cons);
#endif
 p->c_car= jj||first==OBJNULL ? va_arg(ap,object) : first;
 jj=1;
 if (--n == 0)
   {p->c_cdr = Cnil;
    return res;}
 else
   { x= (object) p;
     x->c.c_cdr= (object) ( ++p);}
 goto TOP;
}

object on_stack_list(fixnum n,...) {
  object x,first;
  va_list ap;
  va_start(ap,n);
  first=va_arg(ap,object);
  x=on_stack_list_vector_new(n,first,ap);
  va_end(ap);
  return x;
}


object
list_vector_new(int n,object first,va_list ap) {

  object ans,*p;

  for (p=&ans;n-->0;first=OBJNULL)
    collect(p,make_cons(first==OBJNULL ? va_arg(ap,object) : first,Cnil));
  *p=Cnil;
 return ans;

}
   
#ifdef WIDE_CONS
#define maybe_set_type_of(a,b) set_type_of(a,b)
#else
#define maybe_set_type_of(a,b)
#endif

#define multi_cons(n_,next_,last_)					\
  ({_tm->tm_nfree -= n_;						\
    for(_x=_tm->tm_free,_p=&_x;n_-->0;_p=&(*_p)->c.c_cdr) {		\
      object _z=*_p;							\
      pageinfo(_z)->in_use++;						\
      maybe_set_type_of(_z,t_cons);					\
      _z->c.c_cdr=OBJ_LINK(_z);						\
      _z->c.c_car=next_;						\
    }									\
    _tm->tm_free=*_p;							\
    *_p=SAFE_CDR(last_);						\
    _x;})

#define n_cons(n_,next_,last_)						\
  ({fixnum _n=n_;object _x=Cnil,*_p;					\
    static struct typemanager *_tm=tm_table+t_cons;			\
    if (_n>=0) {/*FIXME vs_top<vs_base*/				\
      BEGIN_NO_INTERRUPT;						\
      if (_n<=_tm->tm_nfree && !stack_alloc_start)			\
	_x=multi_cons(_n,next_,last_);					\
      else {								\
	for (_p=&_x;_n--;)						\
	  collect(_p,make_cons(next_,Cnil));				\
	*_p=SAFE_CDR(last_);						\
      }									\
      END_NO_INTERRUPT;							\
    }									\
    _x;})

object
n_cons_from_x(fixnum n,object x) {

  return n_cons(n,({object _z=x->c.c_car;x=x->c.c_cdr;_z;}),Cnil);

}


object
listqA(int a,int n,va_list ap) {

  return n_cons(n,va_arg(ap,object),a ? va_arg(ap,object) : Cnil);

}

object list(fixnum n,...) { 

  va_list ap;
  object lis;

  va_start(ap,n);
  lis=listqA(0,n,ap);
  va_end(ap);
  return lis;

}

object listA(fixnum n,...) { 

  va_list ap;
  object lis;

  va_start(ap,n);
  lis=listqA(1,n-1,ap);
  va_end(ap);
  return lis;

}



object
append(object x, object y) {

  return n_cons(length(x),({object _t=x->c.c_car;x=x->c.c_cdr;_t;}),y);

}

object
copy_list(x)
object x;
{
	object y;

	if (!consp(x))
		return(x);
	y = make_cons(x->c.c_car, Cnil);
	vs_push(y);
	for (x = x->c.c_cdr; consp(x); x = x->c.c_cdr) {
		y->c.c_cdr = make_cons(x->c.c_car, Cnil);
		y = y->c.c_cdr;
	}
	y->c.c_cdr = x;
	return(vs_pop);
}


DEFUN("CONS",object,fLcons,LISP,2,2,NONE,OO,OO,OO,OO,(object a,object d),"") {

  object x=alloc_object(t_cons);
  x->c.c_car=a;
  x->c.c_cdr=d;
  RETURN1(x);

}

object
make_list(fixnum n) {
  object x =Cnil ;
  while (n-- > 0)
    x = make_cons(Cnil, x);
  return x;
}

static fixnum
list_count(fixnum nargs,object first,object l,va_list ap) {
  fixnum n;
  for (n=0;NEXT_ARG(nargs,ap,l,first,OBJNULL)!=OBJNULL;n++);
  return n;
}

DEFUN("LIST",object,fLlist,LISP,0,MAX_ARGS,NONE,OO,OO,OO,OO,(object first,...),"") {

  object x,l=Cnil;
  va_list ap;
  fixnum nargs=INIT_NARGS(0),n;

  va_start(ap,first);
  n=list_count(nargs,first,l,ap);
  va_end(ap);
  va_start(ap,first);
  x=n_cons(n,NEXT_ARG(nargs,ap,l,first,Cnil),NEXT_ARG(nargs,ap,l,first,Cnil));
  va_end(ap);
  RETURN1(x);

}

DEFUN("LIST*",object,fLlistA,LISP,1,MAX_ARGS,NONE,OO,OO,OO,OO,(object first,...),"") {

  object x,l=Cnil;
  va_list ap;
  fixnum nargs=INIT_NARGS(0),n;

  va_start(ap,first);
  n=list_count(nargs,first,l,ap);
  va_end(ap);
  va_start(ap,first);
  x=n_cons(n-1,NEXT_ARG(nargs,ap,l,first,Cnil),NEXT_ARG(nargs,ap,l,first,Cnil));
  va_end(ap);
  RETURN1(x);

}

void
stack_list(void) {

  object *a;

  a=vs_base;
  vs_base[0]=n_cons(vs_top-vs_base,*a++,Cnil);
  vs_top=vs_base+1;

}
 
object on_stack_make_list(n)
int n;
{ object res=(object) alloca_val;
 struct cons *p = (struct cons *)res;
 if (n<=0) return Cnil;
  TOP:
#ifdef WIDE_CONS
 set_type_of(p,t_cons);
#endif
 p->c_car=Cnil;
 if (--n == 0)
   {p->c_cdr = Cnil;
    return res;}
 else
   {object  x= (object) p;
     x->c.c_cdr= (object) ( ++p);}
 goto TOP;
}


DEFUN("RPLACA",object,fLrplaca,LISP,2,2,NONE,OO,OO,OO,OO,(object o,object c),"") {

  check_type_cons(&o);
  o->c.c_car = c;
  RETURN1(o);

}

DEFUN("RPLACD",object,fLrplacd,LISP,2,2,NONE,OO,OO,OO,OO,(object o,object d),"") {

  check_type_cons(&o);
  o->c.c_cdr = d;
  RETURN1(o);

}

 
void
check_proper_list(alist)
object alist;
{
    object v;
    /*
    if (alist == Cnil)
	 FEwrong_type_argument(sLlist, alist);
    */
    for (v=alist ; consp(v) ; v=v->c.c_cdr);
    if (v != Cnil)
      TYPE_ERROR(alist,siLproper_list);
}


DEFUN("PROPER-LISTP",object,fSproper_listp,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") { 
  check_proper_list(x);
  RETURN1(Ct);
}


bool
member_eq(x, l)
object x, l;
{

	for (;  consp(l);  l = l->c.c_cdr)
		if (x == l->c.c_car)
			return(TRUE);
	return(FALSE);
}

void
delete_eq(x, lp)
object x, *lp;
{
	for (;  consp(*lp);  lp = &(*lp)->c.c_cdr)
		if ((*lp)->c.c_car == x) {
			*lp = (*lp)->c.c_cdr;
			return;
		}
}

DEFUN("STATIC-INVERSE-CONS",object,fSstatic_inverse_cons,SI,1,1,NONE,OI,OO,OO,OO,(fixnum x),"") {

   object y=(object)x;

   return is_imm_fixnum(y) ? Cnil : (is_imm_fixnum(y->c.c_cdr) ? y : (y->d.f||y->d.e ? Cnil : y));

}

void
gcl_init_list_function()
{

	sKtest = make_keyword("TEST");
	sKtest_not = make_keyword("TEST-NOT");
	sKkey = make_keyword("KEY");

	sKinitial_element = make_keyword("INITIAL-ELEMENT");


}
