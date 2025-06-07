/* Copyright (C) 2024 Camm Maguire */
#include "include.h"


DEFUN("TP0",object,fStp0,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp0(x);}
DEFUN("TP1",object,fStp1,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp1(x);}
DEFUN("TP2",object,fStp2,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp2(x);}
DEFUN("TP3",object,fStp3,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp3(x);}
DEFUN("TP4",object,fStp4,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp4(x);}
DEFUN("TP5",object,fStp5,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp5(x);}
DEFUN("TP6",object,fStp6,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp6(x);}
DEFUN("TP7",object,fStp7,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp7(x);}
DEFUN("TP8",object,fStp8,SI,1,1,NONE,IO,OO,OO,OO,(object x),"") {return (object)(fixnum)tp8(x);}

DEFUN("C-OBJECT-==",object,fSc_object_eq,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  RETURN1(x==y?Ct:Cnil);
}
DEFUN("C-FIXNUM-==",object,fSc_fixnum_eq,SI,2,2,NONE,OI,IO,OO,OO,(fixnum x,fixnum y),"") {
  RETURN1(x==y?Ct:Cnil);
}
DEFUN("C-FLOAT-==",object,fSc_float_eq,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  check_type(x,t_shortfloat);
  check_type(y,t_shortfloat);
  RETURN1(sf(x)==sf(y)?Ct:Cnil);
}
DEFUN("C-DOUBLE-==",object,fSc_double_eq,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  check_type(x,t_longfloat);
  check_type(y,t_longfloat);
  RETURN1(lf(x)==lf(y)?Ct:Cnil);
}
DEFUN("C-FCOMPLEX-==",object,fSc_fcomplex_eq,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  check_type(x,t_complex);
  check_type(y,t_complex);
  check_type(x->cmp.cmp_real,t_shortfloat);
  check_type(y->cmp.cmp_real,t_shortfloat);
  RETURN1(sfc(x)==sfc(y)?Ct:Cnil);
}
DEFUN("C-DCOMPLEX-==",object,fSc_dcomplex_eq,SI,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  check_type(x,t_complex);
  check_type(y,t_complex);
  check_type(x->cmp.cmp_real,t_longfloat);
  check_type(y->cmp.cmp_real,t_longfloat);
  RETURN1(lfc(x)==lfc(y)?Ct:Cnil);
}

DEFUN("C+",object,fScp,SI,2,2,NONE,II,IO,OO,OO,(fixnum x,fixnum y),"") {
  RETURN1((object)(x+y));
}
DEFUN("&",object,fSand,SI,2,2,NONE,II,IO,OO,OO,(fixnum x,fixnum y),"") {
  RETURN1((object)(x&y));
}
DEFUN("|",object,fSor,SI,2,2,NONE,II,IO,OO,OO,(fixnum x,fixnum y),"") {
  RETURN1((object)(x|y));
}
DEFUN("^",object,fSxor,SI,2,2,NONE,II,IO,OO,OO,(fixnum x,fixnum y),"") {
  RETURN1((object)(x^y));
}
DEFUN("~",object,fSnot,SI,1,1,NONE,II,OO,OO,OO,(fixnum x),"") {
  RETURN1((object)~x);
}
DEFUN("<<",object,fSlshft,SI,2,2,NONE,II,IO,OO,OO,(fixnum x,fixnum y),"") {
  RETURN1((object)(x<<y));
}
DEFUN(">>",object,fSrshft,SI,2,2,NONE,II,IO,OO,OO,(fixnum x,fixnum y),"") {
  RETURN1((object)(x>>y));
}

static inline bool
TESTA(object x_,object y_,object key,object test,object test_not) {
  object _y=key==Cnil ? y_ : ifuncall1(key,y_);
  if (test!=Cnil)
    return ifuncall2(test,x_,_y)!=Cnil;
  else if (test_not!=Cnil)
    return ifuncall2(test_not,x_,_y)==Cnil;
  else 
    return eql(x_,_y);
}
  
#define MTEST(y_) TESTA(x,y_,key,test,test_not)

#define DEFKTFUN(n_,s_,p_,code_)					\
  DEFUN(n_,object,s_,p_,2,63,NONE,OO,OO,OO,OO,(object x,object y,...),"") { \
									\
  fixnum n=INIT_NARGS(2);						\
  object l=Cnil,f=OBJNULL,*base=vs_top,z,key,test,test_not;		\
  va_list ap;								\
  va_start(ap,y);							\
  for (;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL;)			\
    vs_push(z);								\
  va_end(ap);								\
									\
  parse_key(base,FALSE,FALSE,3,sKtest,sKtest_not,sKkey);		\
  key=base[2];test=base[0];test_not=base[1];vs_top=base;		\
									\
  RETURN1(code_);							\
									\
  }

#define DEFPFUN(n_,s_,p_,test_,call_)					\
  DEFUN(n_,object,s_,p_,2,63,NONE,OO,OO,OO,OO,(object x,object y,...),"") { \
									\
  fixnum n=INIT_NARGS(2);						\
  object l=Cnil,f=OBJNULL,*base=vs_top,z;				\
  va_list ap;								\
									\
  va_start(ap,y);							\
  for (;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL;)			\
    vs_push(z);								\
  va_end(ap);								\
									\
  parse_key(base,FALSE,FALSE,1,sKkey);					\
                                                                        \
  vs_top=base;								\
  RETURN1((VFUN_NARGS=6,FFN(call_)(x,y,test_,sLfuncall,sKkey,base[0]))); \
  }

#define DEFKTPFUN(n_,s_,p_,code_) \
  DEFKTFUN(n_,s_,p_,code_)\
  DEFPFUN(n_ "-IF",Mjoin(s_,_if),p_,sKtest,s_)	\
  DEFPFUN(n_ "-IF-NOT",Mjoin(s_,_if_not),p_,sKtest_not,s_)


DEFKTPFUN("MEMBER",fLmember,LISP,({for (;!endp(y) && !MTEST(y->c.c_car);y=y->c.c_cdr);y;}))
DEFKTFUN("ASSOC",fLassoc,LISP,({for (;!endp(y) && (y->c.c_car==Cnil || !MTEST(y->c.c_car->c.c_car));y=y->c.c_cdr);y->c.c_car;}))
DEFKTFUN("RASSOC",fLrassoc,LISP,({for (;!endp(y) && (y->c.c_car==Cnil || !MTEST(y->c.c_car->c.c_cdr));y=y->c.c_cdr);y->c.c_car;}))

DEFKTFUN("ADJOIN",fLadjoin,LISP,					\
	 ({object z,q=x;						\
	   x=base[2]==Cnil ? x : ifuncall1(base[2],x);			\
	   for (z=y;!endp(z) && !MTEST(z->c.c_car);z=z->c.c_cdr);	\
	   z==Cnil ? MMcons(q,y) : y;}))

DEFUN("TAILP",object,fLtailp,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  for (;consp(y) && y!=x;y=y->c.c_cdr);
  RETURN1(eql(x,y) ? Ct : Cnil);
}

static inline object
subst(object tree,object new,object x,object key,object test,object test_not) {

  if (TESTA(x,tree,key,test,test_not))
    return new;
  else if (consp(tree)) {
    object a=subst(tree->c.c_car,new,x,key,test,test_not),d=subst(tree->c.c_cdr,new,x,key,test,test_not);
    return a==tree->c.c_car && d==tree->c.c_cdr ? tree : MMcons(a,d);
  } else
    return tree;

}

DEFUN("SUBST",object,fLsubst,LISP,3,63,NONE,OO,OO,OO,OO,(object new,object x,object y,...),"") {
  									
  fixnum n=INIT_NARGS(3);						
  object l=Cnil,f=OBJNULL,*base=vs_top,z,key,test,test_not;		
  va_list ap;								

  va_start(ap,y);							
  for (;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL;)			
    vs_push(z);								
  va_end(ap);								
  									
  parse_key(base,FALSE,FALSE,3,sKtest,sKtest_not,sKkey);		
  key=base[2];test=base[0];test_not=base[1];vs_top=base;		
  									
  RETURN1(subst(y,new,x,key,test,test_not));
  									
}

DEFUN("LDIFF",object,fLldiff,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  object first=Cnil,last=Cnil,z;

  if (!listp(x))/*FIXME checktype*/
    TYPE_ERROR(x,sLlist);
  for (;consp(x) && x!=y;x=x->c.c_cdr)
    if (first==Cnil)
      first=last=MMcons(x->c.c_car,Cnil);
    else {
      last->c.c_cdr=(z=MMcons(x->c.c_car,Cnil));
      last=z;
    }
  if (first!=Cnil)
    last->c.c_cdr=eql(x,y) ? Cnil : x;
  RETURN1(first);
}

DEFUN("SUBSETP",object,fLsubsetp,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  if (!listp(x))/*FIXME checktype*/
    TYPE_ERROR(x,sLlist);
  if (!listp(y))/*FIXME checktype*/
    TYPE_ERROR(y,sLlist);
  for (;consp(x);x=x->c.c_cdr)
    if (FFN(fLmember)(x->c.c_car,y)==Cnil)
      RETURN1(Cnil);

  RETURN1(Ct);
}

DEFUN("CAR",object,fLcar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  check_type_list(&x);
  RETURN1(x->c.c_car);
}
DEFUN("CDR",object,fLcdr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  check_type_list(&x);
  RETURN1(x->c.c_cdr);
}
DEFUN("CAAR",object,fLcaar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcar)(x)));
}
DEFUN("CADR",object,fLcadr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcdr)(x)));
}
DEFUN("CDAR",object,fLcdar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcar)(x)));
}
DEFUN("CDDR",object,fLcddr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcdr)(x)));
}
DEFUN("CAAAR",object,fLcaaar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcar)(FFN(fLcar)(x))));
}
DEFUN("CAADR",object,fLcaadr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcar)(FFN(fLcdr)(x))));
}
DEFUN("CADAR",object,fLcadar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcdr)(FFN(fLcar)(x))));
}
DEFUN("CADDR",object,fLcaddr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcdr)(FFN(fLcdr)(x))));
}
DEFUN("CDAAR",object,fLcdaar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcar)(FFN(fLcar)(x))));
}
DEFUN("CDADR",object,fLcdadr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcar)(FFN(fLcdr)(x))));
}
DEFUN("CDDAR",object,fLcddar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcar)(x))));
}
DEFUN("CDDDR",object,fLcdddr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcdr)(x))));
}

DEFUN("CAAAAR",object,fLcaaaar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcar)(FFN(fLcar)(FFN(fLcar)(x)))));
}
DEFUN("CAAADR",object,fLcaaadr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcar)(FFN(fLcar)(FFN(fLcdr)(x)))));
}
DEFUN("CAADAR",object,fLcaadar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcar)(FFN(fLcdr)(FFN(fLcar)(x)))));
}
DEFUN("CAADDR",object,fLcaaddr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcar)(FFN(fLcdr)(FFN(fLcdr)(x)))));
}
DEFUN("CADAAR",object,fLcadaar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcdr)(FFN(fLcar)(FFN(fLcar)(x)))));
}
DEFUN("CADADR",object,fLcadadr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcdr)(FFN(fLcar)(FFN(fLcdr)(x)))));
}
DEFUN("CADDAR",object,fLcaddar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcar)(x)))));
}
DEFUN("CADDDR",object,fLcadddr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcar)(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcdr)(x)))));
}

DEFUN("CDAAAR",object,fLcdaaar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcar)(FFN(fLcar)(FFN(fLcar)(x)))));
}
DEFUN("CDAADR",object,fLcdaadr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcar)(FFN(fLcar)(FFN(fLcdr)(x)))));
}
DEFUN("CDADAR",object,fLcdadar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcar)(FFN(fLcdr)(FFN(fLcar)(x)))));
}
DEFUN("CDADDR",object,fLcdaddr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcar)(FFN(fLcdr)(FFN(fLcdr)(x)))));
}
DEFUN("CDDAAR",object,fLcddaar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcar)(FFN(fLcar)(x)))));
}
DEFUN("CDDADR",object,fLcddadr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcar)(FFN(fLcdr)(x)))));
}
DEFUN("CDDDAR",object,fLcdddar,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcar)(x)))));
}
DEFUN("CDDDDR",object,fLcddddr,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcdr)(FFN(fLcdr)(x)))));
}

DEFUN("COPY-LIST",object,fLcopy_list,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  object y=Cnil,ly=Cnil;
  for (;consp(x);x=x->c.c_cdr) {
    object z=MMcons(x->c.c_car,Cnil);
    if (y==Cnil)
      y=ly=z;
    else {
      ly->c.c_cdr=z;
      ly=z;
    }
  }
  RETURN1(y);
}
    
DEFUN("LAST",object,fLlast,LISP,1,2,NONE,OO,OO,OO,OO,(object x,...),"") {

  fixnum n=INIT_NARGS(1);						
  object l=Cnil,f=OBJNULL,s,t;
  va_list ap;
  enum type tp;

  va_start(ap,x);
  s=NEXT_ARG(n,ap,l,f,make_fixnum(1));

  if (endp(x))
    RETURN1(Cnil);
  tp=type_of(s);
  if ((tp!=t_fixnum && tp!=t_bignum)|| number_minusp(s))
    TYPE_ERROR(s,list(2,sLinteger,make_fixnum(0)));
  n=tp==t_fixnum ? fix(s) : fix(sLarray_dimension_limit->s.s_dbind);
  t=x;
  if (!n)
    while (consp(t))
      t=t->c.c_cdr;
  else {
    while (consp(x->c.c_cdr) && --n)
      x = x->c.c_cdr;
    while (consp(x->c.c_cdr)) {
      t=t->c.c_cdr;
      x = x->c.c_cdr;
    }
  }
  RETURN1(t);
  
}

DEFUN("BUTLAST",object,fLbutlast,LISP,1,2,NONE,OO,OO,OO,OO,(object lis,...),"") {

  fixnum n=INIT_NARGS(1);						
  object l=Cnil,f=OBJNULL,nn;
  va_list ap;

  va_start(ap,lis);
  nn=NEXT_ARG(n,ap,l,f,make_fixnum(1));

  RETURN1(FFN(fLldiff)(lis,(VFUN_NARGS=2,FFN(fLlast)(lis,nn))));

}


DEFUN("APPEND",object,fSappend,LISP,0,63,NONE,OO,OO,OO,OO,(object first,...),"") {
  fixnum n=INIT_NARGS(0);						
  object l=Cnil,f=first,z,y=Cnil,r=Cnil,rp=Cnil;
  va_list ap;
  va_start(ap,first);
  for (;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL;) {
    if (z==Cnil) continue;
    y=FFN(fLcopy_list)(y);
    if (r==Cnil)
      r=rp=y;
    else
      rp->c.c_cdr=y;
    rp=(VFUN_NARGS=1,FFN(fLlast)(rp));
    y=z;
  }
  va_end(ap);
  if (r==Cnil)
    r=rp=y;
  else
    rp->c.c_cdr=y;
  RETURN1(r);
}

DEFUN("ENDP",object,fSendp,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  if (x==Cnil)
    RETURN1(Ct);
  if (!consp(x))
    FEwrong_type_argument(sLlist,x);
  RETURN1(Cnil);
}


DEFUN("LIST-LENGTH",object,fSlist_length,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  fixnum n;
  object fast, slow;

  for (n=0,fast=slow=x;;) {
    if (endp(fast))
      RETURN1(make_fixnum(n));
    if (endp(fast->c.c_cdr))
      RETURN1(make_fixnum(n+1));
    if (fast == slow && n > 0)
      RETURN1(Cnil);
    n += 2;
    fast = fast->c.c_cdr->c.c_cdr;
    slow = slow->c.c_cdr;
  }
}

DEFUN("MAKE-LIST",object,fSmake_list,LISP,1,63,NONE,OI,OO,OO,OO,(fixnum x,...),"") {
  fixnum n=INIT_NARGS(1);
  object l=Cnil,f=OBJNULL,*base=vs_top,z,r=Cnil;
  va_list ap;

  va_start(ap,x);
  for (;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL;)
    vs_push(z);/*FIXME do this on C stack, or better, do a parse_key taking on arg at a time*/
  va_end(ap);
  parse_key(base,FALSE,FALSE,1,sKinitial_element);

  for (;x--;)
    r=MMcons(base[0],r);

  vs_top=base;
  RETURN1(r);

}

static inline object
copy_tree(object x) {
  return consp(x) ? MMcons(copy_tree(x->c.c_car),copy_tree(x->c.c_cdr)) : x;
}

DEFUN("COPY-TREE",object,fScopy_tree,LISP,1,2,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(copy_tree(x));
}

DEFUN("NCONC",object,fSnconc,LISP,0,63,NONE,OO,OO,OO,OO,(object first,...),"") {
  fixnum n=INIT_NARGS(0);
  object l=Cnil,f=first,z,y=Cnil,r=Cnil,rp=Cnil;
  va_list ap;
  va_start(ap,first);
  for (;(z=NEXT_ARG(n,ap,l,f,OBJNULL))!=OBJNULL;) {
    if (z==Cnil) continue;
    if (r==Cnil)
      r=rp=y;
    else
      rp->c.c_cdr=y;
    rp=(VFUN_NARGS=1,FFN(fLlast)(rp));
    y=z;
  }
  va_end(ap);
  if (r==Cnil)
    r=rp=y;
  else
    rp->c.c_cdr=y;
  RETURN1(r);
}

DEFUN("NRECONC",object,fSnreconc,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {
  object r=Cnil;

  for (;consp(x);x=x->c.c_cdr) {
    if (r!=Cnil) {
      r->c.c_cdr=y;
      y=r;
    }
    r=x;
  }
  if (r!=Cnil) {
    r->c.c_cdr=y;
    y=r;
  }
  RETURN1(y);
}
    

DEFUN("NTH",object,fLnth,LISP,2,2,NONE,OO,OO,OO,OO,(object i,object lst),"") { 
  object x = lst;
  fixnum index=fixint(i);
  if (index < 0)
    FEerror("Negative index: ~D.", 1, make_fixnum(index));
  while (1)
    {if (consp(x))
       { if (index == 0)
	   RETURN1(Mcar(x));
	 else {x = Mcdr(x); index--;}}
      else if (x == sLnil) RETURN1(sLnil);
      else FEwrong_type_argument(sLlist, lst);}
}

DEFUN("NTHCDR",object,fLnthcdr,LISP,2,2,NONE,OO,OO,OO,OO,(object i,object lst),"") { 
  object x = lst;
  fixnum index=fixint(i);
  if (index < 0)
    FEerror("Negative index: ~D.", 1, make_fixnum(index));
  while (1)
    {if (consp(x))
       { if (index == 0)
	   RETURN1(x);
	 else {x = Mcdr(x); index--;}}
      else if (x == sLnil) RETURN1(sLnil);
      else FEwrong_type_argument(sLlist, lst);}
}

DEFUN("FIRST",object,fLfirst,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ RETURN1(car(x)) ;}

DEFUN("SECOND",object,fLsecond,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(1),x);}
DEFUN("THIRD",object,fLthird,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(2),x);}
DEFUN("FOURTH",object,fLfourth,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(3),x);}
DEFUN("FIFTH",object,fLfifth,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(4),x);}
DEFUN("SIXTH",object,fLsixth,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(5),x);}
DEFUN("SEVENTH",object,fLseventh,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(6),x);}
DEFUN("EIGHTH",object,fLeighth,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(7),x);}
DEFUN("NINTH",object,fLninth,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(8),x);}
DEFUN("TENTH",object,fLtenth,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"")
{ return FFN(fLnth)(make_fixnum(9),x);}

static inline object
sublis(object alist,object tree,object key,object test,object test_not) {
  object z;

  for (z = alist;  !endp(z);  z = z->c.c_cdr) {
    object w=z->c.c_car;
    if (TESTA(w->c.c_car,tree,key,test,test_not))
      return w->c.c_cdr;
  }
  if (consp(tree)) {
    object a=sublis(alist,tree->c.c_car,key,test,test_not),d=sublis(alist,tree->c.c_cdr,key,test,test_not);
    return a==tree->c.c_car && d==tree->c.c_cdr ? tree : MMcons(a,d);
  } else
    return tree;

}

DEFKTFUN("SUBLIS",fLsublis,LISP,sublis(x,y,key,test,test_not))

DEFUN("WILD-PATHNAME-P",object,fLwild_pathname_p,LISP,1,2,NONE,OO,OO,OO,OO,(object x,...),"") {
  return Cnil;
}

DEFUN("SET-DIFFERENCE",object,fLset_difference,LISP,2,8,NONE,OO,OO,OO,OO,
	  (object x,object y,...),"") {
  object z=Cnil,yy;
  for (;x!=Cnil;x=x->c.c_cdr) {
    for (yy=y;yy!=Cnil && x->c.c_car!=yy->c.c_car;yy=yy->c.c_cdr);
    if (yy==Cnil)
      z=MMcons(x->c.c_car,z);
  }
  RETURN1(z);

}

DEFUN("UNION",object,fLunion,LISP,2,8,NONE,OO,OO,OO,OO,
	  (object x,object y,...),"") {
  object z=y,yy;
  for (;x!=Cnil;x=x->c.c_cdr) {
    for (yy=z;yy!=Cnil && x->c.c_car!=yy->c.c_car;yy=yy->c.c_cdr);
    if (yy==Cnil)
      z=MMcons(x->c.c_car,z);
  }
  RETURN1(z);

}

DEFUN("NUNION",object,fLnunion,LISP,2,8,NONE,OO,OO,OO,OO,
	  (object x,object y,...),"") {
  object z=Cnil,zp=z,yy;
  for (;x!=Cnil;x=x->c.c_cdr) {
    for (yy=y;yy!=Cnil && x->c.c_car!=yy->c.c_car;yy=yy->c.c_cdr);
    if (yy==Cnil) {
      if (zp!=Cnil) zp->c.c_cdr=x; else z=x;
      zp=x;
    }
  }
  if (zp!=Cnil) zp->c.c_cdr=y;
  RETURN1(z!=Cnil ? z : y);

}

DEFUN("INTERSECTION",object,fLintersection,LISP,2,8,NONE,OO,OO,OO,OO,
	  (object x,object y,...),"") {
  object z=Cnil,yy;
  for (;x!=Cnil;x=x->c.c_cdr) {
    for (yy=y;yy!=Cnil && x->c.c_car!=yy->c.c_car;yy=yy->c.c_cdr);
    if (yy!=Cnil)
      z=MMcons(x->c.c_car,z);
  }
  RETURN1(z);

}

DEFUN("SBIT",object,fLsbit,LISP,2,2,NONE,IO,IO,OO,OO,(object x,fixnum i),"") {
  RETURN1((object)fix(fLrow_major_aref(x,i)));

}

DEFUNM("GETHASH",object,fLgethash,LISP,2,3,NONE,OO,OO,OO,OO,(object x,object y,...),"") {

  fixnum nargs=INIT_NARGS(2),vals=(fixnum)fcall.valp;
  object *base=vs_top,l=Cnil,f=OBJNULL,z;
  va_list ap;
  struct cons *e;

  check_type_hash_table(&y);
  e=gethash(x,y);
  if (e->c_cdr != OBJNULL)
    RETURN2(e->c_car,Ct);
  else {
    va_start(ap,y);
    z=NEXT_ARG(nargs,ap,l,f,Cnil);
    va_end(ap);
    RETURN2(z,Cnil);
  }

}

DEFUN("HASH-SET",object,fShash_set,SI,3,3,NONE,OO,OO,OO,OO,(object x,object y,object z),"") {

  check_type_hash_table(&y);
  sethash(x,y,z);
  RETURN1(z);

}

DEFUN("COMPLEX",object,fLcomplex,LISP,1,2,NONE,OO,OO,OO,OO,(object r,...),"") {
  fixnum nargs=INIT_NARGS(1);
  object l=Cnil,f=OBJNULL,i;
  va_list ap;

  va_start(ap,r);
  i=NEXT_ARG(nargs,ap,l,f,make_fixnum(0));
  va_end(ap);

  check_type_or_rational_float(&r);
  check_type_or_rational_float(&i);

  RETURN1(make_complex(r,i));

}

DEFUN("FLOAT",object,fLfloat,LISP,1,2,NONE,OO,OO,OO,OO,(object x,...),"") {

  fixnum nargs=INIT_NARGS(1);
  object l=Cnil,f=OBJNULL,y;
  va_list ap;
  double d;
  enum type t;

  va_start(ap,x);
  y=NEXT_ARG(nargs,ap,l,f,(t=type_of(x))==t_shortfloat || t==t_longfloat ? x : make_longfloat(0.0));
  va_end(ap);

  /* check_type_float(&x); */
  check_type_float(&y);

  t=type_of(y);

  switch (type_of(x)) {
  case t_fixnum:
    if (t == t_shortfloat)
      x = make_shortfloat((shortfloat)(fix(x)));
    else
      x = make_longfloat((double)(fix(x)));
    break;

  case t_bignum:
  case t_ratio:
    d = number_to_double(x);
    if (t == t_shortfloat)
      x = make_shortfloat((shortfloat)d);
    else
      x = make_longfloat(d);
    break;

  case t_shortfloat:
    if (t == t_longfloat)
      x = make_longfloat((double)(sf(x)));
    break;

  case t_longfloat:
    if (t == t_shortfloat)
      x = make_shortfloat((shortfloat)(lf(x)));
    break;

  default:
    FEwrong_type_argument(TSor_rational_float, x);
  }

  RETURN1(x);

}


#ifndef NO_BOOT_H
#include "boot.h"
#endif
