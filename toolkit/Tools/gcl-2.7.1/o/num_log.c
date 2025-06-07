
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
	Logical operations on number
*/
#define NEED_MP_H
#define EXPORT_GMP
#include "include.h"
#include "num_include.h"

   
#ifdef GMP
#include "gmp_num_log.c"
#else
#include "pari_num_log.c"
#endif


inline object
fixnum_big_shift(fixnum x,fixnum w) {
  MPOP(return,shifti,SI_TO_MP(x,big_fixnum1),w);
}

inline object
integer_fix_shift(object x, fixnum w) { 
  if (type_of(x)==t_fixnum) {
    fixnum fx=fix(x);
    return (fx!=MOST_NEGATIVE_FIX || w<0) ? fixnum_shft(fx,w) : fixnum_big_shift(fx,w);
  }
  MPOP(return,shifti,MP(x),w);
}
	
inline object
integer_shift(object x,object y) {
  enum type tx=type_of(x),ty=type_of(y);
  if (ty==t_fixnum)
    return integer_fix_shift(x,fix(y));
  else {
    if (eql(x,make_fixnum(0)))
      return x;
    if (big_sign(y)<0)
      return make_fixnum((tx==t_fixnum ? fix(x) : big_sign(x))<0 ? -1 : 0);
    FEerror("Insufficient memory",0);
    return Cnil;
  }
}
      
inline bool
integer_bitp(object p,object x) {
  enum type tp=type_of(p),tx=type_of(x);

  if (tp==t_fixnum) {
    if (tx==t_fixnum)
      return fixnum_bitp(fix(p),fix(x));
    else 
      return big_bitp(x,fix(p));
  } else if (big_sign(p)<0)
    return 0;
  else if (tx==t_fixnum)/*fixme integer_minusp*/
    return fix(x)<0;
  else return big_sign(x)<0;
}

inline object
integer_length(object x) {
  return make_fixnum(type_of(x)==t_fixnum ? fixnum_length(fix(x)) : MP_SIZE_IN_BASE2(MP(x)));
}

inline object
integer_count(object x) {
  return make_fixnum(type_of(x)==t_fixnum ? fixnum_count(fix(x)) : MP_BITCOUNT(MP(x)));
}

#define DEFLOG(n_,a_,b_,c_)						\
DEFUN(n_,object,Join(fL,a_),LISP,0,63,NONE,OO,OO,OO,OO,(object first,...),"") { \
  fixnum nargs=INIT_NARGS(0),fx=0;					\
  object l=Cnil,x,y;							\
  enum type tx,ty;							\
  va_list ap;								\
									\
  va_start(ap,first);							\
  x=NEXT_ARG(nargs,ap,l,first,c_);					\
  if ((tx=type_of(x))==t_fixnum) {fx=fix(x);x=OBJNULL;}			\
  for (;(y=NEXT_ARG(nargs,ap,l,first,OBJNULL))!=OBJNULL;) {		\
    ty=type_of(y);							\
    if (tx==t_fixnum&&ty==t_fixnum)					\
      fx=fixnum_log_op2(b_,fx,fix(y));					\
    else {								\
      x=normalize_big(integer_log_op2(b_,x==OBJNULL ? make_fixnum(fx) : x,tx,y,ty)); \
      if ((tx=type_of(x))==t_fixnum) {fx=fix(x);x=OBJNULL;}		\
    }									\
  }									\
  va_end(ap);								\
  return x==OBJNULL ? make_fixnum(fx) : maybe_replace_big(x);		\
}									\

DEFLOG("LOGIOR",logior,BOOLIOR,small_fixnum(0));
DEFLOG("LOGXOR",logxor,BOOLXOR,small_fixnum(0));
DEFLOG("LOGAND",logand,BOOLAND,small_fixnum(-1));
DEFLOG("LOGEQV",logeqv,BOOLEQV,small_fixnum(-1));



/* #define IF1(a_) BF(1,a_,f,b) */

/* IF1(bitcount) */
/* IF1(popcount) */
/* IF1(bitlength) */
/* BF(2,sizeinbase,f,b,f) */
/* IF1(get_si) */
/* IF1(get_ui) */
/* IF1(sgn) */

/* BF(1,fac_ui,b,f) */
/* BF(1,fib_ui,b,f) */

/* BF(3,powm,b,b,b,b) */
/* BF(3,powm_ui,b,b,f,b) */
/* BF(2,tdiv_qr,m,b,b) */

/* #define BF1(a_) BF(1,a_,b,b) */

/* BF1(com) */
/* BF1(sqrt) */
/* BF1(neg) */

/* BF(2,cmp,f,b,b) */

/* #define BF2(a_) BF(2,a_,b,b,b) */

/* BF2(invert) */
/* BF2(remove) */
/* BF2(add) */
/* BF2(mul) */
/* BF2(sub) */
/* BF2(and) */
/* BF2(ior) */
/* BF2(xor) */
/* BF2(gcd) */
/* BF2(lcm) */
/* BF2(divexact) */

/* BF(2,tstbit,f,b,f) */
/* BF(2,jacobi,f,b,b) */

/* #define BF2I(a_) BF(2,a_,b,b,f) */

/* BF2I(root) */
/* BF2I(divexact_ui) */
/* BF2I(gcd_ui) */
/* BF2I(bin_ui) */
/* BF2I(lcm_ui) */
/* BF2I(sub_ui) */
/* BF2I(add_ui) */
/* BF2I(mul_ui) */
/* BF2I(mul_si) */
/* BF2I(mul_2exp) */
/* BF2I(fdiv_q_2exp) */


#define BI(n_)\
  DEFUN(#n_,object,Join(fS,n_),SI,1,1,NONE,II,OO,OO,OO,(fixnum x),"") {\
\
    RETURN1((object)(fixnum)Join(__builtin_,n_)(x));	\
\
}

BI(clzl)
BI(ctzl)
BI(ffsl)
BI(parityl)
BI(popcountl)


DEFUN("SHFT",object,fSshft,SI,2,2,NONE,OO,IO,OO,OO,(object x,fixnum y),"") {

  object u=new_bignum();

  ENSURE_MP(x,1);
  shifti(MP(u),MP(x),y);
  RETURN1(normalize_big(u));

}


DEFUN("LOGCB1",object,fSlogcb1,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  object u=new_bignum();

  ENSURE_MP(x,1);
  mpz_com(MP(u),MP(x));
  RETURN1(normalize_big(u));

}

#define B2OP(n_,b_)						\
DEFUN(#n_ "B2",object,Join(Join(fSlog,n_),b2),SI,3,3,NONE,OO,OO,OO,OO,(object x,object y,object z),"") { \
\
  object u=new_bignum();\
\
  ENSURE_MP(x,1);\
  ENSURE_MP(y,2);\
  Join(mpz_,b_)(MP(u),MP(x),MP(y));\
  if (z!=Cnil) mpz_com(MP(u),MP(u));\
  RETURN1(normalize_big(u));\
\
}

B2OP(AND,and)
B2OP(IOR,ior)
B2OP(XOR,xor)


DEFUN("BOOLE",object,fLboole,LISP,3,3,NONE,OO,OO,OO,OO,(object o,object x,object y),"") {
  check_type_integer(&o);
  check_type_integer(&x);
  check_type_integer(&y);
  RETURN1(log_op2(fixint(o),x,y));
}


DEFUN("ASH",object,fLash,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  check_type_integer(&x);
  check_type_integer(&y);
  RETURN1(integer_shift(x,y));

}

DEFUN("LOGBITP",object,fLlogbitp,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  check_type_integer(&x);
  check_type_integer(&y);
  RETURN1(integer_bitp(x,y)?Ct:Cnil);

}

DEFUN("LOGCOUNT",object,fLlogcount,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  check_type_integer(&x);
  RETURN1(integer_count(x));

}

DEFUN("INTEGER-LENGTH",object,fLloglength,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  check_type_integer(&x);
  RETURN1(integer_length(x));

}


void
gcl_init_num_log(void)
{
/*  	int siLbit_array_op(void); */

	make_constant("BOOLE-CLR", make_fixnum(BOOLCLR));
	make_constant("BOOLE-SET", make_fixnum(BOOLSET));
	make_constant("BOOLE-1", make_fixnum(BOOL1));
	make_constant("BOOLE-2", make_fixnum(BOOL2));
	make_constant("BOOLE-C1", make_fixnum(BOOLC1));
	make_constant("BOOLE-C2", make_fixnum(BOOLC2));
	make_constant("BOOLE-AND", make_fixnum(BOOLAND));
	make_constant("BOOLE-IOR", make_fixnum(BOOLIOR));
	make_constant("BOOLE-XOR", make_fixnum(BOOLXOR));
	make_constant("BOOLE-EQV", make_fixnum(BOOLEQV));
	make_constant("BOOLE-NAND", make_fixnum(BOOLNAND));
	make_constant("BOOLE-NOR", make_fixnum(BOOLNOR));
	make_constant("BOOLE-ANDC1", make_fixnum(BOOLANDC1));
	make_constant("BOOLE-ANDC2", make_fixnum(BOOLANDC2));
	make_constant("BOOLE-ORC1", make_fixnum(BOOLORC1));
	make_constant("BOOLE-ORC2", make_fixnum(BOOLORC2));


	sLbit = make_ordinary("BIT");
}

