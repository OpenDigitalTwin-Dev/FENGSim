#include "compbas2.h"
#include "funlink.h"

extern object sLvalues,sLinteger,sLfixnum,big_fixnum1,big_fixnum2,big_fixnum3,big_fixnum4,big_fixnum5;

#define ENSURE_MP(a_,b_) \
  if (type_of(a_)==t_fixnum) {\
  mpz_set_si(MP(Join(big_fixnum,b_)),fix(a_));\
  a_=Join(big_fixnum,b_);\
  }

#define K_bbb_b OO
#define K_bbb_f OI
#define K_bb_b OO
#define K_bb_f OI
#define K_fb_b IO
#define K_fb_f II
#define K_fbb_b IO
#define K_fbb_f II
#define K_m_b OO
#define K_m_f OI
#define K_b_b OO
#define K_f_b IO
#define K_b_f OI
#define K_f_f II
#define KK1(a_,b_) Join(K_,Join(a_,Join(_,b_)))

#define Q11(a_,b_) KK1(a_,b_)
#define Q21(a_,b_) OO
#define Q31(a_,b_) OO
#define Q41(a_,b_) OO
#define Q12(a_,b_,c_) KK1(a_,b_)
#define Q22(a_,b_,c_) KK1(c_,b)
#define Q32(a_,b_,c_) OO
#define Q42(a_,b_,c_) OO
#define Q13(a_,b_,c_,d_) KK1(a_,b_)
#define Q23(a_,b_,c_,d_) KK1(c_,d_)
#define Q33(a_,b_,c_,d_) OO
#define Q43(a_,b_,c_,d_) OO
#define Q14(a_,b_,c_,d_,e_) KK1(a_,b_)
#define Q24(a_,b_,c_,d_,e_) KK1(c_,d_)
#define Q34(a_,b_,c_,d_,e_) KK1(e_,b)
#define Q44(a_,b_,c_,d_,e_) OO
#define Q15(a_,b_,c_,d_,e_,f_) KK1(a_,b_)
#define Q25(a_,b_,c_,d_,e_,f_) KK1(c_,d_)
#define Q35(a_,b_,c_,d_,e_,f_) KK1(e_,f_)
#define Q45(a_,b_,c_,d_,e_,f_) OO

/* #define QR11(a_,b_) KK1(a_,b_) */
/* #define QR21(a_,b_) OO */
/* #define QR31(a_,b_) OO */
/* #define QR41(a_,b_) OO */
/* #define QR12(a_,b_,c_) KK1(a_,b_) */
/* #define QR22(a_,b_,c_) KK1(c_,b) */
/* #define QR32(a_,b_,c_) OO */
/* #define QR42(a_,b_,c_) OO */
/* #define QR13(a_,b_,c_,d_) KK1(a_,b_) */
/* #define QR23(a_,b_,c_,d_) KK1(c_,d_) */
/* #define QR33(a_,b_,c_,d_) OO */
/* #define QR43(a_,b_,c_,d_) OO */
/* #define QR14(a_,b_,c_,d_,e_) KK1(a_,b_) */
/* #define QR24(a_,b_,c_,d_,e_) KK1(c_,d_) */
/* #define QR34(a_,b_,c_,d_,e_) KK1(e_,b) */
/* #define QR44(a_,b_,c_,d_,e_) OO */

/* #define QR_fb(a_...)   fb,b,a_ */
/* #define QR_fbb(a_...)  fbb,b,b,a_ */
/* #define QR_bb(a_...)   bb,b,b,a_ */
/* #define QR_bbb(a_...)  bbb,b,b,b,a_ */
/* #define QR_m(a_...)    m,b,b,a_ */
/* #define QR_b(a_...)    b,b,a_ */
/* #define QR_f(a_...)    f,a_ */
/* #define QR(r_,a_...)   Join(QR_,r_)(a_) */
/* #define QRR(e_,n_,a_...) Join(Join(Q,e_),n_)(a_) */


#define D_fb   fixnum
#define D_fbb  fixnum
#define D_bb   object
#define D_bbb  object
#define D_m    object
#define D_b    object
#define D_f    fixnum
#define D0(a_) Join(D_,a_)
#define D1(a_) D0(a_) x
#define D2(a_,b_) D1(a_),D0(b_) y
#define D3(a_,b_,c_) D2(a_,b_),D0(c_) z
#define D4(a_,b_,c_,d_) D3(a_,b_,c_),D0(d_) w

#define R1(a_) object /*D0(a_)*/

#define EE(a_,b_)
#define E_b ENSURE_MP
#define E_f EE
#define E1(a_) Join(E_,a_)(x,1);
#define E2(a_,b_) E1(a_) Join(E_,b_)(y,2)
#define E3(a_,b_,c_) E2(a_,b_) Join(E_,c_)(z,3)
#define E4(a_,b_,c_,d_) E3(a_,b_,c_) Join(E_,d_)(w,4)

/* #define AA_m object *vals=(object *)fcall.valp,*base=vs_top,u=new_bignum(),v=new_bignum() */
/* #define AA_b object u=new_bignum() */
#define AA_bbb object *vals=(object *)fcall.valp,*base=vs_top,u=big_fixnum3,v=big_fixnum4,v2=big_fixnum5
#define AA_bb object *vals=(object *)fcall.valp,*base=vs_top,u=big_fixnum4,v=big_fixnum5
#define AA_fb fixnum u;object *vals=(object *)fcall.valp,*base=vs_top,v=big_fixnum4
#define AA_fbb fixnum u;object *vals=(object *)fcall.valp,*base=vs_top,v=big_fixnum4,v2=big_fixnum5
#define AA_m object *vals=(object *)fcall.valp,*base=vs_top,u=big_fixnum4,v=big_fixnum5
#define AA_b object u=big_fixnum4
#define AA_f fixnum u
#define AA1(a_) Join(AA_,a_)

/* #define AAR_bbb */
/* #define AAR_bb */
/* #define AAR_fb fixnum u */
/* #define AAR_fbb fixnum u */
/* #define AAR_m */
/* #define AAR_b */
/* #define AAR_f fixnum u */
/* #define AAR1(a_) Join(AAR_,a_) */

/* #define CR_b */
/* #define CR_f */
/* #define CR1(a_) Join(CR_,a_)(x) */
/* #define CR2(a_,b_) CR1(a_),Join(CR_,b_)(y) */
/* #define CR3(a_,b_,c_) CR2(a_,b_),Join(CR_,c_)(z) */
/* #define CR4(a_,b_,c_,d_) CR3(a_,b_,c_),Join(CR_,d_)(w) */

#define C_b MP
#define C_f
#define C1(a_) Join(C_,a_)(x)
#define C2(a_,b_) C1(a_),Join(C_,b_)(y)
#define C3(a_,b_,c_) C2(a_,b_),Join(C_,c_)(z)
#define C4(a_,b_,c_,d_) C3(a_,b_,c_),Join(C_,d_)(w)

#define CC_bbb MP(u),MP(v),MP(v2),
#define CC_bb  MP(u),MP(v),
#define CC_fb  MP(v),
#define CC_fbb MP(v),MP(v2),
#define CC_m   MP(u),MP(v),
#define CC_b   MP(u),
#define CC_f
#define CC1(r_) Join(CC_,r_)

/* #define CCR_bbb u,v,v2, */
/* #define CCR_bb  u,v, */
/* #define CCR_fb  v, */
/* #define CCR_fbb v,v, */
/* #define CCR_m   u,v, */
/* #define CCR_b   u, */
/* #define CCR_f */
/* #define CCR1(r_) Join(CCR_,r_) */

/* #define DR_bbb object u,object v,object v2, */
/* #define DR_bb  object u,object v, */
/* #define DR_fb  object v, */
/* #define DR_fbb object v,object v2, */
/* #define DR_m   object u,object v, */
/* #define DR_b   object u, */
/* #define DR_f */
/* #define DR1(r_) Join(DR_,r_) */

#define W_bbb
#define W_bb
#define W_fb  u=
#define W_fbb u=
#define W_m
#define W_b
#define W_f   u=

/* #define WR_bbb */
/* #define WR_bb */
/* #define WR_fb  fixnum u= */
/* #define WR_fbb fixnum u= */
/* #define WR_m */
/* #define WR_b */
/* #define WR_f   fixnum u= */

/* #define Z_m normalize_big(u),normalize_big(v) */
/* #define Z_b normalize_big(u) */
#define Z_bbb maybe_replace_big(u),maybe_replace_big(v),maybe_replace_big(v2)
#define Z_bb  maybe_replace_big(u),maybe_replace_big(v)
#define Z_fb  (object)u,maybe_replace_big(v)
#define Z_fbb (object)u,maybe_replace_big(v),maybe_replace_big(v2)
#define Z_m   maybe_replace_big(u),maybe_replace_big(v)
#define Z_b   maybe_replace_big(u)
#define Z_f   (object)u

#define PT_bb MMcons(sLvalues,MMcons(sLinteger,MMcons(sLinteger,Cnil)))
#define PT_fb MMcons(sLvalues,MMcons(sLfixnum,MMcons(sLinteger,Cnil)))
#define PT_fbb MMcons(sLvalues,MMcons(sLfixnum,MMcons(sLinteger,MMcons(sLinteger,Cnil))))
#define PT_bbb MMcons(sLvalues,MMcons(sLinteger,MMcons(sLinteger,MMcons(sLinteger,Cnil))))
#define PT_m MMcons(sLvalues,MMcons(sLinteger,MMcons(sLinteger,Cnil)))
#define PT_b sLinteger
#define PT_f sLfixnum
#define PT(a_) Join(PT_,a_)
#define PT1(a_) MMcons(Join(PT_,a_),Cnil)
#define PT2(a_,b_) MMcons(PT1(a_),PT1(b_))
#define PT3(a_,b_,c_) MMcons(PT1(a_),PT2(b_,c_))
#define PT4(a_,b_,c_,d_) MMcons(PT1(a_),PT3(b_,c_,d_))

/* #define PTR_bb sLinteger */
/* #define PTR_fb sLfixnum */
/* #define PTR_fbb sLfixnum */
/* #define PTR_bbb sLinteger */
/* #define PTR_m sLinteger */
/* #define PTR_b sLinteger */
/* #define PTR_f sLfixnum */
/* #define PTR(a_) Join(PTR_,a_) */
/* #define PTR1(a_) MMcons(Join(PTR_,a_),Cnil) */
/* #define PTR2(a_,b_) MMcons(PTR1(a_),PTR1(b_)) */
/* #define PTR3(a_,b_,c_) MMcons(PTR1(a_),PTR2(b_,c_)) */
/* #define PTR4(a_,b_,c_,d_) MMcons(PTR1(a_),PTR3(b_,c_,d_)) */

#define HH_bbb(a_...) RETURN3(a_)
#define HH_bb(a_...)  RETURN2(a_)
#define HH_fb(a_...)  RETURN2(a_)
#define HH_fbb(a_...) RETURN3(a_)
#define HH_m(a_...)   RETURN2(a_)
#define HH_b(a_...)   RETURN1(a_)
#define HH_f(a_...)   RETURN1(a_)

/* #define BF1(n_,b_,r_,a_...)						\ */
/*   DEFUNB("mpz_" #b_,R1(r_),Join(fSmpz_,b_),				\ */
/* 	 GMP,n_,n_,NONE,Join(Q1,n_)(r_,a_),Join(Q2,n_)(r_,a_),Join(Q3,n_)(r_,a_),Join(Q4,n_)(r_,a_), \ */
/* 	 (Join(D,n_)(a_)),PT(r_),"") {					\ */
/*   									\ */
/*     AA1(r_);								\ */
/*     									\ */
/*     Join(E,n_)(a_);							\ */
/*     Join(W_,r_) Join(fSmmpz_,b_)(CCR1(r_)Join(CR,n_)(a_));		\ */
/*     Join(HH_,r_)(Join(Z_,r_));						\ */
/*     									\ */
/*   } */


/* Do not expose big_fixnum registers at lisp level, as typing is undefined */
/* #define BF(n_,m_,s_,b_,r_,a_...)					\ */
/*   DEFUNB("mmpz_" #b_,R1(r_),Join(fSmmpz_,b_),				\ */
/* 	 GMP,n_,n_,NONE,						\ */
/* 	 QRR(1,n_,QR(r_,a_)),QRR(2,n_,QR(r_,a_)),			\ */
/* 	 QRR(3,n_,QR(r_,a_)),QRR(4,n_,QR(r_,a_)),			\ */
/* 	 (DR1(r_)Join(D,m_)(a_)),PTR(r_),"") {				\ */
/*   									\ */
/*     Join(WR_,r_) Join(m__gmpz_,b_)(CC1(r_)Join(C,m_)(a_));		\ */
/*     RETURN1(u);								\ */
/*     									\ */
/*   }									\ */
/*   BF1(m_,b_,r_,a_); */


#define BF(n0_,n_,s_,b_,r_,a_...)					\
  DEFUNB("mpz_" #b_,R1(r_),Join(fSmpz_,b_),				\
	 GMP,n_,n_,NONE,Join(Q1,n_)(r_,a_),Join(Q2,n_)(r_,a_),Join(Q3,n_)(r_,a_),Join(Q4,n_)(r_,a_), \
	 (Join(D,n_)(a_)),PT(r_),"") {					\
  									\
    AA1(r_);								\
    									\
    Join(E,n_)(a_);							\
    Join(W_,r_) Join(m__gmpz_,b_)(CC1(r_)Join(C,n_)(a_));		\
    Join(HH_,r_)(Join(Z_,r_));						\
    									\
  }
