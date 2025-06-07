#ifndef IMMNUM_H
#define IMMNUM_H

#include "fixnum.h"

#if defined (LOW_SHFT)
#define is_imm_fixnum2(x_,y_) is_unmrkd_imm_fixnum(x_)&&is_unmrkd_imm_fixnum(y_)
#define is_imm_fixnum3(x_,y_,z_) is_unmrkd_imm_fixnum(x_)&&is_unmrkd_imm_fixnum(y_)&&is_unmrkd_imm_fixnum(z_)
#define fimoff  0
#else
#define is_imm_fixnum2(x_,y_) is_imm_fixnum(((ufixnum)x_)&((ufixnum)y_))
#define is_imm_fixnum3(x_,y_,z_) is_imm_fixnum(((ufixnum)x_)&((ufixnum)y_)&((ufixnum)z_))
#define fimoff  (IM_FIX_BASE+(IM_FIX_LIM>>1))
#endif

#define mif(x)    make_imm_fixnum(x)/*abbreviations*/
#define fif(x)    fix_imm_fixnum(x)
#define iif(x)    is_imm_fixnum(x)
#define iif2(x,y) is_imm_fixnum2(x,y)


INLINE fixnum
lnabs(fixnum x) {return x<0 ? ~x : x;}

INLINE char
clz(ufixnum x) {
#ifdef HAVE_CLZL
  return x ? __builtin_clzl(x) : sizeof(x)*8;
#else
  {char i;for (i=0;i<sizeof(x)*8 && !((x>>(sizeof(x)*8-1-i))&0x1);i++); return i;}
#endif
}

INLINE char
ctz(ufixnum x) {
#ifdef HAVE_CTZL
  return __builtin_ctzl(x);/*x ? __builtin_clzl(x) : sizeof(x)*8;*/
#else
  {char i;for (i=0;i<sizeof(x)*8 && !((x>>i)&0x1);i++); return i;}
#endif
}

INLINE char
fixnum_length(fixnum x) {return sizeof(x)*8-clz(lnabs(x));}

INLINE object
immnum_length(object x) {return iif(x) ? mif((fixnum)fixnum_length(fif(x))) : integer_length(x);}


#if SIZEOF_LONG == 8
#define POPA 0x5555555555555555UL
#define POPB 0x3333333333333333UL
#define POPC 0x0F0F0F0F0F0F0F0FUL
#define POPD 0x7F
#else
#define POPA 0x55555555UL
#define POPB 0x33333333UL
#define POPC 0x0F0F0F0FUL
#define POPD 0x3F
#endif

INLINE char
fixnum_popcount(ufixnum x) {
  x-=POPA&(x>>1);
  x=(x&POPB)+((x>>2)&POPB);
  x=POPC&(x+(x>>4));
  x+=x>>8;
  x+=x>>16;
#if SIZEOF_LONG == 8
  x+=x>>32;
#endif
  return x&POPD;
}

INLINE char
/* fixnum_count(fixnum x) {return __builtin_popcountl(lnabs(x));} */
fixnum_count(fixnum x) {return fixnum_popcount(lnabs(x));}

INLINE object
immnum_count(object x) {return iif(x) ? mif((fixnum)fixnum_count(fif(x))) : integer_count(x);}

/*bs=sizeof(long)*8;
  lb=bs-clz(labs(x));|x*y|=|x|*|y|<2^(lbx+lby)<2^(bs-1);
  0 bounded by 2^0, +-1 by 2^1,mpf by 2^(bs-1), which is sign bit
  protect labs from most negative fix, here all immfix ok*/
long int labs(long int j);
INLINE bool
fixnum_mul_safe_abs(fixnum x,fixnum y) {return clz(x)+clz(y)>sizeof(x)*8+1;}
INLINE object
safe_mul_abs(fixnum x,fixnum y) {return fixnum_mul_safe_abs(x,y) ? make_fixnum(x*y) : fixnum_times(x,y);}
INLINE bool
fixnum_mul_safe(fixnum x,fixnum y) {return fixnum_mul_safe_abs(labs(x),labs(y));}
INLINE object
safe_mul(fixnum x,fixnum y) {return fixnum_mul_safe(x,y) ? make_fixnum(x*y) : fixnum_times(x,y);}
INLINE object
immnum_times(object x,object y) {return iif2(x,y) ? safe_mul(fif(x),fif(y)) : number_times(x,y);}

INLINE object
immnum_plus(object x,object y) {return iif2(x,y) ? make_fixnum(fif(x)+fif(y)) : number_plus(x,y);}
INLINE object
immnum_minus(object x,object y) {return iif2(x,y) ? make_fixnum(fif(x)-fif(y)) : number_minus(x,y);}
INLINE object
immnum_negate(object x) {return iif(x) ? make_fixnum(-fif(x)) : number_negate(x);}

#define BOOLCLR		0
#define BOOLSET		017
#define BOOL1		03
#define BOOL2		05
#define BOOLC1		014
#define BOOLC2		012
#define BOOLAND		01
#define BOOLIOR		07
#define BOOLXOR		06
#define BOOLEQV		011
#define BOOLNAND	016
#define BOOLNOR		010
#define BOOLANDC1	04
#define BOOLANDC2	02
#define BOOLORC1	015
#define BOOLORC2	013

INLINE fixnum
fixnum_boole(fixnum op,fixnum x,fixnum y) {
  switch(op) {
  case BOOLCLR:	 return 0;
  case BOOLSET:	 return -1;
  case BOOL1:	 return x;
  case BOOL2:	 return y;
  case BOOLC1:	 return ~x;
  case BOOLC2:	 return ~y;
  case BOOLAND:	 return x&y;
  case BOOLIOR:	 return x|y;
  case BOOLXOR:	 return x^y;
  case BOOLEQV:	 return ~(x^y);
  case BOOLNAND: return ~(x&y);
  case BOOLNOR:	 return ~(x|y);
  case BOOLANDC1:return ~x&y;
  case BOOLANDC2:return x&~y;
  case BOOLORC1: return ~x|y;
  case BOOLORC2: return x|~y;
  } 
  return 0;/*FIXME error*/
}
  
INLINE object
immnum_boole(fixnum o,object x,object y) {return iif2(x,y) ? mif(fixnum_boole(o,fif(x),fif(y))) : log_op2(o,x,y);}

#define immnum_bool(o,x,y) immnum_boole(fixint(o),x,y)

#define immnum_ior(x,y)   immnum_boole(BOOLIOR,x,y)
#define immnum_and(x,y)   immnum_boole(BOOLAND,x,y)
#define immnum_xor(x,y)   immnum_boole(BOOLXOR,x,y)
#define immnum_not(x)     immnum_boole(BOOLC1,x,x)
#define immnum_nand(x,y)  immnum_boole(BOOLNAND,x,y)
#define immnum_nor(x,y)   immnum_boole(BOOLNOR,x,y)
#define immnum_eqv(x,y)   immnum_boole(BOOLEQV,x,y)
#define immnum_andc1(x,y) immnum_boole(BOOLANDC1,x,y)
#define immnum_andc2(x,y) immnum_boole(BOOLANDC2,x,y)
#define immnum_orc1(x,y)  immnum_boole(BOOLORC1,x,y)
#define immnum_orc2(x,y)  immnum_boole(BOOLORC2,x,y)

INLINE fixnum
fixnum_div(fixnum x,fixnum y,fixnum d) {
  fixnum z=x/y;
  if (d && x!=y*z && (x*d>0 ? y>0 : y<0))
    z+=d;
  return z;
}
INLINE fixnum
fixnum_rem(fixnum x,fixnum y,fixnum d) {
  fixnum z=x%y;
  if (d && z && (x*d>0 ? y>0 : y<0))
    z+=y;
  return z;
}
INLINE object
immnum_truncate(object x,object y) {return iif2(x,y)&&y!=make_fixnum(0) ? mif(fixnum_div(fif(x),fif(y),0)) : (intdivrem(x,y,0,&x,0),x);}
INLINE object
immnum_floor(object x,object y) {return iif2(x,y)&&y!=make_fixnum(0) ? mif(fixnum_div(fif(x),fif(y),-1)) : (intdivrem(x,y,-1,&x,0),x);}
INLINE object
immnum_ceiling(object x,object y) {return iif2(x,y)&&y!=make_fixnum(0) ? mif(fixnum_div(fif(x),fif(y),1)) : (intdivrem(x,y,1,&x,0),x);}
INLINE object
immnum_mod(object x,object y) {return iif2(x,y)&&y!=make_fixnum(0) ? mif(fixnum_rem(fif(x),fif(y),-1)) : (intdivrem(x,y,-1,0,&y),y);}
INLINE object
immnum_rem(object x,object y) {return iif2(x,y)&&y!=make_fixnum(0) ? mif(fixnum_rem(fif(x),fif(y),0)) : (intdivrem(x,y,0,0,&y),y);}

INLINE fixnum
fixnum_rshft(fixnum x,fixnum y) {
  return y>=sizeof(x)*8 ? (x<0 ? -1 : 0) : x>>y;
}
INLINE object
fixnum_lshft(fixnum x,fixnum y) {
  return clz(labs(x))>y ? make_fixnum(x<<y) : (x ? fixnum_big_shift(x,y) : make_fixnum(0));
}
INLINE object
fixnum_shft(fixnum x,fixnum y) {
  return y<0 ? make_fixnum(fixnum_rshft(x,-y)) : fixnum_lshft(x,y);
}
INLINE object
immnum_shft(object x,object y) {return iif2(x,y) ? fixnum_shft(fif(x),fif(y)) : integer_shift(x,y);}

INLINE bool
fixnum_bitp(fixnum p,fixnum x) {return fixnum_rshft(x,p)&0x1;}

INLINE bool
immnum_bitp(object x,object y) {return iif2(x,y) ? fixnum_bitp(fif(x),fif(y)) : integer_bitp(x,y);}

#define immnum_comp(x,y,c) iif2(x,y) ? ((fixnum)x c (fixnum)y) : (number_compare(x,y) c 0)

INLINE bool
immnum_lt(object x,object y) {return immnum_comp(x,y,<);}
INLINE bool
immnum_le(object x,object y) {return immnum_comp(x,y,<=);}
INLINE bool
immnum_eq(object x,object y) {return immnum_comp(x,y,==);}
INLINE bool
immnum_ne(object x,object y) {return immnum_comp(x,y,!=);}
INLINE bool
immnum_gt(object x,object y) {return immnum_comp(x,y,>);}
INLINE bool
immnum_ge(object x,object y) {return immnum_comp(x,y,>=);}

INLINE bool
immnum_minusp(object x) {return iif(x) ? ((fixnum)x)<((fixnum)make_fixnum(0)) : number_minusp(x);}
INLINE bool
immnum_plusp(object x) {return iif(x) ? ((fixnum)x)>((fixnum)make_fixnum(0)) : number_plusp(x);}
INLINE bool
immnum_zerop(object x) {return iif(x) ? ((fixnum)x)==((fixnum)make_fixnum(0)) : number_zerop(x);}
INLINE bool
immnum_evenp(object x) {return iif(x) ? !(((fixnum)x)&0x1) : number_evenp(x);}
INLINE bool
immnum_oddp(object x) {return iif(x) ? (((fixnum)x)&0x1) : number_oddp(x);}

INLINE object
immnum_signum(object x) {
  fixnum ux=(fixnum)x,uz=((fixnum)make_fixnum(0));
  return iif(x) ? (ux<uz ? mif(-1) : (ux==uz ? mif(0) : mif(1))) : number_signum(x);
}
INLINE object
immnum_abs(object x) {return iif(x) ? make_fixnum(labs(fif(x))) : number_abs(x);}

INLINE fixnum
fixnum_ldb(fixnum s,fixnum p,fixnum i) {
  return ((1UL<<s)-1)&fixnum_rshft(i,p);
}

INLINE object
immnum_ldb(object x,object i) {
  if (iif(i))
    if (consp(x)) {
      object s=x->c.c_car,p=x->c.c_cdr;
      if (iif2(s,p)) {
	fixnum fs=fif(s),fp=fif(p);
	if (fs+fp<sizeof(fs)*8)
	  return make_fixnum(fixnum_ldb(fs,fp,fif(i)));
      }
    }
  return number_ldb(x,i);
}

INLINE bool
immnum_ldbt(object x,object i) {
  if (iif(i))
    if (consp(x)) {
      object s=x->c.c_car,p=x->c.c_cdr;
      if (iif2(s,p)) {
	fixnum fs=fif(s),fp=fif(p);
	if (fs+fp<sizeof(fs)*8)
	  return fixnum_ldb(fs,fp,fif(i));
      }
    }
  return number_ldbt(x,i)!=Cnil;
}

INLINE fixnum
fixnum_dpb(fixnum s,fixnum p,fixnum n,fixnum i) {
  fixnum z=(1UL<<s)-1;
  return (i&~(z<<p))|((n&z)<<p);
}

INLINE object
immnum_dpb(object n,object x,object i) {
  if (iif2(n,i))
    if (consp(x)) {
      object s=x->c.c_car,p=x->c.c_cdr;
      if (iif2(s,p)) {
	fixnum fs=fif(s),fp=fif(p);
	if (fs+fp<sizeof(fs)*8)
	  return make_fixnum(fixnum_dpb(fs,fp,fif(n),fif(i)));
      }
    }
  return number_dpb(n,x,i);
}

INLINE fixnum
fixnum_dpf(fixnum s,fixnum p,fixnum n,fixnum i) {
  fixnum z=((1UL<<s)-1)<<p;
  return (i&~z)|(n&z);
}

INLINE object
immnum_dpf(object n,object x,object i) {
  if (iif2(n,i))
    if (consp(x)) {
      object s=x->c.c_car,p=x->c.c_cdr;
      if (iif2(s,p)) {
	fixnum fs=fif(s),fp=fif(p);
	if (fs+fp<sizeof(fs)*8)
	  return make_fixnum(fixnum_dpf(fs,fp,fif(n),fif(i)));
      }
    }
  return number_dpf(n,x,i);
}

INLINE object
immnum_max(object x,object y) {return iif2(x,y) ? ((fixnum)x>=(fixnum)y ? x : y) : (number_compare(x,y)>=0?x:y);}
INLINE object
immnum_min(object x,object y) {return iif2(x,y) ? ((fixnum)x<=(fixnum)y ? x : y) : (number_compare(x,y)<=0?x:y);}

INLINE bool
immnum_logt(object x,object y) {return iif2(x,y) ? fixnum_boole(BOOLAND,fif(x),fif(y))!=0 : !number_zerop(log_op2(BOOLAND,x,y));}

INLINE fixnum
fixnum_gcd(fixnum x,fixnum y) {

  fixnum t;
  char tx,ty;
  
  if (!x) return y;
  if (!y) return x;

  tx=ctz(x);
  ty=ctz(y);
  tx=tx<ty ? tx : ty;
  x>>=tx;
  y>>=tx;
  t=x&0x1 ? -y : x>>1;
  do {
    t>>=ctz(t);
    if (t>0) x=t; else y=-t;
    t=x-y;
  } while (t);

  return x<<tx;

}

INLINE object
immnum_gcd(object x,object y) {return iif2(x,y) ? mif(fixnum_gcd(labs(fif(x)),labs(fif(y)))) : get_gcd(x,y);}

INLINE object
fixnum_lcm(fixnum x,fixnum y) {
  fixnum g=fixnum_gcd(x,y);
  return g ? safe_mul_abs(x,fixnum_div(y,g,0)) : make_fixnum(0);
}

INLINE object
immnum_lcm(object x,object y) {return iif2(x,y) ? fixnum_lcm(labs(fif(x)),labs(fif(y))) : get_lcm(x,y);}

#endif

