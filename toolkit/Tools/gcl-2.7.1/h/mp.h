

#ifdef GMP

#include "gmp.h"
/* define to show we included mp.h */
#define _MP_H  


#define MP_ALLOCATED(x) MP(x)->_mp_alloc
#define MP_SELF(x) MP(x)->_mp_d
#define MP_SIZE(x) MP(x)->_mp_size
#define MP_LIMB_SIZE sizeof(mp_limb_t)





#define MP(x) (&((x)->big.big_mpz_t))
#define MP_ASSIGN_OBJECT(u,x) (type_of(x) == t_bignum ? mpz_set(u,MP(x)) : mpz_set_si(u,fix(x)))

/* temporary holders to put fixnums in ... */

typedef struct
{ MP_INT mpz;
  mp_limb_t body;
} mpz_int;

/* for integers which are in the fixnum range, we allocate a temporary
   place in the stack which we use to convert this into an MP
*/   
#define SI_TEMP_DECL(w) mpz_int w
            
#define SI_TO_MP(x, temp) (mpz_set_si(MP(temp),(x)), MP(temp))
            

#define INTEGER_TO_MP(x, temp ) \
  (type_of(x) == t_bignum ? MP(x) : SI_TO_MP(fix(x), temp))

#define INTEGER_TO_TEMP_MP(x, temp ) \
  (type_of(x) == t_bignum ? (MP_ASSIGN_OBJECT(MP(temp),x),MP(temp)) : SI_TO_MP(fix(x), temp))

#define MPOP(action,function,x1,x2) \
  do {  \
   function(MP(big_fixnum1) ,x1,x2); \
   action maybe_replace_big(big_fixnum1); \
  } while(0)


#define MPOP_DEST(where,function,x1,x2) \
  do { extern MP_INT *verify_mp(); \
  function(MP(where),x1,x2); \
  verify_big_or_zero(where); \
      } while(0)


/* #define MYmake_fixnum(action,x) \ */
/*   do{register int CMPt1; \ */
/*    action \ */
/*    ((((CMPt1=(x))+1024)&-2048)==0?small_fixnum(CMPt1):make_fixnum1(CMPt1));}while(0) */
     
#define ineg(a_) (sizeof(a_)==sizeof(unsigned) ? (unsigned)-(a_) : (unsigned long)-(a_))

#define addii mpz_add
#define addsi(u,a,b) (a >= 0 ?  mpz_add_ui(u,b,a) : mpz_sub_ui(u,b,ineg(a)))
#define addss(u,a,b) addsi(u,a,SI_TO_MP(b,big_fixnum1))
	    
#define mulii mpz_mul
#define mulsi(u,s,i) mpz_mul_si(u,i,s)
#define mulss(u,s1,s2) mpz_mul_si(u,SI_TO_MP(s1,big_fixnum1),s2)
	    
#define subii mpz_sub
#define subsi(u,a,b) mpz_sub(u,SI_TO_MP(a,big_fixnum1),b)
#define subis(u,a,b) (b >= 0 ?  mpz_sub_ui(u,a,b) : mpz_add_ui(u,a,ineg(b)))
#define subss(u,a,b) subis(u,SI_TO_MP(a,big_fixnum1),b)
#define shifti(u,a,w) (w>=0 ? mpz_mul_2exp(u,a,w) : mpz_fdiv_q_2exp(u,a,ineg(w)))




#define cmpii(a,b) mpz_cmp(a,b)
#define BIG_SIGN(x) mpz_sgn(MP(x))
#define MP_SIGN(x) mpz_sgn(MP(x))
#define signe(u) mpz_sgn(u)
#define ZERO_BIG(x) (mpz_set_ui(MP(x),0))
/* force to be positive or negative according to sign. */
#define SET_BIG_SIGN(x,sign) \
  do{if (sign < 0) {if (big_sign(x) > 0) mpz_neg(MP(x),MP(x)); } \
      else { if (big_sign(x) < 0)  mpz_neg(MP(x),MP(x)); } } while(0)
#define MP_LOW(u,n) (*(u)->_mp_d)
     
/* the bit length of each word in bignum representation */
#define BIG_RADIX 32

/* #define MP_COUNT_BITS(u) mpz_sizeinbase(u,2) */
#define MP_BITCOUNT(u) mpz_bitcount(u)
#define MP_SIZE_IN_BASE2(u) mpz_bitlength(u)


#else

#include "genpari.h"
#undef K

#undef subis
#define subis(y,x) (x== (1<<31) ? addii(ABS_MOST_NEGS,y) : addsi(-x,y))
GEN subss();

#define SI_TO_MP(x,ignore) stoi(x)

#define INT_FLAG 0x1010000

#define MP_ALLOCATED(x) (x)->big.big_length
#define MP_SELF(x) (x)->big.big_self
#define MP_LIMB_SIZE (sizeof(long))


#define MP_SELF(x) MP(x)._mp_d


/* the bit length of each word in bignum representation */
#define BIG_RADIX 32

/* used for gc protecting */
object big_register_1;

object big_minus();
object make_bignum();
object make_integer();
#define BIG_SIGN(x) signe(MP(x))
#define SET_BIG_SIGN(x,sign) setsigne(MP(x),sign)
#define MP(x) ((GEN)((x)->big.big_self))
#define MP_START_LOW(u,x,l)  u = (x)+l
#define MP_START_HIGH(u,x,l)  u = (x)+2
#define MP_NEXT_UP(u) (*(--(u)))
#define MP_NEXT_DOWN(u) (*((u)++))
  /* ith word from the least significant */
#define MP_ITH_WORD(u,i,l) (u)[l-i-1]
#define MP_CODE_WORDS 2
/* MP_LOW(x,lgef(x)) is the least significant  word */
#define MP_LOW(x,l) ((x)[(l)-1])
/* most significant word if l is the lgef(x) */  
#define MP_HIGH(x,l) (x)[2]
#define MP_ONLY_WORD(u) MP_LOW((u),(MP_CODE_WORDS+1))


#define MP_BITCOUNT(u) gen_bitcount(u)
#define MP_SIZE_IN_BASE2(u) gen_bitlength(u)
  
  
  
#define MP_FIRST(x) ((MP(x))[2])
#define MP_SIGN(x) (signe(MP(x)))
#define ZERO_BIG(x) \
  do { 	(x)->big.big_length = 2; \
	(x)->big.big_self = gzero;} while(0)





			   

GEN addss();

#define MPOP(dowith, fun,x1,x2) \
  do{GEN _xgen ; \
     save_avma ; \
     _xgen =fun(x1,x2) ;\
     restore_avma; \
     dowith make_integer(_xgen);  }while(0)


#define MPOP_DEST(where ,fun,x1,x2) \
  do{GEN _xgen ; \
     save_avma ; \
     _xgen =fun(x1,x2) ;\
     restore_avma; \
     gcopy_to_big(_xgen,where);  }while(0)

#endif
