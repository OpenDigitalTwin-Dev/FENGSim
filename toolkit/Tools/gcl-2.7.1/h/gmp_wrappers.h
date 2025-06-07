#ifndef GMP_WRAPPERS_H
#define GMP_WRAPPERS_H

EXTER jmp_buf gmp_jmp;
EXTER int jmp_gmp,gmp_relocatable;

#define join(a_,b_) a_ ## b_
#define Join(a_,b_) join(a_,b_)

/*FIXME : this is slightly excessively conservative as it includes
  comparisons with possible non mpz_t type arguments*/
#define E21 _b==(void *)_c
#define E31 E21||_b==(void *)_d
#define E32 _b==(void *)_d||_c==(void *)_d
#define E41 E31||_b==(void *)_e
#define E42 _b==(void *)_d||_b==(void *)_e||_c==(void *)_d||_c==(void *)_e
#define E53 _b==(void *)_e||_b==(void *)_f||_c==(void *)_e||_c==(void *)_f||_d==(void *)_e||_d==(void *)_f
#define E20 0
#define E11 0
#define E10 0
#define E30 0

/*           if (jmp_gmp++>1) \
              FEerror("gmp jmp loop in" #a_, 0);\
*/

#define GMP_TMP _tmp

#define RF_gmp_ulint unsigned long int
#define RD_gmp_ulint RF_gmp_ulint GMP_TMP
#define RA_gmp_ulint GMP_TMP =
#define RR_gmp_ulint GMP_TMP

#define RF_gmp_lint long int
#define RD_gmp_lint RF_gmp_lint GMP_TMP
#define RA_gmp_lint GMP_TMP =
#define RR_gmp_lint GMP_TMP

#define RF_int int
#define RD_int RF_int GMP_TMP
#define RA_int GMP_TMP =
#define RR_int GMP_TMP

#define RF_gmp_char_star char *
#define RD_gmp_char_star RF_gmp_char_star GMP_TMP
#define RA_gmp_char_star GMP_TMP =
#define RR_gmp_char_star GMP_TMP

#define RF_double double
#define RD_double RF_double GMP_TMP
#define RA_double GMP_TMP =
#define RR_double GMP_TMP

#define RF_size_t size_t
#define RD_size_t RF_size_t GMP_TMP
#define RA_size_t GMP_TMP =
#define RR_size_t GMP_TMP

#define RF_void void
#define RD_void
#define RA_void
#define RR_void

#define RF_mpz_t mpz_t
#define RF_gmp_randstate_t gmp_randstate_t
/* #define RF_gmp_char_star_star char ** */

#define P1(bt_) Join(RF_,bt_) _b
#define P2(bt_,ct_) P1(bt_),Join(RF_,ct_) _c
#define P3(bt_,ct_,dt_) P2(bt_,ct_),Join(RF_,dt_) _d
#define P4(bt_,ct_,dt_,et_) P3(bt_,ct_,dt_),Join(RF_,et_) _e
#define P5(bt_,ct_,dt_,et_,ft_) P4(bt_,ct_,dt_,et_),Join(RF_,ft_) _f

#define A1 _b
#define A2 A1,_c
#define A3 A2,_d
#define A4 A3,_e
#define A5 A4,_f


#define SS_40 4
#define SS_30 3
#define SS_20 2
#define SS_10 1
#define SS_00 0
#define SS_41 3
#define SS_31 2
#define SS_21 1
#define SS_11 0
#define SS_42 2
#define SS_32 1
#define SS_22 0
#define SS_43 1
#define SS_33 0
#define SS_44 0
#define SS_53 2

#define PP_gmp_ulint  1
#define PP_gmp_lint   1
#define PP_int        1
#define PP_size_t     1
#define PP_void       0
#define PP_00 0
#define PP_10 1
#define PP_11 2
#define PP_20 2
#define PP_21 3
#define PP0(a_) Join(PP_,a_)
#define PP1(a_) Join(PP_1,Join(PP_,a_))
#define PP2(a_) Join(PP_2,Join(PP_,a_))


#define QQQ_gmp_ulint  f
#define QQQ_gmp_lint   f
#define QQQ_int        f
#define QQQ_size_t     f
#define QQQ_mpz_t      b

#define QQ10(a_) Join(QQQ_,a_)
#define QQ20(a_,b_) QQ10(a_),QQ10(b_)
#define QQ30(a_,b_,c_) QQ20(a_,b_),QQ10(c_)
#define QQ40(a_,b_,c_,d_) QQ30(a_,b_,c_),QQ10(d_)

#define QQ11(a_) 
#define QQ21(a_,b_) QQ10(b_)
#define QQ31(a_,b_,c_) QQ20(b_,c_)
#define QQ41(a_,b_,c_,d_) QQ30(b_,c_,d_)

#define QQ22(a_,b_) 
#define QQ32(a_,b_,c_) QQ10(c_)
#define QQ42(a_,b_,c_,d_) QQ20(c_,d_)

#define QQ53(a_,b_,c_,d_,e_) QQ20(d_,e_)

#define ZZ_gmp_ulint  f
#define ZZ_gmp_lint   f
#define ZZ_int        f
#define ZZ_size_t     f
#define ZZ_void       
#define ZZ3(a_) Join(Join(ZZ_,a_),bbb)
#define ZZ2(a_) Join(Join(ZZ_,a_),bb)
#define ZZ1(a_) Join(Join(ZZ_,a_),b)
#define ZZ0(a_) Join(ZZ_,a_)

#ifndef BF
#define BF(n_,b_,r_,a_...)
#endif

/* #undef mpz_get_strp */
/* #define mpz_get_strp __gmpz_get_strp */

/* GMP_EXTERN_INLINE char * */
/* __gmpz_get_strp(char **a,int b,mpz_t c) {return __gmpz_get_str(*a,b,c);} /\*FIXME*\/ */

/* GMP_WRAPPERS: the gmp library uses heap allocation in places for
   temporary storage.  This greatly complicates relocatable bignum
   allocation in GCL, which is a big winner in terms of performance.
   The old procedure was to patch gmp to use alloca in such instances.
   Aside from possible silently introducing bugs as gmp evolves, such
   a policy also runs the risk of colliding with gmp's stated policy
   of storing pointers in allocated blocks, a possiblity GCL's
   conservative garbage collector is not designed to handle.  Here we
   implement a policy of preventing garbage collection inside of gmp
   calls in any case.  In case of non-inplace calls, where source and
   destination arguments are distinct, we simply longjmp back to the
   front of the call if a gbc would be needed and try the call again,
   as any previous partial write into the destination is of no
   consequence.  Just as is the case with the alloc_contblock and
   alloc_relblock algorithms themselves, on the second pass (as
   indicated by jmp_gmp) new pages are added if there is still not
   enough room in lieu of GBC.  In case of in-place calls, we schedule
   a GBC call after the gmp call completes, relying on the allocator
   to add pages immediately to the type to satisfy the allocation when
   necessary. jmp_gmp counts the pass for non-in-place calls, and is
   set to -1 otherwise.  20040815 CM*/

#define MEM_GMP_CALL(n_,rt_,a_,s_,b_...) \
   INLINE Join(RF_,rt_) Join(m,a_)(Join(P,n_)(b_)) { \
           int j;\
           Join(RD_,rt_);\
           if (gmp_relocatable) {\
	     jmp_gmp=0;\
             if ((j=setjmp(gmp_jmp)))\
                GBC(j);\
             if (Join(Join(E,n_),s_)) jmp_gmp=-1 ; else jmp_gmp++;\
           }\
           Join(RA_,rt_) a_(Join(A,n_));\
           if (gmp_relocatable) {\
             if (jmp_gmp<-1) GBC(-jmp_gmp);\
             jmp_gmp=0;\
           }\
           return Join(RR_,rt_);\
   }

#define EXPORT_GMP_CALL(n_,rt_,a_,s_,b_...)			\
  MEM_GMP_CALL(n_,rt_,Join(mpz_,a_),s_,b_)			\
  BF(n_,Join(SS_,Join(n_,s_)),s_,a_,Join(ZZ,s_)(rt_),Join(QQ,Join(n_,s_))(b_))

MEM_GMP_CALL(3,void,mpz_urandomm,1,mpz_t,gmp_randstate_t,mpz_t)
MEM_GMP_CALL(2,void,gmp_randseed,1,gmp_randstate_t,mpz_t)
MEM_GMP_CALL(2,void,gmp_randseed_ui,1,gmp_randstate_t,gmp_ulint)
MEM_GMP_CALL(1,void,gmp_randinit_default,0,gmp_randstate_t)

 
EXPORT_GMP_CALL(2,gmp_ulint,scan0,0,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,gmp_ulint,scan1,0,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(3,void,add,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,add_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,sub,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,sub_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,ui_sub,1,mpz_t,gmp_ulint,mpz_t)
EXPORT_GMP_CALL(3,void,mul,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,mul_si,1,mpz_t,mpz_t,gmp_lint)
EXPORT_GMP_CALL(3,void,mul_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,mul_2exp,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,void,neg,1,mpz_t,mpz_t)
EXPORT_GMP_CALL(4,void,tdiv_qr,2,mpz_t,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,fdiv_q_2exp,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,int,cmp,0,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,int,cmpabs,0,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,int,cmpabs_ui,0,mpz_t,gmp_ulint)
/* EXPORT_GMP_CALL(2,int,cmp_si,0,mpz_t,gmp_lint) */ /*macro*/
/* EXPORT_GMP_CALL(2,int,cmp_ui,0,mpz_t,gmp_ulint) */
EXPORT_GMP_CALL(3,void,and,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,xor,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,ior,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,void,com,1,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,int,tstbit,0,mpz_t,gmp_ulint)
 MEM_GMP_CALL(1,void,mpz_init,1,mpz_t)
 MEM_GMP_CALL(2,void,mpz_init_set,1,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,void,set,1,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,void,set_ui,1,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,void,set_si,1,mpz_t,gmp_lint)
  MEM_GMP_CALL(1,double,mpz_get_d,0,mpz_t)
EXPORT_GMP_CALL(1,gmp_lint,get_si,0,mpz_t)
EXPORT_GMP_CALL(1,gmp_lint,get_ui,0,mpz_t)
MEM_GMP_CALL(3,gmp_char_star,mpz_get_str,0,gmp_char_star,int,mpz_t)
MEM_GMP_CALL(3,int,mpz_set_str,0,mpz_t,gmp_char_star,int)/*arg set, but 0 for check as moot*/
EXPORT_GMP_CALL(1,int,fits_sint_p,0,mpz_t)
EXPORT_GMP_CALL(1,int,fits_slong_p,0,mpz_t)
EXPORT_GMP_CALL(1,int,fits_sshort_p,0,mpz_t)
EXPORT_GMP_CALL(1,int,fits_uint_p,0,mpz_t)
EXPORT_GMP_CALL(1,int,fits_ulong_p,0,mpz_t)
EXPORT_GMP_CALL(1,int,fits_ushort_p,0,mpz_t)
EXPORT_GMP_CALL(1,gmp_ulint,popcount,0,mpz_t)

EXPORT_GMP_CALL(1,size_t,size,0,mpz_t)
EXPORT_GMP_CALL(2,size_t,sizeinbase,0,mpz_t,int)
EXPORT_GMP_CALL(3,void,gcd,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(5,void,gcdext,3,mpz_t,mpz_t,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,gmp_ulint,gcd_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,divexact,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,divexact_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,void,fac_ui,1,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(4,void,powm,1,mpz_t,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(4,void,powm_ui,1,mpz_t,mpz_t,gmp_ulint,mpz_t)
EXPORT_GMP_CALL(3,void,ui_pow_ui,1,mpz_t,gmp_ulint,gmp_ulint)
EXPORT_GMP_CALL(3,void,pow_ui,1,mpz_t,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(2,int,probab_prime_p,0,mpz_t,int)
EXPORT_GMP_CALL(2,void,nextprime,1,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,lcm,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,lcm_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,invert,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,int,jacobi,0,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,int,kronecker_si,0,mpz_t,gmp_lint)
EXPORT_GMP_CALL(2,int,kronecker_ui,0,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,int,si_kronecker,0,gmp_lint,mpz_t)
EXPORT_GMP_CALL(2,int,ui_kronecker,0,gmp_ulint,mpz_t)
EXPORT_GMP_CALL(3,gmp_ulint,remove,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,bin_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,bin_uiui,1,mpz_t,gmp_ulint,gmp_ulint)
EXPORT_GMP_CALL(2,void,fib_ui,1,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,fib2_ui,2,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,void,lucnum_ui,1,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,lucnum2_ui,2,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,mod,1,mpz_t,mpz_t,mpz_t)
/* EXPORT_GMP_CALL(3,void,mod_ui,1,mpz_t,mpz_t,gmp_ulint) */ /*alias*/
EXPORT_GMP_CALL(2,gmp_ulint,millerrabin,0,mpz_t,int)
EXPORT_GMP_CALL(2,gmp_ulint,hamdist,0,mpz_t,mpz_t)
/* EXPORT_GMP_CALL(1,int,odd_p,0,mpz_t) */ /*macro*/
/* EXPORT_GMP_CALL(1,int,even_p,0,mpz_t) */

EXPORT_GMP_CALL(3,int,root,1,mpz_t,mpz_t,gmp_ulint)/* mult val*/
EXPORT_GMP_CALL(4,void,rootrem,2,mpz_t,mpz_t,mpz_t,gmp_ulint)/* mult val*/
EXPORT_GMP_CALL(2,void,sqrt,1,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,sqrtrem,2,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(1,int,perfect_power_p,0,mpz_t)
EXPORT_GMP_CALL(1,int,perfect_square_p,0,mpz_t)

EXPORT_GMP_CALL(3,void,cdiv_q,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,cdiv_r,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(4,void,cdiv_qr,2,mpz_t,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,gmp_ulint,cdiv_q_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,gmp_ulint,cdiv_r_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(4,gmp_ulint,cdiv_qr_ui,2,mpz_t,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,gmp_ulint,cdiv_ui,0,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,cdiv_q_2exp,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,cdiv_r_2exp,1,mpz_t,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(3,void,fdiv_q,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,fdiv_r,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(4,void,fdiv_qr,2,mpz_t,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,gmp_ulint,fdiv_q_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,gmp_ulint,fdiv_r_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(4,gmp_ulint,fdiv_qr_ui,2,mpz_t,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,gmp_ulint,fdiv_ui,0,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,fdiv_r_2exp,1,mpz_t,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(3,void,tdiv_q,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,tdiv_r,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,gmp_ulint,tdiv_q_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,gmp_ulint,tdiv_r_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(4,gmp_ulint,tdiv_qr_ui,2,mpz_t,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,gmp_ulint,tdiv_ui,0,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,tdiv_q_2exp,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,tdiv_r_2exp,1,mpz_t,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(2,int,divisible_p,0,mpz_t,mpz_t)
EXPORT_GMP_CALL(2,int,divisible_ui_p,0,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(2,int,divisible_2exp_p,0,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(3,int,congruent_p,0,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,int,congruent_ui_p,0,mpz_t,gmp_ulint,gmp_ulint)
EXPORT_GMP_CALL(3,int,congruent_2exp_p,0,mpz_t,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(3,void,addmul,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,addmul_ui,1,mpz_t,mpz_t,gmp_ulint)
EXPORT_GMP_CALL(3,void,submul,1,mpz_t,mpz_t,mpz_t)
EXPORT_GMP_CALL(3,void,submul_ui,1,mpz_t,mpz_t,gmp_ulint)

EXPORT_GMP_CALL(2,void,abs,1,mpz_t,mpz_t)


/* MEM_GMP_CALL(2,void *,mpz_realloc,mpz_t,mp_size_t)*/
/* MEM_GMP_CALL(2,int,mpz_legendre,0,mpz_t,mpz_t) */ /*alias*/
/* MEM_GMP_CALL(2,int,mpz_kronecker,0,mpz_t,mpz_t) */

     /* FIXME: find a way to have this follow the convention in gmp.h*/

#define __gmpz_urandomm m__gmpz_urandomm
#define __gmp_randseed m__gmp_randseed
#define __gmp_randseed_ui m__gmp_randseed_ui
#define __gmp_randinit_default m__gmp_randinit_default


#define __gmpz_add m__gmpz_add
#define __gmpz_add_ui m__gmpz_add_ui
#define __gmpz_sub m__gmpz_sub
#define __gmpz_sub_ui m__gmpz_sub_ui
#define __gmpz_mul m__gmpz_mul
#define __gmpz_mul_si m__gmpz_mul_si
#define __gmpz_mul_ui m__gmpz_mul_ui
#define __gmpz_mul_2exp m__gmpz_mul_2exp
#define __gmpz_neg m__gmpz_neg
#define __gmpz_tdiv_qr m__gmpz_tdiv_qr
#define __gmpz_tdiv_q m__gmpz_tdiv_q
#define __gmpz_tdiv_r m__gmpz_tdiv_r
#define __gmpz_fdiv_qr m__gmpz_fdiv_qr
#define __gmpz_fdiv_q m__gmpz_fdiv_q
#define __gmpz_fdiv_r m__gmpz_fdiv_r
#define __gmpz_cdiv_qr m__gmpz_cdiv_qr
#define __gmpz_cdiv_q m__gmpz_cdiv_q
#define __gmpz_cdiv_r m__gmpz_cdiv_r
#define __gmpz_fdiv_q_2exp m__gmpz_fdiv_q_2exp
#define __gmpz_cmp m__gmpz_cmp
#define __gmpz_and m__gmpz_and
#define __gmpz_xor m__gmpz_xor
#define __gmpz_ior m__gmpz_ior
#define __gmpz_com m__gmpz_com
#define __gmpz_tstbit m__gmpz_tstbit
#define __gmpz_init m__gmpz_init
#define __gmpz_init_set m__gmpz_init_set
#define __gmpz_set m__gmpz_set
#define __gmpz_set_ui m__gmpz_set_ui
#define __gmpz_set_si m__gmpz_set_si
#define __gmpz_get_d m__gmpz_get_d
#define __gmpz_get_str m__gmpz_get_str
#define __gmpz_set_str m__gmpz_set_str
#define __gmpz_get_si m__gmpz_get_si
#define __gmpz_fits_sint_p m__gmpz_fits_sint_p
#define __gmpz_popcount m__gmpz_popcount
#define __gmpz_size m__gmpz_size
#define __gmpz_sizeinbase m__gmpz_sizeinbase
#define __gmpz_gcd m__gmpz_gcd
#define __gmpz_gcd_ui m__gmpz_gcd_ui
#define __gmpz_divexact m__gmpz_divexact
#define __gmpz_divexact_ui m__gmpz_divexact_ui
#define __gmpz_fac_ui m__gmpz_fac_ui
#define __gmpz_powm m__gmpz_powm
#define __gmpz_powm_ui m__gmpz_powm_ui
#define __gmpz_ui_pow_ui m__gmpz_ui_pow_ui
#define __gmpz_pow_ui m__gmpz_pow_ui

#define __gmpz_probab_prime_p m__gmpz_probab_prime_p
#define __gmpz_nextprime m__gmpz_nextprime
#define __gmpz_lcm m__gmpz_lcm
#define __gmpz_lcm_ui m__gmpz_lcm_ui
#define __gmpz_invert m__gmpz_invert
#define __gmpz_jacobi m__gmpz_jacobi
#define __gmpz_kronecker_si m__gmpz_kronecker_si
#define __gmpz_kronecker_ui m__gmpz_kronecker_ui
#define __gmpz_si_kronecker m__gmpz_si_kronecker
#define __gmpz_ui_kronecker m__gmpz_ui_kronecker
#define __gmpz_remove m__gmpz_remove
#define __gmpz_bin_ui m__gmpz_bin_ui
#define __gmpz_bin_uiui m__gmpz_bin_uiui
#define __gmpz_fib_ui m__gmpz_fib_ui
#define __gmpz_fib2_ui m__gmpz_fib2_ui
#define __gmpz_lucnum_ui m__gmpz_lucnum_ui
#define __gmpz_lucnum2_ui m__gmpz_lucnum2_ui
#define __gmpz_hamdist m__gmpz_hamdist
#define __gmpz_odd_p m__gmpz_odd_p
#define __gmpz_even_p m__gmpz_even_p

#define __gmpz_root m__gmpz_root
#define __gmpz_rootrem m__gmpz_rootrem
#define __gmpz_sqrt m__gmpz_sqrt
#define __gmpz_sqrtrem m__gmpz_sqrtrem
#define __gmpz_perfect_power_p m__gmpz_perfect_power_p
#define __gmpz_perfect_square_p m__gmpz_perfect_square_p

#define __gmpz_cdiv_q m__gmpz_cdiv_q
#define __gmpz_cdiv_r m__gmpz_cdiv_r
#define __gmpz_cdiv_qr m__gmpz_cdiv_qr
#define __gmpz_cdiv_q_ui m__gmpz_cdiv_q_ui
#define __gmpz_cdiv_r_ui m__gmpz_cdiv_r_ui
#define __gmpz_cdiv_qr_ui m__gmpz_cdiv_qr_ui
#define __gmpz_cdiv_ui m__gmpz_cdiv_ui
#define __gmpz_cdiv_q_2exp m__gmpz_cdiv_q_2exp
#define __gmpz_cdiv_r_2exp m__gmpz_cdiv_r_2exp

#define __gmpz_fdiv_q m__gmpz_fdiv_q
#define __gmpz_fdiv_r m__gmpz_fdiv_r
#define __gmpz_fdiv_qr m__gmpz_fdiv_qr
#define __gmpz_fdiv_q_ui m__gmpz_fdiv_q_ui
#define __gmpz_fdiv_r_ui m__gmpz_fdiv_r_ui
#define __gmpz_fdiv_qr_ui m__gmpz_fdiv_qr_ui
#define __gmpz_fdiv_ui m__gmpz_fdiv_ui
#define __gmpz_fdiv_r_2exp m__gmpz_fdiv_r_2exp

#define __gmpz_tdiv_q m__gmpz_tdiv_q
#define __gmpz_tdiv_r m__gmpz_tdiv_r
#define __gmpz_tdiv_q_ui m__gmpz_tdiv_q_ui
#define __gmpz_tdiv_r_ui m__gmpz_tdiv_r_ui
#define __gmpz_tdiv_qr_ui m__gmpz_tdiv_qr_ui
#define __gmpz_tdiv_ui m__gmpz_tdiv_ui
#define __gmpz_tdiv_q_2exp m__gmpz_tdiv_q_2exp
#define __gmpz_tdiv_r_2exp m__gmpz_tdiv_r_2exp

#define __gmpz_divisible_p m__gmpz_divisible_p
#define __gmpz_divisible_ui_p m__gmpz_divisible_ui_p
#define __gmpz_divisible_2exp_p m__gmpz_divisible_2exp_p

#define __gmpz_congruent_p m__gmpz_congruent_p
#define __gmpz_congruent_ui_p m__gmpz_congruent_ui_p
#define __gmpz_congruent_2exp_p m__gmpz_congruent_2exp_p

#define __gmpz_addmul m__gmpz_addmul
#define __gmpz_addmul_ui m__gmpz_addmul_ui
#define __gmpz_submul m__gmpz_submul
#define __gmpz_submul_ui m__gmpz_submul_ui

#define __gmpz_abs m__gmpz_abs

#endif
