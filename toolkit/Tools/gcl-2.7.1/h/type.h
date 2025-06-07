enum type {
  t_cons,
  t_start = 0,
  t_fixnum,
  t_bignum,
  t_ratio,
  t_shortfloat,
  t_longfloat,
  t_complex,
  t_pathname,
  t_string,
  t_simple_string,
  t_simple_bitvector,
  t_bitvector,
  t_simple_vector,
  t_vector,
  t_simple_array,
  t_array,
  t_hashtable,
  t_structure,
  t_character,
  t_symbol,
  t_package,
  t_stream,
  t_random,
  t_readtable,
  t_function,
  t_cfdata,
  t_spice,
  t_contiguous,
  t_end=t_contiguous,
  t_relocatable,
  t_other
};

#define Zcdr(a_)                 (*(object *)(a_))/* ((a_)->c.c_cdr) */ /*FIXME*/

#ifndef WIDE_CONS

#ifndef USE_SAFE_CDR
#define SAFE_CDR(a_)             a_
#define imcdr(a_)                is_imm_fixnum(Zcdr(a_))
#else
#define SAFE_CDR(a_)             ({object _a=(a_);is_imm_fixnum(_a) ? make_fixnum1(fix(_a)) : _a;})
#ifdef DEBUG_SAFE_CDR
#define imcdr(a_)                (is_imm_fixnum(Zcdr(a_)) && (error("imfix cdr"),1))
#else
#define imcdr(a_)                0
#endif
#endif

#else

#define SAFE_CDR(a_)             a_
#define imcdr(a_)                0

#endif

#define is_marked(a_)            (imcdr(a_) ? is_marked_imm_fixnum(Zcdr(a_)) : (a_)->d.m)
#define is_marked_or_free(a_)    (imcdr(a_) ? is_marked_imm_fixnum(Zcdr(a_)) : (a_)->md.mf)
#define mark(a_)                 if (imcdr(a_)) mark_imm_fixnum(Zcdr(a_)); else (a_)->d.m=1
#define unmark(a_)               if (imcdr(a_)) unmark_imm_fixnum(Zcdr(a_)); else (a_)->d.m=0
#define is_free(a_)              (!is_imm_fixnum(a_) && !imcdr(a_) && (a_)->d.f)
#define make_free(a_)            ({(a_)->fw=0;(a_)->d.f=1;(a_)->d.h=(fixnum)OBJNULL ? 1 : 0;})
#define make_unfree(a_)          {(a_)->d.f=0;}

#ifdef WIDE_CONS
#define valid_cdr(a_)            0
#else
#define valid_cdr(a_)            (!(a_)->d.e || imcdr(a_))
#endif

#define type_of(x)       ({register object _z=(object)(x);\
                           (is_imm_fixnum(_z) ? t_fixnum : \
			    (valid_cdr(_z) ?  (_z==Cnil ? t_symbol : t_cons)  : _z->d.t));})
  
#ifdef WIDE_CONS
#define TYPEWORD_TYPE_P(y_) 1
#else
#define TYPEWORD_TYPE_P(y_) (y_!=t_cons)
#endif

#define set_type_of(x,y) ({object _x=(object)(x);enum type _y=(y);_x->d.f=0;\
      if (TYPEWORD_TYPE_P(_y)) {_x->d.e=1;_x->d.t=_y;_x->d.h=(fixnum)OBJNULL ? 1 : 0;}})

#ifndef WIDE_CONS

#define cdr_listp(x)     valid_cdr(x)
#define consp(x)         ({register object _z=(object)(x);\
                           (!is_imm_fixnum(_z) && valid_cdr(_z) && _z!=Cnil);})
#define listp(x)         ({register object _z=(object)(x);\
                           (!is_imm_fixnum(_z) && valid_cdr(_z));})
#define atom(x)          ({register object _z=(object)(x);\
                           (is_imm_fixnum(_z) || !valid_cdr(_z) || _z==Cnil);})

#else

#define cdr_listp(x)     listp(x)
#define consp(x)         (type_of(x)==t_cons)
#define listp(x)         ({object _x=x;type_of(_x)==t_cons || _x==Cnil;})
#define atom(x)          !consp(x)

#endif

#define SPP(a_,b_) (type_of(a_)==Join(t_,b_))
#define streamp(a_)    SPP(a_,stream)
#define packagep(a_)   SPP(a_,package)
#define hashtablep(a_) SPP(a_,hashtable)
#define randomp(a_)    SPP(a_,random)
#define characterp(a_) SPP(a_,character)
#define symbolp(a_)    SPP(a_,symbol)
#define pathnamep(a_)  SPP(a_,pathname)
#define stringp_tp(a_) TS_MEMBER(a_,TS(t_string)|TS(t_simple_string))
#define stringp(a_)    stringp_tp(type_of(a_))
#define fixnump(a_)    SPP(a_,fixnum)
#define readtablep(a_) SPP(a_,readtable)
#define functionp(a_)  (type_of(a_)==t_function)
#define compiled_functionp(a_)  functionp(a_)

#define integerp(a_) ({enum type _tp=type_of(a_); _tp >= t_fixnum     && _tp <= t_bignum;})
#define non_negative_integerp(a_) ({enum type _tp=type_of(a_); (_tp == t_fixnum && fix(a_)>=0) || (_tp==t_bignum && big_sign(a_)>=0);})
#define rationalp(a_)({enum type _tp=type_of(a_); _tp >= t_fixnum     && _tp <= t_ratio;})
#define floatp(a_)   ({enum type _tp=type_of(a_); _tp == t_shortfloat || _tp == t_longfloat;})
#define realp(a_)    ({enum type _tp=type_of(a_); _tp >= t_fixnum     && _tp < t_complex;})
#define numberp(a_)  ({enum type _tp=type_of(a_); _tp >= t_fixnum     && _tp <= t_complex;})
#define arrayp(a_)   ({enum type _tp=type_of(a_); _tp >= t_string     && _tp <= t_array;})
#define vectorp(a_)  ({enum type _tp=type_of(a_); _tp >= t_string     && _tp < t_array;})

#define string_symbolp(a_)                 ({enum type _tp=type_of(a_); stringp_tp(_tp) || _tp == t_symbol;})
#define pathname_string_symbolp(a_)        ({enum type _tp=type_of(a_); _tp==t_pathname || stringp_tp(_tp) \
                                                                     || _tp == t_symbol;})
#define pathname_string_symbol_streamp(a_) ({enum type _tp=type_of(a_); _tp==t_pathname || stringp_tp(_tp) \
                                                                     || _tp == t_symbol || _tp==t_stream;})
/* #define eql_is_eq(a_)    (is_imm_fixnum(a_) || ({enum type _tp=type_of(a_); _tp == t_cons || _tp > t_complex;})) */
#define eql_is_eq(a_) (is_imm_fixnum(a_)||valid_cdr(a_)||(a_->d.t>t_complex))
#define equal_is_eq(a_)  (is_imm_fixnum(a_) || type_of(a_)>t_bitvector)
#define equalp_is_eq(a_) (type_of(a_)>t_structure)



#define tp0(x) is_imm_fixnum(x)/*(((ufixnum)x)>=IM_FIX_BASE)*/

#define tp1(x) (x==Cnil)

#define tp2(x) ({object _x=x;is_imm_fixnum(_x) ? 2 : _x->d.e && !is_imm_fixnum(_x->ff.ff);})/*(((ufixnum)_x)>=IM_FIX_BASE)*/

#define tp3(x) ({object _x=x;_x==Cnil ? 2 : (is_imm_fixnum(_x) ? 3 : _x->d.e && !is_imm_fixnum(_x->ff.ff));})

#define tp4(x) ({object _x=x;is_imm_fixnum(_x) ? -1 : _x->d.e && !is_imm_fixnum(_x->ff.ff) ? _x->d.t : 0;})

#define tp5(x) ({object _x=x;_x==Cnil ? -2 : (is_imm_fixnum(_x) ? -1 : (_x->d.e && !is_imm_fixnum(_x->ff.ff) ? _x->d.t : 0));})


#define tp6(x) ({object _x=x;is_imm_fixnum(_x) ? -1 : (_x->d.e && !is_imm_fixnum(_x->ff.ff) ? _x->fstp.tp : 0);})

#define tp7(x) ({object _x=x;_x==Cnil ? -2 :				\
      (is_imm_fixnum(_x) ? -1 :						\
       (_x->d.e && !is_imm_fixnum(_x->ff.ff) ? _x->fstp.tp : 0));})

#define tp8(x) ({object _x=x;(is_imm_fixnum(_x) ? 0 :			\
	(_x->d.e && !is_imm_fixnum(_x->ff.ff) ?				\
	 (_x->d.t<t_complex ? x->d.t :					\
	  (_x->d.t==t_complex&&x->d.tt<4 ? x->d.t :			\
	   (_x->d.t==t_complex ? x->d.t+x->d.tt-3 :			\
	    0))) : 0));})/*FIXME*/

