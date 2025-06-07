#include <string.h>
#include <stdlib.h>
#include <fenv.h>
#include <errno.h>
#include <signal.h>


#define	CHAR_CODE_LIMIT	256	
#define	READ_TABLE_SIZE CHAR_CODE_LIMIT


EXTER int *cs_org;     
EXTER int GBC_enable;

#define CHAR_SIZE 8
EXTER object sSAnotify_gbcA;

/* symbols which are not needed in compiled lisp code */
EXTER int interrupt_flag,interrupt_enable;
/* void sigint(),sigalrm(); */


EXTER int gc_enabled, saving_system;

EXTER object lisp_package,user_package;
EXTER char *core_end;
EXTER int catch_fatal;
EXTER long real_maxpage;
EXTER char *this_lisp;

EXTER char stdin_buf[],stdout_buf[];

EXTER object user_package;

#define TRUE 1
#define FALSE 0



#define GET_OPT_ARG(min,max) \
  va_list ap; \
  object  opt_arg[max - min]; object *__p= opt_arg ;\
  int _i=min, _nargs = VFUN_NARGS ; \
  va_start(ap); \
  if (_nargs < min || (_nargs > max)) FEerror("wrong number of args"); \
  while(_i++ <= max) { if (_i > _nargs) *__p++ = Cnil; \
			 else *__p++ = va_arg(ap,object);} \
  va_end(ap)

#ifndef NO_DEFUN
/* eg.
   A function taking from 2 to 8 args
   returning object the first args is object, the next 6 int, and last defaults to object.
   note the return type must also be put in the signature.
  DEFUN("AREF",object,fSaref,SI,2,8,NONE,oo,ii,ii,ii)
*/

#define MAKEFUN(pack,string,fname,argd) \
  (pack == SI ? SI_makefun(string,fname,argd) : \
   pack == LISP ? LISP_makefun(string,fname,argd) : \
   error("Bad pack variable in MAKEFUN\n"))

#define MAKEFUNB(pack,string,fname,argd,p)	\
  (GMP_makefunb(string,fname,argd,p))

#define MAKEFUNM(pack,string,fname,argd) \
  (pack == SI ? SI_makefunm(string,fname,argd) : \
   pack == LISP ? LISP_makefunm(string,fname,argd) : \
   error("Bad pack variable in MAKEFUN\n"))

#define mjoin(a_,b_) a_ ## b_
#define Mjoin(a_,b_) mjoin(a_,b_)

#define SI 0
#define LISP 1

#undef FFN
#undef LFD
#undef FFD
#undef STATD
#undef make_function
#undef make_macro_function
#undef make_si_function
#undef make_si_sfun
#undef make_special_form
#ifdef STATIC_FUNCTION_POINTERS
#define FFN(a_) Mjoin(a_,_static)
#define LFD(a_) static void FFN(a_) (); void a_  () { FFN(a_)();} static void FFN(a_)
#define FFD(a_) static void FFN(a_) (object); void a_  (object x) { FFN(a_)(x);} static void FFN(a_)
#define make_function(a_,b_) make_function_internal(a_,FFN(b_))
#define make_macro_function(a_,b_) make_macro_internal(a_,FFN(b_))
#define make_si_function(a_,b_) make_si_function_internal(a_,FFN(b_))
#define make_special_form(a_,b_) make_special_form_internal(a_,FFN(b_))
#define make_si_special_form(a_,b_) make_si_special_form_internal(a_,FFN(b_))
#define make_macro_function(a_,b_) make_macro_internal(a_,FFN(b_))
#define make_si_sfun(a_,b_,c_) make_si_sfun_internal(a_,FFN(b_),c_)
#define STATD static
#else
#define FFN(a_) (a_)
#define LFD(a_) void a_
#define FFD(a_) void a_
#define make_function(a_,b_) make_function_internal(a_,b_)
#define make_macro_function(a_,b_) make_macro_internal(a_,b_)
#define make_si_function(a_,b_) make_si_function_internal(a_,b_)
#define make_special_form(a_,b_) make_special_form_internal(a_,b_)
#define make_si_special_form(a_,b_) make_si_special_form_internal(a_,b_)
#define make_macro_function(a_,b_) make_macro_internal(a_,b_)
#define make_si_sfun(a_,b_,c_) make_si_sfun_internal(a_,b_,c_)
#define STATD
#endif

#undef DEFUN
#define DEFUN(string,ret,fname,pack,min,max, flags, ret0a0,a12,a34,a56,args,doc) STATD ret FFN(fname) args; \
void Mjoin(fname,_init) () {\
  MAKEFUN(pack,string,(void *)FFN(fname),F_ARGD(min,max,(flags|ONE_VAL),ARGTYPES(ret0a0,a12,a34,a56))); \
}\
STATD ret FFN(fname) args

#define DEFUNB(string,ret,fname,pack,min,max, flags, ret0a0,a12,a34,a56,args,p,doc) STATD ret FFN(fname) args; \
void Mjoin(fname,_init) () {\
  MAKEFUNB(pack,string,(void *)FFN(fname),F_ARGD(min,max,(flags|ONE_VAL),ARGTYPES(ret0a0,a12,a34,a56)),p); \
}\
STATD ret FFN(fname) args

#define DEFUNM(string,ret,fname,pack,min,max, flags, ret0a0,a12,a34,a56,args,doc) STATD ret FFN(fname) args;\
void Mjoin(fname,_init) () {\
  MAKEFUNM(pack,string,(void *)FFN(fname),F_ARGD(min,max,flags,ARGTYPES(ret0a0,a12,a34,a56))); \
}\
STATD ret FFN(fname) args

#define DEFUNM_NEW(string,ret,fname,pack,min,max, flags, ret0a0,a12,a34,a56,args,doc) STATD ret FFN(fname) args;\
void Mjoin(fname,_init) () {\
   MAKEFUNM(pack,string,(ret (*)())FFN(fname),F_ARGD(min,max,flags,ARGTYPES(ret0a0,a12,a34,a56)));\
}\
STATD ret FFN(fname) args

/* eg.
   A function taking from 2 to 8 args
   returning object the first args is object, the next 6 int, and last defaults to object.
   note the return type must also be put in the signature.
  DEFUN("AREF",object,fSaref,SI,2,8,NONE,oo,ii,ii,ii)
*/

  /* these are needed to be linked in to be called by incrementally
   loaded code */
#define DEFCOMP(type,fun) type fun

#define  DEFVAR(name,cname,pack,val,doc) object cname
#define  DEFCONST(name,cname,pack,val,doc) object cname
#define  DEF_ORDINARY(name,cname,pack,doc) object cname  
#define DO_INIT(x)   
#endif /* NO_DEFUN */




#define TYPE_OF(x) type_of(x)


/* For a faster way of checking if t0 is in several types,
   is t0 a member of types t1 t2 t3 
TS_MEMBER(t0,TS(t1)|TS(t2)|TS(t3)...)
*/
#define TS(s) (1<<s)
#define TS_MEMBER(t1,ts) ((TS(t1)) & (ts))

#define ASSURE_TYPE(val,t) if (type_of(val)!=t) TYPE_ERROR(val,type_name(t))


/* array to which X is has its body displaced */
#define DISPLACED_TO(x) Mcar(ADISP(x))

/* List of arrays whose bodies are displaced to X */

#define DISPLACED_FROM(x) Mcdr(ADISP(x))

#define FIX_CHECK(x) (Mfix(Iis_fixnum(x)))

#define INITIAL_TOKEN_LENGTH 512

/* externals not needed by cmp */
/* print.d */

/* from format.c */
EXTER VOL object fmt_stream;
EXTER VOL int ctl_origin;
EXTER VOL int ctl_index;
EXTER VOL int ctl_end;
EXTER  object * VOL fmt_base;
EXTER VOL int fmt_index;
EXTER VOL int fmt_end;
EXTER VOL object fmt_iteration_list;
typedef jmp_buf *jmp_bufp;
EXTER jmp_bufp VOL fmt_jmp_bufp;
EXTER VOL int fmt_indents;
EXTER VOL object fmt_string;
EXTER object endp_temp;

/* eval */
EXTER int eval1 ;
/* list.d */
EXTER bool in_list_flag;
EXTER object test_function;
EXTER object item_compared;
EXTER object key_function;


/* string.d */
EXTER  bool left_trim;
EXTER bool right_trim;


/* on most machines this will test in one instruction
   if the pointer is on the C stack or the 0 pointer
   but if the CSTACK_ADDRESS is not negative then we can't use this cheap
   test..
*/
#ifndef NULL_OR_ON_C_STACK

#define NULL_OR_ON_C_STACK(x) ({\
      /* if ((void *)(x)<data_start && ((void *)(x)!=NULL) && ((object)(x))!=Cnil && ((object)(x))!=Ct) */ \
      /* {pp(x);printf("%p %p\n",(void *)(x),data_start);}			*/ \
      ((((void *)(x))<(void *)data_start || ((void *)(x))>=(void *)core_end));})

#endif /* NULL_OR_ON_C_STACK */

#define	siScomma     sSXB
EXTER object sSXB;

#define	inheap(pp)	((char *)(pp) < heap_end)



#undef SAFE_READ
#undef SAFE_FREAD
#ifdef SGC
#define SAFE_READ(a_,b_,c_) \
   ({int _a=(a_),_c=(c_);char *_b=(b_);extern int sgc_enabled;\
     if (sgc_enabled) memset(_b,0,_c); \
     read(_a,_b,_c);})
#define SAFE_FREAD(a_,b_,c_,d_) \
   ({int _b=(b_),_c=(c_);char *_a=(a_);FILE *_d=(d_);extern int sgc_enabled; \
     if (sgc_enabled) memset(_a,0,_b*_c); \
     fread(_a,_b,_c,_d);})
#else
#define SAFE_READ(a_,b_,c_) read((a_),(b_),(c_))
#define SAFE_FREAD(a_,b_,c_,d_) fread((a_),(b_),(c_),(d_))
#endif

#ifdef EXPORT_GMP
#include "bfdef.h"
#endif
#include "gmp_wrappers.h"

char FN1[PATH_MAX],FN2[PATH_MAX],FN3[PATH_MAX],FN4[PATH_MAX],FN5[PATH_MAX];

#define coerce_to_filename(a_,b_) coerce_to_filename1(a_,b_,sizeof(b_))

#define massert(a_) ({errno=0;if (!(a_)) assert_error(#a_,__LINE__,__FILE__,__FUNCTION__);})

extern bool writable_malloc;
#define writable_malloc_wrap(f_,rt_,a_...) ({rt_ v;bool w=writable_malloc;writable_malloc=1;v=f_(a_);writable_malloc=w;v;})
#define fopen(a_,b_) writable_malloc_wrap(fopen,FILE *,a_,b_)

#define Mcar(x)	(x)->c.c_car
#define Mcdr(x)	(x)->c.c_cdr
#define Mcaar(x)	(x)->c.c_car->c.c_car
#define Mcadr(x)	(x)->c.c_cdr->c.c_car
#define Mcdar(x)	(x)->c.c_car->c.c_cdr
#define Mcddr(x)	(x)->c.c_cdr->c.c_cdr
#define Mcaaar(x)	(x)->c.c_car->c.c_car->c.c_car
#define Mcaadr(x)	(x)->c.c_cdr->c.c_car->c.c_car
#define Mcadar(x)	(x)->c.c_car->c.c_cdr->c.c_car
#define Mcaddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_car
#define Mcdaar(x)	(x)->c.c_car->c.c_car->c.c_cdr
#define Mcdadr(x)	(x)->c.c_cdr->c.c_car->c.c_cdr
#define Mcddar(x)	(x)->c.c_car->c.c_cdr->c.c_cdr
#define Mcdddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_cdr
#define Mcaaaar(x)	(x)->c.c_car->c.c_car->c.c_car->c.c_car
#define Mcaaadr(x)	(x)->c.c_cdr->c.c_car->c.c_car->c.c_car
#define Mcaadar(x)	(x)->c.c_car->c.c_cdr->c.c_car->c.c_car
#define Mcaaddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_car->c.c_car
#define Mcadaar(x)	(x)->c.c_car->c.c_car->c.c_cdr->c.c_car
#define Mcadadr(x)	(x)->c.c_cdr->c.c_car->c.c_cdr->c.c_car
#define Mcaddar(x)	(x)->c.c_car->c.c_cdr->c.c_cdr->c.c_car
#define Mcadddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_cdr->c.c_car
#define Mcdaaar(x)	(x)->c.c_car->c.c_car->c.c_car->c.c_cdr
#define Mcdaadr(x)	(x)->c.c_cdr->c.c_car->c.c_car->c.c_cdr
#define Mcdadar(x)	(x)->c.c_car->c.c_cdr->c.c_car->c.c_cdr
#define Mcdaddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_car->c.c_cdr
#define Mcddaar(x)	(x)->c.c_car->c.c_car->c.c_cdr->c.c_cdr
#define Mcddadr(x)	(x)->c.c_cdr->c.c_car->c.c_cdr->c.c_cdr
#define Mcdddar(x)	(x)->c.c_car->c.c_cdr->c.c_cdr->c.c_cdr
#define Mcddddr(x)	(x)->c.c_cdr->c.c_cdr->c.c_cdr->c.c_cdr

#include "prelink.h"

#define prof_block(x) ({\
      sigset_t prof,old;						\
      int r;								\
      sigemptyset(&prof);						\
      sigaddset(&prof,SIGPROF);						\
      sigprocmask(SIG_BLOCK,&prof,&old);				\
      r=x;								\
      sigprocmask(SIG_SETMASK,&old,NULL);				\
      r;})

#define psystem(x) prof_block(vsystem(x))
#define pfork() prof_block(fork())
#define pvfork() prof_block(vfork())

#include "error.h"

#if __GNU_MP_VERSION > 4 || (__GNU_MP_VERSION == 4 && __GNU_MP_VERSION_MINOR >= 2)
extern void __gmp_randget_mt ();
extern void __gmp_randclear_mt ();
extern void __gmp_randiset_mt ();

typedef struct {void *a,*b,*c,*d;} gmp_randfnptr_t;
EXTER gmp_randfnptr_t Mersenne_Twister_Generator_Noseed;
#endif

/* #define BV_BITS CHAR_SIZE */
/* #define BV_ALLOC (BV_BITS*SIZEOF_LONG) */
#define BV_BITS (CHAR_SIZE*SIZEOF_LONG)
#define BV_ALLOC BV_BITS
#ifdef WORDS_BIGENDIAN
#define BV_BIT(i) (1L<<(BV_BITS-1-((i)%BV_BITS)))
#else
#define BV_BIT(i) (1L<<((i)%BV_BITS))
#endif
#define BITREF(x,i) ({ufixnum _i=(i);(BV_BIT(_i)&(x->bv.bv_self[_i/BV_BITS])) ? 1 : 0;})
#define SET_BITREF(x,i)   ({ufixnum _i=(i);(x->bv.bv_self[_i/BV_BITS]) |= BV_BIT(_i);})
#define CLEAR_BITREF(x,i) ({ufixnum _i=(i);(x->bv.bv_self[_i/BV_BITS]) &= ~BV_BIT(_i);})
#define BIT_MASK(n_) ({char _n=n_;(_n==BV_BITS ? -1L : BV_BIT(_n)-1);})

#define VLEN(a_) ({object _a=(a_);(_a)->v.v_hasfillp ? (_a)->v.v_fillp : (_a)->v.v_dim;})
#define VFILLP_SET(a_,b_)					\
  ({object _a=(a_);ufixnum _b=(b_);				\
    if (_a->v.v_hasfillp) _a->v.v_fillp=_b;})
#define VSET_MAX_FILLP(a_) ({object _x=(a_);VFILLP_SET(_x,_x->v.v_dim);})

#define ADISP(a_) ({object _a=(a_);(_a)->a.a_adjustable ? (_a)->a.a_displaced : Cnil;})
#define SET_ADISP(a_,b_) ({object _a=(a_);if ((_a)->a.a_adjustable) (_a)->a.a_displaced=(b_);})

#define str(a_)								\
  ({string_register->st.st_self=(a_);					\
    string_register->st.st_dim=strlen(string_register->st.st_self);	\
    string_register;})

#define BLOCK_EXCEPTIONS(a_) \
  ({fenv_t env;		     \
  feholdexcept(&env);	     \
  a_;			     \
  fesetenv(&env);})

#define collect(p_,f_) (p_)=&(*(p_)=(f_))->c.c_cdr
#define READ_STREAM_OR_FASD(strm_) \
  type_of(strm_)==t_stream ? read_object_non_recursive(strm_) : fSread_fasd_top(strm_)
