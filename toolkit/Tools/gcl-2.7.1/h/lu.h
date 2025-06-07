#include "pbits.h"

typedef long long              lfixnum;
typedef unsigned long long    ulfixnum;

typedef long            fixnum;
typedef unsigned long   ufixnum;

#ifndef WORDS_BIGENDIAN

/* high bit must be clear to distinguish from high immediate fixnum*/
#define FRSTWRD(t_,b_,a_...) ufixnum    e:1,m:1,f:1,    t_:5,t:5,st:3,a_,b_,h:1
#define FRSTWRDF(t_,a_...)   ufixnum    e:1,m:1,f:1,    t_:5,t:5,st:3,a_,h:1
#define FIRSTWORD            ufixnum    e:1,m:1,f:1,    tt:5,t:5,st:3,w:LM(17),h:1
#define FSTPWORD             ufixnum  emf:3,            tp:10,   st:3,w:LM(17),h:1
#define MARKWORD             ufixnum    e:1,   mf:2,    tt:5,t:5,xx:LM(14),h:1
#define SGCMWORD             ufixnum    e:1,mf:2,       tt:5,t:5,xx:LM(14),h:1
#define TYPEWORD             ufixnum  emf:3,            tt:5,t:5,xx:LM(14),h:1

#else

/* high bit must be clear to distinguish from high immediate fixnum*/
#define FRSTWRD(t_,b_,a_...) ufixnum h:1,b_,a_,   st:3,t:5,t_:5,    f:1,m:1,e:1
#define FRSTWRDF(t_,a_...)   ufixnum h:1,a_,   st:3,t:5,t_:5,    f:1,m:1,e:1
#define FIRSTWORD            ufixnum h:1,w:LM(17),st:3,t:5,tt:5,    f:1,m:1,e:1
#define FSTPWORD             ufixnum h:1,w:LM(17),st:3,tp:10,             emf:3
#define MARKWORD             ufixnum h:1,xx:LM(14),    t:5,tt:5,       mf:2,e:1
#define SGCMWORD             ufixnum h:1,xx:LM(14),    t:5,tt:5,       mf:2,e:1
#define TYPEWORD             ufixnum h:1,xx:LM(14),    t:5,tt:5,          emf:3

#endif

#if SIZEOF_LONG < 8
#define SPAD object spad
#else
#define SPAD
#endif

typedef union lispunion * object;

typedef struct cons * htent;
typedef struct rtent * rtentp;

typedef object (*ofunc)();
typedef void   (*vfunc)();

typedef object integer;
typedef object keyword;
typedef object direl;
typedef object plist;
typedef object pack;
typedef object real;
typedef object string;
typedef object structure;
typedef object symbol;
typedef float  shortfloat;
typedef double longfloat;
typedef float  complex fcomplex;
typedef double complex dcomplex;
#undef bool
typedef int bool;
typedef unsigned short int ushort;
typedef unsigned int uint;


#if 2 * SIZEOF_INT == SIZEOF_LONG
typedef int             hfixnum;
typedef unsigned int   uhfixnum;
#elif 2 * SIZEOF_SHORT == SIZEOF_LONG
typedef short           hfixnum;
typedef unsigned short uhfixnum;
#else
#error No hfixnum size detected
#endif

/* typedef char character; */
typedef unsigned char uchar;

#if 4 * SIZEOF_SHORT == SIZEOF_LONG
typedef short           qfixnum;
typedef unsigned short uqfixnum;
#elif 4 * SIZEOF_CHAR == SIZEOF_LONG
typedef char            qfixnum;
typedef unsigned char   uqfixnum;
#else
#error No qfixnum size detected
#endif

struct fixnum_struct {

  FIRSTWORD;

  fixnum FIXVAL;  /*  fixnum value  */

};
struct shortfloat_struct {

  FIRSTWORD;

  shortfloat	SFVAL;	/*  shortfloat value  */

};
struct longfloat_struct {

  FIRSTWORD;

  longfloat LFVAL; /*  longfloat value  */
  SPAD;

};
struct bignum {

  FIRSTWORD;

  __mpz_struct big_mpz_t;  /*defined by gmp/mgmp.h*/

};
struct ratio {

  FIRSTWORD;

  integer rat_den; /*  denominator, must be an integer  */
  integer rat_num; /*  numerator,  must be an integer  */
  SPAD;

};
struct ocomplex {

  FIRSTWORD;

  real cmp_real; /*  real part, must be a number  */
  real cmp_imag; /*  imaginary part, must be a number  */
  SPAD;

};

#define j(a_,b_) a_##b_
#define J(a_,b_) j(a_,b_)

#define ARRAY_DIMENSION_BITS 28
#define ARRAY_DIMENSION_LIMIT (1UL<<ARRAY_DIMENSION_BITS)
#define FILLP_WORD(a_)				\
  ufixnum a_:ARRAY_DIMENSION_BITS;		\
  ufixnum J(fppad,a_):LM(ARRAY_DIMENSION_BITS)

struct character {

  FIRSTWORD;

  object            ch_name;
  ufixnum           pad5;
  uchar             ch_code;/*need address of this*/
  uchar             ch_font;
  uchar             ch_bits;
  uchar             cp1;/*cygwin won't take a bit field pad here*/
#if SIZEOF_LONG == 8
  uchar             cp2,cp3,cp4,cp5;
#endif
  ufixnum           pad1;
  ufixnum           pad2;
  ufixnum           pad3;
  ufixnum           pad4;

};

struct symbol {

  FIRSTWORD;

  fixnum     s_sfdef;        /*  special form definition, coincides with c_car  */
  object     s_dbind;        /*  dynamic binding  */
  string     s_name;         /*  symbol-name */
  object     s_gfdef;        /*  global function definition, for a macro, its expansion function */
  plist      s_plist;        /*  property list  */
  pack       s_hpack;        /*  home package, Cnil for uninterned symbols  */
  uhfixnum   s_pad2:HM(2);   /*  unused  */
  uhfixnum   s_stype:2;      /*  symbol type, of enum stype  */
  uhfixnum   s_pad3:HM(1);   /*  unused  */
  uhfixnum   s_mflag:1;      /*  macro flag  */
  fixnum     s_hash;         /*  cached hash code */
  SPAD;
};
struct package {

  FIRSTWORD;

  string          p_name;              /*  package name, a string  */
  plist           p_nicknames;         /*  nicknames, list of strings  */
  plist           p_shadowings;        /*  shadowing symbol list  */
  plist           p_uselist;           /*  use-list of packages  */
  plist           p_usedbylist;        /*  used-by-list of packages  */
  plist          *p_internal;          /*  hashtable for internal symbols  */
  plist          *p_external;          /*  hashtable for external symbols  */
  ufixnum         p_internal_size;     /* size of internal hash table*/
  ufixnum         p_external_size;     /* size of external hash table */
  ufixnum         p_internal_fp;       /* [rough] number of symbols */
  ufixnum         p_external_fp;       /* [rough]  number of symbols */
  struct package *p_link;              /*  package link  */
  SPAD;

};
struct cons {

  /*   FIRSTWORD; Two word cons, 20050609, CM */

  object c_cdr;  /*  cdr  */
  object c_car;  /*  car  */

};
struct hashtable {           /*  hash table header  */

  FIRSTWORD;

  htent         ht_self;            /*  pointer to the hash table  */
  real          ht_rhsize;          /*  rehash size  */
  real          ht_rhthresh;        /*  rehash threshold  */
  ufixnum       ht_pad1:3;          /*  unused  */
  ufixnum       ht_static:1;        /*  unused  */
  ufixnum       ht_nent:LM(4);      /*  number of entries  */
  ufixnum       ht_pad2:4;          /*  hash table size  */
  ufixnum       ht_size:LM(4);      /*  hash table size  */
  uhfixnum      ht_test:2;          /*  key test function, of enum httest  */
  uhfixnum      ht_pad3:HM(2);      /*  unused */
  uhfixnum      ht_pad4;            /*  unused */
  ufixnum       ht_pad5:4;          /*  unused */
  ufixnum       ht_max_ent:LM(4);   /*  max entries */
  htent         ht_cache;           /*  gethash cache */
  SPAD;

};

#define j(a_,b_) a_##b_
#define J(a_,b_) j(a_,b_)

#define ARRAY_RANK_BITS 6
/* #define ARRAY_RANK_LIMIT ((1UL<<ARRAY_RANK_BITS)-1)/\*FIXME?*\/ */
#define ARRAY_RANK_LIMIT (1UL<<ARRAY_RANK_BITS)

#if SIZEOF_LONG == 8
#ifdef WORDS_BIGENDIAN
#define ARRAYWORD(b_,c_)						\
  FRSTWRDF(J(b_,J(c_,elttype)),						\
	   J(b_,J(c_,eltmode)):3,					\
	   J(b_,J(c_,dim)):ARRAY_DIMENSION_BITS,			\
	   J(b_,J(c_,hasfillp)):1,					\
	   J(b_,J(c_,writable)):1,					\
	   J(b_,J(c_,rank)):ARRAY_RANK_BITS,				\
	   pd1:1,							\
	   J(b_,J(c_,adjustable)):1,					\
	   J(b_,J(c_,offset)):3,					\
	   J(b_,J(c_,eltsize)):3)
#else
#define ARRAYWORD(b_,c_)						\
  FRSTWRDF(J(b_,J(c_,elttype)),						\
	   J(b_,J(c_,hasfillp)):1,					\
	   J(b_,J(c_,writable)):1,					\
	   J(b_,J(c_,rank)):ARRAY_RANK_BITS,				\
	   pd1:1,							\
	   J(b_,J(c_,adjustable)):1,					\
	   J(b_,J(c_,offset)):3,					\
	   J(b_,J(c_,eltsize)):3,					\
	   J(b_,J(c_,eltmode)):3,					\
	   J(b_,J(c_,dim)):ARRAY_DIMENSION_BITS)
#endif

#define atem(a_,b_,c_)				\
  ARRAYWORD(b_,c_);				\
  a_       *J(b_,J(c_,self))

#else

#define ARRAYWORD(b_,c_)						\
  FRSTWRDF(J(b_,J(c_,elttype)),						\
	   J(b_,J(c_,hasfillp)):1,					\
	   J(b_,J(c_,adjustable)):1,					\
	   J(b_,J(c_,writable)):1,					\
	   J(b_,J(c_,offset)):3,					\
	   J(b_,J(c_,eltsize)):3,					\
	   J(b_,J(c_,rank)):ARRAY_RANK_BITS)

#define atem(a_,b_,c_)					\
  ARRAYWORD(b_,c_);					\
  a_       *J(b_,J(c_,self));				\
  ufixnum   J(b_,J(c_,dim)):ARRAY_DIMENSION_BITS;	\
  ufixnum   J(b_,J(c_,eltmode)):3;			\
  ufixnum   pad1:LM(31)
/*FIXME SPAD*/
#endif

#define dimstempl(b_,c_)			\
  ufixnum  *J(b_,J(c_,dims))

#define atempl(a_,b_,c_)			\
  atem(a_,b_,c_);				\
  dimstempl(b_,c_)

#define vfptempl(b_,c_)						\
  FILLP_WORD(J(b_,J(c_,fillp)));				\
  plist     J(b_,J(c_,displaced))

#define vtempl(a_,b_,c_)				\
  atem(a_,b_,c_);					\
  vfptempl(b_,c_)


struct unadjarray {

  atempl(object,sa_,);

};

struct adjarray {

  atempl(object,,m);
  plist     aadj_displaced;
  SPAD;

};

struct array {

  atempl(object,a_,);
  plist     a_displaced;
  SPAD;

};


/* struct unadjmatrix { */

/*   atempl(object,smt_,); */

/* }; */

/* struct adjmatrix { */

/*   atem(object,,); */
/*   dimstempl(madj_,); */

/* }; */

struct matrix {

  atem(object,,m);
  dimstempl(mt_,);
  plist     displaced;
  SPAD;

};


struct unadjvector {

  atem(object,sv_,);
  SPAD;

};

struct vector {

  vtempl(object,v_,);
  SPAD;

};

struct adjvector {

  atem(object,,);
  vfptempl(vadj_,);
  SPAD;

};

struct unadjstring {

  atem(char,sst_,);
  SPAD;

};

struct string {

  vtempl(char,st_,);
  SPAD;

};

struct unadjbitvector {

  atem(ufixnum,sbv_,);
  SPAD;

};

struct bitvector {

  vtempl(ufixnum,bv_,);
  SPAD;

};

struct ustring {

  vtempl(uchar,ust_,);
  SPAD;

};

struct structure {  /*  structure header  */

  FIRSTWORD;

  structure  str_def;  /*  structure definition (a structure)  */
  object    *str_self; /*  structure self  */
  SPAD;

};

struct strstd {  /*  structure header  */

  FIRSTWORD;

  structure  strstd_sdef;  /*  structure definition (a structure)  */
  object    *strstd_sself; /*  structure self  */
  SPAD;

};

struct stream {

  FIRSTWORD;

  void            *sm_fp;          /*  file pointer  */
  object           sm_object0;     /*  some object  */
  object           sm_object1;     /*  some object */
  char            *sm_buffer;      /*  ptr to BUFSIZE block of storage */
  ufixnum          sm_mode:4;      /*  stream mode  */
  ufixnum          sm_flags:6;     /*  flags from gcl_sm_flags */
  ufixnum          sm_fd:6;        /*  file descriptor */
  ufixnum          sm_int:LM(16);  /*  general purpose integer */
     
};
struct random {

  FIRSTWORD;

  __gmp_randstate_struct  rnd_state;

};
struct readtable {       /*  read table  */

  FIRSTWORD;

  rtentp        rt_self; /*  read table itself  */
  keyword       rt_case;
  SPAD;

};
struct pathname {

  FIRSTWORD;

  direl  pn_host;      /*  host  */
  direl  pn_device;    /*  device  */
  plist  pn_directory; /*  directory  */
  direl  pn_name;      /*  name  */
  direl  pn_type;      /*  type  */
  direl  pn_version;   /*  version  */
  string pn_namestring;/*  cached namestring */

};

struct function {

  FRSTWRD(tt,
#if SIZEOF_LONG == 8
	  fw:LM(39),
	  fun_minarg:6,    /* required arguments */
	  fun_maxarg:6,    /* maximum arguments */
	  pd:4,
	  fun_neval:5,     /* maximum extra values set */
	  fun_vv:1         /* variable number of values */
#else
	  fw:LM(29),
	  fun_minarg:6,    /* required arguments */
	  fun_maxarg:6     /* maximum arguments */
#endif
	  );

  ofunc   fun_self;         /* executable code */
  object  fun_data;         /* cfddata structure */
  plist   fun_plist;        /* sig callees callers src file */
#if SIZEOF_LONG == 8
  ufixnum fun_argd;         /* arg/return type checking */
#else
  ufixnum fun_neval:5;      /* maximum extra values set */
  ufixnum fun_vv:1;         /* variable number of values */
  ufixnum fun_argd:LM(6);   /* arg/return type checking */
#endif
  object *fun_env;

};

struct cfdata {

  FRSTWRD(tt,
	  cfw:LM(18),
	  cfd_prof:1       /* profiling */
	  );

  char   *cfd_start;             /* beginning of contblock for fun */
  FILLP_WORD(cfd_size);
  FILLP_WORD(cfd_fillp);
  object *cfd_self;              /* body */
  plist   cfd_dlist;
  string  cfd_name;
  SPAD;

};

struct spice {

  FIRSTWORD;

  fixnum spc_dummy;

};

struct dummy      {FIRSTWORD;};
struct ff         {ufixnum ff;};
struct fstpw      {FSTPWORD;};
union  fstp       {ufixnum ff;struct fstpw t;};
struct mark       {MARKWORD;};
struct typew      {TYPEWORD;};
struct sgcm       {SGCMWORD;};

/*
 Definition of lispunion.
*/
union lispunion {
 struct fixnum_struct     FIX; /*  fixnum  */
 struct bignum            big; /*  bignum  */
 struct ratio             rat; /*  ratio  */
 struct shortfloat_struct  SF; /*  short floating-point number  */
 struct longfloat_struct   LF; /*  plong floating-point number  */
 struct ocomplex          cmp; /*  complex number  */
 struct character          ch; /*  character  */
 struct symbol              s; /*  symbol  */
 struct package             p; /*  package  */
 struct cons                c; /*  cons  */
 struct hashtable          ht; /*  hash table  */
 struct unadjstring       sst; /*  simple string  */
 struct string             st; /*  string  */
 /* struct stdesig            sd; /\*  array character symbol -- phony for c package ref  *\/ */
 struct ustring           ust; /*  unsigned char string  */
 struct unadjbitvector    sbv; /*  simple bit-vector  */
 struct bitvector          bv; /*  bit-vector  */
 struct unadjvector        sv; /*  simple vector  */
 struct vector              v; /*  vector  */
 struct cfdata            cfd; /*  compiled fun data */
 struct adjvector        vadj; /*  adjustable vector  */
 /* struct unadjarray         sa; /\*  simple array  *\/ */
 /* struct unadjmatrix       smt; /\*  simple vector  *\/ */
 struct array               a; /*  array  */
 struct matrix             mt; /*  matrix  */
 struct adjarray         aadj; /*  adjustable array  */
 struct structure         str; /*  structure  */
 struct strstd         strstd; /*  structure  */
 struct stream             sm; /*  stream  */
 struct random            rnd; /*  random-states  */
 struct readtable          rt; /*  read table  */
 struct pathname           pn; /*  path name  */
 struct function          fun; /*  function */
 struct spice             spc; /*  spice  */

 struct dummy               d; /*  dummy  */
 struct fstpw            fstp; /*  fast type  */
 struct ff                 ff; /*  fast type  */
 struct mark               md; /*  mark dummy  */
 struct sgcm              smd; /*  sgc mark dummy  */
 struct typew              td; /*  type dummy  */
 fixnum                    fw;
 void *                    vw;

};
