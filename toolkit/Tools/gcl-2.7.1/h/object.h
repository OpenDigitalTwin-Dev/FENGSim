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
	object.h
*/

/*
	Some system constants.
*/

#define	TRUE		1	/*  boolean true value  */
#define	FALSE		0	/*  boolean false value  */


#define NOT_OBJECT_ALIGNED(a_) ({union lispunion _t={.vw=(void *)(a_)};_t.td.emf;})
#define ROUNDUP(x_,y_) (((unsigned long)(x_)+(y_ -1)) & ~(y_ -1))/*FIXME double eval*/
#define ROUNDDN(x_,y_) (((unsigned long)(x_)) & ~(y_ -1))

#undef PAGESIZE
#define	PAGESIZE	(1L << PAGEWIDTH)	/*  page size in bytes  */


#ifndef plong
#define plong long
#endif

union int_object {
  object o; 
  fixnum i;
};
typedef union int_object iobject;
/* union int_object {object *o; fixnum i;}; */

#define	CHCODELIM	256	/*  character code limit  */
				/*  ASCII character set  */
#define RTABSIZE CHCODELIM        /*  read table size  */

/* #define eql_is_eq(a_)    (is_imm_fixnum(a_) || ({enum type _tp=type_of(a_); _tp == t_cons || _tp > t_complex;})) */
/* #define equal_is_eq(a_)  (is_imm_fixnum(a_) || type_of(a_)>t_bitvector) */




#define Msf(obje) (obje)->SF.SFVAL
#define sf(x) Msf(x)
#define sfc(x) ({object _x=x;sf(_x->cmp.cmp_real)+I*sf(_x->cmp.cmp_imag);})


#define Mlf(obje) (obje)->LF.LFVAL
#define lf(x) Mlf(x)
#define lfc(x) ({object _x=x;lf(_x->cmp.cmp_real)+I*lf(_x->cmp.cmp_imag);})












/* EXTER struct character character_table1[256+128] OBJ_ALIGN; /\*FIXME, sync with char code constants above.*\/ */
/* #define character_table (character_table1+128) */
#define code_char(c)    (object)(character_table+((unsigned char)(c)))
#define char_code(obje) (obje)->ch.ch_code
#define char_font(obje) (obje)->ch.ch_font
#define char_bits(obje) (obje)->ch.ch_bits

enum stype {     /*  symbol type  */

  stp_ordinary,  /*  ordinary  */
  stp_constant,  /*  constant  */
  stp_special    /*  special  */

};

/* #define s_fillp  st_fillp */
/* #define s_self   st_self */



#define NOT_OBJECT_ALIGNED(a_) ({union lispunion _t={.vw=(void *)(a_)};_t.td.emf;})

#define Cnil ((object)&Cnil_body)
#define Ct   ((object)&Ct_body)
#define sLnil Cnil
#define sLt Ct

#define NOT_SPECIAL  (fixnum)Cnil



/*
	The values returned by intern and find_symbol.
	File_symbol may return 0.
*/
#define	INTERNAL	1
#define	EXTERNAL	2
#define	INHERITED	3

/*
	All the packages are linked through p_link.
*/
EXTER struct package *pack_pointer;	/*  package pointer  */

#ifdef WIDE_CONS
#define Scdr(a_) (a_)->c.c_cdr
#else
#define Scdr(a_) ({union lispunion _t={.vw=(a_)->c.c_cdr};unmark(&_t);_t.vw;})
#endif

enum httest {   /*  hash table key test function  */
  htt_eq,       /*  eq  */
  htt_eql,      /*  eql  */
  htt_equal,    /*  equal  */
  htt_equalp    /*  equalp  */
};

/* struct htent {      /\*  hash table entry  *\/ */
/*   object hte_key;   /\*  key  *\/ */
/*   object hte_value; /\*  value  *\/ */
/* }; */



typedef struct {
  void *dflt;
  object *namep;
  unsigned char size;
} aet_type_struct;


#define USHORT_GCL(x,i)  (((unsigned short *)(x)->ust.ust_self)[i])
#define  SHORT_GCL(x,i)  ((( short *)(x)->ust.ust_self)[i])

#define UINT_GCL(x,i)  (((unsigned int *)(x)->ust.ust_self)[i])
#define  INT_GCL(x,i)  ((( int *)(x)->ust.ust_self)[i])

#define BV_OFFSET(x) ((type_of(x)==t_bitvector ? x->bv.bv_offset : \
                       type_of(x)== t_array ? x->a.a_offset : 0))

#define SET_BV_OFFSET(x,val) ((type_of(x)==t_bitvector ? x->bv.bv_offset = val : \
                               type_of(x)== t_array ? x->a.a_offset=val : 0))


struct s_data {

  object name;
  fixnum length;
  object raw;
  object included;
  object includes;
  object staticp;
  object print_function;
  object slot_descriptions;
  object slot_position;
  fixnum size;
  object has_holes;

};

#define S_DATA(x) ((struct s_data *)((x)->str.str_self))
#define SLOT_TYPE(def,i) (((S_DATA(def))->raw->ust.ust_self[i]))
#define SLOT_POS(def,i) USHORT_GCL(S_DATA(def)->slot_position,i)
#define STREF(type,x,i) (*((type *)(((char *)((x)->str.str_self))+(i))))
/* we sometimes have to touch the header of arrays or structures
   to make sure the page is writable */
#ifdef SGC
#define SGC_TOUCH(x) (x)->d.e=1 /* if ((x)->d.m) system_error(); (x)->d.m=0 */
#else
#define SGC_TOUCH(x)
#endif
#define STSET(type,x,i,val)  do{SGC_TOUCH(x);STREF(type,x,i) = (val);} while(0)



enum smmode {      /*  stream mode  */
 smm_input,        /*  input  */
 smm_output,       /*  output  */
 smm_io,           /*  input-output  */
 smm_probe,        /*  probe  */
 smm_file_synonym, /*  synonym to file stream  */
 smm_synonym,      /*  synonym  */
 smm_broadcast,    /*  broadcast  */
 smm_concatenated, /*  concatenated  */
 smm_two_way,      /*  two way  */
 smm_echo,         /*  echo  */
 smm_string_input, /*  string input  */
 smm_string_output,/*  string output  */
 smm_user_defined, /*  for user defined */
 smm_socket        /*  Socket stream  */
};

/* for any stream that takes writec_char, directly (not two_way or echo)
   ie. 	 smm_output,smm_io, smm_string_output, smm_socket
 */
#define STREAM_FILE_COLUMN(str) ((str)->sm.sm_int)

/* for smm_echo */
#define ECHO_STREAM_N_UNREAD(strm) ((strm)->sm.sm_int)

/* file fd for socket */
#define SOCKET_STREAM_FD(strm) ((strm)->sm.sm_fd)
#define SOCKET_STREAM_BUFFER(strm) ((strm)->sm.sm_object1)

/*  for     smm_string_input  */
#define STRING_INPUT_STREAM_NEXT(strm) ((strm)->sm.sm_object0->st.st_fillp)
#define STRING_INPUT_STREAM_END(strm) ((strm)->sm.sm_object0->st.st_dim)

/* for smm_two_way and smm_echo */
#define STREAM_OUTPUT_STREAM(strm) ((strm)->sm.sm_object1)
#define STREAM_INPUT_STREAM(strm) ((strm)->sm.sm_object0)

/* for smm_string_{input,output} */
#define STRING_STREAM_STRING(strm) ((strm)->sm.sm_object0)


/* flags */
#define GET_STREAM_FLAG(strm,name) ((strm)->sm.sm_flags & (1<<(name)))
#define SET_STREAM_FLAG(strm,name,val) {if (val) (strm)->sm.sm_flags |= (1<<(name)); else (strm)->sm.sm_flags &= ~(1<<(name));}

#define GCL_MODE_BLOCKING 1
#define GCL_MODE_NON_BLOCKING 0
#define GCL_TCP_ASYNC 1
     
enum gcl_sm_flags {
  gcl_sm_blocking=1,
  gcl_sm_tcp_async,
  gcl_sm_input,
  gcl_sm_output,
  gcl_sm_closed,
  gcl_sm_had_error
  
};

enum chattrib {			/*  character attribute  */
	cat_whitespace,		/*  whitespace  */
	cat_terminating,	/*  terminating macro  */
	cat_non_terminating,	/*  non-terminating macro  */
	cat_single_escape,	/*  single-escape  */
	cat_multiple_escape,	/*  multiple-escape  */
	cat_constituent		/*  constituent  */
};

/* struct rtent {				/\*  read table entry  *\/ */
/* 	enum chattrib	rte_chattrib;	/\*  character attribute  *\/ */
/* 	object		rte_macro;	/\*  macro function  *\/ */
/* 	object		*rte_dtab;	/\*  pointer to the  *\/ */
/* 					/\*  dispatch table  *\/ */
/* 					/\*  NULL for  *\/ */
/* 					/\*  non-dispatching  *\/ */
/* 					/\*  macro character, or  *\/ */
/* 					/\*  non-macro character  *\/ */
/* }; */

enum chatrait {       /*  character attribute  */
 trait_alpha,         /*  alphabetic  */
 trait_digit,         /*  digits      */
 trait_alphadigit,    /*  alpha/digit */
 trait_package,       /*  package mrk */
 trait_plus,          /*  plus sign   */
 trait_minus,         /*  minus sign  */
 trait_ratio,         /*  ratio mrk   */
 trait_exp,           /*  expon mrk   */
 trait_invalid        /*  unreadable  */
};

struct rtent {               /*  read table entry  */
 enum chattrib rte_chattrib; /*  character attribute  */
 enum chatrait rte_chatrait; /*  constituent trait */
 object        rte_macro;    /*  macro function  */
 object        *rte_dtab;    /*  pointer to the  */
                             /*  dispatch table  */
                             /*  NULL for  */
                             /*  non-dispatching  */
                             /*  macro character, or  */
                             /*  non-macro character  */
};





EXTER object def_env1[2],*def_env;
EXTER object src_env1[2],*src_env;


#define address_int ufixnum

/*
 The struct of free lists.
*/
struct freelist {
	FIRSTWORD;
	address_int f_link;
};
#ifndef INT_TO_ADDRESS
#define INT_TO_ADDRESS(x) ((object )(long )x)
#endif

#define F_LINK(x) ((struct freelist *)(long) x)->f_link
#define FL_LINK F_LINK
#define SET_LINK(x,val) F_LINK(x) = (address_int) (val)
#define OBJ_LINK(x) ((object) INT_TO_ADDRESS(F_LINK(x)))
#define PHANTOM_FREELIST(x) ({struct freelist f;(object)((void *)&x+((void *)&f-(void *)&f.f_link));})
#define FREELIST_TAIL(tm_) ({struct typemanager *_tm=tm_;\
      _tm->tm_free==OBJNULL ? PHANTOM_FREELIST(_tm->tm_free) : _tm->tm_tail;})

struct fasd {
  object stream;   /* lisp object of type stream */
  object table;  /* hash table used in dumping or vector on input*/
  object eof;      /* lisp object to be returned on coming to eof mark */
  object direction;    /* holds Cnil or sKinput or sKoutput */
  object package;  /* the package symbols are in by default */
  object index;     /* integer.  The current_dump index on write  */
  object filepos;   /* nil or the position of the start */
  object table_length; /*    On read it is set to the size dump array needed
		     or 0
		     */
  object evald_items;  /* a list of items which have been eval'd and must
			  not be walked by fasd_patch_sharp */
};

#define	FREE	(-1)		/*  free object  */

/*
 Storage manager for each type.
*/
struct typemanager {
  enum type tm_type;             /*  type  */
  long	    tm_size;             /*  element size in bytes  */
  long      tm_nppage;           /*  number per page  */
  object    tm_free;             /*  free list  */
				 /*  Note that it is of type object.  */
  object    tm_tail;             /*  free list tail  */
				 /*  Note that it is of type object.  */
  long	    tm_nfree;            /*  number of free elements  */
  long	    tm_npage;            /*  number of pages  */
  long	    tm_maxpage;          /*  maximum number of pages  */
  char	   *tm_name;             /*  type name  */
  long	    tm_gbccount;         /*  GBC count  */
  object    tm_alt_free;         /*  Alternate free list (swap with tm_free) */
  long      tm_alt_nfree;        /*  Alternate nfree (length of nfree) */
  long	    tm_alt_npage;        /*  number of pages  */
  long      tm_sgc;              /*  this type has at least this many sgc pages */
  long      tm_sgc_minfree;      /*  number free on a page to qualify for being an sgc page */
  long      tm_sgc_max;          /* max on sgc pages */
  long      tm_min_grow;         /* min amount to grow when growing */
  long      tm_max_grow;         /* max amount to grow when growing */
  long      tm_growth_percent;   /* percent to increase maxpages */
  long      tm_percent_free;     /* percent which must be free after a gc for this type */
  long      tm_distinct;         /* pages of this type are distinct */
  float     tm_adjgbccnt;
  long      tm_opt_maxpage;
  enum type tm_calling_type;     /* calling type  */
};


/*
	The table of type managers.
*/
EXTER struct typemanager tm_table[ 32  /* (int) t_relocatable */];

#define tm_of(t) ({struct typemanager *_tm=tm_table+tm_table[t].tm_type;_tm->tm_calling_type=t;_tm;})

/*
	Contiguous block header.
*/
EXTER bool prefer_low_mem_contblock;
struct contblock {            /*  contiguous block header  */

  fixnum cb_size;             /*  size in bytes  */
  struct contblock  *cb_link; /*  contiguous block link  */
};

/*
	The pointer to the contiguous blocks.
*/
EXTER struct contblock *cb_pointer;	/*  contblock pointer  */

/* SGC cont pages: After SGC_start, old_cb_pointer will be a linked
   list of free blocks on non-SGC pages, and cb_pointer will be
   likewise for SGC pages.  CM 20030827*/
EXTER struct contblock *old_cb_pointer;	/*  old contblock pointer when in SGC  */

/*
	Variables for memory management.
*/
EXTER fixnum ncb;   /*  number of contblocks  */
#define ncbpage    tm_table[t_contiguous].tm_npage
#define maxcbpage  tm_table[t_contiguous].tm_maxpage
#define maxrbpage  tm_table[t_relocatable].tm_maxpage
#define cbgbccount tm_table[t_contiguous].tm_gbccount  
  

EXTER long holepage;			/*  hole pages  */
#define nrbpage tm_table[t_relocatable].tm_npage
#define maxrbpage tm_table[t_relocatable].tm_maxpage
#define rbgbccount tm_table[t_relocatable].tm_gbccount
EXTER fixnum new_holepage,starting_hole_div,starting_relb_heap_mult;

EXTER ulfixnum cumulative_allocation,recent_allocation;
EXTER ufixnum wait_on_abort;
EXTER double gc_alloc_min,mem_multiple,gc_page_min,gc_page_max;
EXTER char *multiprocess_memory_pool;

EXTER char *new_rb_start;		/*  desired relblock start after next gc  */
EXTER char *rb_start;           	/*  relblock start  */
EXTER char *rb_end;			/*  relblock end  */
EXTER char *rb_limit;			/*  relblock limit  */
EXTER char *rb_pointer;		/*  relblock pointer  */
EXTER char *rb_start1;		/*  relblock start in copy space  */
EXTER char *rb_pointer1;		/*  relblock pointer in copy space  */

#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>
#ifndef INLINE
#define INLINE
#endif

INLINE ufixnum
rb_size(void) {
  return rb_end-rb_start;
}

INLINE bool
rb_high(void) {
  return rb_pointer>=rb_end&&rb_size();
}

INLINE char *
rb_begin(void) {
  return rb_high() ? rb_end : rb_start;
}

INLINE bool
rb_emptyp(void) {
  return rb_pointer == rb_begin();
}

INLINE ufixnum
ufmin(ufixnum a,ufixnum b) {
  return a<=b ? a : b;
}

INLINE ufixnum
ufmax(ufixnum a,ufixnum b) {
  return a>=b ? a : b;
}

INLINE int
oemsg(int fd,const char *s,...) {
  va_list args;
  ufixnum n=0;
  void *v=NULL;
  va_start(args,s);
  n=vsnprintf(v,n,s,args)+1;
  va_end(args);
  v=alloca(n);
  va_start(args,s);
  vsnprintf(v,n,s,args);
  va_end(args);
  return write(fd,v,n-1) ? n : -1;
}

#define omsg(a_...) oemsg(1,a_)
#define emsg(a_...) oemsg(2,a_)

EXTER char *heap_end;			/*  heap end  */
EXTER char *core_end;			/*  core end  */
EXTER 
char *tmp_alloc;

/* make f allocate enough extra, so that we can round
   up, the address given to an even multiple.   Special
   case of size == 0 , in which case we just want an aligned
   number in the address range
   */

#define ALLOC_ALIGNED(f, size,align) \
  ({ufixnum _size=size,_align=align;_align <= sizeof(plong) ? (char *)((f)(_size)) :	\
    (tmp_alloc = (char *)((f)(_size+(_size ?(_align)-1 : 0)))+(_align)-1 ,	\
    (char *)(_align * (((unsigned long)tmp_alloc)/_align)));})
#define AR_ALLOC(f,n,type) (type *) \
  (ALLOC_ALIGNED(f,(n)*sizeof(type),sizeof(type)))


#define	RB_GETA		PAGESIZE


#ifdef AV
#define	STATIC	register
#endif

#define	TIME_ZONE	(-9)

/*  For IEEEFLOAT, the double may have exponent in the second word
(little endian) or first word.*/

#if !defined(DOUBLE_BIGENDIAN)
#define HIND 1  /* (int) of double where the exponent and most signif is */
#define LIND 0  /* low part of a double */
#else /* big endian */
#define HIND 0
#define LIND 1
#endif
/* #ifndef VOL */
#define VOL volatile
/* #endif */


#define	isUpper(xxx)	(((xxx)&0200) == 0 && isupper((int)xxx))
#define	isLower(xxx)	(((xxx)&0200) == 0 && islower((int)xxx))
#define	isDigit(xxx)	(((xxx)&0200) == 0 && isdigit((int)xxx))
enum ftype {f_object,f_fixnum};
EXTER char *alloca_val;
#define ALLOCA_CONS_ALIGN(n) ({alloca_val=ZALLOCA((n)*sizeof(struct cons)+sizeof(alloca_val));if (((unsigned long)alloca_val)&sizeof(alloca_val)) alloca_val+=sizeof(alloca_val);alloca_val;})
#define ON_STACK_CONS(x,y) (ALLOCA_CONS_ALIGN(1), on_stack_cons(x,y)) 



/*FIXME -- this is an effort to minimize uninitialized garbage in the
  stack.  THe only comprehensive solution appears to be to wipe the
  stack frame on each function call.  Doubling the overhead of every
  function call appears too expensive, though it has not been
  thoroughly tested.  It is also quesitonable how portable the
  wipe_stack algorithm is.  For now, we've minimized the issue by
  moving the cstack mark origin to the frame right above toplevel.
  20050609 CM. */

/* #include <string.h> */
#define CSP (CSTACK_ALIGNMENT-1)
#if CSTACK_DIRECTION == -1
#define ZALLOCA(n) ({fixnum _x=0,_y=0,_n=((n)+CSP)&~CSP;void *v=NULL;v=alloca(_n+_x+_y);bzero(v,_n+_x+_y); v;})
#else
#define ZALLOCA(n) ({fixnum _x=0,_y=0,_n=((n)+CSP)&~CSP;void *v=NULL;v=alloca(_n+_x+_y);bzero(v,_n+_x+_y); v;})
#endif
/* #define ZALLOCA(n) ({fixnum _x=0,_y=0,_n=((n)+CSP)&~CSP;void *v=NULL;v=alloca(_n+_x+_y);wipe_stack(v+_n); v;}) */
/* #else */
/* #define ZALLOCA(n) ({fixnum _x=0,_y=0,_n=((n)+CSP)&~CSP;void *v=NULL;v=alloca(_n+_x+_y);wipe_stack(v); v;}) */
/* #endif */
#define ZALLOCA1(v,n) ((v)=alloca((n)),__builtin_bzero((v),((n))))

#ifdef DONT_COPY_VA_LIST
#define COERCE_VA_LIST(new,vl,n) new = (object *) (vl)
#else
#define COERCE_VA_LIST(new,vl,n) \
 object Xxvl[65]; \
 {int i; \
  new=Xxvl; \
  if (n >= 65) FEerror("Too plong vl",0); \
  for (i=0 ; i < (n); i++) new[i]=va_arg(vl,object);}
#endif

#ifdef DONT_COPY_VA_LIST
#error Cannot set DONT_COPY_VA_LIST in ANSI C
#else
#define COERCE_VA_LIST_NEW(new,fst,vl,n) \
 object Xxvl[65]; \
 {int i; \
  new=Xxvl; \
  if (n >= 65) FEerror("va_list too long",0); \
  for (i=0 ; i < (n); i++) new[i]=i ? va_arg(vl,object) : fst;}
#define COERCE_VA_LIST_KR_NEW(new,fst,vl,n) \
 object Xxvl[65]; \
 {int i; \
  new=Xxvl; \
  if (n >= 65) FEerror("va_list too long",0); \
  for (i=0 ; i < (n); i++) new[i]=i||fst==OBJNULL ? va_arg(vl,object) : fst;}
#endif



#define make_si_vfun(s,f,min,max) \
  make_si_vfun1(s,f,min | (max << 8))

/* Number of args supplied to a variable arg t_vfun
 Used by the C function to set optionals */

#define  VFUN_NARGS fcall.argd
#define  FUN_VALP   fcall.valp
#define RETURN4(x,y,z,w)  RETURN(3,object,x,(RV(y),RV(z),RV(w)))
#define RETURN3(x,y,z)  RETURN(3,object,x,(RV(y),RV(z)))
#define RETURN2(x,y)    RETURN(2,object,x,(RV(y)))
#define RETURN3I(x,y,z) RETURN(3,fixnum,x,(RV(y),RV(z)))
#define RETURN2I(x,y)   RETURN(2,fixnum,x,(RV(y)))
/* #define RETURN1(x) RETURN(1,object,x,) */
#define RETURN1(x) return(x)
#define RETURN0 do {vs_top=vals ? (object *)vals-1 : base;return Cnil;} while (0)

#define RV(x) ({if (_p) *_p++ = x;})

#define RETURNI(n,val1,listvals) RETURN(n,int,val1,listvals)
#define RETURNO(n,val1,listvals) RETURN(n,object,val1,listvals)

/* eg: RETURN(3,object,val1,(RV(val2),RV(val3))) */
#undef RETURN
#define RETURN(n,typ,val1,listvals) \
  do{typ _val1 = val1; object *_p=(object *)vals; listvals; vs_top=_p ? _p : base; return _val1;} while(0)
/* #define CALL(n,form) (VFUN_NARGS=n,form) */




EXTER object sSlambda_block_expanded;

# ifdef __GNUC__ 
# define assert(ex)\
{if (!(ex)){(void)fprintf(stderr, \
    "Assertion failed: file \"%s\", line %d\n", __FILE__, __LINE__);gcl_abort();}}
# else
# define assert(ex)
# endif

#ifndef CHECK_INTERRUPT
#  define CHECK_INTERRUPT   if (signals_pending) raise_pending_signals(sig_safe)
#endif

#define BEGIN_NO_INTERRUPT \
 plong old_signals_allowed = signals_allowed; \
  signals_allowed = 0

#define END_NO_INTERRUPT \
  ({signals_allowed = old_signals_allowed; if (signals_pending) raise_pending_signals(sig_use_signals_allowed_value);})
/* could add:   if (signals_pending)
   raise_pending_signals(sig_use_signals_allowed_value) */


#define END_NO_INTERRUPT_SAFE \
  signals_allowed = old_signals_allowed; \
  if (signals_pending) \
    do{ if(signals_allowed ==0) /* should not get here*/gcl_abort(); \
   raise_pending_signals(sig_safe)}while(0)


EXTER unsigned plong signals_allowed, signals_pending;

#define endp(a) (consp(a) ? FALSE : ((a)==Cnil ? TRUE : ({TYPE_ERROR((a),sLlist);FALSE;})))


extern void *stack_alloc_start,*stack_alloc_end;



#define stack_alloc_on(n_) ({void *_v=alloca(n_*PAGESIZE+OBJ_ALIGNMENT-1);\
                             if (_v) {\
                                stack_alloc_start=(void *)ROUNDUP(_v,OBJ_ALIGNMENT);\
                                memset(_v,0,stack_alloc_start-_v);\
                                _v+=n_*PAGESIZE+OBJ_ALIGNMENT-1;\
                                stack_alloc_end=(void *)ROUNDDN(_v,OBJ_ALIGNMENT);\
                                memset(stack_alloc_end,0,_v-stack_alloc_end);\
                             };\
                           })
     
#define stack_alloc_off() ({stack_alloc_start=stack_alloc_end=NULL;})
            
#define maybe_alloc_on_stack(n_,t_) ({void *_v=OBJNULL;\
                                      if (stack_alloc_start) {\
                                         unsigned _n=ROUNDUP(n_,OBJ_ALIGNMENT);\
                                         if (stack_alloc_end-stack_alloc_start>_n) {\
                                           _v=stack_alloc_start;\
                                           stack_alloc_start+=_n;\
                                           if (t_>=0) set_type_of(_v,t_);\
                                         } else stack_alloc_off();\
                                      }\
                                      _v;})


#define stack_pages_left ({fixnum _val;int _w;\
                           _val=cs_limit-&_w;\
                           _val=_val<0 ? -_val : _val;\
                           _val=(_val>>PAGEWIDTH);})

#define myfork() ({int _p[2],_j=0;pid_t _pid;\
                   pipe(_p);\
                   _pid=fork();\
                   if (!_pid) { \
                      object _x=sSAchild_stack_allocA->s.s_dbind;\
                      enum type _tp=type_of(_x);\
                      float _fac= _tp==t_shortfloat ? sf(_x) : (_tp==t_longfloat ? lf(_x) : 0.8);\
                      fixnum _n=_fac*stack_pages_left;\
                      if (_n>0) stack_alloc_on(_n);\
                      close(0);close(1);close(2);\
                      _j=1;\
                   } \
                   close(_p[1-_j]);\
		   make_cons(make_fixnum(_pid),make_fixnum(_p[_j]));})

#define make_fd_stream(fd_,mode_,st_,buf_) ({object _x=alloc_object(t_stream);\
                                            _x->sm.sm_mode=mode_;\
                                            _x->sm.sm_fp=fdopen(fd_,st_);\
                                            _x->sm.sm_buffer=buf_;\
                                            setbuf(_x->sm.sm_fp,_x->sm.sm_buffer);\
                                            _x->sm.sm_object0=sLcharacter;\
                                            _x->sm.sm_object1=Cnil;\
                                            _x->sm.sm_fd=fd_;\
                                            _x;})

#define writable_ptr(a_) (((unsigned long)(a_)>=(unsigned long)data_start && (void *)(a_)<(void *)heap_end) || is_imm_fixnum(a_))

#define write_pointer_object(a_,b_) fSwrite_pointer_object(a_,b_)

#define read_pointer_object(a_) fSread_pointer_object(a_)

#define fixnum_float_contagion(a_,b_) \
  ({register object _a=(a_),_x=_a,_b=(b_);\
    register enum type _ta=type_of(_a),_tb=type_of(_b);\
    if (_ta!=_tb)\
       switch(_ta) {\
          case t_shortfloat: if (_tb==t_longfloat) _x=make_longfloat(sf(_a)); break;\
          case t_fixnum: \
              switch(_tb) {\
                  case t_longfloat:  _x=make_longfloat (fix(_a));break;\
                  case t_shortfloat: _x=make_shortfloat(fix(_a));break;\
                  default: break;}\
          break;\
          default: break;}\
   _x;})
                                        

#define FEerror(a_,b_...)   Icall_error_handler(sLerror,null_string,\
                            4,sKformat_control,make_simple_string(a_),sKformat_arguments,list(b_))
#define TYPE_ERROR(a_,b_)   Icall_error_handler(sLtype_error,null_string,\
                            4,sKdatum,(a_),sKexpected_type,(b_))
#define FEinvalid_form(a_,b_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string(a_),\
                      sKformat_arguments,list(1,(b_)))
#define FEinvalid_variable(a_,b_) FEinvalid_form(a_,b_)
#define FEwrong_type_argument(a_,b_) TYPE_ERROR(b_,a_)

#define VA_ARG(_a,_f,_n) \
  ({object _z=_f!=OBJNULL ? _f : va_arg(_a,object);\
    _f=OBJNULL;_n+=((_n<0) ? 1 : -1);_z;})

#define NEXT_ARG(_n,_a,_l,_f,_d)\
  ({object _z;\
    switch (_n) {\
    case -1: _l=VA_ARG(_a,_f,_n);						\
    case  0: if (_l==Cnil) _z=_d; else {_z=_l->c.c_car;_l=_l->c.c_cdr;};break;\
    default: _z=VA_ARG(_a,_f,_n);break;					\
    } _z;}) 

#define INIT_NARGS(_n) ({fixnum _v=VFUN_NARGS;_v=_v<0 ? _v+_n : _v-_n;_v;})

#define object_to_object(x) x

#define proper_list(a) (type_of(a)==t_cons || (a)==Cnil) /*FIXME*/

#define IMMNIL(x) (is_imm_fixnum(x)||x==Cnil)

/* #define eql_is_eq(a_) (is_imm_fixnum(a_)||valid_cdr(a_)||(a_->d.t>t_complex)) */

#define eql(a_,b_)    ({register object _a=(a_);register object _b=(b_);\
      _a==_b ? TRUE : (eql_is_eq(_a)||eql_is_eq(_b)||_a->d.t!=_b->d.t ? FALSE : eql1(_a,_b));})
#define equal(a_,b_)  ({register object _a=(a_);register object _b=(b_);_a==_b ? TRUE : (IMMNIL(_a)||IMMNIL(_b) ? FALSE : equal1(_a,_b));})
#define equalp(a_,b_) ({register object _a=(a_);register object _b=(b_);_a==_b ? TRUE : (_a==Cnil||_b==Cnil ? FALSE : equalp1(_a,_b));})
