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
  GBC.c
  IMPLEMENTATION-DEPENDENT
*/

#define	DEBUG

#define IN_GBC
#define NEED_MP_H
#include <string.h>
#include <stdlib.h>
#include "include.h"
#include "page.h"


#ifdef SGC
static void
sgc_sweep_phase(void);

static void
sgc_mark_phase(void);

static fixnum
sgc_count_read_only(void);

#endif

static void
mark_c_stack(jmp_buf, int, void (*)(void *,void *,int));

static void
mark_contblock(void *, int);

/* the following in line definitions seem to be twice as fast (at
   least on mc68020) as going to the assembly function calls in bitop.c so
   since this is more portable and faster lets use them --W. Schelter
   These assume that DBEGIN is divisible by 32, or else we should have
   #define Shamt(x) (((((int) x -DBEGIN) >> 2) & ~(~0 << 5)))
*/
#define LOG_BITS_CHAR 3

#if CPTR_SIZE == 8
#define LOG_BYTES_CONTBLOCK 3
#elif CPTR_SIZE == 16
#define LOG_BYTES_CONTBLOCK 4
#else
#error Do not recognize CPTR_SIZE
#endif

void *
cb_in(void *p) {
  struct contblock **cbpp;
  int i;

  for (cbpp=&cb_pointer,i=0;*cbpp;cbpp=&((*cbpp)->cb_link),i++) {
    if ((void *)*cbpp<=p && ((void *)(*cbpp)+(*cbpp)->cb_size) >p)
      return *cbpp;
  }
  return NULL;
}

int
cb_print(void) {
  struct contblock **cbpp;
  int i;

  for (cbpp=&cb_pointer,i=0;*cbpp;cbpp=&((*cbpp)->cb_link),i++)
    emsg("%lu at %p\n",(*cbpp)->cb_size,*cbpp);
  emsg("%u blocks\n",i);
  return 0;
}

#ifdef CONTBLOCK_MARK_DEBUG
int
cb_check(void) {
  struct contblock **cbpp;
  struct pageinfo *v;
  void *cbe;

  for (cbpp=&cb_pointer;*cbpp;cbpp=&((*cbpp)->cb_link)) {
    v=get_pageinfo(*cbpp);
    cbe=((void *)(*cbpp)+(*cbpp)->cb_size-1);
    if (cbe>(void *)v+v->in_use*PAGESIZE)
      return 1;
  }
  return 0;
}

int
m_check(void) {
  struct contblock **cbpp;
  void *v,*ve,*p,*pe;
  extern object malloc_list;
  object l;

  for (l=malloc_list;l!=Cnil;l=l->c.c_cdr) {
    p=l->c.c_car->st.st_self;
    pe=p+l->c.c_car->st.st_dim;
    for (cbpp=&cb_pointer;*cbpp;cbpp=&((*cbpp)->cb_link)) {
      v=(void *)(*cbpp);
      ve=(v+(*cbpp)->cb_size-1);
      printf("%p %p  %p %p\n",p,pe,v,ve);
      if ((v<=p && p < ve)||(v<pe && pe<=ve)) 
	return 1;
    }
  }
  return 0;
}

int
off_check(void *v,void *ve,fixnum i,struct pageinfo *pi) {
  massert(i>=0);
  massert(v+i<(void *)pi+pi->in_use*PAGESIZE);
  massert(i<(ve-v));
  return 0;
}
#endif

static inline bool
pageinfo_p(void *v) {

  struct pageinfo *pi=v;

  return pi->magic==PAGE_MAGIC && pi->type<=t_contiguous &&
    (!pi->next || (void *)pi->next>=v+(pi->type==t_contiguous ? pi->in_use : 1)*PAGESIZE);

}

static inline char
get_bit(char *v,struct pageinfo *pi,void *x) {
  void *ve=CB_DATA_START(pi);
  fixnum off=(x-ve)>>LOG_BYTES_CONTBLOCK,i=off>>LOG_BITS_CHAR,s=off&~(~0UL<<LOG_BITS_CHAR);
#ifdef CONTBLOCK_MARK_DEBUG
  off_check(v,ve,i,pi);
#endif
  return (v[i]>>s)&0x1;
}

#define bit_get(v,i,s) ((v[i]>>s)&0x1)
#define bit_set(v,i,s) (v[i]|=(1UL<<s))
#define ptr_get(v,i,s) (v+(((i<<LOG_BITS_CHAR)|s)<<LOG_BYTES_CONTBLOCK))
#define ptr_set(x,v,i,s) ({fixnum _o=(x-v)>>LOG_BYTES_CONTBLOCK;i=_o>>LOG_BITS_CHAR;s=_o&~(~0UL<<LOG_BITS_CHAR);})

static inline void
set_bits(char *v,struct pageinfo *pi,void *x1,void *x2) {

  void *ds=CB_DATA_START(pi);
  fixnum i1,s1,i2,s2,se;

  ptr_set(x1,ds,i1,s1);
  ptr_set(x2,ds,i2,s2);

  if (i1==i2) {
    se=s2;
    s2=0;
  } else
    se=CHAR_SIZE;

  for (;s1<se;s1++)
    bit_set(v,i1,s1);
  if (i2>++i1) memset(v+i1,-1,(i2-i1));
  for (;--s2>=0;)
    bit_set(v,i2,s2);

}

static inline void *
get_bits(char *v,struct pageinfo *pi,void *x) {

  void *ds=CB_DATA_START(pi),*de=CB_DATA_END(pi);
  fixnum i,s,ie=mbytes(pi->in_use);
  bool z;
  char cz;

  ptr_set(x,ds,i,s);
  z=bit_get(v,i,s);
  cz=z?-1:0;

  for (;++s<CHAR_SIZE && z==bit_get(v,i,s););
  if (s==CHAR_SIZE) {
    for (;++i<ie && v[i]==cz;);
    if (i<ie) for (s=-1;++s<CHAR_SIZE && z==bit_get(v,i,s);); else s=CHAR_SIZE-1;
    /* massert(s<CHAR_SIZE); */
  }
  ds=ptr_get(ds,i,s);

  /* for (;x<ds;x+=sizeof(struct contblock)) */
  /*   massert(z==get_bit(v,pi,x)); */

  return ds<de ? ds : de;
}

static inline char
get_mark_bit(struct pageinfo *pi,void *x) {
  return get_bit(CB_MARK_START(pi),pi,x);
}

static inline void *
get_mark_bits(struct pageinfo *pi,void *x) {
  return get_bits(CB_MARK_START(pi),pi,x);
}

static inline void
set_mark_bits(struct pageinfo *pi,void *x1,void *x2) {
  set_bits(CB_MARK_START(pi),pi,x1,x2);
}

#ifdef SGC

static inline char
get_sgc_bit(struct pageinfo *pi,void *x) {
  return get_bit(CB_SGCF_START(pi),pi,x);
}

static inline void *
get_sgc_bits(struct pageinfo *pi,void *x) {
  return get_bits(CB_SGCF_START(pi),pi,x);
}

static inline void
set_sgc_bits(struct pageinfo *pi,void *x1,void *x2) {
  set_bits(CB_SGCF_START(pi),pi,x1,x2);
}

#endif

#ifdef KCLOVM
void mark_all_stacks();
bool ovm_process_created; 
#endif


static int gc_time = -1;
static int gc_start = 0;
static int gc_recursive = 0;
int runtime(void);
#ifdef SGC
int sgc_enabled=0;
#endif
long  first_protectable_page =0;

static char *copy_relblock(char *p, int s);

long real_maxpage;

struct apage {
  char apage_self[PAGESIZE];
};

/* long maxpage; */

object sSAnotify_gbcA;

#ifdef DEBUG
bool debug;
object sSAgbc_messageA;
#endif

#define	MARK_ORIGIN_MAX		300
#define	MARK_ORIGIN_BLOCK_MAX	20

object *mark_origin[MARK_ORIGIN_MAX];
int mark_origin_max;

struct {
  object *mob_addr;	/*  mark origin block address  */
  int	mob_size;	/*  mark origin block size  */
} mark_origin_block[MARK_ORIGIN_BLOCK_MAX];
int mark_origin_block_max;

enum type what_to_collect;

void
enter_mark_origin(object *p) {

  if (mark_origin_max >= MARK_ORIGIN_MAX)
    error("too many mark origins");

  mark_origin[mark_origin_max++] = p;

}

/* Whenever two arrays are linked together by displacement,
   if one is live, the other will be made live */

#define LINK_ARRAY_MARKED(x_) ((*(unsigned long *)(x_))&0x1)
#define MARK_LINK_ARRAY(x_) ((*(unsigned long *)(x_))|=1UL)
#define CLEAR_LINK_ARRAY(x_) ((*(unsigned long *)(x_))&=~(1UL))

/* #define COLLECT_RELBLOCK_P (what_to_collect == t_relocatable || what_to_collect == t_contiguous) */
bool collect_both=0;

#define COLLECT_RELBLOCK_P (what_to_collect == t_relocatable || collect_both)

static void
mark_link_array(void *v,void *ve) {

  void **p,**pe;

  if (NULL_OR_ON_C_STACK(v))
    return;

  if (sSAlink_arrayA->s.s_dbind==Cnil)
    return;

  p=(void *)sSAlink_arrayA->s.s_dbind->v.v_self;
  pe=(void *)p+sSAlink_arrayA->s.s_dbind->v.v_fillp;

  for (;p<pe;p+=2)
    if (*p>=v && *p<ve) {
      massert(!LINK_ARRAY_MARKED(p));
#ifdef SGC
      if(!sgc_enabled || WRITABLE_PAGE_P(page(p)))
#endif
	MARK_LINK_ARRAY(p);
    }

}

static void
prune_link_array(void) {

  void **p,**pe,**n,**ne;

  if (sSAlink_arrayA->s.s_dbind==Cnil)
    return;

  ne=n=p=(void *)sSAlink_arrayA->s.s_dbind->v.v_self;
  pe=(void *)p+sSAlink_arrayA->s.s_dbind->v.v_fillp;

  while (p<pe) {
    if (*p) {
      *ne++=*p++;
      *ne++=*p++;
    } else
      p+=2;
  }

  sSAlink_arrayA->s.s_dbind->v.v_fillp=(ne-n)*sizeof(*n);

}


static void
sweep_link_array(void) {

  void ***p,***pe;

  if (sSAlink_arrayA->s.s_dbind==Cnil)
    return;

  p=(void *)sSAlink_arrayA->s.s_dbind->v.v_self;
  pe=(void *)p+sSAlink_arrayA->s.s_dbind->v.v_fillp;
  for (;p<pe;p+=2)
    if (*p) {
      if (LINK_ARRAY_MARKED(p))
	CLEAR_LINK_ARRAY(p);
      else {
	**p=p[1];
	*p=0;
      }
    }

  prune_link_array();

}

DEFVAR("*LEAF-COLLECTION-THRESHOLD*",sSAleaf_collection_thresholdA,SI,make_fixnum(0),"");

#define MARK_LEAF_DATA_ALIGNED(a_,b_,c_,d_) mark_leaf_data(a_,(void **)&b_,c_,d_)
#define MARK_LEAF_DATA(a_,b_,c_) MARK_LEAF_DATA_ALIGNED(a_,b_,c_,1)

static inline bool
marking(void *p) {
  return (
#ifdef SGC
	  sgc_enabled ? ON_WRITABLE_PAGE_CACHED(p) :
#endif
	  !NULL_OR_ON_C_STACK(p));
}

static inline bool
collecting(void *p) {
  return (p<(void *)heap_end ? what_to_collect==t_contiguous : COLLECT_RELBLOCK_P);
}

static ufixnum ngc_thresh;
static union {struct dummy d;ufixnum f;} rst={.f=-1};
static void *static_promotion_limit;

static inline void
mark_leaf_data(object x,void **pp,ufixnum s,ufixnum r) {

  void *p=*pp,*dp;
  
  if (!marking(p)||!collecting(p))
    return;

  if (what_to_collect!=t_contiguous &&
      x && x->d.st>=ngc_thresh &&
      (dp=alloc_contblock_no_gc(s,static_promotion_limit))) {

    *pp=memcpy(dp,p,s);
    x->d.st=0;

    return;

  }

  if (x && x->d.st<rst.d.st) x->d.st++;

  if (p>=(void *)heap_end)
    *pp=(void *)copy_relblock(p,s);
  else
    mark_contblock(p,s);

}

static void mark_object1(object);
#define mark_object(x) if (marking(x)) mark_object1(x)

static inline void
mark_object_address(object *o,int f) {

  static ufixnum lp;
  static ufixnum lr;
  extern object *min_cfd_self;

  ufixnum p=page(o);

  if (lp!=p || !f) {
    lp=p;
    lr=
#ifdef SGC
      sgc_enabled ? WRITABLE_PAGE_P(lp) :
#endif
      (o>=min_cfd_self && o<((object *)core_end));
  }

  if (lr)
    mark_object(*o);

}

static inline void
mark_object_array(object *o,object *oe) {
  int f=0;

  if (o)
    for (;o<oe;o++,f=1)
      mark_object_address(o,f);

}


static void
mark_object1(object x) {

  fixnum i,j;

  if (is_marked_or_free(x))
    return;

  mark(x);

  switch (type_of(x)) {

  case t_cons:
    mark_object(x->c.c_car);
    mark_object(Scdr(x));/*FIXME*/
    break;

  case t_fixnum:
    break;
    
  case t_bignum:
    MARK_LEAF_DATA(x,MP_SELF(x),MP_ALLOCATED(x)*MP_LIMB_SIZE);
    break;

  case t_ratio:
    mark_object(x->rat.rat_num);
    mark_object(x->rat.rat_den);
    
  case t_shortfloat:
    break;
    
  case t_longfloat:
    break;
    
  case t_complex:
    mark_object(x->cmp.cmp_imag);
    mark_object(x->cmp.cmp_real);
    
  case t_character:
    break;
    
  case t_symbol:
    mark_object(x->s.s_plist);
    mark_object(x->s.s_gfdef);
    mark_object(x->s.s_dbind);
    mark_object(x->s.s_name);
    break;
    
  case t_package:
    mark_object(x->p.p_name);
    mark_object(x->p.p_nicknames);
    mark_object(x->p.p_shadowings);
    mark_object(x->p.p_uselist);
    mark_object(x->p.p_usedbylist);
    mark_object_array(x->p.p_internal,x->p.p_internal+x->p.p_internal_size);
    MARK_LEAF_DATA(x,x->p.p_internal,x->p.p_internal_size*sizeof(object));
    mark_object_array(x->p.p_external,x->p.p_external+x->p.p_external_size);
    MARK_LEAF_DATA(x,x->p.p_external,x->p.p_external_size*sizeof(object));
    break;
    
  case t_hashtable:
    mark_object(x->ht.ht_rhsize);
    mark_object(x->ht.ht_rhthresh);
    if (x->ht.ht_self)
      for (i=0;i<x->ht.ht_size;i++)
	if (x->ht.ht_self[i].c_cdr!=OBJNULL) {/*FIXME*/
	  mark_object_address(&x->ht.ht_self[i].c_cdr,i);
	  mark_object_address(&x->ht.ht_self[i].c_car,i+1);
	}
    i=x->ht.ht_cache-x->ht.ht_self;
    MARK_LEAF_DATA(x,x->ht.ht_self,x->ht.ht_size*sizeof(*x->ht.ht_self));
    if (x->ht.ht_cache) x->ht.ht_cache=x->ht.ht_self+i;
    break;
    
  case t_simple_array:
  case t_array:
    MARK_LEAF_DATA(x,x->a.a_dims,sizeof(*x->a.a_dims)*x->a.a_rank);

  case t_simple_vector:
  case t_simple_bitvector:
  case t_simple_string:
  case t_vector:
  case t_bitvector:
  case t_string:

    if (x->v.v_elttype==aet_object && ADISP(x)->c.c_car==Cnil)
      mark_object_array(x->v.v_self,x->v.v_self+x->v.v_dim);

    j=x->v.v_eltsize ? (1<<(x->v.v_eltsize-1)) : x->v.v_eltsize;

    if ((COLLECT_RELBLOCK_P) && (void *)x->v.v_self>=(void *)heap_end && j>sizeof(long))
      rb_pointer=PCEI(rb_pointer,j); /*FIXME GC space violation*/

    j=j ? j*x->v.v_dim : ceil(BV_OFFSET(x)+x->bv.bv_dim,BV_ALLOC)*sizeof(*x->bv.bv_self);

    if (ADISP(x)->c.c_car==Cnil)
      MARK_LEAF_DATA(x,x->v.v_self,j);

    mark_object(ADISP(x));
    set_displaced_body_ptr(x);

    break;
    
  case t_structure:
    {
      mark_object(x->str.str_def);
      if (x->str.str_self) {
	object def=x->str.str_def;
	unsigned char *s_type= &SLOT_TYPE(def,0);
	unsigned short *s_pos= &SLOT_POS(def,0);
	for (i=0,j=S_DATA(def)->length;i<j;i++)
	  if (s_type[i]==aet_object)
	    mark_object_address(&STREF(object,x,s_pos[i]),i);
	MARK_LEAF_DATA(x,x->str.str_self,S_DATA(def)->size);
      }
    }
    break;
    
  case t_stream:
    switch (x->sm.sm_mode) {
    case smm_input:
    case smm_output:
    case smm_io:
    case smm_socket:  
    case smm_probe:
      mark_object(x->sm.sm_object0);
      mark_object(x->sm.sm_object1);
      /* Only set by malloc, handled by malloc_list */
      /* if (x->sm.sm_fp) { */
      /* 	MARK_LEAF_DATA(x,x->sm.sm_buffer,BUFSIZ); */
      /* } */
      break;

    case smm_file_synonym:
    case smm_synonym:
      mark_object(x->sm.sm_object0);
      break;
      
    case smm_broadcast:
    case smm_concatenated:
      mark_object(x->sm.sm_object0);
      break;
      
    case smm_two_way:
    case smm_echo:
      mark_object(x->sm.sm_object0);
      mark_object(x->sm.sm_object1);
      break;
      
    case smm_string_input:
    case smm_string_output:
      mark_object(x->sm.sm_object0);
      break;
#ifdef USER_DEFINED_STREAMS
    case smm_user_defined:
      mark_object(x->sm.sm_object0);
      mark_object(x->sm.sm_object1);
      break;
#endif
    default:
      error("mark stream botch");
    }
    break;

  case t_random:
    MARK_LEAF_DATA_ALIGNED(x,x->rnd.rnd_state._mp_seed->_mp_d,x->rnd.rnd_state._mp_seed->_mp_alloc*MP_LIMB_SIZE,MP_LIMB_SIZE);
    break;
    
  case t_readtable:
    mark_object(x->rt.rt_case);
    if (x->rt.rt_self) {
      for (i=0;i<RTABSIZE;i++)
	mark_object_address(&x->rt.rt_self[i].rte_macro,i);
      for (i=0;i<RTABSIZE;i++) {
	mark_object_array(x->rt.rt_self[i].rte_dtab,x->rt.rt_self[i].rte_dtab+RTABSIZE);
	MARK_LEAF_DATA(x,x->rt.rt_self[i].rte_dtab,RTABSIZE*sizeof(object));
      }
    }
    MARK_LEAF_DATA(x,x->rt.rt_self,RTABSIZE*sizeof(struct rtent));
    break;
    
  case t_pathname:
    mark_object(x->pn.pn_host);
    mark_object(x->pn.pn_device);
    mark_object(x->pn.pn_directory);
    mark_object(x->pn.pn_name);
    mark_object(x->pn.pn_type);
    mark_object(x->pn.pn_version);
    mark_object(x->pn.pn_namestring);
    break;
    
  case t_function:
    mark_object(x->fun.fun_data);
    mark_object(x->fun.fun_plist);
    if (x->fun.fun_env != def_env && x->fun.fun_env != src_env) {
      mark_object(x->fun.fun_env[0]);
      x->fun.fun_env--;
      MARK_LEAF_DATA(x,x->fun.fun_env,(*(ufixnum *)x->fun.fun_env)*sizeof(*x->fun.fun_env));
      x->fun.fun_env++;
    }
    break;
    
  case t_cfdata:
    
    mark_object(x->cfd.cfd_dlist);
    mark_object(x->cfd.cfd_name);
    mark_object_array(x->cfd.cfd_self,x->cfd.cfd_self+x->cfd.cfd_fillp);
    if (what_to_collect == t_contiguous)
      mark_link_array(x->cfd.cfd_start,x->cfd.cfd_start+x->cfd.cfd_size);
    MARK_LEAF_DATA(NULL,x->cfd.cfd_start,x->cfd.cfd_size);/*Code cannot move*/
    break;

  case t_spice:
    break;

 default:
#ifdef DEBUG
    if (debug)
      printf("\ttype = %d\n", type_of(x));
#endif
    error("mark botch");

  }

}

static long *c_stack_where;

static void
mark_stack_carefully(void *topv, void *bottomv, int offset) {

  long pageoffset;
  long p;
  object x;
  struct typemanager *tm;
  register long *j;
  long *top=topv,*bottom=bottomv;
  
  /* if either of these happens we are marking the C stack
     and need to use a local */
  
  if (top==0) top = c_stack_where;
  if (bottom==0) bottom= c_stack_where;
  
  /* On machines which align local pointers on multiple of 2 rather
     than 4 we need to mark twice
  */
  
  if (offset) 
    mark_stack_carefully((((char *) top) +offset),bottom,0);
  
  for (j=top ; j >= bottom ; j--) {
    
    void *v=(void *)(*j);
    struct pageinfo *pi;
    
    if (!VALID_DATA_ADDRESS_P(v)) continue;
    
    if ((p=page(v))<first_data_page) continue;
    
    pageoffset=v-(void *)pagetochar(p);
    pi=pagetoinfo(p);
    if (!pageinfo_p(pi)) continue;
    
    if (get_pageinfo(pi)) continue;

    tm=tm_of(pi->type);
    if (tm->tm_type>=t_end) continue;

    if (pageoffset<0 || pageoffset>=tm->tm_size*tm->tm_nppage) continue;

    x=(object)(v-pageoffset%tm->tm_size);

    if (is_marked_or_free(x)) continue;

    mark_object(x);

  }

}


static void
mark_phase(void) {

  STATIC fixnum i, j;
  STATIC struct package *pp;
  STATIC bds_ptr bdp;
  STATIC frame_ptr frp;
  STATIC ihs_ptr ihsp;
  
  mark_object(Cnil->s.s_plist);
  mark_object(Cnil->s.s_name);
  mark_object(Ct->s.s_plist);
  mark_object(Ct->s.s_name);
  
  mark_stack_carefully(vs_top-1,vs_org,0);
  mark_stack_carefully(MVloc+(sizeof(MVloc)/sizeof(object)),MVloc,0);

#ifdef DEBUG
  if (debug) {
    printf("value stack marked\n");
    fflush(stdout);
  }
#endif
  
  for (bdp = bds_org;  bdp<=bds_top;  bdp++) {
    mark_object(bdp->bds_sym);
    mark_object(bdp->bds_val);
  }
  
  for (frp = frs_org;  frp <= frs_top;  frp++)
    mark_object(frp->frs_val);
  
  for (ihsp = ihs_org;  ihsp <= ihs_top;  ihsp++)
    mark_object(ihsp->ihs_function);
  
  for (i = 0;  i < mark_origin_max;  i++)
    mark_object(*mark_origin[i]);
  for (i = 0;  i < mark_origin_block_max;  i++)
    for (j = 0;  j < mark_origin_block[i].mob_size;  j++)
      mark_object(mark_origin_block[i].mob_addr[j]);
  
  for (pp = pack_pointer;  pp != NULL;  pp = pp->p_link)
    mark_object((object)pp);
  
#ifdef DEBUG
  if (debug) {
    printf("symbol navigation\n");
    fflush(stdout);
  }
#endif
  
  /* mark the c stack */
#ifndef N_RECURSION_REQD
#define N_RECURSION_REQD 2
#endif
  mark_c_stack(0,N_RECURSION_REQD,mark_stack_carefully);
  
}

#if defined(__ia64__)
	asm("        .text");
	asm("        .psr abi64");
	asm("        .psr lsb");
	asm("        .lsb");
	asm("");
	asm("        .text");
	asm("        .align 16");
	asm("        .global GC_save_regs_in_stack");
	asm("        .proc GC_save_regs_in_stack");
	asm("GC_save_regs_in_stack:");
	asm("        .body");
	asm("        flushrs");
	asm("        ;;");
	asm("        mov r8=ar.bsp");
	asm("        br.ret.sptk.few rp");
	asm("        .endp GC_save_regs_in_stack");

void * GC_save_regs_in_stack();
#endif

#if defined(__hppa__) /* Courtesy of Lamont Jones */
/* the calling sequence */
struct regs {
	void *callee_saves[16];
};
void hppa_save_regs(struct regs);

/* the code */

	asm(".code");
	asm(".export hppa_save_regs, entry");
	asm(".label	hppa_save_regs");
	asm(".proc");
	asm(".callinfo");
	asm(".entry");

	asm("stw	%r3,0(%arg0)");
	asm("stw	%r4,4(%arg0)");
	asm("stw	%r5,8(%arg0)");
	asm("stw	%r6,12(%arg0)");
	asm("stw	%r7,16(%arg0)");
	asm("stw	%r8,20(%arg0)");
	asm("stw	%r9,24(%arg0)");
	asm("stw	%r10,28(%arg0)");
	asm("stw	%r11,32(%arg0)");
	asm("stw	%r12,36(%arg0)");
	asm("stw	%r13,40(%arg0)");
	asm("stw	%r14,44(%arg0)");
	asm("stw	%r15,48(%arg0)");
	asm("stw	%r16,52(%arg0)");
	asm("stw	%r17,56(%arg0)");
	asm("bv	0(%rp)");
	asm("stw	%r18,60(%arg0)");

	asm(".exit");
	asm(".procend");
	asm(".end");
#endif

static void
mark_c_stack(jmp_buf env1, int n, void (*fn)(void *,void *,int)) {

#if defined(__hppa__)
  struct regs hppa_regs;
#endif
  jmp_buf env;
  int where;
  if (n== N_RECURSION_REQD)
    c_stack_where = (long *) (void *) &env;
  if (n > 0 ) {  
#if defined(__hppa__)
    hppa_save_regs(hppa_regs);
#else    
    setjmp(env);
#endif
    mark_c_stack(env,n - 1,fn);
  } else {
      
    /* If the locals of type object in a C function could be
       aligned other than on multiples of sizeof (char *)
       then define this.  At the moment 2 is the only other
       legitimate value besides 0 */
    
#ifndef C_GC_OFFSET
#define C_GC_OFFSET 0
#endif
    if (&where > cs_org)
      (*fn)(0,cs_org,C_GC_OFFSET);
    else
      (*fn)(cs_org,0,C_GC_OFFSET);

  }
  
#if defined(__ia64__)
  {
    extern void * __libc_ia64_register_backing_store_base;
    void * bst=GC_save_regs_in_stack();
    void * bsb=__libc_ia64_register_backing_store_base;

    if (bsb>bst)
      (*fn)(bsb,bst,C_GC_OFFSET);
    else
      (*fn)(bst,bsb,C_GC_OFFSET);

  }
#endif

}

static void
sweep_phase(void) {

  STATIC long j, k, l;
  STATIC object x;
  STATIC char *p;
  STATIC struct typemanager *tm;
  STATIC object f;
  STATIC struct pageinfo *v;
  
  for (j= t_start; j < t_contiguous ; j++) {
    tm_of(j)->tm_free=OBJNULL;
    tm_of(j)->tm_nfree=0;
  }

  for (v=cell_list_head;v;v=v->next) {

    tm = tm_of((enum type)v->type);
    
    p = pagetochar(page(v));
    f = FREELIST_TAIL(tm);
    l = k = 0;
    for (j = tm->tm_nppage; j > 0; --j, p += tm->tm_size) {
      x = (object)p;

      if (is_marked(x)) {
	unmark(x);
	l++;
	continue;
      }

      k++;

      make_free(x);
      SET_LINK(f,x);
      f = x;

    }

    SET_LINK(f,OBJNULL);
    tm->tm_tail = f;
    tm->tm_nfree += k;
    pagetoinfo(page(v))->in_use=l;
    
  }

}

static void
contblock_sweep_phase(void) {

  struct pageinfo *v;
  STATIC char *s, *e, *p, *q;
  ufixnum i;

  reset_contblock_freelist();

  for (i=0;i<contblock_array->v.v_fillp && (v=(void *)contblock_array->v.v_self[i]);i++) {

    bool z;

#ifdef SGC
    if (sgc_enabled && !(v->sgc_flags&SGC_PAGE_FLAG)) continue;
#endif

    s=CB_DATA_START(v);
    e=(void *)v+v->in_use*PAGESIZE;

    z=get_mark_bit(v,s);
    for (p=s;p<e;) {
      q=get_mark_bits(v,p);
      if (!z)
	insert_contblock(p,q-p);
      z=1-z;
      p=q;
    }

    bzero(CB_MARK_START(v),CB_SGCF_START(v)-CB_MARK_START(v));

  }

  sweep_link_array();

}


int (*GBC_enter_hook)() = NULL;
int (*GBC_exit_hook)() = NULL;

fixnum fault_pages=0;

static ufixnum
count_contblocks(void) {

  ufixnum ncb;
  struct contblock *cbp;

  for (ncb=0,cbp=cb_pointer;cbp;cbp=cbp->cb_link,ncb++);

  return ncb;
  
}


void
GBC(enum type t) {

#ifdef DEBUG
  int tm=0;
#endif
  
  BEGIN_NO_INTERRUPT;

  if (t==t_other) {
    collect_both=1;
    t=t_contiguous;
  }

  ngc_thresh=fix(sSAleaf_collection_thresholdA->s.s_dbind);
  cumulative_allocation+=recent_allocation;
  recent_allocation=0;

  if (in_signal_handler && t == t_relocatable)
    error("cant gc relocatable in signal handler");
  
  if (GBC_enter_hook != NULL)
    (*GBC_enter_hook)();
  
  if (!GBC_enable)
      error("GBC is not enabled");
  interrupt_enable = FALSE;
  
  if (saving_system) {

    struct pageinfo *v;
    void *x;
    struct typemanager *tm=tm_of(t_stream);
    unsigned j;

    for (v=cell_list_head;v;v=v->next) 
      if (tm->tm_type==v->type)
	for (x=pagetochar(page(v)),j=tm->tm_nppage;j--;x+=tm->tm_size) {
	  object o=x;
	  if (type_of(o)==t_stream && !is_free(o) && o->sm.sm_fp && o->sm.sm_fp!=stdin && o->sm.sm_fp!=stdout && o->sm.sm_fp!=stderr)
	    close_stream(o);
	}

    gc_time = -1;
    }


#ifdef DEBUG
  debug = symbol_value(sSAgbc_messageA) != Cnil;
#endif
  
  what_to_collect = t;
  
  {
    struct typemanager *tm=tm_of(t);
    tm->tm_gbccount++;
    tm->tm_adjgbccnt++;
  }
  
  if (sSAnotify_gbcA->s.s_dbind != Cnil
#ifdef DEBUG
      || debug
#endif
      ) {

    if (gc_time < 0)
      gc_time=0;

#ifdef SGC
    omsg("[%s for %ld %s pages..",
	 (sgc_enabled ? "SGC" : "GC"),
	 (sgc_enabled ? sgc_count_type(t) : tm_of(t)->tm_npage),
	 (tm_table[(int)t].tm_name)+1);
#else
    omsg("[%s for %ld %s pages..",
	 ("GC"),
	 (tm_of(t)->tm_npage),
	 (tm_table[(int)t].tm_name)+1);
#endif

#ifdef SGC
    if(sgc_enabled)
      printf("(%ld faulted pages, %ld writable, %ld read only)..",
	     fault_pages,(page(core_end)-first_data_page)-(page(rb_start)-page(heap_end))-sgc_count_read_only(),
	     sgc_count_read_only());
#endif	  

    fflush(stdout);

  }

  if (gc_time >=0 && !gc_recursive++) {gc_start=runtime();}
  
  if (COLLECT_RELBLOCK_P) {
    static_promotion_limit=rb_start<new_rb_start ? rb_start : new_rb_start;/*do not allow static promotion to go past this point*/
    setup_rb(0);
  }
  
#ifdef DEBUG
  if (debug) {
    printf("mark phase\n");
    fflush(stdout);
    tm = runtime();
  }
#endif
#ifdef SGC
  if(sgc_enabled)
    sgc_mark_phase();
  else
#endif	
    mark_phase();
#ifdef DEBUG
  if (debug) {
    printf("mark ended (%d)\n", runtime() - tm);
    fflush(stdout);
  }
#endif
  
#ifdef DEBUG
  if (debug) {
    printf("sweep phase\n");
    fflush(stdout);
    tm = runtime();
  }
#endif
#ifdef SGC
  if(sgc_enabled)
    sgc_sweep_phase();
  else
#endif	
    sweep_phase();
#ifdef DEBUG
  if (debug) {
    printf("sweep ended (%d)\n", runtime() - tm);
    fflush(stdout);
  }
#endif
  
  if (COLLECT_RELBLOCK_P) {

#ifdef SGC
    if (sgc_enabled)
      wrimap=(void *)sSAwritableA->s.s_dbind->v.v_self;
#endif

  }

  if (t == t_contiguous) {
#ifdef DEBUG
    if (debug) {
      printf("contblock sweep phase\n");
      fflush(stdout);
      tm = runtime();
    }
#endif
    
    contblock_sweep_phase();
#ifdef DEBUG
    if (debug)
      printf("contblock sweep ended (%d)\n",
	     runtime() - tm);
#endif
  }
  
#ifdef DEBUG
  if (debug) {
    int i,j;
    for (i = 0, j = 0;  i < (int)t_end;  i++) {
      if (tm_table[i].tm_type == (enum type)i) {
	printf("%13s: %8ld used %8ld free %4ld/%ld pages\n",
	       tm_table[i].tm_name,
	       TM_NUSED(tm_table[i]),
	       tm_table[i].tm_nfree,
	       tm_table[i].tm_npage,
	       tm_table[i].tm_maxpage);
	j += tm_table[i].tm_npage;
      } else
	printf("%13s: linked to %s\n",
	       tm_table[i].tm_name,
	       tm_table[(int)tm_table[i].tm_type].tm_name);
    }
    printf("contblock: %ld blocks %ld pages\n", count_contblocks(), ncbpage);
    printf("hole: %lu pages\n", (ufixnum)page(rb_start-heap_end));
    printf("relblock: %ld bytes used %ld bytes free %ld pages\n",
	   (long)(rb_pointer - rb_start), (long)(rb_end - rb_pointer), nrbpage);
    printf("GBC ended\n");
    fflush(stdout);
  }
#endif
  
  interrupt_enable = TRUE;
  
  if (GBC_exit_hook != NULL)
    (*GBC_exit_hook)();
  
  if(gc_time>=0 && !--gc_recursive) {gc_time=gc_time+(gc_start=(runtime()-gc_start));}
  
  if (sSAnotify_gbcA->s.s_dbind != Cnil) {
    
    if (gc_recursive)
      omsg("(T=...).GC finished]\n");
    else
      omsg("(T=%d).GC finished]\n",gc_start);

  }
  
  collect_both=0;

  END_NO_INTERRUPT;

  CHECK_INTERRUPT;

}

static void
FFN(siLheap_report)(void) {

  int i;
  extern void *shared_lib_start;
  
  check_arg(0);
  
  vs_check_push(make_fixnum(sizeof(fixnum)*CHAR_SIZE));
  vs_push(make_fixnum(PAGESIZE));
  vs_push(make_fixnum((ufixnum)data_start));
  vs_push(make_fixnum(((unsigned long)real_maxpage)<<PAGEWIDTH));
  vs_push(make_fixnum((ufixnum)shared_lib_start));
  i=getpagesize();
  vs_push(make_fixnum(((unsigned long)cs_base+i-1)&-i));
  vs_push(make_fixnum(labs(cs_base-cs_org)));
  vs_push(make_fixnum((CSTACK_DIRECTION+1)>>1));
  vs_push(make_fixnum(CSTACK_ALIGNMENT));
  vs_push(make_fixnum(labs(cs_limit-cs_org)));/*CSSIZE*/
  vs_push(make_fixnum(phys_pages));
#if defined(IM_FIX_BASE) && defined(IM_FIX_LIM)
#ifdef LOW_IM_FIX
  vs_push(make_fixnum(-LOW_IM_FIX));
  vs_push(make_fixnum(1UL<<LOW_SHFT));
#else
  vs_push(make_fixnum(IM_FIX_BASE));
  vs_push(make_fixnum(IM_FIX_LIM));
#endif
#else  
  vs_push(make_fixnum(0));
  vs_push(make_fixnum(0));
#endif

}  
DEFUN("CUMULATIVE-ALLOCATION",object,fScumulative_allocation,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  ufixnum x=(cumulative_allocation+recent_allocation)>>20;
  if ((fixnum)x>=0)
    RETURN1(make_fixnum(x));
  mpz_set_ui(MP(big_fixnum1),x);
  RETURN1(replace_big(big_fixnum1));
}

DEFUN("ROOM-REPORT",object,fSroom_report,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {

  object x=Cnil;
  int i;
  
  x=make_cons(make_fixnum(real_maxpage-first_data_page),x);
  x=make_cons(make_fixnum(available_pages),x);
  x=make_cons(make_fixnum(ncbpage),x);
  x=make_cons(make_fixnum(maxcbpage),x);
  x=make_cons(make_fixnum(count_contblocks()),x);
  x=make_cons(make_fixnum(cbgbccount),x);
  x=make_cons(make_fixnum((rb_start-heap_end)>>PAGEWIDTH),x);
  x=make_cons(make_fixnum(rb_pointer - rb_begin()),x);
  x=make_cons(make_fixnum((rb_begin()+rb_size()) - rb_pointer),x);
  x=make_cons(make_fixnum(nrbpage),x);
  x=make_cons(make_fixnum(maxrbpage),x);
  x=make_cons(make_fixnum(rbgbccount),x);
  for (i = 0;  i < (int)t_end;  i++) {
    x=make_cons(make_simple_string(tm_table[i].tm_name+1),x);
    if (tm_table[i].tm_type == (enum type)i) {
      x=make_cons(make_fixnum(TM_NUSED(tm_table[i])),x);
      x=make_cons(make_fixnum(tm_table[i].tm_nfree+tm_table[i].tm_alt_nfree),x);
      x=make_cons(make_fixnum(tm_table[i].tm_npage),x);
      x=make_cons(make_fixnum(tm_table[i].tm_maxpage),x);
      x=make_cons(make_fixnum(tm_table[i].tm_gbccount),x);
      x=make_cons(make_fixnum(tm_table[i].tm_size/sizeof(fixnum)),x);
    } else {
      x=make_cons(Cnil,x);
      x=make_cons(make_simple_string(tm_of(i)->tm_name+1),x);
      x=make_cons(Cnil,x);
      x=make_cons(Cnil,x);
      x=make_cons(Cnil,x);
      x=make_cons(Cnil,x);
    }
  }
  RETURN1(nreverse(x));

}

static void
FFN(siLreset_gbc_count)(void) {

  int i;
  
  check_arg(0);
  
  for (i = 0;  i < t_other;  i++)
    tm_table[i].tm_gbccount = tm_table[i].tm_adjgbccnt = tm_table[i].tm_opt_maxpage = 0;
}

/* copy S bytes starting at P to beyond rb_pointer1 (temporarily)
   but return a pointer to where this will be copied back to,
   when gc is done.  alignment of rb_pointer is kept at a multiple
   of sizeof(char *);
*/

static char *
copy_relblock(char *p, int s) {
 char *q = rb_pointer;

 s = CEI(s,PTR_ALIGN);
 rb_pointer += s;
 memmove(q,p,s);/*FIXME memcpy*/

 return q;

}


static void
mark_contblock(void *p, int s) {

  STATIC char *q;
  STATIC char *x, *y;
  struct pageinfo *v;

  if (NULL_OR_ON_C_STACK(p))
    return;

  q = p + s;
  /* SGC cont pages: contblock pages must be no smaller than
     sizeof(struct contblock).  CM 20030827 */
  x = (char *)PFLR(p,CPTR_SIZE);
  y = (char *)PCEI(q,CPTR_SIZE);
  massert(v=get_pageinfo(x));
#ifdef SGC
  if (!sgc_enabled || (v->sgc_flags&SGC_PAGE_FLAG))
#endif
     set_mark_bits(v,x,y);
 }

DEFUN("CONTIGUOUS-REPORT",object,fScontiguous_report,SI,1,1,NONE,OO,OO,OO,OO,(void),"") {

  struct contblock **cbpp;
  struct pageinfo *v;
  ufixnum i,j,k,s;
  struct typemanager *tm=tm_of(t_cfdata);
  void *p;

  for (i=j=0,cbpp=&cb_pointer;(*cbpp);) {
    for (k=0,s=(*cbpp)->cb_size,p=*cbpp;*cbpp && (*cbpp)->cb_size==s;i+=(*cbpp)->cb_size,j++,k++,cbpp=&(*cbpp)->cb_link);
    emsg("%lu %lu starting at %p\n",k,s,p);
  }
  emsg("\nTotal free %lu in %lu pieces\n\n",i,j);

  for (i=j=k=0;k<contblock_array->v.v_fillp && (v=(void *)contblock_array->v.v_self[k]);k++,i+=v->in_use,j++)
    emsg("%lu pages at %p\n",(unsigned long)v->in_use,v);
  emsg("\nTotal pages %lu in %lu pieces\n\n",i,j);

  for (i=j=0,v=cell_list_head;v;v=v->next)
    if (tm->tm_type==v->type) {
      void *p;
      ufixnum k;
      for (p=pagetochar(page(v)),k=0;k<tm->tm_nppage;k++,p+=tm->tm_size) {
 	object o=p;
 	if (!is_free(o) && type_of(o)==t_cfdata && (void *)o->cfd.cfd_start>=data_start) {
 	  emsg("%lu code bytes at %p\n",(unsigned long)o->cfd.cfd_size,o->cfd.cfd_start);
 	  i+=o->cfd.cfd_size;
 	  j++;
 	}
      }
    }
  emsg("\nTotal code bytes %lu in %lu pieces\n",i,j);

  for (i=j=0,v=cell_list_head;v;v=v->next) {
    struct typemanager *tm=tm_of(v->type);
    void *p;
    ufixnum k;
    for (p=pagetochar(page(v)),k=0;k<tm->tm_nppage;k++,p+=tm->tm_size) {
      object o=p;
      void *d=NULL;
      ufixnum s=0;
      if (!is_free(o)) {
 	switch (type_of(o)) {
 	case t_array:
 	case t_vector:
 	case t_simple_vector:
 	  d=o->a.a_self;
 	  s=o->a.a_dim*sizeof(object);
 	  break;
 	case t_hashtable:
 	  d=o->ht.ht_self;
 	  s=o->ht.ht_size*sizeof(object)*2;
 	  break;
 	case t_simple_string:
 	case t_string:
 	case t_bitvector:
 	case t_simple_bitvector:
 	  d=o->a.a_self;
 	  s=o->a.a_dim;
 	  break;
 	case t_package:
 	  d=o->p.p_external;
 	  s=(o->p.p_external_size+o->p.p_internal_size)*sizeof(object);
 	  break;
 	case t_bignum:
 	  d=o->big.big_mpz_t._mp_d;
 	  s=o->big.big_mpz_t._mp_alloc*MP_LIMB_SIZE;
 	  break;
 	case t_structure:
 	  d=o->str.str_self;
 	  s=S_DATA(o->str.str_def)->length*sizeof(object);
 	  break;
 	case t_random:
 	  d=o->rnd.rnd_state._mp_seed->_mp_d;
 	  s=o->rnd.rnd_state._mp_seed->_mp_alloc*MP_LIMB_SIZE;
 	  break;
 	case t_function:
 	  d=o->fun.fun_env;
 	  s=o->fun.fun_env!=def_env && o->fun.fun_env!=src_env ? ((ufixnum *)o->fun.fun_env)[-1]*sizeof(object) : 0;
 	  break;
 	case t_cfdata:/*FIXME*/
 	  d=o->cfd.cfd_start;
 	  s=o->cfd.cfd_size;
 	  break;
 	case t_readtable:
 	  d=o->rt.rt_self;
 	  s=RTABSIZE*(sizeof(struct rtent));/*FIXME*/
 	  break;
 	default:
 	  break;
 	}
 	if (d>=data_start && d<(void *)heap_end && s) {
 	  emsg("%lu %s bytes at %p, object %p\n",s,tm_table[type_of(o)].tm_name,d,o);
 	  i+=s;
 	  j++;
 	}
      }
    }
  }
  emsg("\nTotal leaf bytes %lu in %lu pieces\n",i,j);

  return Cnil;

}

DEFUN("GBC",object,fSgbc,SI,1,1,NONE,OO,OO,OO,OO,(object x0),"") {

  if (x0 == Ct) {
    tm_table[t_contiguous].tm_adjgbccnt--;
    GBC(t_other);
  } else if (x0 == Cnil) {
    tm_table[t_cons].tm_adjgbccnt--;
    GBC(t_cons);
  } else if (eql(small_fixnum(0),x0)) {
    tm_table[t_contiguous].tm_adjgbccnt--;
    GBC(t_contiguous);
  } else {
    x0 = small_fixnum(1);
    tm_table[t_relocatable].tm_adjgbccnt--;
    GBC(t_relocatable);
  }
  RETURN1(x0);
}

static void
FFN(siLgbc_time)(void) {
  if (vs_top>vs_base)
    gc_time=fix(vs_base[0]);
  else {
    vs_base[0]=make_fixnum(gc_time);
    vs_top=vs_base+1;
  }
}

#ifdef SGC
#include "sgbc.c"
#endif

DEFVAR("*NOTIFY-GBC*",sSAnotify_gbcA,SI,Cnil,"");
#ifdef DEBUG
DEFVAR("*GBC-MESSAGE*",sSAgbc_messageA,SI,Cnil,"");
#endif

void
gcl_init_GBC(void) {

  make_si_function("HEAP-REPORT", siLheap_report);
  make_si_function("RESET-GBC-COUNT", siLreset_gbc_count);
  make_si_function("GBC-TIME",siLgbc_time);
#ifdef SGC
  make_si_function("SGC-ON",siLsgc_on);
#endif

}
