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
	alloc.c
	IMPLEMENTATION-DEPENDENT
*/

#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include "include.h"
#include "page.h"

#ifdef HAVE_MPROTECT
#include <sys/mman.h>
#endif

static int
t_from_type(object);

#include "pool.h"


DEFVAR("*AFTER-GBC-HOOK*",sSAafter_gbc_hookA,SI,sLnil,"");
DEFVAR("*IGNORE-MAXIMUM-PAGES*",sSAignore_maximum_pagesA,SI,sLt,"");
#define IGNORE_MAX_PAGES (sSAignore_maximum_pagesA ==0 || sSAignore_maximum_pagesA->s.s_dbind !=sLnil) 

static void call_after_gbc_hook(int t);

#ifdef DEBUG_SBRK
int debug;
char *
sbrk1(n)
     int n;
{char *ans;
 if (debug){
   printf("\n{sbrk(%d)",n);
   fflush(stdout);}
 ans= (char *)sbrk(n);
 if (debug){
   printf("->[0x%x]", ans);
   fflush(stdout);
   printf("core_end=0x%x,sbrk(0)=0x%x}",core_end,sbrk(0));
   fflush(stdout);}
 return ans;
}
#define sbrk sbrk1
#endif /* DEBUG_SBRK */

long starting_hole_div=10;
long starting_relb_heap_mult=2;
long resv_pages=0;

void *stack_alloc_start=NULL,*stack_alloc_end=NULL;

#ifdef BSD
#include <sys/time.h>
#include <sys/resource.h>
#ifdef RLIMIT_STACK
struct rlimit data_rlimit;
#endif
#endif

static inline void *
bsearchleq(void *i,void *v1,size_t n,size_t s,int (*c)(const void *,const void *)) {

  ufixnum nn=n>>1;
  void *v=v1+nn*s;
  int j=c(i,v);

  if (nn)
    return !j ? v : (j>0 ? bsearchleq(i,v,n-nn,s,c) : bsearchleq(i,v1,nn,s,c));
  else
    return j<=0 ? v : v+s;

}
		     

object contblock_array=Cnil;

static inline void
expand_contblock_array(void) {

  if (contblock_array==Cnil) {
    contblock_array=fSmake_vector(make_fixnum(aet_fix),16,Ct,make_fixnum(0),Cnil,0,Cnil,make_fixnum(0));
    contblock_array->v.v_self[0]=(object)&cb_pointer;
    enter_mark_origin(&contblock_array);
  }

  if (contblock_array->v.v_fillp==contblock_array->v.v_dim) {

    void *v=alloc_relblock(2*contblock_array->v.v_dim*sizeof(fixnum));

    memcpy(v,contblock_array->v.v_self,contblock_array->v.v_dim*sizeof(fixnum));
    contblock_array->v.v_self=v;
    contblock_array->v.v_dim*=2;

  }

}

static void
contblock_array_push(void *p) {

  ufixnum f=contblock_array==Cnil ? 0 : contblock_array->v.v_fillp;/*FIXME*/

  expand_contblock_array();
  memmove(contblock_array->v.v_self+f+1,contblock_array->v.v_self+f,
	  (contblock_array->v.v_fillp-f)*sizeof(*contblock_array->v.v_self));
  contblock_array->v.v_self[f]=p;
  contblock_array->v.v_fillp++;

}
  
static inline int
acomp(const void *v1,const void *v2) {

  void *p1=*(void * const *)v1,*p2=*(void * const *)v2;

  return p1<p2 ? -1 : (p1==p2 ? 0 : 1);

}

struct pageinfo *
get_pageinfo(void *x) {

  struct pageinfo **pp=bsearchleq(&x,contblock_array->v.v_self,contblock_array->v.v_fillp,sizeof(*contblock_array->v.v_self),acomp);
  struct pageinfo *p=(void *)pp>(void *)contblock_array->v.v_self ? pp[-1] : NULL;
  
  return p && (void *)p+p->in_use*PAGESIZE>x ? p : NULL;

}

static inline void
add_page_to_contblock_list(void *p,fixnum m) {
 
  struct pageinfo *pp=pageinfo(p);

  bzero(pp,sizeof(*pp));
  pp->type=t_contiguous;
  pp->in_use=m;
  massert(pp->in_use==m);
  pp->magic=PAGE_MAGIC;
  
  contblock_array_push(p);

  bzero(pagetochar(page(pp)),CB_DATA_START(pp)-(void *)pagetochar(page(pp)));
#ifdef SGC
  if (sgc_enabled && tm_table[t_contiguous].tm_sgc) {
    memset(CB_SGCF_START(pp),-1,CB_DATA_START(pp)-CB_SGCF_START(pp));
    pp->sgc_flags=SGC_PAGE_FLAG;
    }
#endif
  
  ncbpage+=m;
  insert_contblock(CB_DATA_START(pp),CB_DATA_END(pp)-CB_DATA_START(pp));

}

int
icomp(const void *v1,const void *v2) {
  const fixnum *f1=v1,*f2=v2;
  return *f1<*f2 ? -1 : *f1==*f2 ? 0 : +1;
}


void
add_page_to_freelist(char *p, struct typemanager *tm) {

  short t,size;
  long fw;
  object x,xe,f;
  struct pageinfo *pp;

  t=tm->tm_type;

  size=tm->tm_size;
  pp=pageinfo(p);
  bzero(pp,sizeof(*pp));
  pp->type=t;
  pp->magic=PAGE_MAGIC;

  if (cell_list_head==NULL)
    cell_list_tail=cell_list_head=pp;
  else if (pp > cell_list_tail) {
    cell_list_tail->next=pp;
    cell_list_tail=pp;
  }

  x= (object)pagetochar(page(p));
  /* set_type_of(x,t); */
  make_free(x);

#ifdef SGC

  if (sgc_enabled && tm->tm_sgc)
    pp->sgc_flags=SGC_PAGE_FLAG;

#ifndef SGC_WHOLE_PAGE
  if (TYPEWORD_TYPE_P(pp->type))
    x->d.s=(sgc_enabled && tm->tm_sgc) ? SGC_RECENT : SGC_NORMAL;
#endif

  /* array headers must be always writable, since a write to the
     body does not touch the header.   It may be desirable if there
     are many arrays in a system to make the headers not writable,
     but just SGC_TOUCH the header each time you write to it.   this
     is what is done with t_structure */
  /* if (t==(tm_of(t_array)->tm_type)) */
  /*   pp->sgc_flags|=SGC_PERM_WRITABLE; */
  /* The SGC_PERM_WRITABLE facility is no longer used in favor of
     SGC_TOUCH.  Implicitly grouping object types by size is
     unreliable.*/

#endif 

  f=FREELIST_TAIL(tm);
  fw=x->fw;
  xe=(object)((void *)x+tm->tm_nppage*size);
  for (;x<xe;f=x,x=(object)((void *)x+size)) {
    x->fw=fw;
    SET_LINK(f,x);
  }

  SET_LINK(f,OBJNULL);
  tm->tm_tail=f;
  tm->tm_nfree+=tm->tm_nppage;
  tm->tm_npage++;

}

static inline void
maybe_reallocate_page(struct typemanager *ntm,ufixnum count) {

  void **y,**n;
  fixnum *pp,*pp1,*ppe,yp;
  struct typemanager *tm;
  fixnum i,j,e[t_end];
  struct pageinfo *v;

  massert(pp1=pp=alloca(count*sizeof(*pp1)));
  ppe=pp1+count;

  for (v=cell_list_head;v && pp<ppe;v=v->next) {

    if (v->type>=t_end ||
	(tm=tm_of(v->type))==ntm ||
#ifdef SGC
	(sgc_enabled && tm->tm_sgc && v->sgc_flags!=SGC_PAGE_FLAG) ||
#endif
	v->in_use)
      continue;

    count--;
    *pp++=page(v);

  }

#define NEXT_LINK(a_) (void *)&((struct freelist *)*(a_))->f_link
#define FREE_PAGE_P(yp_) bsearch(&(yp_),pp1,ppe-pp1,sizeof(*pp1),icomp)

  ppe=pp;
  bzero(e,sizeof(e));
  for (pp=pp1;pp<ppe;pp++)
    e[pagetoinfo(*pp)->type]++;
  for (i=0;i<sizeof(e)/sizeof(*e);i++) {
    if (!e[i]) continue;
    tm=tm_of(i);
    tm->tm_nfree-=(j=tm->tm_nppage*e[i]);
    tm->tm_npage-=e[i];
    set_tm_maxpage(tm,tm->tm_maxpage-e[i]);
    set_tm_maxpage(ntm,ntm->tm_maxpage+e[i]);
    for (y=(void *)&tm->tm_free;*y!=OBJNULL && j;) {
      for (;*y!=OBJNULL && (yp=page(*y)) && !FREE_PAGE_P(yp);y=NEXT_LINK(y));
      if (*y!=OBJNULL) {
	for (n=NEXT_LINK(y),j--;*n!=OBJNULL && (yp=page(*n)) && FREE_PAGE_P(yp);n=NEXT_LINK(n),j--);
	*y=*n;
      }
    }
    massert(!j);
  }

  for (pp=pp1;pp<ppe;pp++) {
    struct pageinfo *pn=pagetoinfo(*pp)->next;
    add_page_to_freelist(pagetochar(*pp),ntm);
    pagetoinfo(*pp)->next=pn;
  }
      
}


int reserve_pages_for_signal_handler=30;

/* If  (n >= 0 ) return pointer to n pages starting at heap end,
   These must come from the hole, so if that is exhausted you have
   to gc and move the hole.
   if  (n < 0) return pointer to n pages starting at heap end,
   but don't worry about the hole.   Basically just make sure
   the space is available from the Operating system.
   If not in_signal_handler then try to keep a minimum of
   reserve_pages_for_signal_handler pages on hand in the hole
 */

void
setup_rb(bool preserve_rb_pointerp) {

  int lowp=rb_high();

  update_pool(2*(nrbpage-page(rb_size())));
  rb_start=new_rb_start;
  rb_end=rb_start+(nrbpage<<PAGEWIDTH);
  if (!preserve_rb_pointerp)
    rb_pointer=lowp ? rb_start : rb_end;
  rb_limit=rb_begin()+(nrbpage<<PAGEWIDTH);
  pool_check();
  
  alloc_page(-(2*nrbpage+((new_rb_start-heap_end)>>PAGEWIDTH)));
 
}
  
void
resize_hole(ufixnum hp,enum type tp,bool in_placep) {
  
  char *start=rb_begin(),*new_start=heap_end+hp*PAGESIZE;
  ufixnum size=rb_pointer-start;

#define OVERLAP(c_,t_,s_) ((t_)<(c_)+(s_) && (c_)<(t_)+(s_))
  if (!in_placep && (rb_high() ?
		     OVERLAP(start,new_start,size) :
		     OVERLAP(start,new_start+(nrbpage<<PAGEWIDTH),size)
		     /* 0 (20190401  never reached)*/
		     )) {
    if (sSAnotify_gbcA->s.s_dbind != Cnil)
      emsg("[GC Toggling relblock when resizing hole to %lu]\n",hp);
    tm_table[t_relocatable].tm_adjgbccnt--;
    GBC(t_relocatable);
    return resize_hole(hp,tp,in_placep);
  }

  new_rb_start=new_start;

  if (!size || in_placep)
    setup_rb(in_placep);
  else {
    tm_of(tp)->tm_adjgbccnt--;
    GBC(tp);
  }
  
}

void *
alloc_page(long n) {

  bool s=n<0;
  ufixnum nn=s ? -n : n;
  void *v,*e;
  
  if (!s) {

    if (nn>((rb_start-heap_end)>>PAGEWIDTH)) {


      fixnum d=available_pages-nn;

      d*=0.2;
      d=d<0.01*real_maxpage ? available_pages-nn : d;
      d=d<0 ? 0 : d;
      d=(available_pages/3)<d ? (available_pages/3) : d;
      
      if (sSAnotify_gbcA && sSAnotify_gbcA->s.s_dbind != Cnil)
	emsg("[GC Hole overrun]\n");

      resize_hole(d+nn,t_relocatable,0);

    }
  }

  e=heap_end;
  v=e+nn*PAGESIZE;

  if (!s) {

    heap_end=v;
    update_pool(nn);
    pool_check();
    
  } else if (v>(void *)core_end) {
    
    massert(!mbrk(v));
    core_end=v;
    
  }
  
  return(e);

}


#define MAX(a_,b_) ({fixnum _a=(a_),_b=(b_);_a<_b ? _b : _a;})
#define MIN(a_,b_) ({fixnum _a=(a_),_b=(b_);_a<_b ? _a : _b;})

struct pageinfo *cell_list_head=NULL,*cell_list_tail=NULL;;

ufixnum
sum_maxpages(void) {

  ufixnum i,j;

  for (i=t_start,j=0;i<t_other;i++)
    j+=tm_table[i].tm_maxpage;

  return j+tm_table[t_relocatable].tm_maxpage;

}

fixnum
check_avail_pages(void) {
  
  return real_maxpage-page(data_start ? data_start : sbrk(0))-available_pages-resv_pages-sum_maxpages();

}

#include <fenv.h>

fixnum
set_tm_maxpage(struct typemanager *tm,fixnum n) {
  
  fixnum r=tm->tm_type==t_relocatable,j=tm->tm_maxpage,z=(n-j)*(r ? 2 : 1);
  if (z>available_pages) return 0;
  available_pages-=z;
  ({fenv_t f;feholdexcept(&f);tm->tm_adjgbccnt*=((double)j+1)/(n+1);fesetenv(&f);});
  tm->tm_maxpage=n;
  /* massert(!check_avail_pages()); */
  return 1;
}
  
object
type_name(int t) {
  return make_simple_string(tm_table[(int)t].tm_name+1);
}


static void
call_after_gbc_hook(int t) {
  if (sSAafter_gbc_hookA && sSAafter_gbc_hookA->s.s_dbind!= Cnil) {
    ifuncall1(sSAafter_gbc_hookA->s.s_dbind,intern(str((tm_table[(int)t].tm_name+1)),system_package));
  }
}

static fixnum
grow_linear(fixnum old, fixnum fract, fixnum grow_min, fixnum grow_max,fixnum max_delt) {
  
  fixnum delt;

  delt=(old*(fract ? fract : 50))/100;

  delt= (grow_min && delt < grow_min ? grow_min:
	 grow_max && delt > grow_max ? grow_max:
	 delt);

  delt=delt>max_delt ? max_delt : delt;

  return old + delt;

}

/* GCL's traditional garbage collecting algorithm placed heavy emphasis
   on conserving memory.  Maximum page allocations of each object type
   were only increased when the objects in use after GBC exceeded a
   certain percentage threshold of the current maximum.  This allowed
   a situation in which a growing heap would experience significant
   performance degradation due to GBC runs triggered by types making
   only temporary allocations -- the rate of GBC calls would be
   constant while the cost for each GBC would grow with the size of
   the heap.

   We implement here a strategy designed to approximately optimize the
   product of the total GBC call rate times the cost or time taken for
   each GBC.  The rate is approximated from the actual gbccounts so
   far experienced, while the cost is taken to be simply proportional
   to the heap size at present.  This can be further tuned by taking
   into account the number of pointers in each object type in the
   future, but at present objects of several different types but
   having the same size are grouped together in the type manager
   table, so this step becomes more involved.

   After each GBC, we calculate the maximum of the function
   (gbc_rate_other_types + gbc_rate_this_type *
   current_maxpage/new_maxpage)*(sum_all_maxpages-current_maxpage+new_maxpage).
   If the benefit in the product from adopting the new_maxpage is
   greater than 5%, we adopt it, and adjust the gbccount for the new
   basis.  Corrections are put in place for small GBC counts, and the
   possibility that GBC calls of only a single type are ever
   triggered, in which case the optimum new_maxpage would diverge in
   the simple analysis above.

   20040403 CM */

DEFVAR("*OPTIMIZE-MAXIMUM-PAGES*",sSAoptimize_maximum_pagesA,SI,sLnil,"");
#define OPTIMIZE_MAX_PAGES (sSAoptimize_maximum_pagesA ==0 || sSAoptimize_maximum_pagesA->s.s_dbind !=sLnil) 
DEFVAR("*NOTIFY-OPTIMIZE-MAXIMUM-PAGES*",sSAnotify_optimize_maximum_pagesA,SI,sLnil,"");

static object
exhausted_report(enum type t,struct typemanager *tm) {

  available_pages+=resv_pages;
  resv_pages=0;
  CEerror("Continues execution.",
	  "The storage for ~A is exhausted. ~D pages allocated. Use ALLOCATE to expand the space.",
	  2, type_name(t), make_fixnum(tm->tm_npage));

  call_after_gbc_hook(t);

  return alloc_object(t);

}

#ifdef SGC
#define TOTAL_THIS_TYPE(tm) (tm->tm_nppage * (sgc_enabled ? sgc_count_type(tm->tm_type) : tm->tm_npage))
#else
#define TOTAL_THIS_TYPE(tm) (tm->tm_nppage * tm->tm_npage)
#endif

static object cbv=Cnil;
#define cbsrch1 ((struct contblock ***)cbv->v.v_self)
#define cbsrche (cbsrch1+cbv->v.v_fillp)

static inline void
expand_contblock_index_space(void) {

  if (cbv==Cnil) {
    cbv=fSmake_vector(make_fixnum(aet_fix),16,Ct,make_fixnum(0),Cnil,0,Cnil,make_fixnum(0));
    cbv->v.v_self[0]=(object)&cb_pointer;
    enter_mark_origin(&cbv);
  }

  if (cbv->v.v_fillp+1==cbv->v.v_dim) {

    void *v;
    object o=sSAleaf_collection_thresholdA->s.s_dbind;

    sSAleaf_collection_thresholdA->s.s_dbind=make_fixnum(-1);
    v=alloc_relblock(2*cbv->v.v_dim*sizeof(fixnum));
    sSAleaf_collection_thresholdA->s.s_dbind=o;

    memcpy(v,cbv->v.v_self,cbv->v.v_dim*sizeof(fixnum));
    cbv->v.v_self=v;
    cbv->v.v_dim*=2;

  }

}

static inline void *
expand_contblock_index(struct contblock ***cbppp) {

  ufixnum i=cbppp-cbsrch1;

  expand_contblock_index_space();

  cbppp=cbsrch1+i;
  memmove(cbppp+1,cbppp,(cbsrche-cbppp+1)*sizeof(*cbppp));
  cbv->v.v_fillp++;

  return cbppp;

}

static inline void
contract_contblock_index(struct contblock ***cbppp) {

  memmove(cbppp+1,cbppp+2,(cbsrche-cbppp-1)*sizeof(*cbppp));
  cbv->v.v_fillp--;

}

static inline int
cbcomp(const void *v1,const void *v2) {

  ufixnum u1=(**(struct contblock ** const *)v1)->cb_size;
  ufixnum u2=(**(struct contblock ** const *)v2)->cb_size;

  return u1<u2 ? -1 : (u1==u2 ? 0 : 1);

}

static inline struct contblock ***
find_cbppp(struct contblock *cbp) {

  struct contblock **cbpp=&cbp;

  return cbsrche==cbsrch1 ? cbsrch1 : bsearchleq(&cbpp,cbsrch1,cbsrche-cbsrch1,sizeof(*cbsrch1),cbcomp);

}

static inline struct contblock ***
find_cbppp_by_n(ufixnum n) {

  struct contblock cb={n,NULL};

  return find_cbppp(&cb);

}

static inline struct contblock **
find_cbpp(struct contblock ***cbppp,ufixnum n) {

  return *cbppp;

}


static inline struct contblock **
find_contblock(ufixnum n,void **p) {

  *p=find_cbppp_by_n(n);
  return find_cbpp(*p,n);
}

void
print_cb(int print) {

  struct contblock *cbp,***cbppp,**cbpp=&cb_pointer;
  ufixnum k;
  
  for (cbp=cb_pointer,cbppp=cbsrch1;cbp;cbppp++) {
    massert(cbppp<cbsrche);
    massert(*cbppp);
    massert(**cbppp==cbp);
    for (k=0;cbp && cbp->cb_size==(**cbppp)->cb_size;cbpp=&cbp->cb_link,cbp=cbp->cb_link,k++);
    if (print)
      emsg("%lu %p %p %lu %lu\n",(unsigned long)(cbppp-cbsrch1),*cbppp,**cbppp,(**cbppp)->cb_size,k);
  }
  massert(cbppp==cbsrche);
  massert(*cbppp==cbpp);
  massert(!**cbppp);

}
  
void
insert_contblock(void *p,ufixnum s) {

  struct contblock *cbp=p,**cbpp,***cbppp;

  cbpp=find_contblock(s,(void **)&cbppp);

  cbp->cb_size=s;
  cbp->cb_link=*cbpp;
  
  if ((!cbp->cb_link || cbp->cb_link->cb_size!=s)) {
    cbppp=expand_contblock_index(cbppp);
    cbppp[1]=&cbp->cb_link;
  }

  *cbpp=cbp;

}

static inline void
delete_contblock(void *p,struct contblock **cbpp) {

  struct contblock ***cbppp=p;
  ufixnum s=(*cbpp)->cb_size;

  (*cbpp)=(*cbpp)->cb_link;

  if ((!(*cbpp) || (*cbpp)->cb_size!=s))
    contract_contblock_index(cbppp);

}

void
reset_contblock_freelist(void) {

  cb_pointer=NULL;
  cbv->v.v_fillp=0;
  
}

void
empty_relblock(void) {

  object o=sSAleaf_collection_thresholdA->s.s_dbind;

  sSAleaf_collection_thresholdA->s.s_dbind=make_fixnum(0);
  for (;!rb_emptyp();) {
    tm_table[t_relocatable].tm_adjgbccnt--;
    expand_contblock_index_space();
    GBC(t_relocatable);
  }
  sSAleaf_collection_thresholdA->s.s_dbind=o;

}

static inline void *
alloc_from_freelist(struct typemanager *tm,fixnum n) {

  void *p;

  switch (tm->tm_type) {

  case t_contiguous:
    {
      void *pp;
      struct contblock **cbpp=find_contblock(n,&pp);
      
      if ((p=*cbpp)) {
	ufixnum s=(*cbpp)->cb_size;
	delete_contblock(pp,cbpp);
	if (n<s)
	  insert_contblock(p+n,s-n);
      }
      return p;
    }
    break;

  case t_relocatable:
    /* if (rb_pointer>rb_end && rb_pointer+n>rb_limit && rb_pointer+n<rb_end+nrbpage*PAGESIZE)/\**\/ */
    /*   rb_limit=rb_pointer+n; */
    if (rb_limit-rb_pointer>n)
      return ((rb_pointer+=n)-n);
    break;

  default:
    if ((p=tm->tm_free)!=OBJNULL) {
      tm->tm_free = OBJ_LINK(p);
      tm->tm_nfree--;
      return(p);
    }
    break;
  }

  return NULL;

}

static inline void
grow_linear1(struct typemanager *tm) {
  
  if (!sSAoptimize_maximum_pagesA || sSAoptimize_maximum_pagesA->s.s_dbind==Cnil) {

    fixnum maxgro=resv_pages ? available_pages : 0;

    if (tm->tm_type==t_relocatable) maxgro>>=1;

    set_tm_maxpage(tm,grow_linear(tm->tm_npage,tm->tm_growth_percent,tm->tm_min_grow, tm->tm_max_grow,maxgro));

  }

}

static inline int
too_full_p(struct typemanager *tm) {

  fixnum i,j,k,pf=tm->tm_percent_free ? tm->tm_percent_free : 30;
  struct contblock *cbp;
  struct pageinfo *pi;

  switch (tm->tm_type) {
  case t_relocatable:
    return 100*(rb_limit-rb_pointer)<pf*rb_size();
    break;
  case t_contiguous:
    for (cbp=cb_pointer,k=0;cbp;cbp=cbp->cb_link) k+=cbp->cb_size;
    for (i=j=0;i<contblock_array->v.v_fillp;i++) {
      pi=(void *)contblock_array->v.v_self[i];
#ifdef SGC
      if (!sgc_enabled || pi->sgc_flags&SGC_PAGE_FLAG)
#endif
	j+=pi->in_use;
    }
    return 100*k<pf*j*PAGESIZE;
    break;
  default:
    return 100*tm->tm_nfree<pf*TOTAL_THIS_TYPE(tm);
    break;
  }

}

DEFUN("POOL-STAT",object,fSpool_stat,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {

  pool_stat();
  RETURN1(MMcons(make_fixnum(pool_pid),MMcons(make_fixnum(pool_n),MMcons(make_fixnum(pool_s),Cnil))));

}

static inline bool
do_gc_p(struct typemanager *tm,fixnum n) {

  ufixnum cpool,pp;
  
  if (!GBC_enable)
    return FALSE;

  if (!sSAoptimize_maximum_pagesA || sSAoptimize_maximum_pagesA->s.s_dbind==Cnil)
    return tm->tm_npage+tpage(tm,n)>tm->tm_maxpage;

  if ((cpool=get_pool())<=gc_page_min*phys_pages)
    return FALSE;

  pp=gc_page_max*phys_pages;

  return page(recent_allocation)>(1.0+gc_alloc_min-(double)ufmin(cpool,pp)/pp)*data_pages() ||
    2*tpage(tm,n)>available_pages;

}
  
      
static inline void *
alloc_after_gc(struct typemanager *tm,fixnum n) {

  if (do_gc_p(tm,n)) {

    switch (jmp_gmp) {
    case 0: /* not in gmp call*/
      GBC(tm->tm_calling_type);
      break;
    case 1: /* non-in-place gmp call*/
      longjmp(gmp_jmp,tm->tm_type);
      break;
    case -1: /* in-place gmp call */
      jmp_gmp=-tm->tm_type;
      break;
    default:
      break;
    }

    if (IGNORE_MAX_PAGES && too_full_p(tm))
      grow_linear1(tm);

    call_after_gbc_hook(tm->tm_type);

    return alloc_from_freelist(tm,n);

  } else

    return NULL;

}

void
add_pages(struct typemanager *tm,fixnum m) {

  switch (tm->tm_type) {
  case t_contiguous:

    add_page_to_contblock_list(alloc_page(m),m);

    break;

  case t_relocatable:

    if (rb_high() && m>((rb_start-heap_end)>>PAGEWIDTH)) {
      if (sSAnotify_gbcA->s.s_dbind != Cnil)
	emsg("[GC Moving relblock low before expanding relblock pages]\n");
      tm_table[t_relocatable].tm_adjgbccnt--;
      GBC(t_relocatable);
    }
    nrbpage+=m;
    resize_hole(page(rb_start-heap_end)-(rb_high() ? m : 0),t_relocatable,1);
    break;

  default:

    {
      void *p=alloc_page(m),*pe=p+m*PAGESIZE;
      for (;p<pe;p+=PAGESIZE)
	add_page_to_freelist(p,tm);
    }

    break;

  }

}

static inline void *
alloc_after_adding_pages(struct typemanager *tm,fixnum n) {
  
  fixnum m=tpage(tm,n);

  if (tm->tm_npage+m>tm->tm_maxpage) {

    if (!IGNORE_MAX_PAGES) return NULL;

    grow_linear1(tm);

    if (tm->tm_npage+m>tm->tm_maxpage && !set_tm_maxpage(tm,tm->tm_npage+m))
      return NULL;

  }

  add_pages(tm,m);

  return alloc_from_freelist(tm,n);

}

static inline void *
alloc_after_reclaiming_pages(struct typemanager *tm,fixnum n) {

  fixnum m=tpage(tm,n),reloc_min;

  if (tm->tm_type>t_end) return NULL;

  reloc_min=npage(rb_pointer-rb_start);

  if (m<2*(nrbpage-reloc_min)) {

    set_tm_maxpage(tm_table+t_relocatable,reloc_min);
    nrbpage=reloc_min;

    tm_table[t_relocatable].tm_adjgbccnt--;
    GBC(t_relocatable);

    return alloc_after_adding_pages(tm,n);

  }

  if (tm->tm_type>=t_end) return NULL;

  maybe_reallocate_page(tm,tm->tm_percent_free*tm->tm_npage);

  return alloc_from_freelist(tm,n);

}

static inline void *alloc_mem(struct typemanager *,fixnum);

#ifdef SGC
static inline void *
alloc_after_turning_off_sgc(struct typemanager *tm,fixnum n) {

  if (!sgc_enabled) return NULL;
  sgc_quit();
  return alloc_mem(tm,n);

}
#endif

static inline void *
alloc_mem(struct typemanager *tm,fixnum n) {

  void *p;

  CHECK_INTERRUPT;
  
  recent_allocation+=n;

  if ((p=alloc_from_freelist(tm,n)))
    return p;
  if ((p=alloc_after_gc(tm,n)))
    return p;
  if ((p=alloc_after_adding_pages(tm,n)))
    return p;
#ifdef SGC
  if ((p=alloc_after_turning_off_sgc(tm,n)))
    return p;
#endif
  if ((p=alloc_after_reclaiming_pages(tm,n)))
    return p;
  return exhausted_report(tm->tm_type,tm);
}

object
alloc_object(enum type t)  {

  object obj;
  struct typemanager *tm=tm_of(t);
  
  obj=alloc_mem(tm,tm->tm_size);
  set_type_of(obj,t);
  
  pageinfo(obj)->in_use++;

  return(obj);
  
}

void *
alloc_contblock(size_t n) {
  return alloc_mem(tm_of(t_contiguous),CEI(n,CPTR_SIZE));
}

void *
alloc_contblock_no_gc(size_t n,char *limit) {

  struct typemanager *tm=tm_of(t_contiguous);
  void *p;
  
  n=CEI(n,CPTR_SIZE);

  /*This is called from GBC so we do not want to expand the contblock index*/
  if (cbv->v.v_fillp+1==cbv->v.v_dim ||
      contblock_array->v.v_fillp==contblock_array->v.v_dim)
    return NULL;
  
  if ((p=alloc_from_freelist(tm,n)))
    return p;

  if (tpage(tm,n)<(limit-heap_end)>>PAGEWIDTH && (p=alloc_after_adding_pages(tm,n)))
    return p;

  return NULL;

}

void *
alloc_code_space(size_t sz,ufixnum max_code_address) {

  void *v;

  sz=CEI(sz,CPTR_SIZE);

  if (sSAcode_block_reserveA &&
      sSAcode_block_reserveA->s.s_dbind!=Cnil && sSAcode_block_reserveA->s.s_dbind->st.st_dim>=sz) {
    
    v=sSAcode_block_reserveA->s.s_dbind->st.st_self;
    sSAcode_block_reserveA->s.s_dbind->st.st_self+=sz;
    sSAcode_block_reserveA->s.s_dbind->st.st_dim-=sz;
    VSET_MAX_FILLP(sSAcode_block_reserveA->s.s_dbind);
    
  } else
    v=alloc_contblock(sz);

  if (v && (unsigned long)(v+sz)<max_code_address)
    return v;
  else
    FEerror("File ~a has been compiled for a restricted address space,~% and can no longer be loaded in this heap.~%"
#ifdef LARGE_MEMORY_MODEL
	    "You can recompile with :large-memory-model-p t,~% or (setq compiler::*default-large-memory-model-p* t) before recompiling."
#endif
	    ,
	    1,sLAload_pathnameA->s.s_dbind);

  return v;

}

void *
alloc_relblock(size_t n) {

  return alloc_mem(tm_of(t_relocatable),CEI(n,PTR_ALIGN));

}

static inline void
load_cons(object p,object a,object d) {
#ifdef WIDE_CONS
  set_type_of(p,t_cons);
#endif
  p->c.c_cdr=SAFE_CDR(d);
  p->c.c_car=a;
}

object
make_cons(object a,object d) {

  static struct typemanager *tm=tm_table+t_cons;/*FIXME*/
  object obj=alloc_mem(tm,tm->tm_size);

  tm->tm_calling_type=t_cons;

  load_cons(obj,a,d);

  pageinfo(obj)->in_use++;

  return(obj);

}

object
on_stack_cons(object x, object y) {
  object p = (object) alloca_val;
  load_cons(p,x,y);
  return p;
}


DEFUNM("ALLOCATED",object,fSallocated,SI,1,1,NONE,OO,OO,OO,OO,(object typ),"") { 

  struct typemanager *tm=(&tm_table[t_from_type(typ)]);
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;

  if (tm->tm_type == t_relocatable) {
    tm->tm_npage = page(rb_size());
    tm->tm_nfree = rb_limit -rb_pointer;
  } else if (tm->tm_type == t_contiguous) { 
    int cbfree =0;
    struct contblock **cbpp;
    for(cbpp= &cb_pointer; (*cbpp)!=NULL; cbpp= &(*cbpp)->cb_link)
      cbfree += (*cbpp)->cb_size ;
    tm->tm_nfree = cbfree;
  }
  
  RETURN(6,object,make_fixnum(tm->tm_nfree),
	    (RV(make_fixnum(tm->tm_npage)),
	     RV(make_fixnum(tm->tm_maxpage)),
	     RV(make_fixnum(tm->tm_nppage)),
	     RV(make_fixnum(tm->tm_gbccount)),
	     RV(make_fixnum(tm->tm_npage*tm->tm_nppage-tm->tm_nfree))));
}
 
#ifdef SGC_CONT_DEBUG
extern void overlap_check(struct contblock *,struct contblock *);
#endif

DEFUN("PRINT-FREE-CONTBLOCK-LIST",object,fSprint_free_contblock_list,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  
  struct contblock *cbp,*cbp1;

  for (cbp=cb_pointer;cbp;cbp=cbp->cb_link) {
    printf("%p %lu\n",cbp,cbp->cb_size);
    for (cbp1=cbp;cbp1;cbp1=cbp1->cb_link) 
      if ((void *)cbp+cbp->cb_size==(void *)cbp1 ||
	  (void *)cbp1+cbp1->cb_size==(void *)cbp)
	printf("  adjacent to %p %lu\n",cbp1,cbp1->cb_size);
  }

  return Cnil;

}

/* Add a tm_distinct field to prevent page type sharing if desired.
   Not used now, as its never desirable from an efficiency point of
   view, and as the only known place one must separate is cons and
   fixnum, which are of different sizes unless PTR_ALIGN is set too
   high (e.g. 16 on a 32bit machine).  See the ordering of init_tm
   calls for these types below -- reversing would wind up merging the
   types with the current algorithm.  CM 20030827 */

static void
init_tm(enum type t, char *name, int elsize, int nelts, int sgc,int distinct) {

  int i, j;
  int maxpage;
  /* round up to next number of pages */
  maxpage = (((nelts * elsize) + PAGESIZE -1)/PAGESIZE);
  tm_table[(int)t].tm_name = name;
  j=-1;
  if (!distinct)
    for (i = 0;  i < t_end;  i++)
      if (tm_table[i].tm_size != 0 &&
	  tm_table[i].tm_size == elsize &&
	  !tm_table[i].tm_distinct)
	j = i;
  if (j >= 0) {
    tm_table[(int)t].tm_type = (enum type)j;
    set_tm_maxpage(tm_table+j,tm_table[j].tm_maxpage+maxpage);
#ifdef SGC		
    tm_table[j].tm_sgc += sgc;
#endif
    return;
  }
  tm_table[(int)t].tm_type = t;
  tm_table[(int)t].tm_size = elsize ? CEI(elsize,PTR_ALIGN) : 1;
  tm_table[(int)t].tm_nppage = (PAGESIZE-sizeof(struct pageinfo))/tm_table[(int)t].tm_size;
  tm_table[(int)t].tm_free = OBJNULL;
  tm_table[(int)t].tm_nfree = 0;
  /* tm_table[(int)t].tm_nused = 0; */
  /*tm_table[(int)t].tm_npage = 0; */  /* dont zero nrbpage.. */
  set_tm_maxpage(tm_table+t,maxpage);
  tm_table[(int)t].tm_gbccount = 0;
  tm_table[(int)t].tm_adjgbccnt = 0;
  tm_table[(int)t].tm_opt_maxpage = 0;
  tm_table[(int)t].tm_distinct=distinct;

#ifdef SGC	
  tm_table[(int)t].tm_sgc = sgc;
  tm_table[(int)t].tm_sgc_max = 3000;
  tm_table[(int)t].tm_sgc_minfree = (0.4 * tm_table[(int)t].tm_nppage);
#endif
  
}

/* FIXME this is a work-around for the special MacOSX memory
   initialization sequence, which sets heap_end, traditionally
   initialized in gcl_init_alloc.  Mac and windows have non-std
   sbrk-emulating memory subsystems, and their internals need to be
   homogenized and integrated into the traditional unix sequence for
   simplicity.  set_maxpage is overloaded, and the positioning of its
   call is too fragile.  20050115 CM*/
int gcl_alloc_initialized;

object malloc_list=Cnil;

#include <signal.h>

void
maybe_set_hole_from_maxpages(void) {
  if (rb_pointer==rb_begin())
    resize_hole(ufmin(phys_pages,available_pages/3),t_relocatable,0);
}

void
gcl_init_alloc(void *cs_start) {

  fixnum cssize=(1L<<23);

#ifdef GCL_GPROF
  if (raw_image) {
    sigset_t prof;
    sigemptyset(&prof);
    sigaddset(&prof,SIGPROF);
    sigprocmask(SIG_BLOCK,&prof,NULL);
  }
#endif

  prelink_init();
  
#ifdef RECREATE_HEAP
  if (!raw_image) RECREATE_HEAP;
#endif
		    
#if defined(DARWIN)
  init_darwin_zone_compat ();
#endif
  
#if defined(BSD) && defined(RLIMIT_STACK)
  {
    struct rlimit rl;
  
  /* Maybe the soft limit for data segment size is lower than the
   * hard limit.  In that case, we want as much as possible.
   */
    massert(!getrlimit(RLIMIT_DATA, &rl));
    if (rl.rlim_cur != RLIM_INFINITY &&	(rl.rlim_max == RLIM_INFINITY || rl.rlim_max > rl.rlim_cur)) {
      rl.rlim_cur = rl.rlim_max;
      massert(!setrlimit(RLIMIT_DATA, &rl));
    }

    massert(!getrlimit(RLIMIT_STACK, &rl));
    if (rl.rlim_cur!=RLIM_INFINITY && (rl.rlim_max == RLIM_INFINITY || rl.rlim_max > rl.rlim_cur)) {
      rl.rlim_cur = rl.rlim_max; /* == RLIM_INFINITY ? rl.rlim_max : rl.rlim_max/64; */
      massert(!setrlimit(RLIMIT_STACK,&rl));
    }
    cssize = rl.rlim_cur/sizeof(*cs_org) - sizeof(*cs_org)*CSGETA;
  
  }
#endif
  
  cs_org = cs_base = cs_start;
  cs_limit = cs_org + CSTACK_DIRECTION*cssize;

#ifdef __ia64__
  {
    extern void * __libc_ia64_register_backing_store_base;
    cs_org2=cs_base2=__libc_ia64_register_backing_store_base;
  }
#endif
  
#ifdef HAVE_SIGALTSTACK
  {
    /* make sure the stack is 8 byte aligned */
    static double estack_buf[32*SIGSTKSZ];
    static stack_t estack;
    
    estack.ss_sp = estack_buf;
    estack.ss_flags = 0;                                   
    estack.ss_size = sizeof(estack_buf);                             
    massert(sigaltstack(&estack, 0)>=0);
  }
#endif	
  
  install_segmentation_catcher();
  
#ifdef HAVE_MPROTECT
  if (data_start)
    massert(!gcl_mprotect(data_start,(void *)core_end-data_start,PROT_READ|PROT_WRITE|PROT_EXEC));
#endif

#ifdef SGC

  massert(getpagesize()<=PAGESIZE);
  memprotect_test_reset();
  if (sgc_enabled)
    if (memory_protect(1))
      sgc_quit();

#endif

#ifdef INITIALIZE_BRK
  INITIALIZE_BRK;
#endif

  update_real_maxpage();

  cumulative_allocation=recent_allocation=0;

  if (gcl_alloc_initialized) {
    maybe_set_hole_from_maxpages();
    return;
  }
  
#ifdef INIT_ALLOC  
  INIT_ALLOC;
#endif  

  data_start=heap_end;
  first_data_page=page(data_start);
  
  /* Unused (at present) tm_distinct flag added.  Note that if cons
     and fixnum share page types, errors will be introduced.

     Gave each page type at least some sgc pages by default.  Of
     course changeable by allocate-sgc.  CM 20030827 */

  init_tm(t_cons, ".CONS", sizeof(struct cons), 0 ,50,0 );
  init_tm(t_fixnum, "NFIXNUM",sizeof(struct fixnum_struct), 0,20,0);
  init_tm(t_structure, "SSTRUCTURE", sizeof(struct structure),0,1,0 );
  init_tm(t_simple_string, "\'SIMPLE-STRING", sizeof(struct unadjstring),0,1,0);
  init_tm(t_string, "\"STRING", sizeof(struct string),0,1,0  );
  init_tm(t_simple_array, "ASIMPLE-ARRAY", sizeof(struct unadjarray),0,1,0 );
  init_tm(t_array, "aARRAY", sizeof(struct array),0,1,0 );
  init_tm(t_symbol, "|SYMBOL", sizeof(struct symbol),0,1,0 );
  init_tm(t_bignum, "BBIGNUM", sizeof(struct bignum),0,1,0 );
  init_tm(t_ratio, "RRATIONAL", sizeof(struct ratio),0,1,0 );
  init_tm(t_shortfloat, "FSHORT-FLOAT",sizeof(struct shortfloat_struct),0 ,1,0);
  init_tm(t_longfloat, "LLONG-FLOAT",sizeof(struct longfloat_struct),0 ,1,0);
  init_tm(t_complex, "CCOMPLEX", sizeof(struct ocomplex),0 ,1,0);
  init_tm(t_character,"#CHARACTER",sizeof(struct character),0 ,1,0);
  init_tm(t_package, ":PACKAGE", sizeof(struct package),0,1,0);
  init_tm(t_hashtable, "hHASH-TABLE", sizeof(struct hashtable),0,1,0 );
  init_tm(t_simple_vector, "VSIMPLE-VECTOR", sizeof(struct unadjvector),0 ,1,0);
  init_tm(t_vector, "vVECTOR", sizeof(struct vector),0 ,1,0);
  init_tm(t_simple_bitvector, "BSIMPLE-BIT-VECTOR", sizeof(struct unadjbitvector),0 ,1,0);
  init_tm(t_bitvector, "bBIT-VECTOR", sizeof(struct bitvector),0 ,1,0);
  init_tm(t_stream, "sSTREAM", sizeof(struct stream),0 ,1,0);
  init_tm(t_random, "$RANDOM-STATE", sizeof(struct random),0 ,1,0);
  init_tm(t_readtable, "rREADTABLE", sizeof(struct readtable),0 ,1,0);
  init_tm(t_pathname, "pPATHNAME", sizeof(struct pathname),0 ,1,0);
  init_tm(t_function, "xFUNCTION", sizeof(struct function), 85 ,1,0);
  init_tm(t_cfdata, "cCFDATA", sizeof(struct cfdata),0 ,1,0);
  init_tm(t_spice, "!SPICE", sizeof(struct spice),0 ,1,0);
  init_tm(t_relocatable, "%RELOCATABLE-BLOCKS", 0,0,20,1);
  init_tm(t_contiguous, "_CONTIGUOUS-BLOCKS", 0,0,20,1);
  
  
  ncbpage = 0;
  tm_table[t_contiguous].tm_min_grow=256;
  set_tm_maxpage(tm_table+t_contiguous,1);

  set_tm_maxpage(tm_table+t_relocatable,1);
  nrbpage=0;
  
  maybe_set_hole_from_maxpages();
#ifdef SGC	
  tm_table[(int)t_relocatable].tm_sgc = 50;
#endif
  
  expand_contblock_index_space();

  gcl_alloc_initialized=1;
  
}

DEFUN("STATICP",object,fSstaticp,SI,1,1,NONE,OO,OO,OO,OO,(object x),"Tell if the string or vector is static") {
  RETURN1((inheap(x->ust.ust_self) ? sLt : sLnil));
}

/* static void */
/* cant_get_a_type(void) { */
/*   FEerror("Can't get a type.", 0); */
/* } */

static int
t_from_type(object type) {
 
  int i;
  check_type_or_symbol_string(&type);
  type=coerce_to_string(type);
  for (i= t_start ; i < t_other ; i++)
    {struct typemanager *tm = &tm_table[i];
    if(tm->tm_name &&
       0==strncmp((tm->tm_name)+1,type->st.st_self,VLEN(type))
       )
      return i;}
  /* FEerror("Unrecognized type",0); */
  return i;

}
/* When sgc is enabled the TYPE should have at least MIN pages of sgc type,
   and at most MAX of them.   Each page should be FREE_PERCENT free
   when the sgc is turned on.  FREE_PERCENT is an integer between 0 and 100. 
   */

DEFUN("ALLOCATE-SGC",object,fSallocate_sgc,SI
      ,4,4,NONE,OO,II,II,OO,(object type,fixnum min,fixnum max,fixnum free_percent),"") {

  int t=t_from_type(type);
  struct typemanager *tm;
  object res,x,x1,x2;
  tm=tm_of(t);
  x=make_fixnum(tm->tm_sgc);
  x1=make_fixnum(tm->tm_sgc_max);
  x2=make_fixnum((100*tm->tm_sgc_minfree)/tm->tm_nppage);
  res= list(3,x,x1,x2);
  
  if(min<0 || max< min || free_percent < 0 || free_percent > 100)
    goto END;
  tm->tm_sgc_max=max;
  tm->tm_sgc=min;
  tm->tm_sgc_minfree= (tm->tm_nppage *free_percent) /100;
      END:
  RETURN1(res);

}

/* Growth of TYPE will be by at least MIN pages and at most MAX pages.
   It will try to grow PERCENT of the current pages.
   */
DEFUN("ALLOCATE-GROWTH",object,fSallocate_growth,SI,5,5,NONE,OO,II,II,OO,
      (object type,fixnum min,fixnum max,fixnum percent,fixnum percent_free),"")
{int  t=t_from_type(type);
 struct typemanager *tm=t<t_other ? tm_of(t) : NULL;
 object res,x,x1,x2,x3;
 if (!tm) RETURN1(Cnil);
 x=make_fixnum(tm->tm_min_grow);
 x1=make_fixnum(tm->tm_max_grow);
 x2=make_fixnum(tm->tm_growth_percent);
 x3=make_fixnum(tm->tm_percent_free);
 res= list(4,x,x1,x2,x3);
 
 if(min<0 || max< min || min > 3000 || percent < 0 || percent > 500 
    || percent_free <0 || percent_free > 100
    )
    goto END;
 tm->tm_max_grow=max;
 tm->tm_min_grow=min;
 tm->tm_growth_percent=percent;
 tm->tm_percent_free=percent_free;
 END:
 RETURN1(res);
}



DEFUN("ALLOCATE-CONTIGUOUS-PAGES",object,fSallocate_contiguous_pages,SI
	  ,1,2,NONE,OI,OO,OO,OO,(fixnum npages,...),"") {

  object really_do,l=Cnil,f=OBJNULL;
  va_list ap;
  fixnum nargs=INIT_NARGS(1);
  
  va_start(ap,npages);
  really_do=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);

  if  (npages  < 0)
    FEerror("Allocate requires positive argument.", 0);
  if (ncbpage > npages)
    npages=ncbpage;
  if (!set_tm_maxpage(tm_table+t_contiguous,npages))
    FEerror("Can't allocate ~D pages for contiguous blocks.", 1, make_fixnum(npages));
  if (really_do == Cnil) 
    RETURN1(Ct);
  add_pages(tm_of(t_contiguous),npages - ncbpage);

  RETURN1(make_fixnum(npages));

}

DEFUN("ALLOCATED-CONTIGUOUS-PAGES",object,fSallocated_contiguous_pages,SI
       ,0,0,NONE,OO,OO,OO,OO,(void),"")
{
	/* 0 args */
	RETURN1((make_fixnum(ncbpage)));
}

DEFUN("MAXIMUM-CONTIGUOUS-PAGES",object,fSmaximum_contiguous_pages,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  /* 0 args */
  RETURN1((make_fixnum(maxcbpage)));
}


DEFUN("ALLOCATE-RELOCATABLE-PAGES",object,fSallocate_relocatable_pages,SI,1,2,NONE,OI,OO,OO,OO,(fixnum npages,...),"") {

  object really_do,l=Cnil,f=OBJNULL;
  va_list ap;
  fixnum nargs=INIT_NARGS(1);
  
  va_start(ap,npages);
  really_do=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);
    
  if (npages  <= 0)
    FEerror("Requires positive arg",0);
  if (npages<nrbpage) npages=nrbpage;
  if (!set_tm_maxpage(tm_table+t_relocatable,npages))
    FEerror("Can't set the limit for relocatable blocks to ~D.", 1, make_fixnum(npages));
  if (really_do == Cnil) 
    RETURN1(Ct);
  add_pages(tm_of(t_relocatable),npages - nrbpage);
  RETURN1(make_fixnum(npages));

}

DEFUN("ALLOCATE",object,fSallocate,SI
	  ,2,3,NONE,OO,IO,OO,OO,(object type,fixnum npages,...),"") {

  object really_do,l=Cnil,f=OBJNULL;
  va_list ap;
  struct typemanager *tm;
  int t;
  fixnum nargs=INIT_NARGS(2);
  
  va_start(ap,npages);
  really_do=NEXT_ARG(nargs,ap,l,f,Cnil);
  va_end(ap);
  
  t= t_from_type(type);
  if (t == t_contiguous) 
    RETURN1(FUNCALL(2,FFN(fSallocate_contiguous_pages)(npages,really_do)));
  else if (t==t_relocatable) 
    RETURN1(FUNCALL(2,FFN(fSallocate_relocatable_pages)(npages,really_do)));


  if  (npages <= 0)
    FEerror("Allocate takes positive argument.", 1,make_fixnum(npages));
  tm = tm_of(t);
  if (tm->tm_npage > npages) {npages=tm->tm_npage;}
  if (!set_tm_maxpage(tm,npages))
    FEerror("Can't allocate ~D pages for ~A.", 2, make_fixnum(npages), (make_simple_string(tm->tm_name+1)));
  if (really_do == Cnil)
    RETURN1(Ct);
  add_pages(tm,npages - tm->tm_npage);
  RETURN1(make_fixnum(npages));

}

DEFUN("ALLOCATED-RELOCATABLE-PAGES",object,fSallocated_relocatable_pages,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  /* 0 args */
  RETURN1(make_fixnum(nrbpage));
}

DEFUN("GET-HOLE-SIZE",object,fSget_hole_size,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {
  /* 0 args */
  RETURN1(make_fixnum((rb_start-heap_end)>>PAGEWIDTH));
}

DEFUN("SET-STARTING-HOLE-DIVISOR",object,fSset_starting_hole_divisor,SI,1,1,NONE,II,OO,OO,OO,(fixnum div),"") {
  if (div>0 && div <100)
    starting_hole_div=div;
  return (object)starting_hole_div;
}
  
DEFUN("SET-STARTING-RELBLOCK-HEAP-MULTIPLE",object,fSset_starting_relb_heap_multiple,SI,1,1,NONE,II,OO,OO,OO,(fixnum mult),"") {
  if (mult>=0)
    starting_relb_heap_mult=mult;
  return (object)starting_relb_heap_mult;
}
  
DEFUNM("SET-HOLE-SIZE",object,fSset_hole_size,SI,1,2,NONE,OI,IO,OO,OO,(fixnum npages,...),"") {
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;

  RETURN2(make_fixnum((rb_start-heap_end)>>PAGEWIDTH),make_fixnum(reserve_pages_for_signal_handler));

}


void
gcl_init_alloc_function(void) {

  enter_mark_origin(&malloc_list);
  
}


#ifndef DONT_NEED_MALLOC

/*
	UNIX malloc simulator.

	Used by
		getwd, popen, etc.
*/



/*  If this is defined, substitute the fast gnu malloc for the slower
    version below.   If you have many calls to malloc this is worth
    your while.   I have only tested it slightly under 4.3Bsd.   There
    the difference in a test run with 120K mallocs and frees,
    was 29 seconds to 1.9 seconds */
    
#ifdef GNU_MALLOC
#include "malloc.c"
#else

/* a very young malloc may use this simple baby malloc, for the init
 code before we even get to main.c.  If this is not defined, then
 malloc will try to run the init code which will work on many machines
 but some such as WindowsNT under cygwin need this.
 
 */
#ifdef BABY_MALLOC_SIZE

/* by giving an initialization, make it not be in bss, since
   bss may not get loaded until main is reached.  We may
   not even know our own name at this stage. */
static char baby_malloc_data[BABY_MALLOC_SIZE]={1,0};
static char *last_baby= baby_malloc_data;

static char *baby_malloc(n)
     int n;
{
  char *res= last_baby;
  int m;
  n = CEI(n,PTR_ALIGN);
   m = n+ sizeof(int);
  if ((res +m-baby_malloc_data) > sizeof(baby_malloc_data))
    {
     printf("failed in baby malloc");
     do_gcl_abort();
    }
  last_baby += m;
  *((int *)res)=n;
  return res+sizeof(int);
}
#endif

/*  #ifdef HAVE_LIBBFD */

/*  int in_bfd_init=0; */

/* configure size, static init ? */
/*  static char bfd_buf[32768]; */
/*  static char *bfd_buf_p=bfd_buf; */

/*  static void * */
/*  bfd_malloc(int n) { */

/*    char *c; */

/*    c=bfd_buf_p; */
/*    n+=7; */
/*    n>>=3; */
/*    n<<=3; */
/*    if (c+n>bfd_buf+sizeof(bfd_buf)) { */
/*      fprintf(stderr,"Not enough space in bfd_buf %d %d\n",n,sizeof(bfd_buf)-(bfd_buf_p-bfd_buf)); */
/*      exit(1); */
/*    } */
/*    bfd_buf_p+=n; */
/*    return (void *)c; */

/*  } */
/*  #endif */

bool writable_malloc=0;

static void *
malloc_internal(size_t size) {

  if (!gcl_alloc_initialized) {
    static bool recursive_malloc;
    if (recursive_malloc)
      error("Bad malloc");
    recursive_malloc=1;
    gcl_init_alloc(&size);
    recursive_malloc=0;
  }

  CHECK_INTERRUPT;
  
  malloc_list = make_cons(alloc_string(size), malloc_list);
  malloc_list->c.c_car->st.st_self = alloc_contblock(size);
  malloc_list->c.c_car->st.st_writable=writable_malloc;
  
  return(malloc_list->c.c_car->st.st_self);

}

void *
malloc(size_t size) {

  return malloc_internal(size);
  
}


void
free(void *ptr) {

  object *p,pp;
  
  if (ptr == 0)
    return;
  
  for (p = &malloc_list,pp=*p; pp && !endp(pp);  p = &((pp)->c.c_cdr),pp=pp->c.c_cdr)
    if ((pp)->c.c_car->st.st_self == ptr) {
      (pp)->c.c_car->st.st_self = NULL;
      *p = pp->c.c_cdr;
      return;
    }
  {
    static void *old_ptr;
    if (old_ptr==ptr) return;
    old_ptr=ptr;
#ifndef NOFREE_ERR
    FEerror("free(3) error.",0);
#endif
  }
  return;
}
 
void *
realloc(void *ptr, size_t size) {

  object x;
  int i;
  /* was allocated by baby_malloc */
#ifdef BABY_MALLOC_SIZE	
  if (ptr >= (void*)baby_malloc_data && ptr - (void*)baby_malloc_data <BABY_MALLOC_SIZE)
    {
      int dim = ((int *)ptr)[-1];
      if (dim > size)
	return ptr;
      else
	{  char *new= malloc(size);
	bcopy(ptr,new,dim);
	return new;
	}
      
    }
#endif /*  BABY_MALLOC_SIZE	 */
  
  
  if(ptr == NULL) return malloc(size);
  for (x = malloc_list;  !endp(x);  x = x->c.c_cdr)
    if (x->c.c_car->st.st_self == ptr) {
      x = x->c.c_car;
      if (x->st.st_dim >= size) {
	VFILLP_SET(x,size);
	return(ptr);
      } else {
	x->st.st_self = alloc_contblock(size);
	x->st.st_dim = size;
	VSET_MAX_FILLP(x);
	for (i = 0;  i < size;  i++)
	  x->st.st_self[i] = ((char *)ptr)[i];
	return(x->st.st_self);
      }
    }
  FEerror("realloc(3) error.", 0);

  return NULL;

}

#endif /* gnumalloc */


void *
calloc(size_t nelem, size_t elsize)
{
	char *ptr;
	long i;

	ptr = malloc(i = nelem*elsize);
	while (--i >= 0)
		ptr[i] = 0;
	return(ptr);
}


void
cfree(void *ptr) {
  free(ptr);
}

#endif


#ifndef GNUMALLOC
#ifdef WANT_VALLOC
static void *
memalign(size_t align,size_t size) { 
  object x = alloc_string(size);
  x->st.st_self = ALLOC_ALIGNED(alloc_contblock,size,align);
  malloc_list = make_cons(x, malloc_list);
  return x->st.st_self;
}
void *
valloc(size_t size)
{ return memalign(getpagesize(),size);}
#endif

#endif
