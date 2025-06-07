/*  Copyright William Schelter. All rights reserved.
    Copyright 2024 Camm Maguire
    
    Stratified Garbage Collection  (SGC)
    
    Write protects pages to tell which ones have been written
    to recently, for more efficient garbage collection.
    
*/

#ifdef BSD
/* ulong may have been defined in mp.h but the define is no longer needed */
#undef ulong
#include <sys/mman.h>
#define PROT_READ_WRITE_EXEC (PROT_READ | PROT_WRITE |PROT_EXEC)
#define PROT_READ_EXEC (PROT_READ|PROT_EXEC)
#endif
#ifdef AIX3
#include <sys/vmuser.h>
#define PROT_READ_EXEC RDONLY /*FIXME*/
#define  PROT_READ_WRITE_EXEC UDATAKEY
int mprotect();
#endif

#ifdef __MINGW32__
#include <windows.h>
#define PROT_READ_WRITE_EXEC PAGE_EXECUTE_READWRITE
#define PROT_READ_EXEC PAGE_READONLY /*FIXME*/

int gclmprotect ( void *addr, size_t len, int prot ) {
    int old, rv;
    rv = VirtualProtect ( (LPVOID) addr, len, prot, &old );
    if ( 0 == rv ) {
        fprintf ( stderr, "mprotect: VirtualProtect %x %d %d failed\n", addr, len, prot );
        rv = -1;
    } else {
        rv =0;
    }    
    return (rv);
}
/* Avoid clash with libgcc's mprotect */
#define mprotect gclmprotect

#endif

#if defined(DARWIN)
#include <sys/ucontext.h>
#endif

#include <signal.h>

#ifdef SDEBUG
object sdebug;
joe1(){;}
joe() {;}     
#endif

/* structures and arrays of type t, need to be marked if their
   bodies are not write protected even if the headers are.
   So we should keep these on pages particular to them.
   Actually we will change structure sets to touch the structure
   header, that way we won't have to keep the headers in memory.
   This takes only 1.47 as opposed to 1.33 microseconds per set.
*/
static void
sgc_mark_phase(void) {

  STATIC fixnum i, j;
  STATIC struct package *pp;
  STATIC bds_ptr bdp;
  STATIC frame_ptr frp;
  STATIC ihs_ptr ihsp;
  STATIC struct pageinfo *v;
  
  mark_object(Cnil->s.s_plist);
  mark_object(Ct->s.s_plist);
  
  /* mark all non recent data on writable pages */
  {
    long t,i=page(heap_end);
    struct typemanager *tm;
    char *p;
    
    for (v=cell_list_head;v;v=v->next) {
      i=page(v);
      if (v->sgc_flags&SGC_PAGE_FLAG || !WRITABLE_PAGE_P(i)) continue;

      t=v->type;
      tm=tm_of(t);
      p=pagetochar(i);
      for (j = tm->tm_nppage; --j >= 0; p += tm->tm_size) {
	object x = (object) p; 
#ifndef SGC_WHOLE_PAGE
	if (TYPEWORD_TYPE_P(v->type) && x->d.s) continue;
#endif
	mark_object1(x);
      }
    }
  }
  
  /* mark all non recent data on writable contiguous pages */
  if (what_to_collect == t_contiguous)
    for (i=0;i<contblock_array->v.v_fillp && (v=(void *)contblock_array->v.v_self[i]);i++)
      if (v->sgc_flags&SGC_PAGE_FLAG) {
	void *s=CB_DATA_START(v),*e=CB_DATA_END(v),*p,*q;
	bool z=get_sgc_bit(v,s);
	for (p=s;p<e;) {
	  q=get_sgc_bits(v,p);
	  if (!z)
	    set_mark_bits(v,p,q);
	  z=1-z;
	  p=q;
	}
      }
	    
  mark_stack_carefully(vs_top-1,vs_org,0);
  mark_stack_carefully(MVloc+(sizeof(MVloc)/sizeof(object)),MVloc,0);

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
#ifdef KCLOVM
  if (ovm_process_created)
    sgc_mark_all_stacks();
#endif
  
#ifdef DEBUG
  if (debug) {
    printf("symbol navigation\n");
    fflush(stdout);
  }
#endif	
  
  mark_c_stack(0,N_RECURSION_REQD,mark_stack_carefully);
  
}

static void
sgc_sweep_phase(void) {
  STATIC long j, k, l;
  STATIC object x;
  STATIC char *p;
  STATIC struct typemanager *tm;
  STATIC object f;
  int size;
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
    size=tm->tm_size;

    if (v->sgc_flags&SGC_PAGE_FLAG) {

      for (j = tm->tm_nppage; --j >= 0;  p += size) {

	x = (object)p;
	
	if (is_marked(x)) {
	  unmark(x);
	  l++;
	  continue;
	}

#ifndef SGC_WHOLE_PAGE
	if (TYPEWORD_TYPE_P(v->type) && x->d.s == SGC_NORMAL)
	  continue;
#endif
	
	k++;
	make_free(x);
	SET_LINK(f,x);
	f = x;

#ifndef SGC_WHOLE_PAGE
	if (TYPEWORD_TYPE_P(v->type)) x->d.s = SGC_RECENT;
#endif

      }

      SET_LINK(f,OBJNULL);
      tm->tm_tail = f;
      tm->tm_nfree += k;
      v->in_use=l;

    } else if (WRITABLE_PAGE_P(page(v))) /*non sgc_page */
      for (j = tm->tm_nppage; --j >= 0;  p += size) {
	x = (object)p;
	if (is_marked(x)) {
	  unmark(x);
	}
      }
    
  }
}

#undef tm

#ifdef SDEBUG
sgc_count(object yy) {
  fixnum count=0;
  object y=yy;
  while(y)
    {count++;
    y=OBJ_LINK(y);}
  printf("[length %x = %d]",yy,count);
  fflush(stdout);
}

#endif

fixnum writable_pages=0;

/* count read-only pages */
static fixnum
sgc_count_read_only(void) { 

  return sgc_enabled ? sSAwritableA->s.s_dbind->v.v_dim-writable_pages : 0;

}


fixnum
sgc_count_type(int t) {

  if (t==t_relocatable)
    return page(rb_limit)-page(rb_start);
  else
    return tm_of(t)->tm_npage-tm_of(t)->tm_alt_npage;

}

#ifdef SGC_CONT_DEBUG

void
pcb(struct contblock *p) {
  for (;p;p=p->cb_link)
    printf("%p %d\n",p,p->cb_size);
}

void
overlap_check(struct contblock *t1,struct contblock *t2) {

  struct contblock *p;

  for (;t1;t1=t1->cb_link) {

    if (!inheap(t1)) {
      fprintf(stderr,"%p not in heap\n",t1);
      do_gcl_abort();
    }

    for (p=t2;p;p=p->cb_link) {

      if (!inheap(p)) {
	fprintf(stderr,"%p not in heap\n",t1);
	do_gcl_abort();
      }

      if ((p<=t1 && (void *)p+p->cb_size>(void *)t1) ||
	  (t1<=p && (void *)t1+t1->cb_size>(void *)p)) {
	fprintf(stderr,"Overlap %u %p  %u %p\n",t1->cb_size,t1,p->cb_size,p);
	do_gcl_abort();
      }
      
      if (p==p->cb_link) {
	fprintf(stderr,"circle detected at %p\n",p);
	do_gcl_abort();
      }

    }
	
    if (t1==t1->cb_link) {
      fprintf(stderr,"circle detected at %p\n",t1);
      do_gcl_abort();
    }

  }

}

void
tcc(struct contblock *t) {

  for (;t;t=t->cb_link) {

    if (!inheap(t)) {
      fprintf(stderr,"%p not in heap\n",t);
      break;
    }

    fprintf(stderr,"%u at %p\n",t->cb_size,t);

    if (t==t->cb_link) {
      fprintf(stderr,"circle detected at %p\n",t);
      break;
    }

  }

}

#endif	  

typedef enum {memprotect_none,memprotect_cannot_protect,memprotect_sigaction,
	      memprotect_bad_return,memprotect_no_signal,
	      memprotect_multiple_invocations,memprotect_no_restart,
	      memprotect_bad_fault_address,memprotect_success} memprotect_enum;
static volatile memprotect_enum memprotect_result;
static int memprotect_handler_invocations,memprotect_print_enable;
static void *memprotect_test_address;

#define MEM_ERR_CASE(a_) \
  case a_: \
    fprintf(stderr,"The SGC segfault recovery test failed with %s, SGC disabled\n",#a_); \
    break

static void
memprotect_print(void) {

  if (!memprotect_print_enable)
    return;

  switch(memprotect_result) {
  case memprotect_none: case memprotect_success:
    break;

    MEM_ERR_CASE(memprotect_cannot_protect);
    MEM_ERR_CASE(memprotect_sigaction);
    MEM_ERR_CASE(memprotect_bad_return);
    MEM_ERR_CASE(memprotect_no_signal);
    MEM_ERR_CASE(memprotect_no_restart);
    MEM_ERR_CASE(memprotect_bad_fault_address);
    MEM_ERR_CASE(memprotect_multiple_invocations);

  }

}


static void
memprotect_handler_test(int sig, long code, void *scp, char *addr) {

  char *faddr;
  faddr=GET_FAULT_ADDR(sig,code,scp,addr); 

  if (memprotect_handler_invocations) {
    memprotect_result=memprotect_multiple_invocations;
    do_gcl_abort();
  }
  memprotect_handler_invocations=1;
  if (page(faddr)!=page(memprotect_test_address))
    memprotect_result=memprotect_bad_fault_address;
  else
    memprotect_result=memprotect_none;
  gcl_mprotect(memprotect_test_address,PAGESIZE,PROT_READ_WRITE_EXEC);

}

static int
memprotect_test(void) {

  char *b1,*b2;
  unsigned long p=PAGESIZE;
  struct sigaction sa,sao,saob;

  if (memprotect_result!=memprotect_none)
    return memprotect_result!=memprotect_success;
  if (atexit(memprotect_print)) {
    fprintf(stderr,"Cannot setup memprotect_print on exit\n");
    do_gcl_abort();
  }

  if (!(b1=alloca(2*p))) {
    memprotect_result=memprotect_cannot_protect;
    return -1;
  }

  if (!(b2=alloca(p))) {
    memprotect_result=memprotect_cannot_protect;
    return -1;
  }

  memset(b1,32,2*p);
  memset(b2,0,p);
  memprotect_test_address=(void *)(((unsigned long)b1+p-1) & ~(p-1));
  sa.sa_sigaction=(void *)memprotect_handler_test;
  sa.sa_flags=MPROTECT_ACTION_FLAGS;
  if (sigaction(SIGSEGV,&sa,&sao)) {
    memprotect_result=memprotect_sigaction;
    return -1;
  }
  if (sigaction(SIGBUS,&sa,&saob)) {
    sigaction(SIGSEGV,&sao,NULL);
    memprotect_result=memprotect_sigaction;
    return -1;
  }
  { /* mips kernel bug test -- SIGBUS with no faddr when floating point is emulated. */
    float *f1=(void *)memprotect_test_address,*f2=(void *)b2,*f1e=f1+p/sizeof(*f1);
  
    if (gcl_mprotect(memprotect_test_address,p,PROT_READ_EXEC)) {
      memprotect_result=memprotect_cannot_protect;
      return -1;
    }
    memprotect_result=memprotect_bad_return;
    for (;f1<f1e;) *f1++=*f2;
    if (memprotect_result==memprotect_bad_return)
      memprotect_result=memprotect_no_signal;
    if (memprotect_result!=memprotect_none) {
      sigaction(SIGSEGV,&sao,NULL);
      sigaction(SIGBUS,&saob,NULL);
      return -1;
    }
    memprotect_handler_invocations=0;

  }
  if (gcl_mprotect(memprotect_test_address,p,PROT_READ_EXEC)) {
    memprotect_result=memprotect_cannot_protect;
    return -1;
  }
  memprotect_result=memprotect_bad_return;
  memset(memprotect_test_address,0,p);
  if (memprotect_result==memprotect_bad_return)
    memprotect_result=memprotect_no_signal;
  if (memprotect_result!=memprotect_none) {
    sigaction(SIGSEGV,&sao,NULL);
    sigaction(SIGBUS,&saob,NULL);
    return -1;
  }
  if (memcmp(memprotect_test_address,b2,p)) {
    memprotect_result=memprotect_no_restart;
    sigaction(SIGSEGV,&sao,NULL);
    sigaction(SIGBUS,&saob,NULL);
    return -1;
  }
  memprotect_result=memprotect_success;
  sigaction(SIGSEGV,&sao,NULL);
  sigaction(SIGBUS,&saob,NULL);
  return 0;

}

static int
do_memprotect_test(void) {

  int rc=0;

  memprotect_print_enable=1;
  if (memprotect_test()) {
    memprotect_print();
    if (sgc_enabled)
      sgc_quit();
    rc=-1;
  }
  memprotect_print_enable=0;
  return rc;

}

void
memprotect_test_reset(void) {

  memprotect_result=memprotect_none;
  memprotect_handler_invocations=0;
  memprotect_test_address=NULL;

  if (sgc_enabled)
    do_memprotect_test();

}

#define MMIN(a,b) ({long _a=a,_b=b;_a<_b ? _a : _b;})
#define MMAX(a,b) ({long _a=a,_b=b;_a>_b ? _a : _b;})
/* If opt_maxpage is set, don't lose balancing information gained thus
   far if we are triggered 'artificially' via a hole overrun. FIXME --
   try to allocate a small working set with the right proportions
   later on. 20040804 CM*/
#define WSGC(tm) ({struct typemanager *_tm=tm;long _t=MMAX(MMIN(_tm->tm_opt_maxpage,_tm->tm_npage),_tm->tm_sgc);_t*scale;})
/* If opt_maxpage is set, add full pages to the sgc set if needed
   too. 20040804 CM*/
/* #define FSGC(tm) (tm->tm_type==t_cons ? tm->tm_nppage : (tm->tm_opt_maxpage ? 0 : tm->tm_sgc_minfree)) */
#ifdef SGC_WHOLE_PAGE
#define FSGC(tm) tm->tm_nppage
#else
#define FSGC(tm) (!TYPEWORD_TYPE_P(tm->tm_type) ? tm->tm_nppage : tm->tm_sgc_minfree)
#endif

DEFVAR("*WRITABLE*",sSAwritableA,SI,Cnil,"");

unsigned char *wrimap=NULL;

int
sgc_start(void) {

  long i,count,minfree,allocate_more_pages=!saving_system && 10*available_pages>2*(real_maxpage-first_data_page);
  long np;
  struct typemanager *tm;
  struct pageinfo *v;
  object omp=sSAoptimize_maximum_pagesA->s.s_dbind;
  double tmp,scale;

  allocate_more_pages=0;
  if (sgc_enabled)
    return 1;

  sSAoptimize_maximum_pagesA->s.s_dbind=Cnil;
  
  if (memprotect_result!=memprotect_success && do_memprotect_test())
    return 0;

  empty_relblock();

  /* Reset maxpage statistics if not invoked automatically on a hole
     overrun. 20040804 CM*/
  /* if (!hole_overrun) { */
  /*   vs_mark; */
  /*   object *old_vs_base=vs_base; */
  /*   vs_base=vs_top; */
  /*   FFN(siLreset_gbc_count)(); */
  /*   vs_base=old_vs_base; */
  /*   vs_reset; */
  /* } */

  for (i=t_start,scale=1.0,tmp=0.0;i<t_other;i++)
    if (TM_BASE_TYPE_P(i))
      tmp+=WSGC(tm_of(i));
  tmp+=WSGC(tm_of(t_relocatable));
  scale=tmp>available_pages/10 ? (float)available_pages/(10*tmp) : 1.0;

  for (i= t_start; i < t_contiguous ; i++) {
    
    if (!TM_BASE_TYPE_P(i) || !(np=(tm=tm_of(i))->tm_sgc)) continue;

    minfree = FSGC(tm) > 0 ? FSGC(tm) : 1;
    count=0;

  FIND_FREE_PAGES:

    for (v=cell_list_head;v && (count<MMAX(tm->tm_sgc_max,WSGC(tm)));v=v->next) {

      if (v->type!=i || tm->tm_nppage-v->in_use<minfree) continue;

      v->sgc_flags|=SGC_PAGE_FLAG;
      count++;

    }

    if (count<WSGC(tm) && !FSGC(tm)) 
      for (v=cell_list_head;v && (count<MMAX(tm->tm_sgc_max,WSGC(tm)));v=v->next) {

	if (v->type!=i || tm->tm_nppage!=v->in_use) continue;
	
	v->sgc_flags|=SGC_PAGE_FLAG;
	count++;
	if (count >= MMAX(tm->tm_sgc_max,WSGC(tm)))
	  break; 
      }

    /* don't do any more allocations  for this type if saving system */
    if (!allocate_more_pages) 
      continue;
    
    if (count < WSGC(tm)) {
      /* try to get some more free pages of type i */
      long n = WSGC(tm) - count;
      long again=0,nfree = tm->tm_nfree;
      char *p=alloc_page(n);
      if (tm->tm_nfree > nfree) again=1;  /* gc freed some objects */
      if (tm->tm_npage+n>tm->tm_maxpage)
	if (!set_tm_maxpage(tm,tm->tm_npage+n))
	  n=0;
      while (n-- > 0) {
	/* (sgc_enabled=1,add_page_to_freelist(p,tm),sgc_enabled=0); */
	add_page_to_freelist(p,tm);
	p += PAGESIZE;
      }
      if (again) 
	goto FIND_FREE_PAGES;	 
    }

  }


/* SGC cont pages: Here we implement the contblock page division into
   SGC and non-SGC types.  Unlike the other types, we need *whole*
   free pages for contblock SGC, as there is no persistent data
   element (e.g. .m) on an allocated block itself which can indicate
   its live status.  If anything on a page which is to be marked
   read-only points to a live object on an SGC cont page, it will
   never be marked and will be erroneously swept.  It is also possible
   for dead objects to unnecessarily mark dead regions on SGC pages
   and delay sweeping until the pointing type is GC'ed if SGC is
   turned off for the pointing type, e.g. tm_sgc=0. (This was so by
   default for a number of types, including bignums, and has now been
   corrected in gcl_init_alloc in alloc.c.) We can't get around this
   AFAICT, as old data on (writable) SGC pages must be marked lest it
   is lost, and (old) data on now writable non-SGC pages might point
   to live regions on SGC pages, yet might not themselves be reachable
   from the mark origin through an unbroken chain of writable pages.
   In any case, the possibility of a lot of garbage marks on contblock
   pages, especially when the blocks are small as in bignums, makes
   necessary the sweeping of minimal contblocks to prevent leaks. CM
   20030827 */

  {

    void *p=NULL,*pe;
    struct pageinfo *pi;
    fixnum i,j,count=0;
    struct contblock **cbpp;
    
    tm=tm_of(t_contiguous);

    for (i=0;i<contblock_array->v.v_fillp && (pi=(void *)contblock_array->v.v_self[i]) && count<WSGC(tm);i++) {

      p=CB_DATA_START(pi);
      pe=CB_DATA_END(pi);

      for (cbpp=&cb_pointer,j=0;*cbpp;cbpp=&(*cbpp)->cb_link)
	if ((void*)*cbpp>=p && (void *)*cbpp<pe)
	  j+=(*cbpp)->cb_size;

      if (j*tm->tm_nppage<FSGC(tm)*(CB_DATA_END(pi)-CB_DATA_START(pi))) continue;

      pi->sgc_flags=SGC_PAGE_FLAG;
      count+=pi->in_use;

    }
    i=allocate_more_pages ? WSGC(tm) : (saving_system ? 1 : 0);
    
    if (i>count) {
      /* SGC cont pages: allocate more if necessary, dumping possible
	 GBC freed pages onto the old contblock list.  CM 20030827*/
      unsigned long z=(i-count)+1;
      ufixnum fp=contblock_array->v.v_fillp;

      if (maxcbpage<ncbpage+z)
	if (!set_tm_maxpage(tm_table+t_contiguous,ncbpage+z))
	  z=0;

      add_pages(tm_table+t_contiguous,z);

      massert(fp!=contblock_array->v.v_fillp);

      ((struct pageinfo *)contblock_array->v.v_self[fp])->sgc_flags=SGC_PAGE_FLAG;

    }

  }

  sSAwritableA->s.s_dbind=fSmake_vector(sLbit,(page(heap_end)-first_data_page),Ct,Cnil,Cnil,0,Ct,Cnil);
  wrimap=(void *)sSAwritableA->s.s_dbind->v.v_self;

  /* now move the sgc free lists into place.   alt_free should
     contain the others */
  for (i= t_start; i < t_contiguous ; i++)
    if (TM_BASE_TYPE_P(i) && (np=(tm=tm_of(i))->tm_sgc)) {
      object f=tm->tm_free,xf,yf;
      struct freelist x,y;/*the f_link heads have to be separated on the stack*/
      fixnum count=0;
      
      xf=PHANTOM_FREELIST(x.f_link);
      yf=PHANTOM_FREELIST(y.f_link);
      while (f!=OBJNULL) {
#ifdef SDEBUG	     
	if (!is_free(f))
	  printf("Not FREE in freelist f=%d",f);
#endif
	if (pageinfo(f)->sgc_flags&SGC_PAGE_FLAG) {
	  SET_LINK(xf,f);
#ifndef SGC_WHOLE_PAGE
	  if (TYPEWORD_TYPE_P(pageinfo(f)->type)) f->d.s = SGC_RECENT;
#endif
	  xf=f;
	  count++;
	} else {
	  SET_LINK(yf,f);
#ifndef SGC_WHOLE_PAGE
 	  if (TYPEWORD_TYPE_P(pageinfo(f)->type)) f->d.s = SGC_NORMAL;
#endif
	  yf=f;
	}
	f=OBJ_LINK(f);
      }
      SET_LINK(xf,OBJNULL);
      tm->tm_free = OBJ_LINK(&x);
      tm->tm_tail = xf;
      SET_LINK(yf,OBJNULL);
      tm->tm_alt_free = OBJ_LINK(&y);
      tm->tm_alt_nfree = tm->tm_nfree - count;
      tm->tm_nfree=count;
    }

  {

    struct pageinfo *pi;
    ufixnum j;
    
    {

      struct contblock **cbpp;
      void *p=NULL,*pe;
      struct pageinfo *pi;
      ufixnum i;

      old_cb_pointer=cb_pointer;
      reset_contblock_freelist();

      for (i=0;i<contblock_array->v.v_fillp && (pi=(void *)contblock_array->v.v_self[i]);i++) {
	
	if (pi->sgc_flags!=SGC_PAGE_FLAG) continue;
	
	p=CB_DATA_START(pi);
	pe=p+CB_DATA_SIZE(pi->in_use);
	
	for (cbpp=&old_cb_pointer;*cbpp;)
	  if ((void *)*cbpp>=p && (void *)*cbpp<pe) {
	    void *s=*cbpp,*e=s+(*cbpp)->cb_size,*l=(*cbpp)->cb_link;
	    set_sgc_bits(pi,s,e);
	    insert_contblock(s,e-s);
	    *cbpp=l;
	  } else
	    cbpp=&(*cbpp)->cb_link;

      }
      
#ifdef SGC_CONT_DEBUG
      overlap_check(old_cb_pointer,cb_pointer);
#endif
    }

    for (i=t_start;i<t_other;i++)
      tm_of(i)->tm_alt_npage=0;
    writable_pages=0;

    for (pi=cell_list_head;pi;pi=pi->next) {
      if (pi->sgc_flags&SGC_WRITABLE)
	SET_WRITABLE(page(pi));
      else
	tm_of(pi->type)->tm_alt_npage++;
    }
    for (j=0;j<contblock_array->v.v_fillp && (pi=(void *)contblock_array->v.v_self[j]);j++)
      if (pi->sgc_flags&SGC_WRITABLE)
	for (i=0;i<pi->in_use;i++)
	  SET_WRITABLE(page(pi)+i);
      else
	tm_of(t_contiguous)->tm_alt_npage+=pi->in_use;
    {
      extern object malloc_list;
      object x;

      for (x=malloc_list;x!=Cnil;x=x->c.c_cdr)
	if (x->c.c_car->st.st_writable)
	  for (i=page(x->c.c_car->st.st_self);i<=page(x->c.c_car->st.st_self+VLEN(x->c.c_car)-1);i++)
	    SET_WRITABLE(i);
    }

    {
      object v=sSAwritableA->s.s_dbind;
      for (i=page(v->v.v_self);i<=page(v->v.v_self+CEI(v->bv.bv_offset+v->v.v_dim-1,8*sizeof(fixnum))/(8*sizeof(fixnum)));i++)
	SET_WRITABLE(i);
      SET_WRITABLE(page(v));
      SET_WRITABLE(page(sSAwritableA));
    }

    tm_of(t_relocatable)->tm_alt_npage=0;

    fault_pages=0;

  }

  /* Whew.   We have now allocated the sgc space
     and modified the tm_table;
     Turn  memory protection on for the pages which are writable.
  */
  sgc_enabled=1;
  if (memory_protect(1))
    sgc_quit();
  if (sSAnotify_gbcA->s.s_dbind != Cnil)
    emsg("[SGC on]");

  sSAoptimize_maximum_pagesA->s.s_dbind=omp;

  return 1;
  
}

/* int */
/* pdebug(void) { */

/*   extern object malloc_list; */
/*   object x=malloc_list; */
/*   struct pageinfo *v; */
/*   for (;x!=Cnil;x=x->c.c_cdr)  */
/*     printf("%p %d\n",x->c.c_car->st.st_self,x->c.c_car->st.st_dim); */

/*   for (v=contblock_list_head;v;v=v->next) */
/*     printf("%p %ld\n",v,v->in_use<<12); */
/*   return 0; */
/* } */


int
sgc_quit(void) { 

  struct typemanager *tm;
  struct contblock *tmp_cb_pointer,*next;
  unsigned long i,np;
  struct pageinfo *v;

  memory_protect(0);

  if(sSAnotify_gbcA->s.s_dbind != Cnil) 
    emsg("[SGC off]");

  if (sgc_enabled==0) 
    return 0;

  sSAwritableA->s.s_dbind=Cnil;
  wrimap=NULL;

  sgc_enabled=0;

  /* SGC cont pages: restore contblocks, each tmp_cb_pointer coming
     from the new list is guaranteed not to be on the old. Need to
     grab 'next' before insert_contblock writes is.  CM 20030827 */

  if (old_cb_pointer) {
#ifdef SGC_CONT_DEBUG
    overlap_check(old_cb_pointer,cb_pointer);
#endif
    for (tmp_cb_pointer=old_cb_pointer;tmp_cb_pointer;  tmp_cb_pointer=next) {
      next=tmp_cb_pointer->cb_link;
      insert_contblock((void *)tmp_cb_pointer,tmp_cb_pointer->cb_size);
    }
  }

  for (i= t_start; i < t_contiguous ; i++)
    
    if (TM_BASE_TYPE_P(i) && (np=(tm=tm_of(i))->tm_sgc)) {

      object n=tm->tm_free,o=tm->tm_alt_free,f=PHANTOM_FREELIST(tm->tm_free);

      for (;n!=OBJNULL && o!=OBJNULL;)
	if (o!=OBJNULL && (n==OBJNULL || o<n)) {
	  SET_LINK(f,o);
	  f=o;
	  o=OBJ_LINK(o);
	} else {
	  SET_LINK(f,n);
	  f=n;
	  n=OBJ_LINK(n);
	}
      SET_LINK(f,n!=OBJNULL ? n : o);
      tm->tm_tail=f;
      for (;OBJ_LINK(tm->tm_tail)!=OBJNULL;tm->tm_tail=OBJ_LINK(tm->tm_tail));
      tm->tm_nfree += tm->tm_alt_nfree;
      tm->tm_alt_nfree = 0;
      tm->tm_alt_free = OBJNULL;
      
    }

  /*FIXME*/
  /* remove the recent flag from any objects on sgc pages */
#ifndef SGC_WHOLE_PAGE
  for (v=cell_list_head;v;v=v->next)
    if (v->type==(tm=tm_of(v->type))->tm_type && TYPEWORD_TYPE_P(v->type) && v->sgc_flags & SGC_PAGE_FLAG)
      for (p=pagetochar(page(v)),j=tm->tm_nppage;j>0;--j,p+=tm->tm_size)
  	((object) p)->d.s=SGC_NORMAL;
#endif

  for (i=0;i<contblock_array->v.v_fillp &&(v=(void *)contblock_array->v.v_self[i]);i++)
    if (v->sgc_flags&SGC_PAGE_FLAG) 
      bzero(CB_SGCF_START(v),CB_DATA_START(v)-CB_SGCF_START(v));
  
  {
    struct pageinfo *pi;
    for (pi=cell_list_head;pi;pi=pi->next)
      pi->sgc_flags&=SGC_PERM_WRITABLE;
    for (i=0;i<contblock_array->v.v_fillp &&(pi=(void *)contblock_array->v.v_self[i]);i++)
      pi->sgc_flags&=SGC_PERM_WRITABLE;
  }
  
  return 0;
  
}

fixnum debug_fault =0;
fixnum fault_count =0;

static void
memprotect_handler(int sig, long code, void *scp, char *addr) {
  
  unsigned long p;
  void *faddr;  /* Needed because we must not modify signal handler
		   arguments on the stack! */
#ifdef GET_FAULT_ADDR
  faddr=GET_FAULT_ADDR(sig,code,scp,addr); 
  debug_fault = (long) faddr;
#ifdef DEBUG_MPROTECT
  printf("fault:0x%x [%d] (%d)  ",faddr,page(faddr),faddr >= core_end);
#endif 
  if (faddr >= (void *)core_end || faddr < data_start) {
    static void *old_faddr;
    if (old_faddr==faddr)
      if (fault_count++ > 300) error("fault count too high");
    old_faddr=faddr;
    INSTALL_MPROTECT_HANDLER;
    return;
  }
#else
  faddr = addr;
#endif 
  p = page(faddr);
  if (p >= first_protectable_page
      && faddr < (void *)core_end
      && !(WRITABLE_PAGE_P(p))) {
    /*   CHECK_RANGE(p,1); */
#ifdef DEBUG_MPROTECT
    printf("mprotect(0x%x,0x%x,0x%x)\n",
	   pagetoinfo(p),PAGESIZE, sbrk(0));
    fflush(stdout);
#endif     
    
#ifndef BSD
    INSTALL_MPROTECT_HANDLER;
#endif

    massert(!gcl_mprotect(pagetoinfo(p),PAGESIZE,PROT_READ_WRITE_EXEC));
    SET_WRITABLE(p);
    fault_pages++;

    return;

  }
  
#ifndef  BSD
  INSTALL_MPROTECT_HANDLER;
#endif

  segmentation_catcher(sig,code,scp,addr);

}

static int
sgc_mprotect(long pbeg, long n, int writable) {
  /* CHECK_RANGE(pbeg,n);  */
#ifdef DEBUG_MPROTECT
  printf("prot[%d,%d,(%d),%s]\n",pbeg,pbeg+n,writable & SGC_WRITABLE,
	 (writable  & SGC_WRITABLE ? "writable" : "not writable"));
  printf("mprotect(0x%x,0x%x), sbrk(0)=0x%x\n",
	 pagetoinfo(pbeg), n * PAGESIZE, sbrk(0));
  fflush(stdout);
#endif  
  if(gcl_mprotect(pagetoinfo(pbeg),n*PAGESIZE,(writable & SGC_WRITABLE ? PROT_READ_WRITE_EXEC : PROT_READ_EXEC))) {
    perror("sgc disabled");
    return -1;
  }

  return 0;

}



int
memory_protect(int on) {

  unsigned long i,beg,end= page(core_end);
  int writable=1;
  extern void install_segmentation_catcher(void);


  first_protectable_page=first_data_page;

  /* turning it off */
  if (on==0) {
    sgc_mprotect(first_protectable_page,end-first_protectable_page,SGC_WRITABLE);
    install_segmentation_catcher();
    return 0;
  }

  INSTALL_MPROTECT_HANDLER;

  beg=first_protectable_page;
  writable = WRITABLE_PAGE_P(beg);
  for (i=beg ; ++i<= end; ) {

    if (writable==WRITABLE_PAGE_P(i) && i<end) continue;

    if (sgc_mprotect(beg,i-beg,writable))
      return -1;
    writable=1-writable;
    beg=i;

  }

  return 0;

}

static void
FFN(siLsgc_on)(void) {

  if (vs_base==vs_top) {
    vs_base[0]=(sgc_enabled ? Ct :Cnil);
    vs_top=vs_base+1; return;
  }
  check_arg(1);
  if(vs_base[0]==Cnil) 
    sgc_quit();
  else 
    vs_base[0]=sgc_start() ? Ct : Cnil;
}

void
system_error(void) {
  FEerror("System error",0);
}
