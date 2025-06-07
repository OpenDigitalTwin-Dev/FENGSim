#define MAYBE_DATA_P(pp) ((char *)(pp)>= (char *) data_start)/*DBEGIN*/
#define VALID_DATA_ADDRESS_P(pp) (MAYBE_DATA_P(pp) &&  inheap(pp))


#ifndef page
#define page(p)	(((unsigned long)(p))>>PAGEWIDTH)
#define	pagetochar(x)	((char *)((((unsigned long)x) << PAGEWIDTH) + sizeof(struct pageinfo)))
#define pageinfo(x) ((struct pageinfo *)(((ufixnum)x)&(-PAGESIZE)))
#define pagetoinfo(x) ((struct pageinfo *)((((ufixnum)x)<<PAGEWIDTH)))
#endif
  
#ifdef UNIX
#define CHECK_FOR_INTERRUPT \
   if (interrupt_flag) sigint()
#else
#define CHECK_FOR_INTERRUPT
#endif

/* alignment required for pointers */
#ifndef PTR_ALIGN
#define PTR_ALIGN SIZEOF_LONG
#endif

/* minimum size required for contiguous pointers */
#if PTR_ALIGN < SIZEOF_CONTBLOCK
#define CPTR_SIZE SIZEOF_CONTBLOCK
#else
#define CPTR_SIZE PTR_ALIGN
#endif

#define FLR(x,r) ((x)&~(r-1))
#define CEI(x,r) FLR((x)+(r-1),r)
#define PFLR(x,r) ((void *)FLR((ufixnum)x,r))
#define PCEI(x,r) ((void *)CEI((ufixnum)x,r))

#define OBJ_ALIGNED_STACK_ALLOC(x) ({void *v=alloca((x)+OBJ_ALIGNMENT-1);PCEI(v,OBJ_ALIGNMENT);})

#ifdef SGC

#define NORMAL_PAGE 0

/* Contains objects which will be gc'd */
#define SGC_PAGE_FLAG  1       

/* keep writable eg malloc's for system call */
#define SGC_PERM_WRITABLE 2    

#define SGC_WRITABLE  (SGC_PERM_WRITABLE | SGC_PAGE_FLAG)

/* When not 0, the free lists in the type manager are freelists
   on SGC_PAGE's, for those types supporting sgc.
   Marking and sweeping is done specially */
   
int sgc_on;

#define SGC_WHOLE_PAGE /* disallow old data on sgc pages*/

#ifndef SGC_WHOLE_PAGE
/* for the S field of the FIRSTWORD */
enum sgc_type { SGC_NORMAL,   /* not allocated since the last sgc */
                SGC_RECENT    /* allocated since last sgc */
		};
#define SGC_OR_M(x)  (!TYPEWORD_TYPE_P(pageinfo(x)->type)  ? pageinfo(x)->sgc_flags&SGC_PAGE_FLAG : ((object)x)->d.s)
#endif

#define TM_BASE_TYPE_P(i) (tm_table[i].tm_type == i)

/* is this an sgc cell? encompasses all free cells.  Used where cell cannot yet be marked */

#ifndef SIGPROTV
#define SIGPROTV SIGSEGV
#endif

#ifndef INSTALL_MPROTECT_HANDLER
#define INSTALL_MPROTECT_HANDLER gcl_signal(SIGPROTV, memprotect_handler)
#endif

#else  /* END SGC */
#define sgc_quit()
#define sgc_start()
#define sgc_count_type(x) 0
#endif     

extern int sgc_enabled;
#define TM_NUSED(pt) (((pt).tm_npage*(pt).tm_nppage) - (pt).tm_nfree - (pt).tm_alt_nfree)


extern long resv_pages;
extern int reserve_pages_for_signal_handler;

extern struct pageinfo *cell_list_head,*cell_list_tail;
extern object contblock_array;

#define PAGE_MAGIC 0x2e

extern unsigned char *wrimap;
extern fixnum writable_pages;

#define CLEAR_WRITABLE(i) set_writable(i,0)
#define SET_WRITABLE(i) set_writable(i,1)
#define WRITABLE_PAGE_P(i) is_writable(i)
#define CACHED_WRITABLE_PAGE_P(i) is_writable_cached(i)
#define ON_WRITABLE_PAGE(x) WRITABLE_PAGE_P(page(x))
#define ON_WRITABLE_PAGE_CACHED(x) CACHED_WRITABLE_PAGE_P(page(x))


EXTER long first_data_page,real_maxpage,phys_pages,available_pages;
EXTER void *data_start;

#if defined(SGC)
#include "writable.h"
#endif

#define CB_BITS     CPTR_SIZE*CHAR_SIZE
#define ceil(a_,b_) (((a_)+(b_)-1)/(b_))
#define npage(m_)   ceil(m_,PAGESIZE)
#define cpage(m_)   CEI(({ufixnum _m=(m_);ceil(sizeof(struct pageinfo)+_m+2*ceil(_m,(CB_BITS-2)),PAGESIZE);}),1)
#define mbytes(p_)  ceil((p_)*PAGESIZE-sizeof(struct pageinfo),CB_BITS)
#define tpage(tm_,m_) (tm_->tm_type==t_relocatable ? npage(m_-(rb_limit-rb_pointer)+1) : (tm_->tm_type==t_contiguous ? cpage(m_) : npage(m_)))

#define CB_DATA_SIZE(z_)   ({fixnum _z=(z_);_z*PAGESIZE-2*mbytes(_z)-sizeof(struct pageinfo);})
#define CB_MARK_START(pi_) ((void *)(pi_)+sizeof(struct pageinfo))
#define CB_SGCF_START(pi_) ((void *)(pi_)+sizeof(struct pageinfo)+mbytes(pi_->in_use))
#define CB_DATA_START(pi_) ((void *)(pi_)+sizeof(struct pageinfo)+2*mbytes(pi_->in_use))
#define CB_DATA_END(pi_)   ((void *)(pi_)+PAGESIZE*(pi_)->in_use)
