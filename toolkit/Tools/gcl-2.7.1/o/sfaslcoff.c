/* Copyright (C) 2024 Camm Maguire */
#include <string.h>

#include "windows.h"

typedef unsigned char  uc;
typedef unsigned short us;
typedef unsigned int  ul;

struct filehdr {
  us f_magic;	/* magic number			*/
  us f_nscns;	/* number of sections		*/
  ul f_timdat;	/* time & date stamp		*/
  ul f_ptrsym;	/* file pointer to symtab	*/
  ul f_symnum;	/* number of symtab entries	*/
  us f_opthdr;	/* sizeof(optional hdr)		*/
  us f_flags;	/* flags			*/
};

struct opthdr {
  us h_magic;
  uc h_mlv;
  uc h_nlv;
  ul h_tsize;
  ul h_dsize;
  ul h_bsize;
  ul h_maddr;
  ul h_tbase;
  ul h_dbase;           /* = high 32 bits of ibase for PE32+, magic 0x20b*/
  ul h_ibase;
};

struct scnhdr {
  uc	s_name[8];      /* section name  */
  ul	s_paddr;	/* physical address, aliased s_nlib */
  ul  	s_vaddr;	/* virtual address		*/
  ul  	s_size;	        /* section size			*/
  ul  	s_scnptr;	/* file ptr to raw data for section */
  ul  	s_relptr;	/* file ptr to relocation	*/
  ul  	s_lnnoptr;	/* file ptr to line numbers	*/
  us   	s_nreloc;	/* number of relocation entries	*/
  us   	s_nlnno;	/* number of line number entries*/
  ul  	s_flags;	/* flags			*/
};
#define SEC_CODE 0x20
#define SEC_DATA 0x40
#define SEC_BSS  0x80
#define ALLOC_SEC(sec) (sec->s_flags&(SEC_CODE|SEC_DATA|SEC_BSS))
#define  LOAD_SEC(sec) (sec->s_flags&(SEC_CODE|SEC_DATA))

#define NM(sym_,tab_,nm_,op_)				\
  ({char _c=0,*nm_;					\
    if ((sym_)->n.n.n_zeroes)				\
      {(nm_)=(sym_)->n.n_name;_c=(nm_)[8];(nm_)[8]=0;}	\
    else						\
      (nm_)=(tab_)+(sym_)->n.n.n_offset;		\
    op_;						\
    if (_c) (nm_)[8]=_c;				\
  })


struct reloc {
  union {
        ul   r_vaddr;
        ul   r_count;    /* Set to the real count when IMAGE_SCN_LNK_NRELOC_OVFL is set */
    } r;
    ul    r_symndx;
    us    r_type;
} __attribute__ ((packed));
#define R_ABS         0x0000  /* absolute, no relocation is necessary */
#define R_DIR32       0x0006  /* Direct 32-bit reference to the symbols virtual address */
#define R_SECREL32    0x000B  /* Currently ignored, used only for debugging strings FIXME */
#define R_PCRLONG     0x0014  /* 32-bit reference pc relative to the symbols virtual address */

#define IMAGE_REL_AMD64_REL32  0x0004  /* 32-bit reference pc relative to the symbols virtual address */
#define IMAGE_REL_AMD64_ADDR64 0x0001  /* The 64-bit VA of the relocation target */
#define IMAGE_REL_AMD64_ADDR32NB 0x0003  /* The 32-bit address without an image base (RVA) */

struct syment {
  union {
    char n_name[8];
    struct {
      int n_zeroes;
      int n_offset;
    } n;
  } n;
  ul    n_value;
  short n_scnum;
  us    n_type;
  uc    n_sclass;
  uc    n_numaux;
} __attribute__ ((packed));


static int
ovchk(ul v,ul m) {

  m|=m>>1;
  v&=m;

  return (!v || v==m);

}

static int
store_val(ul *w,ul m,ul v) {

  massert(ovchk(v,~m));
  *w=(v&m)|(*w&~m);

  return 0;

}

static int
add_val(ul *w,ul m,ul v) {

  return store_val(w,m,v+(*w&m));

}


static unsigned long self_ibase;
#define sym_lvalue(sym_) (!sym_->n_scnum ? self_ibase+sym_->n_value : (unsigned long)start+sym_->n_value)

static void
relocate(struct scnhdr *sec,struct reloc *rel,struct syment *sym,void *start) {

  ul *where=start+(sec->s_paddr+rel->r.r_vaddr);

  switch(rel->r_type) {

  case R_ABS:
  case R_SECREL32:
    break;

  case IMAGE_REL_AMD64_ADDR64:
    add_val(where,~0L,sym_lvalue(sym));
#if SIZEOF_LONG == 8
    add_val(where+1,~0L,sym_lvalue(sym)>>32);
#endif
    break;

  case IMAGE_REL_AMD64_ADDR32NB:
    add_val(where,~0L,sym->n_value);
    break;

  case R_DIR32:
    add_val(where,~0L,sym_lvalue(sym));
    break;

  case R_PCRLONG:
  case IMAGE_REL_AMD64_REL32:
    add_val(where,~0L,(ul)((void *)sym_lvalue(sym)-(void *)(where+1)));
    break;

  default:
    fprintf(stdout, "%d: unsupported relocation type.", rel->r_type);
    FEerror("The relocation type was unknown",0);

  }

}


static void
find_init_address(struct syment *sym,struct syment *sye,ul *ptr,char *st1) {

  for(;sym<sye;sym++) {

    if (*ptr==0 && sym->n_scnum == 1 && sym->n_value) {
      char *s=sym->n.n.n_zeroes ? sym->n.n_name : st1+sym->n.n.n_offset;
      if (!strncmp(s,"init_",5) || !strncmp(s,"_init_",6))
	*ptr=sym->n_value;
    }

    sym += (sym)->n_numaux;

  }

}    

static ul
get_sym_svalue(const char *name) {

  struct node *answ;

  return (answ=find_sym_ptable(name)) ? answ->address-self_ibase :
    ({massert(!emsg("Unrelocated non-local symbol: %s\n",name));0;});

}

static void
relocate_symbols(struct syment *sym,struct syment *sye,struct scnhdr *sec1,char *st1) {

  long value;

  for (;sym<sye;sym++) {

    if (sym->n_scnum>0)
      sym->n_value = sec1[sym->n_scnum-1].s_paddr;

    else if (!sym->n_scnum) {

      NM(sym,st1,s,value=get_sym_svalue(s));

      sym->n_value=value;

    }

    sym += (sym)->n_numaux;

  }

}

static object
load_memory(struct scnhdr *sec1,struct scnhdr *sece,void *st,ul *init_address) {

  object memory;
  struct scnhdr *sec;
  ul sz,a,ma;

  BEGIN_NO_INTERRUPT;

  for (sec=sec1,ma=sz=0;sec<sece;sec++)
    if (ALLOC_SEC(sec)) {

      a=1<<(((sec->s_flags>>20)&0xf)-1);
      massert(a<=8192);
      ma=ma ? ma : a;
      sz=(sz+a-1)&~(a-1);
      sec->s_paddr=sz;
      sz+=sec->s_size;

    }

  ma=ma>sizeof(struct contblock) ? ma-1 : 0;
  sz+=ma;

  memory=new_cfdata();
  memory->cfd.cfd_size=sz;
  memory->cfd.cfd_start=alloc_code_space(sz,-1UL);

  a=(((unsigned long)memory->cfd.cfd_start+ma)&~ma)-((unsigned long)memory->cfd.cfd_start);
  *init_address+=a;
  for (sec=sec1;sec<sece;sec++) {
    if (ALLOC_SEC(sec)) {
      sec->s_paddr+=a;
      if (LOAD_SEC(sec))
	memcpy((void *)memory->cfd.cfd_start+sec->s_paddr,st+sec->s_scnptr,sec->s_size);
      else
	bzero((void *)memory->cfd.cfd_start+sec->s_paddr,sec->s_size);
    }
  }

  END_NO_INTERRUPT;

  return memory;

}

static int
load_self_symbols() {

  FILE *f;
  void *v1,*v,*ve;
  struct filehdr *fhp;
  struct syment *sy1,*sye,*sym;
  struct scnhdr *sec1,*sec,*sece;
  struct opthdr *h;
  struct node *a;
  char *st1,*st;
  ul ns,sl;
  unsigned long jj;

  massert(f=fopen(kcl_self,"r"));
  massert(v1=get_mmap(f,&ve));

  v=v1+*(ul *)(v1+0x3c);
  massert(!memcmp("PE\0\0",v,4));

  fhp=v+4;
  h=(void *)(fhp+1);
  massert(h->h_magic==0x10b || h->h_magic==0x20b);
  self_ibase=h->h_ibase;
#if SIZEOF_LONG == 8
  if (h->h_magic==0x20b)
    self_ibase=(self_ibase<<32)+h->h_dbase;
#endif

  sec1=(void *)(fhp+1)+fhp->f_opthdr;
  sece=sec1+fhp->f_nscns;

  sy1=v1+fhp->f_ptrsym;
  sye=sy1+fhp->f_symnum;

  st1=(char *)sye;

  for (ns=sl=0,sym=sy1;sym<sye;sym++) {

    if (sym->n_sclass<2 || sym->n_sclass>3 || sym->n_scnum<1)
      continue;
    
    ns++;

    NM(sym,st1,s,sl+=strlen(s)+1);
  
    sym+=sym->n_numaux;

  }

  c_table.alloc_length=ns;
  assert(c_table.ptable=malloc(sizeof(*c_table.ptable)*c_table.alloc_length));
  assert(st=malloc(sl));

  for (a=c_table.ptable,sym=sy1;sym<sye;sym++) {

    if (sym->n_sclass!=2 || sym->n_scnum<1)
      continue;

    NM(sym,st1,s,strcpy(st,s));
    
    sec=sec1+sym->n_scnum-1;
    jj=self_ibase+sym->n_value+sec->s_vaddr;
    
#ifdef FIX_ADDRESS
    FIX_ADDRESS(jj);
#endif       
    
    a->address=jj;
    a->string=st;

    a++;
    st+=strlen(st)+1;
    sym+=sym->n_numaux;
    
  }
  c_table.length=a-c_table.ptable;
  qsort(c_table.ptable,c_table.length,sizeof(*c_table.ptable),node_compare);

  for (c_table.local_ptable=a,sym=sy1;sym<sye;sym++) {

    if (sym->n_sclass!=3 || sym->n_scnum<1)
      continue;

    NM(sym,st1,s,strcpy(st,s));

    sec=sec1+sym->n_scnum-1;
    jj=self_ibase+sym->n_value+sec->s_vaddr;

#ifdef FIX_ADDRESS
    FIX_ADDRESS(jj);
#endif

    a->address=jj;
    a->string=st;

    a++;
    st+=strlen(st)+1;
    sym+=sym->n_numaux;

  }
  c_table.local_length=a-c_table.local_ptable;
  qsort(c_table.local_ptable,c_table.local_length,sizeof(*c_table.local_ptable),node_compare);

  massert(c_table.alloc_length==c_table.length+c_table.local_length);

  massert(!un_mmap(v1,ve));
  massert(!fclose(f));

  return 0;

}

int
seek_to_end_ofile(FILE *fp) {

  void *st,*ve;
  struct filehdr *fhp;
  struct scnhdr *sec1,*sece;
  struct syment *sy1,*sye;
  const char *st1,*ste;
  int i;

  massert(st=get_mmap(fp,&ve));

  fhp=st;
  sec1=(void *)(fhp+1)+fhp->f_opthdr;
  sece=sec1+fhp->f_nscns;
  sy1=st+fhp->f_ptrsym;
  sye=sy1+fhp->f_symnum;
  st1=(void *)sye;
  ste=st1+*(ul *)st1;

  fseek(fp,(void *)ste-st,0);
  while (!(i=getc(fp)));
  ungetc(i, fp);

  massert(!un_mmap(st,ve));

  return 0;

}

object
find_init_string(const char *s) {

  FILE *f;
  struct filehdr *fhp;
  struct scnhdr *sec1,*sece;
  struct syment *sy1,*sym,*sye;
  char *st1,*ste;
  void *st,*est;
  object o=OBJNULL;

  massert(f=fopen(s,"r"));
  massert(st=get_mmap(f,&est));

  fhp=st;
  sec1=(void *)(fhp+1)+fhp->f_opthdr;
  sece=sec1+fhp->f_nscns;
  sy1=st+fhp->f_ptrsym;
  sye=sy1+fhp->f_symnum;
  st1=(void *)sye;
  ste=st1+*(ul *)st1;

  for (sym=sy1;sym<sye;sym++) {

    NM(sym,st1,s,if (!strncmp(s,"_init_",6)) o=make_simple_string(s));

    if (o!=OBJNULL) {
      massert(!un_mmap(st,&est));
      massert(!fclose(f));
      return o;
    }

  }

  massert(!un_mmap(st,&est));
  massert(!fclose(f));
  massert(!"init not found");

  return NULL;

}  

int
fasload(object faslfile) {

  struct filehdr *fhp;
  struct scnhdr *sec1,*sec,*sece;
  struct syment *sy1,*sye;
  struct reloc *rel,*rele;
  object memory;
  FILE *fp;
  char *st1,*ste;
  int i;
  ul init_address=0;
  void *st,*est;

  fp = faslfile->sm.sm_fp;

  massert(st=get_mmap(fp,&est));

  fhp=st;
  sec1=(void *)(fhp+1)+fhp->f_opthdr;
  sece=sec1+fhp->f_nscns;
  sy1=st+fhp->f_ptrsym;
  sye=sy1+fhp->f_symnum;
  st1=(void *)sye;
  ste=st1+*(ul *)st1;

  find_init_address(sy1,sye,&init_address,st1);
	
  memory=load_memory(sec1,sece,st,&init_address);

  relocate_symbols(sy1,sye,sec1,st1);  
	
  for (sec=sec1;sec<sece;sec++)
    if (sec->s_flags&0xe0)
      for (rel=st+sec->s_relptr,rele=rel+(sec->s_flags&0x1000000 ? rel->r.r_count : sec->s_nreloc);rel<rele;rel++)
	relocate(sec,rel,sy1+rel->r_symndx,memory->cfd.cfd_start);
  
  fseek(fp,(void *)ste-st,0);
  while ((i = getc(fp)) == 0);
  ungetc(i, fp);

  massert(!un_mmap(st,est));

#ifdef CLEAR_CACHE
  CLEAR_CACHE;
#endif

  if(symbol_value(sLAload_verboseA)!=Cnil) {
    printf("start address -T %p ", memory->cfd.cfd_start);
    fflush(stdout);
  }

  call_init(init_address,memory,faslfile);

  return(memory->cfd.cfd_size);

}

#include "sfasli.c"
