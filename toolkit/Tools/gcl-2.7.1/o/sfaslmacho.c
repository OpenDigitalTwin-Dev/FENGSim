/* Copyright (C) 2024 Camm Maguire */
#include <alloca.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <mach/mach.h>
#include <mach-o/loader.h>
#include <mach-o/reloc.h>
#include <mach-o/nlist.h>
#include <mach-o/getsect.h>

#ifdef _LP64
#define mach_header			mach_header_64
#define nlist    			nlist_64
#define segment_command			segment_command_64
#undef  LC_SEGMENT
#define LC_SEGMENT			LC_SEGMENT_64
#define section				section_64
#undef MH_MAGIC
#define MH_MAGIC			MH_MAGIC_64
#endif

#ifndef S_16BYTE_LITERALS
#define S_16BYTE_LITERALS 0
#endif

#define ALLOC_SEC(sec) ({ul _fl=sec->flags&SECTION_TYPE;\
      _fl<=S_SYMBOL_STUBS || _fl==S_16BYTE_LITERALS;})

#define LOAD_SEC(sec) ({ul _fl=sec->flags&SECTION_TYPE;\
      (_fl<=S_SYMBOL_STUBS || _fl==S_16BYTE_LITERALS) && _fl!=S_ZEROFILL;})


#define MASK(n) (~(~0ULL << (n)))



typedef unsigned long ul;



#ifdef STATIC_RELOC_VARS
STATIC_RELOC_VARS
#endif



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

#ifndef _LP64
/*redirect trampolines gcc-4.0 gives no reloc for stub sections on x86 only*/
static int
redirect_trampoline(struct relocation_info *ri,ul o,ul rel,
		    struct section *sec1,ul *io1,struct nlist *n1,ul *a) {

  struct section *js=sec1+ri->r_symbolnum-1;

  if (ri->r_extern)
    return 0;

  if ((js->flags&SECTION_TYPE)!=S_SYMBOL_STUBS)
    return 0;

  if (ri->r_pcrel) o+=rel;
  o-=js->addr;
  
  massert(!(o%js->reserved2));
  o/=js->reserved2;
  massert(o>=0 && o<js->size/js->reserved2);
  
  *a=n1[io1[js->reserved1+o]].n_value;
  ri->r_extern=1;
  
  return 0;
  
}
#endif

static int
relocate(struct relocation_info *ri,struct section *sec,
	 struct section *sec1,ul start,ul *io1,struct nlist *n1,ul *got,ul *gote) {

  struct scattered_relocation_info *sri=(void *)ri;
  ul *q=(void *)(sec->addr+(sri->r_scattered ? sri->r_address : ri->r_address));
  ul a,rel=(ul)(q+1);

  if (sri->r_scattered)
    a=sri->r_value;
  else if (ri->r_extern)
    a=n1[ri->r_symbolnum].n_value;
  else
    a=start;

  switch(sri->r_scattered ? sri->r_type : ri->r_type) {
    
#include RELOC_H

  default:
    FEerror("Unknown reloc type\n",0);
    break;
    
  }

  return 0;
  
}  

static int
relocate_symbols(struct nlist *n1,struct nlist *ne,char *st1,ul start) {

  struct nlist *n;
  struct node *nd;

  for (n=n1;n<ne;n++)
    
    if (n->n_sect) 
      n->n_value+=start; 
    else if ((nd=find_sym_ptable(st1+n->n_un.n_strx)))
      n->n_value=nd->address; 
    else if (n->n_type&(N_PEXT|N_EXT))
      massert(!emsg("Unrelocated non-local symbol: %s\n",st1+n->n_un.n_strx));

  return 0;
  
}

static int
find_init_address(struct nlist *n1,struct nlist *ne,const char *st1,ul *init) {

  struct nlist *n;

  for (n=n1;n<ne && strncmp("_init",st1+n->n_un.n_strx,5);n++);
  massert(n<ne);

  *init=n->n_value;

  return 0;

}



static object
load_memory(struct section *sec1,struct section *sece,void *v1,
	    ul *p,ul **got,ul **gote,ul *start) { 

  ul sz,gsz,sa,ma,a,fl;
  struct section *sec;
  object memory;
  
  BEGIN_NO_INTERRUPT;

  for (*p=sz=ma=0,sa=-1,sec=sec1;sec<sece;sec++)
    
    if (ALLOC_SEC(sec)) {
    
      if (sec->addr<sa) {
	sa=sec->addr;
	ma=1<<sec->align;
      }

      a=sec->addr+sec->size;
      if (sz<a) sz=a;

      fl=sec->flags&SECTION_TYPE;
      if (fl==S_NON_LAZY_SYMBOL_POINTERS || fl==S_LAZY_SYMBOL_POINTERS)
	*p+=sec->size*sizeof(struct relocation_info)/sizeof(void *);
      
    }
  
  ma=ma>sizeof(struct contblock) ? ma-1 : 0; 
  sz+=ma;

  gsz=0;
  if (**got) {
    gsz=(**got+1)*sizeof(**got)-1;
    sz+=gsz;
  }
  
  memory=new_cfdata();
  memory->cfd.cfd_size=sz; 
  memory->cfd.cfd_start=alloc_code_space(sz,-1UL);

  a=(ul)memory->cfd.cfd_start;
  a=(a+ma)&~ma;
  for (sec=sec1;sec<sece;sec++)
    if (ALLOC_SEC(sec)) {
      sec->addr+=a;  
      if (LOAD_SEC(sec))
	memcpy((void *)sec->addr,v1+sec->offset,sec->size);
      else
	bzero((void *)sec->sh_addr,sec->sh_size);
    }

  if (**got) {
    sz=**got;
    *got=(void *)memory->cfd.cfd_start+memory->cfd.cfd_size-gsz;
    gsz=sizeof(**got)-1;
    *got=(void *)(((ul)*got+gsz)&~gsz);
    *gote=*got+sz;
  }

  *start=a;
  
  END_NO_INTERRUPT;

  return memory;

} 


static int
parse_file(void *v1,
	   struct section **sec1,struct section **sece,
	   struct nlist **n1,struct nlist **ne,
	   char **st1,char **ste,ul **io1) {
 
  struct mach_header *mh;
  struct load_command *lc;
  struct symtab_command *sym=NULL;
  struct dysymtab_command *dsym=NULL;
  struct segment_command *seg;
  ul i;
  void *v=v1;

  mh=v;
  v+=sizeof(*mh);
  
  for (i=0,*sec1=NULL;(lc=v) && i<mh->ncmds;i++,v+=lc->cmdsize)
    
    switch(lc->cmd) {
      
    case LC_SEGMENT:
      
      if (*sec1 && *sece>*sec1) continue;

      seg=v;
      *sec1=(void *)(seg+1);
      *sece=*sec1+seg->nsects;

      break;
    case LC_SYMTAB:
      massert(!sym);
      sym=v;
      *n1=v1+sym->symoff;
      *ne=*n1+sym->nsyms;
      *st1=v1+sym->stroff;
      *ste=*st1+sym->strsize;
      break;
    case LC_DYSYMTAB:
      massert(!dsym);
      dsym=v;
      *io1=v1+dsym->indirectsymoff;
      break;
    }
  
  return 0;

}


static int
set_symbol_stubs(void *v1,struct nlist *n1,struct nlist *ne,ul *uio,const char *st1) {
 
  struct mach_header *mh;
  struct load_command *lc;
  struct segment_command *seg;
  struct section *sec1,*sec,*sece;
  ul i,ns;
  void *v=v1,*vv;
  int *io1,*io,*ioe;

  mh=v;
  v+=sizeof(*mh);
  
  for (i=0;(lc=v) && i<mh->ncmds;i++,v+=lc->cmdsize)
    
    switch(lc->cmd) {
      
    case LC_SEGMENT:
      
      for (seg=v,sec1=sec=(void *)(seg+1),sece=sec1+seg->nsects;sec<sece;sec++) {
	
	ns=sec->flags&SECTION_TYPE;
	if (ns!=S_SYMBOL_STUBS && 
	    ns!=S_LAZY_SYMBOL_POINTERS &&
	    ns!=S_NON_LAZY_SYMBOL_POINTERS)
	  continue;
	
	io1=(void *)uio;
	io1+=sec->reserved1;
	if (!sec->reserved2) sec->reserved2=sizeof(void *);
	ioe=io1+sec->size/sec->reserved2;

	for (io=io1,vv=(void *)sec->addr;io<ioe;vv+=sec->reserved2,io++)
	  if (*io>=0 && *io<ne-n1)
	    if (!n1[*io].n_value)
	      n1[*io].n_value=(ul)vv;

      }
      
    }
  
  return 0;
  
}


static int
maybe_gen_fake_relocs(void *v1,struct section *sec,void **p,ul *io1) {

  ul fl=sec->flags&SECTION_TYPE,*io;
  struct relocation_info *ri,*re;
  struct scattered_relocation_info *sri;

  if (fl!=S_NON_LAZY_SYMBOL_POINTERS && fl!=S_LAZY_SYMBOL_POINTERS)
    return 0;

  sec->nreloc=sec->size/sizeof(void *);
  sec->reloff=*p-v1;
  ri=*p;
  re=ri+sec->nreloc;
  *p=re;

  io1+=sec->reserved1;
  for (io=io1;ri<re;ri++,io++) {
    
    ri->r_symbolnum=*io;
    ri->r_extern=1;
    ri->r_address=(io-io1)*sizeof(void *);
    ri->r_type=GENERIC_RELOC_VANILLA;
    ri->r_pcrel=0;
    sri=(void *)ri;
    sri->r_scattered=0;
    
  }
  
  return 0;

}


static int
relocate_code(void *v1,struct section *sec1,struct section *sece,
	      void **p,ul *io1,struct nlist *n1,ul *got,ul *gote,ul start) {

  struct section *sec;
  struct relocation_info *ri,*re;

  for (sec=sec1;sec<sece;sec++) {
    
    if (!LOAD_SEC(sec))
      continue;

    maybe_gen_fake_relocs(v1,sec,p,io1);

    for (ri=v1+sec->reloff,re=ri+sec->nreloc;ri<re;ri++)
      relocate(ri,sec,sec1,start,io1,n1,got,gote);
  
  }

  return 0;

}

static int 
load_self_symbols() {

  struct section *sec1=NULL,*sece=NULL;
  struct nlist *sym1=NULL,*sym,*syme=NULL;
  struct node *a;
  ul ns,sl,*uio=NULL;
  char *strtab=NULL,*ste,*s;
  void *addr,*addre;
  FILE *f;
  
  massert(f=fopen(kcl_self,"r"));
  massert(addr=get_mmap(f,&addre));

  parse_file(addr,&sec1,&sece,&sym1,&syme,&strtab,&ste,&uio);

  set_symbol_stubs(addr,sym1,syme,uio,strtab);

  for (ns=sl=0,sym=sym1;sym<syme;sym++) {
    
    if (sym->n_type & N_STAB)
      continue;

    ns++;
    sl+=strlen(sym->n_un.n_strx+strtab)+1;

  }
  
  c_table.alloc_length=ns;
  assert(c_table.ptable=malloc(sizeof(*c_table.ptable)*c_table.alloc_length));
  assert(s=malloc(sl));

  for (a=c_table.ptable,sym=sym1;sym<syme;sym++) {
    
    if ((sym->n_type & N_STAB) || !(sym->n_type & N_EXT))
      continue;

    a->address=sym->n_value;
    a->string=s;
    strcpy(s,sym->n_un.n_strx+strtab);

    a++;
    s+=strlen(s)+1;

  }
  c_table.length=a-c_table.ptable;
  qsort(c_table.ptable,c_table.length,sizeof(*c_table.ptable),node_compare);

  for (c_table.local_ptable=a,sym=sym1;sym<syme;sym++) {

    if ((sym->n_type & N_STAB) || sym->n_type & N_EXT)
      continue;

    a->address=sym->n_value;
    a->string=s;
    strcpy(s,sym->n_un.n_strx+strtab);

    a++;
    s+=strlen(s)+1;

  }
  c_table.local_length=a-c_table.local_ptable;
  qsort(c_table.local_ptable,c_table.local_length,sizeof(*c_table.local_ptable),node_compare);

  massert(c_table.alloc_length==c_table.length+c_table.local_length);

  massert(!un_mmap(addr,addre));
  massert(!fclose(f));

  return 0;

}

int 
seek_to_end_ofile(FILE *f) {

  struct mach_header *mh;
  struct load_command *lc;
  struct symtab_command *st=NULL;
  void *addr,*addre;
  int i;
  
  massert(addr=get_mmap(f,&addre));

  mh=addr;
  lc=addr+sizeof(*mh);
  
  for (i=0;i<mh->ncmds;i++,lc=(void *)lc+lc->cmdsize)
    if (lc->cmd==LC_SYMTAB) {
      st=(void *) lc;
      break;
    }
  massert(st);

  fseek(f,st->stroff+st->strsize,SEEK_SET);

  massert(!un_mmap(addr,addre));

  return 0;

}

#ifndef GOT_RELOC
#define GOT_RELOC(a) 0
#endif

static int
label_got_symbols(void *v1,struct section *sec,struct nlist *n1,struct nlist *ne,ul *gs) {

  struct relocation_info *ri,*re;
  struct nlist *n;

  *gs=0;
  for (n=n1;n<ne;n++)
    n->n_desc=0;

  for (ri=v1+sec->reloff,re=ri+sec->nreloc;ri<re;ri++)

    if (GOT_RELOC(ri)) {
    
      massert(ri->r_extern);
      n=n1+ri->r_symbolnum;
      if (!n->n_desc)
	n->n_desc=++*gs;

    }

  return 0;
  
}

static int
clear_protect_memory(object memory) {

  void *p,*pe;

  p=(void *)((unsigned long)memory->cfd.cfd_start & ~(PAGESIZE-1));
  pe=(void *)((unsigned long)(memory->cfd.cfd_start+memory->cfd.cfd_size + PAGESIZE-1) & ~(PAGESIZE-1));

  return gcl_mprotect(p,pe-p,PROT_READ|PROT_WRITE|PROT_EXEC);

}


int
fasload(object faslfile) {

  FILE *fp;
  ul init_address=-1;
  object memory;
  void *v1,*ve,*p;
  struct section *sec1,*sece=NULL;
  struct nlist *n1=NULL,*ne=NULL;
  char *st1=NULL,*ste=NULL;
  ul gs,*got=&gs,*gote,*io1=NULL,rls,start;

  fp = faslfile->sm.sm_fp;

  massert(v1=get_mmap(fp,&ve));

  parse_file(v1,&sec1,&sece,&n1,&ne,&st1,&ste,&io1);

  label_got_symbols(v1,sec1,n1,ne,got);

  massert(memory=load_memory(sec1,sece,v1,&rls,&got,&gote,&start));
  memory->cfd.cfd_name=faslfile->sm.sm_object1;
  
  massert(p=alloca(rls));
  
  relocate_symbols(n1,ne,st1,start);

  find_init_address(n1,ne,st1,&init_address);

  relocate_code(v1,sec1,sece,&p,io1,n1,got,gote,start);

  fseek(fp,(void *)ste-v1,SEEK_SET);
  
  massert(!clear_protect_memory(memory));

#ifdef CLEAR_CACHE
  CLEAR_CACHE;
#endif
  
  massert(!un_mmap(v1,ve));
  
  if(symbol_value(sLAload_verboseA)!=Cnil) {
    printf(";; start address for %.*s %p\n",
	   (int)VLEN(memory->cfd.cfd_name),memory->cfd.cfd_name->st.st_self,
	   memory->cfd.cfd_start);
    fflush(stdout);
  }

  init_address-=(ul)memory->cfd.cfd_start;
  call_init(init_address,memory,faslfile);
  
  return(memory->cfd.cfd_size);

 }

#include "sfasli.c"
