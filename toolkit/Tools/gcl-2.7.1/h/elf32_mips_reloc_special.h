#include <page.h>

static ul gpd,ggot,ggote,can_gp; static Rel *hr;

typedef struct {
  ul addr_hi,addr_lo,jr,nop;
} mips_26_tramp;

static int
write_26_stub(ul s,ul *got,ul *gote) {

  static mips_26_tramp t1={(0xf<<26)|(0x0<<21)|(0x19<<16),   /*lui t9*/
			   (0xe<<26)|(0x19<<21)|(0x19<<16),  /*ori t9,t9 */
			   0x03200008,                       /*jr t9*/
			   0x00200825};                      /*mv at,at */;
  mips_26_tramp *t=(void *)gote;

  *t=t1;
  t->addr_hi|=s>>16;
  t->addr_lo|=s&0xffff;

  return 0;

}

typedef struct {
  ul entry,addr_hi,addr_lo,lw,jr,lwcan;
} call_16_tramp;

static int
write_stub(ul s,ul *got,ul *gote) {

  static call_16_tramp t1={0,
			   (0xf<<26)|(0x0<<21)|(0x19<<16),   /*lui t9*/
			   (0xe<<26)|(0x19<<21)|(0x19<<16),  /*ori t9,t9 */
			   (0x23<<26)|(0x19<<21)|(0x19<<16), /*lw t9,(0)t9*/
			   0x03200008,                       /*jr t9*/
                           /*stub addresses need veneer setting gp to canonical*/
			   (0x23<<26)|(0x1c<<21)|(0x1c<<16)};/*lw gp,(0)gp*/
  call_16_tramp *t=(void *)gote++;

  *t=t1;
  *got=can_gp;

  t->entry=(ul)gote;
  t->addr_hi|=s>>16;
  t->addr_lo|=s&0xffff;

  return 0;

}

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym,Sym *syme) {
  
  Shdr *sec;
  ul *q,gotsym=0,locgotno=0,stub,stube;
  void *p,*pe;

  massert(sec=get_section(".dynamic",sec1,sece,sn));
  for (p=(void *)sec->sh_addr,pe=p+sec->sh_size;p<pe;p+=sec->sh_entsize) {
    q=p;
    if (q[0]==DT_MIPS_GOTSYM)
      gotsym=q[1];
    if (q[0]==DT_MIPS_LOCAL_GOTNO)
      locgotno=q[1];
    if (q[0]==DT_PLTGOT)
      can_gp=q[1]+0x7ff0;

  }
  massert(gotsym && locgotno && can_gp);

  massert(sec=get_section(".MIPS.stubs",sec1,sece,sn));
  stub=sec->sh_addr;
  stube=sec->sh_addr+sec->sh_size;

  massert(sec=get_section(".got",sec1,sece,sn));
  ggot=sec->sh_addr+locgotno*sec->sh_entsize;
  ggote=sec->sh_addr+sec->sh_size;

  for (ds1+=gotsym,sym=ds1;sym<dse;sym++)
    if (!sym->st_value || (sym->st_value>=stub && sym->st_value<stube))
      sym->st_value=ggot+(sym-ds1)*sec->sh_entsize;

  return 0;

}

static int
label_got_symbols(void *v1,Shdr *sec1,Shdr *sece,Sym *sym1,Sym *syme,const char *st1,const char *sn,ul *gs) {

  Rel *r;
  Sym *sym;
  Shdr *sec,*ssec;
  void *v,*ve;
  ul q;
  struct node *a;

  for (q=0,sym=sym1;sym<syme;sym++) {
    const char *s=st1+sym->st_name;
    if ((sym->st_other=strcmp(s,"_gp_disp") ? (strcmp(s,"__gnu_local_gp") ? 0 : 2) : 1)) {
      q++;
      sym->st_info=ELF_ST_INFO(STB_LOCAL,ELF_ST_TYPE(sym->st_info));
    }
  }
  massert(q<=1);
  
  for (sym=sym1;sym<syme;sym++)
    sym->st_size=0;

  for (*gs=1,sec=sec1;sec<sece;sec++)/*can_gp in got[0]*/
    if (sec->sh_type==SHT_REL)/*no addend*/
      for (v=v1+sec->sh_offset,ve=v+sec->sh_size,r=v;v<ve;v+=sec->sh_entsize,r=v)

	if (!(sym=sym1+ELF_R_SYM(r->r_info))->st_size)

	  switch(ELF_R_TYPE(r->r_info)) {

	  case R_MIPS_26:
	    if (((ul)(pagetochar(page(heap_end))+r->r_offset))>>28) {
	      sym->st_size=++*gs;
	      (*gs)+=sizeof(mips_26_tramp)/sizeof(ul)-1;
	    }
	    break;
	  case R_MIPS_CALL16:
	    sym->st_size=++*gs;
	    if (((ssec=sec1+sym->st_shndx)>=sece || !ALLOC_SEC(ssec)) &&
		(a=find_sym_ptable(st1+sym->st_name)) &&
		a->address>=ggot && a->address<ggote)
	      (*gs)+=sizeof(call_16_tramp)/sizeof(ul)-1;
	    break;
	  case R_MIPS_GOT16:
	    sym->st_size=++*gs;
	    break;
	  }

  return 0;
  
}

#define FIX_HIDDEN_SYMBOLS(st1_,a_,sym1_,sym_,syme_)				\
  ({Sym *p;const char *n=(st1_)+(sym_)->st_name,*s=".pic.",*q;ul z=strlen(s);	\
    if (ELF_ST_VISIBILITY((sym_)->st_other)==STV_HIDDEN) {		\
      for (p=(sym1_);p<(syme_);p++)					\
	if (!strncmp(s,(q=(st1_)+p->st_name),z) && !strcmp(n,q+z)) {	\
	  (*(a_))->address=p->st_value;					\
	  break;							\
	}}})

#undef LOAD_SYM_BY_NAME
#define LOAD_SYM_BY_NAME(sym,st1) (!strncmp(st1+sym->st_name,"__moddi3",8))
