static ul pltgot;

#define ASM21(x) ((x>>20)|(((x>>9)&0x7ff)<<1)|(((x>>7)&0x3)<<14)|(((x>>2)&0x1f)<<16)|(((x>>0)&0x3)<<12))
/* be,l off(sr4,r19),sr0,r31 ; linux userspace sr4-7 const, sr0-3 used by kernel */
#define ASM17(x) ((x>>16)|(((x>>11)&0x1f)<<16)|((x&0x3ff)<<3)|(((x>>10)&0x1)<<2)|(1<<13))

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym,Sym *syme) {
  
  Rela *r;
  Shdr *sec;
  ul *q;
  void *p,*pe;

  massert(sec=get_section(".dynamic",sec1,sece,sn));
  for (p=(void *)sec->sh_addr,pe=p+sec->sh_size;p<pe;p+=sec->sh_entsize) {
    q=p;
    if (q[0]==DT_PLTGOT)
      pltgot=q[1];
    
  }
  massert(pltgot);

  massert(sec=get_section(".rela.plt",sec1,sece,sn));
  p=v+sec->sh_offset;
  pe=p+sec->sh_size;
  for (r=p;p<pe;p+=sec->sh_entsize,r=p) 
    if (!ds1[ELF_R_SYM(r->r_info)].st_value)
      ds1[ELF_R_SYM(r->r_info)].st_value=r->r_offset|0x2;

  return 0;

}

static int
label_got_symbols(void *v1,Shdr *sec1,Shdr *sece,Sym *sym1,Sym *syme,const char *st1,const char *sn,ul *gs) {

  return 0;
  
}

#undef LOAD_SYM_BY_NAME
#define LOAD_SYM_BY_NAME(sym,st1) (!strncmp(st1+sym->st_name,"$$",2))
