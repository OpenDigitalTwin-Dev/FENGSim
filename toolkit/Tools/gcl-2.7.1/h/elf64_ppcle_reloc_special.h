static Sym *toc;

static int tramp[]={0,0,
		    (((0x3a<<10)|(0x9<<5)|0xc)<<16)|0xfff8,/*ld      r9,-8(r12)*/
		    ((0x3a<<10)|(0x9<<5)|0x9)<<16,         /*ld      r9,0(r9)*/
		    0x7d2c4b78,                            /*mr      r12,r9 */
		    0x7d8903a6,                            /*mtctr   r12*/
		    0x4e800420                             /*bctrl*/
};

static int
load_trampolines(void *v,Shdr *sec,Sym *ds1) {

  Rela *r;
  void *ve;
  ul *u,j;

  v+=sec->sh_offset;
  ve=v+sec->sh_size;

  for (j=0,r=v;v<ve;v+=sec->sh_entsize,r=v)
    if (ELF_R_TYPE(r->r_info) && !ds1[ELF_R_SYM(r->r_info)].st_value)
      j++;

  massert(u=malloc(j*sizeof(tramp)));

  v=ve-sec->sh_size;
  for (r=v;v<ve;v+=sec->sh_entsize,r=v)
    if (ELF_R_TYPE(r->r_info) && !ds1[ELF_R_SYM(r->r_info)].st_value) {
      memcpy(u,tramp,sizeof(tramp));
      *u++=r->r_offset;
      ds1[ELF_R_SYM(r->r_info)].st_value=(ul)u;
      u=((void *)(u-1)+sizeof(tramp));
    }

  return 0;

}

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym,Sym *syme) {

  Shdr *sec;

  massert((sec=get_section(".rela.dyn",sec1,sece,sn)));
  massert(!load_trampolines(v,sec,ds1));
  if ((sec=get_section(".rela.plt",sec1,sece,sn)))
    massert(!load_trampolines(v,sec,ds1));

  return 0;

}

static int
label_got_symbols(void *v1,Shdr *sec1,Shdr *sece,Sym *sym1,Sym *syme,const char *st1,const char *sn,ul *gs) {

  Rela *r;
  void *v,*ve;
  Shdr *sec;
  Sym *sym;
  
  for (toc=NULL,sym=sym1;sym<syme;sym++) {
    const char *s=st1+sym->st_name;
    if (!strcmp(s,".TOC.") || !strcmp(s,".toc.")) {
      toc=sym;
      toc->st_info=ELF_ST_INFO(STB_LOCAL,ELF_ST_TYPE(sym->st_info));
      massert((sec=get_section(".bss",sec1,sece,sn)));
      toc->st_shndx=sec-sec1;
    }
  }

  for (sym=sym1;sym<syme;sym++)
   sym->st_size=0;

  for (*gs=0,sec=sec1;sec<sece;sec++)
    if (sec->sh_type==SHT_RELA)
      for (v=v1+sec->sh_offset,ve=v+sec->sh_size,r=v;v<ve;v+=sec->sh_entsize,r=v)
	if (ELF_R_TYPE(r->r_info)==R_PPC64_PLT16_HA||
	    ELF_R_TYPE(r->r_info)==R_PPC64_PLT16_LO_DS) {

	  sym=sym1+ELF_R_SYM(r->r_info);

	  if (!sym->st_size)
	    sym->st_size=++*gs;

	}

  return 0;
  
}
