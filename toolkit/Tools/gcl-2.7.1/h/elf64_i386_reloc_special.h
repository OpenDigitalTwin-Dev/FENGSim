static unsigned char tramp[]={
  0x48,0x89,0x40,0x08, /*mov    %rax,0x10(%rax) */
  0xff,0xe0,           /*jmp    *%rax*/
  0x90,                /*nop*/
  0x90,                /*nop*/
  0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0};

static ul tz=sizeof(tramp)/sizeof(ul);

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym,Sym *syme) {

  /* Shdr *sec,*psec; */
  /* Rel *r; */
  /* ul *p,*pe; */
  /* void *ve; */

  /* /\*plt entries are not of uniform size*\/ */

  /* massert(psec=get_section(".plt",sec1,sece,sn)); */
  /* p=(void *)psec->sh_addr; */
  /* pe=(void *)p+psec->sh_size; */

  /* massert((sec=get_section( ".rel.plt",sec1,sece,sn)) || */
  /* 	  (sec=get_section(".rela.plt",sec1,sece,sn))); */

  /* v+=sec->sh_offset; */
  /* ve=v+sec->sh_size; */

  /* p=next_plt_entry(p,pe);/\*plt0*\/ */

  /* for (r=v;v<ve && p<pe;v+=sec->sh_entsize,r=v,p=next_plt_entry(p,pe)) { */
  /*   if (!ds1[ELF_R_SYM(r->r_info)].st_value) */
  /*     ds1[ELF_R_SYM(r->r_info)].st_value=(ul)p; */
  /* } */

  /* massert(p==pe); */
  /* massert(v==ve); */

  return 0;

}

static int
label_got_symbols(void *v1,Shdr *sec1,Shdr *sece,Sym *sym1,Sym *syme,const char *st1,const char *sn,ul *gs) {

  Rel *r;
  Sym *sym;
  Shdr *sec;
  void *v,*ve;

  for (sym=sym1;sym<syme;sym++)
    sym->st_size=0;

  for (*gs=0,sec=sec1;sec<sece;sec++)
    if (sec->sh_type==SHT_REL)
      for (v=v1+sec->sh_offset,ve=v+sec->sh_size,r=v;v<ve;v+=sec->sh_entsize,r=v)
	if (
	    ELF_R_TYPE(r->r_info)==R_X86_64_PLT32 ||
	    ELF_R_TYPE(r->r_info)==R_X86_64_PC32 ||
	    ELF_R_TYPE(r->r_info)==R_X86_64_32
	    ) {

	  sym=sym1+ELF_R_SYM(r->r_info);

	  if (!sym->st_size) {
	    sym->st_size=++*gs;
	    if (ELF_R_TYPE(r->r_info)==R_X86_64_PLT32)
	      (*gs)+=sizeof(tramp)-1;
	  }
	    

	}

  (*gs)*=tz;

  return 0;

}
