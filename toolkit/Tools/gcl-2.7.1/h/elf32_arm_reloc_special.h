static int tramp[]={0xe59fc000,   /*ldr r12, [pc]*/ /*FIXME?  Can this refer to an earlier address?*/
		    0xe12fff1c};  /*br r12*/
static ul tz=1+sizeof(tramp)/sizeof(ul);

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym,Sym *syme) {

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
	    ELF_R_TYPE(r->r_info)==R_ARM_CALL ||
	    ELF_R_TYPE(r->r_info)==R_ARM_JUMP24
	    ) {

	  sym=sym1+ELF_R_SYM(r->r_info);

	  if (!sym->st_size)
	    sym->st_size=++*gs;

	}

  (*gs)*=tz;

  return 0;

}
