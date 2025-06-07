#define R_LARCH_B16 64
#define R_LARCH_B21 65
#define R_LARCH_B26 66
#define R_LARCH_PCALA_HI20 71
#define R_LARCH_PCALA_LO12 72
#define R_LARCH_GOT_PC_HI20 75
#define R_LARCH_GOT_PC_LO12 76
#define R_LARCH_32_PCREL 99
#define R_LARCH_RELAX 100
#define R_LARCH_ALIGN 102
#define R_LARCH_ADD6 105
#define R_LARCH_SUB6 106

static unsigned int tramp[] = {
				0x1a00000c, /* pcalau12i $t0, %hi(sym) */
				0x4c000180 /* jirl $zero, $t0, %lo(sym) */};

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym,Sym *syme) {
  return 0;

}

static int
label_got_symbols(void *v1,Shdr *sec1,Shdr *sece,Sym *sym1,Sym *syme,const char *st1,const char *sn,ul *gs) {
  Rela *r;
  Sym *sym;
  Shdr *sec;
  void *v,*ve;
  int idx;
  const int gz = sizeof(ul)/sizeof(ul), tz = sizeof(tramp)/sizeof(ul);
  massert(gz==1);
  massert(tz==1);

  for (sym=sym1;sym<syme;sym++)
    sym->st_size=0;

  /* Count the symbols need to be fixed first. */
  for (sec=sec1;sec<sece;sec++)
    if (sec->sh_type==SHT_RELA)
      for (v=v1+sec->sh_offset,ve=v+sec->sh_size,r=v;v<ve;v+=sec->sh_entsize,r=v)
	if (
	    ELF_R_TYPE(r->r_info)==R_LARCH_GOT_PC_HI20 ||
	    ELF_R_TYPE(r->r_info)==R_LARCH_B26
	    ) {
	  sym=sym1+ELF_R_SYM(r->r_info);
	  if (ELF_R_TYPE(r->r_info)==R_LARCH_B26 && LOCAL_SYM(sym))
	    continue;

	  if (ELF_R_TYPE(r->r_info)==R_LARCH_GOT_PC_HI20)
	    sym->st_size|=0x1;
	  if (ELF_R_TYPE(r->r_info)==R_LARCH_B26)
	    sym->st_size|=0x2;
	}

  for (idx=0,sym=sym1;sym<syme;sym++) {
    if (sym->st_size==0)
      continue;
    massert(!(sym->st_size>>2));
    sym->st_size|=idx<<2;
    if (sym->st_size&0x1)
      idx+=gz;
    if (sym->st_size&0x2)
      idx+=tz;
  }

  *gs=idx;
  return 0;
}
