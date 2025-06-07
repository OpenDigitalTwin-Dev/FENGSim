/* #define R_AARCH64_TRAMP 1 */
static int tramp[]={0x58ffffd0, /*ldr 19bit pc relative x16*/
		    0xd61f0200};/*br x16*/
static ul gotp,tz=1+sizeof(tramp)/sizeof(ul);


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

  gotp=0;
  for (sym=sym1;sym<syme;sym++)
    sym->st_size=0;

  for (*gs=0,sec=sec1;sec<sece;sec++)
    if (sec->sh_type==SHT_RELA)
      for (v=v1+sec->sh_offset,ve=v+sec->sh_size,r=v;v<ve;v+=sec->sh_entsize,r=v)
	if (ELF_R_TYPE(r->r_info)==R_AARCH64_JUMP26 ||
	    ELF_R_TYPE(r->r_info)==R_AARCH64_CALL26) {

	  if (r->r_addend)

	    (*gs)+=tz;

	  else {
	  
	    sym=sym1+ELF_R_SYM(r->r_info);
	    
	    if (!sym->st_size) 
	      sym->st_size=++gotp;

	  }

	}

  gotp*=tz;
  (*gs)+=gotp;
  
  return 0;
  
}
