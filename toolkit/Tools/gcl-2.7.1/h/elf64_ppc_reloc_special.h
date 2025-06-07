static ul toc;

static int tramp[]={0,0,0,0,0,0,0,0,
		    ((0x3a<<10)|(0x9<<5)|0x2)<<16,
		    ((0x3a<<10)|(0x9<<5)|0x9)<<16,
		    ((0x3a<<10)|(0xa<<5)|0x9)<<16,
		    (((0x3a<<10)|(0xb<<5)|0x9)<<16)|0x10,
		    0x7d4903a6,
		    (((0x3a<<10)|(0x2<<5)|0x9)<<16)|0x8,
		    0x4e800420,0};

/* static int */
/* make_trampoline(void *v,ul addr) { */

/*   ul *u; */
/*   int *i; */

/*   u=v; */
/*   *u++=(ul)(v+4*sizeof(*u)); */
/*   *u++=(ul)(v+3*sizeof(*u)); */
/*   *u++=0; */
/*   *u++=addr; */
/*   i=(void *)u; */
/*   *i++=((0x3a<<10)|(0x9<<5)|0x2)<<16; */
/*   *i++=((0x3a<<10)|(0x9<<5)|0x9)<<16; */
/*   *i++=((0x3a<<10)|(0xa<<5)|0x9)<<16; */
/*   *i++=(((0x3a<<10)|(0xb<<5)|0x9)<<16)|0x10; */
/*   *i++=0x7d4903a6; */
/*   *i++=(((0x3a<<10)|(0x2<<5)|0x9)<<16)|0x8; */
/*   *i++=0x4e800420; */

/*   return 0; */

/* } */

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym,Sym *syme) {
  
  Shdr *sec;
  Rela *r;
  void *ve,*u;
  ul j;


  massert(sec=get_section(".got",sec1,sece,sn));
  toc=sec->sh_addr;

  init_section_name=".opd";

  massert((sec=get_section(".rel.dyn",sec1,sece,sn))||
	  (sec=get_section(".rela.dyn",sec1,sece,sn)));

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
      ((ul *)u)[0]=(ul)(((ul *)u)+4);
      ((ul *)u)[1]=(ul)(((ul *)u)+3);
      ((ul *)u)[3]=r->r_offset;
      ds1[ELF_R_SYM(r->r_info)].st_value=(ul)u;
      u+=sizeof(tramp);
    }

  return 0;

}

static int
label_got_symbols(void *v1,Shdr *sec1,Shdr *sece,Sym *sym1,Sym *syme,const char *st1,const char *sn,ul *gs) {

  Shdr *sec;

  massert(sec=get_section(".toc",sec1,sece,sn));
  toc=sec->sh_addr;

  return 0;
  
}
