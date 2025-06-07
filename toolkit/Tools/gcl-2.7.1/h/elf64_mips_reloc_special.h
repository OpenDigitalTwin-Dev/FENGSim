static ul ggot,ggote,la; static Rela *hr,*lr;

#undef ELF_R_SYM 
#define ELF_R_SYM(a_) (a_&0xffffffff) 
#define ELF_R_TYPE1(a_) ((a_>>56)&0xff)
#define ELF_R_TYPE2(a_) ((a_>>48)&0xff)
#define ELF_R_TYPE3(a_) ((a_>>40)&0xff)
#define recurse(val) ({							\
      if (ELF_R_TYPE2(r->r_info)) {					\
	ul i=r->r_info;							\
	r->r_info=(((r->r_info>>32)&MASK(24))<<40)|(r->r_info&MASK(32)); \
	relocate(sym1,r,(val)-s,start,got,gote);			\
	r->r_info=i;							\
	break;								\
      }})

#undef ELF_R_TYPE 
#define ELF_R_TYPE(a_) ELF_R_TYPE1(a_)
#define MIPS_HIGH(a_) ({ul _a=(a_);(_a-(short)_a)>>16;})

typedef struct {
  ul entry,gotoff;
  unsigned int ld_gotoff,lw,jr,lwcan;
} call_16_tramp;

static int
write_stub(ul s,ul *got,ul *gote) {

  static call_16_tramp t1={0,0,
			   (0x37<<26)|(0x1c<<21)|(0x19<<16), /*ld t9,(0)gp*/
			   (0x37<<26)|(0x19<<21)|(0x19<<16), /*ld t9,(0)t9*/
			   0x03200008,                       /*jr t9*/
			   0                                 /*nop*/
  };
  call_16_tramp *t=(void *)gote;

  *t=t1;

  t->entry=(ul)(gote+2);
  t->gotoff=s;
  t->ld_gotoff|=((void *)(gote+1)-(void *)got);

  return 0;

}

static int
make_got_room_for_stub(Shdr *sec1,Shdr *sece,Sym *sym,const char *st1,ul *gs) {

  Shdr *ssec=sec1+sym->st_shndx;
  struct node *a;
  if ((ssec>=sece || !ALLOC_SEC(ssec)) &&
      (a=find_sym_ptable(st1+sym->st_name)) &&
      a->address>=ggot && a->address<ggote)
    (*gs)+=sizeof(call_16_tramp)/sizeof(ul)-1;

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
  }
  massert(gotsym && locgotno);

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

  Rela *r;
  Sym *sym;
  Shdr *sec;
  void *v,*ve;
  ul a,b;

  for (sym=sym1;sym<syme;sym++)
    sym->st_other=sym->st_size=0;

  for (sec=sec1;sec<sece;sec++)
    if (sec->sh_type==SHT_RELA)
      for (v=v1+sec->sh_offset,ve=v+sec->sh_size,r=v;v<ve;v+=sec->sh_entsize,r=v)
	if (ELF_R_TYPE(r->r_info)==R_MIPS_CALL16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_DISP||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_HI16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_LO16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_CALL_HI16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_CALL_LO16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_PAGE) {

	  sym=sym1+ELF_R_SYM(r->r_info);

	  /*unlikely to save got space by recording possible holes in addend range*/
	  if ((a=MIPS_HIGH(r->r_addend)+1)>sym->st_other)
	    sym->st_other=a;

	}

  for (*gs=0,sec=sec1;sec<sece;sec++)
    if (sec->sh_type==SHT_RELA)
      for (v=v1+sec->sh_offset,ve=v+sec->sh_size,r=v;v<ve;v+=sec->sh_entsize,r=v)
	if (ELF_R_TYPE(r->r_info)==R_MIPS_CALL16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_DISP||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_HI16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_LO16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_CALL_HI16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_CALL_LO16||
	    ELF_R_TYPE(r->r_info)==R_MIPS_GOT_PAGE) {

	  sym=sym1+ELF_R_SYM(r->r_info);

	  if (sym->st_other) {
	    sym->st_size=++*gs;
	    if (sym->st_other>1)
	      (*gs)+=sym->st_other-1;
	    else
	      massert(!make_got_room_for_stub(sec1,sece,sym,st1,gs));
	    sym->st_other=0;
	  }

	  b=sizeof(r->r_addend)*4; 
	  massert(!(r->r_addend>>b)); 
	  r->r_addend|=((sym->st_size+MIPS_HIGH(r->r_addend))<<b);

	}
  
  return 0;
  
}
