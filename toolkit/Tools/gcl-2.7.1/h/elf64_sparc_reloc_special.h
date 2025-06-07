#undef ELF_R_TYPE
#define ELF_R_TYPE(a) (ELF64_R_TYPE(a)&0xff)
#define ELF_R_ADDEND(a) (((ELF64_R_TYPE(a)>>8)^0x800000)-0x800000)

static int
label_got_symbols(void *v1,Shdr *sec1,Shdr *sece,Sym *sym1,Sym *syme,const char *st1,const char *sn,ul *gs) {

  return 0;

}

static int
find_special_params(void *v,Shdr *sec1,Shdr *sece,const char *sn,
		    const char *st1,Sym *ds1,Sym *dse,Sym *sym1,Sym *syme) {

  return 0;

}


int
store_ival(int *w,ul m,ul v) {

  *w=(v&m)|(*w&~m);

  return 0;

}

int
store_ivals(int *w,ul m,ul v) {

  massert(ovchks(v,~m));
  return store_ival(w,m,v);

}

int
store_ivalu(int *w,ul m,ul v) {

  massert(ovchku(v,~m));
  return store_ival(w,m,v);

}


int
add_ival(int *w,ul m,ul v) {

  return store_ival(w,m,v+(*w&m));

}

int
add_ivalu(int *w,ul m,ul v) {

  return store_ivalu(w,m,v+(*w&m));

}

int
add_ivals(int *w,ul m,ul v) {

  ul l=*w&m,mm;
  
  mm=~m;
  mm|=mm>>1;
  if (l&mm) l|=mm;

  return store_ival(w,m,v+l);

}

int
add_ivalsc(int *w,ul m,ul v) {

  ul l=*w&m,mm;
  
  mm=~m;
  mm|=mm>>1;
  if (l&mm) l|=mm;

  return store_ivals(w,m,v+l);

}
