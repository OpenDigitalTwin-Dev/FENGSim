/* 
Copyright William Schelter. All rights reserved.
Copyright 2024 Camm Maguire */

#if !defined(HAVE_LIBBFD) && !defined(SPECIAL_RSYM)
#error Need either BFD or SPECIAL_RSYM
#endif

#ifndef SPECIAL_RSYM

/* Replace this with gcl's own hash structure at some point */
static int
build_symbol_table_bfd(void) {

  int u,v;
  unsigned long pa;
  asymbol **q;

  bfd_init();
  if (!(bself=bfd_openr(kcl_self,0)))
    FEerror("Cannot open self\n",0);
  if (!bfd_check_format(bself,bfd_object))
    FEerror("I'm not an object",0);
/*    if (link_info.hash) */
/*      bfd_link_hash_table_free(bself,link_info.hash); */
#ifdef HAVE_OUTPUT_BFD
    link_info.output_bfd = bfd_openw("/dev/null", bfd_get_target(bself));
#endif
  if (!(link_info.hash = bfd_link_hash_table_create (bself)))
    FEerror("Cannot make hash table",0);
  if (!bfd_link_add_symbols(bself,&link_info))
    FEerror("Cannot add self symbols\n",0);
  if ((u=bfd_get_symtab_upper_bound(bself))<0)
    FEerror("Cannot get self's symtab upper bound",0);

#ifdef HAVE_ALLOCA
  q=(asymbol **)alloca(u);
#else
  q=(asymbol **)malloc(u);
#endif
  if ((v=bfd_canonicalize_symtab(bself,q))<0)
    FEerror("Cannot canonicalize self's symtab",0);

  for (u=0;u<v;u++) {
    char *c=NULL;
    struct bfd_link_hash_entry *h;

    if (!*q[u]->name)
      continue;

    if (strncmp(q[u]->section->name,"*UND*",5) && !(q[u]->flags & BSF_WEAK))
      continue;

    if ((c=(char *)strstr(q[u]->name,"@@"))) {
      *c=0;
      if (!(h=bfd_link_hash_lookup(link_info.hash,q[u]->name,MY_BFD_TRUE,MY_BFD_TRUE,MY_BFD_TRUE)))
	FEerror("Cannot make new hash entry",0);
      h->type=bfd_link_hash_new;
    } else if 
      (!(h=bfd_link_hash_lookup(link_info.hash,q[u]->name,MY_BFD_FALSE,MY_BFD_FALSE,MY_BFD_TRUE)) &&
       !(h=bfd_link_hash_lookup(link_info.hash,q[u]->name,MY_BFD_TRUE,MY_BFD_TRUE,MY_BFD_TRUE)))
	FEerror("Cannot make new hash entry",0);

    if (h->type!=bfd_link_hash_defined) {
      if (!q[u]->section)
	FEerror("Symbol ~S is missing section",1,make_simple_string(q[u]->name));
      if (!my_plt(q[u]->name,&pa)) {
/* 	 printf("my_plt %s %p\n",q[u]->name,(void *)pa);  */
 	if (q[u]->value && q[u]->value!=pa)
 	  FEerror("plt address mismatch", 0);
 	else
 	  q[u]->value=pa;
      }
      if (q[u]->value) {
	h->type=bfd_link_hash_defined;
	h->u.def.value=q[u]->value+q[u]->section->vma;
	h->u.def.section=q[u]->section;
      }
    }

    if (c) {
      *c='@';
      c=NULL;
    }
  }

#ifndef HAVE_ALLOCA
  free(q);
#endif

  return 0;
  
}

#endif /* special_rsym */

void *min_text;
extern void *data_start;

int
is_text_addr(void *p) {
  extern int initializing_boot;
  if (initializing_boot) return 1;/*FIXME*/
  return p>=min_text && p<data_start && !is_bigger_fixnum(p) ? 1 : 0;
}

LFD(build_symbol_table)(void) {

  printf("Building symbol table for %s ..\n",kcl_self);fflush(stdout);

#ifdef SPECIAL_RSYM

#ifndef USE_DLOPEN
  load_self_symbols();
#endif

#else

  build_symbol_table_bfd();

#endif

  {
    fixnum i;

    min_text=data_start;
    for (i=0;i<c_table.alloc_length;i++) {
      void *p=(void *)c_table.ptable[i].address;
      min_text=p<min_text ? p : min_text;
    }
  }

}

#ifndef _WIN32
int
use_symbols(double d,...) {

  double d2;
#ifndef DARWIN
  extern void sincos(double,double *,double *);

  sincos(d,&d,&d2);

#else

  d=sin(d)+cos(d);
  d2=sin(d)+cos(d);

#endif

  { dcomplex c=d+I*d;
    c=c*c;
    d=c+I*c;
    d=c/d;
    d=d/(c+I*d);
  }
  
  return (int)d;

}
#endif

void
gcl_init_sfasl() {

#ifdef SFASL
 make_si_function("BUILD-SYMBOL-TABLE",build_symbol_table);
#endif

}
