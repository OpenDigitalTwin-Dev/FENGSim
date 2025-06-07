/*
(c) Copyright W. Schelter 1988, All rights reserved.
    Copyright (c) 2024 Camm Maguire
*/

#include <unistd.h>

#include "include.h"
#include "page.h"

#ifdef HAVE_LIBBFD
#ifdef NEED_CONST
#define CONST const
#endif
#define IN_GCC
#include <bfd.h>
#include <bfdlink.h>
#endif

#define FAT_STRING


enum type what_to_collect;

/* start fasdump stuff */
#include "fasdump.c"



object sSAprofile_arrayA;
#ifdef NO_PROFILE
#ifdef DARWIN/*FIXME macosx10.8 has a prototype (which must match here) but unlinkable function in 64bit*/
int profil(char *buf, size_t bufsiz, unsigned long offset, unsigned int scale){return 0;}
#else
void profil(void){;}
#endif
#endif


#ifndef NO_PROFILE
DEFUN("PROFILE",object,fSprofile,SI
       ,2,2,NONE,OO,OO,OO,OO,(object start_address,object scale),
       "Sets up profiling with START-ADDRESS and  SCALE where scale is \
  between 0 and 256")
{				/* 2 args */
  
  object ar=sSAprofile_arrayA->s.s_dbind;
  void *x;
  fixnum a,s;

  if (!stringp(ar))
    FEerror("si:*Profile-array* not a string",0);
  if( type_of(start_address)!=t_fixnum ||   type_of(scale)!=t_fixnum)
    FEerror("Needs start address and scale as args",0);

  massert((a=fix(start_address))>=0);
  massert((s=fix(scale))>=0);

  x=a&&s ? (void *) (ar->ust.ust_self) : NULL;
  profil(x, (ar->ust.ust_dim),fix(start_address),fix(scale) << 8);
  RETURN1(start_address);
}

#endif
DEFUN("FUNCTION-START",object,fSfunction_start,SI
       ,1,1,NONE,OO,OO,OO,OO,(object funobj),"")
{/* 1 args */
 if(/* type_of(funobj)!=t_cfun */
    /* &&  */type_of(funobj)!=t_function)
    FEerror("not compiled function",0);
 funobj=make_fixnum((long) (funobj->fun.fun_self));
 RETURN1(funobj);
}

/* begin fasl stuff*/
/* this is for windows to not include all of windows.h for this..*/

#include "ptable.h"
#ifdef AIX3
#include <sys/ldr.h>
char *data_load_addr =0;
#endif

#define CFUN_LIM 10000

int maxpage;
object sScdefn;

#define CF_FLAG ((unsigned long)1 << (sizeof(long)*CHAR_SIZE-1)) 

static void
cfuns_to_combined_table(unsigned int n) /* non zero n will ensure new table length */
               
{int ii=0;  
 STATIC int j;
 STATIC object x;
 STATIC char *p,*cf_addr;
 STATIC struct typemanager *tm;
 if (! (n || combined_table.ptable)) n=CFUN_LIM;
 if (n && combined_table.alloc_length < n)
   { 
     (combined_table.ptable)=NULL;
     (combined_table.ptable)= (struct node *)malloc(n* sizeof(struct node));
     if(!combined_table.ptable)
       FEerror("unable to allocate",0);
     combined_table.alloc_length=n;}

 {
   struct pageinfo *v;
   for (v=cell_list_head;v;v=v->next) {
     enum type tp=v->type;
     if (tp!=tm_table[(short)t_function].tm_type)
       continue;
     tm = tm_of(tp);
     p = pagetochar(page(v));
     for (j = tm->tm_nppage; j > 0; --j, p += tm->tm_size) {
       x = (object)p;
       if (type_of(x)!=t_function)
	 continue;
       if (is_free(x) || x->fun.fun_self == NULL)
	 continue;
       /* the cdefn things are the proclaimed call types. */
       cf_addr=(char * ) ((unsigned long)(x->fun.fun_self));
       
       SYM_ADDRESS(combined_table,ii)=(unsigned long)cf_addr;
       SYM_STRING(combined_table,ii)= (char *)(CF_FLAG | (unsigned long)x) ;
       /*       (x->cf.cf_name ? x->cf.cf_name->s.st_self : NULL) ; */
       combined_table.length = ++ii;
       if (ii >= combined_table.alloc_length)
	 FEerror("Need a larger combined_table",0);
     }
   }
 }
}

static int
address_node_compare(const void *node1, const void *node2)
{unsigned int a1,a2;
 a1=((struct node *)node1)->address;
 a2=((struct node *)node2)->address;
 if (a1> a2) return 1;
 if (a1< a2) return -1;
 return 0;
}
 

#if defined(HAVE_LIBBFD) && ! defined(SPECIAL_RSYM)

static int bfd_update;

static MY_BFD_BOOLEAN
bfd_combined_table_update(struct bfd_link_hash_entry *h,PTR ct) {

  if (ct!=&combined_table)
    return MY_BFD_FALSE;

  if (h->type!=bfd_link_hash_defined)
    return MY_BFD_TRUE;

  if (!h->u.def.section) {
    FEerror("Symbol without section",0);
    return MY_BFD_FALSE;
  }

  if (bfd_update) {
    if (combined_table.length>=combined_table.alloc_length)
      FEerror("combined table overflow", 0);
    
    SYM_ADDRESS(combined_table,combined_table.length)=h->u.def.value+h->u.def.section->vma;
    SYM_STRING(combined_table,combined_table.length)=(char *)h->root.string;
  }

  combined_table.length++;

  return MY_BFD_TRUE;

}
#endif


DEFUN("SET-UP-COMBINED",object,fSset_up_combined,SI
	  ,0,1,NONE,OO,OO,OO,OO,(object first,...),"") {

  unsigned int n;
  object siz,l=Cnil,f=OBJNULL;
  fixnum nargs=INIT_NARGS(0);
  va_list ap;

  va_start(ap,first);
  siz=NEXT_ARG(nargs,ap,l,f,make_fixnum(0));
  n = (unsigned int) fix(siz);
  cfuns_to_combined_table(n);

#if !defined(HAVE_LIBBFD) && !defined(SPECIAL_RSYM)
#error Need either BFD or SPECIAL_RSYM
#endif

#if defined(SPECIAL_RSYM)
  if (c_table.ptable) {

    int j,k;

    if((k=combined_table.length)+c_table.length >=  combined_table.alloc_length)
      cfuns_to_combined_table(combined_table.length+c_table.length+20);
    
    for(j = 0; j < c_table.length;) { 
      SYM_ADDRESS(combined_table,k) =SYM_ADDRESS(c_table,j);
      SYM_STRING(combined_table,k) =SYM_STRING(c_table,j);
      k++;
      j++;
    }

    combined_table.length += c_table.length ;

  }

#else
#if defined(HAVE_LIBBFD)
  if (link_info.hash) {

    bfd_update=0;
    bfd_link_hash_traverse(link_info.hash,
			   bfd_combined_table_update,&combined_table);
    
    if (combined_table.length >=combined_table.alloc_length)
      cfuns_to_combined_table(combined_table.length);
    
    bfd_update=1;
    bfd_link_hash_traverse(link_info.hash,
			   bfd_combined_table_update,&combined_table);
    bfd_update=0;

  }
#endif
#endif

  qsort(combined_table.ptable,combined_table.length,sizeof(*combined_table.ptable),address_node_compare);

  RETURN1(siz);

}

static int  prof_start;
static int
prof_ind(unsigned int address, int scale)
{address = address - prof_start ;
 if (address > 0) return ((address * scale) >> 8) ;
 return 0;
}

/* sum entries AAR up to DIM entries */
static int
string_sum(register unsigned char *aar, unsigned int dim)
{register unsigned char *endar;
 register unsigned int count = 0;
endar=aar+dim;
 for ( ; aar< endar; aar++)
   count += *aar;
 return count;
}


DEFUN("DISPLAY-PROFILE",object,fSdisplay_profile,SI
       ,2,2,NONE,OO,OO,OO,OO,(object start_addr,object scal),"") {

  if (!combined_table.ptable)
   FEerror("must symbols first",0);
   /* 2 args */
  {
    unsigned int prev,next,upto,dim,total;
    int j,scale,count;
    unsigned char *ar;
    object obj_ar;
    obj_ar=sSAprofile_arrayA->s.s_dbind;
    if (!stringp(obj_ar))
      FEerror("si:*Profile-array* not a string",0);
    ar=obj_ar->ust.ust_self;
    scale=fix(scal);
    prof_start=fix(start_addr);
    vs_top=vs_base;
    dim= (obj_ar->ust.ust_dim);

    total=string_sum(ar,dim);

    j=0;
    {
      int i, finish = combined_table.length-1;
      for(i =0,prev=SYM_ADDRESS(combined_table,i); i< finish;prev=next)	{
	++i;
	next=SYM_ADDRESS(combined_table,i);
	if (prev<prof_start)
	  continue;
	upto=prof_ind(next,scale);
	if (upto >= dim)
	  upto=dim;
	{
	  const char *name; unsigned long uname;
	  count=0;
	  for(;j<upto;j++)
	    count += ar[j];
	  if (count > 0) {
	    name=SYM_STRING(combined_table,i-1);
	    uname = (unsigned long) name;
	    printf("\n%6.2f%% (%5d): ",(100.0*count)/total, count);
	    fflush(stdout);
	    if (CF_FLAG & uname)
	      ;/*{ if (~CF_FLAG & uname) prin1( ((object) (~CF_FLAG & uname))->cf.cf_name,Cnil);} *//*FIXME*/
	     else if (name ) printf("%s",name);};
	  if (upto==dim) goto TOTALS ;

	}
      }
    }
  TOTALS:
    printf("\nTotal ticks %d",total);fflush(stdout);
  }
  RETURN1(start_addr);
}



/* end fasl stuff*/


/* These are some low level hacks to allow determining the address
   of an array body, and to allow jumping to inside the body
   of the array */

DEFUN("ARRAY-ADRESS",object,fSarray_adress,SI
       ,1,1,NONE,OO,OO,OO,OO,(object array),"")
{/* 1 args */
 array=make_fixnum((long) (&(array->st.st_self[0])));
 RETURN1(array);
}

/* This is some very low level code for hacking invokation of
   m68k instructions in a lisp array.  The index used should be
   a byte index.  So invoke(ar,3) jmps to byte ar+3.
   */

#ifdef CLI

invoke(ar)
char *ar;
{asm("movel a6@(8),a0");
 asm("jmp a0@");
}
/* save regs (2 3 4 5 6 7  10 11 12 13 14) and invoke restoring them */
save_regs_invoke(ar)
char *ar;
{asm("moveml #0x3f3e,sp@-");
 invoke(ar);
 asm("moveml a6@(-44),#0x7cfc");
}

/* DEFUNO_NEW("SAVE-REGS-INVOKE",object,fSsave_regs_invoke,SI
   ,2,2,NONE,OO,OO,OO,OO,void,siLsave_regs_invoke,"",(x0,x1))
object x0,x1;
{int x;
  check_type_integer(&x1);
  x=save_regs_invoke((x0->st.st_self)+fix(x1));
 x0=make_fixnum(x);
 RETURN1(x0);
}
*/

#endif

DEFVAR("*PROFILE-ARRAY*",sSAprofile_arrayA,SI,Cnil,"");
void
gcl_init_fat_string(void)
{
 
 make_si_constant("*ASH->>*",(-1==(((int)-1) >> 20))? Ct :Cnil);
/* #ifdef SFASL */
/*  make_si_function("BUILD-SYMBOL-TABLE",build_symbol_table); */
/* #endif */


 init_fasdump();
 
}







