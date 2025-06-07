 /* Copyright William F. Schelter  All Rights Reserved.
    Copyright 2024 Camm Maguire

   Utility for writing out lisp objects and reading them in:
   Basically it attempts to write out only those things which could
   be written out using princ and reread.   It just uses less space
   and is faster.
   

   Primitives for dealing with a `fasd stream'.
   Such a stream is really an array containing some state and a lisp file stream.
   Note that having *print-circle* == nil wil make this faster.  gensyms will
   still be dumped correctly in that case.
   
   open_fasd
   write_fasd_top
   read_fasd_top
   close_fasd
   
   */



#ifndef FAT_STRING
#include "include.h"
#endif

static void
clrhash(object);


object coerce_stream();
static object fasd_patch_sharp(object x, int depth);
object make_pathname ();


static int needs_patching;

struct fasd current_fasd;


enum circ_ind {
  LATER_INDEX,
  NOT_INDEXED,
  FIRST_INDEX,
  };

enum dump_type {
  d_nil,         /* dnil: nil */
  d_eval_skip,        /* deval o1: evaluate o1 after reading it */
  d_delimiter,   /* occurs after d_list,d_general and d_new_indexed_items */
  d_enter_vector,      /* d_enter_vector o1 o2 .. on d_delimiter , make a cf_data with
		    this length.   Used internally by akcl.  Just make
		    an array in other lisps */
  d_cons,        /* d_cons o1 o2: (o1 . o2) */
  d_dot,
  d_list,    /* list* delimited by d_delimiter d_list,o1,o2, ... ,d_dot,on
		for (o1 o2       . on)
		or d_list,o1,o2, ... ,on,d_delimiter  for (o1 o2 ...  on)
	      */
  d_list1,   /* nil terminated length 1  d_list1,o1   */
  d_list2,    /* nil terminated length 2 */
  d_list3,
  d_list4,
  d_eval,
  d_short_symbol,
  d_short_string,
  d_short_fixnum,
  d_short_symbol_and_package,
  d_bignum,
  d_fixnum,
  d_string,
  d_objnull,
  d_structure,
  d_package,
  d_symbol,
  d_symbol_and_package,
  d_end_of_file,
  d_standard_character,
  d_vector,
  d_array,
  d_begin_dump,
  d_general_type,
  d_sharp_equals,              /* define a sharp */
  d_sharp_value,
  d_sharp_value2,
  d_new_indexed_item,
  d_new_indexed_items,
  d_reset_index,
  d_macro,
  d_reserve1,
  d_reserve2,
  d_reserve3,
  d_reserve4,
  d_indexed_item3,       /* d_indexed_item3 followed by 3bytes to give index */
  d_indexed_item2,        /* d_indexed_item2 followed by 2bytes to give index */
  d_indexed_item1,
  d_indexed_item0      /* This must occur last ! */
        
};

/* set whole structures!  */
#define SETUP_FASD_IN(fd) do{ \
  fas_stream= (fd)->stream; \
  dump_index =   fix((fd)->index) ; \
  current_fasd= * (fd);}while(0)

#define SAVE_CURRENT_FASD \
   struct fasd old_fd; \
   int old_dump_index = dump_index; \
   object old_fas_stream = fas_stream; \
   int old_needs_patching = needs_patching; \
   old_fd = current_fasd;


#define  RESTORE_FASD \
    current_fasd =old_fd ; \
    dump_index= old_dump_index ; \
    needs_patching = old_needs_patching ; \
    fas_stream = old_fas_stream
  
  
#define FASD_SHARP_LIMIT 250  /* less than short_max */
#define SETUP_FASD_OUT(fasd) SETUP_FASD_IN(fasd)

#define dump_hash_table (current_fasd.table)

#define SIZE_D_CODE 8
#define SIZE_BYTE 8
#define SIZE_SHORT ((2*SIZE_BYTE) - SIZE_D_CODE)
/* this is not! the maximum short !!  It is shorter */
#define SHORT_MAX ((1<< SIZE_SHORT) -1)


/* given SHORT extract top code (say 4 bits) and bottom byte */
#define TOP(i) (i >> SIZE_BYTE)
#define BOTTOM(i) (i &  ~(~0UL << SIZE_BYTE))

#define FASD_VERSION 2

object fas_stream;
int dump_index;
/* struct htent *gethash(); */
static void read_fasd1(int i, object *loc);
object extended_read();

/* to enable debugging define the following,
   and set debug=1 or debug=2
*/   
/* #define DEBUG */

#ifdef DEBUG /*FIXME debugging versions need sync with getc -> readc_stream, etc.*/

#define PUT(x) writec_stream1((char)x,fas_stream)
#define GET() readc_stream1()
#define D_FWRITE fwrite1
#define D_FREAD fread1

char *dump_type_names[]={ "d_nil",
     "d_eval_skip",
     "d_delimiter",
     "d_enter_vector",
     "d_cons",
     "d_dot",
     "d_list",
     "d_list1",
     "d_list2",
     "d_list3",
     "d_list4",
     "d_eval",
     "d_short_symbol",
     "d_short_string",
     "d_short_fixnum",
     "d_short_symbol_and_package",
     "d_bignum",
     "d_fixnum",
     "d_string",
     "d_objnull",
     "d_structure",
     "d_package",
     "d_symbol",
     "d_symbol_and_package",
     "d_end_of_file",
     "d_standard_character",
     "d_vector",
     "d_array",
     "d_begin_dump",
     "d_general_type",
     "d_sharp_equals",
     "d_sharp_value",
      "d_sharp_value2",
     "d_new_indexed_item",
     "d_new_indexed_items",
     "d_reset_index",
     "d_macro",
     "d_reserve1",
     "d_reserve2",
     "d_reserve3",
     "d_reserve4",
     "d_indexed_item3",
     "d_indexed_item2",
     "d_indexed_item1",
     "d_indexed_item0"};

int debug;
int
print_op(i)
{if (debug)
   {if (i < d_indexed_item0 & i >= 0)
	   {printf("\n<%s>",dump_type_names[i]);}
   else {printf("\n<indexed_item0:%d>",i -d_indexed_item0);}}
 return i;
}

#define PUTD(str,i) putd(str,i)
void
putd(str,i)
char *str;
  int i;
{if (debug)
   {printf("{");
    printf(str,i);
    printf("}");}
 writec_stream(i,fas_stream);}

void
writec_stream1(x)
int x;
{  if (debug) printf("(%x,%d,%c)",x,x,x);
   writec_stream(x,fas_stream);
/*    fflush(stdout); */
 }

int
readc_stream1()
{ int x;
   x= readc_stream(fas_stream);
  if (debug) printf("(%x,%d,%c)",x,x,x);
/*   fflush(stdout); */
  return x;
 }

int
fread1(p,n1,n2,st)
     FILE* st;
     char *p;
     int n1;
     int n2;
{int i,j;
 j=SAFE_FREAD(p,n1,n2,st);
 if(debug)
 {printf("[");
  n1=n1*n2;
  for(i=0;i<n1; i++)
    writec_stream(p[i],sLAstandard_outputA->s.s_dbind);
  printf("]");
/*   fflush(stdout);} */
    return j;

}
 
   
 

int
fwrite1(p,n1,n2,st)
     FILE* st;
     char *p;
     int n1;
     int n2;
{int i,j;
 j=fwrite(p,n1,n2,st);
 if(debug)
 {printf("[");
  n1=n1*n2;
  for(i=0;i<n1; i++)
    writec_stream(p[i],sLAstandard_outputA->s.s_dbind);
  printf("]");}
    return j;
}


#define GET_OP() ((unsigned)print_op((unsigned char)readc_stream(fas_stream)))
#define PUT_OP(x) writec_stream(print_op(x),fas_stream)
 
#define DP(sw)  sw   /*  if (debug) {printf("\ncase sw");} */
#define GETD(str) getd(str)

int
getd(str)
 char *str;
{ int i = (unsigned char)readc_stream(fas_stream);
 if(debug){
   printf("{");
   printf(str,i);
   printf("}");}
  return i;}
#define DPRINTF(a,b)  do{if(debug) printf(a,b);} while(0)
#else
#define PUT(x) writec_stream((char)x,fas_stream)
#define GET() ((unsigned char)readc_stream(fas_stream))
#define GET_OP GET
#define PUT_OP PUT
#define D_FWRITE fwrite_int
#define D_FREAD fread_int
#define DP(sw)  sw
#define PUTD(a,b) PUT(b)
#define GETD(a) GET()
#define DPRINTF(a,b)  
/* #define fwrite_int(a_,b_,c_,d_) {register char *_p=(a_),*_pe=_p+(b_)*(c_);for (;_p<_pe;) writec_stream(*_p++,(d_));} */
/* #define fread_int(a_,b_,c_,d_)  {register char *_p=(a_),*_pe=_p+(b_)*(c_);for (;_p<_pe;) *_p++=readc_stream(d_);} */
#define fwrite_int(a_,b_,c_,d_) {register unsigned _i;for (_i=0;_i<(b_)*(c_);_i++) writec_stream(((char *)(a_))[_i],(d_));}
#define fread_int(a_,b_,c_,d_)  {register unsigned _i;for (_i=0;_i<(b_)*(c_);_i++) ((char *)(a_))[_i]=readc_stream(d_);}


#endif


      
#define D_TYPE_OF(byt) \
  ((enum dump_type )((unsigned int) byt & ~(~0UL << SIZE_D_CODE)))

/* this field may be the top of a short for length, or part of an extended
   code */
#define E_TYPE_OF(byt) ((unsigned int) byt >> (SIZE_D_CODE))
  /* takes two bytes and reconstructs the SIZE_SHORT int from them after
     dropping the code */


/* takes two bytes i and j and returns the SHORT associated */ 
#define LENGTH(i,j) MAKE_SHORT(E_TYPE_OF(i),(j))

#define MAKE_SHORT(top,bot) (((top)<< SIZE_BYTE) + (bot))

#define READ_BYTE1() ((unsigned char)readc_stream(fas_stream))

#define GET8(varx ) \
 do{unsigned long long var=READ_BYTE1();  \
   var |=  ((unsigned long long)READ_BYTE1() << SIZE_BYTE); \
   var |=  ((unsigned long long)READ_BYTE1() << (2*SIZE_BYTE)); \
   var |=  ((unsigned long long)READ_BYTE1() << (3*SIZE_BYTE)); \
   var |=  ((unsigned long long)READ_BYTE1() << (4*SIZE_BYTE)); \
   var |=  ((unsigned long long)READ_BYTE1() << (5*SIZE_BYTE)); \
   var |=  ((unsigned long long)READ_BYTE1() << (6*SIZE_BYTE)); \
   var |=  ((unsigned long long)READ_BYTE1() << (7*SIZE_BYTE)); \
   DPRINTF("{8byte:varx= %ld}", var); \
     varx=var;} while (0)

#define GET4(varx ) \
 do{int  var=READ_BYTE1();  \
   var |=  (READ_BYTE1() << SIZE_BYTE); \
   var |=  (READ_BYTE1() << (2*SIZE_BYTE)); \
   var |=  (READ_BYTE1() << (3*SIZE_BYTE)); \
   DPRINTF("{4byte:varx= %d}", var); \
     varx=var;} while (0)

#define GET2(varx ) \
 do{int  var=READ_BYTE1();  \
   var |=  (READ_BYTE1() << SIZE_BYTE); \
     DPRINTF("{2byte:varx= %d}", var); \
     varx=var;} while (0)

#define GET3(varx ) \
 do{int  var=READ_BYTE1();  \
   var |=  (READ_BYTE1() << SIZE_BYTE); \
   var |=  (READ_BYTE1() << (2*SIZE_BYTE)); \
          DPRINTF("{3byte:varx= %d}", var); \
     varx=var;} while (0)



#define MASK ~(~0UL << 8)
#define WRITE_BYTEI(x,i)  writec_stream((((x) >> (i*SIZE_BYTE)) & MASK),fas_stream)

#define PUTFIX(v_) Join(PUT,SIZEOF_LONG)(v_)
#define GETFIX(v_) Join(GET,SIZEOF_LONG)(v_)

#define PUT8(varx ) \
 do{unsigned long long var= varx ; \
     DPRINTF("{8byte:varx= %ld}", var); \
       WRITE_BYTEI(var,0); \
     WRITE_BYTEI(var,1); \
     WRITE_BYTEI(var,2); \
     WRITE_BYTEI(var,3); \
     WRITE_BYTEI(var,4); \
     WRITE_BYTEI(var,5); \
     WRITE_BYTEI(var,6); \
     WRITE_BYTEI(var,7);} while(0)

#define PUT4(varx ) \
 do{unsigned long var= varx ; \
     DPRINTF("{4byte:varx= %d}", var); \
       WRITE_BYTEI(var,0); \
     WRITE_BYTEI(var,1); \
     WRITE_BYTEI(var,2); \
     WRITE_BYTEI(var,3);} while(0)

#define PUT2(var ) \
 do{unsigned long v=var; \
     DPRINTF("{2byte:var= %d}", v); \
       WRITE_BYTEI(v,0); \
     WRITE_BYTEI(v,1); \
     } while(0)

#define PUT3(var ) \
 do{unsigned long v=var; \
     DPRINTF("{3byte:var= %d}", v); \
       WRITE_BYTEI(v,0); \
     WRITE_BYTEI(v,1); \
       WRITE_BYTEI(v,2); \
     } while(0)




  /* constructs the first byte containing ecode and top
     top either stands for something in extended codes, or for something
     the top part of a SIZE_SHORT int
   */
#define MAKE_CODE(CODE,Top) \
  ((unsigned int)(CODE) | ((unsigned int)(Top) <<  SIZE_D_CODE))


/* write out two bytes encoding the enum d_code  CODE and SHORT SH. */



#define PUT_CODE_AND_SHORT(CODE,SH) \
  PUT(MAKE_CODE(CODE,TOP(SH))); \
  PUT(BOTTOM(SH)); 

#define READ_SYMBOL(leng,pack,to) \
	do { BEGIN_NO_INTERRUPT;{char  *p=alloc_relblock(leng);\
	 D_FREAD(p,1,leng,fas_stream); \
	 string_register->st.st_dim = leng; \
	 string_register->st.st_self = p; \
	 to=(pack==Cnil ? make_symbol(string_register) : intern(string_register,pack)); \
	   END_NO_INTERRUPT;} \
	 }while(0)

#define  READ_STRING(leng,loc) do {BEGIN_NO_INTERRUPT;     \
     *loc = alloc_simple_string(leng); \
     (*loc)->st.st_self=alloc_relblock(leng); END_NO_INTERRUPT; \
/* Now handled in SAFE_FREAD -- CM 20040210 */ \
/*   memset((*loc)->st.st_self,0,leng); */ /* fread won't restart if it triggers an SGC segfault -- CM */ \
  D_FREAD((*loc)->st.st_self,1,leng,fas_stream);} while(0)

/* if try_hash finds it we don't need to write the object
   Otherwise we write the index type and the object
 */
#define NUMBER_ZERO_ITEMS (SHORT_MAX - (int) d_indexed_item0)



static enum circ_ind
do_hash(object obj, int dot)
{    struct cons *e;
     int i;
     e=gethash(obj,dump_hash_table); 
     if (e->c_cdr==OBJNULL) 
/* We won't index things unless they have  < -2 in the hash table */
  {   if(type_of(obj)!=t_package) return NOT_INDEXED;
      sethash(obj,dump_hash_table,make_fixnum(dump_index));
      e=gethash(obj,dump_hash_table);	 
	PUT_OP(d_new_indexed_item);
	DPRINTF("{dump_index=%d}",dump_index);
	dump_index++;
	return FIRST_INDEX;}
	
     i = fix(e->c_car);
     if (i == -1) return NOT_INDEXED; /* don't want to index this baby */
     
     if (dot) PUT_OP(dot);
     if ( i < -1)
       { e->c_car = make_fixnum(dump_index);
	 PUT_OP(d_new_indexed_item);
	 DPRINTF("{dump_index=%d}",dump_index);
	 dump_index++;
	 return FIRST_INDEX;
       }
     if (i < (NUMBER_ZERO_ITEMS))
       {PUT_OP(i+(int)d_indexed_item0); return LATER_INDEX;}
     if (i < (2*SHORT_MAX - (int)d_indexed_item0))
       {PUT_OP((int)d_indexed_item1);
	PUTD("n=%d",i- NUMBER_ZERO_ITEMS);
	return LATER_INDEX;
      }
     if (i < SHORT_MAX*SHORT_MAX)
       {PUT_OP((int)d_indexed_item2);
	PUT2(i);
	return LATER_INDEX;
      }
     if (i < SHORT_MAX*SHORT_MAX*SHORT_MAX)
       {PUT_OP((int)d_indexed_item3);
	 PUT3(i);
	 return LATER_INDEX;
       }
     else
       FEerror("too large an index",0);
     return LATER_INDEX;
   }
 
static void write_fasd(object obj);

DEFUN("WRITE-FASD-TOP",object,fSwrite_fasd_top,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object obj, object x),"") {

/* static object */
/* FFN(write_fasd_top)(object obj, object x) */
/* { */
struct fasd *fd = (struct fasd *) x->v.v_self;
  if (fd->direction == sKoutput)
    SETUP_FASD_IN(fd);
  else FEerror("bad value for open slot of fasd",0);

  write_fasd(obj);
  /* we could really allocate a fixnum and then smash its field if this
     is to costly */
  (fd)->index = make_fixnum(dump_index);
  return obj;
}

/* It is assumed that anything passed to eval should be first
   sharp patched, and that there will be no more patching afterwards.
   The object returned might have arbitrary complexity.
*/   

#define MAYBE_PATCH(result) \
  if (needs_patching)  result =fasd_patch_sharp(result,0)

DEFUN("READ-FASD-TOP",object,fSread_fasd_top,SI
	  ,1,1,NONE,OO,OO,OO,OO,(object x),"") {
/* static object */
/* FFN(read_fasd_top)(object x) */
/* { */
  struct fasd *fd = (struct fasd *)  x->v.v_self;
   VOL int e=0;
   object result;
   SAVE_CURRENT_FASD;
   
   SETUP_FASD_IN(fd);

   frs_push(FRS_PROTECT, Cnil);
   if (nlj_active) {
     e = TRUE;
     goto L;
   }
   needs_patching=0;
   if (current_fasd.direction == sKinput)
     {read_fasd1(GET_OP(),&result);
      MAYBE_PATCH(result);
      (fd)->index = make_fixnum(dump_index);
      fd->direction=current_fasd.direction;

    }
   else
     if(current_fasd.direction== Cnil) result= current_fasd.eof;
   else
       FEerror("Stream not open for input",0);
 L:

   frs_pop();
   
   if (e) {
     nlj_active = FALSE;
     unwind(nlj_fr, nlj_tag);
     fd->direction=Cnil;
     RESTORE_FASD;
     return Cnil;
   }
   else
     { RESTORE_FASD;
     return result;}
 }
#ifdef STATIC_FUNCTION_POINTERS
object
fSread_fasd_top(object x) {
  return FFN(fSread_fasd_top)(x);
}
#endif


object sLeq;
object sSPinit;
void Lmake_hash_table();

DEFUN("OPEN-FASD",object,fSopen_fasd,SI
	  ,4,4,NONE,OO,OO,OO,OO,(object stream, object direction, object eof, object tabl),"") {


/* static object */
/* FFN(open_fasd)(object stream, object direction, object eof, object tabl) */
/* { */
  object str=Cnil;
  object result;
  if(direction==sKinput)
    {str=coerce_stream(stream,0);
    if (tabl==Cnil)
      tabl=alloc_simple_vector(0);
    else
      check_type(tabl,t_simple_vector);}
  if(direction==sKoutput)
    {str=coerce_stream(stream,1);
      if(tabl==Cnil) tabl=gcl_make_hash_table(sLeq);
    else
      check_type(tabl,t_hashtable);}
  massert(str==stream);
  result=alloc_simple_vector(sizeof(struct fasd)/sizeof(object));
  array_allocself(result,1,Cnil);
  {struct fasd *fd= (struct fasd *)result->sv.sv_self;
  fd->table=tabl;
  fd->stream=stream;
  fd->direction=direction;
  fd->eof=eof;
  fd->index=small_fixnum(0);
  fd->package=symbol_value(sLApackageA);
  fd->filepos = make_fixnum(file_position(stream));
  
  SETUP_FASD_IN(fd);
  if (direction==sKoutput){
    PUT_OP((int)d_begin_dump);
    PUTD("version=%d",FASD_VERSION);
    PUT4(0);  /* reserve space for the size of index array needed */
    /*  equivalent to:   write_fasd(current_fasd.package);
	except we don't want to index this, so that we can open
	with an empty array.
    */
    PUT_OP(d_package);
    write_fasd(current_fasd.package->p.p_name);
    
  }
  else			/* input */
    { object tem;
    read_fasd1(GET_OP(),&tem);
    if(tem!=current_fasd.table) FEerror("not positioned at beginning of a dump",0);
    }
  fd->index=make_fixnum(dump_index);
  fd->filepos=current_fasd.filepos;
  fd->package=current_fasd.package;
  fd->table_length=current_fasd.table_length;
  return result;
  }}
#ifdef STATIC_FUNCTION_POINTERS
object
fSopen_fasd(object stream, object direction, object eof, object tabl) {
  return FFN(fSopen_fasd)(stream,direction,eof,tabl);
}
#endif

DEFUN("CLOSE-FASD",object,fSclose_fasd,SI,1,1,NONE,OO,OO,OO,OO,(object ar),"") {
/* static object */
/* FFN(close_fasd)(object ar) */
/* { */
  struct fasd *fd= (struct fasd *)(ar->v.v_self);
   check_type(ar,t_simple_vector);
     if(fd->direction==sKoutput)
       {clrhash(fd->table);
	SETUP_FASD_IN(fd);
	PUT_OP(d_end_of_file);
	{int i = file_position(fd->stream);
	 if(type_of(fd->filepos) == t_fixnum)
	  { file_position_set(fd->stream,fix(fd->filepos) +2);
	    /* record the length of array needed to read the indices */
	    PUT4(fix(fd->index));
	    /* move back to where we were */
	    file_position_set(fd->stream,i);
	  }}
	 
      }
   /*  else FEerror("bad fasd stream",0); */
   fd->direction=Cnil;
   return ar;
  
 }
#ifdef STATIC_FUNCTION_POINTERS
object
fSclose_fasd(object ar) {
  return FFN(fSclose_fasd)(ar);
}
#endif


#define HASHP(x) 1
#define TRY_HASH \
  if(do_hash(obj,0)==LATER_INDEX) return;

static void
write_fasd(object obj)
{  fixnum j,leng;

   /* hook for writing other data in fasd file */


   
   /* check if we have already output the object in a hash table.
      If so just record the index */
   {
     /* if dump_index is too large or the object has not been written before
	we output it now */

     switch(type_of(obj)){

     case DP(t_cons:)
       TRY_HASH;

       /* decide how long we think this list is */
       
       {object x=obj->c.c_cdr;
	int l=0;
	if (obj->c.c_car == siSsharp_comma)
	  { PUT_OP(d_eval);
	    write_fasd(x);
	    break;}
	while(1)
	  { if(x==Cnil)
	      {PUT_OP(d_list1+l);
	       break;}
	    if(consp(x))
	      {if ((int) d_list1 + ++l > (int) d_list4)
	       {PUT_OP(d_list);
		break;}
	       else {x=x->c.c_cdr;
		     continue;}}
	    /* 1 to 4 done */
	    if(l==0)
	      {PUT_OP(d_cons);
	       write_fasd(obj->c.c_car);
	       write_fasd(obj->c.c_cdr);
	       return;}
	    else
	      {PUT_OP(d_list);
	       break;
	     }}}

 /*    WRITE_LIST: */

       write_fasd(obj->c.c_car);
       obj=obj->c.c_cdr;
       {int l=0;
	while(1)
	  {if (consp(obj))
	     { enum circ_ind is_indexed=LATER_INDEX;
	       if(HASHP(t_cons)){
		 is_indexed=do_hash(obj,d_dot);
		 if  (is_indexed == LATER_INDEX)
		 return;
	       if (is_indexed==FIRST_INDEX)
		 { PUT_OP(d_cons);
		   write_fasd(obj->c.c_car);
		   write_fasd(obj->c.c_cdr);
		  return;}}
	       write_fasd(obj->c.c_car);
	       l++;
	       obj=obj->c.c_cdr;}
	   else
	     if(obj==Cnil)
	       {if (l> ((int) d_list4- (int) d_list1))
		  {PUT_OP(d_delimiter);}
		return;}
	   else
	     {PUT_OP(d_dot);
	      write_fasd(obj);
	      return;}}}

     case DP(t_symbol:)
          
       if (obj==Cnil)
	 {PUT_OP(d_nil); return;}
        TRY_HASH;
	leng=VLEN(obj->s.s_name);
       if (current_fasd.package!=obj->s.s_hpack)
	 {{
	   if (leng< SHORT_MAX)
	      {PUT_OP(d_short_symbol_and_package);
	       PUTD("leng=%d",leng);}
	   else
	     { j=leng;
	       PUT_OP(d_symbol_and_package);
	       PUT4(j);}}
	  
	  write_fasd(obj->s.s_hpack);}
       else
	 { if (leng< SHORT_MAX)
	     { PUT_OP(d_short_symbol);
	       PUTD("leng=%d",leng);}
	 else
	   { j=leng;
	     PUT_OP(d_symbol);
	     PUT4(j);}
	   }
       D_FWRITE(obj->s.s_name->st.st_self,1,leng,fas_stream);
       break;
     case DP(t_fixnum:)
       leng=fix(obj);
       if ((leng< (SHORT_MAX/2))
	   && (leng > -(SHORT_MAX/2)))
	 {PUT_OP(d_short_fixnum);
	    PUTD("leng=%d",leng);}
       else
	 {PUT_OP(d_fixnum);
	  j=leng;
	  PUTFIX(j);}
       break;
     case DP(t_character:)
       PUT_OP(d_standard_character);
       PUTD("char=%c",char_code(obj));
       break;
     case DP(t_simple_string:)
     case DP(t_string:)
       leng=VLEN(obj);
       if (leng< SHORT_MAX)
	 {PUT_OP(d_short_string);
	  PUTD("leng=%d",leng);}
       else
	 {j=leng;
	  PUT_OP(d_string);
	  PUT4(j);}
       D_FWRITE(obj->st.st_self,1,leng,fas_stream);
       break;
     case DP(t_bignum:)
       PUT_OP(d_bignum);
#ifdef GMP
     {int l = MP(obj)->_mp_size;
     int m = (l >= 0 ? l : -l);
      
     mp_limb_t *u = MP(obj)->_mp_d;
     /* fix this */
     /* if (sizeof(mp_limb_t) != 4) { FEerror("fix for gmp",0);} */
     PUT4(l);
     while (-- m >=0) {
#if MP_LIMB_BYTES == 8
	 PUT8(*u);
#elif MP_LIMB_BYTES == 4
	 PUT4(*u); 
#else
#error Bad MP_LIMB_BYTES
#endif
       u++;
     }
     break;}
#else     
       {int l = obj->big.big_length;
	plong *u = obj->big.big_self;
	PUT4(l);
	while (-- l >=0)
	  {PUT4(*u) ; u++;}
       break;}
#endif       
     case DP(t_package:)
       TRY_HASH;
       PUT_OP(d_package);
       write_fasd(obj->p.p_name);
       break;
     case DP(t_structure:)

       TRY_HASH;
       {int narg=S_DATA(obj->str.str_def)->length;
	int i;
	object name= S_DATA(obj->str.str_def)->name;
	if(narg >= SHORT_MAX)
	  FEerror("Only dump structures whose length < ~a",1,make_fixnum(SHORT_MAX));
	PUT_OP(d_structure);
	PUTD("narg=%d",narg);
	write_fasd(name);
	for (i = 0;  i < narg;  i++)
	    write_fasd(structure_ref(obj,name,i));}

	break;

      case DP(t_array:)
	TRY_HASH;
	PUT_OP(d_array);
	{ int leng=obj->a.a_dim;
	  int i;
	  PUT4(leng);
	  PUTD("elttype=%d",obj->a.a_elttype);
	  PUTD("rank=%d",obj->a.a_rank);
	  {int i;
	   if (obj->a.a_rank > 1)
	     {
	       for (i=0; i<obj->a.a_rank ; i++)
		 PUT4(obj->a.a_dims[i]);}}
	  for(i=0; i< leng ; i++)
	    write_fasd(aref(obj,i));}
      break;
	
      case DP(t_simple_vector:)
      case DP(t_vector:)
	TRY_HASH;
	PUT_OP(d_vector);
	{ int leng=VLEN(obj);
	  PUT4 (leng);
	  PUTD("eltype=%d",obj->v.v_elttype);
	  {int i;
	   for(i=0; i< leng ; i++)
	     {write_fasd(aref(obj,i));}}}
	break;
      
    
     default:
       PUT_OP(d_general_type);
       prin1(obj,current_fasd.stream);
       PUTD("close general:%c",')');
      
     }}
 }


static void
fasd_patch_sharp_cons(object x, int depth)
{
	for (;;) {
		x->c.c_car = fasd_patch_sharp(x->c.c_car,depth+1);
		if (consp(x->c.c_cdr))
			x = x->c.c_cdr;
		else {
                        x->c.c_cdr = SAFE_CDR(fasd_patch_sharp(x->c.c_cdr,depth+1));
			break;
		}
	}
}

static object
fasd_patch_sharp(object x, int depth)
{
	cs_check(x);
	if (++depth > 1000)
	  { object *p = current_fasd.table->v.v_self;
	    while(*p)
	      { if (x== *p++ && type_of(x)!=t_spice) return x;}}
    /* eval'd forms are already patched, and they might contain
      circular structure */
	{ object p = current_fasd.evald_items;
	  while (p != Cnil)
	    { if (p->c.c_car == x) return x;
	      p = p->c.c_cdr;}}

	switch (type_of(x)) {
	case DP(t_spice:)
	{  if (x->spc.spc_dummy >=  current_fasd.table->v.v_dim)
	     FEerror("bad spice ref",0);
	   return  current_fasd.table->v.v_self[x->spc.spc_dummy ];

	}
	case DP(t_cons:)
	/*
		x->c.c_car = fasd_patch_sharp(x->c.c_car,depth);
		x->c.c_cdr = fasd_patch_sharp(x->c.c_cdr,depth);
	*/
		fasd_patch_sharp_cons(x,depth);
		break;

	case DP(t_simple_vector:)
	case DP(t_vector:)
	{
		int i;

		if ((enum aelttype)x->v.v_elttype != aet_object)
		  break;

		for (i = 0;  i < VLEN(x);  i++)
			x->v.v_self[i] = fasd_patch_sharp(x->v.v_self[i],depth);
		break;
	}
	case DP(t_array:)
	{
		int i, j;
		
		if ((enum aelttype)x->a.a_elttype != aet_object)
		  break;

		for (i = 0, j = 1;  i < x->a.a_rank;  i++)
			j *= x->a.a_dims[i];
		for (i = 0;  i < j;  i++)
			x->a.a_self[i] = fasd_patch_sharp(x->a.a_self[i],depth);
		break;
	}
	case DP(t_structure:)
	{object def = x->str.str_def;
	 int i;
	 i=S_DATA(def)->length;
	 while (i--> 0)
	   structure_set(x,def,i,fasd_patch_sharp(structure_ref(x,def,i),depth));
	 break;
      
       }
          default:
             /* dont have to walk other objs */
           break;
	
	}
	return(x);
}

object sharing_table;

DEFUN("FIND-SHARING-TOP",object,fSfind_sharing_top,SI
	  ,2,2,NONE,OO,OO,OO,OO,(object x,object table),"") {

  sharing_table=table;
  travel_find_sharing(x,table);
  RETURN1(Ct);

}

static object
lisp_eval(object x) {

  SAVE_CURRENT_FASD;
  x=ieval(x);
  RESTORE_FASD;

  return x;

}

#define CHECK_CH(i)    	   do{if ((i)==EOF && stream_at_end(fas_stream)) bad_eof();}while (0)
/* grow vector AR of general type */
static void
grow_vector(object ar)
{   int len=ar->v.v_dim;
    int nl=(int) (1.5*(len+1));
    {BEGIN_NO_INTERRUPT;
     {char *p= (char *)AR_ALLOC(alloc_contblock,nl,object);
    bcopy(ar->v.v_self,p,sizeof(object)* len);
    ar->v.v_self= (object *)p;
    ar->v.v_dim=nl;
    VSET_MAX_FILLP(ar);
    while(--nl >=len)
      ar->v.v_self[nl]=Cnil;
    END_NO_INTERRUPT;}}
  }

static void
bad_eof(void)
{  FEerror("Unexpected end of file",0);}



/* read one starting with byte i into location loc */
static void
read_fasd1(int i, object *loc)
{  object tem;
   int leng;
 BEGIN:
   CHECK_CH(i);
   switch(D_TYPE_OF(i))
     {case DP(d_nil:)
	*loc=Cnil;return;
      case DP(d_cons:)
	read_fasd1(GET_OP(),&tem);
        collect(loc,make_cons(tem,Cnil));
	i=GET_OP();
	goto BEGIN;
      case DP(d_list1:) i=1;goto READ_LIST;
      case DP(d_list2:) i=2;goto READ_LIST;
      case DP(d_list3:) i=3;goto READ_LIST;
      case DP(d_list4:) i=4;goto READ_LIST;
      case DP(d_list:)  i=(1<<30) ; goto READ_LIST;

      READ_LIST:
	while(1)
	  {int j;
	   if (--i < 0) {*loc=Cnil; return;}
	   j=GET_OP();
	   CHECK_CH(j);
	   if (j==d_delimiter)
	     {*loc=Cnil;
	      DPRINTF("{Read end of list(%d)}",i);
	      return;}
	   else
	     if(j==d_dot)
	       { DPRINTF("{Read end of dotted list(%d)}",i);
		 read_fasd1(GET_OP(),loc);
	    
		 return;}
	     else
	       {object tem;
		DPRINTF("{Read next item in list(%d)}",i);
		read_fasd1(j,&tem);
		DPRINTF("{Item=",(debug >= 2 ? pp(tem) : 0));
		DPRINTF("}",0);
		collect(loc,make_cons(tem,Cnil));}}

      case DP(d_delimiter:)
      case DP(d_dot:)
	FEerror("Illegal op at top level",0);
	break;
      case DP(d_eval_skip:)
	read_fasd1(GET_OP(),loc);
	MAYBE_PATCH(*loc);
	lisp_eval(*loc);
	read_fasd1(GET_OP(),loc);
	break;

      case d_reserve1:
      case d_reserve2:
      case d_reserve3:
      case d_reserve4:
       
	FEerror("Op reserved for future use",0);
	break;

      case DP(d_reset_index:)
	dump_index=0;
	break;
       
      case DP(d_short_symbol:)
	leng=GETD("leng=%d");
	leng = LENGTH(i,leng);
	READ_SYMBOL(leng,current_fasd.package,tem);
	*loc=tem;
	return ;
      case DP(d_short_symbol_and_package:)
	{object pack;
	 leng=GETD("leng=%d");
	 leng = LENGTH(i,leng);
	 read_fasd1(GET_OP(),&pack);
	 READ_SYMBOL(leng,pack,tem);
	 *loc=tem;
	 return;}
      case DP(d_short_string:)
	leng=GETD("leng=%d");
	leng = LENGTH(i,leng);
	READ_STRING(leng,loc);
	return;
      case DP(d_string:)
	{int j;
	 GET4(j);
	 READ_STRING(j,loc);
	 return;}
      
      case DP(d_indexed_item3:)
	GET3(i);goto INDEXED;
      case DP(d_indexed_item2:)
	GET2(i);goto INDEXED;
      case DP(d_indexed_item1:)
	i=GET()+ NUMBER_ZERO_ITEMS ; goto INDEXED;
      default:
      case DP(d_indexed_item0:)
	i = i - (int) d_indexed_item0; goto INDEXED;

      INDEXED:	
	  
	*loc= current_fasd.table->v.v_self[i];
	/* if object not yet built make pointer to it */
	if(*loc==0)
	  {*loc=current_fasd.table->v.v_self[i]= alloc_object(t_spice);
	   (*loc)->spc.spc_dummy= i;
	   needs_patching=1;}
	return;

	/* the item`s' case does not return a value but is simply
	   a facility to allow convenient dumping of a list of registers
	   at the beginning, follwed by a delimiter.   read continues on. */

      case DP(d_new_indexed_items:)
      case DP(d_new_indexed_item:)

	{
	 int cindex,k;
	 k=GET_OP();
       MORE:
	 cindex =dump_index;
	 DPRINTF("{dump_index=%d}",dump_index);
	 if (dump_index >= current_fasd.table->v.v_dim)
	   grow_vector(current_fasd.table);
	 /* grow the array */
	 current_fasd.table->v.v_self[dump_index++] = 0;
	 read_fasd1(k,loc);
	 current_fasd.table->v.v_self[cindex] = *loc;
	   
	 if (i==d_new_indexed_items)
	   {int k=GET_OP();
	    if (k==d_delimiter)
	      { DPRINTF("{Reading last of new indexed items}",0);
		read_fasd1(GET_OP(),loc);
		return;}
	    else { 
	      goto MORE;
	    }}
	 return;
       }
      case DP(d_short_fixnum:)
	{int leng=GETD("n=%d");
	 if (leng & (1 << (SIZE_SHORT -1)))
	   leng= leng - (1 << (SIZE_SHORT));
	 *loc=SAFE_CDR(make_fixnum(leng));
	 return;}
    
      case DP(d_fixnum:)
	{fixnum j;
	 GETFIX(j);
	 *loc=SAFE_CDR(make_fixnum(j));
	 return;}
      case DP( d_bignum:)
	{int j,m;
	 object tem;
	 mp_limb_t *u;
	 GET4(j);
#ifdef GMP
	 tem = new_bignum();
	 m = (j >= 0 ? j : -j);
	 _mpz_realloc(MP(tem),m);
	 MP(tem)->_mp_size = j;
	 j = m;
	 u = MP(tem)->_mp_d;
#else	 
        { BEGIN_NO_INTERRUPT;
	 tem = alloc_object(t_bignum);
	 tem->big.big_length = j;
	 tem-> big.big_self = 0;
	 u = tem-> big.big_self = (plong *) alloc_relblock(j*sizeof(plong));
	   END_NO_INTERRUPT;
	 }
	
#endif	 
	while ( --j >=0) {
#if MP_LIMB_BYTES == 8
	    GET8(*u);
#elif MP_LIMB_BYTES == 4
	    GET4(*u);
#else
#error Bad MP_LIMB_BYTES
#endif
	  u++;
	}
	*loc=tem; return;}
     case DP(d_objnull:)

	*loc=0; return;

      case DP(d_structure:)
	{ int narg,i;
          object name;
          narg=GETD("narg=%d");
          read_fasd1(GET_OP(),& name);
          { object *base=vs_top;
	    object *p = base;
	    vs_base=base;
	    vs_top = base + 1 + narg;
	    *p++ = name;
	    for (i=0; i < narg ; i++)
	      read_fasd1(GET_OP(),p++);
	    vs_base=base;
	    vs_top = p;
	    funcall(find_symbol(str("MAKE-STRUCTURE"),system_package));
	    /* siLmake_structure(); */
	    *loc = vs_base[0];
	    vs_top=vs_base=base;
	    return;
	  }}

      case DP(d_symbol:)
	{int i; object tem;
	 GET4(i);
	 READ_SYMBOL(i,current_fasd.package,tem);
	 *loc=tem;
	 return ;}
      case DP(d_symbol_and_package:)
	{int i; object pack;
	 GET4(i);  
	 read_fasd1(GET_OP(),&pack);
	 READ_SYMBOL(i,pack,*loc);
	 return;}
      case DP(d_package:)
	{object pack,tem;
	 read_fasd1(GET_OP(),&tem);
	 pack=find_package(tem);
	 if (pack==Cnil) FEerror("The package named ~a, does not exist",1,tem);
	 *loc=pack;
	 return ;}
      case DP(d_standard_character:)
	*loc=(code_char(GETD("char=%c")));
	return;
      case DP(d_vector:)
	{int leng,j;
	 object y;
	 object x;
	 GET4(leng);
	 {
	   enum aelttype tp=GETD("v_elttype=%d");
	   x= tp==aet_object ? alloc_simple_vector(leng) : alloc_vector(leng,tp);
	 }
	 array_allocself(x,0,Cnil);
	 for (j=0; j< leng ; j++)
	   { DPRINTF("{vector_elt=%d}",j);
	     read_fasd1(GET_OP(),&y);
	     aset(x,j,y);}
	 *loc=x;
	 DPRINTF("{End of length %d vector}",leng);
	 return;}


      case DP(d_array:)
	{BEGIN_NO_INTERRUPT;

	{int leng,i;
	 object y;
	 object x=alloc_object(t_array);
	 GET4(leng);

	 set_array_elttype(x,GETD("a_elttype=%d"));
	 x->a.a_dim=leng;
	 x->a.a_hasfillp=1;
	 x->a.a_rank= GETD("a_rank=%d");
	 x->a.a_self=0;
	 x->a.a_adjustable=1;
	 SET_ADISP(x,Cnil);
	 if (x->a.a_rank > 0)
	   { x->a.a_dims = (ufixnum *)alloc_relblock(sizeof(fixnum)*(x->a.a_rank)); }
	 for (i=0; i< x->a.a_rank ; i++)
	   GET4(x->a.a_dims[i]);
	 array_allocself(x,0,Cnil);
	 END_NO_INTERRUPT;
	 for (i=0; i< leng ; i++)
	   { read_fasd1(GET_OP(),&y);
	     aset(x,i,y);}
	 *loc=x;
	 return;}}
	
      case DP(d_end_of_file:)
	current_fasd.direction =Cnil;
	*loc=current_fasd.eof;
	return;

      case DP(d_begin_dump:)
	{int vers=GETD("version=%d");
	if(vers!=FASD_VERSION) {
	  object x,x1;
	  x=make_fixnum(vers);
	  x1=make_fixnum(FASD_VERSION);
	  FEerror("This file was dumped with FASD version ~a not ~a.",
		  2,x,x1);}}
	{int leng;
	 GET4(leng);
	 current_fasd.table_length=make_fixnum(leng);}
	read_fasd1(GET_OP(),&tem);
        if (type_of(tem)==t_package || tem==Cnil)
	  {current_fasd.package = tem;
	   *loc=current_fasd.table;}
	else FEerror("expected package",0);
	return;
	
      case DP(d_general_type:)
	*loc=read_object_non_recursive(current_fasd.stream);
	if(GETD("close general:%c")!=')') FEerror("general type not followed by ')'",0);
	return;
      

	/* Special type, the forms have been sharp patched separately
	   It is also arranged that it does not 
	   */
	 
      case DP(d_enter_vector:)
	{
	 extern object sSPmemory;
	 int print_only=0;
	 int n = 0;
	 object vv = sSPmemory->s.s_dbind,tem;
	 if (vv == Cnil) print_only = 1;
	 else
	   if (type_of(vv)!=t_cfdata) FEerror("bad VectorToEnter",0);
	 while ((i=GET_OP()) !=d_delimiter)
	   {int eval=(i==d_eval_skip);
	    if (print_only)
	      { if (eval) princ_str("#!",Ct);
		else if (i== d_eval)
		  princ_str("#.",Ct);}
	    if(eval) i=GET_OP();
	    read_fasd1(i, &tem);
	    MAYBE_PATCH(tem);
	    /* the eval entries don't enter it */

	    if (print_only) {princ(tem,Ct);
			     princ_str(";",Ct);
			     princ(make_fixnum(n),Ct);
			     if (eval==0) n++;
			     princ_str("\n",Ct);}
	    else
	      {
	      if(eval)
		lisp_eval(tem);
	      else
		{if (n >= vv->cfd.cfd_fillp) FEerror("cfd too small",0);
		 vv->cfd.cfd_self[n++]=tem;}}}
	 if (print_only==0) vv->cfd.cfd_fillp = n;
	 *loc=vv;
	 return;
       }

      case DP(d_eval:)
	{object tem;
	 read_fasd1(GET_OP(),&tem);
	 MAYBE_PATCH(tem);
	 *loc = lisp_eval(tem);
	 current_fasd.evald_items = make_cons(*loc,current_fasd.evald_items);
	 return;
       }
	
      }}
       
static void
clrhash(object table)
{int i;
   if (table->ht.ht_nent > 0 )
     for(i = 0; i < table->ht.ht_size; i++) {
       table->ht.ht_self[i].c_cdr = OBJNULL;
       table->ht.ht_self[i].c_car = OBJNULL;}
   table->ht.ht_nent =0;}

object IfaslInStream;
/* static void */
/* IreadFasdData(void) */

   /* While  executing this the  siPMemory should be  bound to the cfdata
   and the sSPinit to a vector of addresses. */
/* {object ar=open_fasd(IfaslInStream,sKinput,0,Cnil); */
/*   int n=fix(current_fasd.table_length); */
/*   object result; */
/*  {BEGIN_NO_INTERRUPT; */
/* #ifdef HAVE_ALLOCA */
/*   current_fasd.table->v.v_self */
/*     = (object *)alloca(n*sizeof(object)); */
/* #else */
/*   current_fasd.table->v.v_self */
/*     = (object *)alloc_relblock(n*sizeof(object)); */
/* #endif */
/*   current_fasd.table->v.v_dim=n; */
/*   current_fasd.table->v.v_fillp=n; */
/*   gset( current_fasd.table->v.v_self,0,n,aet_object); */
/*   END_NO_INTERRUPT; */
/* } */
/*   result=read_fasd_top(ar); */
 /* make sure there is nothing still pointing into the stack */
/*   current_fasd.table->v.v_self = 0; */
/*    current_fasd.table->v.v_dim=0; */
/*   current_fasd.table->v.v_fillp=0; */

/* } */

 


static void
init_fasdump(void)
{
/*   make_si_sfun("READ-FASD-TOP",read_fasd_top,1); */
/*   make_si_sfun("WRITE-FASD-TOP",write_fasd_top,2); */
/*   make_si_sfun("OPEN-FASD",open_fasd,4);   */
/*   make_si_sfun("CLOSE-FASD",close_fasd,1); */
/*  make_si_sfun("FASD-I-DATA",fasd_i_macro,1); */
/*   make_si_sfun("FIND-SHARING-TOP",find_sharing_top,2); */
}
