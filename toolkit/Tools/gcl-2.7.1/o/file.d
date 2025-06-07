/* -*-C-*- */
/*
 Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
 Copyright (C) 2024 Camm Maguire

This file is part of GNU Common Lisp, herein referred to as GCL

GCL is free software; you can redistribute it and/or modify it under
the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GCL is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
License for more details.

You should have received a copy of the GNU Library General Public License 
along with GCL; see the file COPYING.  If not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
*/

/*
	file.d
	IMPLEMENTATION-DEPENDENT

	The specification of printf may be dependent on the C library,
	especially for read-write access, append access, etc.
	The file also contains the code to reclaim the I/O buffer
	by accessing the FILE structure of C.
	It also contains read_fasl_data.
*/

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#define IN_FILE
#include "include.h"

#ifdef USE_READLINE
#include <readline/readline.h>
#define kclgetc(FP)		rl_getc_em(FP)
#define kclungetc(C, FP)	rl_ungetc_em(C, FP)
#define kclputc(C, FP)		rl_putc_em(C, FP)
#else
#define	kclgetc(FP)		getc(FP)
#define	kclungetc(C, FP)	ungetc(C, FP)
#define	kclputc(C, FP)		putc(C, FP)
#endif /* USE_READLINE */

#define	xkclfeof(c,FP)		feof(((FILE *)FP))

#ifdef HAVE_AOUT
#undef ATT
#undef BSD
#ifndef HAVE_ELF
#ifndef HAVE_FILEHDR
#define BSD
#endif
#endif
#include HAVE_AOUT
#endif

#ifdef ATT
#include <filehdr.h>
#include <syms.h>
#define HAVE_FILEHDR
#endif

#ifdef E15
#include <a.out.h>
#define exec	bhdr
#define a_text	tsize
#define a_data	dsize
#define a_bss	bsize
#define a_syms	ssize
#define a_trsize	rtsize
#define a_drsize	rdsize
#endif

#if defined(HAVE_ELF_H)
#include <elf.h>
#elif defined(HAVE_ELF_ABI_H)
#include <elf_abi.h>
#endif

#ifndef __MINGW32__
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#else
#  include <winsock2.h>
#  include <windows.h>
#endif
#include <errno.h>

extern void tcpCloseSocket (int fd);

object terminal_io;

object Vverbose;
object LSP_string;


object sSAignore_eof_on_terminal_ioA;

static bool
feof1(fp)
FILE *fp;
{

#ifdef USE_READLINE
  if (readline_on && fp==rl_instream && rl_line_buffer && *rl_line_buffer==EOF)
    return TRUE;
#endif
  if (!feof(fp))
    return(FALSE);
  if (fp == terminal_io->sm.sm_object0->sm.sm_fp) {
     if (symbol_value(sSAignore_eof_on_terminal_ioA) == Cnil)
	return(TRUE);
     fp = freopen("/dev/tty", "r", fp);
     if (fp == NULL)
	error("can't reopen the console");
     return(FALSE);
  }
  return(TRUE);
}

#undef	feof
#define	feof	feof1

void
end_of_stream(object strm) {
  END_OF_FILE(strm);
}

DEFUN("TEMP-STREAM",object,fStemp_stream,SI,2,2,NONE,OO,OO,OO,OO,(object x,object ext),"") {
  
  object st;
#ifdef _WIN32
  DWORD dwRetVal;
  char lpPathBuffer[MAX_PATH];
  
  check_type_string ( &x );
  check_type_string ( &ext );
  
  dwRetVal = GetTempPath ( MAX_PATH, lpPathBuffer );
  if ( dwRetVal + VLEN(ext) + VLEN(x) + 2 > MAX_PATH ) {
    FEerror ( "Length of temporary file path combined with file name is too large.", 0 );
  }
  
  strcat ( lpPathBuffer, x->st.st_self );
  strcat ( lpPathBuffer, "." );
  strcat ( lpPathBuffer, ext->st.st_self );
  st = make_simple_string ( lpPathBuffer );
  x  = open_stream ( st, smm_io, sKsupersede, Cnil );
  
#else
  char *c, *d;
  int l;
  check_type_string(&x);
  check_type_string(&ext);
  
  if (!(c=alloca(VLEN(x)+VLEN(ext)+8)))
    FEerror("Cannot allocate temp name space",0);
  if (!(d=alloca(VLEN(x)+VLEN(ext)+8)))
    FEerror("Cannot allocate temp name space",0);
  memcpy(c,x->st.st_self,VLEN(x));
  memcpy(c+VLEN(x),"XXXXXX",6);
  c[VLEN(x)+6]=0;
  l=mkstemp(c);
  
  memcpy(d,c,VLEN(x)+6);
  memcpy(d+VLEN(x)+6,".",1);
  memcpy(d+VLEN(x)+7,ext->st.st_self,VLEN(ext));
  d[VLEN(x)+VLEN(ext)+7]=0;
  if (rename(c,d))
    FEerror("Cannot rename ~s to ~s",2,make_simple_string(c),make_simple_string(d));
  st=make_simple_string(d);
  x=open_stream(st,smm_output,sKsupersede,Cnil);
  close(l);
#endif
  
  RETURN1(x);
  
}

DEFUN("TERMINAL-INPUT-STREAM-P",object,fSterminal_input_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(type_of(x)==t_stream && x->sm.sm_mode==smm_input && x->sm.sm_fp && isatty(fileno(x->sm.sm_fp)) ? Ct : Cnil);
}


/*
	Input_stream_p(strm) answers
	if stream strm is an input stream or not.
	It does not check if it really is possible to read
	from the stream,
	but only checks the mode of the stream (sm_mode).
*/
static bool
input_stream_p(strm)
object strm;
{
BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_input:
		return(TRUE);

	case smm_output:
		return(FALSE);

	case smm_io:
	case smm_socket:  
		return(TRUE);

	case smm_probe:
		return(FALSE);

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_broadcast:
		return(FALSE);

	case smm_concatenated:
		return(TRUE);

	case smm_two_way:
		return(TRUE);

	case smm_echo:
		return(TRUE);

	case smm_string_input:
		return(TRUE);

	case smm_string_output:
		return(FALSE);

	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(FALSE);
	}
}

/*
	Output_stream_p(strm) answers
	if stream strm is an output stream.
	It does not check if it really is possible to write
	to the stream,
	but only checks the mode of the stream (sm_mode).
*/
static bool
output_stream_p(strm)
object strm;
{
BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_input:
		return(FALSE);

	case smm_output:
		return(TRUE);

	case smm_io:
	case smm_socket:  
		return(TRUE);

	case smm_probe:
		return(FALSE);

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_broadcast:
		return(TRUE);

	case smm_concatenated:
		return(FALSE);

	case smm_two_way:
		return(TRUE);

	case smm_echo:
		return(TRUE);

	case smm_string_input:
		return(FALSE);

	case smm_string_output:
		return(TRUE);

	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(FALSE);
	}
}

static object
stream_element_type(strm)
object strm;
{
	object x;

BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_input:
	case smm_output:
	case smm_io: 
	case smm_probe:
		return(strm->sm.sm_object0);

	case smm_socket:
	    return (sLcharacter);
	    
	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_broadcast:
		x = strm->sm.sm_object0;
		if (endp(x))
			return(Ct);
		return(stream_element_type(x->c.c_car));

	case smm_concatenated:
		x = strm->sm.sm_object0;
		if (endp(x))
			return(Ct);
		return(stream_element_type(x->c.c_car));

	case smm_two_way:
		return(stream_element_type(STREAM_INPUT_STREAM(strm)));

	case smm_echo:
		return(stream_element_type(STREAM_INPUT_STREAM(strm)));

	case smm_string_input:
		return(sLcharacter);

	case smm_string_output:
		return(sLcharacter);

	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(FALSE);
	}
}

void
setup_stream_buffer(object x) {
#ifdef NO_SETBUF
  massert(!setvbuf(x->sm.sm_fp,x->sm.sm_buffer=NULL,_IONBF,0));
#else
  massert(!setvbuf(x->sm.sm_fp,x->sm.sm_buffer=writable_malloc_wrap(malloc,void *,BUFSIZ),_IOFBF,BUFSIZ));
#endif
}	

static void
deallocate_stream_buffer(object strm) {

  if (strm->sm.sm_buffer==NULL)
    return;

  free(strm->sm.sm_buffer);

  massert(!setvbuf(strm->sm.sm_fp,strm->sm.sm_buffer=NULL,_IONBF,0));

}

DEFVAR("*ALLOW-GZIPPED-FILE*",sSAallow_gzipped_fileA,SI,sLnil,"");

static void
cannot_open(object);
static void
cannot_create(object);
/*
	Open_stream(fn, smm, if_exists, if_does_not_exist)
	opens file fn with mode smm.
	Fn is a namestring.
*/
object
open_stream(object fn,enum smmode smm, object if_exists, object if_does_not_exist) {

  object x;
  FILE *fp=NULL;
  vs_mark;

  coerce_to_filename(fn,FN1);

  switch(smm) {

  case smm_input:
  case smm_probe:

    if (!(fp=*FN1=='|' ? popen(FN1+1,"r") : fopen_not_dir(FN1,"r")) && sSAallow_gzipped_fileA->s.s_dbind!=Cnil) {

      struct stat ss;
      massert(snprintf(FN2,sizeof(FN2),"%s.gz",FN1)>0);

      if (!stat(FN2,&ss)) {

	FILE *pp;
	int n;

	massert((fp=tmpfile()));
	massert(snprintf(FN3,sizeof(FN2),"zcat %s",FN2)>0);
	massert(pp=popen(FN3,"r"));
	while ((n=fread(FN4,1,sizeof(FN3),pp)))
	  massert(fwrite(FN4,1,n,fp)==n);
	massert(pclose(pp)>=0);
	massert(!fseek(fp,0,SEEK_SET));

      }

    }

    if (!fp) {

      if (if_does_not_exist==sKerror) cannot_open(fn);
      else if (if_does_not_exist==sKcreate) {
	if (!(fp=fopen_not_dir(FN1,"w"))) cannot_create(fn);
	fclose(fp);
	if (!(fp=fopen_not_dir(FN1,"r"))) cannot_open(fn);
      } else if (if_does_not_exist==Cnil) return(Cnil);
      else FEerror("~S is an illegal IF-DOES-NOT-EXIST option.",1,if_does_not_exist);

    }
    break;

  case smm_output:
  case smm_io:

    if ((fp=*FN1=='|' ? NULL : fopen_not_dir(FN1,"r"))) {

      fclose(fp);
      if (if_exists==sKerror) FILE_ERROR(fn,"File exists");
      else if (if_exists==sKrename) {
	massert(snprintf(FN2,sizeof(FN2),"%-*.*s~",(int)strlen(FN1)-1,(int)strlen(FN1)-1,FN1)>=0);
	unlink(FN2);/*MinGW*/
	massert(!rename(FN1,FN2));
	if (!(fp=fopen(FN1,smm==smm_output ? "w" : "w+"))) cannot_create(fn);
      } else if (if_exists==sKrename_and_delete ||
		 if_exists==sKnew_version ||
		 if_exists==sKsupersede) {
	if (!(fp=fopen(FN1,smm==smm_output ? "w" : "w+"))) cannot_create(fn);
      } else if (if_exists==sKoverwrite) {
	if (!(fp=fopen_not_dir(FN1,"r+"))) cannot_open(fn);
      } else if (if_exists==sKappend) {
	if (!(fp = fopen_not_dir(FN1,smm==smm_output ? "a" : "a+")))
	  FEerror("Cannot append to the file ~A.",1,fn);
      } else if (if_exists == Cnil) return(Cnil);
      else FEerror("~S is an illegal IF-EXISTS option.",1,if_exists);

    } else {

      if (if_does_not_exist == sKerror)
	FILE_ERROR(fn,"The file does not exist");
      else if (if_does_not_exist == sKcreate) {
	if (!(fp=smm==smm_output ? (*FN1=='|' ? popen(FN1+1,"w") : fopen_not_dir(FN1, "w")) : fopen_not_dir(FN1, "w+")))
	  cannot_create(fn);
      } else if (if_does_not_exist==Cnil) return(Cnil);
      else FEerror("~S is an illegal IF-DOES-NOT-EXIST option.",1,if_does_not_exist);
    }
    break;

  default:
    FEerror("Illegal open mode for ~S.",1,fn);
    break;
  }

  vs_push(make_simple_string(FN1));
  x = alloc_object(t_stream);
  x->sm.tt=x->sm.sm_mode = (short)smm;
  x->sm.sm_fp = fp;
  x->sm.sm_buffer = 0;
  x->sm.sm_object0 = sLcharacter;
  x->sm.sm_object1 = vs_head;
  x->sm.sm_int = 0;
  x->sm.sm_flags=0;
  vs_push(x);

  setup_stream_buffer(x);
  vs_reset;

  if (smm==smm_probe)
    close_stream(x);

  return(x);

}

static void
gclFlushSocket(object);

DEFUN("OPEN-STREAM-P",object,fLopen_stream_p,LISP,1,1,NONE,OO,OO,OO,OO,(object x),"") {

  check_type_stream(&x);

  return GET_STREAM_FLAG(x,gcl_sm_closed) ? Cnil : Ct;

}
    /*
	Close_stream(strm) closes stream strm.
	The abort_flag is not used now.
*/

static int
pipe_designator_p(object x) {

  if (x==OBJNULL||x==Cnil)
    return 0;
  coerce_to_filename(x,FN1);
  return FN1[0]=='|' ? 1 : 0;

}

void
close_stream(object strm)  {

  if (FFN(fLopen_stream_p)(strm)==Cnil)
    return;

  switch (strm->sm.sm_mode) {
  case smm_output:
    if (strm->sm.sm_fp == stdout || strm->sm.sm_fp == stderr)
      FEerror("Cannot close the standard output.", 0);
    fflush(strm->sm.sm_fp);
    deallocate_stream_buffer(strm);
    if (pipe_designator_p(strm->sm.sm_object1))
      pclose(strm->sm.sm_fp);
    else
      fclose(strm->sm.sm_fp);
    strm->sm.sm_fp = NULL;
    strm->sm.sm_fd = -1;
    break;

  case smm_socket:
    if (SOCKET_STREAM_FD(strm) < 2)
      emsg("tried Closing %d ! as socket \n",SOCKET_STREAM_FD(strm));
    else {
#ifdef HAVE_NSOCKET
      if (GET_STREAM_FLAG(strm,gcl_sm_output)) {
	gclFlushSocket(strm);
	/* there are two for one fd so close only one */
	tcpCloseSocket(SOCKET_STREAM_FD(strm));
      }
#endif
      SOCKET_STREAM_FD(strm)=-1;
    }

  case smm_input:
    if (strm->sm.sm_fp == stdin)
      FEerror("Cannot close the standard input.", 0);

  case smm_io:
  case smm_probe:
    if (strm->sm.sm_fp == NULL) break; /*FIXME: review this*/
    deallocate_stream_buffer(strm);
    if (pipe_designator_p(strm->sm.sm_object1))
      pclose(strm->sm.sm_fp);
    else
      fclose(strm->sm.sm_fp);
    strm->sm.sm_fp = NULL;
    strm->sm.sm_fd = -1;
    break;

  case smm_file_synonym:
  case smm_synonym:
  case smm_broadcast:
  case smm_concatenated:
    strm->sm.sm_object0=OBJNULL;
    break;

  case smm_two_way:
  case smm_echo:
    STREAM_INPUT_STREAM(strm)=OBJNULL;
    STREAM_OUTPUT_STREAM(strm)=OBJNULL;
    /* 		close_stream(STREAM_INPUT_STREAM(strm)); */
    /* 		close_stream(STREAM_OUTPUT_STREAM(strm)); */
    break;

  case smm_string_input:
  case smm_string_output:
    STRING_STREAM_STRING(strm)=OBJNULL;
    break;

  default:
    error("Illegal stream mode");
  }

  SET_STREAM_FLAG(strm,gcl_sm_closed,1);

}

DEFUN("INTERACTIVE-STREAM-P",object,fLinteractive_stream_p,LISP,1,1,NONE,OO,OO,OO,OO,(object strm),"") {

  check_type_stream(&strm);

  while(type_of(strm)==t_stream)
    switch (strm->sm.sm_mode) {
    case smm_output:
    case smm_input:
    case smm_io:
    case smm_probe:
      if ((strm->sm.sm_fp == stdin) ||
	  (strm->sm.sm_fp == stdout) ||
	  (strm->sm.sm_fp == stderr))
	return Ct;
      return Cnil;
      break;
    case smm_file_synonym:
    case smm_synonym:
      strm = symbol_value(strm->sm.sm_object0);
      if (type_of(strm) != t_stream)
	FEwrong_type_argument(sLstream, strm);
      break;

    case smm_broadcast:
    case smm_concatenated:
      if (( consp(strm->sm.sm_object0) ) &&
	  ( type_of(strm->sm.sm_object0->c.c_car) == t_stream ))
	strm=strm->sm.sm_object0->c.c_car;
      else
	return Cnil;
      break;

    case smm_two_way:
    case smm_echo:
      strm=STREAM_INPUT_STREAM(strm);
      break;
    default:
      return Cnil;
    }

  return Cnil;

}
#ifdef STATIC_FUNCTION_POINTERS
object
fLinteractive_stream_p(object x) {
  return FFN(fLinteractive_stream_p)(x);
}
#endif

object
make_two_way_stream(istrm, ostrm)
object istrm, ostrm;
{
	object strm;

	strm = alloc_object(t_stream);
	strm->sm.tt=strm->sm.sm_mode = (short)smm_two_way;
	strm->sm.sm_fp = NULL;
	strm->sm.sm_buffer = 0;
	STREAM_INPUT_STREAM(strm) = istrm;
	STREAM_OUTPUT_STREAM(strm) = ostrm;
	strm->sm.sm_int = 0;
	strm->sm.sm_flags=0;
	return(strm);
}

static object
make_echo_stream(istrm, ostrm)
object istrm, ostrm;
{
	object strm;

	strm = make_two_way_stream(istrm, ostrm);
	strm->d.tt=strm->sm.sm_mode = (short)smm_echo;
	return(strm);
}

DEFUN("MAKE-STRING-INPUT-STREAM-INT",object,fSmake_string_input_stream_int,SI,3,3,NONE,OO,II,OO,OO,
	  (object strng,fixnum istart,fixnum iend),"") {

  object strm;

  massert(type_of(strng)==t_string);
  strm = alloc_object(t_stream);
  strm->sm.tt=strm->sm.sm_mode = (short)smm_string_input;
  strm->sm.sm_fp = NULL;
  strm->sm.sm_buffer = 0;
  strm->sm.sm_flags=0;
  STRING_STREAM_STRING(strm) = strng;
  strm->sm.sm_object1 = OBJNULL;
  STRING_INPUT_STREAM_NEXT(strm)= istart;
  STRING_INPUT_STREAM_END(strm)= iend;

  RETURN1(strm);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSmake_string_input_stream_int(object x,fixnum y,fixnum z) {
  return FFN(fSmake_string_input_stream_int)(x,y,z);
}
#endif

static bool
tty_stream_p(object strm) {

  if (type_of(strm)!=t_stream)
    return(FALSE);

  switch (strm->sm.sm_mode) {
  case smm_input:
  case smm_output:
  case smm_io:
    return(strm->sm.sm_fp && isatty(fileno(strm->sm.sm_fp)) ? TRUE : FALSE);

  case smm_socket:
  case smm_probe:
  case smm_string_input:
  case smm_string_output:
    return(FALSE);

  case smm_broadcast:
  case smm_concatenated:
    {
      object x;
      for (x=strm->sm.sm_object0;!endp(x);x=x->c.c_cdr)
	if (!tty_stream_p(x->c.c_car))
	  return(FALSE);
      return(TRUE);
    }

  case smm_file_synonym:
  case smm_synonym:
    return(tty_stream_p(symbol_value(strm->sm.sm_object0)));

  case smm_two_way:
  case smm_echo:
    return(tty_stream_p(STREAM_INPUT_STREAM(strm)) && tty_stream_p(STREAM_OUTPUT_STREAM(strm)));

  default:
    FEerror("Illegal stream mode for ~S.",1,strm);
    return(FALSE);

  }

}

DEFUN("TTY-STREAM-P",object,fStty_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  return tty_stream_p(x)  ? Ct : Cnil;
}
DEFUN("STRING-INPUT-STREAM-P",object,fSstring_input_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  return type_of(x)==t_stream && x->sm.sm_mode == (short)smm_string_input ? Ct : Cnil;
}
DEFUN("STRING-OUTPUT-STREAM-P",object,fSstring_output_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  return type_of(x)==t_stream && x->sm.sm_mode == (short)smm_string_output ? Ct : Cnil;
}

object
make_string_output_stream(line_length)
int line_length;
{
	object strng, strm;
	vs_mark;

	strng = alloc_string(line_length);
	strng->st.st_fillp = 0;
	strng->st.st_self = alloc_relblock(line_length);
	strm = alloc_object(t_stream);
	strm->sm.tt=strm->sm.sm_mode = (short)smm_string_output;
	strm->sm.sm_fp = NULL;
	strm->sm.sm_buffer = 0;
	STRING_STREAM_STRING(strm) = strng;
	strm->sm.sm_object1 = OBJNULL;
	strm->sm.sm_int = 0;
	strm->sm.sm_flags=0;
	vs_reset;
	return(strm);
}

static object
get_output_stream_string(strm)
object strm;
{
	object strng;

	strng = copy_simple_string(STRING_STREAM_STRING(strm));
	STREAM_FILE_COLUMN(strm) = 0;
	STRING_STREAM_STRING(strm)->st.st_fillp = 0;
	return(strng);
}

static void
cannot_read(object);

static void
closed_stream(object);
int
readc_stream(strm)
object strm;
{
	int c;

BEGIN:
	switch (strm->sm.sm_mode) {
#ifdef HAVE_NSOCKET
	case smm_socket:
	  return (getCharGclSocket(strm,Ct));
#endif
	case smm_input:
	case smm_io:

		if (strm->sm.sm_fp == NULL)
			closed_stream(strm);
		c = kclgetc(strm->sm.sm_fp);
/* 		if (c == EOF) { */
/*         	  if (xkclfeof(c,strm->sm.sm_fp)) */
/* 			end_of_stream(strm); */
/* 		    else c = kclgetc(strm->sm.sm_fp); */
/* 		    if (c == EOF) end_of_stream(strm); */
/* 		} */
		
/* 		c &= 0377; */
		/* strm->sm.sm_int0++; */
		return(c==EOF ? c : (c&0377));

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_concatenated:
	CONCATENATED:
		if (endp(strm->sm.sm_object0)) {
			end_of_stream(strm);
		}
		if (stream_at_end(strm->sm.sm_object0->c.c_car)) {
			strm->sm.sm_object0
			= strm->sm.sm_object0->c.c_cdr;
			goto CONCATENATED;
		}
		c = readc_stream(strm->sm.sm_object0->c.c_car);
		return(c);

	case smm_two_way:
#ifdef UNIX
		if (strm == terminal_io)			
			flush_stream(STREAM_OUTPUT_STREAM(terminal_io));
#endif
		/* strm->sm.sm_int1 = 0; */
		strm = STREAM_INPUT_STREAM(strm);
		goto BEGIN;

	case smm_echo:
		c = readc_stream(STREAM_INPUT_STREAM(strm));
		if (ECHO_STREAM_N_UNREAD(strm) == 0)
			writec_stream(c, STREAM_OUTPUT_STREAM(strm));
		else
			--(ECHO_STREAM_N_UNREAD(strm));
		return(c);

	case smm_string_input:
		if (STRING_INPUT_STREAM_NEXT(strm)>= STRING_INPUT_STREAM_END(strm))
			end_of_stream(strm);
		return(STRING_STREAM_STRING(strm)->st.st_self
		       [STRING_INPUT_STREAM_NEXT(strm)++]);

	case smm_output:
	case smm_probe:
	case smm_broadcast:
	case smm_string_output:
		cannot_read(strm);
#ifdef USER_DEFINED_STREAMS
	case smm_user_defined:
#define STM_DATA_STRUCT 0
#define STM_READ_CHAR 1
#define STM_WRITE_CHAR 2
#define STM_UNREAD_CHAR 7
#define STM_FORCE_OUTPUT 4
#define STM_PEEK_CHAR 3
#define STM_CLOSE 5
#define STM_TYPE 6
#define STM_NAME 8
{object val;
		object *old_vs_base = vs_base;
		object *old_vs_top = vs_top;
		vs_base = vs_top;
		vs_push(strm);
		super_funcall(strm->sm.sm_object1->str.str_self[STM_READ_CHAR]);
		val = vs_base[0];
		vs_base = old_vs_base;
		vs_top = old_vs_top;
		if (type_of(val) == t_fixnum)
		  return (fix(val));
		if (type_of(val) == t_character)
		  return (char_code(val));
	      }

#endif

	default:	
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(0);
	}
}

int
rl_ungetc_em(int, FILE *);

void
unreadc_stream(int c, object strm) {
BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_socket:
#ifdef HAVE_NSOCKET
	  ungetCharGclSocket(c,strm);
	  return;
#endif
	case smm_input:
	case smm_io:

		if (strm->sm.sm_fp == NULL)
			closed_stream(strm);
		kclungetc(c, strm->sm.sm_fp);
		/* --strm->sm.sm_int0; */  /* use ftell now for position */
		break;

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_concatenated:
		if (endp(strm->sm.sm_object0))
			goto UNREAD_ERROR;
		strm = strm->sm.sm_object0->c.c_car;
		goto BEGIN;

	case smm_two_way:
		strm = STREAM_INPUT_STREAM(strm);
		goto BEGIN;

	case smm_echo:
		unreadc_stream(c, STREAM_INPUT_STREAM(strm));
		ECHO_STREAM_N_UNREAD(strm)++;
		break;

	case smm_string_input:
		if (STRING_INPUT_STREAM_NEXT(strm)<= 0)
			goto UNREAD_ERROR;
		--STRING_INPUT_STREAM_NEXT(strm);
		break;

	case smm_output:
	case smm_probe:
	case smm_broadcast:
	case smm_string_output:
		goto UNREAD_ERROR;

#ifdef USER_DEFINED_STREAMS
        case smm_user_defined:
		{object *old_vs_base = vs_base;
		 object *old_vs_top = vs_top;
		 vs_base = vs_top;
		 vs_push(strm);
		 /* if there is a file pointer and no define unget function,
                  * then call ungetc */
		 if ((strm->sm.sm_fp != NULL ) &&
		     strm->sm.sm_object1->str.str_self[STM_UNREAD_CHAR] == Cnil)
		   kclungetc(c, strm->sm.sm_fp);
		 else
		   super_funcall(strm->sm.sm_object1->str.str_self[STM_UNREAD_CHAR]);
		 vs_top = old_vs_top;
		 vs_base = old_vs_base;
	       }
		break;
#endif
	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
	}
	return;

UNREAD_ERROR:
	FEerror("Cannot unread the stream ~S.", 1, strm);
}

static void
putCharGclSocket(object,int);
int
rl_putc_em(int, FILE *);
static void
cannot_write(object);

object
output_stream(object strm) {

  switch (strm->sm.sm_mode) {
  case smm_output:
  case smm_io:
  case smm_socket:
  case smm_string_output:
  case smm_broadcast:
#ifdef USER_DEFINED_STREAMS
  case smm_user_defined:
#endif
    return strm;

  case smm_file_synonym:
  case smm_synonym:
    return output_stream(symbol_value(strm->sm.sm_object0));

  case smm_two_way:
  case smm_echo:
    return output_stream(STREAM_OUTPUT_STREAM(strm));

  default:
    cannot_write(strm);
    return Cnil;

  }

}

static void
adjust_stream_column(int c,object strm) {

  if (c == '\n')
    STREAM_FILE_COLUMN(strm) = 0;
  else if (c == '\t')
    STREAM_FILE_COLUMN(strm) = (STREAM_FILE_COLUMN(strm)&~07) + 8;
  else
    STREAM_FILE_COLUMN(strm)++;

}

static int
writec_socket_stream(int c,object strm) {

  adjust_stream_column(c,strm);
  if (strm->sm.sm_fd>=0)
    putCharGclSocket(strm,c);
  else if (!GET_STREAM_FLAG(strm,gcl_sm_had_error))
    closed_stream(strm);

  return c;

}

static int
writec_output_stream(int c,object strm) {

  adjust_stream_column(c,strm);
  if (strm->sm.sm_fp!=NULL)
    kclputc(c, strm->sm.sm_fp);
  else if (!GET_STREAM_FLAG(strm,gcl_sm_had_error))
    closed_stream(strm);

  return c;

}

static int
writec_string_output_stream(int c,object strm) {

  object x=STRING_STREAM_STRING(strm);

  adjust_stream_column(c,strm);

  if (x->st.st_fillp >= x->st.st_dim) {

    ufixnum j=x->st.st_dim * 2 + 16;
    char *p;

    if (!x->st.st_adjustable)
      FEerror("The string ~S is not adjustable.",1, x);

    p = (inheap((long)x->st.st_self) ? alloc_contblock : alloc_relblock)(j);
    memcpy(p,x->st.st_self,x->st.st_dim);
    x->st.st_dim=j;
    x->st.st_self=p;

    adjust_displaced(x);

  }

  x->st.st_self[x->st.st_fillp++] = c;

  return c;

}

static int
writec_broadcast_stream(int c,object strm) {
  object x;
  for (x = strm->sm.sm_object0; !endp(x); x = x->c.c_cdr)
    writec_stream(c, x->c.c_car);
  return c;
}

void *
writec_stream_fun(object strm) {
  switch (strm->sm.sm_mode) {
  case smm_output:
  case smm_io:
    return writec_output_stream;
  case smm_socket:
    return writec_socket_stream;
  case smm_broadcast:
    return writec_broadcast_stream;
  case smm_string_output:
    return writec_string_output_stream;
  default:
    return NULL;
  }
}

int
writec_stream(int c, object strm) {
	object x;
	char *p;

BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_output:
	case smm_io:
	case smm_socket:  
		/* strm->sm.sm_int0++; */
		if (c == '\n')
			STREAM_FILE_COLUMN(strm) = 0;
		else if (c == '\t')
			STREAM_FILE_COLUMN(strm) = (STREAM_FILE_COLUMN(strm)&~07) + 8;
		else
			STREAM_FILE_COLUMN(strm)++;
		if (strm->sm.sm_fp == NULL)
		  {
#ifdef HAVE_NSOCKET
		    if (strm->sm.sm_mode == smm_socket && strm->sm.sm_fd>=0)
		      putCharGclSocket(strm,c);
		    else
#endif
		      if (!GET_STREAM_FLAG(strm,gcl_sm_had_error))
			closed_stream(strm);
		  } else {
			
		    kclputc(c, strm->sm.sm_fp);
		  }

		break;

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_broadcast:
		for (x = strm->sm.sm_object0; !endp(x); x = x->c.c_cdr)
			writec_stream(c, x->c.c_car);
		break;

	case smm_two_way:
		/*  this should be on the actual streams 
                strm->sm.sm_int0++;
		if (c == '\n')
			strm->sm.sm_int1 = 0;
		else if (c == '\t')
			strm->sm.sm_int1 = (strm->sm.sm_int1&~07) + 8;
		else
			strm->sm.sm_int1++;
               */
		strm = STREAM_OUTPUT_STREAM(strm);
		goto BEGIN;

	case smm_echo:
		strm = STREAM_OUTPUT_STREAM(strm);
		goto BEGIN;

	case smm_string_output:
		/* strm->sm.sm_int0++; */
		if (c == '\n')
			STREAM_FILE_COLUMN(strm) = 0;
		else if (c == '\t')
			STREAM_FILE_COLUMN(strm) = (STREAM_FILE_COLUMN(strm)&~07) + 8;
		else
			STREAM_FILE_COLUMN(strm)++;
		x = STRING_STREAM_STRING(strm);
		if (x->st.st_fillp >= x->st.st_dim) {

		  ufixnum j=x->st.st_dim * 2 + 16;

		  if (!x->st.st_adjustable)
		    FEerror("The string ~S is not adjustable.",1, x);

		  p = (inheap((long)x->st.st_self) ? alloc_contblock : alloc_relblock)(j);
		  memcpy(p,x->st.st_self,x->st.st_dim);
		  x->st.st_dim=j;
		  x->st.st_self=p;

		  adjust_displaced(x);

		}
		x->st.st_self[x->st.st_fillp++] = c;
		break;

	case smm_input:
	case smm_probe:
	case smm_concatenated:
	case smm_string_input:
		cannot_write(strm);

#ifdef USER_DEFINED_STREAMS
	case smm_user_defined:
		{object *old_vs_base = vs_base;
		 object *old_vs_top = vs_top;
		 vs_base = vs_top;
		 vs_push(strm);
		 vs_push(code_char(c));
		 super_funcall(strm->sm.sm_object1->str.str_self[2]);
		 vs_base = old_vs_base;
		 vs_top = old_vs_top;
		 break;
	       }

#endif
	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
	}
	return(c);
}

void
flush_stream(object strm) {
	object x;

BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_output:
	case smm_io:
	  if (strm->sm.sm_fp == NULL)
	    closed_stream(strm);
	  fflush(strm->sm.sm_fp);
	  break;
	case smm_socket:
#ifdef HAVE_NSOCKET
		if (SOCKET_STREAM_FD(strm) >0)
		  gclFlushSocket(strm);
		else
#endif
		  closed_stream(strm);
		break;
	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_broadcast:
		for (x = strm->sm.sm_object0; !endp(x); x = x->c.c_cdr)
			flush_stream(x->c.c_car);
		break;

	case smm_echo:
	case smm_two_way:
		strm = STREAM_OUTPUT_STREAM(strm);
		goto BEGIN;


	case smm_string_output:
		break;

	case smm_input:
	case smm_probe:
	case smm_concatenated:
	case smm_string_input:
		FEerror("Cannot flush the stream ~S.", 1, strm);
#ifdef USER_DEFINED_STREAMS
        case smm_user_defined:
		{object *old_vs_base = vs_base;
		 object *old_vs_top = vs_top;
		 vs_base = vs_top;
		 vs_push(strm);
		 super_funcall(strm->sm.sm_object1->str.str_self[4]);
		 vs_base = old_vs_base;
		 vs_top = old_vs_top;
		break;
	       }

#endif

	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
	}
}


bool
stream_at_end(object strm) {
#define NON_CHAR -1000
	VOL int c = NON_CHAR;

BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_socket:  
	  listen_stream(strm);
	  if (SOCKET_STREAM_FD(strm)>=0)
	    return(FALSE);
	  else return(TRUE);	  
	case smm_io:
	case smm_input:
		if (strm->sm.sm_fp == NULL)
			closed_stream(strm);
		if (isatty(fileno((FILE *)strm->sm.sm_fp)) && !listen_stream(strm))
		  return(feof(strm->sm.sm_fp) ? TRUE : FALSE);
		{int prev_signals_allowed = signals_allowed;
	       AGAIN:
		signals_allowed= sig_at_read;
		c = kclgetc(strm->sm.sm_fp);
                /* blocking getchar for sockets */
             
                if (c == NON_CHAR) goto AGAIN; 
		signals_allowed=prev_signals_allowed;}
	       
		if (xkclfeof(c,strm->sm.sm_fp))
			return(TRUE);
		else {
			if (c>=0) kclungetc(c, strm->sm.sm_fp);
			return(FALSE);
		}

	case smm_output:
		return(FALSE);

	case smm_probe:
		return(FALSE);

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		check_stream(strm);
		goto BEGIN;

	case smm_broadcast:
		return(FALSE);

	case smm_concatenated:
	CONCATENATED:
		if (endp(strm->sm.sm_object0))
			return(TRUE);
		if (stream_at_end(strm->sm.sm_object0->c.c_car)) {
			strm->sm.sm_object0
			= strm->sm.sm_object0->c.c_cdr;
			goto CONCATENATED;
		} else
			return(FALSE);

	case smm_two_way:
#ifdef UNIX
		if (strm == terminal_io)				/**/
			flush_stream(terminal_io->sm.sm_object1);	/**/
#endif
		strm = STREAM_INPUT_STREAM(strm);
		goto BEGIN;

	case smm_echo:
		strm = STREAM_INPUT_STREAM(strm);
		goto BEGIN;

	case smm_string_input:
		if (STRING_INPUT_STREAM_NEXT(strm)>= STRING_INPUT_STREAM_END(strm))
			return(TRUE);
		else
			return(FALSE);

	case smm_string_output:
		return(FALSE);

#ifdef USER_DEFINED_STREAMS
        case smm_user_defined:
		  return(FALSE);
#endif
	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(FALSE);
	}
}


#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif


#ifdef LISTEN_USE_FCNTL
#include <fcntl.h>
#endif

bool
listen_stream(object strm) {

BEGIN:

	switch (strm->sm.sm_mode) {

#ifdef HAVE_NSOCKET
	case smm_socket:

	  if (SOCKET_STREAM_BUFFER(strm)->ust.ust_fillp>0) return TRUE;

	  /* { */
	  /*   fd_set fds; */
	  /*   struct timeval tv; */
	  /*   FD_ZERO(&fds); */
	  /*   FD_SET(SOCKET_STREAM_FD(strm),&fds); */
	  /*   memset(&tv,0,sizeof(tv)); */
	  /*   return select(SOCKET_STREAM_FD(strm)+1,&fds,NULL,NULL,&tv)>0 ? TRUE : FALSE; */
 	  /* } */
	  { int ch  = getCharGclSocket(strm,Cnil);
	   if (ch == EOF) return FALSE;
	   else unreadc_stream(ch,strm);
	   return TRUE;
	  }
#endif	   

	case smm_input:
	case smm_io:

#ifdef USE_READLINE
	  if (readline_on && strm->sm.sm_fp==rl_instream)
	    if (rl_line_buffer) return *rl_line_buffer ? TRUE : FALSE;
#endif
		if (strm->sm.sm_fp == NULL)
			closed_stream(strm);
		if (feof(strm->sm.sm_fp))
				return(FALSE);
#ifdef LISTEN_FOR_INPUT
		LISTEN_FOR_INPUT(strm->sm.sm_fp);
#else
#ifdef LISTEN_USE_FCNTL
  do { int c = 0;
  FILE *fp = strm->sm.sm_fp;
  int orig;
  int res;
  if (feof(fp)) return TRUE;
  orig = fcntl(fileno(fp), F_GETFL);
  if (! (orig & O_NONBLOCK ) ) {
    res=fcntl(fileno(fp),F_SETFL,orig | O_NONBLOCK);
  }
  c = getc(fp);
  if (! (orig & O_NONBLOCK ) ){
    fcntl(fileno(fp),F_SETFL,orig );
  }
  if (c != EOF)
    { 
      ungetc(c,fp);
      return TRUE;
    }
  return FALSE;
  } while (0);
#endif
#endif
		return TRUE;

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_concatenated:
	  {
	    object x;
	    for (x=strm->sm.sm_object0;!endp(x);x=x->c.c_cdr)
	      if (listen_stream(x->c.c_car))
		return TRUE;
	    return FALSE;
	  }
	  break;

	case smm_two_way:
	case smm_echo:
		strm = STREAM_INPUT_STREAM(strm);
		goto BEGIN;

	case smm_string_input:
		if (STRING_INPUT_STREAM_NEXT(strm)< STRING_INPUT_STREAM_END(strm))
			return(TRUE);
		else
			return(FALSE);

	case smm_output:
	case smm_probe:
	case smm_broadcast:
	case smm_string_output:
		FEerror("Can't listen to ~S.", 1, strm);
		return(FALSE);
	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(FALSE);
	}
}

int
file_position(strm)
object strm;
{
BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_input:
	case smm_output:
	case smm_io:
		/*  return(strm->sm.sm_int0);  */
		if (strm->sm.sm_fp == NULL)
			closed_stream(strm);
		return(ftell(strm->sm.sm_fp));
	case smm_broadcast:
	  for (strm=strm->sm.sm_object0;!endp(strm->c.c_cdr);strm=strm->c.c_cdr);
	  if (strm==Cnil)
	    return 0;
	  else {
	    strm=strm->c.c_car;
	    goto BEGIN;
	  }

	case smm_socket:
	   return -1;
	  

	case smm_string_output:
		return(STRING_STREAM_STRING(strm)->st.st_fillp);

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_probe:
	case smm_concatenated:
	case smm_two_way:
	case smm_echo:
	case smm_string_input:
		return(-1);

	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(-1);
	}
}

int
file_position_set(strm, disp)
object strm;
int disp;
{
BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_socket:
	  return -1;
	case smm_input:
	case smm_output:
	case smm_io:

		if (fseek(strm->sm.sm_fp, disp, 0) < 0)
			return(-1);
		/* strm->sm.sm_int0 = disp; */
		return(0);

	case smm_string_output:
		if (disp < STRING_STREAM_STRING(strm)->st.st_dim) {
			STRING_STREAM_STRING(strm)->st.st_fillp = disp;
			/* strm->sm.sm_int0 = disp; */
		} else {
			disp -= (STRING_STREAM_STRING(strm)->st.st_fillp=
				 STRING_STREAM_STRING(strm)->st.st_dim);
			while (disp-- > 0)
				writec_stream(' ', strm);
		}
		return(0);

	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_probe:
	case smm_broadcast:
	case smm_concatenated:
	case smm_two_way:
	case smm_echo:
	case smm_string_input:
		return(-1);

	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(-1);
	}
}

static int
file_length(strm)
object strm;
{
 BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_input:
	case smm_output:
	case smm_io:
	case smm_probe:
	  if (strm->sm.sm_fp == NULL)
	    closed_stream(strm);
	  return(file_len(strm->sm.sm_fp));
	  
	case smm_broadcast:
	  for (strm=strm->sm.sm_object0;!endp(strm->c.c_cdr);strm=strm->c.c_cdr);
	  if (strm==Cnil)
	    return 0;
	  else {
	    strm=strm->c.c_car;
	    goto BEGIN;
	  }

	case smm_file_synonym:
	case smm_synonym:

		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

	case smm_string_input:
	case smm_string_output:
	case smm_concatenated:
	case smm_two_way:
	case smm_echo:
	case smm_socket:
		return(-1);

	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(-1);
	}
	return(-1);
}

int
file_column(object strm) {
	int i;
	object x;

BEGIN:
	switch (strm->sm.sm_mode) {
	case smm_output:
	case smm_io:
	case smm_socket:  
	case smm_string_output:
		return(STREAM_FILE_COLUMN(strm));

	case smm_echo:
	case smm_two_way:
           strm=STREAM_OUTPUT_STREAM(strm);
           goto BEGIN;
	case smm_file_synonym:
	case smm_synonym:
		strm = symbol_value(strm->sm.sm_object0);
		if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;


	case smm_input:
	case smm_probe:
	case smm_string_input:
		return(-1);

	case smm_concatenated:
		if (endp(strm->sm.sm_object0))
			return(-1);
		strm = strm->sm.sm_object0->c.c_car;
		goto BEGIN;

	case smm_broadcast:
		for (x = strm->sm.sm_object0; !endp(x); x = x->c.c_cdr) {
			i = file_column(x->c.c_car);
			if (i >= 0)
				return(i);
		}
		return(-1);

#ifdef USER_DEFINED_STREAMS
	case smm_user_defined: /* not right but what is? */
		return(-1);
	
#endif
	default:
		FEerror("Illegal stream mode for ~S.",1,strm);
		return(-1);
	}
}

void
load(const char *s) {

  object filename, tfn, strm, x;

  vs_mark;

  if (user_match(s,strlen(s)))
    return;

  filename = make_simple_string(s);
  vs_push(filename);
  massert(realpath(s,FN2));
  tfn = make_simple_string(FN2);
  bds_bind(sLAload_pathnameA,filename);
  bds_bind(sLAload_truenameA,tfn);

  strm = open_stream(filename, smm_input, Cnil, sKerror);
  vs_push(strm);
  for (;;) {
    preserving_whitespace_flag = FALSE;
    detect_eos_flag = TRUE;
    x = read_object_non_recursive(strm);
    if (x == OBJNULL)
      break;
    vs_push(x);
    ieval(x);
    vs_popp;
  }
  close_stream(strm);

  bds_unwind1;
  bds_unwind1;

  vs_reset;

}



static int
file_synonym_stream_p(object x) {
  switch(type_of(x)) {
  case t_stream:
    switch(x->sm.sm_mode) {
    case smm_input:
    case smm_output:
    case smm_io:
    case smm_probe:
    case smm_file_synonym:
      return 1;
    case smm_synonym:
      return file_synonym_stream_p(x->sm.sm_object0->s.s_dbind);
    default:
      return 0;
    }
  default:
    return 0;
  }
}

LFD(Lmake_synonym_stream)()
{
	object x;

	check_arg(1);
	check_type_sym(&vs_base[0]);
	x = alloc_object(t_stream);
	x->sm.tt=x->sm.sm_mode = file_synonym_stream_p(vs_base[0]->s.s_dbind) ? (short)smm_file_synonym : (short)smm_synonym;
	x->sm.sm_fp = NULL;
	x->sm.sm_buffer = 0;
	x->sm.sm_object0 = vs_base[0];
	x->sm.sm_object1 = OBJNULL;
	x->sm.sm_int = 0;
	x->sm.sm_flags=0;
	vs_base[0] = x;
}

LFD(Lmake_broadcast_stream)() {

  object x;
  int narg, i;

  narg = vs_top - vs_base;
  for (i = 0;  i < narg;  i++)
    if (type_of(vs_base[i]) != t_stream ||
	!output_stream_p(vs_base[i]))
      TYPE_ERROR(vs_base[i],
		 MMcons(sLand,MMcons(sLstream,
				     MMcons(MMcons(sLsatisfies,MMcons(sLoutput_stream_p,Cnil)),Cnil))));
      /* cannot_write(vs_base[i]); */

  stack_list();

  x = alloc_object(t_stream);
  x->sm.tt=x->sm.sm_mode = (short)smm_broadcast;
  x->sm.sm_fp = NULL;
  x->sm.sm_buffer = 0;
  x->sm.sm_object0 = vs_base[0];
  x->sm.sm_object1 = OBJNULL;
  x->sm.sm_int = 0;
  x->sm.sm_flags=0;
  vs_base[0] = x;

}

LFD(Lmake_concatenated_stream)() {

  object x;
  int narg, i;

  narg = vs_top - vs_base;
  for (i = 0;  i < narg;  i++)
    if (type_of(vs_base[i]) != t_stream ||
	!input_stream_p(vs_base[i]))
      cannot_read(vs_base[i]);

  stack_list();

  x = alloc_object(t_stream);
  x->sm.tt=x->sm.sm_mode = (short)smm_concatenated;
  x->sm.sm_fp = NULL;
  x->sm.sm_buffer = 0;
  x->sm.sm_object0 = vs_base[0];
  x->sm.sm_object1 = OBJNULL;
  x->sm.sm_int = 0;
  x->sm.sm_flags=0;
  vs_base[0] = x;

}

DEFUN("MAKE-TWO-WAY-STREAM", object, fLmake_two_way_stream,LISP,2,2,NONE,OO,OO,OO,OO,(object x,object y),"") {

  if (type_of(x) != t_stream || !input_stream_p(x)) TYPE_ERROR(x,sLinput_stream);
  if (type_of(y) != t_stream || !output_stream_p(y)) TYPE_ERROR(y,sLoutput_stream);
  RETURN1(make_two_way_stream(x,y));
  
}

LFD(Lmake_echo_stream)()
{
	check_arg(2);

	if (type_of(vs_base[0]) != t_stream ||
	    !input_stream_p(vs_base[0]))
	  TYPE_ERROR(vs_base[0],sLinput_stream);
	if (type_of(vs_base[1]) != t_stream ||
	    !output_stream_p(vs_base[1]))
	  TYPE_ERROR(vs_base[1],sLoutput_stream);
	vs_base[0] = make_echo_stream(vs_base[0], vs_base[1]);
	vs_popp;
}

DEFUN("MAKE-STRING-OUTPUT-STREAM-INT",object,fLmake_string_output_stream_int,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {

  RETURN1(make_string_output_stream(64));

}


DEFUN("GET-OUTPUT-STREAM-STRING",object,fLget_output_stream_string,LISP,
	   1,1,NONE,OO,OO,OO,OO,(object stream),"") {

  if (type_of(stream) != t_stream ||
      (enum smmode)stream->sm.sm_mode != smm_string_output)
    FEerror("~S is not a string-output stream.", 1, stream);
  stream = get_output_stream_string(stream);
  RETURN1(stream);

}

/* LFD(Lget_output_stream_string)() */
/* { */
/* 	check_arg(1); */

/* 	if (type_of(vs_base[0]) != t_stream || */
/* 	    (enum smmode)vs_base[0]->sm.sm_mode != smm_string_output) */
/* 		FEerror("~S is not a string-output stream.", 1, vs_base[0]); */
/* 	vs_base[0] = get_output_stream_string(vs_base[0]); */
/* } */

/*
	(SI:OUTPUT-STREAM-STRING string-output-stream)

		extracts the string associated with the given
		string-output-stream.
*/
LFD(siLoutput_stream_string)()
{
	check_arg(1);
	if (type_of(vs_base[0]) != t_stream ||
	    (enum smmode)vs_base[0]->sm.sm_mode != smm_string_output)
		FEerror("~S is not a string-output stream.", 1, vs_base[0]);
	vs_base[0] = vs_base[0]->sm.sm_object0;
}

object
file_stream(object x) {
  if (type_of(x)==t_stream)
    switch(x->sm.sm_mode) {
    case smm_input:
    case smm_output:
    case smm_io:
    case smm_probe:
      return x;
    case smm_file_synonym:
      return file_stream(x->sm.sm_object0->s.s_dbind);
    default:
      break;
  }
  return Cnil;
}

DEFUN("FILE-STREAM-P",object,fSfile_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(file_stream(x)!=Cnil ? Ct : Cnil);
}

DEFUN("SYNONYM-STREAM-P",object,fSsynonym_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(type_of(x)==t_stream && x->sm.sm_mode==smm_synonym ? Ct : Cnil);
}

DEFUN("BROADCAST-STREAM-P",object,fSbroadcast_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(type_of(x)==t_stream && x->sm.sm_mode==smm_broadcast ? Ct : Cnil);
}

DEFUN("ECHO-STREAM-P",object,fSecho_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(type_of(x)==t_stream && x->sm.sm_mode==smm_echo ? Ct : Cnil);
}

DEFUN("TWO-WAY-STREAM-P",object,fStwo_way_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(type_of(x)==t_stream && x->sm.sm_mode==smm_two_way ? Ct : Cnil);
}

DEFUN("CONCATENATED-STREAM-P",object,fSconcatenated_stream_p,SI,1,1,NONE,OO,OO,OO,OO,(object x),"") {
  RETURN1(type_of(x)==t_stream && x->sm.sm_mode==smm_concatenated ? Ct : Cnil);
}



LFD(Lstreamp)()
{
	check_arg(1);

	if (type_of(vs_base[0]) == t_stream)
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

LFD(Linput_stream_p)()
{
	check_arg(1);

 	check_type_stream(&vs_base[0]);
	if (input_stream_p(vs_base[0]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

LFD(Loutput_stream_p)()
{
	check_arg(1);

	check_type_stream(&vs_base[0]);
	if (output_stream_p(vs_base[0]))
		vs_base[0] = Ct;
	else
		vs_base[0] = Cnil;
}

LFD(Lstream_element_type)()
{
	check_arg(1);

	check_type_stream(&vs_base[0]);
	vs_base[0] = stream_element_type(vs_base[0]);
}

@(defun close (strm &key abort)
@
	check_type_stream(&strm);
	close_stream(strm);
	@(return Ct)
@)

DEFUN("OPEN-INT",object,fSopen_int,SI,8,8,NONE,OO,OO,OO,OO,
	  (object fn,object direction,object element_type,object if_exists,
	   object iesp,object if_does_not_exist,object idnesp,
	   object external_format),"") {

  enum smmode smm=0;
  vs_mark;
  object strm,filename;
  
  filename=fn;
  if (direction == sKinput) {
    smm = smm_input;
    if (idnesp==Cnil)
      if_does_not_exist = sKerror;
  } else if (direction == sKoutput) {
    smm = smm_output;
    if (iesp==Cnil)
      if_exists = sKnew_version;
    if (idnesp==Cnil) {
      if (if_exists == sKoverwrite ||
	  if_exists == sKappend)
	if_does_not_exist = sKerror;
      else
	if_does_not_exist = sKcreate;
    }
  } else if (direction == sKio) {
    smm = smm_io;
    if (iesp==Cnil)
      if_exists = sKnew_version;
    if (idnesp==Cnil) {
      if (if_exists == sKoverwrite ||
	  if_exists == sKappend)
	if_does_not_exist = sKerror;
      else
	if_does_not_exist = sKcreate;
    }
  } else if (direction == sKprobe) {
    smm = smm_probe;
    if (idnesp==Cnil)
      if_does_not_exist = Cnil;
  } else
    FEerror("~S is an illegal DIRECTION for OPEN.", 1, direction);
  strm = open_stream(filename, smm, if_exists, if_does_not_exist);
  if (type_of(strm) == t_stream) {
    strm->sm.sm_object0 = element_type;
    strm->sm.sm_object1 = fn;
  }
  vs_reset;
  RETURN1(strm);
}


static fixnum /*FIXME, this duplicates code in gcl_iolib.lsp somwhat */
chars_per_write(object s) {/*s already a file-stream*/

  fixnum i;
  
  s=s->sm.sm_object0;
  if (consp(s)) {
    s=s->c.c_cdr->c.c_car;
    i=fix(s);
    return ((i/CHAR_SIZE) + (i%CHAR_SIZE ? 1 : 0));
  } else
    return 1;

}

@(defun file_position (file_stream &o position)
  fixnum i=0;
@
	check_type_stream(&file_stream);
	if (position == Cnil) {
		i = file_position(file_stream);
		if (i < 0)
			@(return Cnil)
	        i/=chars_per_write(file_stream);
		@(return `make_fixnum(i)`)
	} else {
		if (position == sKstart)
			i = 0;
		else if (position == sKend)
			i = file_length(file_stream);
		else if (type_of(position) != t_fixnum ||
		    (i = fix((position))) < 0)
			FEerror("~S is an illegal file position~%\
for the file-stream ~S.",
				2, position, file_stream);
		if (file_position_set(file_stream, i) < 0)
			@(return Cnil)
		@(return Ct)
	}	
@)

LFD(Lfile_length)()
{
	int i;

	check_arg(1);
	check_type_stream(&vs_base[0]);
	i = file_length(vs_base[0]);
	if (i < 0)
		FEwrong_type_argument(sLfile_stream,vs_base[0]);
	else {
	  i/=chars_per_write(vs_base[0]);
	  vs_base[0] = make_fixnum(i);
	}
}

LFD(siLfile_column)()
{
	int i;

	check_arg(1);
	check_type_stream(&vs_base[0]);
	i = file_column(vs_base[0]);
	if (i < 0)
		FEwrong_type_argument(sLfile_stream,vs_base[0]);
	else
		vs_base[0] = make_fixnum(i);
}



DEFVAR("*COLLECT-BINARY-MODULES*",sSAcollect_binary_modulesA,SI,sLnil,"");
DEFVAR("*BINARY-MODULES*",sSAbinary_modulesA,SI,Cnil,"");
DEFVAR("*DISABLE-RECOMPILE*",sSAdisable_recompile,SI,Ct,"");

DEFUN("LOAD-STREAM",object,fSload_stream,SI,2,2,NONE,OO,OO,OO,OO,(object strm,object print),"") {

  object x;

  for (;;) {
    preserving_whitespace_flag = FALSE;
    detect_eos_flag = TRUE;
    if ((x = READ_STREAM_OR_FASD(strm))==OBJNULL)
      break;
    {
      object *base = vs_base, *top = vs_top, *lex = lex_env;
      object xx;

      lex_new();
      eval(x);
      xx = vs_base[0];
      lex_env = lex;
      vs_top = top;
      vs_base = base;
      x = xx;
    }
    if (print != Cnil) {
      princ(x,symbol_value(sLAstandard_outputA));
      princ(make_simple_string("\n"),symbol_value(sLAstandard_outputA));
    }
  }

  RETURN1(Ct);

}
#ifdef STATIC_FUNCTION_POINTERS
object
fSload_stream(object strm,object print) {
  return FFN(fSload_stream)(strm,print);
}
#endif

DEFUN("LOAD-FASL",object,fSload_fasl,SI,2,2,NONE,OO,OO,OO,OO,(object fasl_filename,object print),"") {
  
  int i;

  if (sSAcollect_binary_modulesA->s.s_dbind==Ct) {
    object _x=sSAbinary_modulesA->s.s_dbind;
    object _y=Cnil;
    while (_x!=Cnil) {
      _y=_x;
      _x=_x->c.c_cdr;
    }
    if (_y==Cnil)
      sSAbinary_modulesA->s.s_dbind=make_cons(fasl_filename,Cnil);
    else
      _y->c.c_cdr=make_cons(fasl_filename,Cnil);
  }
  i = fasload(fasl_filename);
  if (print != Cnil) {
    object strm=symbol_value(sLAstandard_outputA);
    if (file_column(strm)!=0)
      princ(make_simple_string("\n"),strm);
    princ(make_simple_string(";; Fasload successfully ended.\n"),strm);
  }

  RETURN1(make_fixnum(i));

}

static void
FFN(siLget_string_input_stream_index)()
{
	check_arg(1);
	check_type_stream(&vs_base[0]);
	if ((enum smmode)vs_base[0]->sm.sm_mode != smm_string_input)
		FEerror("~S is not a string-input stream.", 1, vs_base[0]);
	vs_base[0] = make_fixnum(STRING_INPUT_STREAM_NEXT(vs_base[0]));
}

LFD(siLmake_string_output_stream_from_string)()
{
	object strng, strm;

	check_arg(1);
	strng = vs_base[0];
	if (!stringp(strng) || !strng->st.st_hasfillp)
		FEerror("~S is not a string with a fill-pointer.", 1, strng);
	strm = alloc_object(t_stream);
	strm->sm.tt=strm->sm.sm_mode = (short)smm_string_output;
	strm->sm.sm_fp = NULL;
	strm->sm.sm_buffer = 0;
	STRING_STREAM_STRING(strm) = strng;
	strm->sm.sm_object1 = OBJNULL;
	/* strm->sm.sm_int0 = strng->st.st_fillp; */
	STREAM_FILE_COLUMN(strm) = 0;/*FIXME*/
	strm->sm.sm_flags=0;
	vs_base[0] = strm;
}

LFD(siLcopy_stream)()
{
	object in, out;

	check_arg(2);
	check_type_stream(&vs_base[0]);
	check_type_stream(&vs_base[1]);
	in = vs_base[0];
	out = vs_base[1];
	while (!stream_at_end(in))
		writec_stream(readc_stream(in), out);
	flush_stream(out);
	vs_base[0] = Ct;
	vs_popp;
#ifdef AOSVS

#endif
}

static void
cannot_open(fn)
object fn;
{
	FILE_ERROR(fn,"Cannot open");
}

static void
cannot_create(fn)
object fn;
{
	FILE_ERROR(fn,"Cannot create");
}

static void
cannot_read(strm)
object strm;
{
	FEerror("Cannot read the stream ~S.", 1, strm);
}

static void
cannot_write(strm)
object strm;
{
	FEerror("Cannot write to the stream ~S.", 1, strm);
}

#ifdef USER_DEFINED_STREAMS
/* more support for user defined streams */
static void
FFN(siLuser_stream_state)()
{     
  check_arg(1);

  if(vs_base[0]->sm.sm_object1)
      vs_base[0] = vs_base[0]->sm.sm_object1->str.str_self[0]; 
  else
	FEerror("sLtream data NULL ~S", 1, vs_base[0]);
}
#endif

static void
closed_stream(strm)
object strm;
{
  if (!GET_STREAM_FLAG(strm,gcl_sm_had_error))
    {
        SET_STREAM_FLAG(strm,gcl_sm_had_error,1);
	FEerror("The stream ~S is already closed.", 1, strm);
    }

}



/* returns a stream with which one can safely do fwrite to the x->sm.sm_fp
   or nil.
   */


/* coerce stream to one so that x->sm.sm_fp is suitable for fread and fwrite,
   Return nil if this is not possible.
   */

object
coerce_stream(strm,out)
object strm;
int out;
{
 BEGIN:
 if (type_of(strm) != t_stream)
   FEwrong_type_argument(sLstream, strm);
 switch (strm->sm.sm_mode){
 case smm_file_synonym:
 case smm_synonym:
  strm = symbol_value(strm->sm.sm_object0);
  if (type_of(strm) != t_stream)
			FEwrong_type_argument(sLstream, strm);
		goto BEGIN;

 case smm_two_way:
 case smm_echo:
  if (out)strm = STREAM_OUTPUT_STREAM(strm);
    else strm = STREAM_INPUT_STREAM(strm);
  goto BEGIN;
 case smm_output:
  if (!out) cannot_read(strm);
  break;
 case smm_string_output:
   if (!out) cannot_read(strm);
   return (strm);
  break;
 case smm_input: 
    if (out) cannot_write(strm);
  break;
 case smm_string_input:
    if (out) cannot_write(strm);
    return (strm);
  break;
 case smm_io:
/*  case smm_socket: */
 break;
 
 default:
  strm=Cnil;
  }
 if (strm!=Cnil
     && (strm->sm.sm_fp == NULL))
   closed_stream(strm);
 return(strm);
}

static void
FFN(siLfp_output_stream)()
{check_arg(1);
 vs_base[0]=coerce_stream(vs_base[0],1);
}

static void
FFN(siLfp_input_stream)()
{check_arg(1);
 vs_base[0]=coerce_stream(vs_base[0],0);
}
 

DEFUN("FWRITE",object,fSfwrite,SI,4,4,NONE,OO,OO,OO,OO,
	  (object vector,object start,object count,object stream),"") {

  unsigned char *p;
  int n,beg;
  
  stream=coerce_stream(stream,1);
  if (stream==Cnil) RETURN1(Cnil);
  p = vector->ust.ust_self;
  beg = ((type_of(start)==t_fixnum) ? fix(start) : 0);
  n = ((type_of(count)==t_fixnum) ? fix(count) : (VLEN(vector) - beg));
  if (fwrite(p+beg,1,n,stream->sm.sm_fp)) RETURN1(Ct);
  RETURN1(Cnil);
}

DEFUN("FREAD",object,fSfread,SI,4,4,NONE,OO,OO,OO,OO,
	  (object vector,object start,object count,object stream),"") {
  char *p;
  int n,beg;

  stream=coerce_stream(stream,0);
  if (stream==Cnil) RETURN1(Cnil);
  p = vector->st.st_self;
  beg = ((type_of(start)==t_fixnum) ? fix(start) : 0);
  n = ((type_of(count)==t_fixnum) ? fix(count) : (VLEN(vector) - beg));
  if ((n=SAFE_FREAD(p+beg,1,n,stream->sm.sm_fp)))
    RETURN1(make_fixnum(n));
  RETURN1(Cnil);
}

#ifdef HAVE_NSOCKET

#ifdef DODEBUG
#define dprintf(s,arg) emsg(s,arg)
#else 
#define dprintf(s,arg)
#endif     



/*
  putCharGclSocket(strm,ch) -- put one character to a socket
  stream.
  Results:
  Side Effects:  The buffer may be filled, and the fill pointer
  of the buffer may be changed.
 */
static void
putCharGclSocket(strm,ch)
  object strm;
  int ch;
{
  object bufp = SOCKET_STREAM_BUFFER(strm);

 AGAIN:
  if (bufp->ust.ust_fillp < bufp->ust.ust_dim) {
    dprintf("getchar returns (%c)\n",bufp->ust.ust_self[-1+(bufp->ust.ust_fillp)]);
    bufp->ust.ust_self[(bufp->ust.ust_fillp)++]=ch;
    return;
  }
  else {
    gclFlushSocket(strm);
    goto AGAIN;
  }
}

static void
gclFlushSocket(strm)
     object strm;

{
    int fd = SOCKET_STREAM_FD(strm);
    object bufp = SOCKET_STREAM_BUFFER(strm);
    int i=0;
    int err;
    int wrote;
    if (!GET_STREAM_FLAG(strm,gcl_sm_output)
	||   GET_STREAM_FLAG(strm,gcl_sm_had_error))
	 return;
#define AMT_TO_WRITE 500
    while(i< bufp->ust.ust_fillp) {
      wrote =TcpOutputProc ( fd, 
	                     &(bufp->st.st_self[i]),
		             bufp->ust.ust_fillp-i > AMT_TO_WRITE ? AMT_TO_WRITE : bufp->ust.ust_fillp-i,
			     &err
#ifdef __MINGW32__
				, TRUE /* Wild guess as to whether it should block or not */
#endif
);
      if (wrote < 0) {
	SET_STREAM_FLAG(strm,gcl_sm_had_error,1);
	close_stream(strm);
	FEerror("error writing to socket: errno= ~a",1,make_fixnum(err));
	
      }
      i+= wrote;
    }
    bufp->ust.ust_fillp=0;
}

static
object
make_socket_stream(fd,mode,server,host,port,async)
int fd;
enum gcl_sm_flags mode;
object server;
object host;
object port;
object async;
{
  object x;
  if (fd < 0 )
   {
     FEerror("Could not connect",0);
   }
  x = alloc_object(t_stream);
  x->sm.tt=x->sm.sm_mode = smm_socket;
  x->sm.sm_buffer = 0;
  x->sm.sm_object0 = list(3,server,host,port);
  x->sm.sm_object1 = 0;
  x->sm.sm_int = 0;
  x->sm.sm_flags=0;
  SOCKET_STREAM_FD(x)= fd;
  SET_STREAM_FLAG(x,mode,1);
  SET_STREAM_FLAG(x,gcl_sm_tcp_async,(async!=Cnil));
  /*
  if (mode == gcl_sm_output)
     { fp=fdopen(fd,(mode==gcl_sm_input ? "r" : "w"));
       if (fp==NULL)      FEerror("Could not connect",0);
       x->sm.sm_fp = fp;
       setup_stream_buffer(x);
     } else
   */
     {
         object buffer;
         x->sm.sm_fp = NULL;
	 buffer=alloc_string((BUFSIZ < 4096 ? 4096 : BUFSIZ));
	 SOCKET_STREAM_BUFFER(x) =buffer;
	 buffer->ust.ust_self = alloc_contblock(buffer->st.st_dim);
	 buffer->ust.ust_fillp = 0;
     }
  return x;
}
     
static object
maccept(object x) {

  int fd;
  struct sockaddr_in addr;
  unsigned n=sizeof(addr);
  object server,host,port;
  
  if (type_of(x) != t_stream)
    FEerror("~S is not a steam~%",1,x);
  if (x->sm.sm_mode!=smm_two_way)
    FEerror("~S is not a two-way steam~%",1,x);
  memset(&addr,0,sizeof(addr));
  fd=accept(SOCKET_STREAM_FD(STREAM_INPUT_STREAM(x)),(struct sockaddr *)&addr, &n);
  if (fd <0) {
    FEerror("Error ~S on accepting connection to ~S~%",2,make_simple_string(strerror(errno)),x);
    x=Cnil;
  } else {
    server=STREAM_INPUT_STREAM(x)->sm.sm_object0->c.c_car;
    host=STREAM_INPUT_STREAM(x)->sm.sm_object0->c.c_cdr->c.c_car;
    port=STREAM_INPUT_STREAM(x)->sm.sm_object0->c.c_cdr->c.c_cdr->c.c_car;
    x = make_two_way_stream
      (make_socket_stream(fd,gcl_sm_input,server,host,port,Cnil),
       make_socket_stream(fd,gcl_sm_output,server,host,port,Cnil));
  }
  return x;

}

#ifdef BSD
#include <sys/types.h>
#include <sys/resource.h>
#include <signal.h>

#if defined(DARWIN) || defined(FREE_BSD)
#define on_exit(a,b)
#else
static void
rmc(int e,void *pid) {

  kill((long)pid,SIGTERM);

}
#endif
#endif

@(static defun socket (port &key host server async myaddr myport daemon)
             /*
	     HOST is a string then connection is made to that
                          ip or domain address.
	    SERVER  A function to call if this is to be a server 
                            
            ASYNC  socket returned immideiately.  read or flush
                           will block till open if in non blocking mode 
            MYADDR  client's ip address. Useful if have several
                                    net interfaces
            MYPORT  port to use on client side
                        */
int fd;
int isServer = 0;
int inPort;
char buf1[500];
char buf2[500];
char *myaddrPtr=buf1,*hostPtr=buf2;
object x=Cnil;
@
  if (stringp(host)) {
    hostPtr=lisp_copy_to_null_terminated(host,hostPtr,sizeof(buf1));
  } else { hostPtr = NULL; }
  
   if (fLfunctionp(server) == Ct) {
      isServer=1;
   }
   
   if (myaddr != Cnil) {
     myaddrPtr=lisp_copy_to_null_terminated(myaddr,myaddrPtr,sizeof(buf2));
   } else   { myaddrPtr = NULL; }
   if (isServer == 0 && hostPtr == NULL) {
     FEerror("You must supply at least one of :host hostname or :server function",0);
   }
   Iis_fixnum(port);
   inPort = (myport == Cnil ? 0 : fix(Iis_fixnum(myport)));
   
#ifdef BSD

  if (isServer && daemon != Cnil) {

    long pid,i;
    struct rlimit r;
    struct sigaction sa,osa;

    sa.sa_handler=SIG_IGN;
    sa.sa_flags=SA_NOCLDWAIT;
    sigemptyset(&sa.sa_mask);

    massert(!sigaction(SIGCHLD,&sa,&osa));

    switch((pid=pvfork())) {
    case -1:
      FEerror("Cannot fork", 0);
      break;
    case 0:

      massert(setsid()>=0);

      if (daemon == sKpersistent)
	switch(pvfork()) {
	case -1:
	  FEerror("daemon fork error", 0);
	  break;
	case 0:
	  break;
	default:
	  exit(0);
	  break;
	}
      
      massert(!chdir("/"));

      memset(&r,0,sizeof(r));
      massert(!getrlimit(RLIMIT_NOFILE,&r));
      
      for (i=0;i<r.rlim_cur;i++)
      	close(i);/*FIXME some of this will return error*/
      
      massert((i=open("/dev/null",O_RDWR))>=0);
      massert((i=dup(i))>=0);
      massert((i=dup(i))>=0);
      
      umask(0);
      
      fd = CreateSocket(fix(port),hostPtr,isServer,myaddrPtr,inPort,(async!=Cnil));
      
      x = make_two_way_stream
	(make_socket_stream(fd,gcl_sm_input,server,host,port,async),
	 make_socket_stream(fd,gcl_sm_output,server,host,port,async));
    
      for (;;) {
	
	fd_set fds;
	object y;
	
	FD_ZERO(&fds);
	FD_SET(fd,&fds);
	
	if (select(fd+1,&fds,NULL,NULL,NULL)>0) {
	  
	  y=maccept(x);
	  
	  switch((pid=pvfork())) {
	  case 0:
	    massert(!sigaction(SIGCHLD,&osa,NULL));
	    ifuncall1(server,y);
	    exit(0);
	    break;
	  case -1:
	    gcl_abort();
	    break;
	  default:
	    close_stream(y);
	    break;
	  }
	  
	}
      }
      break;
    default:
      if (daemon != sKpersistent) {
	on_exit(rmc,(void *)pid);
	x=make_fixnum(pid);
      } else
	x=Cnil;
      break;
    }

    massert(!sigaction(SIGCHLD,&osa,NULL));

  } else 

#endif

  {
    fd = CreateSocket(fix(port),hostPtr,isServer,myaddrPtr,inPort,(async!=Cnil));
	
    x = make_two_way_stream
      (make_socket_stream(fd,gcl_sm_input,server,host,port,async),
       make_socket_stream(fd,gcl_sm_output,server,host,port,async));

  }
    
  @(return `x`);
   
@)

DEF_ORDINARY("MYADDR",sKmyaddr,KEYWORD,"");
DEF_ORDINARY("MYPORT",sKmyport,KEYWORD,"");
DEF_ORDINARY("ASYNC",sKasync,KEYWORD,"");
DEF_ORDINARY("HOST",sKhost,KEYWORD,"");
DEF_ORDINARY("SERVER",sKserver,KEYWORD,"");
DEF_ORDINARY("DAEMON",sKdaemon,KEYWORD,"");
DEF_ORDINARY("PERSISTENT",sKpersistent,KEYWORD,"");
DEF_ORDINARY("SOCKET",sSsocket,SI,"");


@(static defun accept (x)
@
  x=maccept(x);
  @(return `x`);
@)

#endif /* HAVE_NSOCKET */

object
fresh_synonym_stream_to_terminal_io(void) {

  object x = alloc_object(t_stream);
  x->sm.tt=x->sm.sm_mode = (short)smm_synonym;
  x->sm.sm_fp = NULL;
  x->sm.sm_buffer = 0;
  x->sm.sm_object0 = sLAterminal_ioA;
  x->sm.sm_object1 = OBJNULL;
  x->sm.sm_int = 0; /* unused */
  x->sm.sm_flags=0;
  return x;

}

/* object standard_io; */
object standard_error;
DEFVAR("*TERMINAL-IO*",sLAterminal_ioA,LISP,(gcl_init_file(),terminal_io),"");
DEFVAR("*STANDARD-INPUT*",sLAstandard_inputA,LISP,fresh_synonym_stream_to_terminal_io(),"");
DEFVAR("*STANDARD-OUTPUT*",sLAstandard_outputA,LISP,fresh_synonym_stream_to_terminal_io(),"");
DEFVAR("*ERROR-OUTPUT*",sLAerror_outputA,LISP,standard_error,"");
DEFVAR("*QUERY-IO*",sLAquery_ioA,LISP,fresh_synonym_stream_to_terminal_io(),"");
DEFVAR("*DEBUG-IO*",sLAdebug_ioA,LISP,fresh_synonym_stream_to_terminal_io(),"");
DEFVAR("*TRACE-OUTPUT*",sLAtrace_outputA,LISP,fresh_synonym_stream_to_terminal_io(),"");


void
gcl_init_file(void)
{
	object standard_input;
	object standard_output;
	object standard;

	standard_input = alloc_object(t_stream);
	standard_input->sm.tt=standard_input->sm.sm_mode = (short)smm_input;
	standard_input->sm.sm_fp = stdin;
	standard_input->sm.sm_buffer = 0;
	standard_input->sm.sm_object0 = sLcharacter;
	standard_input->sm.sm_object1 = make_simple_string("stdin");
	standard_input->sm.sm_int = 0; /* unused */
	standard_input->sm.sm_flags = 0;

	standard_output = alloc_object(t_stream);
	standard_output->sm.tt=standard_output->sm.sm_mode = (short)smm_output;
	standard_output->sm.sm_fp = stdout;
	standard_output->sm.sm_buffer = 0;
	standard_output->sm.sm_object0 = sLcharacter;
	standard_output->sm.sm_object1 = make_simple_string("stdout");
	standard_output->sm.sm_int = 0; /* unused */
	standard_output->sm.sm_flags=0;

	standard_error = alloc_object(t_stream);
	standard_error->sm.tt=standard_error->sm.sm_mode = (short)smm_output;
	standard_error->sm.sm_fp = stderr;
	standard_error->sm.sm_buffer = 0;
	standard_error->sm.sm_object0 = sLcharacter;
	standard_error->sm.sm_object1 = make_simple_string("stderr");
	standard_error->sm.sm_int = 0; /* unused */
	standard_error->sm.sm_flags=0;
	enter_mark_origin(&standard_error);

	terminal_io = standard
	= make_two_way_stream(standard_input, standard_output);
	enter_mark_origin(&terminal_io);


}

DEFVAR("*IGNORE-EOF-ON-TERMINAL-IO*",sSAignore_eof_on_terminal_ioA,SI,Cnil,"");
DEFVAR("*LOAD-PATHNAME*",sLAload_pathnameA,LISP,Cnil,"");
DEFVAR("*LOAD-TRUENAME*",sLAload_truenameA,LISP,Cnil,"");
DEFVAR("*LOAD-VERBOSE*",sLAload_verboseA,LISP,Ct,"");
DEFVAR("*LOAD-PRINT*",sLAload_printA,LISP,Cnil,"");

DEF_ORDINARY("ABORT",sKabort,KEYWORD,"");
DEF_ORDINARY("APPEND",sKappend,KEYWORD,"");
DEF_ORDINARY("CREATE",sKcreate,KEYWORD,"");
DEF_ORDINARY("DEFAULT",sKdefault,KEYWORD,"");
DEF_ORDINARY("DIRECTION",sKdirection,KEYWORD,"");
DEF_ORDINARY("ELEMENT-TYPE",sKelement_type,KEYWORD,"");
DEF_ORDINARY("ERROR",sKerror,KEYWORD,"");
DEF_ORDINARY("FILE-ERROR",sKfile_error,KEYWORD,"");
DEF_ORDINARY("PATHNAME-ERROR",sKpathname_error,KEYWORD,"");
DEF_ORDINARY("IF-DOES-NOT-EXIST",sKif_does_not_exist,KEYWORD,"");
DEF_ORDINARY("IF-EXISTS",sKif_exists,KEYWORD,"");
DEF_ORDINARY("INPUT",sKinput,KEYWORD,"");
DEF_ORDINARY("IO",sKio,KEYWORD,"");
DEF_ORDINARY("NEW-VERSION",sKnew_version,KEYWORD,"");
DEF_ORDINARY("OUTPUT",sKoutput,KEYWORD,"");
DEF_ORDINARY("OVERWRITE",sKoverwrite,KEYWORD,"");
DEF_ORDINARY("PRINT",sKprint,KEYWORD,"");
DEF_ORDINARY("PROBE",sKprobe,KEYWORD,"");
DEF_ORDINARY("RENAME",sKrename,KEYWORD,"");
DEF_ORDINARY("RENAME-AND-DELETE",sKrename_and_delete,KEYWORD,"");
DEF_ORDINARY("SET-DEFAULT-PATHNAME",sKset_default_pathname,KEYWORD,"");
DEF_ORDINARY("SUPERSEDE",sKsupersede,KEYWORD,"");
DEF_ORDINARY("VERBOSE",sKverbose,KEYWORD,"");


DEFUN("STREAM-OBJECT0",object,fSstream_object0,SI,1,1,NONE,OO,OO,OO,OO,(object strm),"") {
  
  object x;
  
  if (type_of(strm)!=t_stream)
    TYPE_ERROR(strm,sLstream);
  x=strm->sm.sm_object0;
  x=x==OBJNULL ? Cnil : x;
  RETURN1(x);
  
}

DEFUN("STREAM-OBJECT1",object,fSstream_object1,SI,1,1,NONE,OO,OO,OO,OO,(object strm),"") { 

  object x;

  if (type_of(strm)!=t_stream)
    TYPE_ERROR(strm,sLstream);
  x=strm->sm.sm_object1;
  x=x==OBJNULL ? Cnil : x;
  RETURN1(x);

}

void
gcl_init_file_function()
{


#ifdef UNIX
	FASL_string = make_simple_string("o");
	make_si_constant("*EOF*",make_fixnum(EOF));
#endif
#ifdef AOSVS

#endif
	enter_mark_origin(&FASL_string);
#ifdef UNIX
	LSP_string = make_simple_string("lsp");
#endif
#ifdef AOSVS

#endif
	enter_mark_origin(&LSP_string);
	make_si_function("FP-INPUT-STREAM",	siLfp_input_stream);
	make_si_function("FP-OUTPUT-STREAM",	siLfp_output_stream);

	make_function("MAKE-SYNONYM-STREAM", Lmake_synonym_stream);
	make_function("MAKE-BROADCAST-STREAM", Lmake_broadcast_stream);
	make_function("MAKE-CONCATENATED-STREAM",
		      Lmake_concatenated_stream);
/* 	make_function("MAKE-TWO-WAY-STREAM", Lmake_two_way_stream); */
	make_function("MAKE-ECHO-STREAM", Lmake_echo_stream);
/* 	make_function("MAKE-STRING-OUTPUT-STREAM", */
/* 		      Lmake_string_output_stream); */
/* 	make_function("GET-OUTPUT-STREAM-STRING", */
/* 		      Lget_output_stream_string); */

	make_si_function("OUTPUT-STREAM-STRING", siLoutput_stream_string);
/* 	make_si_function("FWRITE",Lfwrite); */
/* 	make_si_function("FREAD",Lfread); */
#ifdef HAVE_NSOCKET
	make_si_function("SOCKET",Lsocket);
	make_si_function("ACCEPT",Laccept);
#endif
	make_function("STREAMP", Lstreamp);
	make_function("INPUT-STREAM-P", Linput_stream_p);
	make_function("OUTPUT-STREAM-P", Loutput_stream_p);
	make_function("STREAM-ELEMENT-TYPE", Lstream_element_type);
	make_function("CLOSE", Lclose);

/* 	make_si_function("OPEN1", Lopen1); */

	make_function("FILE-POSITION", Lfile_position);
	make_function("FILE-LENGTH", Lfile_length);
	make_si_function("FILE-COLUMN", siLfile_column);
/* 	make_function("READ-SEQUENCE", Lread_sequence); */
/* 	make_function("WRITE-SEQUENCE", Lwrite_sequence); */

	/* make_si_function("LOAD1", Lload1); */

	make_si_function("GET-STRING-INPUT-STREAM-INDEX",
			 siLget_string_input_stream_index);
	make_si_function("MAKE-STRING-OUTPUT-STREAM-FROM-STRING",
			 siLmake_string_output_stream_from_string);
	make_si_function("COPY-STREAM", siLcopy_stream);

#ifdef USER_DEFINED_STREAMS
	make_si_function("USER-STREAM-STATE", siLuser_stream_state);
#endif

#ifdef USE_READLINE
	gcl_init_readline_function();
#endif
}
