/*
 Copyright (C) 1994 Rami el Charif, W. Schelter 
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

#define IN_SOCKETS
#include "include.h"

#ifdef HAVE_NSOCKET

#include "sheader.h"

#include <sys/types.h>
#ifndef __MINGW32__
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#else
#  include <winsock2.h>
#  include <windows.h>
#endif

#ifdef __STDC__
#endif

#ifndef __MINGW32__
# include <netdb.h> 
#endif

#include <sys/time.h>
#ifndef NO_UNISTD_H
#include <unistd.h>
#endif
#include <fcntl.h>
/*#include <signal.h> */

#include <errno.h> 

static void write_timeout_error();
static void connection_failure();

#ifdef __MINGW32__
/* Keep track of socket initialisations */
int w32_socket_initialisations = 0;
WSADATA WSAData;

int w32_socket_init(void)
{
    int rv = 0;
    if (w32_socket_initialisations++) {
	rv = 0;
    } else {
        if (WSAStartup(0x0101, &WSAData)) {
            w32_socket_initialisations = 0;
            emsg("WSAStartup failed\n" );
            WSACleanup();
            rv = -1;
        }
    }

    return rv;
}

int w32_socket_exit(void)
{
    int rv = 0;

    if ( w32_socket_initialisations == 0 ||
         --w32_socket_initialisations > 0 ) {
	rv = 0;
    } else {
        rv = WSACleanup();
    }
    
    return rv;
}

#endif

#define BIND_MAX_RETRY		128
#define BIND_ADDRESS_INCREMENT	16
#define BIND_INITIAL_ADDRESS	5000
#define BIND_LAST_ADDRESS	65534
static unsigned int iLastAddressUsed = BIND_INITIAL_ADDRESS;

DEFUN("OPEN-NAMED-SOCKET",object,fSopen_named_socket,SI,1,1,NONE,OI,OO,OO,OO,(fixnum port),
"Open a socket on PORT and return (cons fd portname) where file \
descriptor is a small fixnum which is the write file descriptor for \
the socket.  If PORT is zero do automatic allocation of port") 
{
#ifdef __MINGW32__
    SOCKET s;
#else    
    int s;
#endif    
  int n, rc;
  struct sockaddr_in addr;

#ifdef __MINGW32__  
  if ( w32_socket_init() < 0 ) {
      perror("ERROR !!! Windows socket DLL initialisation failed in sock_connect_to_name\n");
      return Cnil;
  }
#endif
  
  /* Using TCP layer */
  s = socket(PF_INET, SOCK_STREAM, 0);
#ifdef __MINGW32__
    if ( s == INVALID_SOCKET )  
#else    
  if (s < 0)
#endif      
    {
      perror("ERROR !!! socket creation failed in sock_connect_to_name\n");
      return Cnil;
    }

  addr.sin_family = PF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  memset(addr.sin_zero, 0, 8);
  n = sizeof addr;

  if (port == 0)
    {
#define MY_HTONS(x) htons((unsigned short)((x) & 0xffff))      
      int cRetry = 0;
      do {
	addr.sin_port = MY_HTONS(iLastAddressUsed);
	rc = bind(s, (struct sockaddr *)&addr, n);

	cRetry++;
	iLastAddressUsed += BIND_ADDRESS_INCREMENT;
	if (iLastAddressUsed > BIND_LAST_ADDRESS)
	  iLastAddressUsed = BIND_INITIAL_ADDRESS;
      } while ((rc < 0) &&
#ifdef __MINGW32__                
                (errno == WSAEADDRINUSE) &&
#else                
                (errno == EADDRINUSE) &&
#endif                
                (cRetry < BIND_MAX_RETRY));
      if (0)
	  emsg("\nAssigned automatic address to socket : port(%d), errno(%d), bind_rc(%d), iLastAddressUsed(%d), retries(%d)\n"
		  , addr.sin_port, errno, rc, iLastAddressUsed, cRetry
		  );
    }
  else
    {
      addr.sin_port = MY_HTONS(port);
      rc = bind(s, (struct sockaddr *)&addr, n);
    }
  if (rc < 0)
    {
      perror("ERROR !!! Failed to bind socket in sock_open_named_socket\n");
      close(s);
      return Cnil;
    }
  rc = listen(s, 3);
  if (rc < 0)
    {
      perror("ERROR ! listen failed on socket in sock_open_named_socket");
      close(s);
      return Cnil;
    }

  return make_cons(make_fixnum(s), make_fixnum(ntohs(addr.sin_port)));
}

DEFUN("CLOSE-FD",object,fSclose_fd,SI,1,1,NONE,OI,OO,OO,OO,(fixnum fd),
      "Close the file descriptor FD")

{RETURN1(0==close(fd) ? Ct : Cnil);}

DEFUN("CLOSE-SD",object,fSclose_sfd,SI,1,1,NONE,OO,OO,OO,OO,(object sfd),
      "Close the socket connection sfd")

{ int res;
  free(OBJ_TO_CONNECTION_STATE(sfd)->read_buffer);
  res = close(OBJ_TO_CONNECTION_STATE(sfd)->fd);
  free (OBJ_TO_CONNECTION_STATE(sfd));
#ifdef __MINGW32__  
  w32_socket_exit();
#endif  
  RETURN1(res ? Ct : Cnil);
}


DEFUN("ACCEPT-SOCKET-CONNECTION",object,fSaccept_socket_connection,
	  SI,1,1,NONE,OO,OO,OO,OO,(object named_socket),
      "Given a NAMED_SOCKET it waits for a connection on this \
and returns (list* named_socket fd name1) when one is established")

{
  socklen_t n;
  int fd;
  struct sockaddr_in addr;
  object x; 
  n = sizeof addr;
  fd = accept(fix(car(named_socket)) , (struct sockaddr *)&addr, &n);
  if (fd < 0)
    {
      emsg("ERROR ! accept on socket failed in sock_accept_connection");
      return Cnil;
    }
  x = alloc_string(sizeof(struct connection_state));
  x->ust.ust_self = (void *)setup_connection_state(fd);
  return make_cons(
		   make_cons(x
			     , make_simple_string(
						  inet_ntoa(addr.sin_addr))),
		   named_socket
		   );
}

/* static object */
/* sock_hostname_to_hostid_list(host_name) */
/*      char *host_name; */
/* { */
/*   struct hostent *h; */
/*   object addr_list = Cnil; */
/*   int i; */

/*   h = gethostbyname(host_name); */

/*   for (i = 0; h->h_addr_list[i] != 0; i++) */
/*     { */
/*       addr_list = make_cons(make_simple_string(inet_ntoa(*(struct in_addr *)h->h_addr_list[i])), addr_list); */
/*     } */
/*   return addr_list; */
/* } */

    
      

DEFUN("HOSTNAME-TO-HOSTID",object,fShostname_to_hostid,SI,1,1,
      NONE,OO,OO,OO,OO,(object host),"")
{
  struct hostent *h;
  char buf[300];
  char *p;
  p = lisp_copy_to_null_terminated(host,buf,sizeof(buf));
  h = 
#ifdef STATIC_LINKING
    NULL;
#else
  gethostbyname(p);
#endif
  if (p != buf) free (p);
  if (h && h->h_addr_list[0])
    return
     make_simple_string(inet_ntoa(*(struct in_addr *)h->h_addr_list[0]));
  else return Cnil;
}

DEFUN("GETHOSTNAME",object,fSgethostname,SI,0,0,NONE,OO,OO,OO,OO,(void),
      "Returns HOSTNAME of the local host")
     
{char buf[300];
 if (0 == gethostname(buf,sizeof(buf)))
   return make_simple_string(buf);
 else return Cnil;
}

DEFUN("HOSTID-TO-HOSTNAME",object,fShostid_to_hostname,SI,
      1,10,NONE,OO,OO,OO,OO,(object host_id),"")

{char *hostid;
  struct in_addr addr;
  struct hostent *h;
  char buf[300];
  hostid = lisp_copy_to_null_terminated(host_id,buf,sizeof(buf));
  addr.s_addr = inet_addr(hostid);
  h = 
#ifdef STATIC_LINKING
    NULL;
#else
  gethostbyaddr((char *)&addr, 4, AF_INET);
#endif
  if (h && h->h_name && *h->h_name)
    return make_simple_string(h->h_name);
  else
    return Cnil;
}

/* static object */
/* sock_get_name(s) */
/*      int s; */
/* { */
/*   struct sockaddr_in addr; */
/*   int m = sizeof(addr); */
/*   getsockname(s, (struct sockaddr *)&addr, &m); */
/*   return make_cons( */
/* 		   make_cons( */
/* 			     make_fixnum(addr.sin_port) */
/* 			     , make_simple_string(inet_ntoa(addr.sin_addr)) */
/* 			     ) */
/* 		   ,make_cons(make_fixnum(addr.sin_family) */
/* 			      , make_fixnum(s)) */
/* 		   ); */
/* } */

#include "comm.c"


DEFUN("CONNECTION-STATE-FD",object,fSconnection_state_fd,SI,1,1,NONE,OO,OO,OO,OO,(object sfd),"") 
{ return make_fixnum(OBJ_TO_CONNECTION_STATE(sfd)->fd);
}
     
DEFUN("OUR-WRITE",object,fSour_write,SI,3,3,NONE,OO,OI,OO,OO,(object sfd,object buffer,fixnum nbytes),"")

{ return make_fixnum(write1(OBJ_TO_CONNECTION_STATE(sfd),buffer->st.st_self,nbytes));
}

DEFUN("OUR-READ-WITH-OFFSET",object,fSour_read_with_offset,SI,5,5,NONE,
	  OO,OI,II,OO,(object fd,object buffer,fixnum offset,fixnum nbytes,fixnum timeout),
      "Read from STATE-FD into string BUFFER putting data at OFFSET and reading NBYTES, waiting for TIMEOUT before failing")

{ return make_fixnum(read1(OBJ_TO_CONNECTION_STATE(fd),&((buffer)->st.st_self[offset]),nbytes,timeout));
}


enum print_arglist_codes {
    normal,
    no_leading_space,
    join_follows,
    end_join,
    begin_join,
    begin_join_no_leading_space,
    no_quote,
    no_quote_no_leading_space,
    no_quote_downcase,
    no_quotes_and_no_leading_space
  };

  /* push object X into the string with fill pointer STR, according to CODE
     */
  

#define PUSH(_c) do{if (--left < 0) goto FAIL; \
		     *xx++ = _c;}while(0)


#define BEGIN_QUOTE '"'
#define END_QUOTE '"'

static int needs_quoting[256];

DEFUN("PRINT-TO-STRING1",object,fSprint_to_string1,SI,3,3,NONE,OO,OO,OO,OO,(object str,object x,object the_code),
      "Print to STRING the object X according to CODE.   The string must have \
fill pointer, and this will be advanced.")

{ enum type t = type_of(x);
  int fp = VLEN(str);
  char *xx = &(str->st.st_self[fp]);
  int left = str->st.st_dim - fp;
  char buf[30];
  char *p;
  enum print_arglist_codes code = fix(the_code);

  if (code==no_quote || code == no_quotes_and_no_leading_space)
       { needs_quoting['"']=0;
	 needs_quoting['$']=0;
	 needs_quoting['\\']=0;
	  needs_quoting['[']=0;
/*	 needs_quoting[']']=0; */
       }
  else { needs_quoting['"']=1;
	 needs_quoting['$']=1;
	 needs_quoting['\\']=1;
	 needs_quoting['[']=1;
/*	 needs_quoting[']']=1; */
       }
 { 
  int downcase ;
  int do_end_quote = 0;
  if(!stringp(str))
    FEerror("Must be given string with fill pointer",0);
  if (t==t_symbol) downcase=1;
  else downcase=0;
  
  switch (code){

  case no_quote_downcase:
    downcase = 1;
  case no_quote:
    PUSH(' ');
  case  no_quotes_and_no_leading_space:
  case no_quote_no_leading_space:
  break;

  case normal:
    PUSH(' ');
  case no_leading_space:
    if (stringp_tp(t))
      { do_end_quote = 1;
	PUSH(BEGIN_QUOTE);
      }
    break;
    
  case begin_join:
    PUSH(' ');
  case begin_join_no_leading_space:
    PUSH(BEGIN_QUOTE);
    break;
  case  end_join:
    do_end_quote=1;
    break;
  case join_follows:


    break;
  default: do_gcl_abort();
  }
  
  switch (t) {
  case t_symbol:
    if (x->s.s_hpack == keyword_package)
      {if (code == normal)
	 PUSH('-');}
    x=x->s.s_name;
  case t_simple_string:/*FIXME?*/
  case t_string:
    {int len = VLEN(x);
      p = &x->st.st_self[0];
     if (downcase)
     while (--len>=0)
       { char c = *p++;
	 c=tolower((int)c);
	 if(needs_quoting[(unsigned char)c])
	   PUSH('\\');
	 PUSH(c);}
     else
       while (--len>=0)
       { char c = *p++;
	 if(needs_quoting[(unsigned char)c])
	   PUSH('\\');	 
	 PUSH(c);}}
   break;
   case t_fixnum:
     sprintf(buf,"%ld",fix(x));
     p = buf;
     while(*p) {PUSH(*p);p++;}
     break;
   case t_longfloat:
     sprintf(buf,"%.2f",lf(x));
     p = buf;
     while(*p) {PUSH(*p);p++;}
     break;
   case t_shortfloat:
     sprintf(buf,"%.2f",sf(x));
     p = buf;
     while(*p) {PUSH(*p);p++;}
     break;
   case t_bignum:
     goto FAIL;
   default:
     FEerror("Bad type for print_string ~s",1,x);
   }
     if(do_end_quote) PUSH('"');
    str->st.st_fillp += (xx - &(str->st.st_self[fp]));
    return Ct;
 FAIL:

  /* either ran out of storage or tried to print a bignum.
     The caller will handle these two cases
     */
   return Cnil;
   }
}

static void
not_defined_for_os()
{ FEerror("Function not defined for this operating system",0);}


DEFUN("SET-SIGIO-FOR-FD",object,fSset_sigio_for_fd,SI,1,1,NONE,OI,OO,OO,OO,(fixnum fd),"")

{ 
  /* for the moment we will use SIGUSR1 to notify, instead of depending on SIGIO,
     since LINUX does not support the latter yet...
     So right  now this does nothing... 
  */   
#if !defined(FASYNC) || !defined(SET_FD_TO_GIVE_SIGIO)
  not_defined_for_os();

#else
#ifdef SET_FD_TO_GIVE_SIGIO
  SET_FD_TO_GIVE_SIGIO(fd);
#else
  /* want something like this... but wont work on all machines. */
  flags = fcntl(fd,F_GETFL,0);
  if (flags == -1
      || ( flags |=  FASYNC , 0)
      || -1 ==   fcntl(fd,F_SETFL,flags)
      || -1 ==   fcntl(fd,F_SETOWN,getpid()))
    {perror("Could not set ASYNC IO for SIGIO:");
     return Cnil;}
#endif
#endif

  return (Ct);

}
     
DEFUN("RESET-STRING-INPUT-STREAM",object,fSreset_string_input_stream,SI,4,4,NONE,OO,OI,IO,OO,(object strm,object string,fixnum start,fixnum end),
      "Reuse a string output STREAM by setting its output to STRING \
and positioning the ouput/input to start at START and end at END")

{ massert(type_of(string)==t_string);
  strm->sm.sm_object0 = string;
  STRING_INPUT_STREAM_NEXT(strm) = start;
  STRING_INPUT_STREAM_END(strm) = end;
  return strm;
}

DEFUN("CHECK-STATE-INPUT",object,fScheck_state_input,SI,2,2,NONE,IO,IO,OO,OO,(object osfd,fixnum timeout),
      "") 
{
  return (object)fScheck_dsfd_for_input(OBJ_TO_CONNECTION_STATE(osfd),timeout);

}

DEFUN("CLEAR-CONNECTION-STATE",object,fSclear_connection_state,
	  SI,1,1,NONE,OO,OO,OO,OO,(object osfd),
      "Read on FD until nothing left to read.  Return number of bytes read")

{ 
  struct connection_state *sfd = OBJ_TO_CONNECTION_STATE(osfd);
  int n=fix(FFN(fSclear_connection)(sfd->fd));
  
  sfd->valid_data = sfd->read_buffer;
  sfd->valid_data_size = 0;
  sfd->bytes_received_not_confirmed += n;
 return make_fixnum(n);
}

#endif

static void
write_timeout_error(s)
     char *s;
{FEerror("Write timeout: ~s",1,make_simple_string(s));
}

static void
connection_failure(s)
     char *s;
{FEerror("Connect failure: ~s",1,make_simple_string(s));
}


