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

#define IN_GUIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif
    
#include <sys/types.h>

#ifndef _WIN32
#  include <netinet/in.h>
#  ifdef PLATFORM_NEXT
#     include <bsd/netdb.h>
#     include <libc.h>
#  else
#     include <netdb.h>
#     include <arpa/inet.h>
#  endif
#endif
    
/* #include <sys/types.h> */

#include <sys/time.h>

#ifndef _WIN32  
#include <sys/socket.h>
#endif    

#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#ifdef __cplusplus
#ifdef PLATFORM_NEXT
extern unsigned long inet_addr( char *cp );
extern char *inet_ntoa ( struct in_addr in );
#endif
}
#endif
#ifdef PLATFORM_LINUX
#include <termios.h>
#endif
#include <errno.h>

#ifdef __svr4__
#include <sys/file.h>
#endif

#ifdef PLATFORM_NEXT /* somehow, this is getting lost... */
#undef bzero
#define bzero(b,len) memset(b,0,len)
#endif


#include "guis.h"

#ifndef TRUE
#define TRUE (1)
#define FALSE (0)
#endif

FILE *pstreamDebug;
int fDebugSockets;

/* #ifdef PLATFORM_SUNOS */
/* static void notice_input( ); */
/* #else */
/* static void notice_input(); */
/* #endif */

int hdl = -1;

void TkX_Wish ();

pid_t parent;
 
int debug;

#ifdef _WIN32

#include <windows.h>
#include <winsock2.h>

/* Keep track of socket initialisations */
int w32_socket_initialisations = 0;

WSADATA WSAData;


/* Use threads instead of fork() */
/* Struct to hold args for thread. */
typedef struct _TAS {
    char **argv;
    int    argc;
    int    rv;
    int    delay;
} TAS;

#endif

#include "comm.c"

#ifdef _WIN32

#define SET_SESSION_ID() 0

UINT WINAPI tf1 ( void *tain )
{
    TAS *ta = (TAS *) tain;
    UINT rv = 0;
    if (SET_SESSION_ID() == -1) {
        fprintf ( stderr, "tf: Error - set session id failed : %d\n", errno );
    }
    if ( w32_socket_init() >= 0 ) {
        dsfd = sock_connect_to_name ( ta->argv[1], atoi ( ta->argv[2] ), 0);
        if ( dsfd ) {
            fprintf ( stderr, "connected to %s %s\n", ta->argv[1], ta->argv[2] );
            TkX_Wish ( ta->argc, ta->argv );
            fprintf ( stderr, "Wish shell done\n" );
            sock_close_connection ( dsfd );
            ta->rv = 0;
        } else {
            fprintf ( stderr,
                       "Error: Can't connect to socket host=%s, port=%s, errno=%d\n",
                       ta->argv[1], ta->argv[2], errno );
            fflush ( stderr );
            ta->rv = -1;
        }
        w32_socket_exit();
    } else {
        fprintf ( stderr, "tf: Can't initialise sockets - w32_socket_init failed.\n" );
    }
    _endthreadex ( 0 );
    return ( 0 );
}

int w32_socket_init(void)
{
    int rv = 0;
    if (w32_socket_initialisations++) {
	rv = 0;
    } else {
        if (WSAStartup(0x0101, &WSAData)) {
            w32_socket_initialisations = 0;
            fprintf ( stderr, "WSAStartup failed\n" );
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


/* Start up our Graphical User Interface connecting to 
   NETWORK-ADDRESS on PORT to process PID.  If fourth
   argument WAITING causes debugging flags to be turned
   on and also causes a wait in a loop for WAITING seconds
   (giving a human debugger time to attach to the forked process).
 */

#ifdef SGC
int sgc_enabled=0;
#endif

int delay;
int main(argc, argv,envp)
int argc;
char *argv[];
char *envp[];
{
    int rv = 0; 
    {
        int i = argc;
        pstreamDebug  = stderr; 
        while (--i > 3) {
            if (strcmp(argv[i],"-delay")==0)
                { delay = atoi(argv[i+1]);}
            if (strcmp(argv[i],"-debug")==0)
                {debug = 1; fDebugSockets = -1;}
        }
    }

    if (argc >= 4) {

#ifdef _WIN32
        UINT dwThreadID;
        HANDLE hThread;
        TAS targs;
        void *pTA   = (void *) &targs;
        targs.argv  = argv;
        targs.argc  = argc;
        targs.rv    = 0;
        targs.delay = delay;

        hThread = (HANDLE) _beginthreadex (
                                            NULL,
                                            0,
                                            tf1,
                                            pTA,
                                            0,
                                            &dwThreadID
                                            );
        if ( 0 == hThread ) {
            dfprintf ( stderr, "Error: Couldn't create thread.\n" );
            rv = -1;
        }
        if ( WAIT_OBJECT_0 != WaitForSingleObject ( hThread, INFINITE ) ) {
            dfprintf ( stderr, "Error: Couldn't wait for thread to exit.\n" );
            rv = -1;
        }
        CloseHandle ( hThread );
        
#else  /* _WIN32 */
        pid_t p;

        parent = atoi(argv[3]);
        dfprintf(stderr,"guis, parent is : %d\n", parent);

#ifdef MUST_USE_VFORK
        p = vfork();
#else
        p = fork();
#endif
        dfprintf(stderr, "guis, vfork returned : %d\n", p);

        if (p == -1)
            {
                dfprintf(stderr, "Error !!! vfork failed %d\n", errno);

                return -1;
            }
        else if (p)
            {
                dfprintf(stderr, "guis,vforked child : %d\n", p);

                _exit(p);
                /*
                   return p;
                   */
            }
        else
            {

#ifndef SET_SESSION_ID	  
#if defined(__svr4__) || defined(ATT) 
#define SET_SESSION_ID() setsid()
#else
#ifdef BSD
#define SET_SESSION_ID() (setpgrp() ? -1 : 0)
#endif
#endif	  
#endif
                
                if (SET_SESSION_ID() == -1)
                    {   dfprintf(stderr, "Error !!! setsid failed : %d\n", errno);
                    }


                dsfd = sock_connect_to_name(argv[1], atoi(argv[2]), 0);
                if (dsfd) {
                    dfprintf(stderr, "connected to %s %s"
                              , argv[1], argv[2]);
                    /* give chance for someone to attach with gdb and
                       to set waiting to 0 */
                    while (-- delay >=0) sleep(1);
                    {
                        TkX_Wish(argc, argv);
                    }
                    
                    dfprintf(stderr, "Wish shell done\n");
                    
                    sock_close_connection(dsfd);
                    return 0;
                } else {
                    dfprintf(stderr,
                              "Error !!! Can't connect to socket host=%s, port=%s, errno=%d\n"
                              , argv[1], argv[2], errno);
                    fflush(stderr);
                    return -1;
                }
            }
#endif  /* _WIN32 */
    } else {
        int i;
        fprintf ( stderr, "gcltkaux: Error - expecting more arguments, but found:\n" );
        fflush(stderr);
        for ( i = 0; i<argc; i++ ) {
            fprintf ( stderr, "    argv[%d] = %s\n", i, argv[i] );
            fflush(stderr);
        }
        fflush(stderr);
        return -1;
    }
    return ( rv );
}

struct connection_state *
sock_connect_to_name(host_id,  name, async)
     char *host_id;
     int name;
     int async;
     
{
  struct sockaddr_in addr;
  int fd, n, rc;

  fd = socket( PF_INET, SOCK_STREAM, 0 );

  addr.sin_family = PF_INET;
  addr.sin_port = htons((unsigned short)(name & 0xffff));
  addr.sin_addr.s_addr = inet_addr( host_id );
  memset( addr.sin_zero, 0, 8 );
    
  n = sizeof addr;
  rc = connect( fd, (struct sockaddr *)&addr, n );
  if (rc != 0)
    return 0;

  return setup_connection_state(fd);
}

void
sock_close_connection(sfd)
struct connection_state *sfd;     
{
  close( sfd->fd );
  free(sfd->read_buffer);
  free(sfd);
  
}
  

/* #ifdef PLATFORM_SUNOS */
/* static void */
/* notice_input( int sig, int code, struct sigcontext *s, char *a ) */
/* #else */
/* static void */
/* notice_input( sig ) */
/*      int sig; */
/* #endif */
/* { */
/*   signal( SIGIO, notice_input ); */
/*   dfprintf(stderr, "\nNoticed input!\n" ); */

/* } */

static int message_id;

int
sock_write_str2( sfd, type, hdr,
		hdrsize,text, length )

struct connection_state *sfd;
enum mtype type;
 char *hdr;
int hdrsize;
const char *text;
int length;
     
{
  char buf[0x1000];
  char *p = buf;
  int m;
  int n_written;
  struct message_header *msg;
  msg = (struct message_header *) buf;

  if (length == 0)
    length = strlen(text);
  m = length + hdrsize;

  msg->magic1=MAGIC1;
  msg->magic2=MAGIC2;
  msg->type = type;
  msg->flag = 0;
  STORE_3BYTES(msg->size,m);
  STORE_3BYTES(msg->msg_id,message_id);
  message_id++;
  p = buf + MESSAGE_HEADER_SIZE;
  bcopy(hdr,p,hdrsize);
  p+= hdrsize;
  
  if (sizeof(buf) >= (length + hdrsize + MESSAGE_HEADER_SIZE))
    { bcopy(text,p,length);
      n_written = write1(sfd,buf,(length + hdrsize + MESSAGE_HEADER_SIZE));
    }
  else
    { n_written = write1(sfd,buf, hdrsize + MESSAGE_HEADER_SIZE);
      n_written += write1(sfd, text, length);
    }

  if (n_written != (length + hdrsize + MESSAGE_HEADER_SIZE))
    {perror("sock_write_str: Did not write full message");
     return -1;}
  return n_written;
  
}


#define READ_BUF_STRING_AVAIL	1
#define READ_BUF_DATA_ON_PORT	2



#define DEFAULT_TIMEOUT_FOR_TK_READ (100 * HZ)



struct message_header *
guiParseMsg1(sfd,buf,bufleng)
  char *buf;
int bufleng;
struct connection_state *sfd;
{ int m;
  int body_length;
  int tot;
  struct message_header *msg;
  msg = (struct message_header *) buf;
  m= read1(sfd,(void *)msg,MESSAGE_HEADER_SIZE,DEFAULT_TIMEOUT_FOR_TK_READ);
  if (m == MESSAGE_HEADER_SIZE)
    {
     if ( msg->magic1!=MAGIC1
	 ||  msg->magic2!=MAGIC2)
       { fprintf(stderr,"bad magic..flushing buffers");
	 while(read1(sfd,buf,bufleng,0) > 0);
	 return 0;}
      GET_3BYTES(msg->size,body_length);
      tot = body_length+MESSAGE_HEADER_SIZE;
      if (tot >= bufleng)
         {msg = (void *)malloc(tot+1);
	  bcopy(buf,msg,MESSAGE_HEADER_SIZE);}
      m = read1(sfd,(void *)&(msg->body),
		   body_length,DEFAULT_TIMEOUT_FOR_TK_READ);
     if (m == body_length)
       { return msg;}}
  if (m < 0) exit(1);
  { static int bad_read_allowed=4;
    if (bad_read_allowed-- < 0) exit(1);
  }
    
  dfprintf(stderr,"reading from lisp timed out or not enough read");
  return 0;
}  
      
void
error(s)
     char *s;
{ fprintf(stderr,"%s",s); abort();
}

void
write_timeout_error(s)
     char *s;
{ fprintf(stderr,"write timeout: %s",s); abort();
}
void
connection_failure(s)
     char *s;
{ fprintf(stderr,"connection_failure:%s",s); abort();
}
