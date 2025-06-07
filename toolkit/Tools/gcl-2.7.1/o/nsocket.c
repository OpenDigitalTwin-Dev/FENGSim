/* Copyright (C) 2024 Camm Maguire */
/* the following file compiles under win95 using cygwinb19 */ 
#include "include.h"

#ifdef DODEBUG
#define dprintf(s,arg) emsg(s,arg)
#else 
#define dprintf(s,arg)
#endif     

#ifdef HAVE_NSOCKET



#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>



/************* for the sockets ******************/ 
#include <sys/socket.h>		/* struct sockaddr, SOCK_STREAM, ... */
#ifndef NO_UNAME
#   include <sys/utsname.h>	/* uname system call. */
#endif
#include <netinet/in.h>		/* struct in_addr, struct sockaddr_in */
#include <arpa/inet.h>		/* inet_ntoa() */
#include <netdb.h>		/* gethostbyname() */

/****************end for sockets *******************/



/*
 * These bits may be ORed together into the "flags" field of a TcpState
 * structure.
 */


#define TCP_ASYNC_SOCKET	(1<<0)	/* Asynchronous socket. */
#define TCP_ASYNC_CONNECT	(1<<1)	/* Async connect in progress. */

/*
 * The following defines the maximum length of the listen queue. This is
 * the number of outstanding yet-to-be-serviced requests for a connection
 * on a server socket, more than this number of outstanding requests and
 * the connection request will fail.
 */

#ifndef	SOMAXCONN
#define SOMAXCONN	100
#endif

#if	(SOMAXCONN < 100)
#undef	SOMAXCONN
#define	SOMAXCONN	100
#endif

#define VOID void
#define ERROR_MESSAGE(msg)     do{ emsg(msg); gcl_abort() ; } while(0)

#ifdef STAND

main(argc,argv)
     char *argv[];
     int argc;
{
  char buf[1000];
  char out[1000];
  char op[10];
  int n,fd;
  int x,y,ans,errno;
  char *bp;
  fd_set readfds;
  struct timeval timeout;
  
  
  bp = buf;
  fd = doConnect(argv[1],atoi(argv[2]));
  if (fd < 0) {
    perror("cant connect");
    do_gcl_abort();
  }

  while (1) { int high;
    timeout.tv_sec = 20;
    timeout.tv_usec = 0;
    FD_ZERO(&readfds);
    FD_SET(fd,&readfds);
    
    high = select(fd+1,&readfds,NULL,NULL,&timeout);
    if (high > 0)
      {
	int n;
	n = read(fd,buf,sizeof(buf));
	if (3 == sscanf(buf,"%d %s %d",&x,op,&y)) {
	  switch (op[0]) {
	    
	  case '+':  	  sprintf(out,"%d\n",x+y);
	    break;
	  case '*':  sprintf(out,"%d\n",x*y);
	    break;
	  default:
	    sprintf(out,"bad operation\n");
	  }
	  write(fd,out,strlen(out));
	}
      }
  }
}

#endif


/*
 *----------------------------------------------------------------------
 *
 * CreateSocketAddress --
 *
 *	This function initializes a sockaddr structure for a host and port.
 *
 * Results:
 *	1 if the host was valid, 0 if the host could not be converted to
 *	an IP address.
 *
 * Side effects:
 *	Fills in the *sockaddrPtr structure.
 *
 *----------------------------------------------------------------------
 */

static int
CreateSocketAddress(struct sockaddr_in *sockaddrPtr, char *host, int port)
                                    	/* Socket address */
               				/* Host.  NULL implies INADDR_ANY */
             				/* Port number */
{
    struct hostent *hostent;		/* Host database entry */
    struct in_addr addr;		/* For 64/32 bit madness */

    (void) memset((VOID *) sockaddrPtr, '\0', sizeof(struct sockaddr_in));
    sockaddrPtr->sin_family = AF_INET;
    sockaddrPtr->sin_port = htons((unsigned short) (port & 0xFFFF));
    if (host == NULL) {
	addr.s_addr = INADDR_ANY;
    } else {
        addr.s_addr = inet_addr(host);
        if (addr.s_addr == -1) {
            hostent = 
#ifdef STATIC_LINKING
	      NULL;
#else
	    gethostbyname(host);
#endif
            if (hostent != NULL) {
                memcpy((VOID *) &addr,
                        (VOID *) hostent->h_addr_list[0],
                        (size_t) hostent->h_length);
            } else {
#ifdef	EHOSTUNREACH
                errno = EHOSTUNREACH;
#else
#ifdef ENXIO
                errno = ENXIO;
#endif
#endif
                return 0;	/* error */
            }
        }
    }
        
    /*
     * NOTE: On 64 bit machines the assignment below is rumored to not
     * do the right thing. Please report errors related to this if you
     * observe incorrect behavior on 64 bit machines such as DEC Alphas.
     * Should we modify this code to do an explicit memcpy?
     */

    sockaddrPtr->sin_addr.s_addr = addr.s_addr;
    return 1;	/* Success. */
}



/* return -1 on failure, or else an fd */
int 
CreateSocket(int port, char *host, int server, char *myaddr, int myport, int async)
             			/* Port number to open. */
               			/* Name of host on which to open port.
				 * NULL implies INADDR_ANY */
               			/* 1 if socket should be a server socket,
				 * else 0 for a client socket. */
                 		/* Optional client-side address */
               			/* Optional client-side port */
              			/* If nonzero and creating a client socket,
                                 * attempt to do an async connect. Otherwise
                                 * do a synchronous connect or bind. */
{
    int status, sock, /* asyncConnect,  */curState, origState;
    struct sockaddr_in sockaddr;	/* socket address */
    struct sockaddr_in mysockaddr;	/* Socket address for client */

    sock = -1;
    origState = 0;
    if (! CreateSocketAddress(&sockaddr, host, port)) {
	goto addressError;
    }
    if ((myaddr != NULL || myport != 0) &&
	    ! CreateSocketAddress(&mysockaddr, myaddr, myport)) {
	goto addressError;
    }

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
	goto addressError;
    }

    /*
     * Set the close-on-exec flag so that the socket will not get
     * inherited by child processes.
     */

    fcntl(sock, F_SETFD, FD_CLOEXEC);
    
    /* asyncConnect = 0; */
    status = 0;
    if (server) {

	/*
	 * Set up to reuse server addresses automatically and bind to the
	 * specified port.
	 */
    
	status = 1;
	(void) setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *) &status,
		sizeof(status));
	status = bind(sock, (struct sockaddr *) &sockaddr,
                sizeof(struct sockaddr));
	if (status != -1) {
	    status = listen(sock, SOMAXCONN);
	} 
    } else {
	if (myaddr != NULL || myport != 0) { 
	    curState = 1;
	    (void) setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                    (char *) &curState, sizeof(curState));
	    status = bind(sock, (struct sockaddr *) &mysockaddr,
		    sizeof(struct sockaddr));
	    if (status < 0) {
		goto bindError;
	    }
	}

	/*
	 * Attempt to connect. The connect may fail at present with an
	 * EINPROGRESS but at a later time it will complete. The caller
	 * will set up a file handler on the socket if she is interested in
	 * being informed when the connect completes.
	 */

        if (async) {
#ifndef	USE_FIONBIO
            origState = fcntl(sock, F_GETFL);
            curState = origState | O_NONBLOCK;
            status = fcntl(sock, F_SETFL, curState);
#endif

#ifdef	USE_FIONBIO
            curState = 1;
            status = ioctl(sock, FIONBIO, &curState);
#endif            
        } else {
            status = 0;
        }
        if (status > -1) {
            status = connect(sock, (struct sockaddr *) &sockaddr,
                    sizeof(sockaddr));
            if (status < 0) {
                if (errno == EINPROGRESS) {
                    /* asyncConnect = 1; */
                    status = 0;
                }
            }
        }
    }

bindError:
    if (status < 0) {

            ERROR_MESSAGE("couldn't open socket:");

        if (sock != -1) {
            close(sock);
        }
        return -1;
    }

      return sock;

addressError:
    if (sock != -1) {
        close(sock);
    }

      ERROR_MESSAGE("couldn't open socket:");

    return -1;
}


#ifdef STAND
int
doConnect(host,port)
     	  char *host;          /*name of host we are trying to connect to */
	  int port;            /* port number to use */
{
    return CreateSocket(port, host, 0 , NULL , 0 , 0);    
}
#endif



#define SOCKET_FD(strm) ((strm)->sm.sm_fp ? fileno((strm)->sm.sm_fp) : -1)

DEFUN("GETPEERNAME",object,fSgetpeername,SI,1,1,NONE,OO,OO,OO,OO,(object sock),
 "Return a list of three elements: the address, the hostname and the port for the other end of the socket.  If hostname is not available it will be equal to the address.  Invalid on server sockets. Return NIL on failure.")
{
 struct sockaddr_in peername;
 socklen_t size = sizeof(struct sockaddr_in);
 struct hostent *hostEntPtr;
 object address,host;
 check_socket(sock);
 if (getpeername(SOCKET_FD(sock), (struct sockaddr *) &peername, &size)
		>= 0) {
           address=make_simple_string(inet_ntoa(peername.sin_addr));
           hostEntPtr = 
#ifdef STATIC_LINKING
	     NULL;
#else
	   gethostbyaddr((char *) &(peername.sin_addr),
			 sizeof(peername.sin_addr), AF_INET);
#endif
            if (hostEntPtr != (struct hostent *) NULL) 
               host = make_simple_string(hostEntPtr->h_name);
            else host = address;
	    return list(3,address,host,make_fixnum(ntohs(peername.sin_port)));
 } else {
   return Cnil;
 }
}
	    

DEFUN("GETSOCKNAME",object,fSgetsockname,SI,1,1,NONE,OO,OO,OO,OO,(object sock),
 "Return a list of three elements: the address, the hostname and the port for the socket.  If hostname is not available it will be equal to the address. Return NIL on failure. ")
{ struct sockaddr_in sockname;
 socklen_t size = sizeof(struct sockaddr_in);
 struct hostent *hostEntPtr;
 object address,host;

 check_socket(sock);
 if (getsockname(SOCKET_FD(sock), (struct sockaddr *) &sockname, &size)
		>= 0) {
  address= make_simple_string(inet_ntoa(sockname.sin_addr));
  hostEntPtr = 
#ifdef STATIC_LINKING
    NULL;
#else
  gethostbyaddr((char *) &(sockname.sin_addr),
		sizeof(sockname.sin_addr), AF_INET);
#endif
  if (hostEntPtr != (struct hostent *) NULL)
   host = make_simple_string(hostEntPtr->h_name);
  else host=address;
  return list(3,address,host,make_fixnum(ntohs(sockname.sin_port)));
 } else {
   return Cnil;
 }
}

/*
  TcpBlocking --
    Use on a tcp socket to alter the blocking or non blocking.
  Results 0 if succeeds and errno if fails.

  Side effects:
     the channel is setto blocking or nonblocking mode. 
*/  

DEFUN("SET-BLOCKING",object,fSset_blocking,SI,2,2,NONE,OO,OO,OO,OO,(object sock,object setBlocking),
      "Set blocking on if MODE is T otherwise off.  Return 0 if succeeds. Otherwise the error number.")
{
      int setting;
      int fd ;
   AGAIN:
      check_stream(sock);
      /* set our idea of whether blocking on or off 
        setBlocking==Cnil <==> blocking turned off.  */
     SET_STREAM_FLAG(sock,gcl_sm_tcp_async,setBlocking==Cnil);
      if (sock->sm.sm_mode == smm_two_way) {
	/* check for case they are sock streams and so
	   share the same fd */
	if (STREAM_INPUT_STREAM(sock)->sm.sm_fp != NULL
	    &&STREAM_OUTPUT_STREAM(sock)->sm.sm_fp != NULL
	    && (SOCKET_FD(STREAM_INPUT_STREAM(sock))==
		SOCKET_FD(STREAM_OUTPUT_STREAM(sock))))
	  {
	    SET_STREAM_FLAG(STREAM_OUTPUT_STREAM(sock),
			    gcl_sm_tcp_async,setBlocking==Cnil);
	    sock = STREAM_INPUT_STREAM(sock);
	    /* they share an 'fd' and so only do one. */
	    goto AGAIN;
	  }
	else
	{
	  int x1 = fix(FFN(fSset_blocking)(STREAM_INPUT_STREAM(sock),setBlocking));
	  int x2 = fix(FFN(fSset_blocking)(STREAM_OUTPUT_STREAM(sock),setBlocking));
	  /* if either is negative result return negative. (ie fail)
	     If either is positive return positive (ie fail)
	     Zero result means both ok.  (ie succeed)
	     */       
	  
	  return make_fixnum((x1 < 0 || x2 < 0 ? -2 : x1 > 0  ? x1 : x2));
	}
      }
	
      if (sock->sm.sm_fp == NULL)
	return make_fixnum(-2);
      fd = SOCKET_FD(sock);


#ifndef	USE_FIONBIO
    setting = fcntl(fd, F_GETFL);
    if (setBlocking != Cnil) {
      setting &= (~(O_NONBLOCK));
    } else {
      setting |= O_NONBLOCK;
    }
    if (fcntl(fd, F_SETFL, setting) < 0) {
        return make_fixnum(errno);
    }
#endif

#ifdef	USE_FIONBIO
    if (setBlocking != Cnil) {
        setting = 0;
        if (ioctl(fd, (int) FIONBIO, &setting) == -1) {
            return make_fixnum(errno);
        }
    } else {
        setting = 1;
        if (ioctl(fd, (int) FIONBIO, &setting) == -1) {
            return make_fixnum(errno);
        }
    }
#endif
  return make_fixnum(0);
}

/* with 2 args return the function if any.
*/

/*setHandler(stream,readable,function)
     object stream;     stream to watch 
     object readable;   keyword readable,writable 
     object function;   the handler function to be invoked with arg stream 
{
  
}
*/
/* goes through the streams does a select with 0 timeout, and invokes
   any handlers */
/*
update ()
{

}
*/

static int
joe(int x) { return x; }

/*
  get a character from FP but block, if it would return
  the EOF, but the stream is not closed.
*/   
int
getOneChar(FILE *fp)
{
  fd_set readfds;
  struct timeval timeout;
  int fd= fileno(fp);
  int high;
  /*  fprintf(stderr,"<socket 0x%x>",fp);
  fflush(stderr); */
  emsg("in getOneChar, fd=%d,fp=%p",fd,fp);
  if (fd == 0)
   { joe(fd);
   return -1;
   }

  while (1) {
  timeout.tv_sec = 0;
  timeout.tv_usec = 200000;
  FD_ZERO(&readfds);
  FD_SET(fd,&readfds);
  CHECK_INTERRUPT;	 
  high = select(fd+1,&readfds,NULL,NULL,&timeout);
  if (high > 0)
    {
      int ch ;
      emsg("in getOneChar, fd=%d,fp=%p",fd,fp);
      ch = getc(fp);
      if ( ch != EOF || feof(fp) ) {
	/*      fprintf(stderr,"< 0x%x returning %d,%c>\n",fp,ch,ch);
      fflush(stderr);
      */
      }
      emsg("in getOneChar, ch= %c,%d\n",ch,ch);
      CHECK_INTERRUPT;	 
      if (ch != EOF) return ch;
      if (feof(fp)) return EOF;
    }
     
  }
}

#ifdef DODEBUG
#define dprintf(s,arg) emsg(s,arg)
#else 
#define dprintf(s,arg)
#endif     
     
void
ungetCharGclSocket(int c, object strm)
                  /* the character to unget */
                  /* stream */
{  object bufp = SOCKET_STREAM_BUFFER(strm);
  if (c == EOF) return;
  dprintf("pushing back %c\n",c);
  if (bufp->ust.ust_fillp < bufp->ust.ust_dim) {
    bufp->ust.ust_self[(bufp->ust.ust_fillp)++]=c;
  } else {
    FEerror("Tried to unget too many chars",0);
  }
}


/*
 *----------------------------------------------------------------------
 *
 * TcpOutputProc --
 *
 *	This procedure is invoked by the generic IO level to write output
 *	to a TCP socket based channel.
 *
 *	NOTE: We cannot share code with FilePipeOutputProc because here
 *	we must use send, not write, to get reliable error reporting.
 *
 * Results:
 *	The number of bytes written is returned. An output argument is
 *	set to a POSIX error code if an error occurred, or zero.
 *
 * Side effects:
 *	Writes output on the output device of the channel.
 *
 *----------------------------------------------------------------------
 */

int
TcpOutputProc(int fd, char *buf, int toWrite, int *errorCodePtr)
            		/* Socket state. */
              				/* The data buffer. */
                			/* How many bytes to write? */
                      			/* Where to store error code. */
{
    int written;

    *errorCodePtr = 0;
    written = send(fd, buf, (size_t) toWrite, 0);
    if (written > -1) {
        return written;
    }
    *errorCodePtr = errno;
    return -1;
}

void
tcpCloseSocket(int fd)
{
  close(fd);

}

static void
doReverse(char *s, int n)
{ char *p=&s[n-1];
  int m = n/2;
  while (--m>=0) {
    int tem = *s;
    *s = *p;
    *p = tem;
    s++; p--;
  }
}



/*
  getCharGclSocket(strm,block) -- get one character from a socket
  stream.
  Results: a character or EOF if at end of file
  Side Effects:  The buffer may be filled, and the fill pointer
  of the buffer may be changed.
 */
int
getCharGclSocket(object strm, object block) {

  object bufp=SOCKET_STREAM_BUFFER(strm);
  int fd=SOCKET_STREAM_FD(strm);

  if (VLEN(bufp) > 0)
    return bufp->ust.ust_self[--(bufp->ust.ust_fillp)];

  if (fd>=0) {

    fd_set readfds;
    struct timeval t,t1={0,10000},*tp=block==Ct ? NULL : &t;
    int high,n;

    FD_ZERO(&readfds);
    FD_SET(fd,&readfds);

    for (;(errno=0,t=t1,high=select(fd+1,&readfds,NULL,NULL,tp))==-1 && !tp && errno==EINTR;);

    if (high > 0) {

      massert((n=SAFE_READ(fd,bufp->st.st_self,bufp->ust.ust_dim))>=0);

      if (n) {
	doReverse(bufp->st.st_self,n);
	bufp->ust.ust_fillp=n;
      } else
	SOCKET_STREAM_FD(strm)=-1;

      return getCharGclSocket(strm,block);

    }

  }

  return EOF;

}

#else
int
getOneChar(fp)
     FILE *fp;
{
  return getc(fp);
}

#endif



