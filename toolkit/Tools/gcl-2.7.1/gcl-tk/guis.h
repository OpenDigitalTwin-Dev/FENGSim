#ifndef _GUIS_H_
#define _GUIS_H_

#include <stdlib.h>

#define NO_PRELINK_UNEXEC_DIVERSION
#define IMMNUM_H
#define GMP_WRAPPERS_H
#define ERROR_H
#undef INLINE

#include "include.h"

#ifdef NeXT
typedef int pid_t;
#endif

#ifndef _ANSI_ARGS_
#ifdef __STDC__
#define _ANSI_ARGS_(x) x
#else
#define _ANSI_ARGS_(x) ()
#endif
#endif

#define STRING_HEADER_FORMAT	"%4.4d"
#define CB_STRING_HEADER	(5)
/*
#define GET_STRING_SIZE_FROM_HEADER(__buf, __plgth)	\
sscanf(__buf, STRING_HEADER_FORMAT, __plgth);
*/

/* sscanf is braindead on SunOS */
#define GET_STRING_SIZE_FROM_HEADER(__buf, __plgth)	\
{\
   __buf[CB_STRING_HEADER - 1] = 0;\
   *__plgth = atoi(__buf);\
   __buf[4] = '';\
}

/* need to have opportunity to collapse message to reduce trafic */
#define MSG_STRAIGHT_TCL_CMD		0
#define MSG_CREATE_COMMAND	1
/*
#define MSG_
*/

typedef struct _guiMsg {

  pid_t pidSender;
  int vMajor;
  int vMinor;
  int idx;
  int fSignal;
  int fAck;
  int IdMsg;
  char *szData;
  char *szMsg;

} guiMsg;

#define MSG_IDX(__p)			(__p->idx)
#define MSG_COMMAND(__p)		(__p->IdMsg)
#define MSG_NEED_ACK(__p)		(__p->fAck)
#define MSG_NEED_SIGNAL_PARENT(__p)	(__p->fSignal)
#define MSG_TCL_STR(__p)		(__p->szData)
#define MSG_DATA_STR(__p)		(__p->szData)
/*
#define MSG_(__p)		(__p->)
*/

#include "sheader.h"
struct message_header * guiParseMsg1();


extern pid_t parent;

struct connection_state *
sock_connect_to_name();
void sock_close_connection( );
int sock_read_str();

guiMsg *guiParseMsg();
void guiFreeMsg();

void
guiCreateThenBindCallback();
int guiBindCallback();

#endif

int
sock_write_str2(struct connection_state *,enum mtype, char *,
		int,const char *,int);


object
fSclear_connection(fixnum);


object
fScheck_fd_for_input(fixnum,fixnum);

#define SI_makefun(a_,b_,c_)
