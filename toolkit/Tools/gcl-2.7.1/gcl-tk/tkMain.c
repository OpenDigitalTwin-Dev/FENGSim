/* 
 * main.c --
 *
 *	This file contains the main program for "wish", a windowing
 *	shell based on Tk and Tcl.  It also provides a template that
 *	can be used as the basis for main programs for other Tk
 *	applications.
 *
 * Copyright (c) 1990-1993 The Regents of the University of California.
 * Copyright (c) 2024 Camm Maguire
 * All rights reserved.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 * 
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF
 * CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

/*  #ifndef lint */
/*  static char rcsid[] = "$Header$ SPRITE (Berkeley)"; */
/*  #endif */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <tcl.h>
#include <tk.h>
#if TCL_MAJOR_VERSION >= 9
#include <wordexp.h>
#endif


#if (TK_MINOR_VERSION==0 && TK_MAJOR_VERSION==4)
#define TkCreateMainWindow Tk_CreateMainWindow
#endif
#if TCL_MAJOR_VERSION >= 8
#define INTERP_RESULT(interp) Tcl_GetStringResult(interp)
#else
#define INTERP_RESULT(interp) (interp)->result
#endif


/*-------------------------------------------------------------------*/
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <errno.h>

int writable_malloc=0; /*FIXME, don't wrap fopen here, exclude notcomp.h or equivalent */

#include "guis.h"
struct connection_state *dsfd;
/*-------------------------------------------------------------------*/

/*
 * Declarations for various library procedures and variables (don't want
 * to include tkInt.h or tkConfig.h here, because people might copy this
 * file out of the Tk source directory to make their own modified versions).
 */

/* extern void		exit _ANSI_ARGS_((int status));
extern int		isatty _ANSI_ARGS_((int fd));
extern int		read _ANSI_ARGS_((int fd, char *buf, size_t size));
extern char *		strrchr _ANSI_ARGS_((CONST char *string, int c));
*/
extern int Tcl_AppInit(Tcl_Interp *interp);

/*
 * Global variables used by the main program:
 */

/* static Tk_Window mainWindow;    The main window for the application.  If
				 * NULL then the application no longer
				 * exists. */
static Tcl_Interp *interp;	/* Interpreter for this application. */
char *tcl_RcFileName;		/* Name of a user-specific startup script
				 * to source if the application is being run
				 * interactively (e.g. "~/.wishrc").  Set
				 * by Tcl_AppInit.  NULL means don't source
				 * anything ever. */
static Tcl_DString command;	/* Used to assemble lines of terminal input
				 * into Tcl commands. */
static int tty;			/* Non-zero means standard input is a
				 * terminal-like device.  Zero means it's
				 * a file. */
static char errorExitCmd[] = "exit 1";

/*
 * Command-line options:
 */

static int synchronize = 0;
static char *fileName = NULL;
static char *name = NULL;
static char *display = NULL;
static char *geometry = NULL;
int debug = 0;

static void guiCreateCommand _ANSI_ARGS_((int idLispObject, int iSlot , char *arglist));

void
dfprintf(FILE *fp,char *s,...) {

  va_list args;

  if (debug) {
    va_start(args,s);
    fprintf(fp,"\nguis:");
    vfprintf(fp,s,args);
    fflush(fp);
    va_end(args);
  }
}

#define CMD_SIZE 4000
#define SIGNAL_ERROR TCL_signal_error

static void
TCL_signal_error(x)
     char *x;
{char buf[300] ;
  snprintf(buf,sizeof(buf),"error %s",x);
 Tcl_Eval(interp,buf);
 dfprintf(stderr,x);
}



static Tk_ArgvInfo argTable[] = {
    {"-file", TK_ARGV_STRING, (char *) NULL, (char *) &fileName,
	"File from which to read commands"},
    {"-geometry", TK_ARGV_STRING, (char *) NULL, (char *) &geometry,
	"Initial geometry for window"},
    {"-display", TK_ARGV_STRING, (char *) NULL, (char *) &display,
	"Display to use"},
    {"-name", TK_ARGV_STRING, (char *) NULL, (char *) &name,
	"Name to use for application"},
    {"-sync", TK_ARGV_CONSTANT, (char *) 1, (char *) &synchronize,
	"Use synchronous mode for display server"},
    {(char *) NULL, TK_ARGV_END, (char *) NULL, (char *) NULL,
	(char *) NULL}
};

/*
 * Declaration for Tcl command procedure to create demo widget.  This
 * procedure is only invoked if SQUARE_DEMO is defined.
 */

extern int SquareCmd _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, int argc, char *argv[]));

/*
 * Forward declarations for procedures defined later in this file:
 */

static void		StdinProc _ANSI_ARGS_((ClientData clientData,
			    int mask));

/*
 *----------------------------------------------------------------------
 *
 * main --
 *
 *	Main program for Wish.
 *
 * Results:
 *	None. This procedure never returns (it exits the process when
 *	it's done
 *
 * Side effects:
 *	This procedure initializes the wish world and then starts
 *	interpreting commands;  almost anything could happen, depending
 *	on the script being interpreted.
 *
 *----------------------------------------------------------------------
 */
/*
int
main(argc, argv)
*/

/* FIXME, should come in from tk header or not be called */
EXTERN Tk_Window	TkCreateMainWindow _ANSI_ARGS_((Tcl_Interp * interp, 
				char * screenName, char * baseName));

void
TkX_Wish (argc, argv)
    int argc;				/* Number of arguments. */
    char **argv;			/* Array of argument strings. */
{
    char *args, *p;
    const char *msg;
    char buf[20];
    int code;

    interp = Tcl_CreateInterp();
#ifdef TCL_MEM_DEBUG
    Tcl_InitMemory(interp);
#endif

    /*
     * Parse command-line arguments.
     */

    if (Tk_ParseArgv(interp, (Tk_Window) NULL, &argc, (void *)argv, argTable, 0)
	    != TCL_OK) {
	fprintf(stderr, "%s\n", INTERP_RESULT(interp));
	exit(1);
    }
    if (name == NULL) {
	if (fileName != NULL) {
	    p = fileName;
	} else {
	    p = argv[0];
	}
	name = strrchr(p, '/');
	if (name != NULL) {
	    name++;
	} else {
	    name = p;
	}
    }

    /*
     * If a display was specified, put it into the DISPLAY
     * environment variable so that it will be available for
     * any sub-processes created by us.
     */

    if (display != NULL) {
	Tcl_SetVar2(interp, "env", "DISPLAY", display, TCL_GLOBAL_ONLY);
    }

    /*
     * Initialize the Tk application.
     */

/*     mainWindow = TkCreateMainWindow(interp, display, name/\*  , "Tk" *\/);  */
/*     if (mainWindow == NULL) { */
/* 	fprintf(stderr, "%s\n", INTERP_RESULT(interp)); */
/* 	exit(1); */
/*     } */
/* #ifndef __MINGW32__     */
/*     if (synchronize) { */
/* 	XSynchronize(Tk_Display(mainWindow), True); */
/*     } */
/* #endif     */
/*     Tk_GeometryRequest(mainWindow, 200, 200); */
/*     Tk_UnmapWindow(mainWindow); */

    /*
     * Make command-line arguments available in the Tcl variables "argc"
     * and "argv".  Also set the "geometry" variable from the geometry
     * specified on the command line.
     */

    args = Tcl_Merge(argc-1, (const char **)argv+1);
    Tcl_SetVar(interp, "argv", args, TCL_GLOBAL_ONLY);
    ckfree(args);
    sprintf(buf, "%d", argc-1);
    Tcl_SetVar(interp, "argc", buf, TCL_GLOBAL_ONLY);
    Tcl_SetVar(interp, "argv0", (fileName != NULL) ? fileName : argv[0],
	    TCL_GLOBAL_ONLY);
    if (geometry != NULL) {
	Tcl_SetVar(interp, "geometry", geometry, TCL_GLOBAL_ONLY);
    }

    /*
     * Set the "tcl_interactive" variable.
     */

    tty = isatty(dsfd->fd);
    Tcl_SetVar(interp, "tcl_interactive",
	    ((fileName == NULL) && tty) ? "1" : "0", TCL_GLOBAL_ONLY);

    /*
     * Add a few application-specific commands to the application's
     * interpreter.
     */

/* #ifdef SQUARE_DEMO */
/*     Tcl_CreateCommand(interp, "square", SquareCmd, (ClientData) mainWindow, */
/* 	    (void (*)()) NULL); */
/* #endif */

    /*
     * Invoke application-specific initialization.
     */

    if (Tcl_AppInit(interp) != TCL_OK) {
	fprintf(stderr, "Tcl_AppInit failed: %s\n", INTERP_RESULT(interp));
    }

    /*
     * Set the geometry of the main window, if requested.
     */

    if (geometry != NULL) {
	code = Tcl_VarEval(interp, "wm geometry . ", geometry, (char *) NULL);
	if (code != TCL_OK) {
	    fprintf(stderr, "%s\n", INTERP_RESULT(interp));
	}
    }

    /*
     * Invoke the script specified on the command line, if any.
     */

    if (fileName != NULL) {
	code = Tcl_VarEval(interp, "source ", fileName, (char *) NULL);
	if (code != TCL_OK) {
	    goto error;
	}
	tty = 0;
    } else {
	/*
	 * Commands will come from standard input, so set up an event
	 * handler for standard input.  If the input device is aEvaluate the
	 * .rc file, if one has been specified, set up an event handler
	 * for standard input, and print a prompt if the input
	 * device is a terminal.
	 */

	if (tcl_RcFileName != NULL) {
	    char *fullName;
	    FILE *f;
#if TCL_MAJOR_VERSION >= 9
	    wordexp_t exp_result;
	    wordexp(tcl_RcFileName, &exp_result, WRDE_NOCMD);
	    fullName = exp_result.we_wordv[0];
#else
	    Tcl_DString buffer;
    
	    fullName = Tcl_TildeSubst(interp, tcl_RcFileName, &buffer);
#endif
	    if (fullName == NULL) {
		fprintf(stderr, "%s\n", INTERP_RESULT(interp));
	    } else {
		f = fopen(fullName, "r");
		if (f != NULL) {
		    code = Tcl_EvalFile(interp, fullName);
		    if (code != TCL_OK) {
			fprintf(stderr, "%s\n", INTERP_RESULT(interp));
		    }
		    fclose(f);
		}
	    }
#if TCL_MAJOR_VERSION >= 9
	    wordfree(&exp_result);
#else
	    Tcl_DStringFree(&buffer);
#endif
	}

	dfprintf(stderr, "guis : Creating file handler for %d\n", dsfd->fd);
#ifndef __MINGW32__	
	Tcl_CreateFileHandler(dsfd->fd, TCL_READABLE, StdinProc, (ClientData) 0);
#endif        
    }
    fflush(stdout);
    Tcl_DStringInit(&command);

    /*
     * Loop infinitely, waiting for commands to execute.  When there
     * are no windows left, Tk_MainLoop returns and we exit.
     */

    Tk_MainLoop();

    /*
     * Don't exit directly, but rather invoke the Tcl "exit" command.
     * This gives the application the opportunity to redefine "exit"
     * to do additional cleanup.
     */

    Tcl_Eval(interp, "exit");
    exit(1);

error:
    msg = Tcl_GetVar(interp, "errorInfo", TCL_GLOBAL_ONLY);
    if (msg == NULL) {
	msg = INTERP_RESULT(interp);
    }
    dfprintf(stderr, "%s\n", msg);
    Tcl_Eval(interp, errorExitCmd);
    return;			/* Needed only to prevent compiler warnings. */
}

static char *being_set_by_lisp;

static char *
tell_lisp_var_changed(
                clientData,
               interp,
               name1,
               name2,
                flags)

          ClientData clientData;
               Tcl_Interp *interp;
               char *name1;
               char *name2;
               int flags;     
     
{

  if (being_set_by_lisp == 0)
    { const char *val = Tcl_GetVar2(interp,name1,name2, TCL_GLOBAL_ONLY);
      char buf[3];
      STORE_3BYTES(buf,(long) clientData);
      if(sock_write_str2(dsfd,   m_set_lisp_loc, buf, 3 ,
				 val, strlen(val))
		 < 0)
		{		/* what do we want to do if the write failed */}
#ifndef __MINGW32__	      
    if (parent > 0)  kill(parent, SIGUSR1);
#endif      
    }
  else
  /* avoid going back to lisp if it is lisp that is doing the setting! */
    if (strcmp(being_set_by_lisp,name1))
      { fprintf(stderr,"recursive setting of vars %s??",name1);}
  /* normal */
  return 0;
}


/*
 *----------------------------------------------------------------------
 *
 * StdinProc --
 *
 *	This procedure is invoked by the event dispatcher whenever
 *	standard input becomes readable.  It grabs the next line of
 *	input characters, adds them to a command being assembled, and
 *	executes the command if it's complete.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Could be almost arbitrary, depending on the command that's
 *	typed.
 *
 *----------------------------------------------------------------------
 */

    /* ARGSUSED */
static void
StdinProc(clientData, mask)
     ClientData clientData;		/* Not used. */
     int mask;				/* Not used. */
{
  int fNotDone;
  char *cmd;
  int code, count;
  struct message_header *msg;
  char buf[0x4000];
  msg = (struct message_header *) buf;

  /*
   * Disable the stdin file handler while evaluating the command;
   * otherwise if the command re-enters the event loop we might
   * process commands from stdin before the current command is
   * finished.  Among other things, this will trash the text of the
   * command being evaluated.
   */
  dfprintf(stderr, "\nguis : Disabling file handler for %d\n", dsfd->fd);

/*  Tcl_CreateFileHandler(dsfd->fd, 0, StdinProc, (ClientData) 0); */

  do
    { 

      msg = guiParseMsg1(dsfd,buf,sizeof(buf));

      if (msg == NULL)
	{
	  /*dfprintf(stderr, "Yoo !!! Empty command\n"); */
	  if (debug)perror("zero message");
#ifndef __MINGW32__          
	  Tcl_CreateFileHandler(dsfd->fd, TCL_READABLE, StdinProc, (ClientData) 0);
#endif          
	  return;
	}

      /* Need to switch to table lookup */
      switch (msg->type){
      case m_create_command:
	  {
	    int iSlot;
	    GET_3BYTES(msg->body,iSlot);
	    guiCreateCommand(0, iSlot, &(msg->body[3]));
	  }
	  break;
	case  m_tcl_command :
	case m_tcl_command_wait_response:
	  count = strlen(msg->body);
	  cmd = Tcl_DStringAppend(&command, msg->body, count);

	  code = Tcl_RecordAndEval(interp, cmd, 0);

	  if (msg->type == m_tcl_command_wait_response
	      || code)
	    {
	      char buf[4];
	      char *p = buf, *string;
	      /*header */
	      *p++ = (code ? '1' : '0');
	      bcopy(msg->msg_id,p,3);
	      /* end header */
	      string = (char *)INTERP_RESULT(interp);
	      if(sock_write_str2(dsfd, m_reply, buf, 4, string, strlen(string))
		 < 0)
		{		/* what do we want to do if the write failed */}
	      
	      if (msg->type == m_tcl_command_wait_response)
		{ /* parent is waiting so dong signal */ ;}
#ifndef __MINGW32__              
	      else
		if (parent> 0)kill(parent, SIGUSR1);
#endif              
	    }

	  Tcl_DStringFree(&command);
	  break;
	case m_tcl_clear_connection:
	  /* we are stuck... */
	  {
	    Tcl_DStringInit(&command);
	    Tcl_DStringFree(&command);
	    fSclear_connection(dsfd->fd);
	  }
	  break;
	case m_tcl_set_text_variable:
	  { int n = strlen(msg->body);
	    if(being_set_by_lisp) fprintf(stderr,"recursive set?");
	    /* avoid a trace on this set!! */
	    
	    being_set_by_lisp = msg->body;
	    Tcl_SetVar2(interp,msg->body,0,msg->body+n+1,
			TCL_GLOBAL_ONLY);
	    being_set_by_lisp = 0;
	     }
	  break;

	case m_tcl_link_text_variable:
	  {long i;
	   GET_3BYTES(msg->body,i);
	   Tcl_TraceVar2(interp,msg->body+3 ,0,
			   TCL_TRACE_WRITES
			   | TCL_TRACE_UNSETS
			   | TCL_GLOBAL_ONLY
			   , tell_lisp_var_changed, (ClientData) i);
	 }
	   break;

	case m_tcl_unlink_text_variable:
	  {long i;
	   GET_3BYTES(msg->body,i);
	   Tcl_UntraceVar2(interp,msg->body+3 ,0,
			   TCL_TRACE_WRITES
			   | TCL_TRACE_UNSETS
			   | TCL_GLOBAL_ONLY
			   , tell_lisp_var_changed, (ClientData) i);
	 }
	  break;

	default :
	  dfprintf(stderr, "Error !!! Unknown command %d\n"
		   , msg->type);
	}
      fNotDone = fScheck_dsfd_for_input(dsfd,0);
      
      if (fNotDone > 0)
	{
	  dfprintf(stderr, "\nguis : in StdinProc, not done, executed %s"
		  ,  msg->body);

	}
    } while (fNotDone > 0);


  /* Tcl_CreateFileHandler(dsfd->fd, TCL_READABLE, StdinProc, (ClientData) 0); */
  if ((void *)msg != (void *) buf)
    free ((void *) msg);
}

/* ----------------------------------------------------------------- */
typedef struct _ClientDataLispObject {
  int id;
  int iSlot;
  char *arglist;
} ClientDataLispObject;

static int
TclGenericCommandProcedure( clientData,
			   pinterp,
			    argc, argv)
     ClientData clientData;
     Tcl_Interp *pinterp;
     int argc;
     char *argv[];
{
  char szCmd[CMD_SIZE];
  ClientDataLispObject *pcdlo = (ClientDataLispObject *)clientData;
  int cb=0;
  char *q = szCmd;
  char *p = pcdlo->arglist;

  STORE_3BYTES(q,(pcdlo->iSlot));
  q += 3;
  if (p == 0)
    { char *arg = (argc > 1 ? argv[1] : "");
      int m = strlen(arg);
      if (m > CMD_SIZE -50)
	SIGNAL_ERROR("too big command");
      bcopy(arg,q,m);
      q += m ;}
  else
    { int i,n;
      *q++ = '(';
      n = strlen(p);
      for (i=1; i< argc; i++)
	{ if (i < n && p[i]=='s')   { *q++ = '"';}
	  strcpy(q,argv[i]);
	  q+= strlen(argv[i]);
	  if (i < n && p[i]=='s')   { *q++ = '"';}
	}
      *q++ = ')';
    }
  *q = 0;
     
  dfprintf(stderr, "TclGenericCommandProcedure : %s\n"
	  , szCmd
	  );

  if (sock_write_str2(dsfd,m_call, "",0, szCmd, q-szCmd) == -1)
    {
      dfprintf(stderr,
      "Error\t(TclGenericCommandProcedure) !!!\n\tFailed to write [%s] to socket %d (%d) cb=%d\n"
	      , szCmd, dsfd->fd, errno, cb);

    }
#ifndef __MINGW32__  
  if (parent > 0)kill(parent, SIGUSR1);
#endif  
  return TCL_OK;
}



static void
guiCreateCommand( idLispObject,  iSlot , arglist)
     int idLispObject; int iSlot ; char *arglist;
{
  char szNameCmdProc[2000],*c;
  ClientDataLispObject *pcdlo;

  sprintf(szNameCmdProc, "callback_%d",iSlot);

  pcdlo = (ClientDataLispObject *)malloc(sizeof(ClientDataLispObject));
  pcdlo->id = idLispObject;
  pcdlo->iSlot = iSlot;
  if (arglist[0] == 0)
    { pcdlo->arglist = 0;}
  else
  {c= malloc(strlen(arglist)+1);
   strcpy(c,arglist);
   pcdlo->arglist = c;}
  Tcl_CreateCommand(interp
		    , szNameCmdProc, TclGenericCommandProcedure
		    , (ClientData *)pcdlo, free);
  dfprintf(stderr, "TCL creating callback : %s\n", szNameCmdProc);

/*  guiBindCallback(szNameCmdProc, szTclObject, szModifier,arglist); */
}

/*
int
guiBindCallback(char *szNameCmdProc, char *szTclObject, char *szModifier,char* arglist)
{
  int code;
  char szCmd[2000];

  sprintf(szCmd, "bind %s %s {%s %s}"
	  , szTclObject
	  , szModifier
	  , szNameCmdProc
	  , (arglist ? arglist : "")
	  );
  dfprintf(stderr, "TCL BIND : %s\n", szCmd);

  code = Tcl_Eval(interp, szCmd);
  if (code != TCL_OK)
    {
      dfprintf(stderr, "TCL Error int bind : %s\n", INTERP_RESULT(interp));

    }
  return code;
}
*/
/* static void */
/* guiDeleteCallback(szCallback) */
/*      char *szCallback; */
/* { */
/*   dfprintf(stderr, "Tcl Deleting command : %s\n", szCallback); */

/*   Tcl_DeleteCommand(interp, szCallback); */
/* } */

/*  */

