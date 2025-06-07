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

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <spawn.h>
#ifndef __MINGW32__
#include <sys/wait.h>
#endif

#include "include.h"

#if !defined(__MINGW32__) && !defined(__CYGWIN__)


int
vsystem(char *command) {

  char *c;
  const char *x1[]={"/bin/sh","-c",NULL,NULL},*spc=" \n\t",**p1,**pp,**pe;
  int s;
  pid_t pid;
  posix_spawnattr_t attr;
  posix_spawn_file_actions_t file_actions;
  extern char **environ;

  if (strpbrk(command,"\"'$<>"))

    (p1=x1)[2]=command;

  else {

    p1=(void *)FN2;
    pe=p1+sizeof(FN2)/sizeof(*p1);
    for (pp=p1,c=command;pp<pe && (*pp=strtok(c,spc));c=NULL,pp++);
    massert(pp<pe);

  }

  massert(!posix_spawn_file_actions_init(&file_actions));
  massert(!posix_spawnattr_init(&attr));

  massert(!posix_spawnp(&pid, *p1, &file_actions, &attr,  (void *)p1, environ));

  massert(!posix_spawnattr_destroy(&attr));
  massert(!posix_spawn_file_actions_destroy(&file_actions));

  massert(pid>0);
  massert(pid==waitpid(pid,&s,0));

  if ((s>>8)&128)
    emsg("execvp failure when executing '%s': %s\n",command,strerror((s>>8)&0x7f));

  return s;

}
#elif defined(__CYGWIN__)

#include <tchar.h>
#include <time.h>
#include <windows.h>
#include <sys/cygwin.h>

int
vsystem(const char *command) {

  STARTUPINFO s={0};
  PROCESS_INFORMATION p={0};
  unsigned int e;
  char *cmd=NULL,*r;

  massert((r=strpbrk(command," \n\t"))-command<sizeof(FN2));
  memcpy(FN2,command,r-command);
  FN2[r-command]=0;
  cygwin_conv_path(CCP_POSIX_TO_WIN_A,FN2,FN3,sizeof(FN3));
  massert(snprintf(FN1,sizeof(FN1),"%s %s",FN3,r)>=0);
  command=FN1;


  s.cb=sizeof(s);
  massert(CreateProcess(cmd,(void *)command,NULL,NULL,FALSE,0,NULL,NULL,&s,&p));
  massert(!WaitForSingleObject(p.hProcess,INFINITE));
  massert(GetExitCodeProcess(p.hProcess,&e));
  massert(CloseHandle(p.hProcess));
  massert(CloseHandle(p.hThread));

  return e;

}

#endif


#ifdef ATT3B2
#include <signal.h>
int
system(command)
char *command;
{
	char buf[4];
	extern sigint();

	signal(SIGINT, SIG_IGN);
	write(4, command, strlen(command)+1);
	read(5, buf, 1);
	signal(SIGINT, sigint);
	return(buf[0]<<8);
}
#endif

#ifdef E15
#include <signal.h>
int
system(command)
char *command;
{
	char buf[4];
	extern sigint();

	signal(SIGINT, SIG_IGN);
	write(4, command, strlen(command)+1);
	read(5, buf, 1);
	signal(SIGINT, sigint);
	return(buf[0]<<8);
}
#endif

int
msystem(char *s) {

  return psystem(s);

}

static void
FFN(siLsystem)(void)
{
	static char command[32768];
	int i;

	check_arg(1);
	check_type_string(&vs_base[0]);
	if (VLEN(vs_base[0]) >= 32768)
		FEerror("Too long command line: ~S.", 1, vs_base[0]);
	for (i = 0;  i < VLEN(vs_base[0]);  i++)
		command[i] = vs_base[0]->st.st_self[i];
	command[i] = '\0';
	{int old = signals_allowed;
	 int res;
	 signals_allowed = sig_at_read;
	 res = msystem(command) ;
	 signals_allowed = old;
	 vs_base[0] = make_fixnum(res >> 8);
	 vs_base[1] = make_fixnum((res & 0xff));
	 vs_top++;
       }
}

DEFUN("GETPID",object,fSgetpid,SI,0,0,NONE,OO,OO,OO,OO,(void),
      "getpid  returns  the  process  ID  of the current process") {
  return make_fixnum(getpid());
}


void
gcl_init_unixsys(void) {

  make_si_function("SYSTEM", siLsystem);

}
