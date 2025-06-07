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
	unixsave.c
*/

#define IN_UNIXSAVE
#ifndef FIRSTWORD
#include "include.h"
#endif

#ifdef UNIXSAVE
#include UNIXSAVE
#else

#ifdef HAVE_FCNTL
#include <fcntl.h>
#else
#include <sys/file.h>
#endif

#ifdef HAVE_AOUT
#undef BSD
#undef ATT
#define BSD
#endif



#ifdef BSD
#include HAVE_AOUT
#endif

#ifdef DOS
void 
binary_file_mode()
{_fmode = O_BINARY;}
#endif


#ifdef ATT
#include <filehdr.h>
#include <aouthdr.h>
#include <scnhdr.h>
#endif

#ifdef E15
#include <a.out.h>
extern	char etext;
#endif


filecpy(to, from, n)
FILE *to, *from;
register int n;
{
	char buffer[BUFSIZ];

	for (;;)
		if (n > BUFSIZ) {
			fread(buffer, BUFSIZ, 1, from);
			fwrite(buffer, BUFSIZ, 1, to);
			n -= BUFSIZ;
		} else if (n > 0) {
			fread(buffer, 1, n, from);
			fwrite(buffer, 1, n, to);
			break;
		} else
			break;
}

static void
memory_save(original_file, save_file)
char *original_file, *save_file;
{	MEM_SAVE_LOCALS;
	char *data_begin, *data_end;
	int original_data;
	FILE *original, *save;
	register int n;
	register char *p;
	extern char *sbrk();

	original = freopen(original_file,"r",stdin);
/*	fclose(stdin); 
	original = fopen(original_file, "r");
*/	

	if (stdin != original || original->_file != 0) {
		emsg("Can't open the original file.\n");
		do_gcl_abort();
	}
	setbuf(original, stdin_buf);
	fclose(stdout);
	unlink(save_file);
	n = open(save_file, O_CREAT|O_WRONLY, 0777);
	if (n != 1 || (save = fdopen(n, "w")) != stdout) {
		emsg("Can't open the save file.\n");
		do_gcl_abort();
	}
	setbuf(save, stdout_buf);

	READ_HEADER;
	FILECPY_HEADER;

	for (n = header.a_data, p = data_begin;  ;  n -= BUFSIZ, p += BUFSIZ)
		if (n > BUFSIZ)
			fwrite(p, BUFSIZ, 1, save);
		else if (n > 0) {
			fwrite(p, 1, n, save);
			break;
		} else
			break;

	fseek(original, original_data, 1);

	COPY_TO_SAVE;

	fclose(original);
	fclose(save);
}

extern void _cleanup();

LFD(Lsave)() {
  
  check_arg(1);
  check_type_or_pathname_string_symbol_stream(&vs_base[0]);
  coerce_to_filename(vs_base[0], FN1);
  
  _cleanup();
  
  memory_save(kcl_self, FN1);
  exit(0);
  /*  no return  */
}

#endif /* UNIXSAVE include */

void
gcl_init_unixsave(void)
{
	make_si_function("SAVE", siLsave);
}

