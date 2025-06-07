#include "linux.h"

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("BSD386"); \
      	             ADD_FEATURE("MC68020")


#define	I386
#define SGC

#ifndef SA_NOCLDWAIT
#define SA_NOCLDWAIT 0 /*fixme handler does waitpid(-1, ..., WNOHANG)*/
#endif
#define PATH_MAX 4096 /*fixme dynamic*/
#define MAXPATHLEN 4096 /*fixme dynamic*/
/* #define MAX_BRK 0x70000000 */ /*GNU Hurd fragmentation bug*/

#define RELOC_H "elf32_i386_reloc.h"

#define NEED_STACK_CHK_GUARD

#undef HAVE_D_TYPE /*FIXME defined, but not implemented in readdir*/
/* #define NO_FILE_LOCKING */ /*FIXME*/


