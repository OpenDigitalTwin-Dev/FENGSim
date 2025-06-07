#define ElfW(a) Elf32_ ## a
#if !defined(HAVE_LIBBFD) && !defined(USE_DLOPEN)
#define __ELF_NATIVE_CLASS 32
#include <sys/elf_SPARC.h>
#endif
#include "linux.h"

#define ADDITIONAL_FEATURES \
                ADD_FEATURE("SUN"); \
                           ADD_FEATURE("SPARC")

#define        SPARC
#define SGC

#define PTR_ALIGN 8

#undef LISTEN_FOR_INPUT
#undef SIG_UNBLOCK_SIGNALS
#define NO_SYSTEM_TIME_ZONE

void bcopy (const void *,void *,size_t);
void bzero(void *,size_t);
int bcmp(const void *,const void *,size_t);
