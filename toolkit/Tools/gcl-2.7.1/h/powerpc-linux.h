#include "linux.h"

#define SGC

#define CLEAR_CACHE_LINE_SIZE 32
#define CLEAR_CACHE do {void *v=memory->cfd.cfd_start,*ve=v+memory->cfd.cfd_size; \
                        v=(void *)((unsigned long)v & ~(CLEAR_CACHE_LINE_SIZE - 1));\
                        for (;v<ve;v+=CLEAR_CACHE_LINE_SIZE) \
                           asm __volatile__ ("dcbst 0,%0\n\tsync\n\ticbi 0,%0\n\tsync\n\tisync": : "r" (v) : "memory");\
                        } while(0)

#if SIZEOF_LONG == 4
#define RELOC_H "elf32_ppc_reloc.h"
#else
#ifdef WORDS_BIGENDIAN
#define RELOC_H "elf64_ppc_reloc.h"
#define SPECIAL_RELOC_H "elf64_ppc_reloc_special.h"
#define STATIC_FUNCTION_POINTERS
#else
#define RELOC_H "elf64_ppcle_reloc.h"
#define SPECIAL_RELOC_H "elf64_ppcle_reloc_special.h"
#endif
#define C_GC_OFFSET 4
#endif
