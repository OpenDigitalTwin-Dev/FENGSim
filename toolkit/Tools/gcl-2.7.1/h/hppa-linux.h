#include "linux.h"

#define SGC
#define STATIC_FUNCTION_POINTERS

#ifdef IN_SFASL
#include <sys/mman.h>
#define CLEAR_CACHE_LINE_SIZE 32
#define CLEAR_CACHE {\
   void *v1=memory->cfd.cfd_start,*v,*ve=v1+memory->cfd.cfd_size;	\
   v1=(void *)((unsigned long)v1 & ~(CLEAR_CACHE_LINE_SIZE - 1));\
   for (v=v1;v<ve;v+=CLEAR_CACHE_LINE_SIZE) asm __volatile__ ("fdc 0(%0)" : : "r" (v) : "memory");\
   asm __volatile__ ("syncdma\n\tsync" : : "r" (v) : "memory");\
   for (v=v1;v<ve;v+=CLEAR_CACHE_LINE_SIZE) asm __volatile__ ("fic 0(%%sr4,%0)" : : "r" (v) : "memory");\
   asm __volatile__ ("syncdma\n\tsync" : : "r" (v) : "memory");}
#endif

#define RELOC_H "elf32_hppa_reloc.h"
#define SPECIAL_RELOC_H "elf32_hppa_reloc_special.h"
