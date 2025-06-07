#include "linux.h"

#ifdef IN_GBC
/* GET_FAULT_ADDR is a bit complicated to implement on m68k, because the fault
   address can't be found directly in the sigcontext. One has to look at the
   CPU frame, and that one is different for each CPU.
   */

/* #define GET_FAULT_ADDR(sig,code,sv,a)		\ */
/*     ({\ */
/* 	struct sigcontext *scp1 = (struct sigcontext *)(sv); \ */
/* 	int format = (scp1->sc_formatvec >> 12) & 0xf; \ */
/* 	unsigned long *framedata = (unsigned long *)(scp1 + 1); \ */
/* 	unsigned long ea; \ */
/* 	if (format == 0xa || format == 0xb) \ */
/* 			/\* 68020/030 *\/	\ */
/*           ea = framedata[2]; \ */
/* 	else if (format == 7) \ */
/* 			/\* 68040 *\/ \ */
/*           ea = framedata[3]; \ */
/* 	else if (format == 4) {	\ */
/* 			/\* 68060 *\/ \ */
/*           ea = framedata[0]; \ */
/*           if (framedata[1] & 0x08000000) \ */
/* 			/\* correct addr on misaligned access *\/ \ */
/*             ea = (ea+4095)&(~4095); \ */
/* 	} \ */
/*         else {\ */
/*            FEerror("Unknown m68k cpu",0);\ */
/*            ea=0;\ */
/*         } \ */
/* 	(char *)ea; }) */
#endif

#define ADDITIONAL_FEATURES \
		     ADD_FEATURE("BSD386"); \
         	     ADD_FEATURE("MC68020")



#define	M68K
/* #define SGC *//*FIXME:  Unknown m68k cpu in modern emulators*/

#include <asm/cachectl.h>
int cacheflush(void *,int,int,int);
#define CLEAR_CACHE_LINE_SIZE 32
#define CLEAR_CACHE do {void *v=memory->cfd.cfd_start,*ve=v+memory->cfd.cfd_size; \
                        v=(void *)((unsigned long)v & ~(CLEAR_CACHE_LINE_SIZE - 1));\
                        cacheflush(v,FLUSH_SCOPE_PAGE,FLUSH_CACHE_BOTH,ve-v);\
                    } while(0)

#define C_GC_OFFSET 2

#define RELOC_H "elf32_m68k_reloc.h"

#define NEED_STACK_CHK_GUARD

/* #define DEFINED_REAL_MAXPAGE (1UL<<18) /\*FIXME brk probe broken*\/ */
