/* Copyright (C) 2024 Camm Maguire */
#ifndef FIRSTWORD
#include "include.h"
#endif

static void
memory_save(char *original_file, char *save_file)
{
#ifdef DO_BEFORE_SAVE
  DO_BEFORE_SAVE ;
#endif    
  
  unexec(save_file,original_file,0,0,0);
}

#ifdef USE_CLEANUP
extern void _cleanup();
#endif

LFD(siLsave)(void) {

  extern char *kcl_self;

  check_arg(1);

  gcl_cleanup(1);

  coerce_to_filename(vs_base[0], FN1);

#ifdef MEMORY_SAVE
  MEMORY_SAVE(kcl_self,FN1);
#else	  
  memory_save(kcl_self, FN1);
#endif	

  /*  no return  */
  exit(0);

}
