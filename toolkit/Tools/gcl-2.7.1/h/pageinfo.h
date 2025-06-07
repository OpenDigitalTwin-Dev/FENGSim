#include "pbits.h"

struct pageinfo {
  unsigned long type:6;
  unsigned long magic:7;
  unsigned long sgc_flags:2;
  unsigned long in_use:LM(15);
  struct pageinfo *next;
};
  
