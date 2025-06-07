#include "linux.h"

/* #define SGC *//*FIXME ia64 specific fread/getc restart failure and hang*/

#define STATIC_FUNCTION_POINTERS
#define BRK_DOES_NOT_GUARANTEE_ALLOCATION

#define NOFREE_ERR
