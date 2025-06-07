#include <stdarg.h>
#define _VA_LIST_DEFINED
#ifndef EXTER
#define EXTER extern
#endif
#ifndef INLINE
#ifdef OLD_INLINE
#define INLINE extern inline
#else
#define INLINE inline
#endif
#endif
