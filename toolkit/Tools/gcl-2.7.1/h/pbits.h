#define mjoin(a_,b_) a_ ## b_
#define Mjoin(a_,b_) mjoin(a_,b_)

#include "arth.h"

#define LM(a_) AM(AT(SIZEOF_LONG,8),a_)
#define HM(a_) AM(AT(AD(SIZEOF_LONG,2),8),a_)
#define QM(a_) AM(AT(AD(SIZEOF_LONG,4),8),a_)
#if SIZEOF_LONG == 4
#define LL 2
#elif SIZEOF_LONG == 8
#define LL 3
#else
#error "unknown SIZEOF_LONG"
#endif 
