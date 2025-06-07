#include <stdlib.h>
void bzero(void *b, size_t length)
{ char *c=b;

 while(length-->0)
   *c++ = 0;
}
