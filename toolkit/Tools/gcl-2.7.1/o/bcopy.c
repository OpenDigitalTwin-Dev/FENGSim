#include <stdlib.h>
void bcopy(const void *s1, void *s2, size_t n)
{ const char *c1=s1;
  char *c2=s2;
  while (n-- > 0) {
    *c2++ = *c1++;
}
}


