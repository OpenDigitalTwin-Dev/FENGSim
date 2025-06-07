#include <stdlib.h>
int bcmp(const void *s1, const void *s2, size_t n)
{  const char *c1=s1,*c2=s2;
   while (n-- > 0)
	{if (*c1++ != *c2++)
	 return 1;}
      return 0;
    }


  
