/*  Copyright (C) 2024 Camm Maguire */

#include <string.h>
#include <stdio.h>

static char *
match(char *c) {

  char *d;

  if (!(c=strstr(c,"DEF")))
    return NULL;

  for (d=c;*d && (*d=='_' || (*d>='A'&& *d<='Z'));d++);

  return *d=='(' ? c : match(d);

}

int
main() {

  char buf[4096],*c,*d=(void *)-1,*e;

  for (;fgets(buf,sizeof(buf),stdin);) {

    if (!strchr(buf,'\n')) {
      fprintf(stderr,"Line too long, %s\n",buf);
      return -1;
    }

    for (c=buf;(c=!d&&*c!='\n' ? c : match(c));c=e) {

      d=strstr(c,"\")");
      e=d ? d+2 : buf+strlen(buf)-1;
      printf("%-.*s\n",(int)(e-c),c);

    }

  }

}
