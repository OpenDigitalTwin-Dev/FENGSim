// file:    ctools.h
// purpose: prototype
// author:  Christian Wieners
// date:    Oct 11, 2002, now: Feb 25, 2006

#ifndef _CTOOLS_H_
#define _CTOOLS_H_

void Rename (const char* fname);

char* NumberName (const char* path, const char* name, const char* ext, char* namebuffer, int i);
char* NumberName (const char* name, char* namebuffer, int i);
char* NumberName (const char* name, char* namebuffer, int i, const char* ext);

char* pNumberName (const char* name, char* namebuffer);
char* pNumberName (const char* name, char* namebuffer, int i);
char* pNumberName (const char* name, char* namebuffer, int i, const char* ext );
char* pNumberName (const char* name, char* namebuffer, const char* ext);

char* pNumberOldName (const char* name, char* namebuffer);


#endif
