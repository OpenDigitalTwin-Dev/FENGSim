#include "Compiler.h"
#include "Parallel.h"

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

void Rename (const char* fname) {
    time_t T;
    struct stat fstat;
    char new_fname[128];
    strcpy(new_fname,fname);
    strcat(new_fname,".");
    if (stat(fname, &fstat)<0) exit(1);
    T = fstat.st_mtime;
    strftime(new_fname+strlen(fname)+1,64,"%y-%m-%d-%H:%M:%S",localtime(&T));
    if (rename(fname,new_fname)) exit(1);
}

char* NumberName (const char* path, const char* name, const char* ext,
                         char* namebuffer, int i) {
    if (i < 0) sprintf(namebuffer,"%s/%s.%s",path,name,ext);
    else sprintf(namebuffer,"%s/%s.%04d.%s",path,name,i,ext);
    return namebuffer;
}

char* NumberName (const char* name, char* namebuffer, int i) {
    if (i < 0) sprintf(namebuffer,"%s",name);
    else sprintf(namebuffer,"%s.%04d",name,i);
    return namebuffer;
}

char* NumberName (const char* name, char* namebuffer, 
			 int i, const char* ext) {
    if (i < 0) sprintf(namebuffer,"%s.%s",name,ext);
    else sprintf(namebuffer,"%s.%04d.%s",name,i,ext);
    return namebuffer;
}

char* pNumberName (const char* name, char* namebuffer) {
    sprintf(namebuffer,"%s.p%04d",name,PPM->proc());
    return namebuffer;
}

char* pNumberName (const char* name, char* namebuffer, int i) {
    sprintf(namebuffer,"%s.p%04d.%04d",name,PPM->proc(),i);
    return namebuffer;
}

char* pNumberName (const char* name, char* namebuffer, 
			  int i, const char* ext ) {
    sprintf(namebuffer,"%s.p%04d.%04d.%s",name,PPM->proc(),i,ext);
    return namebuffer;
}

char* pNumberName (const char* name, char* namebuffer, const char* ext){
    sprintf(namebuffer,"%s.p%04d.%s",name,PPM->proc(),ext);
    return namebuffer;
}

char* pNumberOldName (const char* name, char* namebuffer) {
    sprintf(namebuffer,"%s-old.p%04d",name,PPM->proc());
    return namebuffer;
}
