#ifndef _REGEXP
#define _REGEXP 1

#define NSUBEXP  19
typedef struct regexp {
	char *startp[NSUBEXP];
	char *endp[NSUBEXP];
	char regstart;		/* Internal use only. */
	char reganch;		/* Internal use only. */
	char *regmust;		/* Internal use only. */
	int regmlen;		/* Internal use only. */
	unsigned char regmaybe_boyer;
	char program[1];	/* Unwarranted chumminess with compiler. */
} regexp;

#if __STDC__ == 1
#define _ANSI_ARGS_(x) x
#else
#define _ANSI_ARGS_(x) ()
#endif

/* extern regexp *regcomp _ANSI_ARGS_((char *exp)); */
/* extern int regexec _ANSI_ARGS_((regexp *prog, char *string, char *start,int length )); */
extern void regsub _ANSI_ARGS_((regexp *prog, char *source, char *dest));
#ifndef regerror
extern void regerror _ANSI_ARGS_((char *msg));
#endif

#endif /* REGEXP */
