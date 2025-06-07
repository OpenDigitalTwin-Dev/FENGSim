/* original regexp.c file written by Henry Spencer.
   many changes made [see below] made by W. Schelter.
   These changes Copyright (c) 1994 W. Schelter
   These changes Copyright (c) 2024 Camm Maguire

This file is part of GNU Common Lisp, herein referred to as GCL

GCL is free software; you can redistribute it and/or modify it under
the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GCL is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
License for more details.

You should have received a copy of the GNU Library General Public License 
along with GCL; see the file COPYING.  If not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
   

   Various enhancements made by William Schelter when converting
   for use by GCL:
     1) allow case_fold_search:  If this variable is not nil,
     then 'a' and 'A' are considered equivalent.
     2) Various speed ups, useful when searching a long string
     [eg body of a file etc.]
     Timings searching a 47k byte file for patterns

     The following table shows how many times longer it took the original
     implementation, to search for a given pattern.   Comparison is also
     made with the re-search-forward function of gnu emacs.   For example
     in searching for the pattern 'not_there' the search took 20 times
     longer in the original implementation, and about the same time in gnu
     emacs.
					 
     Pattern:                            current original  gnu emacs 
     not_there                              1       20      1
     not_there|really_not                   1       200     30
     not_there|really_not|how is[a-z]       1       115     15
     not_there|really_not|how is[a-z]y      1       30      4
     [a-u]bcdex                             1      194      60
     a.bcde                                 1       10      7.5  
     
     of a character.
     3).  Allow string length to be specified, and string not null
     terminated.  If length specified as zero, string assumed null
     terminated.   If string NOT null terminated, then string area
     must be writable (commonly strings in non writable area are
     already null terminated..).

     To do:  1)Still lots of improvement possible:  eg the pattern
     x[^x]*nice_pattern, should be searched for by doing search
     for nice_pattern, and then backing up.   To do easily requires
     backward search. eg:  "FRONT TAIL" search for TAIL and then
     search back for "FRONT $"
     2) do backward search.
     
     
*/
#include <stdio.h>
#include "regexp.h"

static int
min_initial_branch_length(regexp *, unsigned char *, int);


/*
 * The "internal use only" fields in regexp.h are present to pass info from
 * compile to execute that permits the execute phase to run lots faster on
 * simple cases.  They are:
 *
 * regstart	char that must begin a match; '\0' if none obvious
 * reganch	is the match anchored (at beginning-of-line only)?
 * regmust	string (pointer into program) that match must include, or NULL
 * regmlen	length of regmust string
 *
 * Regstart and reganch permit very fast decisions on suitable starting points
 * for a match, cutting down the work a lot.  Regmust permits fast rejection
 * of lines that cannot possibly match.  The regmust tests are costly enough
 * that regcomp() supplies a regmust only if the r.e. contains something
 * potentially expensive (at present, the only such thing detected is * or +
 * at the start of the r.e., which can involve a lot of backup).  Regmlen is
 * supplied because the test in regexec() needs it and regcomp() is
 * computing it anyway.
 */

/*
 * Structure for regexp "program".  This is essentially a linear encoding
 * of a nondeterministic finite-state machine (aka syntax charts or
 * "railroad normal form" in parsing technology).  Each node is an opcode
 * plus a "next" pointer, possibly plus an operand.  "Next" pointers of
 * all nodes except BRANCH implement concatenation; a "next" pointer with
 * a BRANCH on both ends of it is connecting two alternatives.  (Here we
 * have one of the subtle syntax dependencies:  an individual BRANCH (as
 * opposed to a collection of them) is never concatenated with anything
 * because of operator precedence.)  The operand of some types of node is
 * a literal string; for others, it is a node leading into a sub-FSM.  In
 * particular, the operand of a BRANCH node is the first node of the branch.
 * (NB this is *not* a tree structure:  the tail of the branch connects
 * to the thing following the set of BRANCHes.)  The opcodes are:
 */

/* definition	number	opnd?	meaning */
#define	END	0	/* no	End of program. */
#define	BOL	1	/* no	Match "" at beginning of line. */
#define	EOL	2	/* no	Match "" at end of line. */
#define	ANY	3	/* no	Match any one character. */
#define	ANYOF	4	/* str	Match any character in this string. */
#define	ANYBUT	5	/* str	Match any character not in this string. */
#define	BRANCH	6	/* node	Match this alternative, or the next... */
#define	BACK	7	/* no	Match "", "next" ptr points backward. */
#define	EXACTLY	8	/* str	Match this string. */
#define	NOTHING	9	/* no	Match empty string. */
#define	STAR	10	/* node	Match this (simple) thing 0 or more times. */
#define	PLUS	11	/* node	Match this (simple) thing 1 or more times. */
#define	OPEN	20	/* no	Mark this point in input as start of #n. */
			/*	OPEN+1 is number 1, etc. */
#define	CLOSE	(OPEN+NSUBEXP)	/* no	Analogous to OPEN. */

/*
 * Opcode notes:
 *
 * BRANCH	The set of branches constituting a single choice are hooked
 *		together with their "next" pointers, since precedence prevents
 *		anything being concatenated to any individual branch.  The
 *		"next" pointer of the last BRANCH in a choice points to the
 *		thing following the whole choice.  This is also where the
 *		final "next" pointer of each individual branch points; each
 *		branch starts with the operand node of a BRANCH node.
 *
 * BACK		Normal "next" pointers all implicitly point forward; BACK
 *		exists to make loop structures possible.
 *
 * STAR,PLUS	'?', and complex '*' and '+', are implemented as circular
 *		BRANCH structures using BACK.  Simple cases (one character
 *		per match) are implemented with STAR and PLUS for speed
 *		and to minimize recursive plunges.
 *
 * OPEN,CLOSE	...are numbered at compile time.
 */

/*
 * A node is one char of opcode followed by two chars of "next" pointer.
 * "Next" pointers are stored as two 8-bit pieces, high order first.  The
 * value is a positive offset from the opcode of the node containing it.
 * An operand, if any, simply follows the node.  (Note that much of the
 * code generation knows about this implicit relationship.)
 *
 * Using two bytes for the "next" pointer is vast overkill for most things,
 * but allows patterns to get big without disasters.
 */
#define	OP(p)	(*(p))
#define	NEXT(p)	(((*((p)+1)&0377)<<8) + (*((p)+2)&0377))
#define	OPERAND(p)	((p) + 3)

/*
 * See regmagic.h for one further detail of program structure.
 */


/*
 * Utility definitions.
 */
#ifndef CHARBITS
#define	UCHARAT(p)	((int)*(unsigned char *)(p))
#else
#define	UCHARAT(p)	((int)*(p)&CHARBITS)
#endif

#define	FAIL(m)	{ regerror(m); return(NULL); }
#define	ISMULT(c)	((c) == '*' || (c) == '+' || (c) == '?')
#undef META
#define	META	"^$.[()|?+*\\"

/*
 * Flags to be passed up and down.
 */
#define	HASWIDTH	01	/* Known never to match null string. */
#define	SIMPLE		02	/* Simple enough to be STAR/PLUS operand. */
#define	SPSTART		04	/* Starts with * or +. */
#define	WORST		0	/* Worst case. */

/*
 * Global work variables for regcomp().
 */
static char *regparse;		/* Input-scan pointer. */
static int regnpar;		/* () count. */
static char regdummy;
static char *regcode;		/* Code-emit pointer; &regdummy = don't. */
static long regsize;		/* Code size. */

/*
 * The first byte of the regexp internal "program" is actually this magic
 * number; the start node begins in the second byte.
 */
#define	MAGIC	0234

/*
 * Forward declarations for regcomp()'s friends.
 */
#ifndef STATIC
#define	STATIC	static
#endif
STATIC char *reg(int paren, int *flagp);
STATIC char *regbranch(int *flagp);
STATIC char *regpiece(int *flagp);
STATIC char *regatom(int *flagp);
STATIC char *regnode(char op);
STATIC char *regnext(register char *p);
STATIC void regc(char b);
STATIC void reginsert(char op, char *opnd);
STATIC void regtail(char *p, char *val);
STATIC void regoptail(char *p, char *val);

int case_fold_search = 0;
/*
 - regcomp - compile a regular expression into internal code
 *
 * We can't allocate space until we know how big the compiled form will be,
 * but we can't compile it (and thus know how big it is) until we've got a
 * place to put the code.  So we cheat:  we compile it twice, once with code
 * generation turned off and size counting turned on, and once "for real".
 * This also means that we don't allocate space until we are sure that the
 * thing really will compile successfully, and we never have to move the
 * code and thus invalidate pointers into it.  (Note that it has to be in
 * one piece because free() must be able to free it all.)
 *
 * Beware that the optimization-preparation code in here knows about some
 * of the structure of the compiled regexp.
 */
static regexp *
regcomp(char *exp,ufixnum *sz)
{
	register regexp *r;
	register char *scan;
	register char *longest;
	register int len;
	int flags;

	if (exp == NULL)
		FAIL("NULL argument");

	/* First pass: determine size, legality. */
	regparse = exp;
	regnpar = 1;
	regsize = 0L;
	regcode = &regdummy;
	regc(MAGIC);
	if (reg(0, &flags) == NULL)
		return(NULL);

	/* Small enough for pointer-storage convention? */
	if (regsize >= 32767L)		/* Probably could be 65535L. */
		FAIL("regexp too big");

	/* Allocate space. */
	*sz=sizeof(regexp) + (unsigned)regsize;
	r = (regexp *)alloc_relblock(*sz);
	if (r == NULL)
		FAIL("out of space");

	/* Second pass: emit code. */
	regparse = exp;
	regnpar = 1;
	regcode = r->program;
	regc(MAGIC);
	if (reg(0, &flags) == NULL)
		return(NULL);

	/* Dig out information for optimizations. */
	r->regstart = '\0';	/* Worst-case defaults. */
	r->reganch = 0;
	r->regmust = NULL;
	r->regmlen = 0;
	r->regmaybe_boyer =0;
	scan = r->program+1;			/* First BRANCH. */
	if (0&& OP(regnext(scan)) == END) {		/* Only one top-level choice. */
		scan = OPERAND(scan);

		/* Starting-point info. */
		if (OP(scan) == EXACTLY)
			{r->regstart = *OPERAND(scan);
			 r->regmaybe_boyer = strlen(OPERAND(scan));}
		else if (OP(scan) == BOL)
			r->reganch++;


		/*
		 * If there's something expensive in the r.e., find the
		 * longest literal string that must appear and make it the
		 * regmust.  Resolve ties in favor of later strings, since
		 * the regstart check works with the beginning of the r.e.
		 * and avoiding duplication strengthens checking.  Not a
		 * strong reason, but sufficient in the absence of others.
		 */
		if (flags&SPSTART) {
			longest = NULL;
			len = 0;
			for (; scan != NULL; scan = regnext(scan))
				if (OP(scan) == EXACTLY && ((int) strlen(OPERAND(scan))) >= len) {
					longest = OPERAND(scan);
					len = strlen(OPERAND(scan));
				}
			r->regmust = longest;
			r->regmlen = len;
		}
	}
	else { r->regmaybe_boyer = min_initial_branch_length(r,0,0);}


	return(r);
}

/*
 - reg - regular expression, i.e. main body or parenthesized thing
 *
 * Caller must absorb opening parenthesis.
 *
 * Combining parenthesis handling with the base level of regular expression
 * is a trifle forced, but the need to tie the tails of the branches to what
 * follows makes it hard to avoid.
 */
static char *
reg(int paren, int *flagp)
          			/* Parenthesized? */
           
{
	register char *ret;
	register char *br;
	register char *ender;
	register int parno = 0;
	int flags;

	*flagp = HASWIDTH;	/* Tentatively. */

	/* Make an OPEN node, if parenthesized. */
	if (paren) {
		if (regnpar >= NSUBEXP)
			FAIL("too many ()");
		parno = regnpar;
		regnpar++;
		ret = regnode(OPEN+parno);
	} else
		ret = NULL;

	/* Pick up the branches, linking them together. */
	br = regbranch(&flags);
	if (br == NULL)
		return(NULL);
	if (ret != NULL)
		regtail(ret, br);	/* OPEN -> first. */
	else
		ret = br;
	if (!(flags&HASWIDTH))
		*flagp &= ~HASWIDTH;
	*flagp |= flags&SPSTART;
	while (*regparse == '|') {
		regparse++;
		br = regbranch(&flags);
		if (br == NULL)
			return(NULL);
		regtail(ret, br);	/* BRANCH -> BRANCH. */
		if (!(flags&HASWIDTH))
			*flagp &= ~HASWIDTH;
		*flagp |= flags&SPSTART;
	}

	/* Make a closing node, and hook it on the end. */
	ender = regnode((paren) ? CLOSE+parno : END);	
	regtail(ret, ender);

	/* Hook the tails of the branches to the closing node. */
	for (br = ret; br != NULL; br = regnext(br))
		regoptail(br, ender);

	/* Check for proper termination. */
	if (paren && *regparse++ != ')') {
		FAIL("unmatched ()");
	} else if (!paren && *regparse != '\0') {
		if (*regparse == ')') {
			FAIL("unmatched ()");
		} else
			FAIL("junk on end");	/* "Can't happen". */
		/* NOTREACHED */
	}

	return(ret);
}

/*
 - regbranch - one alternative of an | operator
 *
 * Implements the concatenation operator.
 */
static char *
regbranch(int *flagp)
{
	register char *ret;
	register char *chain;
	register char *latest;
	int flags;

	*flagp = WORST;		/* Tentatively. */

	ret = regnode(BRANCH);
	chain = NULL;
	while (*regparse != '\0' && *regparse != '|' && *regparse != ')') {
		latest = regpiece(&flags);
		if (latest == NULL)
			return(NULL);
		*flagp |= flags&HASWIDTH;
		if (chain == NULL)	/* First piece. */
			*flagp |= flags&SPSTART;
		else
			regtail(chain, latest);
		chain = latest;
	}
	if (chain == NULL)	/* Loop ran zero times. */
		(void) regnode(NOTHING);

	return(ret);
}

/*
 - regpiece - something followed by possible [*+?]
 *
 * Note that the branching code sequences used for ? and the general cases
 * of * and + are somewhat optimized:  they use the same NOTHING node as
 * both the endmarker for their branch list and the body of the last branch.
 * It might seem that this node could be dispensed with entirely, but the
 * endmarker role is not redundant.
 */
static char *
regpiece(int *flagp)
{
	register char *ret;
	register char op;
	register char *next;
	int flags;

	ret = regatom(&flags);
	if (ret == NULL)
		return(NULL);

	op = *regparse;
	if (!ISMULT(op)) {
		*flagp = flags;
		return(ret);
	}

	if (!(flags&HASWIDTH) && op != '?')
		FAIL("*+ operand could be empty");
	*flagp = (op != '+') ? (WORST|SPSTART) : (WORST|HASWIDTH);

	if (op == '*' && (flags&SIMPLE))
		reginsert(STAR, ret);
	else if (op == '*') {
		/* Emit x* as (x&|), where & means "self". */
		reginsert(BRANCH, ret);			/* Either x */
		regoptail(ret, regnode(BACK));		/* and loop */
		regoptail(ret, ret);			/* back */
		regtail(ret, regnode(BRANCH));		/* or */
		regtail(ret, regnode(NOTHING));		/* null. */
	} else if (op == '+' && (flags&SIMPLE))
		reginsert(PLUS, ret);
	else if (op == '+') {
		/* Emit x+ as x(&|), where & means "self". */
		next = regnode(BRANCH);			/* Either */
		regtail(ret, next);
		regtail(regnode(BACK), ret);		/* loop back */
		regtail(next, regnode(BRANCH));		/* or */
		regtail(ret, regnode(NOTHING));		/* null. */
	} else if (op == '?') {
		/* Emit x? as (x|) */
		reginsert(BRANCH, ret);			/* Either x */
		regtail(ret, regnode(BRANCH));		/* or */
		next = regnode(NOTHING);		/* null. */
		regtail(ret, next);
		regoptail(ret, next);
	}
	regparse++;
	if (ISMULT(*regparse))
		FAIL("nested *?+");

	return(ret);
}

/*
 - regatom - the lowest level
 *
 * Optimization:  gobbles an entire sequence of ordinary characters so that
 * it can turn them into a single node, which is smaller to store and
 * faster to run.  Backslashed characters are exceptions, each becoming a
 * separate node; the code is simpler that way and it's not worth fixing.
 */
static char *
regatom(int *flagp)
{
	register char *ret;
	int flags;

	*flagp = WORST;		/* Tentatively. */

	switch (*regparse++) {
	case '^':
		ret = regnode(BOL);
		break;
	case '$':
		ret = regnode(EOL);
		break;
	case '.':
		ret = regnode(ANY);
		*flagp |= HASWIDTH|SIMPLE;
		break;
	case '[': {char buf[1000];
		   char result[256];
		   char *regcp=buf;
		   int matches = 1;
#define REGC(x) (*regcp++ = (x))
	              {     
			register int clss;
			register int classend;
			ret = regnode(ANYOF);

			if (*regparse == '^') {	/* Complement of range. */
				matches = 0;
				regparse++;}
			if (*regparse == ']' || *regparse == '-')
				REGC(*regparse++);
			while (*regparse != '\0' && *regparse != ']') {
				if (*regparse == '-') {
					regparse++;
					if (*regparse == ']' || *regparse == '\0')
						REGC('-');
					else {
						clss = UCHARAT(regparse-2)+1;
						classend = UCHARAT(regparse);
						if (clss > classend+1)
							FAIL("invalid [] range");
						for (; clss <= classend; clss++)
							REGC(clss);
						regparse++;
					}
				} else
					REGC(*regparse++);
			}
			REGC('\0');
			if (*regparse != ']')
				FAIL("unmatched []");
			regparse++;
			*flagp |= HASWIDTH|SIMPLE;
		}
		 if (regcp - buf > sizeof(buf))
		   { emsg("wow that is badly defined regexp..");
		     do_gcl_abort();}
		regcp --;
		{ char *p=buf;

		  /* set default vals */
		  p = result;
		  while (p < &result[sizeof(result)])
		     *p++ = (!matches );
		  
                  p = buf;
		  while (p < regcp)
		    { result[*(unsigned char *)p] = matches;
		      if (case_fold_search)
			{result[tolower(*p)] = matches;
			 result[toupper(*p)] = matches; p++;}
		      else
		      result[*(unsigned char *)p++] = matches;
		      
		    }
		  p = result;
		  while (p < &result[sizeof(result)])
		    { regc(*p++);}}
		break;
		 }
	case '(':
		ret = reg(1, &flags);
		if (ret == NULL)
			return(NULL);
		*flagp |= flags&(HASWIDTH|SPSTART);
		break;
	case '\0':
	case '|':
	case ')':
		FAIL("internal urp");	/* Supposed to be caught earlier. */
		/* NOTREACHED */
		break;
	case '?':
	case '+':
	case '*':
		FAIL("?+* follows nothing");
		/* NOTREACHED */
		break;
	case '\\':
		if (*regparse == '\0')
			FAIL("trailing \\");
		ret = regnode(EXACTLY);
		regc(*regparse++);
		regc('\0');
		*flagp |= HASWIDTH|SIMPLE;
		break;
	default: {
			register int len;
			register char ender;

			regparse--;
			len = strcspn(regparse, META);
			if (len <= 0)
				FAIL("internal disaster");
			ender = *(regparse+len);
			if (len > 1 && ISMULT(ender))
				len--;		/* Back off clear of ?+* operand. */
			*flagp |= HASWIDTH;
			if (len == 1)
				*flagp |= SIMPLE;
			ret = regnode(EXACTLY);
			while (len > 0) {
				regc(*regparse++);
				len--;
			}
			regc('\0');
		}
		break;
	}

	return(ret);
}

/*
 - regnode - emit a node
 */
static char *			/* Location. */
regnode(char op)
{
	register char *ret;
	register char *ptr;

	ret = regcode;
	if (ret == &regdummy) {
		regsize += 3;
		return(ret);
	}

	ptr = ret;
	*ptr++ = op;
	*ptr++ = '\0';		/* Null "next" pointer. */
	*ptr++ = '\0';
	regcode = ptr;

	return(ret);
}

/*
 - regc - emit (if appropriate) a byte of code
 */
static void
regc(char b)
{
	if (regcode != &regdummy)
		*regcode++ = b;
	else
		regsize++;
}

/*
 - reginsert - insert an operator in front of already-emitted operand
 *
 * Means relocating the operand.
 */
static void
reginsert(char op, char *opnd)
{
	register char *src;
	register char *dst;
	register char *place;

	if (regcode == &regdummy) {
		regsize += 3;
		return;
	}

	src = regcode;
	regcode += 3;
	dst = regcode;
	while (src > opnd)
		*--dst = *--src;

	place = opnd;		/* Op node, where operand used to be. */
	*place++ = op;
	*place++ = '\0';
	*place++ = '\0';
}

/*
 - regtail - set the next-pointer at the end of a node chain
 */
static void
regtail(char *p, char *val)
{
	register char *scan;
	register char *temp;
	register int offset;

	if (p == &regdummy)
		return;

	/* Find last node. */
	scan = p;
	for (;;) {
		temp = regnext(scan);
		if (temp == NULL)
			break;
		scan = temp;
	}

	if (OP(scan) == BACK)
		offset = scan - val;
	else
		offset = val - scan;
	*(scan+1) = (offset>>8)&0377;
	*(scan+2) = offset&0377;
}

/*
 - regoptail - regtail on operand of first argument; nop if operandless
 */
static void
regoptail(char *p, char *val)
{
	/* "Operandless" and "op != BRANCH" are synonymous in practice. */
	if (p == NULL || p == &regdummy || OP(p) != BRANCH)
		return;
	regtail(OPERAND(p), val);
}

/*
 * regexec and friends
 */

/*
 * Global work variables for regexec().
 */
static char *reginput;		/* String-input pointer. */
static char *regbol;		/* Beginning of input, for ^ check. */
static char **regstartp;	/* Pointer to startp array. */
static char **regendp;		/* Ditto for endp. */

/*
 * Forwards.
 */
STATIC int regtry(regexp *prog, char *string);
STATIC int regmatch(char *prog);
STATIC int regrepeat(char *p);

#ifdef DEBUG
int regnarrate = 0;
void regdump();
STATIC char *regprop();
#endif

/*
 - regexec - match a regexp against a string
 PROG is the compiled regexp and STRING is the string one is searching in
 and START is a pointer relative to STRING, to tell if a substring of the
 original STRING is being passed.  LENGTH can be 0 or the strlen(STRING).
 If it is not 0 and is large, then a fast checking will be enabled. 

 */
static int
regexec(register regexp *prog, register char *string, char *start, int length)
{
	register char *s;
	char saved,*savedp=NULL;
	int value;

	/* Be paranoid... */
	if (prog == NULL || string == NULL) {
		regerror("NULL parameter");
		return(0);
	}

	/* Check validity of program. */
	if (UCHARAT(prog->program) != MAGIC) {
		regerror("corrupted program");
		return(0);
	}

	/* If there is a "must appear" string, look for it. */
	/* to do:fix this for case_fold_search, and also to detect
	   x[^x]*MUST pattern, searching for MUST, and then
	   backing up to the 'x'.  The regmust thing is bad in
	   case of a long string. */
	if (0 && prog->regmust != NULL) {
		s = string;
		while ((s = strchr(s, prog->regmust[0])) != NULL) {
			if (strncmp(s, prog->regmust, prog->regmlen) == 0)
				break;	/* Found it. */
			s++;
		}
		if (s == NULL)	/* Not present. */
			return(0);
	}

	/* null terminate string */
	if (length)
	  { savedp = &string[length];
	    saved = *savedp;
	    if (saved) *savedp=0;
	  }
	else saved=0;
#define RETURN_VAL(i) do {value=i; goto DO_RETURN;}while(0)

	/* Mark beginning of line for ^ . */
	regbol = start;

	/* Simplest case:  anchored match need be tried only once. */
	if (prog->reganch)
		RETURN_VAL(regtry(prog, string));

	/* Messy cases:  unanchored match. */
	s = string;
	/* only do if long enough to warrant compile time
	   really  length/prog->regmaybe_boyer > 1000 is
	   probably better (and >=2 !)
	 */
	if (length > 2 && prog->regmaybe_boyer>= 1)
	  { unsigned char buf[256];
	  /*  int advance= reg_compboyer(prog,buf); */
	    int advance=prog->regmaybe_boyer;


	    
	    int amt;
	    unsigned char *s = (unsigned char *)string+ advance -1;
	    min_initial_branch_length(prog, buf,advance);
	    switch(advance) {
	    case 1:
	      while (1)
	      { if (buf[*s]==0)
		  { if (*s == 0) RETURN_VAL(0);
		    else
		      if (regtry(prog,(char *)s-(1-1))) RETURN_VAL(1);}
		s++; }
	    RETURN_VAL(0);
	      
	    case 2:
	    while (length > 0)
	      { 
		amt = (buf[s[0]]);
		if (amt == 0)
		  {
		    amt = buf[s[-1]]-1;
		    if (amt <=0) {
		      if (regtry(prog,(char *)s-(advance-1))) 
			RETURN_VAL(1);
		      else 
			amt =1;
		    }
		  }
		s += amt; length -= amt;
	      }
	    RETURN_VAL(0);
	  case 3:
	    while (length > 0)
	      { amt = (buf[s[0]]);
		if (amt == 0)
		  {amt = buf[s[-1]]-1;
		   if (amt <=0)
		     {amt = buf[s[-2]]-2;
		      if (amt <=0)
		        {if (regtry(prog,(char *)s-(advance-1))) RETURN_VAL(1);
			else amt =1;}}}
		s += amt; length -= amt;}
	  case 4:
	    while (length > 0)
	      { amt = (buf[s[0]]);
		if (amt == 0)
		  {amt = buf[s[-1]]-1;
		   if (amt <=0)
		     {amt = buf[s[-2]]-2;
		      if (amt <=0)
			{amt = buf[s[-3]]-3;
			 if (amt <=0)
			   {if (regtry(prog,(char *)s-(advance-1))) RETURN_VAL(1);
			   else amt =1;}}}}
		s += amt; length -= amt;}

	  default:
	    while (length > 0)
	      { amt = (buf[s[0]]);
		if (amt == 0)
		  {amt = buf[s[-1]]-1;
		   if (amt <=0)
		     {amt = buf[s[-2]]-2;
		      if (amt <=0)
			{amt = buf[s[-3]]-3;
			 if (amt <=0)
			   {amt = buf[s[-4]]-4;
			    if (amt <=0)
			   {if (regtry(prog,(char *)s-(advance-1))) RETURN_VAL(1);
			   else amt =1;}}}}}
		s += amt; length -= amt;}
	  }
	    RETURN_VAL(0);
	  }
	else
	if (prog->regstart != '\0')
		/* We know what char it must start with. */
	  { if (case_fold_search)
	      {char ch = tolower(prog->regstart);
	       while (*s)
		 { if (tolower(*s)==ch)
		     {if (regtry(prog, s))
			RETURN_VAL(1);}
		   s++;}}
	    else
	      while ((s = strchr(s, prog->regstart)) != NULL) {
		if (regtry(prog, s))
		  RETURN_VAL(1);
		s++;
	      }
	  }
	else
		/* We don't -- general case. */
		do {
			if (regtry(prog, s))
				RETURN_VAL(1);
		} while (*s++ != '\0');

	/* Failure. */
	RETURN_VAL(0);
      DO_RETURN:
	if(saved) *savedp=saved;
	return value;
	
}

#ifdef OLD_VERSION
reg_compboyer(r,buf)
     regexp *r;
     char *buf;
{
  char *scan;
  scan = r->program+1;			/* First BRANCH. */
  if (OP(regnext(scan)) == END) {/* Only one top-level choice. */
    scan = OPERAND(scan);
    /* Starting-point info. */
#define MIN(n,m) (n > m ? m : n)
    if (OP(scan) == EXACTLY)
      { char *op = OPERAND(scan);
	char *p = buf;
	int advance = strlen(op);
	int i = 256;
	if (advance > 255) advance = 255;
	if (advance < 1) regerror("Impossible");
	while (--i >= 0) *p++ = advance;
	i = advance;
	p = op;
	while (--i >= 0)
	  { if (case_fold_search)
	    { buf[tolower(*p)] = i;
	      buf[toupper(*p)] = i;
	    }
	    else  buf[(*p)] = i;
	    p++;
	      
	  }
	buf[0]=0;
	return advance;
      }}
    regerror("Should be impossible");
    return 1;
}
#endif

/*
 - regtry - try match at specific point
 */
static int			/* 0 failure, 1 success */
regtry(regexp *prog, char *string)
{
	register int i;
	register char **sp;
	register char **ep;

	reginput = string;
	regstartp = prog->startp;
	regendp = prog->endp;

	sp = prog->startp;
	ep = prog->endp;
	for (i = NSUBEXP; i > 0; i--) {
		*sp++ = NULL;
		*ep++ = NULL;
	}
	if (regmatch(prog->program + 1)) {
		prog->startp[0] = string;
		prog->endp[0] = reginput;
		return(1);
	} else
		return(0);
}

/*
 - regmatch - main matching routine
 *
 * Conceptually the strategy is simple:  check to see whether the current
 * node matches, call self recursively to see whether the rest matches,
 * and then act accordingly.  In practice we make some effort to avoid
 * recursion, in particular by going through "ordinary" nodes (that don't
 * need to know whether the rest of the match failed) by a loop instead of
 * by recursion.
 */
static int			/* 0 failure, 1 success */
regmatch(char *prog)
{
	register char *scan;	/* Current node. */
	char *next;		/* Next node. */

	scan = prog;
#ifdef DEBUG
	if (scan != NULL && regnarrate)
		emsg("%s(\n", regprop(scan));
#endif
	while (scan != NULL) {
#ifdef DEBUG
		if (regnarrate)
			emsg("%s...\n", regprop(scan));
#endif
		next = regnext(scan);

		switch (OP(scan)) {
		case BOL:
			if (reginput != regbol)
				return(0);
			break;
		case EOL:
			if (*reginput != '\0')
				return(0);
			break;
		case ANY:
			if (*reginput == '\0')
				return(0);
			reginput++;
			break;
		case EXACTLY: {
				register char *opnd;
				char * ch = reginput;

				opnd = OPERAND(scan);
				if (case_fold_search)
				while (*opnd )
				  { if (tolower(*opnd) != tolower(*ch))
				       return 0;
				    else { ch++; opnd++;}}
				else
				while (*opnd )
				  { if (*opnd != *ch)
				       return 0;
				    else { ch++; opnd++;}}
				/* a match */
				reginput += (opnd - OPERAND(scan));
			}
			break;
		case ANYOF:
 			if (*reginput == '\0' ||
			    OPERAND(scan)[*(unsigned char *)reginput] == 0)
				return(0);
			reginput++;
			break;
		case ANYBUT:
 			if (*reginput == '\0' ||
    			    OPERAND(scan)[*(unsigned char *)reginput] != 0)
				return(0);
			reginput++;
			break;
		case NOTHING:
			break;
		case BACK:
			break;
		case OPEN+1 ... OPEN+NSUBEXP-1:
		  {
				register int no;
				register char *save;

				no = OP(scan) - OPEN;
				save = reginput;

				if (regmatch(next)) {
					/*
					 * Don't set startp if some later
					 * invocation of the same parentheses
					 * already has.
					 */
					if (regstartp[no] == NULL)
						regstartp[no] = save;
					return(1);
				} else
					return(0);
			}
			/* NOTREACHED */
			break;
		case CLOSE+1 ... CLOSE+NSUBEXP-1:
		  {
				register int no;
				register char *save;

				no = OP(scan) - CLOSE;
				save = reginput;

				if (regmatch(next)) {
					/*
					 * Don't set endp if some later
					 * invocation of the same parentheses
					 * already has.
					 */
					if (regendp[no] == NULL)
						regendp[no] = save;
					return(1);
				} else
					return(0);
			}
			/* NOTREACHED */
			break;
		case BRANCH: {
				register char *save;

				if (OP(next) != BRANCH)		/* No choice. */
					next = OPERAND(scan);	/* Avoid recursion. */
				else {
					do {
						save = reginput;
						if (regmatch(OPERAND(scan)))
							return(1);
						reginput = save;
						scan = regnext(scan);
					} while (scan != NULL && OP(scan) == BRANCH);
					return(0);
					/* NOTREACHED */
				}
			}
			/* NOTREACHED */
			break;
		case STAR:
		case PLUS: {
				register char nextch;
				register int no;
				register char *save;
				register int min;

				/*
				 * Lookahead to avoid useless match attempts
				 * when we know what character comes next.
				 */
				nextch = '\0';
				if (OP(next) == EXACTLY)
					nextch = *OPERAND(next);
				if (case_fold_search)
				  nextch = tolower(nextch);
				min = (OP(scan) == STAR) ? 0 : 1;
				save = reginput;
				no = regrepeat(OPERAND(scan));
				while (no >= min) {
					/* If it could work, try it. */
					if (nextch == '\0' ||
					    *reginput == nextch
					    || (case_fold_search &&
					      tolower(*reginput) == nextch))
						if (regmatch(next))
							return(1);
					/* Couldn't or didn't -- back up. */
					no--;
					reginput = save + no;
				}
				return(0);
			}
			/* NOTREACHED */
			break;
		case END:
			return(1);	/* Success! */
			/* NOTREACHED */
			break;
		default:
			regerror("memory corruption");
			return(0);
			/* NOTREACHED */
			break;
		}

		scan = next;
	}

	/*
	 * We get here only if there's trouble -- normally "case END" is
	 * the terminating point.
	 */
	regerror("corrupted pointers");
	return(0);
}

/*
 - regrepeat - repeatedly match something simple, report how many
 */
static int
regrepeat(char *p)
{
	register int count = 0;
	register char *scan;
	register char *opnd;

	scan = reginput;
	opnd = OPERAND(p);
	switch (OP(p)) {
	case ANY:
		count = strlen(scan);
		scan += count;
		break;
	case EXACTLY:
		{ char ch = *opnd;
		if (case_fold_search)
		  { ch = tolower(*opnd);
		    while (ch == tolower(*scan))
		      {
			count++;
			scan++;}}
		else
		  while (ch  == *scan) {
		    count++;
		    scan++;
		}}
		break;
	case ANYOF:
		while (*scan != '\0' &&
		       opnd[*(unsigned char *)scan] != 0)
		  {
			count++;
			scan++;
		}
		break;
	case ANYBUT:
		while (*scan != '\0' &&
		       opnd[*(unsigned char *)scan] == 0)
	  {
			count++;
			scan++;
		}
		break;
	default:		/* Oh dear.  Called inappropriately. */
		regerror("internal foulup");
		count = 0;	/* Best compromise. */
		break;
	}
	reginput = scan;

	return(count);
}

/*
 - regnext - dig the "next" pointer out of a node
 */
static char *
regnext(register char *p)
{
	register int offset;

	if (p == &regdummy)
		return(NULL);

	offset = NEXT(p);
	if (offset == 0)
		return(NULL);

	if (OP(p) == BACK)
		return(p-offset);
	else
		return(p+offset);
}

#ifdef DEBUG

STATIC char *regprop();

/*
 - regdump - dump a regexp onto stdout in vaguely comprehensible form
 */
void
regdump(r)
regexp *r;
{
	register char *s;
	register char op = EXACTLY;	/* Arbitrary non-END op. */
	register char *next;


	s = r->program + 1;
	while (op != END) {	/* While that wasn't END last time... */
		op = OP(s);
		printf("%2d%s", s-r->program, regprop(s));	/* Where, what. */
		next = regnext(s);
		if (next == NULL)		/* Next ptr. */
			printf("(0)");
		else 
			printf("(%d)", (s-r->program)+(next-s));
		s += 3;
		if (op == ANYOF || op == ANYBUT)
		  { int i=-1;
		    while (i++ < 256)
		      if (s[i]) printf("%c",i);
		    s +=256;
		  }
		

		else
		if (op == EXACTLY) {
			/* Literal string, where present. */
			while (*s != '\0') {
				putchar(*s);
				s++;
			}
			s++;
		}
		putchar('\n');
	}

	/* Header fields of interest. */
	if (r->regstart != '\0')
		printf("start `%c' ", r->regstart);
	if (r->reganch)
		printf("anchored ");
	if (r->regmust != NULL)
		printf("must have \"%s\"", r->regmust);
	printf("\n");
}

/*
 - regprop - printable representation of opcode
 */
static char *
regprop(op)
char *op;
{
	register char *p;
	static char buf[50];

	(void) strcpy(buf, ":");

	switch (OP(op)) {
	case BOL:
		p = "BOL";
		break;
	case EOL:
		p = "EOL";
		break;
	case ANY:
		p = "ANY";
		break;
	case ANYOF:
		p = "ANYOF";
		break;
	case ANYBUT:
		p = "ANYBUT";
		break;
	case BRANCH:
		p = "BRANCH";
		break;
	case EXACTLY:
		p = "EXACTLY";
		break;
	case NOTHING:
		p = "NOTHING";
		break;
	case BACK:
		p = "BACK";
		break;
	case END:
		p = "END";
		break;
	case OPEN+1 ... OPEN+NSUBEXP-1:
		sprintf(buf+strlen(buf), "OPEN%d", OP(op)-OPEN);
		p = NULL;
		break;
	case CLOSE+1 ... CLOSE+NSUBEXP-1:
		sprintf(buf+strlen(buf), "CLOSE%d", OP(op)-CLOSE);
		p = NULL;
		break;
	case STAR:
		p = "STAR";
		break;
	case PLUS:
		p = "PLUS";
		break;
	default:
		regerror("corrupted opcode");
		break;
	}
	if (p != NULL)
		(void) strcat(buf, p);
	return(buf);
}
#endif

/*
 * The following is provided for those people who do not have strcspn() in
 * their C libraries.  They should get off their butts and do something
 * about it; at least one public-domain implementation of those (highly
 * useful) string routines has been published on Usenet.
 */
/*
 * strcspn - find length of initial segment of s1 consisting entirely
 * of characters not from s2
 */

#ifdef NEVER_WE_PUT_IT_IN_LIB
size_t
strcspn(s1, s2)
char *s1;
char *s2;
{
	register char *scan1;
	register char *scan2;
	register int count;

	count = 0;
	for (scan1 = s1; *scan1 != '\0'; scan1++) {
		for (scan2 = s2; *scan2 != '\0';)	/* ++ moved down. */
			if (*scan1 == *scan2++)
				return(count);
		count++;
	}
	return(count);
}
#endif
/* if min_initial_branch_length(prog,0,0) > 2
   it is possible to have an initial matching routine
   This means that each toplevel branch has an initial segment of
   characters which is at least 2 and which  
   */

#define MINIMIZE(loc,val) if (val < loc) loc=val
static int
min_initial_branch_length(regexp *x, unsigned char *buf, int advance)
{ char *s = x->program+1;
  int overall = 10000;
  int i= -1;
  char *next ;
  char op = EXACTLY;
  int n = advance;
  if (buf)
    { buf[0]=0;
      for (i=256; --i>0 ; ){buf[i]=n;};
   }
  while(op != END)
    { op = OP(s);
      next = (s) + NEXT(s);
      if (op != END && op != BRANCH)
	do_gcl_abort();
      s = s+3;
      { int this = 0;
	int anythis =0;
	int ok = 1;
	char op ;
	int i;
	while (1)
	  { if (ok == 0) goto LEND;
	  AGAIN:
	    if(buf && n <= 0) {break;}
	    op = OP(s);
	    advance = n;
	    s = OPERAND(s);
	    if (op == EXACTLY)
	      { int m = strlen(s);
		if (buf)
		  { char *ss = s;
		    n--;
		    while(1)
		      { if (case_fold_search)
			  {MINIMIZE(buf[tolower(*ss)],n);
			   MINIMIZE(buf[toupper(*ss)],n);
			  }
			else
			  { MINIMIZE(buf[*(unsigned char *)ss],n);}
			
			ss++;
			if (*ss==0 || n ==0) break;
			--n;}}
		else {
		this += m + anythis;
		anythis = 0;}
		
		s += m+1;}
	    else if (op == ANYOF)
	      { if (buf)
		  { --n;
		    for(i=256; --i>0;)
		      {if (s[i]) MINIMIZE(buf[i],n);}}
		else
		  {
		    anythis += 1;
		    /* if this seems like a random choice of letters they
		       are and they are not */
		    if (s['f']==0 || s['a']==0 ||s['y']==0 || s['v']==0)
		      { this += anythis;
			anythis = 0;
		      }}

		s += 256;}
	    else if (op == ANY)
	      {if (buf)
		  { --n;
		    for(i=256; --i>0;)
		      { MINIMIZE(buf[i],n);}}
	       else
		anythis += 1;
	       }
	    else if (op == PLUS)
	      {	
		ok = 0; goto AGAIN;
	       }
	    else
	      {
	      LEND:
#ifdef DEBUG		
		if (buf==0)printf("[Br=%d]",this);
#endif		
		 if (overall > this) { overall = this;}
		 break;}
	  }}
      s = next;
      op = OP(s);
      n = advance;

    }
#ifdef DEBUG		
  if (buf==0)  printf("[overall=%d]\n",overall);
#endif		
  return overall;
}

#ifndef regerror
void
regerror(char *s)
{
    emsg("regexp error %s\n", s);
}
#endif
  
