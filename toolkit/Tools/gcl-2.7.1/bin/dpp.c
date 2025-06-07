/*
	dpp.c

	defun preprocessor
*/

/*
	Usage:
		dpp file

	The file named file.d is preprocessed and the output will be
	written to the file whose name is file.c.

	
	;changes: remove \n from beginning of main output so debuggers
	can find the right foo.d source file name.--wfs
	;add \" to the line output for ansi C --wfs

	The function definition:

	@(defun name ({var}*
		      [&optional {var | (var [initform [svar]])}*]
		      [&rest]
		      [&key {var |
			     ({var | (keyword var)} [initform [svar]])}*
			    [&allow_other_keys]]
		      [&aux {var | (var [initform])}*])

		C-declaration

	@

		C-body

	@)

	&optional may be abbreviated as &o.
	&rest may be abbreviated as &r.
	&key may be abbreviated as &k.
	&allow_other_keys may be abbreviated as &aok.
	&aux may be abbreviated as &a.

	Each variable becomes a macro name
	defined to be an expression of the form
		vs_base[...].

	Each supplied-p parameter becomes a boolean C variable.

	Initforms are C expressions.
	It an expression contain non-alphanumeric characters,
	it should be surrounded by backquotes (`).


	Function return:

		@(return {form}*)

	It becomes a C block.

*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "gclincl.h"
#include "config.h"

#ifdef UNIX
#include <ctype.h>
#define	isalphanum(c)	isalnum(c)
#endif

#define	POOLSIZE	2048
#define	MAXREQ		16
#define	MAXOPT		16
#define	MAXKEY		16
#define	MAXAUX		16
#define	MAXRES		16

#define	TRUE		1
#define	FALSE		0

typedef int bool;

FILE *in, *out;

char filename[BUFSIZ];
int line;
int tab;
int tab_save;

char pool[POOLSIZE];
char *poolp;

char *function;
int fstatic;

char *required[MAXREQ];
int nreq;

struct optional {
	char *o_var;
	char *o_init;
	char *o_svar;
} optional[MAXOPT];
int nopt;

bool rest_flag;

bool key_flag;
struct keyword {
	char *k_key;
	char *k_var;
	char *k_init;
	char *k_svar;
} keyword[MAXKEY];
int nkey;
bool allow_other_keys_flag;

struct aux {
	char *a_var;
	char *a_init;
} aux[MAXAUX];
int naux;

char *result[MAXRES];
int nres;

void
error(s)
char *s;
{
	printf("Error in line %d: %s.\n", line, s);
	exit(0);
}

int
readc()
{
	int c;

	c = getc(in);
	if (feof(in)) {
		if (function != NULL)
			error("unexpected end of file");
		exit(0);
	}
	if (c == '\n') {
		line++;
		tab = 0;
	} else if (c == '\t')
		tab++;
	return(c);
}

int
nextc()
{
	int c;

	while (isspace(c = readc()))
		;
	return(c);
}

void
unreadc(c)
int c;
{
	if (c == '\n')
		--line;
	else if (c == '\t')
		--tab;
	ungetc(c, in);
}

void
put_tabs(n)
int n;
{
	int i;

	for (i = 0;  i < n;  i++)
		putc('\t', out);
}

void
pushc(c)
int c;
{
	if (poolp >= &pool[POOLSIZE])
		error("buffer bool overflow");
	*poolp++ = c;
}

char *
read_token()
{
	int c;
	char *p;

	p = poolp;
	if ((c = nextc()) == '`') {
		while ((c = readc()) != '`')
			pushc(c);
		pushc('\0');
		return(p);
	}
	do
		pushc(c);
	while (isalphanum(c = readc()) || c == '_');
	pushc('\0');
	unreadc(c);
	return(p);
}

void
reset()
{
	int i;

	poolp = pool;
	function = NULL;
	nreq = 0;
	for (i = 0;  i < MAXREQ;  i++)
		required[i] = NULL;
	nopt = 0;
	for (i = 0;  i < MAXOPT;  i++)
		optional[i].o_var
		= optional[i].o_init
		= optional[i].o_svar
		= NULL;
	rest_flag = FALSE;
	key_flag = FALSE;
	nkey = 0;
	for (i = 0;  i < MAXKEY;  i++)
		keyword[i].k_key
		= keyword[i].k_var
		= keyword[i].k_init
		= keyword[i].k_svar
		= NULL;
	allow_other_keys_flag = FALSE;
	naux = 0;
	for (i = 0;  i < MAXAUX;  i++)
		aux[i].a_var
		= aux[i].a_init
		= NULL;
}

void
get_function()
{
	function = read_token();
}

void
get_lambda_list()
{
	int c;
	char *p;

	if ((c = nextc()) != '(')
		error("( expected");
	for (;;) {
		if ((c = nextc()) == ')')
			return;
		if (c == '&') {
			p = read_token();
			goto OPTIONAL;
		}
		unreadc(c);
		p = read_token();
		if (nreq >= MAXREQ)
			error("too many required variables");
		required[nreq++] = p;
	}

OPTIONAL:
	if (strcmp(p, "optional") != 0 && strcmp(p, "o") != 0)
		goto REST;
	for (;;  nopt++) {
		if ((c = nextc()) == ')')
			return;
		if (c == '&') {
			p = read_token();
			goto REST;
		}
		if (nopt >= MAXOPT)
			error("too many optional argument");
		if (c == '(') {
			optional[nopt].o_var = read_token();
			if ((c = nextc()) == ')')
				continue;
			unreadc(c);
			optional[nopt].o_init = read_token();
			if ((c = nextc()) == ')')
				continue;
			unreadc(c);
			optional[nopt].o_svar = read_token();
			if (nextc() != ')')
				error(") expected");
		} else {
			unreadc(c);
			optional[nopt].o_var = read_token();
		}
	}

REST:
	if (strcmp(p, "rest") != 0 && strcmp(p, "r") != 0)
		goto KEYWORD;
	rest_flag = TRUE;
	if ((c = nextc()) == ')')
		return;
	if (c != '&')
		error("& expected");
	p = read_token();
	goto KEYWORD;

KEYWORD:
	if (strcmp(p, "key") != 0 && strcmp(p, "k") != 0)
		goto AUX_L;
	key_flag = TRUE;
	for (;;  nkey++) {
		if ((c = nextc()) == ')')
			return;
		if (c == '&') {
			p = read_token();
			if (strcmp(p, "allow_other_keys") == 0 ||
			    strcmp(p, "aok") == 0) {
				allow_other_keys_flag = TRUE;
				if ((c = nextc()) == ')')
					return;
				if (c != '&')
					error("& expected");
				p = read_token();
			}
			goto AUX_L;
		}
		if (nkey >= MAXKEY)
			error("too many optional argument");
		if (c == '(') {
			if ((c = nextc()) == '(') {
				p = read_token();
				if (p[0] != ':' || p[1] == '\0')
					error("keyword expected");
				keyword[nkey].k_key = p + 1;
				keyword[nkey].k_var = read_token();
				if (nextc() != ')')
					error(") expected");
			} else {
				unreadc(c);
				keyword[nkey].k_key
				= keyword[nkey].k_var
				= read_token();
			}
			if ((c = nextc()) == ')')
				continue;
			unreadc(c);
			keyword[nkey].k_init = read_token();
			if ((c = nextc()) == ')')
				continue;
			unreadc(c);
			keyword[nkey].k_svar = read_token();
			if (nextc() != ')')
				error(") expected");
		} else {
			unreadc(c);
			keyword[nkey].k_key
			= keyword[nkey].k_var
			= read_token();
		}
	}

AUX_L:
	if (strcmp(p, "aux") != 0 && strcmp(p, "a") != 0)
		error("illegal lambda-list keyword");
	for (;;) {
		if ((c = nextc()) == ')')
			return;
		if (c == '&')
			error("illegal lambda-list keyword");
		if (naux >= MAXAUX)
			error("too many auxiliary variable");
		if (c == '(') {
			aux[naux].a_var = read_token();
			if ((c = nextc()) == ')')
				continue;
			unreadc(c);
			aux[naux].a_init = read_token();
			if (nextc() != ')')
				error(") expected");
		} else {
			unreadc(c);
			aux[naux].a_var = read_token();
		}
		naux++;
	}
}

void
get_return()
{
	int c;

	nres = 0;
	for (;;) {
		if ((c = nextc()) == ')')
			return;
		unreadc(c);
		result[nres++] = read_token();
	}
}

void
put_fhead()
{
#ifdef STATIC_FUNCTION_POINTERS
	fprintf(out, "static void L%s_static ();\n",function);
	if (!fstatic)
	  fprintf(out,"void\nL%s()\n{ L%s_static();}\n\n",function,function);
	fprintf(out,"static void\nL%s_static()\n{",function);
#else
	fprintf(out, "%svoid\nL%s()\n{", fstatic ? "static " : "",function);
#endif
}

void
put_declaration()
{
	int i;

	if (nopt || rest_flag || key_flag)
	  fprintf(out, "\tint narg;\n");
	fprintf(out, "\tregister object *DPPbase=vs_base;\n");
	
	for (i = 0;  i < nopt;  i++)
		if (optional[i].o_svar != NULL)
			fprintf(out, "\tbool %s;\n",
				optional[i].o_svar);
	for (i = 0;  i < nreq;  i++)
		fprintf(out, "#define\t%s\tDPPbase[%d]\n",
			required[i], i);
	for (i = 0;  i < nopt;  i++)
		fprintf(out, "#define\t%s\tDPPbase[%d+%d]\n",
			optional[i].o_var, nreq, i);
	for (i = 0;  i < nkey;  i++)
		fprintf(out, "#define\t%s\tDPPbase[%d+%d+%d]\n",
			keyword[i].k_var, nreq, nopt, i);
	for (i = 0;  i < nkey;  i++)
		if (keyword[i].k_svar != NULL)
			fprintf(out, "\tbool %s;\n", keyword[i].k_svar);
	for (i = 0;  i < naux;  i++)
		fprintf(out, "#define\t%s\tDPPbase[%d+%d+2*%d+%d]\n",
			aux[i].a_var, nreq, nopt, nkey, i);
	fprintf(out, "\n");
	if (nopt == 0 && !rest_flag && !key_flag)
		fprintf(out, "\tcheck_arg(%d);\n", nreq);
	else {
	        fprintf(out, "\tnarg = vs_top - vs_base;\n");
		fprintf(out, "\tif (narg < %d)\n", nreq);
		fprintf(out, "\t\ttoo_few_arguments();\n");
	}
	for (i = 0;  i < nopt;  i++)
		if (optional[i].o_svar != NULL) {
			fprintf(out, "\tif (narg > %d + %d)\n",
				nreq, i);
			fprintf(out, "\t\t%s = TRUE;\n",
				optional[i].o_svar);
			fprintf(out, "\telse {\n");
			fprintf(out, "\t\t%s = FALSE;\n",
				optional[i].o_svar);
			fprintf(out, "\t\tvs_push(%s);\n",
				optional[i].o_init);
			fprintf(out, "\t\tnarg++;\n");
			fprintf(out, "\t}\n");
		} else if (optional[i].o_init != NULL) {
			fprintf(out, "\tif (narg <= %d + %d) {\n",
				nreq, i);
			fprintf(out, "\t\tvs_push(%s);\n",
				optional[i].o_init);
			fprintf(out, "\t\tnarg++;\n");
			fprintf(out, "\t}\n");
		} else {
			fprintf(out, "\tif (narg <= %d + %d) {\n",
				nreq, i);
			fprintf(out, "\t\tvs_push(Cnil);\n");
			fprintf(out, "\t\tnarg++;\n");
			fprintf(out, "\t}\n");
		}
	if (nopt > 0 && !key_flag && !rest_flag) {
		fprintf(out, "\tif (narg > %d + %d)\n", nreq, nopt);
		fprintf(out, "\t\ttoo_many_arguments();\n");
	}
	if (key_flag) {
		fprintf(out, "\tparse_key(vs_base+%d+%d,FALSE, %s, %d,\n",
			nreq, nopt,
			allow_other_keys_flag ? "TRUE" : "FALSE", nkey);
		if (nkey > 0) {
			i = 0;
			for (;;) {
				fprintf(out, "\t\tsK%s", keyword[i].k_key);
				if (++i == nkey) {
					fprintf(out, ");\n");
					break;
				} else
					fprintf(out, ",\n");
			}
		} else
			fprintf(out, "\t\tCnil);");
		fprintf(out, "\tvs_top = vs_base + %d+%d+2*%d;\n",
			nreq, nopt, nkey);
		for (i = 0;  i < nkey;  i++) {
			if (keyword[i].k_init == NULL)
				continue;
			fprintf(out, "\tif (vs_base[%d+%d+%d+%d]==Cnil)\n",
				nreq, nopt, nkey, i);
			fprintf(out, "\t\t%s = %s;\n",
				keyword[i].k_var, keyword[i].k_init);
		}
		for (i = 0;  i < nkey;  i++)
			if (keyword[i].k_svar != NULL)
				fprintf(out,
				"\t%s = vs_base[%d+%d+%d+%d] != Cnil;\n",
				keyword[i].k_svar, nreq, nopt, nkey, i);
	}
	for (i = 0;  i < naux;  i++)
                if (aux[i].a_init != NULL)
			fprintf(out, "\tvs_push(%s);\n", aux[i].a_init);
		else
			fprintf(out, "\tvs_push(Cnil);\n");
}

void
put_ftail()
{
	int i;

	for (i = 0;  i < nreq;  i++)
		fprintf(out, "#undef %s\n", required[i]);
	for (i = 0;  i < nopt;  i++)
		fprintf(out, "#undef %s\n", optional[i].o_var);
	for (i = 0;  i < nkey;  i++)
		fprintf(out, "#undef %s\n", keyword[i].k_var);
	for (i = 0;  i < naux;  i++)
		fprintf(out, "#undef %s\n", aux[i].a_var);
	fprintf(out, "}");
}

void
put_return()
{
	int i, t;

	t = tab_save + 1;
	if (nres == 0) {
		fprintf(out, "{\n");
		put_tabs(t);
		fprintf(out, "vs_top = vs_base;\n");
		put_tabs(t);
		fprintf(out, "vs_base[0] = Cnil;\n");
		put_tabs(t);
		fprintf(out, "return;\n");
		put_tabs(tab_save);
		fprintf(out, "}");
	} else if (nres == 1) {
		fprintf(out, "{\n");
		put_tabs(t);
		fprintf(out, "vs_base[0] = %s;\n", result[0]);
		put_tabs(t);
		fprintf(out, "vs_top = vs_base + 1;\n");
		put_tabs(t);
		fprintf(out, "return;\n");
		put_tabs(tab_save);
		fprintf(out, "}");
	} else {
		fprintf(out, "{\n");
		for (i = 0;  i < nres;  i++) {
			put_tabs(t);
			fprintf(out, "object R%d;\n", i);
		}
		for (i = 0;  i < nres;  i++) {
			put_tabs(t);
			fprintf(out, "R%d = %s;\n", i, result[i]);
		}
		for (i = 0;  i < nres;  i++) {
			put_tabs(t);
			fprintf(out, "vs_base[%d] = R%d;\n", i, i);
		}
		put_tabs(t);
		fprintf(out, "vs_top = vs_base + %d;\n", nres);
		put_tabs(t);
		fprintf(out, "return;\n");
		put_tabs(tab_save);
		fprintf(out, "}");
	}
}

void
main_loop()
{
	int c;
	char *p;

	line = 1;
	fprintf(out, "# line %d \"%s\"\n", line, filename);
LOOP:
	reset();
	fprintf(out, "\n# line %d \"%s\"\n", line, filename);
	while ((c = readc()) != '@')
		putc(c, out);
	if (readc() != '(')
		error("@( expected");
	p = read_token();
	fstatic=0;
	if (strcmp(p, "static") == 0) {
	  fstatic=1;
	  p = read_token();
	}
	if (strcmp(p, "defun") == 0) {
		get_function();
		get_lambda_list();
		put_fhead();
		fprintf(out, "\n# line %d \"%s\"\n", line, filename);
		while ((c = readc()) != '@')
			putc(c, out);
		put_declaration();

	BODY:
		fprintf(out, "\n# line %d \"%s\"\n", line, filename);
		while ((c = readc()) != '@')
			putc(c, out);
		if ((c = readc()) == ')') {
			put_ftail();
			goto LOOP;
		} else if (c != '(')
			error("@( expected");
		p = read_token();
		if (strcmp(p, "return") == 0) {
			tab_save = tab;
			get_return();
			put_return();
			goto BODY;
		} else
			error("illegal symbol");
	} else
		error("illegal symbol");
}

int
main(int argc, char *argv[]) {

  if (argc != 3)
    error("arg count");
  if (sscanf(argv[1],"%s.d",filename)!=1)
    error("bad filename\n");
  if (!(in = fopen(argv[1], "r")))
    error("can't open input file");
  out = fopen(argv[2], "w");
  if (!(out = fopen(argv[2], "w")))
    error("can't open output file");
  printf("dpp: %s -> ", argv[1]);
  printf("%s\n", argv[2]);
  main_loop();
  return 0;
}
