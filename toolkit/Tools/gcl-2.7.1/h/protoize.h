/* alloc.c:89:OF */ extern void *alloc_page (long n); /* (n) int n; */
/* alloc.c:149:OF */ void add_page_to_freelist (char *p, struct typemanager *tm); /* (p, tm) char *p; struct typemanager *tm; */
/* alloc.c:196:OF */ extern object type_name (int t); /* (t) int t; */
/* alloc.c:213:OF */ object alloc_object (enum type t); /* (t) enum type t; */
/* alloc.c:213:OF */ void add_pages(struct typemanager *,fixnum);
/* alloc.c:296:OF */ extern object make_cons (object a, object d); /* (a, d) object a; object d; */
/* alloc.c:364:OF */ extern object on_stack_cons (object x, object y); /* (x, y) object x; object y; */
/* alloc.c:480:OF */ extern void insert_contblock (void *p, ufixnum s); /* (p, s) char *p; int s; */
/* alloc.c:480:OF */ extern void insert_maybe_sgc_contblock (char *p, int s); /* (p, s) char *p; int s; */
/* alloc.c:611:OF */ extern void set_maxpage (void); /* () */
/* alloc.c:635:OF */ extern void gcl_init_alloc (void *); /* () */
/* alloc.c:1000:OF */ extern void gcl_init_alloc_function (void); /* () */
/* alloc.c:1126:OF */ extern void free (void *ptr); /* (ptr) void *ptr; */
/* array.c:57:OF */ extern void Laref (void); /* () */
/* array.c:262:OF */ extern void siLaset (void); /* () */
/* array.c:321:OF */ extern void siLsvset (void); /* () */
/* array.c:480:OF */ extern void siLmake_vector (void); /* () */
/* array.c:738:OF */ extern void adjust_displaced (object x); /* (x, diff) object x; int diff; */
/* array.c:790:OF */ extern void gset (void *p1, void *val, fixnum n, int typ); /* (p1, val, n, typ) char *p1; char *val; int n; int typ; */
/* array.c:879:OF */ extern void array_allocself (object x, int staticp, object dflt); /* (x, staticp, dflt) object x; int staticp; object dflt; */
/* array.c:920:OF */ extern void siLfill_pointer_set (void); /* () */
/* array.c:944:OF */ extern void Lfill_pointer (void); /* () */
/* array.c:986:OF */ extern void Larray_element_type (void); /* () */
/* array.c:995:OF */ extern void Ladjustable_array_p (void); /* () */
/* array.c:1002:OF */ extern void siLdisplaced_array_p (void); /* () */
/* array.c:1010:OF */ extern void Larray_rank (void); /* () */
/* array.c:1020:OF */ extern void Larray_dimension (void); /* () */
/* array.c:1090:OF */ extern void siLreplace_array (void); /* () */
/* array.c:1160:OF */ extern void gcl_init_array_function (void); /* () */
/* assignment.c:62:OF */ extern void setq (object sym, object val); /* (sym, val) object sym; object val; */
/* assignment.c:128:OF */ extern void Lset (void); /* () */
/* assignment.c:142:OF */ extern void siLfset (void); /* () */
/* assignment.c:228:OF */ extern void Lfmakunbound (void); /* () */
/* assignment.c:547:OF */ extern object clear_compiler_properties (object sym, object code); /* (sym, code) object sym; object code; */
/* assignment.c:591:OF */ extern void gcl_init_assignment (void); /* () */
/* backq.c:259:OF */ extern int backq_car (object x); /* (x) object x; */
/* backq.c:381:OF */ extern void gcl_init_backq (void); /* () */
/* bds.c:31:OF */ extern void bds_unwind (bds_ptr new_bds_top); /* (new_bds_top) bds_ptr new_bds_top; */
/* gmp_big.c:96:OF */ extern void gcl_init_big1 (void); /* () */
/* gmp_big.c:108:OF */ extern object new_bignum (void); /* () */
/* gmp_big.c:161:OF */ extern object make_integer (__mpz_struct *u); /* (u) __mpz_struct *u; */
/* gmp_big.c:207:OF */ extern int big_compare (object x, object y); /* (x, y) object x; object y; */
/* gmp_big.c:214:OF */ extern object normalize_big_to_object (object x); /* (x) object x; */
/* gmp_big.c:230:OF */ extern void add_int_big (int i, object x); /* (i, x) int i; object x; */
/* gmp_big.c:244:OF */ extern void mul_int_big (int i, object x); /* (i, x) int i; object x; */
/* gmp_big.c:289:OF */ extern object normalize_big (object x); /* (x) object x; */
/* gmp_big.c:302:OF */ extern object big_minus (object x); /* (x) object x; */
/* gmp_big.c:324:OF */ extern double big_to_double (object x); /* (x) object x; */
/* gmp_big.c:454:OF */ extern object maybe_replace_big (object x); /* (x) object x; */
/* gmp_big.c:454:OF */ extern object replace_big (object x); /* (x) object x; */
/* gmp_big.c:472:OF */ extern object bignum2 (unsigned int h, unsigned int l); /* (h, l) unsigned int h; unsigned int l; */
/* gmp_big.c:482:OF */ extern void integer_quotient_remainder_1 (object x, object y, object *qp, object *rp,fixnum z); /* (x, y, qp, rp) object x; object y; object *qp; object *rp; */
/* gmp_big.c:482:OF */ extern void integer_quotient_remainder_1_ui (object x, unsigned long y, object *qp, object *rp,fixnum z); /* (x, y, qp, rp) object x; object y; object *qp; object *rp; */
/* gmp_big.c:502:OF */ extern object coerce_big_to_string (object x, int printbase); /* (x, printbase) object x; int printbase; */
/* gmp_big.c:521:OF */ extern void gcl_init_big (void); /* () */
/* big.c:72:OF */ extern int big_sign (object x); /* (x) object x; */
/* big.c:78:OF */ extern void set_big_sign (object x, int sign); /* (x, sign) object x; int sign; */
/* big.c:85:OF */ extern void zero_big (object x); /* (x) object x; */
/* bind.c:74:OF */ extern void lambda_bind (object *arg_top); /* (arg_top) object *arg_top; */
/* bind.c:564:OF */ extern void bind_var (object var, object val, object spp); /* (var, val, spp) object var; object val; object spp; */
/* bind.c:610:OF */ extern object find_special (object body, struct bind_temp *start, struct bind_temp *end,object *); /* (body, start, end) object body; struct bind_temp *start; struct bind_temp *end; */
/* bind.c:670:OF */ extern object let_bind (object body, struct bind_temp *start, struct bind_temp *end); /* (body, start, end) object body; struct bind_temp *start; struct bind_temp *end; */
/* bind.c:688:OF */ extern object letA_bind (object body, struct bind_temp *start, struct bind_temp *end); /* (body, start, end) object body; struct bind_temp *start; struct bind_temp *end; */
/* bind.c:712:OF */ extern void parse_key (object *base, bool rest, bool allow_other_keys, register int n, ... ); 
/* bind.c:820:OF */ extern void check_other_key (object l, int n, ...); 
struct key {short n,allow_other_keys;
	    iobject *defaults;
	    iobject keys[1];
	   };

/* bind.c:866:OF */ extern int parse_key_new_new (int n, object *base, struct key *keys, object first, va_list ap); /* (n, base, keys, ap) int n; object *base; struct key *keys; va_list ap; */
/* bind.c:916:OF */ extern int parse_key_rest_new (object rest, int n, object *base, struct key *keys, object first, va_list ap); /* (rest, n, base, keys, ap) object rest; int n; object *base; struct key *keys; va_list ap; */
/* bind.c:975:OF */ extern void set_key_struct (struct key *ks, object data); /* (ks, data) struct key *ks; object data; */
/* bind.c:995:OF */ extern void gcl_init_bind (void); /* () */
/* block.c:121:OF */ extern void gcl_init_block (void); /* () */
/* bsearch.c:5:OF */ extern void *bsearch (const void *key, const void *base, size_t nel, size_t keysize, int (*compar) (const void *,const void *)); /* (key, base, nel, keysize, compar) char *key; char *base; unsigned int nel; unsigned int keysize; int (*compar)(); */
#if defined (__MINGW32__)
/* bzero.c:3:OF */ /*  extern void bzero (char *b, size_t length); */ /* (b, length) char *b; int length; */
#endif
/* catch.c:166:OF */ extern void gcl_init_catch (void); /* () */
/* cfun.c:37:OF */ extern object make_cfun (void (*self)(), object name, object data, char *start, int size); /* (self, name, data, start, size) int (*self)(); object name; object data; char *start; int size; */
/* cfun.c:56:OF */ extern object make_sfun (object name, object (*self)(), int argd, object data,fixnum nval); /* (name, self, argd, data) object name; int (*self)(); int argd; object data; */
/* cfun.c:91:OF */ extern object make_cclosure_new (void *self, object name, object env, object data,object cl,fixnum argd,fixnum sizes); /* (self, name, env, data) int (*self)(); object name; object env; object data; */
/* cfun.c:91:OF */ extern object make_cclosure(void *self,object data,object call,object key,ufixnum argd,ufixnum sizes,object *envp,ufixnum nargs,...);
/* cfun.c:91:OF */ extern void  add_to_env(object fun,ufixnum nargs,...);
/* cfun.c:283:OF */ extern object make_function_internal (char *s, void(*f)()); /* (s, f) char *s; int (*f)(); */
/* cfun.c:283:OF */ extern object make_macro_internal (char *s, void(*f)()); /* (s, f) char *s; int (*f)(); */
/* cfun.c:299:OF */ extern object make_si_sfun_internal (char *s, object (*f)(), int argd); /* (s, f, argd) char *s; int (*f)(); int argd; */
/* cfun.c:322:OF */ extern object make_si_function_internal (char *s, void (*f) ()); /* (s, f) char *s; int (*f)(); */
/* cfun.c:341:OF */ extern object make_special_form_internal (char *s, void *f); /* (s, f) char *s; int (*f)(); */
/* cfun.c:341:OF */ extern object make_si_special_form_internal (char *s, void *f); /* (s, f) char *s; int (*f)(); */
/* cfun.c:371:OF */ extern void turbo_closure (object fun); /* (fun) object fun; */
/* cfun.c:403:OF */ extern void gcl_init_cfun (void); /* () */
/* cmac.c:191:OF */ extern void gcl_init_cmac (void); /* () */
/* cmpaux.c:33:OF */ extern void siLspecialp (void); /* () */
/* cmpaux.c:95:OF */ extern void gcl_init_cmpaux (void); /* () */
/* cmpaux.c:106:OF */ /* extern int ifloor (int x, int y); */ /* (x, y) int x; int y; */
/* cmpaux.c:124:OF */ /* extern int imod (int x, int y); */ /* (x, y) int x; int y; */
/* cmpaux.c:185:OF */ extern int object_to_int (object x); /* (x) object x; */
/* cmpaux.c:185:OF */ extern fixnum object_to_fixnum (object x); /* (x) object x; */
/* cmpaux.c:263:OF */ extern char *object_to_string (object x); /* (x) object x; */
typedef int (*FUNC)();
/* cmpaux.c:294:OF */ extern void call_init (int init_address,object memory,object faslfile); /* (init_address, memory, fasl_vec, fptr) int init_address; object memory; object fasl_vec; FUNC fptr; */
/* cmpaux.c:339:OF */ extern void do_init (object *statVV); /* (statVV) object *statVV; */
/* cmpaux.c:416:OF */ extern void gcl_init_or_load1 (void (*fn) (void), const char *file); /* (fn, file) int (*fn)(); char *file; */
/* conditional.c:200:OF */ extern void gcl_init_conditional (void); /* () */
/* error.c:38:OF */ extern void terminal_interrupt (int correctable); /* (correctable) int correctable; */
/* error.c:147:OF */ extern void Lerror (void); /* () */
/* error.c:164:OF */ extern void Lcerror (void); /* () */
/* error.c:561:OF */ extern void check_arg_failed (int n); /* (n) int n; */
/* error.c:568:OF */ extern void too_few_arguments (void); /* () */
/* error.c:573:OF */ extern void too_many_arguments (void); /* () */
/* error.c:586:OF */ extern void ck_larg_exactly (int n, object x); /* (n, x) int n; object x; */
/* error.c:595:OF */ extern void invalid_macro_call (void); /* () */
/* error.c:618:OF */ extern object wrong_type_argument (object typ, object obj); /* (typ, obj) object typ; object obj; */
/* error.c:625:OF */ extern void illegal_declare (object form); /* (form) int form; */
/* error.c:635:OF */ extern void not_a_string_or_symbol (object x); /* (x) object x; */
/* error.c:641:OF */ extern void not_a_symbol (object obj); /* (obj) object obj; */
/* error.c:647:OF */ extern int not_a_variable (object obj); /* (obj) object obj; */
/* error.c:653:OF */ extern void illegal_index (object x, object i); /* (x, i) object x; object i; */
/* error.c:660:OF */ extern void check_socket (object x); /* (x) object x; */
/* error.c:670:OF */ extern void check_stream (object strm); /* (strm) object strm; */
/* error.c:697:OF */ extern void check_arg_range (fixnum nn,int n, int m); /* (n, m) int n; int m; */
/* error.c:727:OF */ extern void gcl_init_error (void); /* () */
/* eval.c:143:OF */ extern void funcall (object fun); /* (fun) object fun; */
/* eval.c:375:OF */ extern void lispcall (object *funp, int narg); /* (funp, narg) object *funp; int narg; */
/* eval.c:461:OF */ extern void symlispcall (object sym, object *base, int narg); /* (sym, base, narg) object sym; object *base; int narg; */
/* eval.c:549:OF */ extern object simple_lispcall (object *funp, int narg); /* (funp, narg) object *funp; int narg; */
/* eval.c:645:OF */ extern object simple_symlispcall (object sym, object *base, int narg); /* (sym, base, narg) object sym; object *base; int narg; */
/* eval.c:739:OF */ extern void super_funcall (object fun); /* (fun) object fun; */
/* eval.c:752:OF */ extern void super_funcall_no_event (object fun); /* (fun) object fun; */
/* eval.c:936:OF */ extern object Ievaln (object form,object *vals); /* (form) object form; */
#define Ieval1(a_) Ievaln(a_,0)
/* eval.c:944:OF */ extern void eval (object form); /* (form) object form; */
/* eval.c:1189:OF */ extern void Leval (void); /* () */
/* eval.c:1203:OF */ extern void Levalhook (void); /* () */
/* eval.c:1269:OF */ extern void Lconstantp (void); /* () */
/* eval.c:1293:OF */ extern object ieval (object x); /* (x) object x; */
/* eval.c:1309:OF */ extern object ifuncall1 (object fun, object arg1); /* (fun, arg1) object fun; object arg1; */
/* eval.c:1328:OF */ extern object ifuncall2 (object fun, object arg1, object arg2); /* (fun, arg1, arg2) object fun; object arg1; object arg2; */
/* eval.c:1348:OF */ extern object ifuncall3 (object fun, object arg1, object arg2, object arg3);
/* eval.c:1348:OF */ extern object ifuncall4 (object fun, object arg1, object arg2, object arg3, object arg4);
typedef void (*funcvoid)(void);
/* eval.c:1545:OF */ extern void gcl_init_eval (void); /* () */
/* fasdump.c:1465:OF */ extern object read_fasl_vector (object in); /* (in) object in; */
/* fat_string.c:435:OF */ extern void gcl_init_fat_string (void); /* () */
/* sfasli.c::OF */ extern void gcl_init_sfasl (void); /* () */
/* format.c:2084:OF */ extern void Lformat (void); /* () */
/* format.c:2171:OF */ extern void gcl_init_format (void); /* () */
/* frame.c:32:OF */ extern void unwind (frame_ptr fr, object tag) NO_RETURN; /* (fr, tag) frame_ptr fr; object tag; */
/* frame.c:58:OF */ extern frame_ptr frs_sch (object frame_id); /* (frame_id) object frame_id; */
/* frame.c:69:OF */ extern frame_ptr frs_sch_catch (object frame_id); /* (frame_id) object frame_id; */
/* funlink.c:19:OF */ extern void call_or_link (object sym, int setf, void **link); /* (sym, link) object sym; void **link; */
/* funlink.c:41:OF */ extern void call_or_link_closure (object sym, int setf,void **link, void **ptr); /* (sym, link, ptr) object sym; void **link; object *ptr; */
/* funlink.c:696:OF */ extern object call_proc0 (object sym, int setf,void *link); /* (sym, link) object sym; void *link; */
/* funlink.c:784:OF */ extern int clear_stack (object *beg, object *limit); /* (beg, limit) object *beg; object *limit; */
/* funlink.c:821:OF */ extern void gcl_init_links (void); /* () */
/* gbc.c:151:OF */ extern void enter_mark_origin (object *p); /* (p) object *p; */
/* gbc.c:938:OF */ extern void GBC (enum type t); /* (t) enum type t; */
/* sgbc.c:924:OF */ extern fixnum sgc_count_type (int t); /* (t) int t; */
/* sgbc.c:938:OF */ extern int sgc_start (void); /* () */
/* sgbc.c:1068:OF */ extern int sgc_quit (void); /* () */
/* sgbc.c:1131:OF */ extern void make_writable (unsigned long beg, unsigned long i); /* (beg, i) int beg; int i; */
#ifndef __MINGW32__
/* #include <signal.h> */
#endif
/* sgbc.c:1246:OF */ extern int memory_protect (int on); /* (on) int on; */
/* sgbc.c:1306:OF */ extern void perm_writable (char *p, long n); /* (p, n) char *p; int n; */
/* sgbc.c:1321:OF */ extern void system_error (void); /* () */
/* gbc.c:1357:OF */ extern void gcl_init_GBC (void); /* () */
/* gnumalloc.c:286:OF */ extern void malloc_init (char *start, void (*warnfun) (/* ??? */)); /* (start, warnfun) char *start; void (*warnfun)(); */
/* gnumalloc.c:301:OF */ extern int malloc_usable_size (char *mem); /* (mem) char *mem; */
/* gnumalloc.c:737:OF */ extern int get_lim_data (void); /* () */
/* grab_defs.c:35:OF */ extern int read_some (char *buf, int n, int start_ch, int copy); /* (buf, n, start_ch, copy) char *buf; int n; int start_ch; int copy; */
/* grab_defs.c:71:OF */ /*  extern int main (void); */ /* () */
/* iteration.c:457:OF */ extern void gcl_init_iteration (void); /* () */
/* let.c:29:OF */ extern void let_var_list (object var_list); /* (var_list) object var_list; */
/* let.c:321:OF */ extern void gcl_init_let (void); /* () */
/* lex.c:34:OF */ extern object assoc_eq (object key, object alist); /* (key, alist) object key; object alist; */
/* lex.c:47:OF */ extern void lex_fun_bind (object name, object fun); /* (name, fun) object name; object fun; */
/* lex.c:59:OF */ extern void lex_macro_bind (object name, object exp_fun); /* (name, exp_fun) object name; object exp_fun; */
/* lex.c:70:OF */ extern void lex_tag_bind (object tag, object id); /* (tag, id) object tag; object id; */
/* lex.c:82:OF */ extern void lex_block_bind (object name, object id); /* (name, id) object name; object id; */
/* lex.c:95:OF */ extern object lex_tag_sch (object tag); /* (tag) object tag; */
/* lex.c:110:OF */ extern object lex_block_sch (object name); /* (name) object name; */
/* lex.c:125:OF */ extern void gcl_init_lex (void); /* () */
/* macros.c:139:OF */ extern object Imacro_expand1 (object exp_fun, object form); /* (exp_fun, form) object exp_fun; object form; */
/* macros.c:173:OF */ extern void Lmacroexpand (void); /* () */
/* macros.c:224:OF */ extern void Lmacroexpand_1 (void); /* () */
/* macros.c:265:OF */ extern object macro_expand (object form); /* (form) object form; */
/* macros.c:344:OF */ extern void gcl_init_macros (void); /* () */
/* main.c:111:OF */ extern int main (int argc, char **argv, char **envp); /* (argc, argv, envp) int argc; char **argv; char **envp; */
/* main.c:346:OF */ extern void install_segmentation_catcher (void); /* () */
/* main.c:359:OF */ extern void error (char *s); /* (s) char *s; */
/* main.c:519:OF */ extern object vs_overflow (void); /* () */
/* main.c:528:OF */ extern void bds_overflow (void); /* () */
/* main.c:537:OF */ extern void frs_overflow (void); /* () */
/* main.c:546:OF */ extern void ihs_overflow (void); /* () */
/* main.c:556:OF */ extern void segmentation_catcher (int,long,void *,char *); /* () */
/* main.c:587:OF */ extern void Lby (void); /* () */
/* main.c:607:OF */ extern void Lquit(void); /* () */
/* main.c:612:OF */ extern void Lexit(void); /* () */
/* main.c:619:OF */ extern int c_trace (void); /* () */
/* main.c:695:OF */ extern void siLreset_stack_limits (void); /* (arg) int arg; */
/* main.c:797:OF */ extern void Lidentity(void); /* () */
/* main.c:805:OF */ extern void Llisp_implementation_version(void); /* () */
/* makefun.c:10:OF */ extern object MakeAfun (object (*addr)(object,object), unsigned int argd, object data); /* (addr, argd, data) int (*addr)(); unsigned int argd; object data; */
/* makefun.c:122:OF */ extern void SI_makefun (char *strg, object (*fn) (/* ??? */), unsigned int argd); /* (strg, fn, argd) char *strg; object (*fn)(); unsigned int argd; */
/* makefun.c:131:OF */ extern void LISP_makefun (char *strg, object (*fn) (/* ??? */), unsigned int argd); /* (strg, fn, argd) char *strg; object (*fn)(); unsigned int argd; */
/* makefun.c:122:OF */ extern void GMP_makefunb (char *strg, object (*fn) (/* ??? */), unsigned int argd,object p); /* (strg, fn, argd) char *strg; object (*fn)(); unsigned int argd; */
/* makefun.c:122:OF */ extern void SI_makefunm (char *strg, object (*fn) (/* ??? */), unsigned int argd); /* (strg, fn, argd) char *strg; object (*fn)(); unsigned int argd; */
/* makefun.c:131:OF */ extern void LISP_makefunm (char *strg, object (*fn) (/* ??? */), unsigned int argd); /* (strg, fn, argd) char *strg; object (*fn)(); unsigned int argd; */
/* mapfun.c:324:OF */ extern void gcl_init_mapfun (void); /* () */
/* multival.c:32:OF */ extern void Lvalues (void); /* () */
/* multival.c:37:OF */ extern void Lvalues_list (void); /* () */
/* multival.c:134:OF */ extern void gcl_init_multival (void); /* () */
object funcall_vec(object,fixnum,object *);
/* nfunlink.c:190:OF */ extern object IapplyVector (object fun, int nargs, object *base); /* (fun, nargs, base) object fun; int nargs; object *base; */
/* nfunlink.c:269:OF */ extern void Iinvoke_c_function_from_value_stack (object (*f)(), ufixnum fargd); /* (f, fargd) object (*f)(); int fargd; */
/* nsocket.c:190:OF */ extern int CreateSocket (int port, char *host, int server, char *myaddr, int myport, int async); /* (port, host, server, myaddr, myport, async) int port; char *host; int server; char *myaddr; int myport; int async; */
/* nsocket.c:484:OF */ extern int getOneChar (FILE *fp); /* (fp) FILE *fp; */
/* nsocket.c:539:OF */ extern void ungetCharGclSocket (int c, object strm); /* (c, strm) int c; object strm; */
#ifndef __MINGW32__
/* nsocket.c:592:OF */ extern void tcpCloseSocket (int fd); /* (fd) int fd; */
/* nsocket.c:575:OF */ extern int TcpOutputProc (int fd, char *buf, int toWrite, int *errorCodePtr); /* (fd, buf, toWrite, errorCodePtr) int fd; char *buf; int toWrite; int *errorCodePtr; */
#endif
/* nsocket.c:619:OF */ extern int getCharGclSocket (object strm, object block); /* (strm, block) object strm; object block; */
/* num_arith.c:31:OF */ extern object fixnum_add (fixnum i, fixnum j); /* (i, j) int i; int j; */
/* num_arith.c:48:OF */ extern object fixnum_sub (fixnum i, fixnum j); /* (i, j) int i; int j; */
/* num_arith.c:100:OF */ extern object number_plus (object x, object y); /* (x, y) object x; object y; */
/* num_arith.c:246:OF */ extern object one_plus (object x); /* (x) object x; */
/* num_arith.c:292:OF */ extern object number_minus (object x, object y); /* (x, y) object x; object y; */
/* num_arith.c:438:OF */ extern object one_minus (object x); /* (x) object x; */
/* num_arith.c:478:OF */ extern object number_negate (object x); /* (x) object x; */
/* num_arith.c:520:OF */ extern object number_times (object x, object y); /* (x, y) object x; object y; */
/* num_arith.c:670:OF */ extern object number_divide (object x, object y); /* (x, y) object x; object y; */
/* num_arith.c:818:OF */ extern object integer_divide1 (object x, object y,fixnum z); /* (x, y) object x; object y; */
/* num_arith.c:818:OF */ extern object integer_divide2 (object x, object y,fixnum z,object *r); /* (x, y) object x; object y; */
/* num_arith.c:828:OF */ extern object get_gcd (object x, object y); /* (x, y) object x; object y; */
/* num_arith.c:873:OF */ extern void Lplus (void); /* () */
/* num_arith.c:889:OF */ extern void Lminus (void); /* () */
/* num_arith.c:907:OF */ extern void Ltimes (void); /* () */
/* num_arith.c:923:OF */ extern void Ldivide (void); /* () */
/* num_arith.c:1029:OF */ extern void gcl_init_num_arith (void); /* () */
/* num_co.c:292:OF */ extern object double_to_integer (double d); /* (d) double d; */
/* num_co.c:372:OF */ extern void Lfloat (void); /* () */
/* num_co.c:424:OF */ extern void Lnumerator (void); /* () */
/* num_co.c:432:OF */ extern void Ldenominator (void); /* () */
/* num_co.c:442:OF */ extern void Lfloor (void); /* () */
/* num_co.c:563:OF */ extern void Lceiling (void); /* () */
/* num_co.c:684:OF */ extern void Ltruncate (void); /* () */
/* num_co.c:766:OF */ extern void Lround (void); /* () */
/* num_co.c:896:OF */ extern void Lmod (void); /* () */
/* num_co.c:987:OF */ extern void Lfloat_radix (void); /* () */
/* num_co.c:1089:OF */ extern void Linteger_decode_float (void); /* () */
/* num_co.c:1114:OF */ extern void Lcomplex (void); /* () */
/* num_co.c:1136:OF */ extern void Lrealpart (void); /* () */
/* num_co.c:1147:OF */ extern void Limagpart (void); /* () */
/* num_co.c:1185:OF */ extern void gcl_init_num_co (void); /* () */
/* num_comp.c:40:OF */ extern int number_compare (object x, object y); /* (x, y) object x; object y; */
/* num_comp.c:269:OF */ extern void Lmonotonically_increasing (void); /* () */
/* num_comp.c:271:OF */ extern void Lmonotonically_nondecreasing (void); /* () */
/* num_comp.c:272:OF */ extern void Lmonotonically_nonincreasing (void); /* () */
/* num_comp.c:292:OF */ extern void Lmin (void); /* () */
/* num_comp.c:309:OF */ extern void gcl_init_num_comp (void); /* () */
/* num_log.c:224:OF */ extern object integer_fix_shift (object x, fixnum w); /* (x, w) object x; int w; */
/* num_log.c:258:OF */ extern void Llogior (void); /* () */
/* num_log.c:279:OF */ extern void Llogxor (void); /* () */
/* num_log.c:299:OF */ extern void Llogand (void); /* () */
/* num_log.c:339:OF */ extern void Lboole (void); /* () */
/* num_log.c:380:OF */ extern void Llogbitp (void); /* () */
/* num_log.c:420:OF */ extern void Lash (void); /* () */
/* num_log.c:482:OF */ extern void Linteger_length (void); /* () */
/* num_log.c:549:OF */ extern void gcl_init_num_log (void); /* () */
/* num_log.c:585:OF */ extern void siLbit_array_op (void); /* () */
/* num_pred.c:31:OF */ extern int number_zerop (object x); /* (x) object x; */
/* num_pred.c:67:OF */ extern int number_plusp (object x); /* (x) object x; */
/* num_pred.c:107:OF */ extern int number_minusp (object x); /* (x) object x; */
/* num_pred.c:147:OF */ extern int number_oddp (object x); /* (x) object x; */
/* num_pred.c:161:OF */ extern int number_evenp (object x); /* (x) object x; */
/* num_pred.c:240:OF */ extern void gcl_init_num_pred (void); /* () */
/* num_rand.c:111:OF */ extern void Lrandom (void); /* () */
/* num_rand.c:151:OF */ extern void gcl_init_num_rand (void); /* () */
/* num_sfun.c:91:OF */ extern object number_expt (object x, object y); /* (x, y) object x; object y; */
/* num_sfun.c:453:OF */ extern void Lexp (void); /* () */
/* num_sfun.c:469:OF */ extern void Llog (void); /* () */
/* num_sfun.c:488:OF */ extern void Lsqrt (void); /* () */
/* num_sfun.c:495:OF */ extern void Lsin (void); /* () */
/* num_sfun.c:502:OF */ extern void Lcos (void); /* () */
/* num_sfun.c:516:OF */ extern void Latan (void); /* () */
/* num_sfun.c:535:OF */ extern void gcl_init_num_sfun (void); /* () */
/* number.c:35:OF */ extern long int fixint (object x); /* (x) object x; */
/* number.c:44:OF */ extern int fixnnint (object x); /* (x) object x; */
/* number.c:81:OF */ extern object make_fixnum1 (long i); /* (i) int i; */
/* number.c:102:OF */ extern object make_ratio (object num, object den,int); /* (num, den) object num; object den; */
/* number.c:144:OF */ extern object make_shortfloat (float f); /* (f) double f; */
/* number.c:157:OF */ extern object make_longfloat (longfloat f); /* (f) longfloat f; */
/* number.c:170:OF */ extern object make_complex (object r, object i); /* (r, i) object r; object i; */
/* number.c:229:OF */ extern double number_to_double (object x); /* (x) object x; */
/* number.c:254:OF */ extern void gcl_init_number (void); /* () */
/* peculiar.c:14:OF */ /*  extern int main (void); */ /* () */
/* predicate.c:46:OF */ extern void Lsymbolp (void); /* () */
/* predicate.c:176:OF */ extern void Lcomplexp (void); /* () */
/* predicate.c:238:OF */ extern void Lsimple_string_p (void); /* () */
/* predicate.c:253:OF */ extern void Lsimple_bit_vector_p (void); /* () */
/* predicate.c:268:OF */ extern void Lsimple_vector_p (void); /* () */
/* predicate.c:301:OF */ extern void Lpackagep (void); /* () */
/* predicate.c:313:OF */ extern void Lfunctionp (void); /* () */
/* predicate.c:344:OF */ extern void Lcompiled_function_p (void); /* () */
/* predicate.c:393:OF */ extern bool eql1 (object x, object y); /* (x, y) object x; object y; */
/* predicate.c:393:OF */ extern bool oeql (object x, object y); /* (x, y) object x; object y; */
/* predicate.c:469:OF */ extern bool equal1 (register object x, register object y); /* (x, y) register object x; register object y; */
/* predicate.c:469:OF */ extern bool oequal (register object x, register object y); /* (x, y) register object x; register object y; */
/* predicate.c:557:OF */ extern bool equalp1 (object x, object y); /* (x, y) object x; object y; */
/* predicate.c:557:OF */ extern bool oequalp (object x, object y); /* (x, y) object x; object y; */
/* predicate.c:750:OF */ extern bool contains_sharp_comma (object x); /* (x) object x; */
/* predicate.c:833:OF */ extern void gcl_init_predicate_function (void); /* () */
/* prog.c:48:OF */ extern void Ftagbody (object body); /* (body) object body; */
/* prog.c:246:OF */ extern void Fprogn (object body); /* (body) object body; */
/* prog.c:303:OF */ extern void gcl_init_prog (void); /* () */
/* reference.c:32:OF */ extern void Lfboundp (void); /* () */
/* reference.c:49:OF */ extern object symbol_function (object sym); /* (sym) object sym; */
/* reference.c:69:OF */ extern void Lsymbol_function (void); /* () */
/* reference.c:143:OF */ extern void Lsymbol_value (void); /* () */
/* reference.c:156:OF */ extern void Lboundp (void); /* () */
/* reference.c:169:OF */ extern void Lmacro_function (void); /* () */
/* reference.c:180:OF */ extern void Lspecial_form_p (void); /* () */
/* reference.c:191:OF */ extern void gcl_init_reference (void); /* () */
/*  #include "regexp.h" */
/* regexp.c:1588:OF */ extern void regerror (char *s); /* (s) char *s; */
/* save.c:17:OF */ extern void siLsave (void); /* () */
#include <unistd.h>
/* sbrk.c:9:OF */ /*  extern void * sbrk (int n); */ /* (n) int n; */
/* strcspn.c:3:OF */ /*  extern size_t strcspn (const char *s1, const char *s2); */ /* (s1, s2) char *s1; char *s2; */
/* structure.c:59:OF */ extern object structure_ref (object x, object name, fixnum i); /* (x, name, i) object x; object name; int i; */
/* structure.c:107:OF */ extern object structure_set (object x, object name, fixnum i, object v); /* (x, name, i, v) object x; object name; int i; object v; */
/* structure.c:164:OF */ extern object structure_to_list (object x); /* (x) object x; */
/* structure.c:188:OF */ extern void siLmake_structure (void); /* () */
/* structure.c:281:OF */ extern void siLstructure_set (void); /* () */
/* structure.c:326:OF */ extern void siLlist_nth (void); /* () */
/* structure.c:439:OF */ extern void gcl_init_structure_function (void); /* () */
/* toplevel.c:211:OF */ extern void gcl_init_toplevel (void); /* () */
/* typespec.c:294:OF */ extern void Ltype_of (void); /* () */
/* typespec.c:493:OF */ extern void gcl_init_typespec (void); /* () */
/* typespec.c:497:OF */ extern void gcl_init_typespec_function (void); /* () */
/* unexec-19.29.c:1016:OF */ extern int write_segment (int new, register char *ptr, register char *end); /* (new, ptr, end) int new; register char *ptr; register char *end; */
/* unexec.c:1016:OF */ extern int write_segment (int new, register char *ptr, register char *end); /* (new, ptr, end) int new; register char *ptr; register char *end; */
/* unexlin.c:808:OF */ extern int write_segment (int new, register char *ptr, register char *end); /* (new, ptr, end) int new; register char *ptr; register char *end; */
/* unixfasl.c:409:OF */ extern void gcl_init_unixfasl (void); /* () */
/* unixfsys.c:145:OF */ extern char *getwd (char *buffer); /* (buffer) char *buffer; */
/* unixfsys.c:209:OF */ extern void coerce_to_filename1 (object pathname, char *p,unsigned sz); /* (pathname, p) object pathname; char *p; */
/* unixfsys.c:209:OF */ extern void coerce_to_local_filename1 (object pathname, char *p,unsigned sz); /* (pathname, p) object pathname; char *p; */
/* unixfsys.c:329:OF */ extern bool file_exists (object file); /* (file) object file; */
/* unixfsys.c:359:OF */ extern FILE *backup_fopen (char *filename, char *option); /* (filename, option) char *filename; char *option; */
/* unixfsys.c:359:OF */ extern FILE *fopen_not_dir (char *filename, char *option); /* (filename, option) char *filename; char *option; */
/* unixfsys.c:372:OF */ extern int file_len (FILE *fp); /* (fp) FILE *fp; */
/* unixfsys.c:382:OF */ extern object truename (object); /* () */
/* unixfsys.c:382:OF */ extern void Ltruename (void); /* () */
/* unixfsys.c:456:OF */ extern void Lprobe_file (void); /* () */
/* unixfsys.c:533:OF */ extern void Ldirectory (void); /* () */
/* unixfsys.c:777:OF */ extern void gcl_init_unixfsys (void); /* () */
/* unixsave.c:173:OF */ extern void gcl_init_unixsave (void); /* () */
/* unixsys.c:87:OF */ extern void gcl_init_unixsys (void); /* () */
/* unixtime.c:67:OF */ extern int runtime (void); /* () */
/* unixtime.c:82:OF */ extern object unix_time_to_universal_time (int i); /* (i) int i; */
/* unixtime.c:173:OF */ extern void gcl_init_unixtime (void); /* () */
/* user_init.c:2:OF */ extern object user_init (void); /* () */
/* user_init.c:2:OF */ extern int user_match (const char *,int n); /* () */
/* usig.c:49:OF */ extern void gcl_signal (int signo, void (*handler) (/* ??? */)); /* (signo, handler) int signo; void (*handler)(); */
/* usig.c:92:OF */ extern int unblock_signals (int n, int m); /* (n, m) int n; int m; */
/* usig.c:119:OF */ extern void unblock_sigusr_sigio (void); /* () */
/* usig.c:182:OF */ extern void install_default_signals (void); /* () */
/* usig2.c:142:OF */ extern void gcl_init_safety (void); /* () */
/* usig2.c:158:OF */ extern object sSsignal_safety_required (fixnum signo,fixnum safety); /* (signo, safety) int signo; int safety; */
#ifdef __MINGW32__
/* usig2.c:167:OF */ extern void main_signal_handler (int signo); /* (signo) int signo */
#else
/* /\* usig2.c:167:OF *\/ extern void main_signal_handler (int signo, siginfo_t *a, void *b); /\* (signo, a, b) int signo; int a; int b; *\/ */
#endif
/* usig2.c:375:OF */ extern void raise_pending_signals (int cond); /* (cond) int cond; */
/* utils.c:12:OF */ extern object IisSymbol (object f); /* (f) object f; */
/* utils.c:20:OF */ extern object IisFboundp (object f); /* (f) object f; */
/* utils.c:30:OF */ extern object IisArray (object f); /* (f) object f; */
/* utils.c:44:OF */ extern object Iis_fixnum (object f); /* (f) object f; */
/* utils.c:61:OF */ extern object Iapply_ap_new (fixnum n,object (*f) (/* ??? */), object first, va_list ap); /* (f, ap) object (*f)(); va_list ap; */
/* utils.c:178:OF */ extern object Icheck_one_type (object x, enum type t); /* (x, t) object x; enum type t; */
/* utils.c:202:OF */ extern object Ivs_values (void); /* () */
/* utils.c:227:OF */ extern char *lisp_copy_to_null_terminated (object string, char *buf, int n); /* (string, buf, n) object string; char *buf; int n; */


/*  readline.d */
extern int readline_on;
void gcl_init_readline_function(void);
void gcl_init_readline(void);

/*  sys_gcl.c */
void gcl_init_init(void);

/* misc */
void gcl_init_symbol(void);

void gcl_init_package(void);

void gcl_init_character(void);

void gcl_init_read(void);

void gcl_init_pathname(void);

void gcl_init_print(void);

void gcl_init_character_function(void);

void gcl_init_file_function(void);

void gcl_init_list_function(void);

void gcl_init_package_function(void);

void gcl_init_pathname_function(void);

void gcl_init_print_function(void);

void gcl_init_read_function(void);

void gcl_init_sequence_function(void);

void gcl_init_string_function(void);

void gcl_init_symbol_function(void);

void gcl_init_socket_function(void);

void gcl_init_hash(void);

void import(object,object);

void export(object,object);

void NewInit(void);

void gcl_init_system(object);

void set_up_string_register(char *);

bool endp1(object);

void stack_cons(void);

bool char_equal(object,object);

bool string_equal(object,object);

bool string_eq(object,object);

bool remf(object *,object);

bool keywordp(object);

int pack_hash(object);

void load(const char *);

bool member_eq(object,object);

void delete_eq(object,object *);

int length(object);

int rl_getc_em(FILE *);

void setupPRINTdefault(object,object);

void write_str(char *);

void cleanupPRINT(void);

int fasload(object);

int readc_stream(object);

void unreadc_stream(int,object);

void end_of_stream(object);

bool stream_at_end(object);

int digitp(int,int);

bool char_eq(object,object);

bool listen_stream(object);

void get_string_start_end(object,object,object,int *,int *);

int file_column(object);

int writec_stream(int,object);

int writec_pstream(int,object);

void
write_codes_pstream(object,fixnum,fixnum,fixnum,fixnum);

void *writec_stream_fun(object);

object output_stream(object);

int digit_weight(int,int);

void flush_stream(object);

void writestr_pstream(char *,object);

void write_string(object,object);

void edit_double(int, double, int *, char *, int *, int);

void sethash(object,object,object);

int file_position(object);

int file_position_set(object, int);

void princ_str(char *s, object);

void close_stream(object);

void build_symbol_table(void);

void gcl_init_file(void);

object aset1(object,fixnum,object);

void dfprintf(FILE *,char *,...);

void Lmake_list(void);

void Llast(void);

void Lgensym(void);

void Lldiff(void);

void Lintern(void);

void Lgensym(void);

void Lldiff(void);

void Lgensym(void);

void Lintern(void);

void Lintern(void);

void Lreconc(void);

void Lmember(void);

void Ladjoin(void);

void Llist(void);

void Lappend(void);

void Lread(void);

void Lread_char(void);

void Lchar_eq(void);

void Lwrite_char(void);

void Lforce_output(void);

void Lchar_neq(void);

void Llist(void);

void Lwrite(void);

void Lfresh_line(void);

void Lsymbol_package(void);

void Lfind_package(void);

void Lfind_symbol(void);

void Lpackage_name(void);

void Lsymbol_plist(void);

void Lpackage_nicknames(void);

void Lpackage_use_list(void);

void Lpackage_used_by_list(void);

void Lstandard_char_p(void);

void Lstring_char_p(void);

void Lchar_code(void);

void Lchar_bits(void);

void Lchar_font(void);

void Lread_line(void);

void siLpackage_internal(void);

void siLpackage_external(void);

void Llist_all_packages(void);

void Lgensym(void);

void Lread(void);

void Lwrite(void);

void Lstring_equal(void);

void Lclose(void);

void Lnamestring(void);

void Lmake_echo_stream(void);

void Lmake_broadcast_stream(void);

void Lmake_two_way_stream(void);

void Lbutlast(void);

void Ladjoin(void);

void Lstring_downcase(void);

void Lmember(void);

void Lgensym(void);

void Llist_all_packages(void);

void Lfind_symbol(void);

void Lstring_equal(void);

void Lfind_package(void);

void siLpackage_internal(void);

void siLpackage_external(void);

void Lpackage_use_list(void);

void Lreconc(void);

void Lstandard_char_p(void);

void Lstring_char_p(void);

void Lcharacter(void);

void Llength(void);

void Lreconc(void);

void Llength(void);

void Lgensym(void);

void Llist_length(void);

void Lgensym(void);

void Lbutlast(void);

void Lnconc(void);

void Lfind_package(void);

void Lpackage_name(void);

void Llist(void);

void Lfresh_line(void);

void Lread_char(void);

void Lunread_char(void);

void Lread_line(void);

void Lread(void);

void Lforce_output(void);

void Lwrite(void);

void Lmember(void);

void siLpackage_internal(void);

void siLpackage_external(void);

void Lmake_pathname(void);

void Lnamestring(void);

void Lclose(void);

void Lgensym(void);

void Lfresh_line(void);

void Llist(void);

void Lread_char(void);

void Lchar_eq(void);

void Lfinish_output(void);

void Lchar_neq(void);

void Lwrite(void);

void Lgensym(void);

void Lmember(void);

void Lappend(void);

void Lcopy_tree(void);

void Ladjoin(void);

void Lgetf(void);

void Lsubst(void);

void Lsymbol_package(void);

void Lcopy_list(void);

void Lintern(void);

void Lfind_package(void);

void LlistA(void);

void Llist(void);

void Lgetf(void);

void Lstreamp(void);

void Lpeek_char(void);

void Lread_char(void);

void Lread_line(void);

void Lset_macro_character(void);

void Lclrhash(void);

void siLhash_set(void);

void Lgethash(void);

struct cons * gethash(object,object);

void Lremhash(void);

void Llist_all_packages(void);

void Lintern(void);

void Lunintern(void);

void Lsubseq(void);

void Lsymbol_package(void);

void Lfind_package(void);

void siLpackage_internal(void);

void siLpackage_external(void);

void Lread_char(void);

void Lfile_length(void);

void Lfile_position(void);

void Lclose(void);

void Lsubseq(void);

void Lnamestring(void);

void Lmerge_pathnames(void);

void Lcopy_list(void);

void Lread_line(void);

void Lgensym(void);

void Lcopy_list(void);

void Lintern(void);

void Lappend(void);

void Lgensym(void);

void Lcopy_list(void);

void Lmember(void);

void Lintern(void);

void Lappend(void);

void Lfind_package(void);

void Lpackage_name(void);

void Lpackage_nicknames(void);

void Lpackage_use_list(void);

void siLpackage_external(void);

void siLpackage_internal(void);

void Lsymbol_package(void);

void Lappend(void);

void Lgentemp(void);

void Lgensym(void);

void Lassoc(void);

void Ladjoin(void);

void Lstring_eq(void);

void Lmember(void);

void Lgethash(void);

void Lfinish_output(void);

void Lread(void);

void Lmake_hash_table(void);

void siLhash_set(void);

void Lrevappend(void);

void Lreconc(void);

void Lcopy_list(void);

void LlistA(void);

void Lfind_package(void);

void siLpackage_internal(void);

void siLpackage_external(void);

void princ_char(int,object);

void Ldigit_char_p(void);

void Lwrite_byte(void);

#ifdef SPECIAL_RSYM
void read_special_symbols(char *);

/* int */
/* node_compare(const void *,const void *); */
#endif

void FEpackage_error(object,const char *s);

void FEcannot_coerce(object, object);

int system_time_zone_helper(void);

object call_proc_new(object sym,ufixnum clp,ufixnum vald,void **link,ufixnum argd,object first,va_list ll);

object  call_vproc_new(object,int setf,int pop_one_arg,void *,object,va_list);

void funcall_with_catcher(object, object);

void siLset_symbol_plist(void);

void Lhash_table_p(void);

void Lreadtablep(void);

fixnum fixnum_expt(fixnum,fixnum);

void check_alist(object);

void ck_larg_at_least(int,object);

void vfun_wrong_number_of_args(object);

/* FIXME from lfun_list.lsp -- should be automatically generated */
void Lgensym(void);
void Lsubseq(void);
void Lminusp(void);
void Linteger_decode_float(void);
void Lminus(void);
void Lint_char(void);
void Lchar_int(void);
void Lall_different(void);
void Lcopy_seq(void);
void Lkeywordp(void);
void Lname_char(void);
void Lchar_name(void);
void Lrassoc_if(void);
void Lmake_list(void);
void Lhost_namestring(void);
void Lmake_echo_stream(void);
void Lnth(void);
void Lsin(void);
void Lnumerator(void);
void Larray_rank(void);
void Lcaar(void);
void Lboth_case_p(void);
void Lnull(void);
void Lrename_file(void);
void Lfile_author(void);
void Lstring_capitalize(void);
void Lmacroexpand(void);
void Lnconc(void);
void Lboole(void);
void Ltailp(void);
void Lconsp(void);
void Llistp(void);
void Lmapcan(void);
void Llength(void);
void Lrassoc(void);
void Lpprint(void);
void Lpathname_host(void);
void Lnsubst_if_not(void);
void Lfile_position(void);
void Lstring_l(void);
void Lreverse(void);
void Lstreamp(void);
void siLputprop(void);
void Lremprop(void);
void Lsymbol_package(void);
void Lnstring_upcase(void);
void Lstring_ge(void);
void Lrealpart(void);
void Lnbutlast(void);
void Larray_dimension(void);
void Lcdr(void);
void Leql(void);
void Llog(void);
void Ldirectory(void);
void Lstring_not_equal(void);
void Lshadowing_import(void);
void Lmapc(void);
void Lmapl(void);
void Lmakunbound(void);
void Lcons(void);
void Llist(void);
void Luse_package(void);
void Lfile_length(void);
void Lmake_symbol(void);
void Lstring_right_trim(void);
void Lenough_namestring(void);
void Lprint(void);
void Lcddaar(void);
void Lcdadar(void);
void Lcdaadr(void);
void Lcaddar(void);
void Lcadadr(void);
void Lcaaddr(void);
void Lset_macro_character(void);
void Lforce_output(void);
void Lnthcdr(void);
void Llogior(void);
void Lchar_downcase(void);
void Lstream_element_type(void);
void Lpackage_used_by_list(void);
void Ldivide(void);
void Lmaphash(void);
void Lstring_eq(void);
void Lpairlis(void);
void Lsymbolp(void);
void Lchar_not_lessp(void);
void Lone_plus(void);
void Lby(void);
void Lnsubst_if(void);
void Lcopy_list(void);
void Ltan(void);
void Lset(void);
void Lfunctionp(void);
void Lwrite_byte(void);
void Llast(void);
void Lmake_string(void);
void Lcaaar(void);
void Llist_length(void);
void Lcdddr(void);
void Lprin1(void);
void Lprinc(void);
void Llower_case_p(void);
void Lchar_le(void);
void Lstring_equal(void);
void Lclear_output(void);
void CERROR(void);
void Lterpri(void);
void Lnsubst(void);
void Lunuse_package(void);
void Lstring_not_greaterp(void);
void Lstring_g(void);
void Lfinish_output(void);
void Lspecial_form_p(void);
void Lstringp(void);
void Lget_internal_run_time(void);
void Ltruncate(void);
void Lcode_char(void);
void Lchar_code(void);
void Lsimple_string_p(void);
void Lrevappend(void);
void Lhash_table_count(void);
void Lpackage_use_list(void);
void Lrem(void);
void Lmin(void);
void Lapplyhook(void);
void Lexp(void);
void Lchar_lessp(void);
void Lcdar(void);
void Lcadr(void);
void Llist_all_packages(void);
void Lcdr(void);
void Lcopy_symbol(void);
void Lacons(void);
void Ladjustable_array_p(void);
void Lsvref(void);
void Lapply(void);
void Ldecode_float(void);
void Lsubst_if_not(void);
void Lrplaca(void);
void Lsymbol_plist(void);
void Lwrite_string(void);
void Llogeqv(void);
void Lstring(void);
void Lstring_upcase(void);
void Lceiling(void);
void Lgethash(void);
void Ltype_of(void);
void Lbutlast(void);
void Lone_minus(void);
void Lmake_hash_table(void);
void Lstring_neq(void);
void Lmonotonically_nondecreasing(void);
void Lmake_broadcast_stream(void);
void Limagpart(void);
void Lintegerp(void);
void Lread_char(void);
void Lpeek_char(void);
void Lchar_font(void);
void Lstring_greaterp(void);
void Loutput_stream_p(void);
void Lash(void);
void Llcm(void);
void Lelt(void);
void Lcos(void);
void Lnstring_downcase(void);
void Lcopy_alist(void);
void Latan(void);
void Ldelete_file(void);
void Lfloat_radix(void);
void Lsymbol_name(void);
void Lclear_input(void);
void Lfind_symbol(void);
void Lchar_l(void);
void Lhash_table_p(void);
void Levenp(void);
void siLcmod(void);
void siLcplus(void);
void siLctimes(void);
void siLcdifference(void);
void Lzerop(void);
void Lcaaaar(void);
void Lchar_ge(void);
void Lcdddar(void);
void Lcddadr(void);
void Lcdaddr(void);
void Lcadddr(void);
void Lfill_pointer(void);
void Lmapcar(void);
void Lfloatp(void);
void Lshadow(void);
void Lmacroexpand_1(void);
void Lsxhash(void);
void Llisten(void);
void Larrayp(void);
void Lmake_pathname(void);
void Lpathname_type(void);
void Lfuncall(void);
void Lclrhash(void);
void Lgraphic_char_p(void);
void Lfboundp(void);
void Lnsublis(void);
void Lchar_not_equal(void);
void Lmacro_function(void);
void Lsubst_if(void);
void Lcomplexp(void);
void Lread_line(void);
void Lpathnamep(void);
void Lmax(void);
void Lin_package(void);
void Lreadtablep(void);
void Lfloat_sign(void);
void Lcharacterp(void);
void Lread(void);
void Lnamestring(void);
void Lunread_char(void);
void Lcdaar(void);
void Lcadar(void);
void Lcaadr(void);
void Lchar_eq(void);
void Lalpha_char_p(void);
void Lstring_trim(void);
void Lmake_package(void);
void Lclose(void);
void Ldenominator(void);
void Lfloat(void);
void Lcar(void);
void Lround(void);
void Lsubst(void);
void Lupper_case_p(void);
void Larray_element_type(void);
void Ladjoin(void);
void Llogand(void);
void Lmapcon(void);
void Lintern(void);
void Lvalues(void);
void Lexport(void);
void Ltimes(void);
void Lmonotonically_increasing(void);
void Lcomplex(void);
void Lset_syntax_from_char(void);
void Lchar_bit(void);
void Linteger_length(void);
void Lpackagep(void);
void Linput_stream_p(void);
void Lmonotonically_nonincreasing(void);
void Lpathname(void);
void Leq(void);
void Lmake_char(void);
void Lfile_namestring(void);
void Lcharacter(void);
void Lsymbol_function(void);
void Lconstantp(void);
void Lchar_equal(void);
void Ltree_equal(void);
void Lcddr(void);
void Lgetf(void);
void Lsave(void);
void Lmake_random_state(void);
void Lchar_not_greaterp(void);
void Lexpt(void);
void Lsqrt(void);
void Lscale_float(void);
void Lchar_g(void);
void Lldiff(void);
void Lassoc_if_not(void);
void Lbit_vector_p(void);
void Lnstring_capitalize(void);
void Lsymbol_value(void);
void Lrplacd(void);
void Lboundp(void);
void Lequalp(void);
void Lsimple_bit_vector_p(void);
void Lmember_if_not(void);
void Lmake_two_way_stream(void);
void Lparse_integer(void);
void Lplus(void);
void Lall_the_same(void);
void Lgentemp(void);
void Lrename_package(void);
void Lcommonp(void);
void Lnumberp(void);
void Lcopy_readtable(void);
void Lrandom_state_p(void);
void Ldirectory_namestring(void);
void Lstandard_char_p(void);
void Ltruename(void);
void Lidentity(void);
void Lnreverse(void);
void Lpathname_device(void);
void Lunintern(void);
void Lunexport(void);
void Lfloat_precision(void);
void Lstring_downcase(void);
void Lcar(void);
void Lconjugate(void);
void Lnull(void);
void Lread_char_no_hang(void);
void Lfresh_line(void);
void Lwrite_char(void);
void Lparse_namestring(void);
void Lstring_not_lessp(void);
void Lchar(void);
void Laref(void);
void Lpackage_nicknames(void);
void Lendp(void);
void Loddp(void);
void Lchar_upcase(void);
void LlistA(void);
void Lvalues_list(void);
void Lequal(void);
void Ldigit_char_p(void);
void Lchar_neq(void);
void Lpathname_directory(void);
void Lcdaaar(void);
void Lcadaar(void);
void Lcaadar(void);
void Lcaaadr(void);
void Lcddddr(void);
void Lget_macro_character(void);
void Lformat(void);
void Lcompiled_function_p(void);
void Lsublis(void);
void Lpathname_name(void);
void Limport(void);
void Llogxor(void);
void Lrassoc_if_not(void);
void Lchar_greaterp(void);
void Lmake_synonym_stream(void);
void Lalphanumericp(void);
void Lremhash(void);
void Lreconc(void);
void Lmonotonically_decreasing(void);
void Llogbitp(void);
void Lmaplist(void);
void Lvectorp(void);
void Lassoc_if(void);
void Lget_properties(void);
void Lstring_le(void);
void Levalhook(void);
void Lfile_write_date(void);
void Llogcount(void);
void Lmerge_pathnames(void);
void Lmember_if(void);
void Lread_byte(void);
void Lsimple_vector_p(void);
void Lchar_bits(void);
void Lcopy_tree(void);
void Lgcd(void);
void Lby(void);
void Lget(void);
void Lmod(void);
void Ldigit_char(void);
void Lprobe_file(void);
void Lstring_left_trim(void);
void Lpathname_version(void);
void Lwrite_line(void);
void Leval(void);
void Latom(void);
void Lcddar(void);
void Lcdadr(void);
void Lcaddr(void);
void Lfmakunbound(void);
void Lsleep(void);
void Lpackage_name(void);
void Lfind_package(void);
void Lassoc(void);
void Lset_char_bit(void);
void Lfloor(void);
void Lwrite(void);
void Lplusp(void);
void Lfloat_digits(void);
void Lread_delimited_list(void);
void Lappend(void);
void Lmember(void);
void Lstring_lessp(void);
void Lrandom(void);
void siLspecialp(void);
void siLoutput_stream_string(void);
void siLstructurep(void);
void siLcopy_stream(void);
void siLinit_system(void);
void siLstring_to_object(void);
void siLreset_stack_limits(void);
void siLdisplaced_array_p(void);
void siLrplaca_nthcdr(void);
void siLlist_nth(void);
void siLmake_vector(void);
void siLaset(void);
void siLsvset(void);
void siLfill_pointer_set(void);
void siLreplace_array(void);
void siLfset(void);
void siLhash_set(void);
void Lboole(void);
void siLpackage_internal(void);
void siLpackage_external(void);
void siLelt_set(void);
void siLchar_set(void);
void siLmake_structure(void);
void siLstructure_name(void);
void siLstructure_ref(void);
void siLstructure_set(void);
void siLput_f(void);
void siLrem_f(void);
void siLset_symbol_plist(void);
void siLbit_array_op(void);

object cmod(object);
object ctimes(object,object);
object cdifference(object,object);
object cplus(object,object);

object Icall_gen_error_handler(object,object,object,object,ufixnum,...);

/* #define Icall_error_handler(a_,b_,c_,d_...) \ */
/*   Icall_gen_error_handler(Cnil,null_string,a_,b_,c_,##d_) */
/* #define Icall_continue_error_handler(a_,b_,c_,d_,e_...) \ */
/*   Icall_gen_error_handler(Ct,a_,b_,c_,d_,##e_) */
/* object */
/* Icall_error_handler(object,object,int,...); */

/* object */
/* Icall_continue_error_handler(object,object,object,int,...); */

void * gcl_gmp_alloc(size_t);

void init_gmp_rnd_state(__gmp_randstate_struct *);

int my_plt(const char *,unsigned long *);

int my_pltp(const char *,unsigned long *);

int parse_plt(void);

int sgc_count_read_only_type(int);

int  gcl_isnormal_double(double);

int  gcl_isnormal_float(float);

int
gcl_isnan(object);

int
gcl_is_not_finite(object);

object powm_bbb(object,object,object);
object powm_bfb(object,fixnum,object);
object powm_fbb(fixnum,object,object);
object powm_ffb(fixnum,fixnum,object);
object powm_bbf(object,object,fixnum);
object powm_bff(object,fixnum,fixnum);
object powm_fbf(fixnum,object,fixnum);
object powm_fff(fixnum,fixnum,fixnum);

object find_init_name1(char *,unsigned);

int
gcl_isnan(object);

long opt_maxpage(struct typemanager *);

typedef MP_INT * GEN;

MP_INT * otoi(object);

MP_INT * stoi(fixnum);

object read_byte1(object,object);

#ifdef SGC
void memprotect_test_reset(void);
#endif


#if defined (__MINGW32__)
int bcmp ( const void *s1, const void *s2, size_t n );
void bcopy ( const void *s1, void *s2, size_t n );
void bzero(void *b, size_t length);
int TcpOutputProc ( int fd, char *buf, int toWrite, int *errorCodePtr, int block );
void gcl_init_shared_memory ( void );
void fix_filename ( object pathname, char *filename1 );
void alarm ( int n );
void *sbrk ( ptrdiff_t increment );
void sigemptyset( sigset_t *set);
void sigaddset ( sigset_t *set, int n);
int sigismember ( sigset_t *set, int n );
int sigprocmask ( int how, const sigset_t *set, sigset_t *oldset );
#endif

#if defined (__MINGW32__) || defined (__CYGWIN__)
void recreate_heap1 ( void );
#endif

void
gprof_cleanup(void);


unsigned long ihash_equal1(object,int);

object interactive_stream_p(object);

void reinit_gmp(void);

object macro_def_int(object);

/* void call_after_gbc_hook(int); */

int reset_plt(void);

int msystem(char *);

fcomplex object_to_fcomplex(object);

object make_fcomplex(fcomplex);

dcomplex object_to_dcomplex(object);

void assert_error(const char *,unsigned,const char *,const char *);

#ifdef _WIN32
void detect_wine(void);

void init_shared_memory(void);
#endif

void * object_to_pointer(object);

void * alloca(unsigned long);

object make_dcomplex(dcomplex);

object find_init_string(const char *);

object quick_call_function_cs(object,...);

object call_proc_cs(object,...);

void *
get_mmap(FILE *,void **);

void *
get_mmap_shared(FILE *,void **);

object call_proc_cs1(object,...);

int un_mmap(void *,void *);

object call_proc_cs2(object,...);

void isetq_fix(MP_INT *,int);
int mpz_to_mpz1(MP_INT *,MP_INT *,void *);
int mpz_to_mpz(MP_INT *,MP_INT *);
int obj_to_mpz1(object,MP_INT *,void *);
int obj_to_mpz(object,MP_INT *);

int update_real_maxpage(void);

fixnum set_tm_maxpage(struct typemanager *,fixnum);

fixnum elt_size(fixnum);

fixnum elt_mode(fixnum);

void init_gmp_rnd_state(__gmp_randstate_struct *);

/* void set_sgc_bit(struct pageinfo *,void *); */

void reinit_gmp(void);

object mod(object,object);

void intdivrem(object,object,fixnum,object *,object *);

object integer_count(object);

object integer_length(object);

bool integer_bitp(object,object);

object  fixnum_times(fixnum,fixnum);

object log_op2(fixnum,object,object);

object fixnum_big_shift(fixnum,fixnum);

object integer_shift(object,object);

object number_abs(object);

object number_signum(object);


object number_ldb(object,object);
object number_ldbt(object,object);
object number_dpb(object,object,object);
object number_dpf(object,object,object);

extern void *feval_src;
#if defined(DARWIN)
void init_darwin_zone_compat ();
#endif

int get_cstack_dir(VOL fixnum);

int
gcl_mprotect(void *,unsigned long,int);

void *
alloc_code_space(size_t,ufixnum);

void *
alloc_contblock_no_gc(size_t,char *);

struct pageinfo *
get_pageinfo(void *);

void
reset_contblock_freelist(void);

void
setup_rb(bool);

void
empty_relblock(void);

void
close_pool(void);

void
gcl_cleanup(int);

void
do_gcl_abort(void);

object
n_cons_from_x(fixnum,object);

int
mbrk(void *);

void
prelink_init(void);

fixnum
check_avail_pages(void);

void
resize_hole(ufixnum,enum type,bool);

void
maybe_set_hole_from_maxpages(void);

size_t
dir_name_length(const char *);

object
new_cfdata(void);

void
set_displaced_body_ptr(object);

void
travel_find_sharing(object,object);

object
coerce_funcall_object_to_function(object);

object
gcl_make_hash_table(object);

int
home_namestring1(const char *,int,char *,int);

object
double_to_rational(double);

object
fresh_synonym_stream_to_terminal_io(void);

void
set_array_elttype(object,enum aelttype);

object
apply_format_function(object,object,object,object,object,object);

object
fSstring_match2(object,object);

object
aelttype_list(void);

object alloc_simple_string(fixnum);
object alloc_string(fixnum);
object append(object,object);
object car(object);
object cdr(object);
object copy_list(object);
object copy_simple_string(object);
object current_package(void);
object find_package(object);
object find_symbol(object,object);
object getf(object,object,object);
object intern(object,object);
object make_constant(char *,object);
object make_keyword(char *);
object make_ordinary(char *);
object make_si_constant(char *,object);
object make_si_ordinary(char *);
object make_si_special(char *,object);
object make_special(char *,object);
object nreverse(object);
double object_to_double(object);
object open_stream(object,enum smmode,object,object);
object putf(object,object,object);
object putprop(object,object,object);
object read_object(object);
object read_object_non_recursive(object);
object make_symbol(object);
object reverse(object);
object alloc_bitvector(fixnum);
object alloc_simple_bitvector(fixnum);
object alloc_simple_vector(fixnum);
object alloc_vector(fixnum,enum aelttype);
object coerce_to_character(object);
object peek_char(bool,object);
object prin1(object,object);
object read_char(object);
object make_string_output_stream(int);
object make_gmp_ordinary(char *);
void *malloc(size_t);
void *realloc(void *,size_t);
void *alloc_contblock(size_t);
void *alloc_relblock(size_t);
object ifuncall(object,int,...);
object list(fixnum,...);
object listA(fixnum,...);
object vs_overflow(void);
object make_fixnum1(long);
object make_shortfloat(float);
long fixint(object);
object read_char1(object,object);
object Iapply_fun_n(object,int,int,...);
object Iapply_fun_n2(object,int,int,...);
object Ifuncall_n(object,int,...);
object funcall_cfun(void(*)(),int,...);
int gcl_init_cmp_anon(void);
int is_bigger_fixnum(void *);
int is_text_addr(void *);
int seek_to_end_ofile(FILE *);
void stack_list(void);
void *msbrk(intptr_t);
int msbrk_init(void);
int msbrk_end(void);
