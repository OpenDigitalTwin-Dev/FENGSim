bool eql1(object,object);
bool equal1(object,object);
bool equalp1(object,object);
bool file_exists(object);
bool integer_bitp(object,object);
double big_to_double(object);
frame_ptr frs_sch_catch(object);
frame_ptr frs_sch(object);
int length(object);
int number_compare(object,object);
int number_evenp(object);
int number_minusp(object);
int number_oddp(object);
int number_plusp(object);
int number_zerop(object);
long int fixint(object);
object alloc_object(enum type);
object call_proc_cs2(object,...);
object call_proc_new(object,ufixnum,ufixnum,void **,ufixnum,object,va_list);
object call_proc_new_nval(object,ufixnum,ufixnum,void **,ufixnum,object,...);
object coerce_to_string();
object fixnum_big_shift(fixnum,fixnum);
object fixnum_times(fixnum,fixnum);
object fSgensym0(void);
object fSgensym1ig(object);
object fSgensym1s(object);
object fSinit_function(object,object,object,object, fixnum,fixnum,fixnum);
object fSsputprop(object,object,object);
object Icall_gen_error_handler(object,object,object,object,ufixnum,...);
object get(object,object,object);
object get_gcd(object,object);
object get_lcm(object,object);
object integer_count(object);
object integer_length(object);
object integer_shift(object,object);
object listA(fixnum,...);
object list(fixnum,...);
object log_op2(fixnum,object,object);
object make_complex(object, object);
object make_cons(object, object);
object make_dcomplex(dcomplex);
object make_fcomplex(fcomplex);
object make_fixnum1(long);
object make_list(fixnum);
object make_longfloat(longfloat);
object make_shortfloat(float);
object make_simple_string(const char *);
object number_abs(object);
object number_divide(object, object);
object number_dpb(object,object,object);
object number_dpf(object,object,object);
object number_ldb(object,object);
object number_ldbt(object,object);
object number_minus(object,object);
object number_negate(object);
object number_plus(object,object);
object number_signum(object);
object number_times(object,object);
object princ(object,object);
object prin1(object,object);
object print(object,object);
object read_char1(object,object);
object structure_ref(object,object,fixnum);
object structure_set(object,object,fixnum,object);
object symbol_function(object);
object symbol_name(object);
object symbol_value(object);
object terpri(object);
object vs_overflow(void);
void bds_overflow(void);
void bds_unwind(bds_ptr);
void do_init(object *);
void frs_overflow(void);
void intdivrem(object,object,fixnum,object *,object *);
void princ_char(int,object);
void princ_str(char *,object);
void sethash(object,object,object);
void setq(object,object);
void super_funcall_no_event(object);
void unwind(frame_ptr,object) NO_RETURN;
int object_to_int(object);
fixnum object_to_fixnum(object);
dcomplex object_to_dcomplex(object);
char object_to_char(object);
void not_a_symbol(object);
object number_expt(object,object);
object fLrow_major_aref(object,fixnum);
void *alloca(unsigned long);
object cmod(object);
object ctimes(object,object);
object cdifference(object,object);
object cplus(object,object);
double sqrt(double);
float sqrtf(float);
object Icall_gen_error_handler(object,object,object,object,ufixnum,...);
object Icall_gen_error_handler_noreturn(object,object,object,object,ufixnum,...) __attribute__((noreturn));
object file_stream(object);
fixnum fixnum_expt(fixnum, fixnum);
char *gcl_gets(char *,int);
int gcl_puts(const char *);
int setjmp();
int _setjmp();
char *object_to_string(object);
