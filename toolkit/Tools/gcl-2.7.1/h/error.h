#ifndef ERROR_H
#define ERROR_H

#define Icall_error_handler(a_,b_,c_,d_...)			\
  Icall_gen_error_handler_noreturn(Cnil,null_string,a_,b_,c_,##d_)
#define Icall_continue_error_handler(a_,b_,c_,d_,e_...) \
  Icall_gen_error_handler(Ct,a_,b_,c_,d_,##e_)


extern enum type t_vtype;
extern int vtypep_fn(object);
extern void Check_type(object *,int (*)(object),object);

#define PFN(a_) INLINE int Join(a_,_fn)(object x) {return a_(x);}

PFN(integerp)
PFN(non_negative_integerp)
PFN(rationalp)
PFN(floatp)
PFN(realp)
PFN(numberp)
PFN(characterp)
PFN(symbolp)
PFN(stringp)
PFN(pathnamep)
PFN(string_symbolp)
PFN(packagep)
PFN(consp)
PFN(listp)
PFN(streamp)
PFN(pathname_string_symbolp)
PFN(pathname_string_symbol_streamp)
PFN(randomp)
PFN(hashtablep)
PFN(arrayp)
PFN(vectorp)
PFN(readtablep)
PFN(functionp)

#define TPE(a_,b_,c_) if (!(b_)(*(a_))) FEwrong_type_argument((c_),*(a_))

#define check_type(a_,b_)                               ({t_vtype=(b_);TPE(&a_,vtypep_fn,type_name(t_vtype));})
#define check_type_function(a_)                         TPE(a_,functionp_fn,sLfunction)
#define check_type_integer(a_)                          TPE(a_,integerp_fn,sLinteger)
#define check_type_non_negative_integer(a_)             TPE(a_,non_negative_integerp_fn,TSnon_negative_integer)
#define check_type_rational(a_)                         TPE(a_,rationalp_fn,sLrational)
#define check_type_float(a_)                            TPE(a_,floatp_fn,sLfloat)
#define check_type_real(a_)                             TPE(a_,realp_fn,sLreal)
#define check_type_or_rational_float(a_)                TPE(a_,realp_fn,sLreal)
#define check_type_number(a_)                           TPE(a_,numberp_fn,sLnumber)
#define check_type_stream(a_)                           TPE(a_,streamp_fn,sLstream)
#define check_type_hash_table(a_)                       TPE(a_,hashtablep_fn,sLhash_table)
#define check_type_character(a_)                        TPE(a_,characterp_fn,sLcharacter)
#define check_type_sym(a_)                              TPE(a_,symbolp_fn,sLsymbol)
#define check_type_string(a_)                           TPE(a_,stringp_fn,sLstring)
#define check_type_pathname(a_)                         TPE(a_,pathnamep_fn,sLpathname)
#define check_type_or_string_symbol(a_)                 TPE(a_,string_symbolp_fn,TSor_symbol_string)
#define check_type_or_symbol_string(a_)                 TPE(a_,string_symbolp_fn,TSor_symbol_string)
#define check_type_or_pathname_string_symbol_stream(a_) TPE(a_,pathname_string_symbol_streamp_fn,TSor_pathname_string_symbol_stream)
#define check_type_or_Pathname_string_symbol(a_)        TPE(a_,pathname_string_symbolp_fn,TSor_pathname_string_symbol)
#define check_type_package(a_)                          TPE(a_,packagep_fn,sLpackage)
#define check_type_cons(a_)                             TPE(a_,consp_fn,sLcons)
#define check_type_list(a_)                             TPE(a_,listp_fn,sLlist)
#define check_type_stream(a_)                           TPE(a_,streamp_fn,sLstream)
#define check_type_array(a_)                            TPE(a_,arrayp_fn,sLarray)
#define check_type_vector(a_)                           TPE(a_,vectorp_fn,sLvector)
#define check_type_readtable_no_default(a_)             TPE(a_,readtablep_fn,sLreadtable)
#define check_type_readtable(a_)                        ({if (*(a_)==Cnil) *(a_)=standard_readtable;TPE(a_,readtablep_fn,sLreadtable);})
#define check_type_random_state(a_)                     TPE(a_,randomp_fn,sLrandom_state)

#define stack_string(a_,b_) struct string _s={0};\
                            object a_=(object)&_s;\
                            set_type_of((a_),t_string);\
                            (a_)->st.st_self=(void *)(b_);\
                            (a_)->st.st_dim=(a_)->st.st_fillp=strlen(b_)

#define stack_fixnum(a_,b_) struct fixnum_struct _s={0};\
                            object a_;\
                            if (is_imm_fix(b_)) (a_)=make_fixnum(b_); else {\
                            (a_)=(object)&_s;\
                            set_type_of((a_),t_fixnum);\
                            (a_)->FIX.FIXVAL=(b_);}

object ihs_top_function_name(ihs_ptr h);
#define FEerror(a_,b_...)   Icall_error_handler(sLerror,null_string,\
                            4,sKformat_control,make_simple_string(a_),sKformat_arguments,list(b_))
#define CEerror(a_,b_,c_...)   Icall_continue_error_handler(make_simple_string(a_),sLerror,null_string,\
                               4,sKformat_control,make_simple_string(b_),sKformat_arguments,list(c_))

#define TYPE_ERROR(a_,b_)   Icall_error_handler(sLtype_error,null_string,\
                            4,sKdatum,(a_),sKexpected_type,(b_))
#define FEwrong_type_argument(a_,b_) TYPE_ERROR(b_,a_)
#define FEcannot_coerce(a_,b_)       TYPE_ERROR(b_,a_)
#define FEinvalid_function(a_)       TYPE_ERROR(a_,sLfunction)

#define CONTROL_ERROR(a_) Icall_error_handler(sLcontrol_error,null_string,4,sKformat_control,make_simple_string(a_),sKformat_arguments,Cnil)

#define PROGRAM_ERROR(a_,b_) Icall_error_handler(sLprogram_error,null_string,4,\
                                                 sKformat_control,make_simple_string(a_),sKformat_arguments,list(1,(b_)))
#define FEtoo_few_arguments(a_,b_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string("~S [or a callee] requires more than ~R argument~:p."),\
                      sKformat_arguments,list(2,ihs_top_function_name(ihs_top),make_fixnum((b_)-(a_))))
#define FEwrong_no_args(a_,b_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string(a_),\
                      sKformat_arguments,list(2,ihs_top_function_name(ihs_top),(b_)))
#define FEtoo_few_argumentsF(a_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string("Too few arguments."),\
                      sKformat_arguments,list(2,ihs_top_function_name(ihs_top),(a_)))

#define FEtoo_many_arguments(a_,b_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string("~S [or a callee] requires less than ~R argument~:p."),\
                      sKformat_arguments,list(2,ihs_top_function_name(ihs_top),make_fixnum((b_)-(a_))))
#define FEtoo_many_argumentsF(a_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string("Too many arguments."),\
                      sKformat_arguments,list(2,ihs_top_function_name(ihs_top),(a_)))
#define FEinvalid_macro_call() \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string("Invalid macro call to ~S."),\
                      sKformat_arguments,list(1,ihs_top_function_name(ihs_top)))
#define FEunexpected_keyword(a_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string("~S does not allow the keyword ~S."),\
                      sKformat_arguments,list(2,ihs_top_function_name(ihs_top),(a_)))
#define FEinvalid_form(a_,b_) \
  Icall_error_handler(sLprogram_error,null_string,4,\
                      sKformat_control,make_simple_string(a_),\
                      sKformat_arguments,list(1,(b_)))
#define FEinvalid_variable(a_,b_) FEinvalid_form(a_,b_)

#define PARSE_ERROR(a_)   Icall_error_handler(sLparse_error,null_string,4,\
                                              sKformat_control,make_simple_string(a_),sKformat_arguments,Cnil)
#define STREAM_ERROR(a_,b_) Icall_error_handler(sLstream_error,null_string,6,\
                                                sKstream,a_,\
                                                sKformat_control,make_simple_string(b_),sKformat_arguments,Cnil)
#define READER_ERROR(a_,b_) Icall_error_handler(sLreader_error,null_string,6,\
                                                sKstream,a_,\
                                                sKformat_control,make_simple_string(b_),sKformat_arguments,Cnil)
#define PRINT_NOT_READABLE(a_,b_) Icall_error_handler(sLprint_not_readable,null_string,6,\
                                                sKobject,a_,\
                                                sKformat_control,make_simple_string(b_),sKformat_arguments,Cnil)
#define FILE_ERROR(a_,b_) Icall_error_handler(sLfile_error,null_string,6,\
                                                sKpathname,a_,\
                                                sKformat_control,make_simple_string(b_),sKformat_arguments,Cnil)
#define END_OF_FILE(a_) Icall_error_handler(sLend_of_file,null_string,2,sKstream,a_)
#define PACKAGE_ERROR(a_,b_) Icall_error_handler(sLpackage_error,null_string,6,\
                                                 sKpackage,a_,\
                                                 sKformat_control,make_simple_string(b_),sKformat_arguments,Cnil)
#define FEpackage_error(a_,b_) PACKAGE_ERROR(a_,b_)
#define PACKAGE_CERROR(a_,b_,c_,d_...) \
                     Icall_continue_error_handler(make_simple_string(b_),\
                                                  sLpackage_error,null_string,6,\
                                                  sKpackage,a_,\
                                                  sKformat_control,make_simple_string(c_),sKformat_arguments,list(d_))
#define NEW_INPUT(a_) (a_)=Ieval1(read_object(sLAstandard_inputA->s.s_dbind))


#define CELL_ERROR(a_,b_) Icall_error_handler(sLcell_error,null_string,6,\
                                              sKname,a_,\
                                              sKformat_control,make_simple_string(b_),sKformat_arguments,Cnil)
#define UNBOUND_VARIABLE(a_) Icall_error_handler(sLunbound_variable,null_string,2,sKname,a_)
#define FEunbound_variable(a_) UNBOUND_VARIABLE(a_)

#define UNBOUND_SLOT(a_,b_) Icall_error_handler(sLunbound_slot,null_string,4,sKname,a_,sKinstance,b_)
#define UNDEFINED_FUNCTION(a_) Icall_error_handler(sLundefined_function,null_string,2,sKname,a_)
#define FEundefined_function(a_) UNDEFINED_FUNCTION(a_)

#define ARITHMETIC_ERROR(a_,b_) Icall_error_handler(sLarithmetic_error,null_string,4,sKoperation,a_,sKoperands,b_)
#define DIVISION_BY_ZERO(a_,b_) Icall_error_handler(sLdivision_by_zero,null_string,4,sKoperation,a_,sKoperands,b_)
#define FLOATING_POINT_OVERFLOW(a_,b_) Icall_error_handler(sLfloating_point_overflow,null_string,4,sKoperation,a_,sKoperands,b_)
#define FLOATING_POINT_UNDERFLOW(a_,b_) Icall_error_handler(sLfloating_point_underflow,null_string,4,sKoperation,a_,sKoperands,b_)
#define FLOATING_POINT_INEXACT(a_,b_) Icall_error_handler(sLfloating_point_inexact,null_string,4,sKoperation,a_,sKoperands,b_)
#define FLOATING_POINT_INVALID_OPERATION(a_,b_) Icall_error_handler(sLfloating_point_invalid_operation,null_string,4,sKoperation,a_,sKoperands,b_)

#define PATHNAME_ERROR(a_,b_,c_...) Icall_error_handler(sLfile_error,null_string,6,\
                                                        sKpathname,(a_),\
				   		        sKformat_control,make_simple_string(b_),\
                                                        sKformat_arguments,list(c_))
#define WILD_PATH(a_) ({object _a=(a_);PATHNAME_ERROR(_a,"File ~s is wild",1,_a);})


#define NERROR(a_)  ({object fmt=make_simple_string(a_ ": line ~a, file ~a, function ~a");\
                    {object line=make_fixnum(__LINE__);\
                    {object file=make_simple_string(__FILE__);\
                    {object function=make_simple_string(__FUNCTION__);\
                     Icall_error_handler(sKerror,fmt,3,line,file,function);}}}})

#define ASSERT(a_) do {if (!(a_)) NERROR("The assertion " #a_ " failed");} while (0)

#define gcl_abort()  ({\
   frame_ptr fr=frs_sch_catch(sSPtop_abort_tagP->s.s_dbind);\
   vs_base[0]=sSPtop_abort_tagP->s.s_dbind;\
   vs_top=vs_base+1;\
   if (fr) unwind(fr,sSPtop_abort_tagP->s.s_dbind);\
   abort();\
 })

#endif /*ERROR_H*/
