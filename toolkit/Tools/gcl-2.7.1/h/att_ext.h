/*
 Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
 Copyright (C) 2024 Camm Maguire

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

*/
#ifndef COM_LENG
#define COM_LENG
#endif

/*  alloc.c  */
/* void * memalign(size_t,size_t); */

/*  array.c  */
EXTER object sLarray_dimension_limit;
EXTER object sLarray_total_size_limit;

/*  assignment.c  */



/*  backq.c  */
EXTER int backq_level;
EXTER object sLlistA;
EXTER object sLappend;
EXTER object sLnconc;


/*  bds.c  */

/*  big.c  */
EXTER  struct bignum big_fixnum1_body,big_fixnum2_body,big_fixnum3_body,big_fixnum4_body,big_fixnum5_body;
EXTER object big_fixnum1,big_fixnum2,big_fixnum3,big_fixnum4,big_fixnum5;




/* bind.c */
EXTER object ANDoptional;
EXTER object ANDrest;
EXTER object ANDkey;
EXTER object ANDallow_other_keys;
EXTER object ANDaux;
EXTER object sKallow_other_keys;

/* block.c */

/*  cfun.c  */

/*  character.d  */
EXTER object STreturn;
EXTER object STspace;
EXTER object STrubout;
EXTER object STpage;
EXTER object STtab;
EXTER object STbackspace;
EXTER object STlinefeed;
EXTER object STnewline;

/*  catch.c  */

/*  cmpaux.c  */

/*  error.c  */
EXTER object sKerror,sKparse_error,sKreader_error,sKprogram_error;
EXTER object sKwrong_type_argument;
EXTER object sKcontrol_error;
EXTER object sKcatch;
EXTER object sKprotect;
EXTER object sKcatchall;
EXTER object sKdatum;
EXTER object sKexpected_type;
EXTER object sKpackage;
EXTER object sKformat_control;
EXTER object sKformat_arguments;
EXTER object sSuniversal_error_handler;
EXTER object sSPminus_most_negative_fixnumP;

/*  eval.c  */
EXTER object sLapply;
EXTER object sLfuncall;
EXTER object siVevalhook;
EXTER object siVapplyhook;

/*  unixfasl.c  fasload.c  */

/*  file.d  */
EXTER object sKabort;
EXTER object sKappend;
EXTER object sKcreate;
EXTER object sKdefault;
EXTER object sKdirection;
EXTER object sKelement_type;
EXTER object sKif_does_not_exist;
EXTER object sKif_exists;
EXTER object sKinput;
EXTER object sKio;
EXTER object sKnew_version;
EXTER object sKoutput;
EXTER object sKoverwrite;
EXTER object sKprint;
EXTER object sKprobe;
EXTER object sKrename;
EXTER object sKrename_and_delete;
EXTER object sKset_default_pathname;
EXTER object sKsupersede;
EXTER object sKverbose;

EXTER object sLAstandard_inputA;
EXTER object sLAstandard_outputA;
EXTER object sLAerror_outputA;
EXTER object sLAquery_ioA;
EXTER object sLAdebug_ioA;
EXTER object sLAterminal_ioA;
EXTER object sLAtrace_outputA;
EXTER object terminal_io;
EXTER object standard_io;
EXTER object standard_error;

EXTER object sLAload_verboseA;
EXTER object FASL_string;

#ifdef UNIX
/*  unixfsys.c  */
#else
/*  filesystem.c  */
#endif

/*  frame.c  */

/*  gbc.c  */
EXTER bool GBC_enable;

#ifdef CAN_UNRANDOMIZE_SBRK
EXTER bool gcl_unrandomized;
#endif

/*  let.c  */

/*  lex.c  */

/*  list.d  */
EXTER object sKtest;
EXTER object sKtest_not;
EXTER object sKkey;
EXTER object sKinitial_element;
/* EXTER object sKrev; */

/*  macros.c  */
EXTER object sLAmacroexpand_hookA;
EXTER object sSdefmacroA;

/*  main.c  */
EXTER char * system_directory;
EXTER int ARGC;
EXTER char **ARGV;
#ifdef UNIX
EXTER char **ENVP;
#endif

EXTER object sSAsystem_directoryA;
#ifdef UNIX
EXTER char *kcl_self;
#endif
#if !defined(IN_MAIN) || !defined(ATT)
EXTER bool raw_image;
#endif


EXTER object sLquote;

EXTER object sLlambda;

EXTER object sSlambda_block;
EXTER object sSlambda_closure;
EXTER object sSlambda_block_closure;

EXTER object sLfunction;
EXTER object sSmacro;
EXTER object sStag;
EXTER object sLblock;


/*  mapfun.c  */

/*  multival.c  */

/*  number.c  */
EXTER object shortfloat_zero;
EXTER object longfloat_zero;
/* #define make_fixnum(a) ({fixnum _a=(a);((_a+SMALL_FIXNUM_LIMIT)&(-2*SMALL_FIXNUM_LIMIT))==0?small_fixnum(_a):make_fixnum1(_a);}) */
/*  num_pred.c  */

/*  num_comp.c  */

/*  num_arith  */

/*  num_co.c  */

/*  num_log.c  */

/*  package.d  */
EXTER object lisp_package;
EXTER object user_package;
EXTER object keyword_package;
EXTER object system_package;
EXTER object gmp_package;
EXTER object sLApackageA;
EXTER object sKinternal;
EXTER object sKexternal;
EXTER object sKinherited;
EXTER object sKnicknames;
EXTER object sKuse;
EXTER int intern_flag;
EXTER object uninterned_list;

/*  pathname.d  */
EXTER object Vdefault_pathname_defaults;
EXTER object sKwild;
EXTER object sKnewest;
EXTER object sKstart;
EXTER object sKend;
EXTER object sKjunk_allowed;
EXTER object sKhost;
EXTER object sKdevice;
EXTER object sKdirectory;
EXTER object sKname;
EXTER object sKtype;
EXTER object sKversion;
EXTER object sKdefaults;

EXTER object sKabsolute;
EXTER object sKrelative;
EXTER object sKup;


/*  print.d  */
EXTER object sKupcase;
EXTER object sKdowncase;
EXTER object sKpreserve;
EXTER object sKinvert;
EXTER object sKcapitalize;
EXTER object sKpreserve;
EXTER object sKinvert;
EXTER object sKstream;
EXTER object sKreadably;
EXTER object sKescape;
EXTER object sKpretty;
EXTER object sKcircle;
EXTER object sKbase;
EXTER object sKradix;
EXTER object sKcase;
EXTER object sKgensym;
EXTER object sKlevel;
EXTER object sKlength;
EXTER object sKarray;
EXTER object sKlinear;
EXTER object sKmiser;
EXTER object sKfill;
EXTER object sKmandatory;
EXTER object sKcurrent;
EXTER object sKblock;
EXTER object sLAprint_readablyA;
EXTER object sLAprint_escapeA;
EXTER object sLAprint_prettyA;
EXTER object sLAprint_circleA;
EXTER object sLAprint_baseA;
EXTER object sLAprint_radixA;
EXTER object sLAprint_caseA;
EXTER object sLAprint_gensymA;
EXTER object sLAprint_levelA;
EXTER object sLAprint_lengthA;
EXTER object sLAprint_arrayA;
EXTER object sSAprint_contextA;
EXTER object sSAprint_context_headA;
EXTER object sSpretty_print_format;
EXTER int  line_length;

/*  Read.d  */
EXTER object standard_readtable;
EXTER object Vreadtable;
EXTER object sLAread_default_float_formatA;
EXTER object sLAread_baseA;
EXTER object sLAread_suppressA;
EXTER object READtable;
EXTER int READdefault_float_format;
EXTER int READbase;
EXTER bool READsuppress;
EXTER bool READeval;
EXTER object siSsharp_comma;
EXTER bool escape_flag;
EXTER object delimiting_char;
EXTER bool detect_eos_flag;
/* bool in_list_flag; */
EXTER bool dot_flag;
EXTER bool preserving_whitespace_flag;
EXTER object default_dispatch_macro;
EXTER object big_register_0;
EXTER int sharp_eq_context_max;

/* fasdump.c */
EXTER object sharing_table;

/*  reference.c  */


/*  sequence.d  */

/*  structure.c  */
EXTER object sSs_data;

/*  string.d  */
EXTER int string_sign, string_boundary;

/*  symbol.d  */
EXTER object string_register;
/* EXTER object gensym_prefix; */
/* EXTER int gensym_counter; */
/* EXTER object sLgensym_counter; */
EXTER object gentemp_prefix;
EXTER int gentemp_counter;
EXTER object token;

#ifdef UNIX
/*  unixsys.c  */
#else
/*  sys.c  */
#endif

#ifdef UNIX
/*  unixtime.c  */
#else
/*  time.c  */
#endif

/*  toplevel.c  */
EXTER object sLspecial,sLdeclare;
EXTER object sSvariable_documentation;
EXTER object sSfunction_documentation;
EXTER object sSsetf_function;
#define setf_fn_form(a_) (consp(a_) && MMcar(a_)==sLsetf &&\
                          consp(MMcdr(a_)) && type_of(MMcadr(a_))==t_symbol &&\
                          MMcddr(a_)==Cnil)

/*  typespec.c  */
EXTER object sLcommon,sLnull,sLcons,sLlist,siLproper_list,sLsymbol,sLarray,sLvector,sLbit_vector,sLstring;
EXTER object sLsequence,sLsimple_array,sLsimple_vector,sLsimple_bit_vector,sLsimple_string;
EXTER object sLcompiled_function,sLpathname,sLcharacter,sLnumber,sLrational,sLfloat;
EXTER object sLinteger,sLratio,sLshort_float,sLstandard_char;

EXTER object sLchar,sLnon_negative_char,sLnegative_char,sLsigned_char,sLunsigned_char;
EXTER object sLshort,sLnon_negative_short,sLnegative_short,sLsigned_short,sLunsigned_short;
EXTER object sLfixnum,sLnon_negative_fixnum,sLnegative_fixnum,sLsigned_fixnum,sLunsigned_fixnum;
EXTER object sLlfixnum,sLnon_negative_lfixnum,sLnegative_lfixnum;
EXTER object sLsigned_lfixnum,sLunsigned_lfixnum,sLnegative_bignum,sLnon_negative_bignum,sLbase_char;

EXTER object sLsigned_int,sLnon_negative_int,sLnegative_int,sLunsigned_int;

EXTER object sLseqind,sLrnkind;

EXTER object sLcomplex;
EXTER object sLsingle_float,sLpackage,sLbignum,sLrandom_state,sLdouble_float,sLstream,sLbit,sLreadtable;
EXTER object sLlong_float,sLhash_table,sLstructure,sLboolean,sLfile_stream,sLinput_stream,sLoutput_stream,sLtype_error;
EXTER object sLbroadcast_stream,sLconcatenated_stream,sLecho_stream,sLfile_stream,sLstring_stream;
EXTER object sLsynonym_stream,sLtwo_way_stream;


EXTER object sLsatisfies;
EXTER object sLmember;
EXTER object sLnot;
EXTER object sLor;
EXTER object sLand;
EXTER object sLvalues;
EXTER object sLmod;
EXTER object sLsigned_byte;
EXTER object sLunsigned_byte;
EXTER object sSsigned_char;
EXTER object sSunsigned_char;
EXTER object sSsigned_short;
EXTER object sSunsigned_short;
EXTER object sLA;
EXTER object sLplusp;
EXTER object TSor_symbol_string;
EXTER object TSor_string_symbol;
EXTER object TSor_symbol_string_package;
EXTER object TSnon_negative_integer;
EXTER object TSpositive_number;
EXTER object TSor_integer_float;
EXTER object TSor_rational_float;
#ifdef UNIX
EXTER object TSor_pathname_string_symbol;
#endif
EXTER object TSor_pathname_string_symbol_stream;

EXTER int interrupt_flag;		/* console interupt flag */
EXTER int interrupt_enable;		/* console interupt enable */

EXTER object sSAlink_arrayA;

/* nfunlink.c */
/* object Icall_proc(); */
EXTER object sSPmemory;
EXTER object sSPinit;

/* string.d */
int  (*casefun)();
