#include "sys.c"

void
gcl_init_init()
{

  build_symbol_table();

#if defined(pre_gcl) || defined(ansi_gcl)
  {
    object features=find_symbol(make_simple_string("*FEATURES*"),system_package);
#if defined(pre_gcl)
    {
      extern int in_pre_gcl;
      features->s.s_dbind=make_cons(make_keyword("PRE-GCL"),features->s.s_dbind);
      in_pre_gcl=1;
    }
#else
    features->s.s_dbind=make_cons(make_keyword("ANSI-CL"),make_cons(make_keyword("COMMON-LISP"),features->s.s_dbind));
#endif
  }
#endif
  
  lsp_init("lsp","gcl_export");

  lsp_init("lsp","gcl_defmacro");/*Just for defvar in top*/
  lsp_init("lsp","gcl_evalmacros");

  lsp_init("lsp","gcl_top");

  lsp_init("lsp","gcl_autoload");

}

void
gcl_init_system(object no_init)
{

  if (type_of(no_init)!=t_symbol)
    error("Supplied no_init is not of type symbol\n");

  check_init(lsp,gcl_s,no_init);
  check_init(lsp,gcl_sf,no_init);
  check_init(lsp,gcl_rm,no_init);
  check_init(lsp,gcl_dl,no_init);
  check_init(lsp,gcl_fle,no_init);
  check_init(lsp,gcl_defmacro,no_init);
  check_init(lsp,gcl_hash,no_init);
  check_init(lsp,gcl_evalmacros,no_init);
  check_init(lsp,gcl_module,no_init);
  check_init(lsp,gcl_predlib,no_init);
  check_init(lsp,gcl_deftype,no_init);
  check_init(lsp,gcl_typeof,no_init);
  check_init(lsp,gcl_subtypep,no_init);
  check_init(lsp,gcl_bit,no_init);
#ifndef pre_gcl
  check_init(lsp,gcl_bnum,no_init);
#endif
#ifdef pre_gcl/*FIXME coerce in compiled funcall*/
  check_init(lsp,gcl_type,no_init);
  check_init(lsp,gcl_typecase,no_init);
#endif
  check_init(lsp,gcl_typep,no_init);
#ifndef pre_gcl
  check_init(lsp,gcl_type,no_init);
  check_init(lsp,gcl_typecase,no_init);
#endif

#ifndef pre_gcl
  check_init(lsp,gcl_c,no_init);
  check_init(lsp,gcl_listlib,no_init);
#else
  check_init(lsp,gcl_defseq,no_init);
#endif

  check_init(lsp,gcl_top,no_init);
  lsp_init("lsp","gcl_module");
  check_init(lsp,gcl_setf,no_init);
  check_init(lsp,gcl_arraylib,no_init);

  check_init(lsp,gcl_seq,no_init);
  check_init(lsp,gcl_seqlib,no_init);
#ifndef pre_gcl
  check_init(lsp,gcl_sc,no_init);
#endif
  check_init(lsp,gcl_assert,no_init);
  check_init(lsp,gcl_defstruct,no_init);
  check_init(lsp,gcl_restart,no_init);
  check_init(lsp,gcl_serror,no_init);
  check_init(lsp,gcl_sharp,no_init);

  check_init(lsp,gcl_logical_pathname_translations,no_init);
  check_init(lsp,gcl_make_pathname,no_init);
  check_init(lsp,gcl_parse_namestring,no_init);
  check_init(lsp,gcl_merge_pathnames,no_init);
  check_init(lsp,gcl_pathname_match_p,no_init);
  check_init(lsp,gcl_namestring,no_init);
  check_init(lsp,gcl_wild_pathname_p,no_init);
  check_init(lsp,gcl_translate_pathname,no_init);
  check_init(lsp,gcl_truename,no_init);
  check_init(lsp,gcl_directory,no_init);
  check_init(lsp,gcl_rename_file,no_init);
  
  check_init(lsp,gcl_callhash,no_init);
  check_init(lsp,gcl_describe,no_init);
#ifdef pre_gcl
  check_init(lsp,gcl_bnum,no_init);
#endif
#ifndef pre_gcl
  check_init(lsp,gcl_mnum,no_init);
#endif
  check_init(lsp,gcl_numlib,no_init);
  check_init(lsp,gcl_mislib,no_init);
  check_init(lsp,gcl_iolib,no_init);
  check_init(lsp,gcl_nr,no_init);
#ifndef pre_gcl
  check_init(lsp,gcl_lr,no_init);
  check_init(lsp,gcl_sym,no_init);
#endif
  check_init(lsp,gcl_trace,no_init);
  check_init(lsp,gcl_sloop,no_init);
  check_init(lsp,gcl_packlib,no_init);
  check_init(lsp,gcl_fpe,no_init);
	
  check_init(cmpnew,gcl_cmptype,no_init);
  check_init(cmpnew,gcl_cmpinline,no_init);
  check_init(cmpnew,gcl_cmputil,no_init);

  check_init(lsp,gcl_debug,no_init);
  check_init(lsp,gcl_info,no_init);

  check_init(cmpnew,gcl_cmpbind,no_init);
  check_init(cmpnew,gcl_cmpblock,no_init);
  check_init(cmpnew,gcl_cmptop,no_init);
  check_init(cmpnew,gcl_cmpvar,no_init);
  check_init(cmpnew,gcl_cmpeval,no_init);
  check_init(cmpnew,gcl_cmpcall,no_init);
  check_init(cmpnew,gcl_cmpcatch,no_init);
  check_init(cmpnew,gcl_cmpenv,no_init);
  check_init(cmpnew,gcl_cmpflet,no_init);
  check_init(cmpnew,gcl_cmpfun,no_init);
  check_init(cmpnew,gcl_cmptag,no_init);
  check_init(cmpnew,gcl_cmpif,no_init);
  check_init(cmpnew,gcl_cmplabel,no_init);
  check_init(cmpnew,gcl_cmploc,no_init);
  check_init(cmpnew,gcl_cmpmap,no_init);
  check_init(cmpnew,gcl_cmpmulti,no_init);
  check_init(cmpnew,gcl_cmpspecial,no_init);
  check_init(cmpnew,gcl_cmplam,no_init);
  check_init(cmpnew,gcl_cmplet,no_init);
  check_init(cmpnew,gcl_cmpvs,no_init);
  check_init(cmpnew,gcl_cmpwt,no_init);
  check_init(cmpnew,gcl_cmpmain,no_init);

#ifndef pre_gcl  

#ifndef gcl

#ifdef HAVE_XGCL
  lsp_init("xgcl-2","sysdef.lisp");
  check_init(xgcl-2,gcl_Xlib,no_init);
  check_init(xgcl-2,gcl_Xutil,no_init);
  check_init(xgcl-2,gcl_X,no_init);
  check_init(xgcl-2,gcl_XAtom,no_init);
  check_init(xgcl-2,gcl_defentry_events,no_init);
  check_init(xgcl-2,gcl_Xstruct,no_init);
  check_init(xgcl-2,gcl_XStruct_l_3,no_init);
  check_init(xgcl-2,gcl_general,no_init);
  check_init(xgcl-2,gcl_keysymdef,no_init);
  check_init(xgcl-2,gcl_X10,no_init);
  check_init(xgcl-2,gcl_Xinit,no_init);
  check_init(xgcl-2,gcl_dwtrans,no_init);
  check_init(xgcl-2,gcl_tohtml,no_init);
  check_init(xgcl-2,gcl_index,no_init);
#endif
  
  check_init(mod,gcl_ansi_io,no_init);
  check_init(mod,gcl_destructuring_bind,no_init);
  check_init(mod,gcl_loop,no_init);
  check_init(mod,gcl_defpackage,no_init);
  check_init(mod,gcl_make_defpackage,no_init);


#ifndef mod_gcl
  lsp_init("pcl","package.lisp");
  check_init(pcl,gcl_pcl_pkg,no_init);
  check_init(pcl,gcl_pcl_walk,no_init);
  check_init(pcl,gcl_pcl_iterate,no_init);
  check_init(pcl,gcl_pcl_macros,no_init);
  check_init(pcl,gcl_pcl_low,no_init);
  check_init(pcl,gcl_pcl_impl_low,no_init);
  check_init(pcl,gcl_pcl_fin,no_init);
  check_init(pcl,gcl_pcl_defclass,no_init);
  check_init(pcl,gcl_pcl_defs,no_init);
  check_init(pcl,gcl_pcl_fngen,no_init);
  check_init(pcl,gcl_pcl_cache,no_init);
  check_init(pcl,gcl_pcl_dlisp,no_init);
  check_init(pcl,gcl_pcl_dlisp2,no_init);
  check_init(pcl,gcl_pcl_boot,no_init);
  check_init(pcl,gcl_pcl_vector,no_init);
  check_init(pcl,gcl_pcl_slots_boot,no_init);
  check_init(pcl,gcl_pcl_combin,no_init);
  check_init(pcl,gcl_pcl_dfun,no_init);
  check_init(pcl,gcl_pcl_fast_init,no_init);
  check_init(pcl,gcl_pcl_precom1,no_init);
  check_init(pcl,gcl_pcl_precom2,no_init);
  check_init(pcl,gcl_pcl_braid,no_init);
  check_init(pcl,gcl_pcl_generic_functions,no_init);
  check_init(pcl,gcl_pcl_slots,no_init);
  check_init(pcl,gcl_pcl_init,no_init);
  check_init(pcl,gcl_pcl_std_class,no_init);
  check_init(pcl,gcl_pcl_cpl,no_init);
  check_init(pcl,gcl_pcl_fsc,no_init);
  check_init(pcl,gcl_pcl_methods,no_init);
  check_init(pcl,gcl_pcl_fixup,no_init);
  check_init(pcl,gcl_pcl_defcombin,no_init);
  check_init(pcl,gcl_pcl_ctypes,no_init);
  check_init(pcl,gcl_pcl_env,no_init);
  check_init(pcl,gcl_pcl_compat,no_init);

#ifndef pcl_gcl
  lsp_init("clcs","package.lisp");
  check_init(clcs,gcl_clcs_precom,no_init);
  check_init(clcs,gcl_clcs_conditions,no_init);
  check_init(clcs,gcl_clcs_condition_definitions,no_init);
#endif
#endif
#endif
#endif
  
}
