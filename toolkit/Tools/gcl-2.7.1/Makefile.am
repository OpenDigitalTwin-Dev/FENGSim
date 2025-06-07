export C_INCLUDE_PATH=$(srcdir)/h:$(srcdir)/gcl-tk

AM_CPPFLAGS=$(BASE_CPPFLAGS)
AM_CFLAGS=$(BASE_CFLAGS)
AM_LDFLAGS=$(BASE_LDFLAGS)
AUTOMAKE_OPTIONS=subdir-objects info-in-builddir


my_libdir=$(libdir)/$(PACKAGE)-$(PACKAGE_VERSION)
my_unixportdir=$(my_libdir)/unixport
my_gcltkdir=$(my_libdir)/gcl-tk
my_xgcl2dir=$(my_libdir)/xgcl-2
my_pcldir=$(my_libdir)/pcl
my_lspdir=$(my_libdir)/lsp
my_hdir=$(my_libdir)/h
my_cmpnewdir=$(my_libdir)/cmpnew
my_clcsdir=$(my_libdir)/clcs

docdir=$(datarootdir)/doc/$(PACKAGE)
xgcl2docdir=$(docdir)/xgcl-2
gcltkdocdir=$(docdir)/gcl-tk

# primaries

noinst_PROGRAMS=bin/dpp o/grab_defs
noinst_LIBRARIES=lib/libbase_gcl.a

my_unixport_PROGRAMS=unixport/saved_gcl unixport/saved_ansi_gcl
my_unixport_LIBRARIES=unixport/libgcl.a unixport/libansi_gcl.a
bin_SCRIPTS=bin/gcl
check_SCRIPTS=sb_ansi-tests/test_results sb_bench/timing_results
info_TEXINFOS=info/gcl.texi info/gcl-si.texi
my_unixport_DATA=$(addprefix unixport/,init_raw.lsp gcl.script libboot.so \
				       gcl_cmpnopt_gcl.lsp gcl_cmpnopt_ansi_gcl.lsp)
my_h_DATA=h/cmpinclude.h
my_cmpnew_DATA=sb_cmpnew/gcl_collectfn.o cmpnew/gcl_lfun_list.lsp cmpnew/gcl_cmpopt.lsp
my_pcl_DATA=pcl/package.lisp
my_lsp_DATA=lsp/gcl_auto_new.lsp lsp/gcl_autoload.lsp lsp/gcl_defmacro.lsp \
	    lsp/gcl_evalmacros.lsp lsp/gcl_export.lsp lsp/gcl_module.lsp lsp/gcl_top.lsp
my_clcs_DATA=clcs/package.lisp

dist_man1_MANS=man/man1/gcl.1
lisp_DATA=elisp/add-default.el elisp/ansi-doc.el elisp/dbl.el elisp/doc-to-texi.el \
	  elisp/gcl.el elisp/man1-to-texi.el elisp/smart-complete.el elisp/sshell.el

dist_noinst_DATA=

# conditionals

if AMM_GPROF

LIBGPROF=lib/libgprof.a
noinst_LIBRARIES+=$(LIBGPROF) lib/libbase_gcl_gprof.a
my_unixport_PROGRAMS+=unixport/saved_gcl_gprof unixport/saved_ansi_gcl_gprof
my_unixport_LIBRARIES+=unixport/libgcl_gprof.a unixport/libansi_gcl_gprof.a

endif

#it would be nice someday to sandbox xgcl-2 and gcl-tk builds in tmpdirs

if AMM_XGCL

X_LIB=lib/libxgcl.a
noinst_LIBRARIES+=$(X_LIB)

dist_xgcl2doc_DATA=xgcl-2/Xakcl.paper xgcl-2/README
dist_xgcl2doc_DATA+=$(addprefix xgcl-2/gcl_,\
			$(addsuffix .lsp,\
				menu-set pcalc draw dwindow editors ice-cream lispserver Xakcl.example))
dist_my_xgcl2_DATA=xgcl-2/sysdef.lisp \
		   $(addprefix xgcl-2/gcl_,$(addsuffix .lsp,\
			dwtest dwtestcases drawtrans editorstrans lispservertrans menu-settrans draw-gates))

X_OBJS=$(addprefix gcl_,$(addsuffix .o,\
		Xlib Xutil X XAtom defentry_events Xstruct XStruct_l_3 \
		general keysymdef X10 Xinit dwtrans tohtml index))
dist_noinst_DATA+=$(addprefix xgcl-2/,dwdoc.tex gnu.license dec.copyright version\
			$(patsubst %.o,%.lsp,$(X_OBJS)))

info_TEXINFOS+=info/gcl-dwdoc.texi

if AMM_GPROF
noinst_LIBRARIES+=lib/libxgcl_gprof.a
endif

endif

if AMM_TK

my_gcltk_PROGRAMS=gcl-tk/gcltkaux
my_gcltk_SCRIPTS=gcl-tk/gcltksrv
my_gcltk_DATA=gcl-tk/tkl.o gcl-tk/tinfo.o gcl-tk/demos/gc-monitor.o
gcltkdoc_DATA=gcl-tk/demos/index.lsp
dist_my_gcltk_DATA=gcl-tk/tk-package.lsp gcl-tk/gcl.tcl
dist_gcltkdoc_DATA=$(addprefix gcl-tk/demos/,$(addsuffix .lisp,\
			gc-monitor mkBasic mkCanvText mkdialog mkEntry2 mkEntry mkForm mkHScale  mkItems \
			mkLabel mkListbox mkPlot mkRadio mkRuler mkSearch mkStyles mkTextBind mkVScale \
			nqthm-stack showVars widget))
dist_noinst_DATA+=gcl-tk/tkl.lisp gcl-tk/tinfo.lsp
info_TEXINFOS+=info/gcl-tk.texi

endif

EXTRA_DIST=$(D_SRC) $(INCL_C) $(INIT_L) $(INCL_TEXI)\
	   lsp cmpnew mod pcl clcs ansi-tests elisp xbin bench \
	   majvers minvers git.tag release

# end of primaries

INCL_TEXI=$(addprefix info/,$(addsuffix .texi,\
   bind c-interface chap-1 chap-10 chap-11 chap-12 chap-13 chap-14 chap-15 chap-16 chap-17 chap-18 chap-19 \
   chap-2 chap-20 chap-21 chap-22 chap-23 chap-24 chap-25 chap-26 chap-3 chap-4 chap-5 chap-6 chap-7 chap-8 \
   chap-9 chap-a character compile compiler-defs control debug doc form \
   general internal io iteration japi list misc number sequence si-defs structure symbol system type \
   user-interface widgets))

INCL_C=o/sgbc.c o/gmp_big.c o/gmp.c o/gmp_num_log.c o/cmac.c o/regexp.c o/unexelf.c o/unexmacosx.c o/unexnt.c \
       o/save.c o/xdrfuns.c o/fasdump.c o/usig2_aux.c o/sfaslelf.c o/sfaslmacho.c o/sfaslcoff.c gcl-tk/comm.c \
       o/sfasli.c o/firstfile.c o/lastfile.c
INIT_L=unixport/sys_init.c unixport/sys.c o/boot.c unixport/cinit.lisp
BASE_H= h/compbas2.h h/compbas.h h/compprotos.h h/cstack.h h/enum.h h/error.h h/eval.h h/fixnum.h h/frame.h \
	h/funlink.h h/globals.h h/gmp_wrappers.h h/immnum.h h/include.h h/lex.h h/linux.h h/lu.h h/make-init.h \
	h/mp.h h/notcomp.h h/num_include.h h/object.h h/options.h h/page.h h/pageinfo.h h/pbits.h h/pool.h \
	h/prelink.h h/protoize.h h/ptable.h h/rgbc.h h/sfun_argd.h h/stacks.h h/type.h h/usig.h h/vs.h \
        h/writable.h o/regexp.h h/arth.h h/bsd.h h/bds.h h/att_ext.h h/bfdef.h h/compat.h h/apply_n.h \
	gcl-tk/sheader.h h/make-decl.h h/defun.h o/ntheap.h
CMPI_H=	h/compdefs.h h/cmpincl1.h h/mgmp.h h/compprotos.h h/compbas2.h h/cmponly_last.h
ARCHT_H=h/elf32_armhf_reloc.h h/elf32_armhf_reloc_special.h h/elf32_arm_reloc.h h/elf32_arm_reloc_special.h \
	h/elf32_hppa_reloc.h h/elf32_hppa_reloc_special.h h/elf32_i386_reloc.h h/elf32_m68k_reloc.h \
	h/elf32_mips_reloc.h h/elf32_mips_reloc_special.h h/elf32_ppc_reloc.h h/elf32_s390_reloc.h \
	h/elf32_sh4_reloc.h h/elf32_sparc_reloc.h h/elf64_aarch64_reloc.h h/elf64_aarch64_reloc_special.h \
	h/elf64_alpha_reloc.h h/elf64_alpha_reloc_special.h h/elf64_i386_reloc.h h/elf64_i386_reloc_special.h \
	h/elf64_loongarch64_reloc.h h/elf64_loongarch64_reloc_special.h h/elf64_mips_reloc.h \
	h/elf64_mips_reloc_special.h h/elf64_ppcle_reloc.h h/elf64_ppcle_reloc_special.h h/elf64_ppc_reloc.h \
	h/elf64_ppc_reloc_special.h h/elf64_riscv64_reloc.h h/elf64_s390_reloc.h h/elf64_sparc_reloc.h \
	h/elf64_sparc_reloc_special.h h/mach32_i386_reloc.h h/mach32_ppc_reloc.h h/mach64_i386_reloc.h \
	h/sh4-linux.h h/amd64-linux.h h/amd64-kfreebsd.h h/386-linux.h h/riscv64-linux.h \
	h/386-kfreebsd.h h/amd64-gnu.h h/386-gnu.h h/m68k-linux.h h/alpha-linux.h h/mips-linux.h \
	h/mipsel-linux.h h/sparc-linux.h h/aarch64-linux.h h/armhf-linux.h h/arm-linux.h h/s390-linux.h \
	h/ia64-linux.h h/hppa-linux.h h/loongarch64-linux.h h/powerpc-linux.h h/powerpc-macosx.h \
	h/386-macosx.h h/mingw.h h/gnuwin95.h h/FreeBSD.h h/solaris.h h/solaris-i386.h
BUILT_H=h/new_decl.h o/boot.h h/cmpinclude.h

C_SRC=o/typespec.c o/alloc.c o/gbc.c o/bitop.c o/main.c o/eval.c o/macros.c o/lex.c o/bds.c o/frame.c\
      o/predicate.c o/reference.c o/assignment.c o/bind.c o/let.c o/conditional.c o/block.c o/iteration.c\
      o/prog.c o/multival.c o/catch.c o/cfun.c o/cmpaux.c o/big.c o/number.c o/num_pred.c o/num_comp.c\
      o/num_arith.c o/num_sfun.c o/num_co.c o/num_log.c o/num_rand.c o/earith.c\
      o/array.c o/regexpr.c o/structure.c o/toplevel.c o/backq.c o/format.c\
      o/unixfsys.c o/unixfasl.c o/error.c o/unixtime.c o/unixsys.c o/unixsave.c o/funlink.c  o/fat_string.c\
      o/run_process.c o/nfunlink.c o/usig.c o/usig2.c o/utils.c o/makefun.c o/sockets.c o/gmp_wrappers.c\
      o/clxsocket.c o/nsocket.c o/prelink.c o/sfasl.c o/msbrk.c \
      o/bcmp.c o/bcopy.c o/bzero.c o/user_init.c o/user_match.c o/mapfun.c

D_SRC=o/character.d o/file.d o/gcl_readline.d o/hash.d o/list.d o/package.d o/pathname.d o/print.d\
      o/read.d o/sequence.d o/string.d o/symbol.d
BUILT_C=o/character.c o/file.c o/gcl_readline.c o/hash.c o/list.c o/package.c o/pathname.c o/print.c\
        o/read.c o/sequence.c o/string.c o/symbol.c o/new_init.c

INI_FILES=$(patsubst %.c,%.ini,$(C_SRC)) $(patsubst %.d,%.ini,$(D_SRC))
if AMM_GPROF
INI_FILES+=o/gprof.ini
endif

BUILT_SOURCES=$(BUILT_H) $(BUILT_C)
CLEANFILES=$(BUILT_SOURCES) $(INI_FILES) o/boot.ini

lib_libbase_gcl_a_SOURCES=$(C_SRC) $(BASE_H) $(CMPI_H) $(ARCHT_H)
nodist_lib_libbase_gcl_a_SOURCES=$(BUILT_C)
lib_libgprof_a_SOURCES=o/gprof.c
lib_libgprof_a_CFLAGS=$(AM_CFLAGS) -fno-omit-frame-pointer -pg
lib_libbase_gcl_gprof_a_SOURCES=$(lib_libbase_gcl_a_SOURCES) o/gprof.c
nodist_lib_libbase_gcl_gprof_a_SOURCES=$(nodist_lib_libbase_gcl_a_SOURCES)
lib_libbase_gcl_gprof_a_CFLAGS=$(AM_CFLAGS) -fno-omit-frame-pointer -pg
lib_libbase_gcl_gprof_a_CPPFLAGS=$(AM_CPPFLAGS) -DGCL_GPROF
X_SRC=xgcl-2/Events.c xgcl-2/general-c.c xgcl-2/XStruct-2.c xgcl-2/XStruct-4.c xgcl-2/Xutil-2.c
lib_libxgcl_a_SOURCES=$(X_SRC)
lib_libxgcl_gprof_a_SOURCES=$(X_SRC)
lib_libxgcl_gprof_a_CFLAGS=$(AM_CFLAGS) -fno-omit-frame-pointer -pg

unixport_saved_gcl_SOURCES=
unixport/saved_gcl$(EXEEXT):
unixport_saved_ansi_gcl_SOURCES=
unixport/saved_ansi_gcl$(EXEEXT):
unixport_saved_gcl_gprof_SOURCES=
unixport/saved_gcl_gprof$(EXEEXT):
unixport_saved_ansi_gcl_gprof_SOURCES=
unixport/saved_ansi_gcl_gprof$(EXEEXT):
unixport_libgcl_a_SOURCES=
unixport/libgcl.a:
unixport_libansi_gcl_a_SOURCES=
unixport/libansi_gcl.a:
unixport_libgcl_gprof_a_SOURCES=
unixport/libgcl_gprof.a:
unixport_libansi_gcl_gprof_a_SOURCES=
unixport/libansi_gcl_gprof.a:

GCLTK_SRC=gcl-tk/guis.c gcl-tk/tkAppInit.c gcl-tk/tkMain.c gcl-tk/guis.h
gcl_tk_gcltkaux_SOURCES=$(GCLTK_SRC)
gcl_tk_gcltkaux_CPPFLAGS=$(AM_CPPFLAGS) $(TK_INCLUDE) $(TCL_INCLUDE) $(TK_XINCLUDES)
gcl_tk_gcltkaux_LDADD=$(TK_LIB_SPEC) $(TCL_LIB_SPEC)

MY_DIRS=gcl0 gcl1 gcl2 gcl3 gcl mod_gcl0 mod_gcl pcl_gcl ansi_gcl

L_STUBS=s sf c listlib predlib deftype typeof subtypep bit type typep typecase arraylib \
        seq seqlib bnum fle dl rm nr lr sym hash sharp \
	cmptype cmpeval cmpvar cmpwt cmpif \
	cmplet cmptag cmpinline cmpenv cmplam cmptop cmpbind cmpblock cmpcall cmpcatch \
	cmpflet cmpfun cmplabel cmploc cmpmap cmpmulti cmpspecial cmputil cmpvs cmpmain \
	callhash assert defmacro defstruct \
	describe evalmacros sc logical_pathname_translations make_pathname parse_namestring \
	merge_pathnames pathname_match_p namestring wild_pathname_p translate_pathname \
	truename directory rename_file restart iolib mislib module numlib packlib \
	setf top trace sloop debug info serror mnum fpe
L_OBJS=$(addprefix gcl_,$(addsuffix .o,$(L_STUBS)))
LC_OBJS=$(filter gcl_cmp%,$(L_OBJS))
LL_OBJS=$(filter-out gcl_cmp%,$(L_OBJS))
L_FOBJS=$(addprefix gcl/,$(L_OBJS))

MOD_STUBS=ansi_io defpackage destructuring_bind loop make_defpackage
MMOD_OBJS=$(addprefix gcl_,$(addsuffix .o,$(MOD_STUBS)))
MOD_OBJS=$(MMOD_OBJS) $(X_OBJS)
MOD_FOBJS=$(addprefix mod_gcl/,$(MOD_OBJS)) $(L_FOBJS) $(X_LIB)

PCL_STUBS=boot braid cache combin compat cpl ctypes defclass defcombin defs dfun dlisp2\
	  dlisp env fast_init fin fixup fngen fsc generic_functions impl_low init iterate\
	  low macros methods pkg precom1 precom2 slots_boot slots std_class vector walk
PCL_OBJS=$(addprefix gcl_pcl_,$(addsuffix .o,$(PCL_STUBS)))
PCL_FOBJS=$(addprefix pcl_gcl/,$(PCL_OBJS)) $(MOD_FOBJS)

CLCS_STUBS=condition_definitions conditions precom
CLCS_OBJS=$(addprefix gcl_clcs_,$(addsuffix .o,$(CLCS_STUBS)))
ANSI_FOBJS=$(addprefix ansi_gcl/,$(CLCS_OBJS)) $(PCL_FOBJS)

CLEANFILES+=$(addprefix unixport/raw_,gcl gcl_gprof ansi_gcl ansi_gcl_gprof)
$(addprefix unixport/saved_,gcl ansi_gcl gcl_gprof ansi_gcl_gprof):\
unixport/saved_%: unixport/raw_% # rebuild these only when out of date

unixport/saved_%: | unixport/raw_% unixport/gcl_cmpnopt_%.lsp \
		    unixport/libboot.so unixport/init_raw.lsp
	rm -rf sb_$* # FIXME sandbox ugliness for parallel builds
	mkdir sb_$*
	cd sb_$* && \
		ar x ../unixport/lib$*.a $$(ar t ../unixport/lib$*.a |grep ^gcl_) && \
		ln -snf gcl_cmpnopt_$*.lsp ../unixport/gcl_cmpnopt.lsp && \
		mkdir h && \
		ln -snf ../../h/cmpinclude.h h/ && \
		GCL_LSPSYSDIR=../$(srcdir)/unixport/ \
			../unixport/raw_$* $$(dirname $$(pwd))/unixport/ -libdir $$(dirname $$(pwd))/ \
			    < <(cat ../unixport/init_raw.lsp <(echo "(system:save-system \"../$@\")")) && \
		rm -f ../unixport/gcl_cmpnopt.lsp
	rm -rf sb_$*

unixport/raw_%: unixport/lib%.a
	$(CC) $(AM_LDFLAGS) -rdynamic -Wl,-z,relro $(LDFLAGS) -o $@ $< $(LIBS) #FIXME relro

unixport/gcl_cmpnopt_gcl_gprof.lsp unixport/gcl_cmpnopt_ansi_gcl_gprof.lsp:\
unixport/gcl_cmpnopt_%_gprof.lsp: unixport/gcl_cmpnopt_%.lsp
	ln -snf $$(basename $<) $@
unixport/gcl_cmpnopt_pre_gcl.lsp: # FIXME necessary?
	touch $@
unixport/gcl_cmpnopt_%.lsp: unixport/lib%.a | unixport/%
	echo "(mapc (quote load) (directory \"$*/*.hsh\"))" \
	     "(compiler::dump-inl-hash \"$@\")" | $|

$(addprefix unixport/lib,$(addsuffix .a,pre_gcl $(MY_DIRS))): \
unixport/lib%.a: lib/libbase_gcl.a $(LIBGPROF) unixport/sys_%.o
$(addprefix unixport/lib,$(addsuffix .a,gcl_gprof ansi_gcl_gprof)): \
unixport/lib%.a:lib/libbase_gcl_gprof.a unixport/sys_%.o

.POSIX: # parallel job ordering
E0=s typep nr deftype cmptype cmpinline cmpflet
L0=c listlib seqlib type evalmacros
E0_OBJS=$(addprefix gcl_,$(addsuffix .o,$(E0)))
M0_OBJS=$(filter-out $(E0_OBJS) $(L0_OBJS),$(L_OBJS))
L0_OBJS=$(addprefix gcl_,$(addsuffix .o,$(L0)))
unixport/libgcl0.a: $(addprefix gcl0/,$(E0_OBJS) $(M0_OBJS) $(L0_OBJS))

$(addprefix unixport/lib,$(addsuffix .a,gcl1 gcl2 gcl3 gcl)):\
unixport/lib%.a: $(addprefix %/,$(L_OBJS))
unixport/libmod_gcl0.a unixport/libmod_gcl.a:\
unixport/lib%.a: $(addprefix %/,$(MOD_OBJS)) $(L_FOBJS) $(X_LIB)
unixport/libgcl.a unixport/libmod_gcl.a: unixport/lib%.a: %/recompile
unixport/libpcl_gcl.a: $(PCL_FOBJS)
unixport/libansi_gcl.a: $(ANSI_FOBJS)

unixport/libgcl_gprof.a: $(patsubst %.o,%.go,$(L_FOBJS))
unixport/libansi_gcl_gprof.a: $(patsubst %.o,%.go,$(ANSI_FOBJS))

unixport/lib%.a: | xbin/ar_merge
	$| $(ARFLAGS)s $@ $^

%/recompile: | unixport/%
	$| -batch \
	   -eval "(let ((si::*do-recomp-output-dir* \"$(@D)\")) (si::do-recomp t))" \
	   -eval "(compiler::dump-inl-hash \"$(@D)/all.hsh\")"
	touch $@

unixport/sys_%.o: unixport/sys_init.c
	i=$$(echo $* | sed 's,[0-9],,g' | sed 's,_gprof,,g'); \
	   $(CC) $(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS) -D $$i -D FLAVOR="$$i" -c $< -o $@

unixport/gcl0: | unixport/saved_pre_gcl
unixport/gcl1: | unixport/saved_gcl0
unixport/gcl2: | unixport/saved_gcl1
unixport/gcl3: | unixport/saved_gcl2
unixport/gcl:  | unixport/saved_gcl3
unixport/mod_gcl0:  | unixport/saved_gcl
unixport/mod_gcl:  | unixport/saved_mod_gcl0
unixport/pcl_gcl:  | unixport/saved_mod_gcl
unixport/ansi_gcl:  | unixport/saved_pcl_gcl
$(addprefix unixport/,gcl0 gcl1): unixport/% : unixport/cinit.lisp | %
	$(word 2,$|) < <(cat $< <(echo "(system:save-system \"$@\")"))
$(addprefix unixport/,gcl2 gcl3 gcl mod_gcl): unixport/% : | %
	ln -snf $$(basename $(word 2,$|)) $@

unixport/mod_gcl0: xgcl-2/sysdef.lisp | mod_gcl0
	rm -f $(@D)/in
	[ "$(X_OBJS)" = "" ] || echo "(load \"$<\")" >$(@D)/in
	echo "(compiler::cdebug)(si::save-system \"$@\")" >>$(@D)/in
	$(word 2,$|) <$(@D)/in
	rm $(@D)/in
	cp gcl/all.hsh $(word 1,$|) #FIXME

unixport/pcl_gcl: clcs/package.lisp cmpnew/gcl_collectfn.lsp pcl/defsys.lisp  | pcl_gcl
	echo  "(let ((*features* (remove :kcl *features*))) (mapc (quote load) (list $(patsubst %,\"%\",$^))))" \
	      "(compiler::cdebug)" \
	      "(setq compiler::*assert-ftype-proclamations* t)" \
	      "(setq pcl::*pcl-directory* (cons \"$$(dirname $(word 3,$^))/\" \"$(@F)/\"))" \
	      "(print pcl::*pcl-directory*)" \
	      "(setq pcl::*default-pathname-extensions* (cons \"lisp\" \"o\"))" \
	      "(setq pcl::*pathname-extensions* (cons \"lisp\" \"o\"))" \
	      "(si::save-system \"$@\")" | $(word 2,$|)

unixport/ansi_gcl: clcs/package.lisp clcs/gcl_clcs_precom.lisp \
		   clcs/gcl_clcs_conditions.lisp clcs/gcl_clcs_condition_definitions.lisp \
		   | ansi_gcl
	echo "(mapc (quote load) (list $(patsubst %,\"%\",$^)))" \
             "(compiler::cdebug)" \
             "(si::save-system \"$@\")" | $(word 2,$|)
	cp pcl_gcl/all.hsh $(word 1,$|) #FIXME

$(addprefix gcl0/,$(LL_OBJS)): gcl0/%.o : lsp/%.lsp
$(addprefix gcl0/,$(LC_OBJS)): gcl0/%.o : cmpnew/%.lsp
gcl0/%.o: | unixport/gcl0
	$| -eval "(mapc 'load (directory \"$(@D)/*.done\"))" -compile $< -o $@
	[ "$*" = "gcl_c" ] || [ "$*" = "gcl_listlib" ] || \
		ln -f $@ $$(echo $@ |sed 's,\.o,\.done,g') #FIXME directory link

$(addprefix gcl1/,$(LL_OBJS)): gcl1/%.o : lsp/%.lsp
$(addprefix gcl1/,$(LC_OBJS)): gcl1/%.o : cmpnew/%.lsp
gcl1/%.o: | unixport/gcl1
	$| -eval "(mapc 'load (directory \"$(@D)/*.done\"))" \
	   -eval "(setq compiler::*dump-inl-hash* t)" \
	   -compile $< -o $@
	[ "$*" = "gcl_sym" ] || ln -f $@ $$(echo $@ |sed 's,\.o,\.done,g') # FIXME listlib must come before sym

$(addprefix gcl2/,$(LL_OBJS)): gcl2/%.o : lsp/%.lsp
$(addprefix gcl2/,$(LC_OBJS)): gcl2/%.o : cmpnew/%.lsp
gcl2/%.o: | unixport/gcl2
	$| -eval "(compiler::cdebug)(setq compiler::*dump-inl-hash* t)" -compile $< -o $@

$(addprefix gcl3/,$(LL_OBJS)): gcl3/%.o : lsp/%.lsp
$(addprefix gcl3/,$(LC_OBJS)): gcl3/%.o : cmpnew/%.lsp
gcl3/%.o: | unixport/gcl3
	$| -eval "(compiler::cdebug)(setq compiler::*dump-inl-hash* t)" -compile $< -o $@

gcl/%.o: gcl3/%.o | gcl
	cp $(patsubst %.o,%.*,$<) $(@D)

$(addprefix mod_gcl0/,$(MMOD_OBJS)): mod_gcl0/%.o : mod/%.lsp
$(addprefix mod_gcl0/,$(X_OBJS)):    mod_gcl0/%.o : xgcl-2/%.lsp
mod_gcl0/%.o: | unixport/mod_gcl0
	$| -eval "(setq compiler::*dump-inl-hash* t)" -compile $< -o $@

mod_gcl/%.o: mod_gcl0/%.o | unixport/mod_gcl
	cp $(patsubst %.o,%.*,$<) $(@D)


%/c1: | unixport/%
	echo "(pcl::compile-pcl)" | $|
	touch $@

%/sys-package.lisp: %/c1 | unixport/%
	echo "(pcl::load-pcl)" \
	     "(compiler::get-packages-ansi \
	         (quote (:walker :iterate :pcl :slot-accessor-name)) \
	         \"$@\")" | $|

%/sys-proclaim.lisp: %/c1 | unixport/%
	echo "(pcl::load-pcl)" \
	     "(si::do-recomp2 \
		\"$@\" \
		(mapcar (quote namestring) \
		  (directory \
		     (merge-pathnames \
			\"*.*p\" \
			(make-pathname \
			  :directory (pathname-directory \
				        (si::file (quote pcl::renew-sys-files))))))))" | $|
%/p1.lisp: | unixport/%
	echo "(in-package :si)" \
	     "(export (quote %structure-name))" \
	     "(export (quote %compiled-function-name))" \
	     "(export (quote %set-compiled-function-name))" \
	     "(in-package :pcl)" >$@

%/all.hs1 : %/sys-package.lisp %/p1.lisp %/sys-proclaim.lisp | unixport/%
	rm -rf $*/*.o
	echo "(mapc (quote load) (list $(patsubst %,\"%\",$^)))" \
	     "(pcl::compile-pcl)" \
	     "(compiler::dump-inl-hash \"$@\")" | $|

%/all.hsh: %/p1.lisp %/all.hs1 | unixport/pcl_gcl
	echo "pcl conflicts:"
	echo "(pcl::load-pcl)(si::all-conflicts)" | $|
	cat $^ > $@

$(addprefix pcl_gcl/,$(PCL_OBJS)): pcl_gcl/all.hsh 
	touch $@


ansi_gcl/%.o: clcs/%.lisp | unixport/ansi_gcl
	$| -eval "(setq compiler::*dump-inl-hash* t)" -compile $< -o $@

%.go: %.o mod_gcl/recompile #FIXME parallel
	$(CC) $(AM_CPPFLAGS) -I $(<D) $(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS) \
	      -fno-omit-frame-pointer -pg -c $*.c -o $@
	cat $*.data >>$@

CMPINCLUDE_FILES=h/cmpincl1.h h/gclincl.h h/compbas.h h/type.h h/mgmp.h \
		 h/lu.h h/globals.h  h/vs.h h/bds.h h/frame.h h/lex.h \
		 h/mstdint.h h/compbas2.h h/compprotos.h  h/immnum.h

CLEANFILES+=h/cmpinclude.h h/mstdint.h h/cmpincludea.h h/mcompdefs.h

h/mstdint.h:
	echo "#include <stdint.h>" | $(CC) -E -I./h/ - | $(AWK) '/fsid/ {next} {print}' >$@

h/mcompdefs.h: h/compdefs.h h/new_decl.h
	$(AWK) 'BEGIN {print "#include \"include.h\"";print "#include \"page.h\"";print "---"} {a=$$1;gsub("\\.\\.\\.","",a);print "\"#define " $$1 "\" " a}' $< |\
	$(CC) $(AM_CPPFLAGS) $(AM_CFLAGS) -E -P -I./h/ - |\
	$(AWK) '/^\-\-\-$$/ {i=1;next} {if (!i) next} {gsub("\"","");print}' >$@

h/cmpincludea.h: $(filter-out gclincl.h,$(CMPINCLUDE_FILES)) | h/gclincl.h # FIXME!
	cat $< $| $(filter-out $<,$^) | \
		$(CC) $(AM_CPPFLAGS) $(AM_CFLAGS) -E -I./h/ - | \
		$(AWK) '/^# |^$$|^#pragma/ {next}{print}' > $@

h/cmpinclude.h: h/mcompdefs.h h/cmpincludea.h h/cmponly_last.h
	@cat $^ >new_$(@F)
	@([ -e $@ ] && cmp new_$(@F) $@) || mv -v new_$(@F) $@
	@rm -f new_$(@F)

h/new_decl.h: $(INI_FILES)
	echo '#include "make-decl.h"' > foo.c
	cat $^ |sed 's,DEFBFUN,DEFUN,g' >> foo.c
	$(CPP) $(AM_CPPFLAGS) $(CPPFLAGS) foo.c | sed -n -e '/#/d' -e '/DO_/d' -e '/[a-zA-Z;]/p' > $@
	rm -f foo.c

o/boot.h: %.h: %.ini
	echo '#include "make-init.h"' > $@
	echo 'void gcl_init_boot(void){' >> $@
	cat $< >> $@
	echo '}' >> $@

CLEANFILES+=unixport/libboot.so
unixport/libboot.so: o/boot.c o/boot.h
	$(CC) $(AM_CPPFLAGS) -Io $(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS) \
		-fPIC -shared -Wl,-undefined -Wl,dynamic_lookup $< -o $@

o/boot.ini: CPPFLAGS += -DNO_BOOT_H
# parallel builds can only have one target accessing an intermediate file
# solved with BUILT_SOURCES
o/%.ini:  o/%.c | o/grab_defs
	@$(CPP) $(AM_CPPFLAGS) -DINICOMP -DNO_DEFUN $(CPPFLAGS) $< | $| > new_$(@F)
	@([ -e $@ ] && cmp new_$(@F) $@) || mv -v new_$(@F) $@
	@rm -f new_$(@F)

o/new_init.c: $(INI_FILES)
	echo '#include "make-init.h"' > $@
	echo 'void NewInit(void){' >> $@
	cat $^ >> $@
	echo '}' >> $@
	! cat $@ | awk -F, '/DEFUN/ {print $$1,$$2}' | grep -v object || (rm $@ && false)

sb_ansi-tests/test_results: ansi-tests | unixport/saved_ansi_gcl
	[ -d $(@D) ] || (mkdir $(@D) && cp $</*.lsp $(@D)) # FIXME testsuite in separate directory, hard
	cd $(@D) && echo '(load "gclload")' | ../$| |tee $(@F)

BENCH=boyer browse ctak dderiv deriv destru destru-mod div2 fft fft-mod fprint fread frpoly \
      puzzle puzzle-mod stak tak takl tak-mod takr tprint traverse triang triang-mod

sb_bench/timing_results: $(addprefix sb_bench/,$(addsuffix .tim,$(BENCH))) \
			 | bench/gcl_tim bench/sbcl_tim bench/ecl_tim bench/clisp_tim
	echo "_            gcl_$$(date +'%m%d')" >$(@D)/tmp
	cat $^ >>$(@D)/tmp
	paste $(@D)/tmp $| | \
		$(AWK) '{if (n++) \
		    printf("%-10.10s %15.3f %15.3f %15.3f %15.3f %15.3f\n",$$1,$$2,$$4,$$6,$$8,$$10); else \
		    printf("%-10.10s %15.15s %15.15s %15.15s %15.15s %15.15s\n",$$1,$$2,$$4,$$7,$$10,$$13);}' \
		>$@
	cat $@

sb_bench:
	mkdir $@

sb_bench/fread.tim: sb_bench/fprint.tim
sb_bench/%.tim: bench/%.cl | unixport/saved_ansi_gcl sb_bench
	echo "(load \"$(<D)/make-declare.lsp\")" \
	     "(proclaim-file \"$<\")" \
	     "(compile-file \"$<\" :output-file \"$(@D)/$*.o\")" \
	     "(load \"$(<D)/test-help.lsp\")" \
	     "(do-test \"$(@D)/$*\" \"$(@D)/$*.tim\")" | $(word 1,$|)

gcl-tk/%.o: gcl-tk/%.lisp | unixport/saved_ansi_gcl
	[ -d $(@D) ] || mkdir -p $(@D)
	$(word 1,$|) -compile $< -o $@

gcl-tk/%.o: gcl-tk/%.lsp | unixport/saved_ansi_gcl
	[ -d $(@D) ] || mkdir -p $(@D)
	$(word 1,$|) -compile $< -o $@

CLEANFILES+=gcl-tk/demos/index.lsp gcl-tk/demos/gc-monitor.o
gcl-tk/demos/index.lsp: gcl-tk/tkl.o | unixport/saved_gcl
	[ -d $(@D) ] || mkdir -p $(@D)
	cd $(@D) && echo "(load \"../../$<\")(tk::get-autoloads (directory \"*.lisp\"))" | ../../$|

sb_cmpnew/gcl_collectfn.o: cmpnew/gcl_collectfn.lsp | unixport/saved_gcl
	[ -d $(@D) ] || mkdir -p $(@D)
	$(word 1,$|) -compile $< -o $@

%.c: %.d | bin/dpp
	bin/dpp $< $@

gcl-tk/demos gcl-tk $(MY_DIRS): % :
	[ -d $@ ] || mkdir -p $@

# package extension

bin/gcl.in:
	echo

bin/gcl%: bin/gcl
	sed -e "s,^EXT=.*,EXT=$*,g" $< >$@
	chmod a+x $@

man/man1/gcl%.1: man/man1/gcl.1
	sed -e 's,\(\b\)gcl\(\b\),\1gcl$*\2,g' $< > $@

info/gcl%-dwdoc.texi: info/gcl-dwdoc.texi # FIXME one rule
	sed -e 's,gcl\([:.-]\),gcl$*\1,g' $< >$@
info/gcl%-si.texi: info/gcl-si.texi
	sed -e 's,gcl\([:.-]\),gcl$*\1,g' $< >$@
info/gcl%-tk.texi: info/gcl-tk.texi
	sed -e 's,gcl\([:.-]\),gcl$*\1,g' $< >$@
info/gcl%.texi: info/gcl.texi
	sed -e 's,gcl\([:.-]\),gcl$*\1,g' $< >$@

%.info: %.texi | xbin/mktmp # FIXME parallel ugliness
	i=$$($| $@);ln -f $< $$i;$(MAKEINFO) $$i --output $@;rm -f $$i
%.pdf: %.dvi | xbin/mktmp
	i=$$($| $@);j=$$i.$$(echo $< | sed 's,.*\.,,g');ln -f $< $$j;\
		dvipdfm $$j -o $@ && rm -f $$i*
%.dvi: %.texi | xbin/mktmp
	i=$$($| $@);j=$$i.$$(echo $< | sed 's,.*\.,,g');ln -f $< $$j;\
		TEXINPUTS=$(srcdir):$$TEXINPUTS \
		tex -output-directory=$(@D) $$j|tail && mv $$i.dvi $@ && rm -f $$i*
%.html: %.texi | xbin/mktmp
	mkdir -p $@
	i=$$($| $@);ln -f $< $$i;$(MAKEINFO) --html $$i --output $@;rm -f $$i

# end package extension

install-all: install install-dvi install-pdf install-html

clean_%:
	rm -rf $* $(addprefix unixport/,$* lib$*.a saved_$* sys_$*.o gcl_cmpnopt_$*.lsp)

clean-local: $(addprefix clean_,pre_gcl $(MY_DIRS) gcl_gprof ansi_gcl_gprof)
	rm -rf sb_ansi-tests sb_cmpnew sb_bench

distclean-local:
	rm -rf gcl.script unixport/gcl.script #FIXME
	rm -rf h/config.h #FIXME

AM_ETAGSFLAGS=--regex='/[ \n\t\#\.`]*(defun[ \n\t]+\([^ \n\t]+\)/' \
	      --regex='/[ \n\t\#\.`]*(defmfun[ \n\t]+"\([^ \n\t"]+\)"[ \n\t]*)/\1/' \
	      --regex='/[ \n\t\#\.`]*(defmfun[ \n\t]+"[^ \n\t"]+"[ \n\t]+\([^ \n\t]+\)/\1/'

TAGS_FILES=lsp/*.lsp cmpnew/*.lsp mod/*.lsp pcl/*sp clcs/*sp xgcl-2/*p
