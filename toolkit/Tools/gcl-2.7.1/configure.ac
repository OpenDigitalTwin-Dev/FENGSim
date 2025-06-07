# Copyright 2024 Camm Maguire
AC_INIT(gcl,2.7.1)
AC_PREREQ([2.0])
AC_CONFIG_HEADERS([h/gclincl.h])
AM_INIT_AUTOMAKE([1.16 -Wno-portability tar-pax])
AM_PATH_LISPDIR
AC_CONFIG_FILES([Makefile])
#AC_USE_SYSTEM_EXTENSIONS
#LT_INIT

MAJVERS=`cat $srcdir/majvers`
MINVERS=`cat $srcdir/minvers`
GIT_TAG=`cat $srcdir/git.tag`
RELEASE=`cat $srcdir/release`
VERSION=$MAJVERS.$MINVERS
AC_SUBST(VERSION)

dnl AC_ARG_ENABLE(xgcl,[ --enable-xgcl=yes will compile in support for XGCL],
dnl [enable_xgcl=$enableval],[enable_xgcl="yes"])

#
# Host information 
#

AC_CHECK_PROGS(AWK,gawk nawk awk)

AC_CANONICAL_HOST
canonical=$host
my_host_kernel=`echo $host_os | ${AWK} '{j=split($1,A,"-");print A[[1]]}'`
my_host_system=`echo $host_os | ${AWK} '{j=split($1,A,"-");if (j>=2) print A[[2]]}'`
AC_DEFINE_UNQUOTED(HOST_CPU,"`echo $host_cpu | ${AWK} '{print toupper($0)}'`",[Host cpu])
AC_DEFINE_UNQUOTED(HOST_KERNEL,"`echo $my_host_kernel | ${AWK} '{print toupper($0)}'`",[Host kernel])
if test "$my_host_system" != "" ; then 
	AC_DEFINE_UNQUOTED(HOST_SYSTEM,"`echo $my_host_system | ${AWK} '{print toupper($0)}'`",[Host system])
fi
AC_MSG_RESULT(host=$host)

#use=unknown
case $canonical in
    sh4*linux*) use=sh4-linux;;
    *x86_64*linux*) use=amd64-linux;;
    *x86_64*kfreebsd*)	use=amd64-kfreebsd;;
    *86*linux*) use=386-linux;;
    *riscv64*linux*) use=riscv64-linux;;
    *86*kfreebsd*) use=386-kfreebsd;;
    *86_64*gnu*)	use=amd64-gnu;;
    *86*gnu*)	use=386-gnu;;
    m68k*linux*) use=m68k-linux;;
    alpha*linux*) use=alpha-linux;;
    mips*linux*) use=mips-linux;;
    mipsel*linux*) use=mipsel-linux;;
    sparc*linux*) use=sparc-linux;;
    aarch64*linux*) use=aarch64-linux;;
    arm*linux*hf) use=armhf-linux;;
    arm*linux*) use=arm-linux;;
    s390*linux*) use=s390-linux;;
    ia64*linux*) use=ia64-linux;;
    hppa*linux*) use=hppa-linux;;
    loongarch64*linux*) use=loongarch64-linux;;
    powerpc*linux*) use=powerpc-linux;;
    powerpc-*-darwin*) use=powerpc-macosx;;
    *86*darwin*) use=386-macosx;;
    i*mingw*|i*msys*) use=mingw;;
    *cygwin*)
        if $CC -v 2>&1 | fgrep ming > /dev/null ;
        then use=mingw
        else use=gnuwin95
        fi;;
    *openbsd*) use=FreeBSD;;
    sparc-sun-solaris*) use=solaris;;
    i?86-pc-solaris*) use=solaris-i386;;
esac

AC_ARG_ENABLE([machine],[  --enable-machine=XXXX will force the use of one of the definitions in h/XXXX.defs],
	      [echo enable_machine=$enableval ; use=$enableval])

AC_MSG_RESULT([use=$use])

def_pic="no";
case $use in 
    *kfreebsd)
	if ! test -d h ; then mkdir h; fi;;
#	ln -snf ../$srcdir/h/linux.defs h/$use.defs;;
    *gnu)
	if ! test -d h ; then mkdir h; fi;;
#	ln -snf ../$srcdir/h/linux.defs h/$use.defs;;
    *linux)
	if ! test -d h ; then mkdir h; fi;
#	ln -snf ../$srcdir/h/linux.defs h/$use.defs;
	case $use in
	    hppa*) # FIXME
		def_pic="yes" ;;
	esac;;
esac

AC_ARG_ENABLE([widecons],[  --enable-widecons will use a three word cons with simplified typing],
	      [if test "$enableval" = "yes" ; then AC_DEFINE([WIDE_CONS],[1],[three word cons]) fi])

AC_ARG_ENABLE([safecdr],[  --enable-safecdr will protect cdr from immfix and speed up type processing],
	      [if test "$enableval" = "yes" ; then
		   AC_DEFINE([USE_SAFE_CDR],[1],[protect cdr from immfix and speed up type processing])
		   AC_ARG_ENABLE([safecdrdbg],[  --enable-safecdrdbg will debug safecdr code],
				 [if test "$enableval" = "yes" ; then AC_DEFINE([DEBUG_SAFE_CDR],[1],[debug safecdr code]) fi])
	       fi])

AC_ARG_ENABLE([prelink],[  --enable-prelink will insist that the produced images may be prelinked],
	      [if test "$enable_prelink" = "yes" ; then PRELINK_CHECK=t; fi])
AC_SUBST(PRELINK_CHECK)

AC_ARG_ENABLE([vssize],[  --enable-vssize=XXXX will compile in a value stack of size XXX],
	      [AC_DEFINE_UNQUOTED(VSSIZE,$enableval,[value stack size])])
AC_ARG_ENABLE([bdssize],[  --enable-bdssize=XXXX will compile in a binding stack of size XXX],
	      [AC_DEFINE_UNQUOTED(BDSSIZE,$enableval,[binding stack size])])
AC_ARG_ENABLE([ihssize],[  --enable-ihssize=XXXX will compile in a invocation history stack of size XXX],
	      [AC_DEFINE_UNQUOTED(IHSSIZE,$enableval,[invocation history stack size])])
AC_ARG_ENABLE([frssize],[  --enable-frssize=XXXX will compile in a frame stack of size XXX],
	      [AC_DEFINE_UNQUOTED(FRSSIZE,$enableval,[frame stack size])])

AC_ARG_ENABLE([infodir],[--enable-infodir=XXXX will force the use of a INFO_DIR=XXXXX as place to look for info],
	      [INFO_DIR=$enableval],[INFO_DIR=$prefix/share/info])
INFO_DIR=`eval echo $INFO_DIR/`

AC_ARG_ENABLE([xgcl],[  --enable-xgcl=yes will compile in support for XGCL],,[enable_xgcl=yes])

AC_ARG_ENABLE([debug],[  --enable-debug builds gcl with -g in CFLAGS to enable running under gdb],
	      ,[enable_debug=$def_debug])
AC_ARG_ENABLE([static],[  --enable-static will link your GCL against static as opposed to shared system libraries],
	      ,[enable_static=$def_static])
AC_ARG_ENABLE([pic],[  --enable-pic builds gcl with -fPIC in CFLAGS],,[enable_pic=$def_pic])


#
# System programs
#

# We set the default CFLAGS below, and don't want the autoconf default
# CM 20040106
if test "$CFLAGS" = "" ; then
    CFLAGS=" "
fi
if test "$LDFLAGS" = "" ; then
    LDFLAGS=" "
fi

AC_USE_SYSTEM_EXTENSIONS
AC_PROG_RANLIB
AC_PROG_CC
AC_PROG_CPP
AC_SUBST(CC)
GCL_CC=`basename $CC`
if echo $GCL_CC |grep gcc |grep -q win; then
    GCL_CC=gcc
fi
AC_SUBST(GCL_CC)
AC_SUBST(CPP)

add_arg_to_cflags() {

    AC_MSG_CHECKING([for CFLAG $1])
    CFLAGS_ORI=$CFLAGS
    CFLAGS="$CFLAGS -Werror $1 `echo $1|sed 's,-Wno-,-W,1'`"
    AC_RUN_IFELSE(
	[AC_LANG_PROGRAM([[]],[[]])],
	[CFLAGS="$CFLAGS_ORI $1";AC_MSG_RESULT([yes]);return 0],
	[AC_MSG_RESULT([no])],
	[AC_MSG_RESULT([no])])
    CFLAGS=$CFLAGS_ORI
    return 1

}

assert_arg_to_cflags() {
    if ! add_arg_to_cflags $1 ; then AC_MSG_RESULT([cannot add $1 to CFLAGS]); exit 1 ; fi
    return 0
}

add_args_to_cflags() {

    while test "$#" -ge 1 ; do
  	add_arg_to_cflags $1
	shift
    done
}

add_arg_to_ldflags() {

    AC_MSG_CHECKING([for LDFLAG $1])
    LDFLAGS_ORI=$LDFLAGS
    LDFLAGS="$LDFLAGS -Werror $1"
    AC_RUN_IFELSE(
	[AC_LANG_PROGRAM([[]],[[]])],
	[LDFLAGS="$LDFLAGS_ORI $1";AC_MSG_RESULT([yes]);return 0],
	[AC_MSG_RESULT([no])],
	[AC_MSG_RESULT([no])])
    LDFLAGS=$LDFLAGS_ORI
    return 1

}

assert_arg_to_ldflags() {
    if ! add_arg_to_ldflags $1 ; then AC_MSG_RESULT([cannot add $1 to LDFLAGS]); exit 1 ; fi
    return 0
}

add_args_to_ldflags() {

    while test "$#" -ge 1 ; do
  	add_arg_to_ldflags $1
	shift
    done
}

remove_arg_from_ldflags() {

    NEW_LDFLAGS=""
    for i in $LDFLAGS; do
	if ! test "$i" = "$1" ; then
      	    NEW_LDFLAGS="$NEW_LDFLAGS $i"
	else
            AC_MSG_RESULT([removing $1 from LDFLAGS])
	fi
    done
    LDFLAGS=$NEW_LDFLAGS

    return 0

}

add_args_to_cflags  -fsigned-char -pipe -fcommon \
                    -fno-builtin-malloc -fno-builtin-free \
                    -fno-PIE -fno-pie -fno-PIC -fno-pic \
	            -std=gnu17 \
	            -Wall \
	            -Wno-builtin-requires-header -Wno-empty-body -Wno-self-assign \
	            -Wno-unused-but-set-variable -D_FILE_OFFSET_BITS=64 -D_TIME_BITS=64

add_args_to_ldflags -no-pie # -Wl,-z,lazy

AC_MSG_CHECKING([for inline semantics])
AC_COMPILE_IFELSE(
    [AC_LANG_SOURCE([[
			inline int foo(int i) {return i;}

			int
			bar(int i) {return foo(i);}
		     ]])],
    [if `nm conftest.o |grep foo |awk '{if (NF==3) exit(-1)}'` ; then
    	AC_MSG_RESULT([new])
     else
	AC_COMPILE_IFELSE(
	    [AC_LANG_SOURCE([[
	    			extern inline int foo(int i) {return i;}

				int
				bar(int i) {return foo(i);}
			    ]])],
	    [if `nm conftest.o |grep foo |awk '{if (NF==3) exit(-1)}'` ; then
	    	AC_MSG_RESULT([old])
		AC_DEFINE([OLD_INLINE],[1],[extern inline semantics])
	     else
	       	AC_MSG_ERROR([need working inline semantics])
	     fi],
    	    [AC_MSG_ERROR([need to probe inline semantics])])
     fi],
     [AC_MSG_ERROR([need to probe inline semantics])])

case $use in
    *mingw*)
	assert_arg_to_cflags -fno-zero-initialized-in-bss
	assert_arg_to_cflags -mms-bitfields
	for i in makedefc windows/gcl.iss windows/sysdir.bat windows/install.lsp h/gclincl.h;  do
	    cat $i.in | sed 's,[^\r]\n$,\r\n,g' >tmp && mv tmp $i.in;
	done
	OLD_LDFLAGS=$LDFLAGS
	assert_arg_to_ldflags -pg
	GPL_FLAG="-pg"
	LDFLAGS=$OLD_LDFLAGS;;
    *gnuwin*)
	assert_arg_to_cflags -fno-zero-initialized-in-bss
	assert_arg_to_cflags -mms-bitfields
        assert_arg_to_ldflags -Wl,--stack,8000000
	OLD_LDFLAGS=$LDFLAGS
	assert_arg_to_ldflags -pg
	GPL_FLAG="-pg"
	LDFLAGS=$OLD_LDFLAGS;;
    386-linux)
	if ! add_arg_to_cflags -msse2 || ! add_arg_to_cflags -mfpmath=sse ; then
	   add_arg_to_cflags -ffloat-store;
        fi;;
    loongarch64-linux)
        add_arg_to_cflags -mno-relax
        add_arg_to_cflags -Wa,-mno-relax;;
    386-macosx)
#	assert_arg_to_cflags -Wno-error=implicit-function-declaration
	add_arg_to_cflags -Wno-incomplete-setjmp-declaration
	assert_arg_to_ldflags -Wl,-no_pie
	if test "$build_cpu" = "x86_64" ; then
	    assert_arg_to_cflags -m64
	    assert_arg_to_ldflags -m64
	    assert_arg_to_ldflags -Wl,-headerpad,72
        else
	    assert_arg_to_cflags -m32
	    assert_arg_to_ldflags -m32
	    assert_arg_to_ldflags -Wl,-headerpad,56
        fi;;
    FreeBSD) assert_arg_to_ldflags -Z;;
esac

if test "$enable_static" = "yes" ; then
    assert_arg_to_ldflags -static
    assert_arg_to_ldflags -Wl,-zmuldefs
    AC_DEFINE(STATIC_LINKING,1,[staticly linked images])
fi

TO3FLAGS=""
TO2FLAGS=""
TOSFLAGS=""

case "$use" in
    *mingw*)
  	TFPFLAG="";;
    m68k*)#FIXME gcc 4.x bug workaround
	TFPFLAG="";;
    *)
	TFPFLAG="-fomit-frame-pointer";;
esac

AC_CHECK_PROGS(AWK,[gawk nawk awk])
GCL_CC_ARGS=`echo $CC | ${AWK} '{$1="";print}'`
GCL_CC="`basename $CC` $GCL_CC_ARGS"
if echo $GCL_CC |grep gcc |grep -q win; then
    GCL_CC=gcc
fi
AC_SUBST(GCL_CC)


GPROF="gprof_objs"
AC_ARG_ENABLE([gprof],[  --enable-gprof builds gcl with -pg in CFLAGS to enable profiling with gprof],
	      [if test "$enableval" != "yes" ; then GPROF=""; fi])

if test "$GPROF" != "" ; then
   case $use in
       sh4*)  GPROF="";;
       m68k*) GPROF="";;
       ia64*) GPROF="";;
       gnuwin95*) GPROF="";;
   esac
   OLD_CFLAGS=$CFLAGS
   if ! add_arg_to_cflags -pg ; then GPROF="" ; fi
   CFLAGS=$OLD_CFLAGS
   AC_MSG_CHECKING([working gprof])
   if test "$GPROF" = "" ; then
       AC_MSG_RESULT([disabled])
   else
       AC_MSG_RESULT([ok])
       AC_DEFINE(USE_GPROF,1,[use gprof])
   fi
fi
dnl AC_SUBST(GPROF)
AM_CONDITIONAL([AMM_GPROF],[test "$GPROF" != ""])

if test "$enable_debug" = "yes" ; then
    assert_arg_to_cflags -g
    # for subconfigurations
    CFLAGS="$CFLAGS -g"
else
    TOSFLAGS="-O2" # "-Os $TFPFLAG"
    TO3FLAGS="-O3 $TFPFLAG"
    TO2FLAGS="-O"
fi

# gcc on ppc cannot compile our new_init.c with full opts --CM
TONIFLAGS=""
case $use in 
    powerpc*macosx) assert_arg_to_cflags -mlongcall;;
    *linux)
	case $use in
	    alpha*)
		assert_arg_to_cflags -mieee
#		if test "$enable_debug" != "yes" ; then TO3FLAGS="-O0" ; fi #FIXME needed asof gcc 4.8.1
		;;
	    aarch64*)
		TLIBS="$TLIBS -lgcc_s";;
	    hppa*)
		assert_arg_to_cflags -mlong-calls
		TLIBS="$TLIBS -lgcc_s" # workaround hppa __moddi3 local func symbols with default linker flags
		if test "$enable_debug" != "yes" ; then TO3FLAGS="-O2" ; TFPFLAG=""; fi #FIXME needed asof gcc 4.8.1
		;;
	    mips*)
		case $canonical in
		    mips64*linux*)
#			assert_arg_to_cflags -mxgot
			assert_arg_to_ldflags -Wl,-z,now;;
		esac
		;;
	    ia64*)
		if test "$enable_debug" != "yes" ; then TO3FLAGS="-O" ; fi #FIXME needed asof gcc 4.8.1
		;;
	    arm*)
		assert_arg_to_cflags -fdollars-in-identifiers
		assert_arg_to_cflags -g #?
		;;
	    powerpc*)
		assert_arg_to_cflags -mlongcall
		if test "$host_cpu" != "powerpc64le" ; then assert_arg_to_cflags -mno-pltseq; fi
		;;
	esac;;
esac
if test "$enable_pic" = "yes" ; then
    assert_arg_to_cflags -fPIC
fi

FDEBUG=`echo $CFLAGS | tr ' ' '\012' |grep "^-g$"|tr '\012' ' '`
#CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-g$"`
FOMITF=`echo $CFLAGS | tr ' ' '\012' |grep "^-fomit-frame-pointer$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-fomit-frame-pointer$"|tr '\012' ' '`
FOOPT3=`echo $CFLAGS | tr ' ' '\012' |grep "^-O3$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O3$"|tr '\012' ' '`
FOOPT2=`echo $CFLAGS | tr ' ' '\012' |grep "^-O2$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O2$"|tr '\012' ' '`
FOOPT1=`echo $CFLAGS | tr ' ' '\012' |grep "^-O1$"|tr '\012' ' '`
TMPF=`echo $CFLAGS | tr ' ' '\012' |grep "^-O$"|tr '\012' ' '`
FOOPT1="$FOOPT1$TMPF"
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O1$"|grep -v "^-O$"|tr '\012' ' '`
FOOPT0=`echo $CFLAGS | tr ' ' '\012' |grep "^-O0$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O0$"|tr '\012' ' '`


if test "$FOOPT0" != "" ; then
   TO3FLAGS=`echo $TO3FLAGS | sed 's,-O[[123 ]],-O0 ,g' | sed 's,-O$,-O0 ,g'`
   TO2FLAGS=`echo $TO2FLAGS | sed 's,-O[[123 ]],-O0 ,g' | sed 's,-O$,-O0 ,g'`
   TOSFLAGS=`echo $TOSFLAGS | sed 's,-O[[123 ]],-O0 ,g' | sed 's,-O$,-O0 ,g'`
else 
if test "$FOOPT1" != "" ; then
   TO3FLAGS=`echo $TO3FLAGS | sed 's,-O[[2-3]],-O1,g'`
   TO2FLAGS=`echo $TO2FLAGS | sed 's,-O[[2-3]],-O1,g'`
   TOSFLAGS=`echo $TOSFLAGS | sed 's,-O[[2-3]],-O1,g'`
else 
if test "$FOOPT2" != "" ; then
   TO3FLAGS=`echo "$TO3FLAGS" | sed 's,-O3,-O2,g'`
   TO2FLAGS=`echo "$TO2FLAGS" | sed 's,-O3,-O2,g'`
   TOSFLAGS=`echo "$TOSFLAGS" | sed 's,-O3,-O2,g'`
fi
fi
fi

if test "$FDEBUG" != "" ; then
   TO3FLAGS=`echo $TO3FLAGS | sed 's,-fomit-frame-pointer,,g'`
   TO2FLAGS=`echo $TO2FLAGS | sed 's,-fomit-frame-pointer,,g'`
   TOSFLAGS=`echo $TOSFLAGS | sed 's,-fomit-frame-pointer,,g'`
fi

if test "$FOMITF" != "" ; then
   TO3FLAGS="$TO3FLAGS $FOMITF"
fi

FDEBUG=`echo $CFLAGS | tr ' ' '\012' |grep "^-g$"|tr '\012' ' '`
#CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-g$"`
FOMITF=`echo $CFLAGS | tr ' ' '\012' |grep "^-fomit-frame-pointer$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-fomit-frame-pointer$"|tr '\012' ' '`
FOOPT3=`echo $CFLAGS | tr ' ' '\012' |grep "^-O3$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O3$"|tr '\012' ' '`
FOOPT2=`echo $CFLAGS | tr ' ' '\012' |grep "^-O2$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O2$"|tr '\012' ' '`
FOOPT1=`echo $CFLAGS | tr ' ' '\012' |grep "^-O1$"|tr '\012' ' '`
TMPF=`echo $CFLAGS | tr ' ' '\012' |grep "^-O$"|tr '\012' ' '`
FOOPT1="$FOOPT1$TMPF"
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O1$"|grep -v "^-O$"|tr '\012' ' '`
FOOPT0=`echo $CFLAGS | tr ' ' '\012' |grep "^-O0$"|tr '\012' ' '`
CFLAGS=`echo $CFLAGS | tr ' ' '\012' |grep -v "^-O0$"|tr '\012' ' '`

if test "$FOOPT0" != "" ; then
    TO3FLAGS=`echo $TO3FLAGS | sed 's,-O[[123 ]],-O0 ,g' | sed 's,-O$,-O0 ,g'`
    TO2FLAGS=`echo $TO2FLAGS | sed 's,-O[[123 ]],-O0 ,g' | sed 's,-O$,-O0 ,g'`
else
    if test "$FOOPT1" != "" ; then
	TO3FLAGS=`echo $TO3FLAGS | sed 's,-O[[2-3]],-O1,g'`
	TO2FLAGS=`echo $TO2FLAGS | sed 's,-O[[2-3]],-O1,g'`
    else
	if test "$FOOPT2" != "" ; then
	    TO3FLAGS=`echo "$TO3FLAGS" | sed 's,-O3,-O2,g'`
	    TO2FLAGS=`echo "$TO2FLAGS" | sed 's,-O3,-O2,g'`
	fi
    fi
fi

if test "$FDEBUG" != "" ; then
    TO3FLAGS=`echo $TO3FLAGS | sed 's,-fomit-frame-pointer,,g'`
    TO2FLAGS=`echo $TO2FLAGS | sed 's,-fomit-frame-pointer,,g'`
fi

if test "$FOMITF" != "" ; then
    TO3FLAGS="$TO3FLAGS $FOMITF"
fi

# Step 1: set the variable "system" to hold the name and version number
# for the system.  This can usually be done via the "uname" command, but
# there are a few systems, like Next, where this doesn't work.

AC_CHECK_PROGS(MAKEINFO,makeinfo,"false")
AC_SUBST(MAKEINFO)

AC_MSG_CHECKING([system version (for dynamic loading)])
if machine=`uname -m` ; then true; else machine=unknown ; fi

if test -f /usr/lib/NextStep/software_version; then
    system=NEXTSTEP-`$AWK '/3/,/3/' /usr/lib/NextStep/software_version`
else
    system=`uname -s`-`uname -r`
    if test "$?" -ne 0 ; then
	AC_MSG_RESULT([unknown (cannot find uname command)])
	system=unknown
    else
	# Special check for weird MP-RAS system (uname returns weird
	# results, and the version is kept in special file).

	if test -r /etc/.relid -a "X`uname -n`" = "X`uname -s`" ; then
	    system="MP-RAS-`${AWK} '{print $3}' '/etc/.relid'`"
	fi
	if test "`uname -s`" = "AIX" ; then
	    system=AIX-`uname -v`.`uname -r`
	fi
	AC_MSG_RESULT($system)
    fi
fi

case $use in
    *macosx)
        AC_CHECK_HEADERS(malloc/malloc.h,,[AC_MSG_ERROR([need malloc.h on macosx])])
	AC_CHECK_MEMBER([struct _malloc_zone_t.memalign],
 			AC_DEFINE(HAVE_MALLOC_ZONE_MEMALIGN,1,[memalign element present]), [],
			[
			    #include <malloc/malloc.h>
			])
	AC_SUBST(HAVE_MALLOC_ZONE_MEMALIGN)
	;;
esac

AC_CHECK_HEADERS(
    [setjmp.h],
    [AC_MSG_CHECKING([sizeof jmp_buf])
     AC_RUN_IFELSE(
	 [AC_LANG_PROGRAM(
		 [[
		     #include <stdio.h>
		     #include <setjmp.h>
		 ]],
		 [[
		     FILE *fp=fopen("conftest1","w");
		     fprintf(fp,"%lu\n",sizeof(jmp_buf));
		     fclose(fp);
		 ]])],
	 [sizeof_jmp_buf=`cat conftest1`
	  AC_MSG_RESULT($sizeof_jmp_buf)
	  AC_DEFINE_UNQUOTED(SIZEOF_JMP_BUF,$sizeof_jmp_buf,[sizeof jmp_buf])],
	 [AC_MSG_RESULT([no])])])

# sysconf
AC_CHECK_HEADERS(
    [unistd.h],
    [AC_CHECK_LIB(
	    [c],[sysconf],
	    [AC_MSG_CHECKING([_SC_CLK_TCK])
	     hz=0
	     AC_RUN_IFELSE(
		 [AC_LANG_PROGRAM(
			 [[
			     #include <unistd.h>
			     #include <stdio.h>
			 ]],
			 [[
			     FILE *fp=fopen("conftest1","w");
			     fprintf(fp,"%lu\n",sysconf(_SC_CLK_TCK));
			     fclose(fp);
			 ]],
			 [hz=`cat conftest1`
			  AC_DEFINE_UNQUOTED(HZ,$hz,[time system constant])])])
	     AC_MSG_RESULT($hz)])])


AC_CHECK_HEADERS([gmp.h],
   [AC_CHECK_LIB([gmp],[__gmpz_init],
	[AC_MSG_CHECKING([for external gmp version])
	 AC_RUN_IFELSE(
	     [AC_LANG_PROGRAM(
		     [[
			 #include <gmp.h>
		     ]],
		     [[
			 #if __GNU_MP_VERSION > 3
			 return 0;
			 #else
			 return -1;
			 #endif
		     ]])],
	     [AC_MSG_RESULT([good])
	      TLIBS="$TLIBS -lgmp"
	      echo "#include \"gmp.h\"" >foo.c
	      echo "int main() {return 0;}" >>foo.c
	      MP_INCLUDE=`$CC -E foo.c | $AWK '/(\/|\\\\)gmp.h/ {if (!i) print $3;i=1}' | tr -d '"'`
	      rm -f foo.c])])])

if test "$MP_INCLUDE" = "" ; then
	AC_MSG_ERROR([Cannot use dynamic gmp lib])
fi


AC_MSG_CHECKING([for leading underscore in object symbols])
cat>foo.c <<EOFF
#include <math.h>
#include <stdio.h>
int main() {FILE *f;double d=0.0;getc(f);d=cos(d);return 0;}
EOFF
$CC -c foo.c -o foo.o
if nm foo.o |grep " U " | grep "_cos" >/dev/null || nm foo.o |grep " U " | grep " _getc" >/dev/null ; then
    LEADING_UNDERSCORE=1
    AC_DEFINE(LEADING_UNDERSCORE,1,[symbol name mangling convention])
    AC_MSG_RESULT("yes")
else
    LEADING_UNDERSCORE=""
    AC_MSG_RESULT("no")
fi

AC_MSG_CHECKING([for size of gmp limbs])
AC_RUN_IFELSE([AC_LANG_PROGRAM([[
				   #include <stdio.h>
				   #include "$MP_INCLUDE"
			       ]],
			       [[
				   FILE *fp=fopen("conftest1","w");
				   fprintf(fp,"%u",sizeof(mp_limb_t));
				   fclose(fp);
			       ]])],[mpsize=`cat conftest1`],[AC_MSG_ERROR([Cannot determine mpsize])])
AC_DEFINE_UNQUOTED(MP_LIMB_BYTES,$mpsize,[sizeof mp_limb in gmp library])
AC_MSG_RESULT($mpsize) 

AC_MSG_CHECKING([_SHORT_LIMB])
AC_RUN_IFELSE([AC_LANG_PROGRAM([[
				   #include <stdio.h>
				   #include "$MP_INCLUDE"
			       ]],
			       [[
				   #ifdef _SHORT_LIMB
				   return 0;
				   #else
				   return 1;
				   #endif
			       ]])],[AC_DEFINE(__SHORT_LIMB,1,[short gmp3 limbs]) AC_MSG_RESULT(yes)],[AC_MSG_RESULT(no)])

AC_MSG_CHECKING([_LONG_LONG_LIMB])
AC_RUN_IFELSE([AC_LANG_PROGRAM([[
				   #include <stdio.h>
				   #include "$MP_INCLUDE"
			       ]],
			       [[
				   #ifdef _LONG_LONG_LIMB
				   return 0;
				   #else
				   return 1;
				   #endif
			       ]])],[AC_DEFINE(__LONG_LONG_LIMB,1,[long gmp3 limbs]) AC_MSG_RESULT(yes)],[AC_MSG_RESULT(no)])

GMP=1
AC_DEFINE(GMP,1,[using gmp])
AC_SUBST(GMP)
AC_SUBST(GMPDIR)

AC_MSG_CHECKING([for GNU ld option -Map])
touch map foo.c
$CC -o foo [ -Wl,-Map ] map foo.c >/dev/null 2>&1
if test `cat map | wc -l` != "0" ; then
	AC_MSG_RESULT([yes])
	AC_DEFINE(HAVE_GNU_LD,1,[gnu linker present])
	GNU_LD=1
else
	AC_MSG_RESULT([no])
	GNU_LD=
fi
rm -f foo.c foo.o foo map

#
# X windows
# 

if test "$enable_xgcl" = "yes" ; then 

    AC_PATH_X

    AC_CHECK_LIB(X11,main,
		 [X_LIBS="$X_LIBS -lX11" AC_DEFINE(HAVE_XGCL,1,[using xgcl])],
		 [AC_MSG_RESULT([missing x libraries -- cannot compile xgcl])])

fi
AM_CONDITIONAL([AMM_XGCL],[test "$X_LIBS" != ""])


AC_SUBST(X_LIBS)
AC_SUBST(X_CFLAGS)

#
# Dynamic loading
#

# boot.so requires dlopen
AC_CHECK_LIB([dl],[dlopen],,AC_MSG_ERROR([Cannot find dlopen]))


AC_ARG_ENABLE([xdr],[  --enable-xdr=yes will compile in support for XDR])

if test "$enable_xdr" != "no" ; then
    XDR_LIB=""
    AC_CHECK_FUNC([xdr_double],XDR_LIB=" ",
		  [AC_CHECK_LIB([tirpc],[xdr_double],[XDR_LIB=tirpc],
				[AC_CHECK_LIB([gssrpc],[xdr_double],[XDR_LIB=gssrpc],
					      [AC_CHECK_LIB([rpc],[xdr_double],[XDR_LIB=rpc],
							    [AC_CHECK_LIB([oncrpc],[xdr_double],[XDR_LIB=oncrpc])])])])])

    if test "$XDR_LIB" != ""; then
	AC_DEFINE(HAVE_XDR,1,[have xdr extensions])
	if test "$XDR_LIB" != " "; then
	    TLIBS="$TLIBS -l$XDR_LIB"
	    add_arg_to_cflags -I/usr/include/$XDR_LIB
	fi
    fi
fi


AC_MSG_CHECKING([__builtin_clzl])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
		#include <unistd.h>
	    ]],
	    [[
		unsigned long u;
		long j;
		if (__builtin_clzl(0)!=sizeof(long)*8)
      		   return -1;
		   for (u=1,j=sizeof(long)*8-1;j>=0;j--,u<<=1)
      		       if (__builtin_clzl(u)!=j)
  			  return -1;
	    ]])],
    [AC_MSG_RESULT([yes])
     AC_DEFINE(HAVE_CLZL,[1],[clzl instruction])],
    [AC_MSG_RESULT([no])])

AC_MSG_CHECKING([__builtin_ctzl])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
		#include <unistd.h>
	    ]],
	    [[
		unsigned long u;
		long j;
		if (__builtin_ctzl(0)!=sizeof(long)*8)
      		   return -1;
		   for (u=1,j=0;j<sizeof(long)*8-1;j++,u<<=1)
      		       if (__builtin_ctzl(u)!=j)
  			  return -1;
	    ]])],
    [AC_MSG_RESULT([yes])
     AC_DEFINE(HAVE_CTZL,[1],[ctzl instruction])],
    [AC_MSG_RESULT([no])])


case $use in
    sh4*) ;;     #FIXME, these exceptions needed as of gcc 4.7
    hppa*) ;;    #FIXME
    powerpc*) ;; #FIXME
    alpha*) ;;   #FIXME
    ia64*) ;;    #FIXME
    *)
	AC_MSG_CHECKING(__builtin___clear_cache)
	AC_RUN_IFELSE(
	    [AC_LANG_PROGRAM(
		    [[]],
		    [[
			void *v,*ve;
			__builtin___clear_cache(v,ve);
		    ]])],
	    [AC_DEFINE(HAVE_BUILTIN_CLEAR_CACHE,1,[have __builtin__clear_cache instruction])
	     AC_MSG_RESULT(yes)],
	    AC_MSG_RESULT(no));;
esac

#AC_CONFIG_SUBDIRS($MY_SUBDIRS)

# Find where Data begins.  This is used by the storage allocation
# mechanism, in the PAGE macro.  This offset is subtracted from
# addresses, in calculating a page for an address in the heap.

AC_CHECK_SIZEOF(long,0)
AC_CHECK_SIZEOF(short,0)
AC_CHECK_SIZEOF(int,0)
AC_CHECK_SIZEOF(char,0)


AC_ARG_ENABLE([immfix],[  --enable-immfix will enable an immediate fixnum table above the C stack])

AC_ARG_ENABLE([fastimmfix],[  --enable-fastimmfix=XXXX will reject low immediate fixnums unless 2^XXX can be attained],,[enable_fastimmfix=64])


# Should really find a way to check for prototypes, but this
# basically works for now.  CM
#
AC_CHECK_HEADERS(math.h,AC_DEFINE(HAVE_MATH_H,1,[have math.h]))
AC_CHECK_HEADERS(complex.h,AC_DEFINE(HAVE_COMPLEX_H,1,[have complex.h]))

####  Memory areas and alignment

AC_MSG_CHECKING(for byte order)
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[]],
	    [[
		/* Are we little or big endian?  Adapted from Harbison&Steele.  */
		union {long l;char c[sizeof(long)];} u;
		u.l = 1;
		return u.c[sizeof(long)-1] ? 1 : 0;
	    ]])],[
	AC_MSG_RESULT(little)],
    [AC_MSG_RESULT(big)
     AC_DEFINE(WORDS_BIGENDIAN,1,[big endian byte order])])
AC_SUBST(WORDS_BIGENDIAN)


AC_MSG_CHECKING(for word order)
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[]],
	    [[
		/* Are we little or big endian?  Adapted from Harbison&Steele.  */
		union {double d;int l[sizeof(double)/sizeof(int)];} u;
		u.d = 1.0;
		return u.l[sizeof(double)/sizeof(int)-1] ? 0 : 1;
	    ]])],
    [AC_MSG_RESULT(little)],
    [AC_MSG_RESULT(big)
     AC_DEFINE(DOUBLE_BIGENDIAN,1,[big endian word order])])
AC_SUBST(DOUBLE_BIGENDIAN)


# pagewidth
case $use in
    mips*) min_pagewidth=14;;
    *) min_pagewidth=12;;
esac

AC_MSG_CHECKING([for hugepagewidth])
j=""
if test -e /proc/meminfo ; then
   j=`awk '/^Hugepagesize:/ {if ($3!="kB") next;j=1;for (i=0;i<20 && j!=$2;i++) j=j*2;if (j==$2) printf("%d\n",i+10)}'  /proc/meminfo`
fi
if test "$j" != "" ; then
   AC_MSG_RESULT([$j])

   AC_CHECK_HEADER([sys/mman.h],
      [AC_CHECK_FUNC([madvise],
         [AC_CHECK_DECL([MADV_HUGEPAGE],
            [AC_MSG_CHECKING([madvise works])
	     AC_RUN_IFELSE(
	       [AC_LANG_PROGRAM(
	   	    [[
			#include <sys/mman.h>
			#include <stdlib.h>
		    ]],
	    	    [[
	    		unsigned long k=(1UL<<$j);
			void *p=malloc(2*k);
			int i;
			p=(void *)((((unsigned long)p)+k-1)&~(k-1));
			i=madvise(p,k,MADV_HUGEPAGE);
			return i;
         	    ]])],
		[AC_MSG_RESULT([yes])
		 AC_DEFINE(HAVE_MADVISE_HUGEPAGE,[1],[can madvise hugepages])
		 if test $min_pagewidth -lt $j ; then min_pagewidth=$j ; fi],
		[AC_MSG_RESULT([no])])]
	     ,[],[#include <sys/mman.h>])])])
else
   AC_MSG_RESULT([not found])
fi

AC_ARG_ENABLE(min_pagewidth,[ --enable-min_pagewidth=xxx sets 1<<xxx byte pages],
             [min_pagewidth=$enableval],[])

AC_MSG_CHECKING(for pagewidth)
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
		#include <unistd.h>
		#ifdef __CYGWIN__
		#define getpagesize() 4096
		#endif
	    ]],
	    [[
		size_t i=getpagesize(),j;
		FILE *fp=fopen("conftest1","w");
		for (j=0;i>>=1;j++);
		    j=j<$min_pagewidth ? $min_pagewidth : j;
		    fprintf(fp,"%u",j);
	    ]])],
    [PAGEWIDTH=`cat conftest1`],
    [PAGEWIDTH=0])
AC_MSG_RESULT($PAGEWIDTH)
AC_DEFINE_UNQUOTED(PAGEWIDTH,$PAGEWIDTH,[system pagewidth])
AC_SUBST(PAGEWIDTH)

AC_MSG_CHECKING([for required object alignment])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
		#include <stdlib.h>
		#include <complex.h>
		#define EXTER
		#define INLINE
		#include "$MP_INCLUDE"
		#include "$srcdir/h/enum.h"
		#define OBJ_ALIGN
		#include "$srcdir/h/type.h"
		#include "$srcdir/h/lu.h"
		#include "$srcdir/h/object.h"
	    ]],
	    [[
		unsigned long i;
   		FILE *fp=fopen("conftest1","w");
		for (i=1;i && NOT_OBJECT_ALIGNED(i); i<<=1);
                if (!i) return -1;
		fprintf(fp,"%lu",i);
		fclose(fp);
		return 0;
	    ]])],
    [obj_align=`cat conftest1`
     AC_MSG_RESULT($obj_align)
     AC_DEFINE_UNQUOTED(OBJ_ALIGNMENT,$obj_align,[needed object alignment bytes])],
    [AC_MSG_ERROR([Cannot find object alignent])])

AC_MSG_CHECKING([for C extension variable alignment])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM([[]],
		     [[
			 char *v __attribute__ ((aligned ($obj_align)));
		     ]])],[obj_align="__attribute__ ((aligned ($obj_align)))"],[AC_MSG_ERROR([Need alignment attributes])])
AC_MSG_RESULT($obj_align)
AC_DEFINE_UNQUOTED(OBJ_ALIGN,$obj_align,[can use C extension for object alignment])

AC_MSG_CHECKING([for C extension noreturn function attribute])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM([[]],
		     [[
			 extern int v() __attribute__ ((noreturn));
		     ]])],
    [no_return="__attribute__ ((noreturn))"],[no_return=])
AC_MSG_RESULT($no_return)
AC_DEFINE_UNQUOTED(NO_RETURN,$no_return,[can use C extension for functions that do not return])

AC_MSG_CHECKING([sizeof struct contblock])

AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
		#include <stdlib.h>
		#include <complex.h>
		#define EXTER
		#define INLINE
		#include "$MP_INCLUDE"
		#include "$srcdir/h/enum.h"
		#include "$srcdir/h/type.h"
		#include "$srcdir/h/lu.h"
		#include "$srcdir/h/object.h"
	    ]],
	    [[
		FILE *f=fopen("conftest1","w");
		fprintf(f,"%u",sizeof(struct contblock));
		fclose(f);
	    ]])],
    [sizeof_contblock=`cat conftest1`],
    [AC_MSG_ERROR([Cannot find sizeof struct contblock])],
    [AC_MSG_ERROR([Cannot find sizeof struct contblock])])

AC_MSG_RESULT($sizeof_contblock)
AC_DEFINE_UNQUOTED(SIZEOF_CONTBLOCK,$sizeof_contblock,[sizeof linked list for contiguous pages])

AC_MSG_CHECKING(CSTACK_DIRECTION)
AC_RUN_IFELSE(
    [AC_LANG_SOURCE(
	    [[
                #include <unistd.h>
		#include <stdio.h>
		#include <stdlib.h>
		int main(int argc,char **argv,char **envp) {
		    FILE *fp = fopen("conftest1","w");
		    fprintf(fp,"%d",(alloca(sizeof(void *))>alloca(sizeof(void *))) ? -1 : 1);
		    fclose(fp);
		    return 0;
		}]])],
    [cstack_direction=`cat conftest1`],[cstack_direction=0])
AC_DEFINE_UNQUOTED(CSTACK_DIRECTION,$cstack_direction,[whether C stack grows up or down])
AC_MSG_RESULT($cstack_direction)


AC_MSG_CHECKING([finding CSTACK_ALIGNMENT])
AC_RUN_IFELSE(
    [AC_LANG_SOURCE(
	    [[
                #include <unistd.h>
		#include <stdio.h>
		#include <stdlib.h>
		int main(int argc,char **argv,char **envp) {
		    void *b,*c;
		    FILE *fp = fopen("conftest1","w");
		    long n;
		    b=alloca(sizeof(b));
		    c=alloca(sizeof(c));
		    n=b>c ? b-c : c-b;
		    n=n>sizeof(c) ? n : 1;
		    fprintf(fp,"%ld",n);
		    fclose(fp);
		    return 0;
		}]])],
    [cstack_alignment=`cat conftest1`],[cstack_alignment=0])
AC_DEFINE_UNQUOTED(CSTACK_ALIGNMENT,$cstack_alignment,[C stack alignment])
AC_MSG_RESULT($cstack_alignment)

if test $cstack_direction -eq 1 ; then

   AC_MSG_CHECKING(CSTACK_TOP)
   AC_RUN_IFELSE(
       [AC_LANG_SOURCE(
   	    [[
   		#include <stdio.h>
   		#include <stdlib.h>
   		#include <unistd.h>
   		int
   		main(int argc,char **argv,char **envp) {
   		    FILE *fp = fopen("conftest1","w"),*f=fopen("/proc/self/maps","r");
   		    unsigned long i,j;
		    char b[4096];
   		    i=(unsigned long)alloca(sizeof(void *));
		    for (j=0;j<i && fgets(b,sizeof(b),f);)
		    	sscanf(b,"%x-",&j);
		    j-=1;
		    if (j<i) j=-1;
		    fclose(f);
   		    fprintf(fp,"0x%lx",j);
   		    fclose(fp);
   		    return 0;
   		}]])],
       [cstack_top=`cat conftest1`],[cstack_top=0])
   AC_MSG_RESULT($cstack_top)

   AC_MSG_CHECKING([relocated CSTACK_TOP])
   AC_RUN_IFELSE(
       [AC_LANG_SOURCE(
   	    [[
   		#include <stdio.h>
   		#include <stdlib.h>
   		#include <sys/mman.h>
   		#include <unistd.h>
   		#include <errno.h>
   		int
   		main(int argc,char **argv,char **envp) {
   		    FILE *fp = fopen("conftest1","w"),*f=fopen("/proc/self/maps","r");
   		    unsigned long i,j;
		    char b[4096],*stack_map_base;
		    #include "$srcdir/h/cstack.h"
   		    i=(unsigned long)alloca(sizeof(void *));
		    for (j=0;j<i && fgets(b,sizeof(b),f);)
		    	sscanf(b,"%x-",&j);
		    j-=1;
		    if (j<i) j=-1;
		    fclose(f);
   		    fprintf(fp,"0x%lx",j);
   		    fclose(fp);
   		    return 0;
   		}]])],
      [cstack_top=`cat conftest1`],[cstack_top=0])
   AC_MSG_RESULT($cstack_top)

else

   AC_MSG_CHECKING(CSTACK_TOP)
   AC_RUN_IFELSE(
       [AC_LANG_SOURCE(
   	    [[
   		#include <stdio.h>
   		#include <stdlib.h>
   		#include <unistd.h>
   		int
   		main(int argc,char **argv,char **envp) {
   		    FILE *fp = fopen("conftest1","w");
   		    unsigned long i,j;
   		    j=getpagesize();
   		    i=(unsigned long)alloca(sizeof(void *));
   		    j--;
   		    i+=j;
   		    i&=~j;
   		    fprintf(fp,"0x%lx",i-1);
   		    fclose(fp);
   		    return 0;
   		}]])],
       [cstack_top=`cat conftest1`],[cstack_top=0])
   AC_MSG_RESULT($cstack_top)

   AC_MSG_CHECKING([relocated CSTACK_TOP])
   AC_RUN_IFELSE(
       [AC_LANG_SOURCE(
   	    [[
   		#include <stdio.h>
   		#include <stdlib.h>
   		#include <sys/mman.h>
   		#include <unistd.h>
   		#include <errno.h>
   		int
   		main(int argc,char **argv,char **envp) {
   		    FILE *fp = fopen("conftest1","w");
   		    unsigned long i,j;
		    char *stack_map_base;
   		    #include "$srcdir/h/cstack.h"
   		    j=getpagesize();
   		    i=(unsigned long)alloca(sizeof(void *));
   		    j--;
   		    i+=j;
   		    i&=~j;
   		    fprintf(fp,"0x%lx",i-1);
   		    fclose(fp);
   		    return 0;
   		}]])],
      [cstack_top=`cat conftest1`],[cstack_top=0])
   AC_MSG_RESULT($cstack_top)

fi

AC_MSG_CHECKING([cstack bits])
AC_RUN_IFELSE(
    [AC_LANG_SOURCE(
	    [[
		#include <stdio.h>
		#include <stdlib.h>
		#include <unistd.h>
		#include <sys/mman.h>
		#include <unistd.h>
		#include <errno.h>
		int
		main(int argc,char **argv,char **envp) {
		    FILE *fp = fopen("conftest1","w");
		    long i,j;
		    char *stack_map_base;
		    #include "$srcdir/h/cstack.h"
		    j=getpagesize();
		    i=$cstack_top;
		    j--;
		    i+=j;
		    i&=~j;
		    for (j=0;(i>>j)!=(i>>(sizeof(long)*8-1));j++);
		    fprintf(fp,"%ld",j);
		    fclose(fp);
		    return 0;
		}]])],
    [cstack_bits=`cat conftest1`],[cstack_bits=0])
AC_MSG_RESULT($cstack_bits)


AC_MSG_CHECKING(NEG_CSTACK_ADDRESS)
AC_RUN_IFELSE(
    [AC_LANG_SOURCE(
	    [[
		#include <stdio.h>
		#include <stdlib.h>
		#include <unistd.h>
		#include <sys/mman.h>
		#include <unistd.h>
		#include <errno.h>
		int
		main(int argc,char **argv,char **envp) {
	            char *stack_map_base;
		    #include "$srcdir/h/cstack.h"
		    return (long)$cstack_top<0 ? 0 : -1;
		}]])],
    [AC_MSG_RESULT(yes)
     neg_cstack_address=1],
    [AC_MSG_RESULT(no)
     neg_cstack_address=0])


AC_ARG_ENABLE([immfix],[  --enable-immfix will enable an immediate fixnum table above the C stack])

AC_ARG_ENABLE([fastimmfix],[  --enable-fastimmfix=XXXX will reject low immediate fixnums unless 2^XXX can be attained],,[enable_fastimmfix=64])

AC_MSG_CHECKING([finding default linker script])
if ! test -d unixport ; then mkdir unixport ; fi
touch unixport/gcl.script
echo "int main() {return 0;}" >foo.c
$CC $LDFLAGS -Wl,--verbose foo.c -o foo 2>&1 | \
	$AWK '/==================================================/ {i=1-i;next} {if (i) print}' >gcl.script
rm -rf foo.c foo

if test "`cat gcl.script | wc -l`" != "0" ; then
	AC_MSG_RESULT(got it)

	AC_MSG_CHECKING([output_arch])
	output_arch=`cat gcl.script |grep OUTPUT_ARCH|head -n 1|sed 's,.*(\(.*\)).*,\1:,1'|cut -f1 -d:|tr '-' '_'`;
	if test "$output_arch" != "" ; then
	    AC_DEFINE_UNQUOTED(OUTPUT_ARCH,bfd_arch_${output_arch},[bfd output arch])
 	    AC_MSG_RESULT([bfd_arch_${output_arch}])
	else
 	    AC_MSG_RESULT([not found])
	fi
	AC_MSG_CHECKING([output_mach])
	output_mach=`cat gcl.script |grep OUTPUT_ARCH|head -n 1|sed 's,.*(\(.*\)).*,\1:,1'|cut -f2 -d:|tr '-' '_'|tr -d '.'`;

    if test "$output_mach" = "common" ; then #FIXME
	   output_mach=""
	fi
	defaulted=""
    if test "$output_mach" = "" ; then
		if test "$output_arch" = "i386" ; then
		   output_mach="i386_i386";
		   defaulted="(defaulted)"
		fi
	fi
	if test "$output_mach" != "" ; then
	    AC_DEFINE_UNQUOTED(OUTPUT_MACH,bfd_mach_${output_mach},[bfd output mach])
 	    AC_MSG_RESULT([$defaulted bfd_mach_${output_mach}])
	else
 	    AC_MSG_RESULT([not found])
	fi

	AC_MSG_NOTICE([trying to adjust text start])
	cp gcl.script gcl.script.def

	n=-1;
	k=0;
	lim=`$AWK 'END {printf("%d\n",m*8-2)}' m=$ac_cv_sizeof_long`;
	max=0;
	min=$lim;
	while test $n -lt $lim ; do
        j=`$AWK 'END {for (i=j=0;j<=n;j++) i=i ? i*2 : 1;printf("%x\n",3*i)}' n=$n </dev/null`
        cat gcl.script.def | $AWK '/SEGMENT_START/ {gsub("0x[[0-9]]*","0x" j,$0);} {print}' j=$j >gcl.script
	    #        diff -u gcl.script.def gcl.script
        echo "int main() {return 0;}" >foo.c
        if ( $CC $LDFLAGS -Wl,-T gcl.script foo.c -o foo >/dev/null 2>&1 && ./foo >/dev/null 2>&1 ) >/dev/null 2>&1 ; then
		if test $n -lt $min; then min=$n; fi;
		if test $n -gt $max; then max=$n; fi;
	    elif test $max -gt 0 ; then
	    	# Workaround for false island of acceptability on riscv64, 20240716
	    	if test `$AWK 'END {print n-m}' m=$min n=$max </dev/null` -gt 2 ; then
		   break;
		else
		   min=$lim
		   max=0;
		fi
        fi;
        n=`$AWK 'END {print n+1}' n=$n </dev/null`
	done

	AC_MSG_NOTICE([min log text start $min])
	AC_MSG_NOTICE([max log text start $max])

	if test $neg_cstack_address -eq 1 ; then #FIXME test this
  	    if test $cstack_bits -lt $max ; then
		max=$cstack_bits;
		AC_MSG_NOTICE([max log text start reduced to $max considering c stack address])
	    fi
	fi

	# Need to get over default cstack and related maps between 0x1000000 and 0x2000000
	if test "$use" == "386-gnu" || test "$use" == "amd64-gnu" ; then
	   if test "$use" == "386-gnu" ; then
	      q=24;
	   else
	      q=23;
	   fi
	   if test $min -lt $q && test $q -lt $max && test $q -lt 30; then
	      min=$q
	      AC_MSG_NOTICE([min log text start increased to $min over c stack address])
	   fi
	fi


	j=-1;
	low_shft="";
	if test $min -le $max ; then
        if test $max -ge $enable_fastimmfix && test "$enable_immfix" != "no" ; then
	        j=`$AWK 'END {for (i=j=0;j<=n;j++) i=i ? i*2 : 1;printf("%x\n",3*i)}' n=$max </dev/null`
		low_shft=`$AWK 'END {print n+1}' n=$max </dev/null`;
		AC_MSG_NOTICE([raising log text to 0x$j for a $max bit wide low immfix table])
        else
	        j=`$AWK 'END {for (i=j=0;j<=n;j++) i=i ? i*2 : 1;printf("%x\n",3*i)}' n=$min </dev/null`
		AC_MSG_NOTICE([lowering log text to 0x$j to maximize data area])
	    fi
	fi

	if test "$low_shft" != "" ; then
        AC_DEFINE_UNQUOTED(LOW_SHFT,$low_shft,[upper immediate fixnum bound])
        AC_DEFINE_UNQUOTED(OBJNULL,(object)0x$j,[lowest address non-object])
	else 
        AC_DEFINE_UNQUOTED(OBJNULL,NULL,[lowest address non-object])
	fi 

	#      echo $j;
	AC_MSG_CHECKING([our linker script])
	if test "$j" != "-1" ; then
        cat gcl.script.def | $AWK '/SEGMENT_START/ {gsub("0x[[0-9]]*","0x" j,$0);} {print}' j=$j >gcl.script
	    AC_MSG_RESULT([done])
	    rm -f gcl.script.def
	    assert_arg_to_ldflags -Wl,-T,gcl.script
	    cp gcl.script unixport
	else
	    AC_MSG_RESULT([none found or not needed])
	    rm -f gcl.script gcl.script.def
	fi
	rm -rf foo.c foo
else
    AC_DEFINE_UNQUOTED(OBJNULL,NULL,[lowest address non-object])
    AC_MSG_RESULT([not found])
fi

#else

#    AC_DEFINE_UNQUOTED(OBJNULL,NULL,[lowest address non-object])

#fi
AC_DEFINE_UNQUOTED(CSSIZE,$enable_cssize,[maximum C stack size])

mem_top=0
mem_range=0
AC_MSG_CHECKING(mem top)
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
	    ]],
	    [[
		unsigned long i,j,k,l;
		FILE *fp = fopen("conftest1","w");

		for (i=2,k=1;i;k=i,i<<=1);
		l=$cstack_top;
		for (i=j=k;j && i<l;j>>=1,i|=j);
		if (j<(k>>3)) i=0;
		j=1;
		j<<=$PAGEWIDTH;
		j<<=4;
		j--;
		i+=j;
		i&=~j;
		fprintf(fp,"0x%lx",i);
		fclose(fp);
		return 0;
	    ]])],
    [mem_top=`cat conftest1`],[mem_top="0x0"])
AC_MSG_RESULT($mem_top)

if test "$mem_top" != "0x0" ; then
    AC_MSG_CHECKING(finding upper mem half range)
    AC_RUN_IFELSE(
	[AC_LANG_PROGRAM(
		[[
		    #include <stdio.h>
		]],
		[[
		    unsigned long j;
		    FILE *fp = fopen("conftest1","w");
 
		    for (j=1;j && !(j& $mem_top);j<<=1);
			fprintf(fp,"0x%lx",j>>1);
			fclose(fp);
			return 0;
		]])],
	[mem_range=`cat conftest1`],[mem_range="0x0"])
    AC_MSG_RESULT($mem_range)
fi

if test "$enable_immfix" != "no" ; then
    if test "$mem_top" != "0x0" ; then
	if test "$mem_range" != "0x0" ; then
	    AC_DEFINE_UNQUOTED(IM_FIX_BASE,${mem_top}UL,[beginning address for immediate fixnum range])
            AC_DEFINE_UNQUOTED(IM_FIX_LIM,${mem_range}UL,[size of immediate fixnum address space])
	fi
    fi
fi

AC_MSG_CHECKING([sizeof long long int])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
	    ]],
	    [[
		if (sizeof(long long int) == 2*sizeof(long)) return 0;
		   return 1;
	    ]])],
    [AC_DEFINE(HAVE_LONG_LONG,1,[long long is available]) AC_MSG_RESULT(yes)],
    [AC_MSG_RESULT(no)])

AC_SUBST(HAVE_LONG_LONG)

AC_CHECK_HEADERS([dirent.h],
		 AC_MSG_CHECKING([for d_type])
		 AC_RUN_IFELSE(
		     [AC_LANG_PROGRAM(
			     [[
				 #include <dirent.h>
				 #include <string.h>
			     ]],
			     [[
				 struct dirent *d;
				 DIR *r=opendir("./");
				 for (;(d=readdir(r)) && strcmp("config.log",d->d_name););
				 return d && d->d_type==DT_REG ? 0 : -1;
			     ]])],
		     [AC_MSG_RESULT([yes]) AC_DEFINE(HAVE_D_TYPE,1,[have struct dirent d_type field])],
		     AC_MSG_RESULT([no]),AC_MSG_RESULT([no])))


# Check if Posix compliant getcwd exists, if not we'll use getwd.
AC_CHECK_FUNCS(getcwd)
AC_CHECK_FUNCS(getwd)
AC_CHECK_FUNCS(rename)
AC_CHECK_FUNC(uname, , AC_DEFINE(NO_UNAME,1,[no uname call]))

AC_CHECK_FUNCS(readlinkat,[],[AC_MSG_ERROR([must have readlinkat])])

AC_CHECK_HEADERS(sys/ioctl.h)

# OpenBSD has elf_abi.h instead of elf.h
AC_CHECK_HEADERS(elf.h elf_abi.h)

AC_CHECK_HEADERS(sys/sockio.h)


if test "$use" != "mingw" ; then
   AC_CHECK_LIB(m,sin,LIBS="${LIBS} -lm",true)
fi
AC_CHECK_LIB(mingwex,main,LIBS="${LIBS} -lmingwex",true)


#
# For DBL_MAX et. al. on (only) certain Linux arches, apparently CM
#
AC_CHECK_HEADERS(values.h,AC_DEFINE(HAVE_VALUES_H,1,[have values.h]))

#
# Sparc solaris keeps this in float.h, rework either/or with values.h later
#
AC_CHECK_HEADERS(float.h,AC_DEFINE(HAVE_FLOAT_H,1,[have float.h]))

#
# The second alternative is for solaris.  This needs to be
# a more comprehensive later, i.e. checking that the fpclass
# test makes sense.  CM
#
AC_MSG_CHECKING([for isnormal])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#define _GNU_SOURCE
		#include <math.h>
	    ]],
	    [[
		float f;
		return isnormal(f) || !isnormal(f) ? 0 : 1;
	    ]])],
    [AC_DEFINE(HAVE_ISNORMAL,1,[Have isnormal function]) AC_MSG_RESULT(yes)],
    [AC_MSG_CHECKING([for fpclass of ieeefp.h])
     AC_RUN_IFELSE(
	 [AC_LANG_PROGRAM(
		 [[
		     #include <ieeefp.h>
		 ]],
		 [[
		     float f;
		     return fpclass(f)>=FP_NZERO || fpclass(f)<FP_NZERO ? 0 : 1;
		 ]])],[AC_DEFINE(HAVE_IEEEFP,1,[Have ieeefp fpclass function]) AC_MSG_RESULT(yes)],
	 [AC_MSG_RESULT(no)])])

AC_MSG_CHECKING([for isfinite])
AC_RUN_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#define _GNU_SOURCE
		#include <math.h>
	    ]],
	    [[
		float f;
		return isfinite(f) || !isfinite(f) ? 0 : 1;
	    ]])],[AC_DEFINE(HAVE_ISFINITE,1,[Have isfinite function]) AC_MSG_RESULT(yes)],
    [AC_MSG_CHECKING([for finite()])
     AC_RUN_IFELSE(
	 [AC_LANG_PROGRAM(
		 [[
		     #include <math.h>
		     #include <ieeefp.h>
		 ]],
		 [[
		     float f;
		     return finite(f) || !finite(f) ? 0 : 1;
		 ]])],
	 [AC_DEFINE(HAVE_FINITE,1,[Have finite function]) AC_MSG_RESULT(yes)],
   	 [AC_MSG_ERROR(no)])])

#--------------------------------------------------------------------
#	Check for the existence of the -lsocket and -lnsl libraries.
#	The order here is important, so that they end up in the right
#	order in the command line generated by make.  Here are some
#	special considerations:
#	1. Use "connect" and "accept" to check for -lsocket, and
#	   "gethostbyname" to check for -lnsl.
#	2. Use each function name only once:  can't redo a check because
#	   autoconf caches the results of the last check and won't redo it.
#	3. Use -lnsl and -lsocket only if they supply procedures that
#	   aren't already present in the normal libraries.  This is because
#	   IRIX 5.2 has libraries, but they aren't needed and they're
#	   bogus:  they goof up name resolution if used.
#	4. On some SVR4 systems, can't use -lsocket without -lnsl too.
#	   To get around this problem, check for both libraries together
#	   if -lsocket doesn't work by itself.
#--------------------------------------------------------------------
AC_MSG_CHECKING([for sockets])
tcl_checkBoth=0
AC_CHECK_FUNC(connect, tcl_checkSocket=0, tcl_checkSocket=1)
if test "$tcl_checkSocket" = 1; then
    AC_CHECK_LIB(socket, main, TLIBS="$TLIBS -lsocket", tcl_checkBoth=1)
fi


if test "$tcl_checkBoth" = 1; then
    tk_oldLibs=$TLIBS
    TLIBS="$TLIBS -lsocket -lnsl"
    AC_CHECK_FUNC(accept, tcl_checkNsl=0, [TLIBS=$tk_oldLibs])
fi
AC_CHECK_FUNC(gethostbyname, , AC_CHECK_LIB(nsl, main, [TLIBS="$TLIBS -lnsl"]))

# readline
AC_ARG_ENABLE(readline,[  --enable-readline    enables command line completion via the readline library ])

if test "$use" = "mingw" ; then
    enable_readline=no
fi

if test "$enable_readline" != "no" ; then
    AC_CHECK_HEADERS([readline/readline.h],
			AC_CHECK_LIB([readline],[rl_initialize],
				[AC_DEFINE(USE_READLINE,1,[use readline library])
				 AC_CHECK_LIB([readline],[el_getc],AC_DEFINE(READLINE_IS_EDITLINE,1,[readline is editline]))
				 # These tests discover differences between readline 4.1 and 4.3
				 AC_CHECK_LIB([readline],[rl_completion_matches],
					[AC_DEFINE(HAVE_DECL_RL_COMPLETION_MATCHES,1,[have readline completion matches])
				 	 AC_DEFINE(HAVE_RL_COMPENTRY_FUNC_T,1,[have readline completion matches])])
			         AC_MSG_CHECKING([RL_COMPLETION_ENTRY_FUNCTION_TYPE_FUNCTION])
                                 AC_COMPILE_IFELSE(
                                        [AC_LANG_PROGRAM(
	                                       [[
                                                  #include <stdio.h>
                                                  #include <readline/readline.h>
						  extern Function *rl_completion_entry_function __attribute__((weak));
	    				       ]],
					       [[]])],
                                        [AC_DEFINE(RL_COMPLETION_ENTRY_FUNCTION_TYPE_FUNCTION,1,[rl_completion_entry_function returns type Function])
                                         AC_MSG_RESULT([yes])],
                                        [AC_MSG_RESULT([no])
                                         AC_MSG_CHECKING([RL_COMPLETION_ENTRY_FUNCTION_TYPE_RL_COMPENTRY_FUNC_T])
                                         AC_COMPILE_IFELSE(
                                                [AC_LANG_PROGRAM(
	                                               [[
						          #include <stdio.h>
		                                          #include <readline/readline.h>
							  extern rl_compentry_func_t *rl_completion_entry_function __attribute__((weak));
	                                               ]],
						       [[]])],
	                                        [AC_DEFINE(RL_COMPLETION_ENTRY_FUNCTION_TYPE_RL_COMPENTRY_FUNC_T,1,[rl_completion_entry_function returns type rl_compentry_func_t])
	                                         AC_MSG_RESULT([yes])],
                                                [AC_MSG_RESULT([no])
	                                         AC_MSG_ERROR([Unknown rl_completion_entry_function return type])])])

                                 AC_MSG_CHECKING([RL_READLINE_NAME_TYPE_CHAR])
                                 AC_COMPILE_IFELSE(
                                        [AC_LANG_PROGRAM(
	                                       [[
					          #include <stdio.h>
		                                  #include <readline/readline.h>
		                                  extern char *rl_readline_name __attribute__((weak));
	                                       ]],
	                                       [[]])],
                                        [AC_DEFINE(RL_READLINE_NAME_TYPE_CHAR,1,[rl_readline_name returns type char])
                                         AC_MSG_RESULT([yes])],
                                        [AC_MSG_RESULT([no])
                                         AC_MSG_CHECKING([RL_READLINE_NAME_TYPE_CONST_CHAR])
                                         AC_COMPILE_IFELSE(
                                                [AC_LANG_PROGRAM(
	                                               [[
						          #include <stdio.h>
		                                          #include <readline/readline.h>
		                                          extern const char *rl_readline_name __attribute__((weak));
	                                               ]],
	                                               [[]])],
                                                [AC_DEFINE(RL_READLINE_NAME_TYPE_CONST_CHAR,1,[rl_readline_name returns type const char])
                                                 AC_MSG_RESULT([yes])],
                                                [AC_MSG_RESULT([no])
						 AC_MSG_ERROR([Unknown rl_readline_name return type])])])
			 	 TLIBS="$TLIBS -lreadline" #some machines don't link this, e.g. Slackware
				 RL_OBJS=gcl_readline]),
				 [],[AC_INCLUDES_DEFAULT([#include <stdio.h>])])
fi

AC_SUBST(RL_OBJS)
AC_SUBST(RL_LIB)

# sockets

AC_MSG_CHECKING([For network code for nsocket.c])
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <sys/time.h>
		#include <sys/types.h>
		#include <unistd.h>

		#include <errno.h>
		#include <fcntl.h>
		#include <stdio.h>

		/************* for the sockets ******************/
		#include <sys/socket.h>		/* struct sockaddr, SOCK_STREAM, ... */
		#ifndef NO_UNAME
		#   include <sys/utsname.h>	/* uname system call. */
		#endif
		#include <netinet/in.h>		/* struct in_addr, struct sockaddr_in */
		#include <arpa/inet.h>		/* inet_ntoa() */
		#include <netdb.h>		/* gethostbyname() */
	    ]],
	    [[
		connect(0,(struct sockaddr *)0,0);
		gethostbyname("jil");
		socket(AF_INET, SOCK_STREAM, 0);
	    ]])],
    [AC_DEFINE(HAVE_NSOCKET,1,[can use nsocket library])
     AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])])


AC_MSG_CHECKING([check for listen using fcntl])
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <stdio.h>
		#include <fcntl.h>
	    ]],
	    [[
		FILE *fp=fopen("configure.in","r");
		int orig;
		orig = fcntl(fileno(fp), F_GETFL);
		if (! (orig & O_NONBLOCK )) return 0;
	    ]])],
    [AC_DEFINE(LISTEN_USE_FCNTL,1,[can use fcntl for listen function])
     AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])])


AC_CHECK_FUNC(profil, ,[AC_DEFINE(NO_PROFILE,1,[no profil system call])])
AC_SUBST(NO_PROFILE)
AC_CHECK_FUNC(setenv,[AC_DEFINE(HAVE_SETENV,1,[have setenv call])],no_setenv=1 )
AC_SUBST(HAVE_SETENV)
if test "$no_setenv" = "1" ; then
    AC_CHECK_FUNC(putenv,[AC_DEFINE(HAVE_PUTENV,1,[have putenv call])],)
    AC_SUBST(HAVE_PUTENV)
fi

AC_CHECK_FUNC(_cleanup, [AC_DEFINE(USE_CLEANUP,1,[have _cleanup function])],)
AC_SUBST(USE_CLEANUP)

AC_MSG_CHECKING([FIONBIO vs. O_NONBLOCK for nonblocking I/O])

case $system in
    OSF*)
	AC_DEFINE(USE_FIONBIO,1,[use fionbio for non-blocking io])
	AC_MSG_RESULT(FIONBIO)
	;;
    SunOS-4*)
	AC_DEFINE(USE_FIONBIO,1,[use fionbio for non-blocking io])
	AC_MSG_RESULT(FIONBIO)
	;;
    ULTRIX-4.*)
	AC_DEFINE(USE_FIONBIO,1,[use fionbio for non-blocking io])
	AC_MSG_RESULT(FIONBIO)
	;;
    *)
	AC_MSG_RESULT(O_NONBLOCK)
	;;
esac


AC_MSG_CHECKING(check for SV_ONSTACK)
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <signal.h>
		int joe=SV_ONSTACK;
	    ]],
	    [[]])],
    [AC_DEFINE(HAVE_SV_ONSTACK,1,[have sv_onstack])
     AC_SUBST(HAVE_SV_ONSTACK)
     AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])])

AC_MSG_CHECKING(check for SIGSYS)
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <signal.h>
		int joe=SIGSYS;
	    ]],[[]])],
    [AC_DEFINE(HAVE_SIGSYS,1,[have SIGSYS signal])
     AC_SUBST(HAVE_SIGSYS)
     AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])])


AC_MSG_CHECKING(check for SIGEMT)
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM(
	    [[
		#include <signal.h>
		int joe=SIGEMT;
	    ]],[[]])],
    [AC_DEFINE(HAVE_SIGEMT,1,[have SIGEMT signal])
     AC_SUBST(HAVE_SIGEMT)
     AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])])

AC_CHECK_FUNCS(sigaltstack)
AC_CHECK_FUNCS(feenableexcept)

AC_CHECK_HEADERS(dis-asm.h,
		 MLIBS=$LIBS
		 AC_CHECK_LIB(opcodes,init_disassemble_info)
		 AC_CHECK_LIB(dl,dlopen,#opcodes changes too quickly to link directly
			      AC_CHECK_FUNCS(print_insn_i386,LIBS="$MLIBS -ldl")))


AC_ARG_ENABLE([tcltk],[  --enable-tcltk will try to build gcl-tk])
AC_ARG_ENABLE([tkconfig],
	      [  --enable-tkconfig=XXXX will force the use of a TK_CONFIG_PREFIX=XXXXX as place to look for tkConfig.sh and tclConfig.sh],
	      [TK_CONFIG_PREFIX=$enableval],)
AC_ARG_ENABLE([tclconfig],
	      [  --enable-tclconfig=XXXX will force the use of a TCL_CONFIG_PREFIX=XXXXX as place to look for tclConfig.sh and tclConfig.sh],
	      [TCL_CONFIG_PREFIX=$enableval],)

if test "$enable_tcltk" != "no" ; then

    if test -d "${TCL_CONFIG_PREFIX}"  ; then true ; else

	AC_CHECK_PROG(TCLSH,tclsh,tclsh,${TCLSH})

	if test "${TCLSH}" = "" ; then true ; else
	    TCL_VERSION=`echo '[puts [set tcl_version]]' | ${TCLSH}`
	fi

	if test -e /usr/lib/tcl$TCL_VERSION/tclConfig.sh ; then
	    TCL_CONFIG_PREFIX=/usr/lib/tcl$TCL_VERSION
	fi

    fi

    if test -e ${TCL_CONFIG_PREFIX}/tclConfig.sh  ; then . ${TCL_CONFIG_PREFIX}/tclConfig.sh ; fi

    if test -d "${TK_CONFIG_PREFIX}"  ; then true ; else
	if test -e ${TCL_CONFIG_PREFIX}/tkConfig.sh ; then
	    TK_CONFIG_PREFIX=${TCL_CONFIG_PREFIX}
	else
	    if test -e `echo $TCL_CONFIG_PREFIX | sed 's,tcl,tk,g'`/tkConfig.sh ; then
		TK_CONFIG_PREFIX=`echo $TCL_CONFIG_PREFIX | sed 's,tcl,tk,g'`
	    fi
	fi
    fi

    if test -e ${TK_CONFIG_PREFIX}/tkConfig.sh  ; then . ${TK_CONFIG_PREFIX}/tkConfig.sh ; fi

    if test -d ${TCL_CONFIG_PREFIX}/tcl${TCL_VERSION} ; then
	TCL_LIBRARY=${TCL_CONFIG_PREFIX}/tcl${TCL_VERSION}
    else
	if test -d ${TCL_CONFIG_PREFIX}/../tcl${TCL_VERSION} ; then
	    TCL_LIBRARY=${TCL_CONFIG_PREFIX}/../tcl${TCL_VERSION}
	fi
    fi

    if test -d ${TK_CONFIG_PREFIX}/tk${TK_VERSION} ; then
	TK_LIBRARY=${TK_CONFIG_PREFIX}/tk${TK_VERSION}
    else
	if test -d ${TK_CONFIG_PREFIX}/../tk${TK_VERSION} ; then
	    TK_LIBRARY=${TK_CONFIG_PREFIX}/../tk${TK_VERSION}
	fi
    fi

    if test -e ${TCL_CONFIG_PREFIX}/../include/tcl.h ; then
	TCL_INCLUDE=-I${TCL_CONFIG_PREFIX}/../include
    else
	if test -e /usr/include/tcl${TCL_VERSION}/tcl.h ; then
	    TCL_INCLUDE=-I/usr/include/tcl${TCL_VERSION}
	fi
    fi

    if test -e ${TK_CONFIG_PREFIX}/../include/tk.h ; then
	TK_INCLUDE=-I${TK_CONFIG_PREFIX}/../include
    else
	if test -e /usr/include/tcl${TCL_VERSION}/tk.h ; then
	    TK_INCLUDE=-I/usr/include/tcl${TCL_VERSION}
	fi
    fi

fi
AM_CONDITIONAL([AMM_TK],[test "$TK_CONFIG_PREFIX" != ""])


AC_MSG_CHECKING([for tcl/tk])
if test -d "${TK_CONFIG_PREFIX}" ; then
    AC_MSG_RESULT([using TK_VERSION=${TK_VERSION} of ${TK_CONFIG_PREFIX}])
    AC_SUBST(TK_CONFIG_PREFIX)
    AC_SUBST(TK_XLIB_DIR)
    AC_SUBST(TK_LIBRARY)
    AC_SUBST(TCL_LIBRARY)
    AC_SUBST(TK_INCLUDE)
    AC_SUBST(TCL_INCLUDE)
    AC_SUBST(TK_LIB_SPEC)
    AC_SUBST(TCL_LIB_SPEC)
    AC_CONFIG_FILES([gcl-tk/gcltksrv])
else
    AC_MSG_RESULT([not found])
fi

AC_SUBST(EXT)
AC_CONFIG_FILES([bin/gcl])


AC_CHECK_HEADERS(sys/mman.h,AC_CHECK_FUNCS(mprotect))
AC_CHECK_HEADERS(alloca.h)
AC_FUNC_ALLOCA


#LDFLAGS="`echo $GPL_FLAG $LDFLAGS | sed 's,gcl.script,../unixport/gcl.script,g'`"
LDFLAGS="`echo $GPL_FLAG $LDFLAGS`"
#AM_LDFLAGS = $LDFLAGS
BASE_LDFLAGS="$LDFLAGS"
LDFLAGS=""
AC_SUBST(BASE_LDFLAGS)
LIBS="$X_PRE_LIBS $X_LIBS $X_EXTRA_LIBS $LIBS $TLIBS"
#AM_LIBS = $LIBS
AC_SUBST(LIBS)
CFLAGS="$CFLAGS $GP_FLAG"
FINAL_CFLAGS="$CFLAGS $CPPFLAGS $X_CFLAGS"
AC_SUBST(FINAL_CFLAGS)
# Work around bug with gcc on ppc -- CM
NIFLAGS="$CFLAGS $CPPFLAGS $TONIFLAGS -I o"
AC_SUBST(NIFLAGS)
CFLAGS="$CFLAGS $CPPFLAGS $TO3FLAGS -I o"
BASE_CFLAGS="$CFLAGS"
CFLAGS=""
AC_SUBST(BASE_CFLAGS)
BASE_CPPFLAGS="-I h -I /usr/include/tirpc $CPPFLAGS"
CPPFLAGS=
AC_SUBST(BASE_CPPFLAGS)
O3FLAGS=$TOSFLAGS
AC_SUBST(O3FLAGS)
O2FLAGS=$TO2FLAGS
AC_SUBST(O2FLAGS)

AC_SUBST(EXTRA_LOBJS)
AC_SUBST(LEADING_UNDERSCORE)
AC_SUBST(GNU_LD)

LI_EXTVERS=`echo $MINVERS | cut -f2 -d.`
AC_SUBST(LI_EXTVERS)
LI_MINVERS=`echo $MINVERS | cut -f1 -d.`
AC_SUBST(LI_MINVERS)
LI_MAJVERS=$MAJVERS
AC_SUBST(LI_MAJVERS)
LI_GITTAG="$GIT_TAG"
AC_SUBST(LI_GITTAG)
LI_RELEASE="$RELEASE"
AC_SUBST(LI_RELEASE)
LI_CC="\"$GCL_CC -c `echo " $FINAL_CFLAGS" | sed 's,-pg\b,,g'`\""
AC_SUBST(LI_CC)
LI_DFP="\"$GPL_FLAG\""
AC_SUBST(LI_DFP)
LI_LD="\"$GCL_CC $BASE_LDFLAGS -o\""
AC_SUBST(LI_LD)
LI_LD_LIBS="\"$LIBS\""
AC_SUBST(LI_LD_LIBS)
LI_OPT_THREE="\"$O3FLAGS\""
AC_SUBST(LI_OPT_THREE)
LI_OPT_TWO="\"$O2FLAGS\""
AC_SUBST(LI_OPT_TWO)
LI_INIT_LSP="\"init_raw.lsp\""
AC_SUBST(LI_INIT_LSP)

AC_CONFIG_FILES([unixport/init_raw.lsp])

if test "$use" != ""  ; then

    AC_OUTPUT
    cmp $srcdir/h/$use.h h/config.h || cp $srcdir/h/$use.h h/config.h
    echo configuration for $use done

else
    echo "Unable to guess machine type"
    echo use configure --enable-machine=XXX,  for XXX such that h/XXX.defs exists, ie in h/*.defs
    exit 1
fi
