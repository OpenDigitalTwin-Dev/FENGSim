Summary: dialog - display dialog boxes from shell scripts
%define AppProgram dialog
%define AppVersion 1.3
%define AppRelease 20240619
%define ActualProg c%{AppProgram}
# $XTermId: dialog.spec,v 1.193 2024/06/19 16:57:51 tom Exp $
Name: %{ActualProg}
Version: %{AppVersion}
Release: %{AppRelease}
License: LGPL
Group: Applications/System
URL: https://invisible-island.net/%{AppProgram}
Source0: https://invisible-island.net/archives/%{AppProgram}-%{AppVersion}-%{AppRelease}.tgz
Packager: Thomas Dickey <dickey@invisible-island.net>

%package devel
Summary: Development headers/library for the dialog package.
Requires: %{ActualProg}, ncurses-devel

%description
Dialog is a program that will let you present a variety of questions or
display messages using dialog boxes from a shell script.  These types
of dialog boxes are implemented (though not all are necessarily compiled
into dialog):

     buildlist, calendar, checklist, dselect, editbox, form, fselect,
     gauge, infobox, inputbox, inputmenu, menu, mixedform,
     mixedgauge, msgbox (message), passwordbox, passwordform, pause,
     prgbox, programbox, progressbox, radiolist, rangebox, tailbox,
     tailboxbg, textbox, timebox, treeview, and yesno (yes/no).

This package installs as "cdialog" to avoid conflict with other packages.

%description devel
This is the development package "cdialog", which includes the header files,
the linkage information and library documentation.
%prep

%define debug_package %{nil}

%setup -q -n %{AppProgram}-%{AppVersion}-%{AppRelease}

%build

cp -v package/dialog.map package/%{ActualProg}.map

INSTALL_PROGRAM='${INSTALL}' \
%configure \
  --target %{_target_platform} \
  --prefix=%{_prefix} \
  --bindir=%{_bindir} \
  --libdir=%{_libdir} \
  --mandir=%{_mandir} \
  --with-package=%{ActualProg} \
  --enable-header-subdir \
  --enable-nls \
  --enable-pc-files \
  --enable-stdnoreturn \
  --enable-widec \
  --with-shared \
  --with-ncursesw \
  --with-shlib-version=abi \
  --with-versioned-syms \
  --disable-rpath-hack

make

%install
[ "$RPM_BUILD_ROOT" != "/" ] && rm -rf $RPM_BUILD_ROOT

make install          DESTDIR=$RPM_BUILD_ROOT
make install-full     DESTDIR=$RPM_BUILD_ROOT
make install-examples DESTDIR=$RPM_BUILD_ROOT

strip $RPM_BUILD_ROOT%{_bindir}/%{ActualProg}
chmod 755 $RPM_BUILD_ROOT%{_libdir}/lib%{ActualProg}.so.*

%files
%defattr(-,root,root)
%{_bindir}/%{ActualProg}
%{_mandir}/man1/%{ActualProg}.*
%{_libdir}/lib%{ActualProg}.so.*
%{_datadir}/locale/*/LC_MESSAGES/%{ActualProg}.mo 
%doc README COPYING CHANGES
%{_datadir}/doc/%{ActualProg}/*

%files devel
%defattr(-,root,root)
%{_bindir}/%{ActualProg}-config
%{_includedir}/%{ActualProg}.h
%{_includedir}/%{ActualProg}/dlg_colors.h
%{_includedir}/%{ActualProg}/dlg_config.h
%{_includedir}/%{ActualProg}/dlg_keys.h
%{_libdir}/lib%{ActualProg}.so
%{_libdir}/pkgconfig/%{ActualProg}.pc
%{_mandir}/man3/%{ActualProg}.*

%changelog
# each patch should add its ChangeLog entries here

* Thu Feb 09 2023 Thomas Dickey
- add ".pc" file

* Tue Feb 07 2023 Thomas Dickey
- install examples in doc directory

* Wed Feb 01 2023 Thomas Dickey
- change shared-library configuration to ABI

* Wed Mar 24 2021 Thomas Dickey
- use C11 _Noreturn

* Wed Jul 24 2019 Thomas Dickey
- split-out "-devel" package

* Sat Dec 09 2017 Thomas Dickey
- update ftp url

* Thu Apr 21 2016 Thomas Dickey
- remove stray call to libtool

* Tue Oct 18 2011 Thomas Dickey
- add executable permissions for shared libraries, discard ".la" file.

* Thu Dec 30 2010 Thomas Dickey
- initial version
