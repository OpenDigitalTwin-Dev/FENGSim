; Copyright (c) 1994 William F. Schelter
; Copyright (c) 2024 Camm Maguire

; See the files gnu.license and dec.copyright .

; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 1, or (at your option)
; any later version.

; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.

; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

; Some of the files that interface to the Xlib are adapted from DEC/MIT files.
; See the file dec.copyright for details.

(unless (find-package :xlib) (make-package :XLIB :use '(:lisp :si)));(load "package.lisp")
(in-package :XLIB)

(defvar *files* '( "gcl_Xlib"
      "gcl_Xutil"
      "gcl_X"
      "gcl_XAtom"
      "gcl_defentry_events"
      "gcl_Xstruct"
      "gcl_XStruct_l_3"
      "gcl_general"
      "gcl_keysymdef"
      "gcl_X10"
      "gcl_Xinit"
      "gcl_dwtrans"
      "gcl_tohtml"
      "gcl_index"
;      "gcl_sysinit"
      ))


(defun compile-xgcl()
  #+(or m68k sh4)
  (progn (trace si::readdir si::opendir si::closedir si::pathname-match-p)
	 (print (directory "*.c"))
	 (untrace si::readdir si::opendir si::closedir si::pathname-match-p))
  (mapc (lambda (x) 
	  (let ((x (concatenate 'string compiler::*cc* " -I../h " (namestring x))))
	    (unless (zerop (system x))
	      (error "compile failure: ~s~%" x))))
	(or (directory "*.c")
	    #+(or m68k sh4)
	    (progn (print "qemu/readdir issue still present")
		   (mapcar (lambda (x) (truename (merge-pathnames ".c" x))) '("XStruct-4" "general-c" "Xutil-2" "Events" "XStruct-2")))))
  (let ((compiler::*default-c-file* t)
	(compiler::*default-h-file* t)
	(compiler::*default-data-file* t)
	(compiler::*default-system-p* t))
    (mapc (lambda (x)
	    (compile-file (format nil "~a.lsp" x) :system-p t))
	  *files*)))


(defun load-xgcl()
  (mapcar (lambda (x) (load (format nil "~a.o" x))) *files*))

(defun load-xgcl-interp()
  (mapcar (lambda (x) (load (format nil "~a.lsp" x))) *files*))

(defun save-xgcl (pn)
  (let* ((x (mapcar (lambda (x) (probe-file (concatenate 'string x ".o"))) *files*))
	 (y (directory "*.o"))
	 (z (set-difference y x :test 'equal)))
    (compiler::link x (namestring pn) (format nil "(load ~s)(mapc 'load '~s)" "sysdef.lisp" x)
		    (reduce (lambda (&rest xy) (when xy (concatenate 'string (namestring (car xy)) " " (cadr xy)))) z
			    :initial-value " -lXmu -lXt -lXext -lXaw -lX11" :from-end t)
		    nil)))









