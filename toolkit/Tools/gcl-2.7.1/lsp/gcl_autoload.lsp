;; Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
;; Copyright (C) 2024 Camm Maguire

;; This file is part of GNU Common Lisp, herein referred to as GCL
;;
;; GCL is free software; you can redistribute it and/or modify it under
;;  the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
;; the Free Software Foundation; either version 2, or (at your option)
;; any later version.
;; 
;; GCL is distributed in the hope that it will be useful, but WITHOUT
;; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
;; FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
;; License for more details.
;; 
;; You should have received a copy of the GNU Library General Public License 
;; along with GCL; see the file COPYING.  If not, write to the Free Software
;; Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.



;;;;    AUTOLOAD


(in-package :si)

(export '(clines defentry defcfun)); defla


(defun lisp-implementation-type nil "GNU Common Lisp (GCL)")

(defun machine-type nil nil)

(defun machine-version nil (machine-type))

(defun machine-instance nil (machine-type))

(defun software-version nil nil)

(defun software-version nil (software-type))

(defun short-site-name nil nil)

(defun long-site-name nil nil)


;;; Compiler functions.

(defun proclaim (d)
       (when (eq (car d) 'special) (mapc #'si:*make-special (cdr d))))

(defun proclamation (d)
  (and (eq (car d) 'special)
       (dolist (var (cdr d) t)
               (unless (si:specialp var) (return nil)))))

(defun compile-file (&rest args)
       (error "COMPILE-FILE is not defined in this load module."))
(defun compile (&rest args)
       (error "COMPILE is not defined in this load module."))
(defun disassemble (&rest args)
       (error "DISASSEMBLE is not defined in this load module."))


;;; Editor.

;
(defun get-decoded-time nil
  (decode-universal-time (get-universal-time)))


; System dependent Temporary directory.
(defun temp-dir nil
  "A system dependent path to a temporary storage directory as a string." 
  (si::getenv "TEMP"))

;  Set the default system editor to a fairly certain bet.
(defvar *gcl-editor* "vi")
;; #+winnt(defvar *gcl-editor* "notepad")

(defun new-ed (editor-name)
  "Change the editor called by (ed) held in *gcl-editor*."
  (setf *gcl-editor* editor-name))

(defun ed (&optional name)
  "Edit a file using the editor named in *gcl-editor*; customise with new-ed()."
  (if (null name)
      (system *gcl-editor*)
    (cond ((stringp name) 
	   (system (format nil "~A ~A" *gcl-editor* name))) ; If string, assume file name.
	  ((pathnamep name)
	   (system (format nil "~A ~A" *gcl-editor* (namestring name)))) ; If pathname.
	  (t 
	   (let ((body (symbol-function name)))
	     (cond ((compiled-function-p body) (error "You can't edit compiled functions."))
		   ((and body
			 (consp body)
			 (eq (car body) 'lambda-block)) ; If lambda block, save file and edit.
		    (let ((ed-file (concatenate 'string
						(temp-dir)
						(format nil "~A" (cadr body))
						".lisp")))
		      (with-open-file
		       (st ed-file :direction :output)
		       (print `(defun ,name ,@ (cddr body)) st))
		      (system (format nil "~A ~A" *gcl-editor* ed-file))))
		   (t (system (format nil "~A ~A" *gcl-editor* name))))))))) ; Use symbol as filename

;;; C Interface.

(defmacro Clines (&rest r) (declare (ignore r)) nil)
(defmacro defCfun (&rest r) (declare (ignore r)) nil)
(defmacro defentry (&rest r) (declare (ignore r)) nil)

(defmacro defla (&rest r) (cons 'defun r))

;;; Help.

(defun user::help (&optional (symbol nil s))
  (if s (print-doc symbol)
      (progn
        (princ "
Welcome to GNU Common Lisp (GCL for short).
Here are some functions you should learn first.

	(HELP symbol) prints the online documentation associated with the
	symbol.  For example, (HELP 'CONS) will print the useful information
	about the CONS function, the CONS data type, and so on.

	(HELP* string) prints the online documentation associated with those
	symbols whose print-names have the string as substring.  For example,
	(HELP* \"PROG\") will print the documentation of the symbols such as
	PROG, PROGN, and MULTIPLE-VALUE-PROG1.

	(SI::INFO <some string>) chooses from a list of all references in the
        on-line documentation to <some string>.

	(APROPOS <some string>) or (APROPOS <some string> '<a package>) list
        all symbols containing <some string>.

	(DESCRIBE '<symbol>) or (HELP '<symbol>) describe particular symbols.

	(XGCL-DEMO) will demo the xgcl interface if installed.

	(GCL-TK-DEMO) will demo the gcl-tk interface if installed.

	(BYE) or (BY) ends the current GCL session.

Good luck!				 The GCL Development Team")
        (values))))

(defun user::help* (string &optional (package (find-package "LISP")))
  (apropos-doc string package))

;;; Pretty-print-formats.
;;;
;;;	The number N as the property of a symbol SYMBOL indicates that,
;;;	in the form (SYMBOL f1 ... fN fN+1 ... fM), the subforms fN+1,...,fM
;;;	are the 'body' of the form and thus are treated in a special way by
;;;	the KCL pretty-printer.

;; (setf (get 'lambda 'si:pretty-print-format) 1)
;; (setf (get 'lambda-block 'si:pretty-print-format) 2)
;; (setf (get 'lambda-closure 'si:pretty-print-format) 4)
;; (setf (get 'lambda-block-closure 'si:pretty-print-format) 5)

;; (setf (get 'block 'si:pretty-print-format) 1)
;; (setf (get 'case 'si:pretty-print-format) 1)
;; (setf (get 'catch 'si:pretty-print-format) 1)
;; (setf (get 'ccase 'si:pretty-print-format) 1)
;; (setf (get 'clines 'si:pretty-print-format) 0)
;; (setf (get 'compiler-let 'si:pretty-print-format) 1)
;; (setf (get 'cond 'si:pretty-print-format) 0)
;; (setf (get 'ctypecase 'si:pretty-print-format) 1)
;; (setf (get 'defcfun 'si:pretty-print-format) 2)
;; (setf (get 'define-setf-method 'si:pretty-print-format) 2)
;; (setf (get 'defla 'si:pretty-print-format) 2)
;; (setf (get 'defmacro 'si:pretty-print-format) 2)
;; (setf (get 'defsetf 'si:pretty-print-format) 3)
;; (setf (get 'defstruct 'si:pretty-print-format) 1)
;; (setf (get 'deftype 'si:pretty-print-format) 2)
;; (setf (get 'defun 'si:pretty-print-format) 2)
;; (setf (get 'do 'si:pretty-print-format) 2)
;; (setf (get 'do* 'si:pretty-print-format) 2)
;; (setf (get 'do-symbols 'si:pretty-print-format) 1)
;; (setf (get 'do-all-symbols 'si:pretty-print-format) 1)
;; (setf (get 'do-external-symbols 'si:pretty-print-format) 1)
;; (setf (get 'dolist 'si:pretty-print-format) 1)
;; (setf (get 'dotimes 'si:pretty-print-format) 1)
;; (setf (get 'ecase 'si:pretty-print-format) 1)
;; (setf (get 'etypecase 'si:pretty-print-format) 1)
;; (setf (get 'eval-when 'si:pretty-print-format) 1)
;; (setf (get 'flet 'si:pretty-print-format) 1)
;; (setf (get 'labels 'si:pretty-print-format) 1)
;; (setf (get 'let 'si:pretty-print-format) 1)
;; (setf (get 'let* 'si:pretty-print-format) 1)
;; (setf (get 'locally 'si:pretty-print-format) 0)
;; (setf (get 'loop 'si:pretty-print-format) 0)
;; (setf (get 'macrolet 'si:pretty-print-format) 1)
;; (setf (get 'multiple-value-bind 'si:pretty-print-format) 2)
;; (setf (get 'multiple-value-prog1 'si:pretty-print-format) 1)
;; (setf (get 'prog 'si:pretty-print-format) 1)
;; (setf (get 'prog* 'si:pretty-print-format) 1)
;; (setf (get 'prog1 'si:pretty-print-format) 1)
;; (setf (get 'prog2 'si:pretty-print-format) 2)
;; (setf (get 'progn 'si:pretty-print-format) 0)
;; (setf (get 'progv 'si:pretty-print-format) 2)
;; (setf (get 'return 'si:pretty-print-format) 0)
;; (setf (get 'return-from 'si:pretty-print-format) 1)
;; (setf (get 'tagbody 'si:pretty-print-format) 0)
;; (setf (get 'the 'si:pretty-print-format) 1)
;; (setf (get 'throw 'si:pretty-print-format) 1)
;; (setf (get 'typecase 'si:pretty-print-format) 1)
;; (setf (get 'unless 'si:pretty-print-format) 1)
;; (setf (get 'unwind-protect 'si:pretty-print-format) 0)
;; (setf (get 'when 'si:pretty-print-format) 1)
;; (setf (get 'with-input-from-string 'si:pretty-print-format) 1)
;; (setf (get 'with-open-file 'si:pretty-print-format) 1)
;; (setf (get 'with-open-stream 'si:pretty-print-format) 1)
;; (setf (get 'with-standard-io-syntax 'si:pretty-print-format) 1)
;; (setf (get 'with-output-to-string 'si:pretty-print-format) 1)


(in-package :si)

(defvar *lib-directory* (namestring (truename "../")))

(import '(*lib-directory* *load-path* *system-directory*) 'si::user) 
