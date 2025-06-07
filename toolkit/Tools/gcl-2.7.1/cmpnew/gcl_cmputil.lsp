;;; CMPUTIL  Miscellaneous Functions.
;;;
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


(in-package :compiler)

(export '(*suppress-compiler-warnings*
          *suppress-compiler-notes*
          *compiler-break-enable*))

(defmacro safe-compile (&rest forms) `(when *safe-compile* ,@forms))

(defvar *current-form* '|compiler preprocess|)
(defvar *first-error* t)
(defvar *error-count* 0)

(defconstant *cmperr-tag* (cons nil nil))

(defun cmperr (string &rest args &aux (*print-case* :upcase))
  (print-current-form *error-output*)
  (si::error-format "ERROR: ")
  (apply #'si::error-format string args)
  (force-output *error-output*)
  (incf *error-count*)
  (throw *cmperr-tag* (c1nil)))

(defmacro cmpck (condition string &rest args)
  `(if ,condition (cmperr ,string ,@args)))

(defun too-many-args (name upper-bound n)
  (cmperr "~S requires at most ~R argument~:p, ~
           but ~R ~:*~[were~;was~:;were~] supplied.~%"
          name upper-bound n))

(defun too-few-args (name lower-bound n)
  (cmperr "~S requires at least ~R argument~:p, ~
           but only ~R ~:*~[were~;was~:;were~] supplied.~%"
          name lower-bound n))

(defvar *warning-note-stack*)

(defvar *suppress-compiler-warnings* nil)

(defmacro maybe-to-wn-stack (&rest body)
  (let ((cf (sgen "MTWSCF"))(sri (sgen "MTWSSRI")))
  `(if (and (boundp '*warning-note-stack*) (not *note-keys*));FIXME
       (let ((,cf *current-form*)(,sri *src-inline-recursion*)) 
	 (push (lambda nil
		 (let ((*current-form* ,cf)
		       (*src-inline-recursion* ,sri))
		   ,@body))
	       *warning-note-stack*))
     (progn ,@body))))

(defun output-warning-note-stack nil
  (when (boundp '*warning-note-stack*)
    (do ((*warning-note-stack* (nreverse *warning-note-stack*)))
	((not *warning-note-stack*))
      (funcall (pop *warning-note-stack*)))))
  

(defun cmpwarn (string &rest args &aux (*print-case* :upcase))
  (unless *suppress-compiler-warnings*
    (maybe-to-wn-stack
     (warn 'warning
	   :function-name (print-current-form nil)
	   :format-control "~?"
	   :format-arguments (list string args))))
  nil)

(defun cmpstyle-warn (string &rest args &aux (*print-case* :upcase))
  (unless *suppress-compiler-warnings*
    (maybe-to-wn-stack
     (warn 'style-warning
	   :function-name (print-current-form nil)
	   :format-control "~?"
	   :format-arguments (list string args))))
  nil)

(defvar *suppress-compiler-notes* t)
(defvar *note-keys* nil)
(defun watch (key)
  (pushnew key *note-keys*))
(defun unwatch (&rest keys)
  (setq *note-keys* (when keys (nset-difference *note-keys* keys))))

(defun cmpnote (string &rest args &aux (*print-case* :upcase))
  (maybe-to-wn-stack
   (print-current-form *debug-io*)
   (format *debug-io* ";; Note: ")
   (apply #'format *debug-io* string args)
   (terpri *debug-io*))
  nil)

(defun do-keyed-cmpnote (k string &rest args &aux (*print-case* :upcase))
  (do ((k k (when (consp k) (cdr k)))) ((not k))
      (let ((k (if (consp k) (car k) k)))
	(when (member k *note-keys* :test (lambda (x y) (or (eq x y) (eq 'all y))))
	  (apply 'cmpnote string args)
	  (return)))))
  
(defmacro keyed-cmpnote (key string &rest args)
  `(when *note-keys*
     (do-keyed-cmpnote ,key ,string ,@args)))

(defun print-current-form (&optional (strm t)
			   &aux (*print-length* 2)(*print-level* 2)(f *current-form*))
  (when (or (eq *first-error* t)
	    (not (eq (car *first-error*) *current-form*))
	    (not (eq (cdr *first-error*) *src-inline-recursion*)))
    (setq *first-error* (cons *current-form* *src-inline-recursion*))
    (let ((args (list ";; When compiling ~s~%~{;;   inlining ~s~%~}"
		      (if (and (consp f) (eq (car f) '|#,|)) (cdr f) f)
		      (mapcan (lambda (s) (unless (eq (caar s) f) (list (cons (name-sir (car s)) (cdr s)))))
			      (butlast *src-inline-recursion*)))))
      (if (eq *error-output* strm)
	  (apply 'si::error-format args)
	  (apply 'format strm args)))))

(defun undefined-variable (sym &aux (*print-case* :upcase))
  (cmpwarn
   ";; The variable ~s is undefined.~%~
           ;; The compiler will assume this variable is a global.~%"
   sym))

(defun baboon nil (cmperr "A bug was found in the compiler.  Contact Taiichi.~%"))

(defun cmp-eval (form)
  (multiple-value-bind 
   (x y) (cmp-toplevel-eval `(eval ',form))
    (cond (x
	   (cmpwarn "The form ~s was not evaluated successfully.  You are recommended to compile again.~%"
		    form)
	   `(error "Evaluation of ~s failed at compile time." ',form))
	  (y))))

;(si::putprop 'setf 'c1setf 'c1special)

;;The PLACE may be a local macro, so we must take care to expand it
;;before trying to call the macro form of setf, or an error will

;(defun c1setf (args &aux fd)
;  (cond ((and
;	   (consp (car args))
;	   (symbolp (caar args))
;	   (setq fd (cmp-macro-function (caar args))))
;	 (c1expr `(setf ,(cmp-expand-macro fd (caar args) (cdar args))
;			,@ (cdr args))))
;	(t       
;         (c1expr (cmp-expand-macro (macro-function 'setf)
;				   'setf
;				   args)))))

(defmacro macroexpand-helper (pre meth form)
  (let ((c (sgen "MHC"))(x (sgen "MHX")))
    `(let ((,c (when (consp ,form) (car ,form))))
       ,@(when pre `(,pre))
       (cond ((not ,c) ,form)
	     ((not (symbolp ,c)) ,form)
	     ((not (cmp-macro-function ,c)) ,form);FIXME needed?
	     ((let* ((,x (multiple-value-list (cmp-toplevel-eval `,,meth))))
		(cond ((car ,x)
		       (cmpwarn "The macro form ~s was not expanded successfully.~%" ,form)
		       `(error "Macro-expansion of ~s failed at compile time." ',,form))
		      ((cadr ,x)))))))))

(defun cmp-macroexpand (form)
  (macroexpand-helper nil `(macroexpand ',form ',(funs-to-macrolet-env)) form))

(defun cmp-macroexpand-1 (form)
  (macroexpand-helper nil `(macroexpand-1 ',form ',(funs-to-macrolet-env)) form))

(defun cmp-expand-macro (fd fname args)
  (let ((x (cons fname args)))
    (macroexpand-helper
     (and *record-call-info* (add-macro-callee fname))
     `(funcall *macroexpand-hook* ',fd ',x ',(funs-to-macrolet-env))
     x)))

(defun cmp-expand-macro-w (fd x)
  (macroexpand-helper
   (and *record-call-info* (add-macro-callee (car x)))
   `(funcall *macroexpand-hook* ',fd ',x ',(funs-to-macrolet-env))
   x))

(defvar *compiler-break-enable* nil)

(defun cmp-toplevel-eval (form)
   (let* ((si::*ihs-base* si::*ihs-top*)
          (si::*ihs-top* (1- (si::ihs-top)))
          (si::*break-enable* *compiler-break-enable*)
          (si::*break-hidden-packages*
           (cons (find-package 'compiler)
                 si::*break-hidden-packages*)))
         (si:error-set form)))


(defun  compiler-def-hook (symbol code) symbol code nil)

;; (defun compiler-clear-compiler-properties (symbol code)
;;   code
;;   (let ((v (symbol-plist symbol)) w)
;;     (tagbody
;;       top
;;       (setq w (car v))
;;       (cond ((and (symbolp w)
;; 		  (get w 'compiler-prop))

;; 	     (setq v (cddr v))
;; 	     (remprop symbol w))
;; 	    (t (setq v (cddr v))))
;;       (or (null v) (go top)))
;;     (compiler-def-hook symbol code)
;;     ))

;hi
