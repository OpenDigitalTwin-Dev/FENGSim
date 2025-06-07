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


;;;;        trace.lsp 
;;;;
;;;;        Tracer package for Common Lisp

;;;;;; Modified by Matt Kaufmann to allow tracing options.


;; If you are working in another package you should (import 'si::arglist)
;; to avoid typing the si::

;; (in-package 'lisp)

;; (export '(trace untrace))
;; (export 'step)


(in-package :system)

;;(proclaim '(optimize (safety 2) (space 3)))


(defvar *trace-level* 0)
(defvar *trace-list* nil)


(defmacro trace (&rest r)
  (if (null r)
      '(mapcar 'car *trace-list*)
    `(let ((old (copy-list *trace-list*)) finish-flg)
       (unwind-protect
	   (prog1 (mapcan 'trace-one ',r)
	     (setq finish-flg t))
	 (when (null finish-flg)
	       (format *standard-output* "~%Newly traced functions:  ~S"
		       (mapcar 'car (set-difference *trace-list* old :test #'equal))))))))

(defmacro untrace (&rest r)
  `(mapcan 'untrace-one ',(or r (mapcar 'car *trace-list*))))

(defun trace-one-preprocess (x)
  (if (symbolp x)
      (trace-one-preprocess (list x))
    (do ((tail (cdr x) (cddr tail))
	 (declarations)
	 (entryform `(cons (quote ,(car x)) arglist))
	 (exitform `(cons (quote ,(car x)) values))
	 (condform t)
	 (entrycondform t)
	 (exitcondform t)
	 (depth) (depthvar))
	((null tail)
	 (when depth
	   ;; Modify the :cond so that it first checks depth, and then
	   ;; modify the :entry so that it first increments depth.  Notice
	   ;; that :cond will be fully evaluated before depth is incremented.
	   (setq depthvar (gensym))
	   ;; now reset the condform
	   (if
	       (eq condform t)
	       (setq condform
		     `(< ,depthvar ,depth))
	     (setq condform `(if (< ,depthvar ,depth) ,condform nil)))
	   (setq declarations (cons (cons depthvar 0) declarations))
	   ;; I'll have the depth be incremented for all the entry stuff and no exit stuff,
	   ;; since I don't see any more uniform, logical way to do this.
	   (setq entrycondform
		 `(progn
		    (setq ,depthvar (1+ ,depthvar))
		    ,entrycondform))
	   (setq exitcondform
		 `(progn
		    (setq ,depthvar (1- ,depthvar))
		    ,exitcondform)))
	 `(,(car x) ,declarations
	   (quote ,condform)
	   (quote ,entrycondform) (quote ,entryform)
	   (quote ,exitcondform) (quote ,exitform)))
	(case (car tail)
	      (:declarations
	       (setq declarations
		     (do ((decls (cadr tail) (cdr decls))
			  (result))
			 ((null decls) result)
			 (setq result
			       (cons (if (symbolp (car decls))
					 (cons (car decls) nil)
				       (cons (caar decls) (eval (cadar decls))))
				     result)))))
	      (:cond (setq condform (cadr tail)))
	      (:entrycond (setq entrycondform (cadr tail)))
	      (:entry (setq entryform (cadr tail)))
	      (:exitcond (setq exitcondform (cadr tail)))
	      (:exit (setq exitform (cadr tail))) 
	      (:depth (setq depth (cadr tail)))
	      (otherwise nil)))))

(defun check-trace-spec (form)
  (or (symbolp form)
      (if (and (consp form) (null (cdr (last form))))
	  (check-trace-args form (cdr form) nil)
	(error "Each trace spec must be a symbol or a list terminating in NIL, but ~S is not~&."
	       form))))

(defun check-declarations (declarations &aux decl)
  (when (consp declarations)
	(setq decl (if (consp (car declarations)) (car declarations) (list (car declarations) nil)))
	(when (not (symbolp (car decl)))
	      (error "Declarations are supposed to be of symbols, but ~S is not a symbol.~&"
		     (car decl)))
	(when (cddr decl)
	      (error "Expected a CDDR of NIL in ~S.~&"
		     decl))
	(when (assoc (car decl) (all-trace-declarations))
	      (error "The variable ~A is already declared for tracing"
		     (car decl)))))

(defun check-trace-args (form args acc-keywords)
  (when args
	(cond
	 ((null (cdr args))
	  (error "A trace spec must have odd length, but ~S does not.~&"
		 form))
	 ((member (car args) acc-keywords)
	  (error "The keyword ~A occurred twice in the spec ~S~&"
		 (car args) form))
	 (t
	  (case (car args)
		((:entry :exit :cond :entrycond :exitcond)
		 (check-trace-args form (cddr args) (cons (car args) acc-keywords)))
		(:depth
		 (when (not (and (integerp (cadr args))
				 (> (cadr args) 0)))
		       (error
			"~&Specified depth should be a positive integer, but~&~S is not.~&"
			(cadr args)))
		 (check-trace-args form (cddr args) (cons :depth acc-keywords)))
		(:declarations
		 (check-declarations (cadr args))
		 (check-trace-args form (cddr args) (cons :declarations acc-keywords)))
		(otherwise
		 (error "Expected :entry, :exit, :cond, :depth, or :declarations~&~
                         in ~S where instead there was ~S~&"
			form (car args))))))))

(defun trace-one (form &aux f)
   (let* ((n (funid-sym-p form))
	  (n1 (or n (funid-sym (car form))))
	  (ofname (if n form (car form)))
	  (form (or n (cons n1 (cdr form))))
	  (fname n1))
     (check-trace-spec form)
     (when (null (fboundp fname))
       (format *trace-output* "The function ~S is not defined.~%" fname)
       (return-from trace-one nil))
     (when (special-operator-p fname)
       (format *trace-output* "~S is a special form.~%" fname)
       (return-from trace-one nil))
     (when (macro-function fname)
       (format *trace-output* "~S is a macro.~%" fname)
       (return-from trace-one nil))
     (when (get fname 'traced)
       (untrace-one ofname))
     (setq form (trace-one-preprocess form))
     (let ((x (get fname 'state-function))) (when x (break-state 'trace x)))
     (fset (setq f (gensym)) (symbol-function fname))
     (eval `(defun ,fname (&rest args) (trace-call ',f args ,@(cddr form))))
     (putprop fname f 'traced)
     (setq *trace-list* (cons (cons ofname (cadr form)) *trace-list*))
     (list ofname)))

(defun reset-trace-declarations (declarations)
  (when declarations
	(set (caar declarations) (cdar declarations))
	(reset-trace-declarations (cdr declarations))))

(defun all-trace-declarations ( &aux result)
  (dolist (v *trace-list*)
	  (setq result (append result (cdr v))))
  result)
	  
(defun trace-call (temp-name args cond entrycond entry exitcond exit
			 &aux (*trace-level* *trace-level*) (*print-circle* t) vals indent)
  (when (= *trace-level* 0)
	(reset-trace-declarations (all-trace-declarations)))
  (cond
   ((eval `(let ((arglist (quote ,args))) ,cond))
    (setq *trace-level* (c+ 1 *trace-level*))
    (setq indent (let ((x (c+ *trace-level* *trace-level*))) (if (si::<2 x 20) x 20)))
    (fresh-line *trace-output*)
    (when (or (eq entrycond t)		;optimization for common value
	      (eval `(let ((arglist (quote ,args))) ,entrycond)))
	  ;; put out the prompt before evaluating
	  (format *trace-output*
		  "~V@T~D> "
		  indent *trace-level*)
	  (format *trace-output*
		  "~S~%"
		  (eval `(let ((arglist (quote ,args))) ,entry)))
	  (fresh-line *trace-output*))
    (setq vals (multiple-value-list (apply temp-name args)))
    (when (or (eq exitcond t)		;optimization for common value
	      (eval `(let ((arglist (quote ,args)) (values (quote ,vals)))
		       ,exitcond)))
	  ;; put out the prompt before evaluating
	  (format *trace-output*
		  "~V@T<~D "
		  indent
		  *trace-level*) 
	  (format *trace-output*
		  "~S~%"
		  (eval `(let ((arglist (quote ,args)) (values (quote ,vals))) ,exit))))
    (setq *trace-level* (1- *trace-level*))
    (values-list vals))
   (t (apply temp-name args))))

(defun traced-sym (fname)
  (let* ((sym (when (symbolp fname) (get fname 'traced)))
	 (fn (when (and sym (symbolp sym) (fboundp fname)) 
	       (function-lambda-expression (symbol-function fname))))
	 (fn (and (consp fn) (third fn)))
	 (fn (and (consp fn) (third fn))))
    (and (consp fn) (eq (car fn) 'trace-call) sym)))

(defun untrace-one (fname)
  (let* ((ofname fname)
	 (fname (funid-sym fname))
	 (sym (traced-sym fname))
	 (g (get fname 'traced)))
    (unless sym
      (cond ((not g) (warn "The function ~S is not traced.~%" fname))
	    ((fboundp fname) (warn "The function ~S was traced, but redefined.~%" ofname))
	    ((warn "The function ~S was traced, but is no longer defined.~%"  ofname))))
    (remprop fname 'traced)
    (setq *trace-list* (delete-if #'(lambda (u) (equal (car u) ofname)) *trace-list* :count 1))
    (when sym
      (fset fname (symbol-function sym)))
    (when g (list ofname))))

#| Example of tracing a function "fact" so that only the outermost call is traced.

(defun fact (n) (if (= n 0) 1 (* n (fact (1- n)))))

;(defvar in-fact nil)
(trace (fact :declarations ((in-fact nil))
	     :cond
	     (null in-fact)
	     :entry
	     (progn
	       (setq in-fact t)
	       (princ "Here comes input ")
	       (cons 'fact arglist))
             :exit
             (progn (setq in-fact nil)
		    (princ "Here comes output ")
                    (cons 'fact values))))

; Example of tracing fact so that only three levels are traced

(trace (fact :declarations
	     ((fact-depth 0))
	     :cond
	     (and (< fact-depth 3)
		  (setq fact-depth (1+ fact-depth)))
	     :exit
	     (progn (setq fact-depth (1- fact-depth)) (cons 'fact values))))
|#



(defvar *step-level* 0)
(defvar *step-quit* nil)
(defvar *step-function* nil)

(defvar *old-print-level* nil)
(defvar *old-print-length* nil)


(defun step-read-line ()
  (do ((char (read-char *debug-io*) (read-char *debug-io*)))
      ((or (char= char #\Newline) (char= char #\Return)))))

(defmacro if-error (error-form form)
  (let ((v (gensym)) (f (gensym)) (b (gensym)))
    `(let (,v ,f)
       (block ,b
         (unwind-protect (setq ,v ,form ,f t)
           (return-from ,b (if ,f ,v ,error-form)))))))

(defmacro step (form)
  `(let* ((*old-print-level* *print-level*)
          (*old-print-length* *print-length*)
          (*print-level* 2)
          (*print-length* 2))
     (read-line)
     (format *debug-io* "Type ? and a newline for help.~%")
     (setq *step-quit* nil)
     (stepper ',form nil)))

(defun stepper (form &optional env
                &aux values (*step-level* *step-level*) indent)
  (when (eq *step-quit* t)
    (return-from stepper (evalhook form nil nil env)))
  (when (numberp *step-quit*)
    (if (>= (1+ *step-level*) *step-quit*)
        (return-from stepper (evalhook form nil nil env))
        (setq *step-quit* nil)))
  (when *step-function*
    (if (and (consp form) (eq (car form) *step-function*))
        (let ((*step-function* nil))
          (return-from stepper (stepper form env)))
        (return-from stepper (evalhook form #'stepper nil env))))
  (setq *step-level* (1+ *step-level*))
  (setq indent (min (* *step-level* 2) 20))
  (loop
    (format *debug-io* "~VT~S " indent form)
    (finish-output *debug-io*)
    (case (do ((char (read-char *debug-io*) (read-char *debug-io*)))
              ((and (char/= char #\Space) (char/= char #\Tab)) char))
          ((#\Newline #\Return)
           (setq values
                 (multiple-value-list
                  (evalhook form #'stepper nil env)))
           (return))
          ((#\n #\N)
           (step-read-line)
           (setq values
                 (multiple-value-list
                  (evalhook form #'stepper nil env)))
           (return))
          ((#\s #\S)
           (step-read-line)
           (setq values
                 (multiple-value-list
                  (evalhook form nil nil env)))
           (return))
          ((#\p #\P)
           (step-read-line)
           (write form
                  :stream *debug-io*
                  :pretty t :level nil :length nil)
           (terpri))
          ((#\f #\F)
           (let ((*step-function*
                  (if-error nil
                            (prog1 (read-preserving-whitespace *debug-io*)
                                   (step-read-line)))))
             (setq values
                   (multiple-value-list
                    (evalhook form #'stepper nil env)))
             (return)))
          ((#\q #\Q)
           (step-read-line)
           (setq *step-quit* t)
           (setq values
                 (multiple-value-list
                  (evalhook form nil nil env)))
           (return))
          ((#\u #\U)
           (step-read-line)
           (setq *step-quit* *step-level*)
           (setq values
                 (multiple-value-list
                  (evalhook form nil nil env)))
           (return))
          ((#\e #\E)
           (let ((env1 env))
             (dolist (x
                      (if-error nil
                                (multiple-value-list
                                 (evalhook
                                  (if-error nil
                                            (prog1
                                             (read-preserving-whitespace
                                              *debug-io*)
                                             (step-read-line)))
                                  nil nil env1))))
                     (write x
                            :stream *debug-io*
                            :level *old-print-level*
                            :length *old-print-length*)
                     (terpri *debug-io*))))
          ((#\r #\R)
           (let ((env1 env))
             (setq values
                   (if-error nil
                             (multiple-value-list
                              (evalhook
                               (if-error nil
                                         (prog1
                                          (read-preserving-whitespace
                                           *debug-io*)
                                          (step-read-line)))
                               nil nil env1)))))
           (return))
          ((#\b #\B)
           (step-read-line)
           (let ((*ihs-base* (1+ *ihs-top*))
                 (*ihs-top* (1- (ihs-top)))
                 (*current-ihs* *ihs-top*))
             (simple-backtrace)))
          (t
           (step-read-line)
           (terpri)
           (format *debug-io*
                  "Stepper commands:~%~
		n (or N or Newline):	advances to the next form.~%~
		s (or S):		skips the form.~%~
		p (or P):		pretty-prints the form.~%~
                f (or F) FUNCTION:	skips until the FUNCTION is called.~%~
                q (or Q):		quits.~%~
                u (or U):		goes up to the enclosing form.~%~
                e (or E) FORM:		evaluates the FORM ~
					and prints the value(s).~%~
                r (or R) FORM:		evaluates the FORM ~
					and returns the value(s).~%~
                b (or B):		prints backtrace.~%~
		?:			prints this.~%")
           (terpri))))
  (when (or (constantp form) (and (consp form) (eq (car form) 'quote)))
        (return-from stepper (car values)))
  (if (endp values)
      (format *debug-io* "~V@T=~%" indent)
      (do ((l values (cdr l))
           (b t nil))
          ((endp l))
        (if b
            (format *debug-io* "~V@T= ~S~%" indent (car l))
            (format *debug-io* "~V@T& ~S~%" indent (car l)))))
  (setq *step-level* (- *step-level* 1))
  (values-list values))

