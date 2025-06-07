;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                    ;;;;;
;;;     Copyright (c) 1989 by William Schelter,University of Texas     ;;;;;
;;;     Copyright (c) 2024 Camm Maguire
;;;     All rights reserved                                            ;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; See the doc/DOC file for information on emit-fn and
;; make-all-proclaims.  The basic idea is to utilize information gathered
;; by the compiler in a compile of a system of files in order to generate
;; better code on subsequent compiles of the system.  To do this a file
;; sys-proclaim.lisp should be produced.

;; Additionally cross reference information about functions in the system is
;; collected.

(in-package :compiler)
;(import 'sloop::sloop)

(defstruct fn
  name           ;; name of THIS FUNCTION
  def            ;; defun, defmacro 
  value-type     ;; If this function's body contained
                 ;; (cond ((> a 3) 7)
		 ;;	  ((> a 1) (foo)))
                 ;; then the return type of 7 is known at compile time
                 ;; and value-type would be fixnum. [see return-type]
  fun-values     ;; list of functions whose values are the values of THIS FN
                 ;; (foo) in the previous example.
  callees        ;; list of all functions called by THIS FUNCTION
  return-type    ;; Store a return-type computed from the fun-values
                 ;; and value-type field.  This computation is done later.
  arg-types      ;; non optional arg types.  
  no-emit        ;; if not nil don't emit declaration.
  macros
  )

(si::freeze-defstruct 'fn)

(defvar *other-form* (make-fn))
(defvar *all-fns* nil)
(defvar *call-table* (make-hash-table))
(defvar *current-fn* nil)
(defun add-callee (fname)
  (cond ((consp fname)
	 (or (eq (car fname) 'values)
	     (add-callee (car fname))))
	((eq fname 'single-value))
	(fname (pushnew fname (fn-callees (current-fn))))))

(defun add-macro-callee (fname)
  (or
    ;; make sure the macro fname is not shadowed in the current environment.
    (sloop::sloop for v in *funs*
		  when (and (consp v) (eq (car v) fname))
		  do (return t))
    (pushnew fname (fn-macros (current-fn)))))

(defun clear-call-table ()
  (setf *current-fn* nil)
  (setq *all-fns* nil)
  (setq *other-form* (make-fn :name 'other-form))
  (clrhash *call-table*)
  (setf (gethash 'other-form *call-table*) *other-form*) 
  )

(defun emit-fn (flag)
  (declare (ignore flag))
;  (setq *record-call-info* flag)
  )

(defun type-or (a b)
  (if (eq b '*) '*
    (case a
      ((nil) b)
      ((t inline) t)
      ((fixnum inline-fixnum fixnum-value) (if (eq b 'fixnum) 'fixnum
					     (type-or t b)))
      (otherwise '*)
      )))

(defun current-fn ()
  (cond ((and (consp *current-form*)
	      (member (car *current-form*) '(defun defmacro))
	      (let ((sym (si::funid-sym (second *current-form*))))
		    (symbol-package sym)));;don't record gensym'd
	 (cond ((and *current-fn*
		     (equal (second *current-form*)  (fn-name *current-fn*)))
		*current-fn*)
	       (t
		 (unless
		   (setq *current-fn*
			 (gethash (second *current-form*) *call-table*))
		   (setq *current-fn* (make-fn :name (second *current-form*)
					       :def (car *current-form*)))
		   (setf (gethash (second *current-form*) *call-table*)
			 *current-fn*)
		   (setq *all-fns* (cons *current-fn* *all-fns*)))
		 *current-fn*)))
	;; catch all for other top level forms
	(t *other-form*)))

(defun who-calls (f)
  (sloop::sloop for (ke val) in-table *call-table*
	 when (or (member f (fn-callees val))
		  (member f (fn-macros val)))
	 collect ke))


(defun add-value-type (x fn &aux (current-fn (current-fn)))
  (cond (fn (pushnew fn
		     (fn-fun-values current-fn) :test 'equal))
	(t
	  (setf (fn-value-type current-fn)
		(type-or (fn-value-type current-fn) x)))))


(defun get-var-types (lis)
  (sloop::sloop for v in lis collect (or (si::si-classp (var-type v)) (si::structurep (var-type v)) (var-type v))))

(defun record-arg-info( lambda-list &aux (cf (current-fn)))
  (setf (fn-arg-types cf) (get-var-types (car lambda-list)))
  (when (sloop::sloop for v in (cdr lambda-list)
		      for w in '(&optional &rest &key
					   nil &allow-other-keys
					   )
		      when (and v w) do (return '*))
	(setf (fn-arg-types cf) (nconc(fn-arg-types cf) (list '*)))
	))

(defvar *depth* 0)
(defvar *called-from* nil)

(defun get-value-type (fname)
  (cond ((member fname *called-from* :test 'eq) nil)
	(t
	 (let ((tem (cons fname *called-from*)))
	   (declare (dynamic-extent tem))
	   (let ((*called-from* tem))
	     (get-value-type1 fname))))))

(defun get-value-type1 (fname
			&aux tem (*depth* (the fixnum (+ 1 (the fixnum
								*depth* )))))
  (cond ((> (the fixnum *depth*) 100) '*)
	((setq tem (gethash fname *call-table*))
	 (or
	  (fn-return-type tem)
	  (sloop::sloop with typ = (fn-value-type tem)
		for v in (fn-fun-values tem)
		when (symbolp v)
		do (setq typ (type-or typ (get-value-type v)))
		else
		when (and (consp v) (eq (car v) 'values))
		do
		(setq typ (type-or typ (if (eql (cdr v) 1) t '*)))
		else do (error "unknown fun value ~a" v)
		finally
		;; if there is no visible return, then we can assume
		;; one value.
		(or typ (fn-value-type tem)
		    (fn-fun-values tem)
		    (setf typ t))
		(setf (fn-return-type tem) typ)
		(return typ)
		)))
	((get fname 'proclaimed-return-type))
	(t '*)))
	
(defun result-type-from-loc (x)
  (cond ((consp x)
	 (case (car x)
	   ((fixnum-value inline-fixnum) 'fixnum)
	   (var (var-type (second x)))
	   ;; eventually separate out other inlines
	   (t (cond ((and (symbolp (car x))
			  (get (car x) 'wt-loc))
		     t)
		    (t (print (list 'type '* x)) '*)))))
	((or (eq x t) (null x)) t)
	(t (print (list 'type '*2 x)) '*)))


(defun small-all-t-p (args ret)
  (and (eq ret t)
       (< (length args) 10)
       (sloop::sloop for v in args always (eq v t))))

;; Don't change return type but pretend all these are optional args.

(defun no-make-proclaims-hack ()
  (sloop::sloop for (ke val) in-table *call-table*
	 do (progn ke) (setf (fn-no-emit val) 1)))

(defun set-closure ()
  (setf (fn-def (current-fn)) 'closure))
  
(defun make-proclaims ( &optional (st *standard-output*)
				  &aux (ht (make-hash-table :test 'equal))
				  *print-length* *print-level* 
				  (si::*print-package* t)
				  )
;  (require "VLFUN"
;	 (concatenate 'string si::*system-directory*
;		      "../cmpnew/lfun_list.lsp"))
  
  (print `(in-package ,(package-name *package*)) st)
  (sloop::sloop with ret with at
		for (ke val) in-table *call-table* 
		do
		(cond ((eq (fn-def val) 'closure)
		       (push ke (gethash 'proclaimed-closure ht)))
		      ((or (eql 1 (fn-no-emit val))
			   (not (eq (fn-def val) 'defun))))
		      (t (setq ret (get-value-type ke))
			 (setq at (fn-arg-types val))
			 (push ke   (gethash (list at ret) ht)))))
  (sloop::sloop for (at fns) in-table ht
		do 
		(print
		 (if (symbolp at) `(mapc (lambda (x) (setf (get x 'compiler::proclaimed-closure) t)) '(,@fns))
		   `(proclaim '(ftype (function ,@ at) ,@ fns)))
		 st)))
		 
(defun setup-sys-proclaims()
  (or (gethash 'si::call-test *call-table*)
      (get 'si::call-test 'proclaimed-function)
      (load (concatenate 'string si::*system-directory*
			 "../lsp/sys-proclaim.lisp"))
      (no-make-proclaims-hack)
      ))

(defun make-all-proclaims (&rest files)
  (declare (ignore files))
  ;; (setup-sys-proclaims)
  ;; (dolist (v files)
  ;; 	  (mapcar 'load (directory v)))
  (write-sys-proclaims "sys-proclaim.lisp"))

;; (defun write-sys-proclaims ()
;;   (with-open-file (st "sys-proclaim.lisp" :direction :output)
;;     (make-proclaims st)))

(defvar *file-table* (make-hash-table :test 'eq)) 

(defvar *warn-on-multiple-fn-definitions* t)

(defun add-fn-data (lis &aux tem (file (truename *load-pathname*)));*load-truename*
  (dolist (v lis)
    (cond ((eql (fn-name v) 'other-form)
	   (setf (fn-name v) (intern
			      (concatenate 'string "OTHER-FORM-"
					   (namestring file))))
	   (setf (get (fn-name v) 'other-form) t)))
    (setf (gethash (fn-name v) *call-table*) v)
    (when *warn-on-multiple-fn-definitions*
      (when (setq tem (gethash (fn-name v) *file-table*))
	(unless (equal tem file)
	  (warn 'simple-warning :format-control "~% ~a redefined in ~a. Originally in ~a."
		:format-arguments (list (fn-name v) file tem)))))
    (setf (gethash (fn-name v) *file-table*) file)))

(defun dump-fn-data (&optional (file "fn-data.lsp")
			       &aux (*package* (find-package "COMPILER"))
			       (*print-length* nil)
			       (*print-level* nil)
			       )
  (with-open-file (st file :direction :output)
    (format st "(in-package :compiler)(init-fn)~%(~s '(" 'add-fn-data)
    (sloop::sloop for (ke val) in-table *call-table*
		  do (progn ke) (print val st))
    (princ "))" st)
    (truename st)))

(defun record-call-info (loc fname)
  (cond ((and fname (symbolp fname))
	 (add-callee fname)))
  (cond ((eq loc 'record-call-info) (return-from record-call-info nil)))
  (case *value-to-go*
    (return
      (if (eq loc 'fun-val)
	  (add-value-type nil (or fname  'unknown-values))
	(add-value-type (result-type-from-loc loc) nil)))
    (return-fixnum
      (add-value-type 'fixnum nil))
    (return-object
      (add-value-type t nil))

    (top  (setq *top-data* (cons fname nil))
	 ))
     )

(defun list-undefined-functions (&aux undefs)
  (sloop::sloop for (name fn) in-table *call-table*
		declare (ignore name)
		do (sloop::sloop for w in (fn-callees fn)
			  when (not (or (fboundp w)
					(gethash w *call-table*)
					(get w 'inline-always)
					(get w 'inline-unsafe)
					(get w 'other-form)
					))
			  do (pushnew w undefs)))
  undefs)		



;(dolist (v '(throw coerce single-value  sort delete remove char-upcase
;		   si::fset typep))
;	(si::putprop v t 'return-type))

(defun init-fn () nil)

(defun list-uncalled-functions ( )
  (let* ((size (sloop::sloop for (ke v)
			     in-table *call-table* count t
			     do (progn ke v nil)))
	 (called (make-hash-table :test 'eq :size (+ 3 size))))
    (sloop::sloop for (ke fn) in-table *call-table*
		  declare (ignore ke)
		  do (sloop::sloop for w in (fn-callees fn)
				   do
				   (setf (gethash w called) t))
		  (sloop::sloop for w in (fn-macros fn)
				   do
				   (setf (gethash w called) t))
		  
		  )
    (sloop::sloop for (ke fn) in-table *call-table*
		  when(and
		       (not (gethash ke called))
		       (member (fn-def fn) '(defun defmacro)
			       :test 'eq))
		  collect ke)))

;; redefine the stub in defstruct.lsp
(defun si::record-fn (name def arg-types return-type)
  (if (null return-type) (setq return-type t))
  (and *record-call-info*
       *compiler-in-use*
       (let ((fn (make-fn :name name
		      :def def
		      :return-type return-type
		      :arg-types arg-types)))
	 (push fn *all-fns*)
	 (setf (gethash name *call-table*) fn))))

(defun get-packages (&optional (st "sys-package.lisp") pass
			       &aux (si::*print-package* t))
  (flet ((pr (x) (format st "~%~s" x)))
     (cond ((null pass)
	    (with-open-file (st st :direction :output)
	      (get-packages st 'establish)
	      (get-packages st 'export)
	      (get-packages st 'shadow)
	      (format st "~2%")
	      (return-from get-packages nil))))
	(dolist (p  (list-all-packages))
	   (unless
	    (member (package-name p)
		    '("SLOOP"
		      "COMPILER" "SYSTEM" "KEYWORD" "LISP" "USER")
		    :test 'equal
		    )
	    (format st "~2%;;; Definitions for package ~a of type ~a"
		    (package-name p) pass)
	    (ecase pass
	      (establish
	       (let ((SYSTEM::*PRINT-PACKAGE* t))
		 (pr 
		  `(in-package ,(package-name p) :use nil
			       ,@ (if (package-nicknames p)
				      `(:nicknames ',(package-nicknames p)))))))
	      (export
	       (let ((SYSTEM::*PRINT-PACKAGE* t))
		 (pr 
		  `(in-package ,(package-name p)
			       :use
			       '(,@
				 (mapcar 'package-name (package-use-list p)))
			       ,@(if (package-nicknames p)
				     `(:nicknames ',(package-nicknames p))))))
	       (let (ext (*package* p)
			 imps)
		 (do-external-symbols (sym p) (push sym ext)
				      (or (eq (symbol-package sym) p)
					  (push sym imps)))
		 (pr `(import ',imps))
		 (pr `(export ',ext))))
	      (shadow
	       (let ((SYSTEM::*PRINT-PACKAGE* t))
		 (pr `(in-package ,(package-name p))))
	       (let (in out (*package* (find-package "LISP")))
		 (dolist (v (package-shadowing-symbols p))
			 (cond ((eq (symbol-package v) p)
				(push v in))
			       (t (push v out))))
		 (pr `(shadow ',in))
		 (pr `(shadowing-import ',out))
		 (let (imp)
		   (do-symbols (v p)
			       (cond ((not (eq (symbol-package v) p))
				      (push v imp))))
		   (pr `(import ',imp))))))))))

(defun get-packages-ansi (pl &optional (st "sys-package.lisp") pass &aux (si::*print-package* t))
  (flet ((pr (x) (format st "~%~s" x)))
     (cond ((null pass)
	    (with-open-file (st st :direction :output)
	      (setq pl (sort (copy-list pl)
			     (lambda (x y)
			       (member (find-package y) (package-used-by-list (find-package x))))))
	      (get-packages-ansi pl st 'establish)
	      (get-packages-ansi pl st 'export)
	      (get-packages-ansi pl st 'shadow)
	      (format st "~2%")
	      (return-from get-packages-ansi nil))))
     (dolist (p pl)
	   (unless
	       (member (package-name p)
		       '("SLOOP" "COMPILER" "SYSTEM" "KEYWORD" "LISP" "USER")
		       :test 'equal)
	     (format st "~2%;;; Definitions for package ~a of type ~a" (package-name p) pass)
	     (ecase pass
	       (establish
		(pr `(unless (find-package ,(package-name p))
		       (make-package ,(package-name p)
				     :use ',(mapcar 'package-name (package-use-list p))
				     ,@(when (package-nicknames p)
					 `(:nicknames ',(package-nicknames p)))))))
	       (export
		(let (ext (*package* p) imps)
		  (do-external-symbols (sym p)
		    (push sym ext)
		    (unless (eq (symbol-package sym) p)
		      (push sym imps)))
		  (pr `(import ',imps ,(package-name p)))
		  (pr `(export ',ext ,(package-name p)))))
	       (shadow (print p)
		(let (in out (*package* (find-package "CL")))
		  (dolist (v (package-shadowing-symbols p))
		    (if (eq (symbol-package v) p) (push v in) (push v out)));FIXME push if
		  (pr `(shadow ',in ,(package-name p)))
		  (pr `(shadowing-import ',out ,(package-name p)))
		  (let (imp)
		    (do-symbols (v p)
		      (unless (eq (symbol-package v) p)
		       (push v imp)))
		    (pr `(import ',imp ,(package-name p)))))))))))
