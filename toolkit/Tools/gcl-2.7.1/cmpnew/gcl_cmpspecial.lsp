;;; CMPSPECIAL  Miscellaneous special forms.
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

(si:putprop 'quote 'c1quote 'c1special)
(si:putprop 'function 'c1function 'c1special)
(si:putprop 'function 'c2function 'c2)
(si:putprop 'the 'c1the 'c1special)
(si:putprop 'eval-when 'c1eval-when 'c1special)
(si:putprop 'declare 'c1declare 'c1special)
(si:putprop 'compiler-let 'c1compiler-let 'c1special)
(si:putprop 'compiler-let 'c2compiler-let 'c2)

(defun c1quote (args)
  (when (endp args) (too-few-args 'quote 1 0))
  (unless (endp (cdr args)) (too-many-args 'quote 1 (length args)))
  (c1constant-value (car args) t))

(defun c1eval-when (args)
  (when (endp args) (too-few-args 'eval-when 1 0))
  (dolist (situation (car args) (c1nil))
    (case situation
          ((eval :execute) (return-from c1eval-when (c1progn (cdr args))))
          ((load :load-toplevel compile  :compile-toplevel))
          (otherwise
           (cmperr "The situation ~s is illegal." situation))))
  )

(defun c1declare (args)
  (cmperr "The declaration ~s was found in a bad place." (cons 'declare args)))

(defun c1the (args &aux info form dtype);FIXME rethink this whole function
  (when (or (endp args) (endp (cdr args)))
    (too-few-args 'the 2 (length args)))
  (unless (endp (cddr args))
    (too-many-args 'the 2 (length args)))
  (setq form (c1expr (cadr args)))
  (setq info (copy-info (cadr form)))
  (setq dtype (cmp-norm-tp (cadr (si::ftype-to-sig (list nil (car args))))))
  (when (exit-to-fmla-p) (setq dtype (type-or1 (when (type-and #tnull dtype) #tnull) (when (type-and #t(not null) dtype) #ttrue))));FIXME
  (when (equal dtype #tboolean)
    (unless (type>= dtype (info-type info))
      (return-from c1the (c1expr `(when ,(cadr args) t)))))
;  (setq type (type-and dtype (info-type info)))

  (setq form (list* (car form) info (cddr form)))
  (set-form-type form dtype (type>= #tboolean dtype));FIXME understand boolean exception
;  (if (type>= #tboolean dtype) (setf (info-type (cadr form)) type) (set-form-type form dtype));FIXME
;  (setf (info-type info) type)
  form)

(defun c1compiler-let (args &aux (symbols nil) (values nil))
  (when (endp args) (too-few-args 'compiler-let 1 0))
  (dolist (spec (car args))
    (cond ((consp spec)
           (cmpck (not (and (symbolp (car spec))
                            (or (endp (cdr spec))
                                (endp (cddr spec)))))
                  "The variable binding ~s is illegal." spec)
           (push (car spec) symbols)
           (push (if (endp (cdr spec)) nil (eval (cadr spec))) values))
          ((symbolp spec)
           (push spec symbols)
           (push nil values))
          (t (cmperr "The variable binding ~s is illegal." spec))))
  (setq symbols (reverse symbols))
  (setq values (reverse values))
  (setq args (progv symbols values (c1progn (cdr args))))
  (list 'compiler-let (cadr args) symbols values args)
  )

(defun c2compiler-let (symbols values body)
  (progv symbols values (c2expr body)))

(defvar *fun-id-hash* (make-hash-table :test 'eq))
(defvar *fun-ev-hash* (make-hash-table :test 'eq))
(defvar *fun-tp-hash* (make-hash-table :test 'eq))

(defvar *fn-src-fn* (make-hash-table :test 'eq))

(defun coerce-to-funid (fn)
  (cond ((symbolp fn) fn)
	((local-fun-p fn) fn)
	((not (functionp fn)) nil)
	((fn-get fn 'id))
	((si::function-name fn))))
;	((portable-closure-src fn))

(defun find-special-var (l f)
  (when (consp l)
    (case (car l)
      (lambda (find-special-var (fifth l) f))
      (decl-body (find-special-var (fourth l) f))
      (let* (car (member-if f (third l)))))))



(defun is-narg-le (l) (caadr (caddr l)))

;; (defun is-narg-le (l)
;;   (find-special-var l 'is-narg-var))

(defun mv-var (l)
  (find-special-var l 'is-mv-var))

(defun fun-var (l)
  (find-special-var l 'is-fun-var))

(defun export-sig (sig)
  (uniq-sig `((,@(mapcar 'export-type (car sig))) ,(export-type (cadr sig)))))


(defun lam-e-to-sig (l &aux (args (caddr l)) (regs (car args)) (regs (if (is-first-var (car regs)) (cdr regs) regs)))
  (export-sig
   `((,@(mapcar 'var-type regs)
	,@(when (or (is-narg-le l) (member-if 'identity (cdr args))) `(*)))
     ,(info-type (cadar (last l))))))

(defun compress-fle (l y z)
  (let* ((fname (pop l))
	 (fname (or z fname))
	 (args  (pop l))
	 (w   (make-string-output-stream))
	 (out (pd fname args l))
	 (out (if y `(lambda-closure ,y nil nil ,@(cdr out)) out)))
    (if *compiler-compile* out
	(let ((ss  (si::open-fasd w :output nil nil)))
	  (si::find-sharing-top out (aref ss 1))
	  (si::write-fasd-top out ss)
	  (si::close-fasd ss)
	  (get-output-stream-string w)))))



(defun mc nil (let ((env (cons nil nil))) (lambda nil env)))

(defun afe (a f)
  (push a (car (funcall f)))
  f)

(defun fn-get (fn prop)
  (cdr (assoc prop (car (funcall fn)))))


(defun mf (id &optional fun)
  (let* ((f (mc)))
;    (when (consp id) (setf (caddr (si::call f)) (compress-fle id nil nil)))
    (when fun
      (afe (cons 'fun fun) f))
    (afe (cons 'id id) f)
    (when (or fun (consp id))
      (afe (cons 'df (current-env)) f))
    f))


(defun funid-to-fn (funid &aux fun)
  (cond ((setq fun (local-fun-p funid)) (fun-fn fun))
;	((gethash funid *fn-src-fn*))
;	((setf (gethash funid *fn-src-fn*) (mf funid)))
	((symbolp funid) (or (gethash funid *fn-src-fn*) (setf (gethash funid *fn-src-fn*) (mf funid))))
	((mf funid))
	))


(defvar *prov* nil)

(defun c1function (args &optional (b 'cb) f &aux fd)

  (when (endp args) (too-few-args 'function 1 0))
  (unless (endp (cdr args)) (too-many-args 'function 1 (length args)))
  
  (let* ((funid (si::funid (car args)))
	 (funid (mark-toplevel-src (if (consp funid) (effective-safety-src funid) funid)))
	 (fn (afe (cons 'ce (current-env)) (funid-to-fn funid)))
	 (tp (if fn (object-type fn) #tfunction))
	 (info (make-info :type tp)))
    (cond ((setq fd (c1local-fun funid t))
	   (add-info info (cadr fd))
	   `(function ,info ,fd))
	  ((symbolp funid)
	   (setf (info-flags info) (logior (info-flags info) (iflags sp-change)))
;	   (setf (info-sp-change info) (if (null (get funid 'no-sp-change)) 1 0))
	   `(function ,info (call-global ,info ,funid)))
	  ((let* ((fun (or f (make-fun :name 'lambda :src funid :c1cb t :fn fn :info (make-info :type '*))))
		  (fd (if *prov* (list fun) (process-local-fun b fun funid tp))))
	     (add-info info (cadadr fd))
	     (when *prov*
	       (pushnew funid *prov-src*)
	       (setf (info-flags info) (logior (info-flags info) (iflags provisional))))
	     `(function ,info ,fd))))))

(defun update-closure-indices (cl)
  (mapc (lambda (x &aux (y (var-ref-ccb (car x))))
	  (setf (cadr x) (if (integerp y) (- y *initial-ccb-vs*) (baboon))
		(car x) (var-name (car x))))
	(second (third cl)))
  cl)


(defun c2function (funob);FIXME
  (case (car funob)
        (call-global
         (unwind-exit (list 'symbol-function (caddr funob))))
        (call-local
	 (let* ((funob (caddr funob))(fun (pop funob)))
	   (unwind-exit (if (cadr funob) (list 'ccb-vs (fun-ref-ccb fun)) (list 'vs* (fun-ref fun))))))
        (otherwise
	 (let* ((fun (pop funob))
		(lam (car funob))
		(cl (update-closure-indices (fun-call fun)))
		(sig (car cl))
		(at (car sig))
		(rt (cadr sig))
		(clc (export-call-struct cl)))
	   
	   (pushnew (list 'closure (if (null *clink*) nil (cons 0 0)) *ccb-vs* fun lam)
		    *local-funs* :key 'fourth)
	   
	   (cond (*clink*
		  (let ((clc (cons '|#,| clc)))
		    (unwind-exit (list 'make-cclosure (fun-cfun fun) (fun-name fun) 
				       (or (fun-vv fun) clc)
				       (new-proclaimed-argd at rt) (argsizes at rt (xa lam))
				       *clink*))
		    (unless (fun-vv fun)
		      (setf (fun-vv fun) clc))))
		 (t  
		  (unless (fun-vv fun)
		    (setf (fun-vv fun)
			  (cons '|#,| `(init-function 
					,clc
					,(add-address (c-function-name "&LC" (fun-cfun fun) (fun-name fun)))
					nil nil
					-1 ,(new-proclaimed-argd at rt) ,(argsizes at rt (xa lam))))))
		  (unwind-exit (list 'vv (fun-vv fun)))))))))

(si:putprop 'symbol-function 'wt-symbol-function 'wt-loc)
(si:putprop 'make-cclosure 'wt-make-cclosure 'wt-loc)

(defun wt-symbol-function (vv)
  (if *safe-compile*
      (wt "symbol_function(" (vv-str vv) ")")
    (wt "(" (vv-str vv) "->s.s_gfdef)")))

(defun wt-make-cclosure (cfun fname call argd sizes &rest r &aux (args (car r)))
  (declare (dynamic-extent r))
  (declare (ignore args))
  (wt "fSinit_function(")
  (wt-vv call)
  (wt ",(void *)" (c-function-name "LC" cfun fname) ",Cdata,")
  (wt-clink)
  (wt ",-1," argd "," sizes ")"))

