;;; CMPENV  Environments of the Compiler.
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

;;; Only these flags are set by the user.
;;; If *safe-compile* is ON, some kind of run-time checks are not
;;; included in the compiled code.  The default value is OFF.

(defvar *dlinks* (make-hash-table :test 'equal))

(defun init-env ()
  (setq *tmpsyms* +tmpsyms+)
  (setq *gensym-counter* 0)
  (setq *next-cvar* 0)
  (setq *next-cmacro* 0)
  (setq *next-vv* -1)
  (setq *next-cfun* 0)
  (setq *last-label* 0)
  (setq *src-hash* (make-hash-table :test 'eq))
  (setq *fn-src-fn* (make-hash-table :test 'eq))
  (setq *objects* (make-hash-table :test 'eq))
  (setq *dlinks* (make-hash-table :test 'equal))
  (setq *local-funs* nil)
  (setq *hash-eq* nil)
  (setq *global-funs* nil)
  (setq *global-entries* nil)
  (setq *undefined-vars* nil)
  (setq *reservations* nil)
  (setq *top-level-forms* nil)
  (setq *function-declarations* nil)
  (setq *inline-functions* nil)
  (setq *function-links* nil)
  (setq *inline-blocks* 0)
  (setq *notinline* nil)
  )


(defvar *next-cvar* 0)
(defvar *next-cmacro* 0)
(defvar *next-vv* -1)
(defvar *next-cfun* 0)
(defvar *tmp-pack* nil)

;;; *next-cvar* holds the last cvar number used.
;;; *next-cmacro* holds the last cmacro number used.
;;; *next-vv* holds the last VV index used.
;;; *next-cfun* holds the last cfun used.

(defmacro next-cfun () '(incf *next-cfun*))

(defun add-libc (x)
  (add-dladdress (strcat "dl" x) (mdlsym x)))

(defun add-dladdress (n l) 
  (unless (gethash n *dlinks*) 
    (wt-h "static void *" n #+static"=" #+static(symbol-name l) ";")
    (setf (gethash n *dlinks*) t)
    (add-init `(si::mdl ',(symbol-name l) ',(package-name (symbol-package l)) ,(add-address (concatenate 'string "&" n))))))

;(defun add-symbol (symbol) symbol)

(defun add-object2 (object)
  (let* ((init (if (when (consp object) (eq (car object) '|#,|)) (cdr object) `',object)))
;    (unless init (break))
    (cond ((gethash object *objects*))
	  ((push-data-incf nil)
	   (when init (add-init `(setvv ,*next-vv* ,init)))
	   (setf (gethash object *objects*) *next-vv*)))))

;; Write to a string with all the *print-.. levels bound appropriately.
(defun wt-to-string (x &aux
		       (*compiler-output-data* (make-string-output-stream))
		       *fasd-data*)
  (wt-data1 x)
  (get-output-stream-string *compiler-output-data*))

(defun nani-eq (x y)
  (and (consp x) (consp y)
       (eq (car x) 'si::nani) (eq (car y) 'si::nani)
       (eq (cadr x) (cadr y))))

;(defun add-object (object) object)


(defun add-constant (symbol) (cons '|#,| symbol))

(defmacro next-cvar () '(incf *next-cvar*))
(defmacro next-cmacro () '(incf *next-cmacro*))

;;; Tail recursion information.
(defvar *do-tail-recursion* t)
;(defvar *tail-recursion-info* nil)
;;; Tail recursion optimization never occurs if *do-tail-recursion* is NIL.
;;; *tail-recursion-info* holds NIL, if tail recursion is impossible.
;;; If possible, *tail-recursion-info* holds
;;;	( fname  required-arg .... required-arg ),
;;; where each required-arg is a var-object.


(defvar *function-declarations* nil)
;;; *function-declarations* holds :
;;;	(... ( { function-name | fun-object } arg-types return-type ) ...)
;;; Function declarations for global functions are ASSOCed by function names,
;;; whereas those for local functions are ASSOCed by function objects.
;;;
;;; The valid argment type declaration is:
;;;	( {type}* [ &optional {type}* ] [ &rest type ] [ &key {type}* ] )
;;; though &optional, &rest, and &key return types are simply ignored.

;; (defmacro t-to-nil (x) (let ((s (tmpsym))) `(let ((,s ,x)) (if (eq ,s t) nil ,s))))

;; (defmacro nil-to-t (x) `(or ,x t))


(defun is-global-arg-type (x)
  (let ((x (promoted-c-type x)))
    (or (equal x #tt) (member x +c-global-arg-types+))))
(defun is-local-arg-type (x)
  (let ((x (promoted-c-type x)))
    (or (equal x #tt) (member x +c-local-arg-types+))))
(defun is-local-var-type (x)
  (let ((x (promoted-c-type x)))
    (or (equal x #tt) (member x +c-local-var-types+))))

;; (defun coerce-to-one-value (type)
;;   (or (not type) (type-and type t)))

(defun readable-tp (x) (cmp-unnorm-tp (cmp-norm-tp x)))

(defun function-arg-types (arg-types) (mapcar 'readable-tp arg-types))
;; (defun function-arg-types (arg-types &aux vararg (types nil) result)
;;   (setq result
;; 	(do ((al arg-types (cdr al))
;; 	     (i 0 (the fixnum (+ 1 i))))
;; 	    ((endp al)
;; 	     (reverse types))
;; 	    (declare (fixnum i))
;; 	    (cond ((or (member (car al) '(&optional &rest &key))
;; 		       (equal (car al) '* ))
;; 		   (setq vararg t)
;; 		   (return (reverse (cons '* types)))))
;; 	    ;; only the first 9 args may have proclaimed type different from T
;; 	    (push       (cond 
;; 			       ((< i 9)
;; 				(let ((tem
;; 				       (type-filter (car al))))
;; 				  (if (is-local-arg-type tem) (nil-to-t (car al)) t)));FIXME
;; 			      (t (if (eq (car al) '*) '* t)))
;; 			types)))
;;   ;;only type t args for var arg so far.
;;   (cond (vararg (do ((v result (cdr v)))
;; 		    ((null v))
;; 		    (setf (car v) (if (eq (car v) '*) '* t)))))
		    
;;   result)


;;; The valid return type declaration is:
;;;	(( VALUES {type}* )) or ( {type}* ).

(defun function-return-type (return-types)
  (cond ((endp return-types) nil)
        ((cmpt return-types)
	 (cmp-norm-tp `(,(car return-types) ,@(function-return-type (cdr return-types)))))
        ((cmpt (car return-types))
	 (cmp-norm-tp `(,(caar return-types) ,@(function-return-type (cdar return-types)))))
      	((mapcar 'readable-tp return-types))))

(defun add-function-declaration (fname arg-types return-types)
  (cond ((symbolp fname)
         (push (list (sch-local-fun fname)
                     (function-arg-types arg-types)
                     (function-return-type return-types))
               *function-declarations*))
        (t (warn "The function name ~s is not a symbol." fname))))

(defvar *assert-ftype-proclamations* nil)

(defun get-arg-types (fname &aux x)
  (cond ((when *assert-ftype-proclamations* (setq x (when (symbolp fname) (get fname 'proclaimed-signature)))) (car x))
	((setq x (assoc fname *function-declarations*)) (mapcar 'cmp-norm-tp (cadr x)))
	((setq x (local-fun-p fname)) (caar (fun-call x)))
	((setq x (gethash fname *sigs*)) (caar x))
	((setq x (si::sig fname)) (car x))
	((setq x (when (symbolp fname) (get fname 'proclaimed-signature))) (car x))
	('(*))))

(defun get-return-type (fname &aux x)
  (cond ((when *assert-ftype-proclamations* (setq x (when (symbolp fname) (get fname 'proclaimed-signature)))) (cadr x))
	((setq x (assoc fname *function-declarations*)) (cmp-norm-tp (caddr x)))
	((setq x (local-fun-p fname)) (cadar (fun-call x)))
	((setq x (gethash fname *sigs*)) (cadar x))
	((setq x (si::sig fname)) (cadr x))
	((setq x (when (symbolp fname) (get fname 'proclaimed-signature))) (cadr x))
	('*)))

(defun get-sig (fname)
  (list (get-arg-types fname) (get-return-type fname)))

(defun cclosure-p (fname)
  (not 
   (let ((x (or (fifth (gethash fname *sigs*)) (si::props fname))))
     (when x
       (logbitp 0 x)))))

(defun get-local-arg-types (fun &aux x)
  (if (setq x (assoc fun *function-declarations*))
      (cadr x)
      nil))

(defun get-local-return-type (fun &aux x)
  (if (setq x (assoc fun *function-declarations*))
      (caddr x)
      nil))

(defvar *vs-base-ori-used* nil)
(defvar *sup-used* nil)
(defvar *base-used* nil)
(defvar *frame-used* nil)
(defvar *bds-used*   nil)

(defun reset-top ()
  (wt-nl "vs_top=sup;")
  (setq *sup-used* t))

(defmacro base-used () '(setq *base-used* t))

;;; Proclamation and declaration handling.

(defvar *alien-declarations* nil)
(defvar *inline* nil)
(defvar *notinline* nil)

(defun inline-asserted (fname)
  (unless *compiler-push-events*
    (or 
     (member fname *inline*)
     (local-fun-fn fname)
     (get fname 'cmp-inline)
     (member (symbol-package fname)
	     (load-time-value (mapcar #'symbol-package (list 'c-t-tt (mdlsym "sin") (mdlsym "memmove"))))))))

;; (defun inline-asserted (fname)
;;   (unless *compiler-push-events*
;;     (or 
;;      (member fname *inline*)
;;      (local-fun-fn fname)
;;      (get fname 'cmp-inline))))

;; (defun inline-asserted (fname)
;;   (unless *compiler-push-events*
;;     (or 
;;      (member fname *inline*)
;;      (local-fun-fun fname)
;;      (get fname 'cmp-inline))))

(defun inline-possible (fname)
  (cond ((eq fname 'funcall));FIXME
	((eq fname 'apply));FIXME
	((not (or *compiler-push-events*
		  (member fname *notinline*)
		  (get fname 'cmp-notinline))))))

;; (defun inline-possible (fname)
;;   (not (or *compiler-push-events*
;; 	   (member fname *notinline*)
;; 	   (get fname 'cmp-notinline))))


(defun max-vtp (tp) (coerce-to-one-value (cmp-norm-tp tp)));FIXME lose coerce?

(defun body-safety (others &aux
			   (*compiler-check-args* *compiler-check-args*)
			   (*compiler-new-safety* *compiler-new-safety*)
			   (*compiler-push-events* *compiler-push-events*)
			   (*safe-compile* *safe-compile*))
  (mapc (lambda (x) (when (eq (car x) 'optimize) (local-compile-decls (cdr x)))) others)
  (this-safety-level))

(defun c1body (body doc-p &aux ss is ts others cps)
  (multiple-value-bind
   (doc decls ctps body)
   (parse-body-header body (unless doc-p ""))
   (dolist (decl decls)
     (dolist (decl (cdr decl))
       (cmpck (not (consp decl)) "The declaration ~s is illegal." decl)
       (let ((dtype (car decl)))
	 (if (consp dtype)
	     (let* ((dtype (max-vtp dtype))
		    (stype (if (consp dtype) (car dtype) dtype)))
	       (case stype
		     (satisfies (push decl others))
		     (otherwise
		      (dolist (var (cdr decl))
			(cmpck (not (symbolp var)) "The type declaration ~s contains a non-symbol ~s."
			       decl var)
			(push (cons var dtype) ts)))))
	   (let ((stype dtype))
	     (cmpck (not (symbolp stype)) "The declaration ~s is illegal." decl)
	     (case stype
		   (special
		    (dolist (var (cdr decl))
		      (cmpck (not (symbolp var)) "The special declaration ~s contains a non-symbol ~s."
			     decl var)
		      (push var ss)))
		   ((ignore ignorable)
		    (dolist (var (cdr decl))
		      (cmpck (not (typep var '(or symbol (cons (member function) (cons function-name null)))))
			     "The ignore declaration ~s is illegal ~s."
			     decl var)
		      (when (eq stype 'ignorable)
			(push 'ignorable is))
		      (push var is)))
		   ((optimize ftype inline notinline)
		    (push decl others))
		   ((hint type)
		    (cmpck (endp (cdr decl))  "The type declaration ~s is illegal." decl)
		    (let ((type (max-vtp (cadr decl))))
		      (when type
			(dolist (var (cddr decl))
			  (cmpck (not (symbolp var)) "The type declaration ~s contains a non-symbol ~s." decl var)
			  (cond ((unless (get var 'tmp) (eq stype 'hint)) (push (cons var type) cps) ;FIXME
				 (push (cons var (global-type-bump type)) ts))
				((push (cons var type) ts)))))))
		   (class ;FIXME pcl
		    (cmpck (cdddr decl) "The type declaration ~s is illegal." decl)
		    (let ((type (max-vtp (or (caddr decl) (car decl)))))
		      (when type
			(let ((var (cadr decl)))
			  (cmpck (not (symbolp var)) "The type declaration ~s contains a non-symbol ~s."
				 decl var)
			  (push (cons var type) ts)))))
		   (object
		    (dolist (var (cdr decl))
		      (cmpck (not (symbolp var)) "The object declaration ~s contains a non-symbol ~s."
			     decl var)
		      (push (cons var 'object) ts)))
		   (:register
		    (dolist (var (cdr decl))
		      (cmpck (not (symbolp var)) "The register declaration ~s contains a non-symbol ~s."
			     decl var)
		      (push (cons var  'register) ts)))
		   ((:dynamic-extent dynamic-extent)
		    (dolist (var (cdr decl))
		      (cmpck (not (symbolp var)) "The type declaration ~s contains a non-symbol ~s."
			     decl var)
		      (push (cons var 'dynamic-extent) ts)))
		   (otherwise
		    (let ((type (unless (member stype *alien-declarations*) (max-vtp stype))))
		      (if type
			  (unless (eq type t)
			    (dolist (var (cdr decl))
			      (cmpck (not (symbolp var))
				     "The type declaration ~s contains a non-symbol ~s."
				     decl var)
			      (push (cons var type) ts)))
			(push decl others))))))))))

  (dolist (l ctps) 
    (when (and (cadr l) (symbolp (cadr l))) 
      (let ((tp (or (eq (car l) 'assert) (max-vtp (caddr l)))))
	(unless (eq tp t) 
	  (push (cons (cadr l) tp) cps)))))

  (let ((s (> (body-safety others) (if (top-level-src-p) 0 1))))
    (when ctps
      (setq body (nconc (if s ctps
			    (nconc (mapcar (lambda (x) `(infer-tp ,(car x) ,(cdr x))) cps)
				   (mapcan (lambda (x) (when (eq (car x) 'assert) (list (cadr x)))) ctps)))
			body))))
  (values body ss ts is others (when doc-p doc) cps)))


(defun c1decl-body (decls body &aux dl)
  (let ((*function-declarations* *function-declarations*)
	(*alien-declarations* *alien-declarations*)
	(*notinline* *notinline*)
	(*inline* *inline*)
	(*space* *space*)
	(*compiler-check-args* *compiler-check-args*)
	(*compiler-new-safety* *compiler-new-safety*)
	(*compiler-push-events* *compiler-push-events*)
	(*safe-compile* *safe-compile*))
    (dolist (decl decls dl)
      (case (car decl)
	    (optimize
	     (dolist (d (cdr decl)) (push d dl))
	     (local-compile-decls (cdr decl)))
	    (ftype
	     (if (or (endp (cdr decl))
		     (not (consp (cadr decl)))
		     (not (eq (caadr decl) 'function))
		     (endp (cdadr decl)))
		 (cmpwarn "The function declaration ~s is illegal." decl)
	       (dolist (fname (cddr decl))
		 (add-function-declaration
		  fname (cadadr decl) (cddadr decl)))))
	    (function
	     (if (or (endp (cdr decl))
		     (endp (cddr decl))
		     (not (symbolp (cadr decl))))
		 (cmpwarn "The function declaration ~s is illegal." decl)
	       (add-function-declaration
		(cadr decl) (caddr decl) (cdddr decl))))
	    (inline
	      (dolist (fun (cdr decl))
		(if (symbolp fun)
		    (progn (push (list 'inline fun) dl)
			   (pushnew fun *inline*)
			   (setq *notinline* (remove fun *notinline*)))
		  (cmpwarn "The function name ~s is not a symbol." fun))))
	    (notinline
	     (dolist (fun (cdr decl))
	       (if (symbolp fun)
		   (progn (push (list 'notinline fun) dl)
			  (pushnew fun *notinline*)
			  (setq *inline* (remove fun *inline*)))
		 (cmpwarn "The function name ~s is not a symbol." fun))))
	    (declaration
	     (dolist (x (cdr decl))
	       (if (symbolp x)
		   (unless (member x *alien-declarations*)
		     (push x *alien-declarations*))
		 (cmpwarn "The declaration specifier ~s is not a symbol."
			  x))))
	    (otherwise
	     (unless (member (car decl) *alien-declarations*)
	       (cmpwarn "The declaration specifier ~s is unknown." (car decl))))))
    (let ((c1b (c1progn body)))
      (cond ((null dl) c1b)
	    ((member (car c1b) '(var lit)) c1b)
	    ((eq (car c1b) 'decl-body) (setf (third c1b) (nunion dl (third c1b))) c1b)
	    ((list 'decl-body (copy-info (cadr c1b)) dl c1b))))))

(si:putprop 'decl-body 'c2decl-body 'c2)

(defun local-compile-decls (decls)
  (dolist (decl decls)
    (unless (consp decl) (setq decl (list decl 3)))
    (case (car decl)
	  (debug (setq *debug* (cadr decl)))
	  (safety
	   (let* ((tl (this-safety-level))(level (if (>= tl 3) tl (cadr decl))))
	     (declare (fixnum level))
	     (when (top-level-src-p)
	       (setq *compiler-check-args* (>= level 1)
		     *safe-compile* (>= level 2)
		     *compiler-new-safety* (>= level 3)
		     *compiler-push-events* (>= level 4)))));FIXME
	  (space (setq *space* (cadr decl)))
	  (notinline (push (cadr decl) *notinline*))
	  (speed) ;;FIXME
	  (compilation-speed) ;;FIXME
	  (inline (setq *notinline* (remove (cadr decl) *notinline*)))
	  (otherwise (baboon)))))

(defun c2decl-body (decls body)
  (let ((*compiler-check-args* *compiler-check-args*)
        (*safe-compile* *safe-compile*)
        (*compiler-push-events* *compiler-push-events*)
        (*compiler-new-safety* *compiler-new-safety*)
        (*notinline* *notinline*)
        (*space* *space*)
        (*debug* *debug*))
    (local-compile-decls decls)
    (c2expr body)))

(defun check-vdecl (vnames ts is)
  (dolist (d ts)
    (unless (member (car d) vnames);FIXME check error without this
      (keyed-cmpnote (list 'free-type-declaration (car d))
		     "free type declaration ~s ~s" (car d) (cdr d))
      (c1infer-tp (list (car d) (cdr d)))))
  (dolist (x is)
    (unless (or (eq x 'ignorable) (member x vnames :test 'equal))
      (cmpwarn "Ignore/ignorable declaration was found for not bound variable ~s." x))))


