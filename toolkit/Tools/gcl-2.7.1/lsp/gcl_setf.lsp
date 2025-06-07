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


;;;;        setf.lsp
;;;;
;;;;                                setf routines


;; (in-package 'lisp)


;; (export '(setf psetf shiftf rotatef
;;           define-modify-macro defsetf
;;           getf remf incf decf push pushnew pop
;; ;         define-setf-method
;; 	  define-setf-expander
;; ;	  get-setf-method
;; 	  get-setf-expansion
;; ;	  get-setf-method-multiple-value
;; 	  ))


(in-package :system)


;(eval-when (compile) (proclaim '(optimize (safety 2) (space 3))))
;(eval-when (eval compile) (defun si:clear-compiler-properties (symbol code)))
(eval-when (eval compile) (setq si:*inhibit-macro-special* nil))

(defvar *setf-syms* nil)
(defvar *setf-set* nil)
(defconstant +setf-prefix+ "SETF")
(defconstant +setf-syms+ (let ((*gensym-counter* 0))
			   (mapl (lambda (x) (rplaca x (gensym +setf-prefix+)))
				   (make-list 19))))
;(defun setf-set nil (setq *setf-syms* nil *gensym-counter* 0))
;(defun setf-set nil (or *setf-set* (setq *setf-syms* +setf-syms+)));FIXME, this does not seem possible
(defun setf-gensym nil (if *setf-syms* (prog1 (car *setf-syms*) (setq *setf-syms* (cdr *setf-syms*))) (gensym +setf-prefix+)))

;;; DEFSETF macro.
(defmacro defsetf (access-fn &rest rest)
  (declare (optimize (safety 2)))
  (cond ((and (car rest) (or (symbolp (car rest)) (functionp (car rest))))
         `(eval-when(compile eval load)
                 (si:putprop ',access-fn ',(car rest) 'setf-update-fn)
                 (remprop ',access-fn 'setf-lambda)
                 (remprop ',access-fn 'setf-method)
                 (si:putprop ',access-fn
                             ,(when (not (endp (cdr rest)))
                                    (unless (stringp (cadr rest))
                                            (error "A doc-string expected."))
                                    (unless (endp (cddr rest))
                                            (error "Extra arguments."))
                                    (cadr rest))
                             'setf-documentation)
                 ',access-fn))
	(t
	 (unless (= (list-length (cadr rest)) 1)
		 (error "(store-variable) expected."))
	 (multiple-value-bind
	  (doc decls ctps body)
	  (parse-body-header (cddr rest))
	  (declare (ignore ctps))
	  `(eval-when (compile eval load)
		      (si:putprop 
		       ',access-fn 
		       (lambda ,(car rest) ,@decls (lambda ,(cadr rest) (block ,access-fn ,@body))) 'setf-lambda)
		      (remprop ',access-fn 'setf-update-fn)
		      (remprop ',access-fn 'setf-method)
		      (si:putprop ',access-fn
				  ,doc
				  'setf-documentation)
		      ',access-fn)))))
  
  
;;; DEFINE-SETF-METHOD macro.
(defmacro define-setf-method (access-fn &rest rest &aux body)
  (multiple-value-bind
   (args env) 
   (get-&environment (car rest))
   (setq body (cdr rest))
   (cond (env (setq args (cons env args)))
	 ((setq args (cons (sgen "DEFINE-SETF-METHOD") args))
	  (push `(declare (ignore ,(car args))) body)))
   `(eval-when (compile eval load)
	       (si:putprop ',access-fn
			   #'(lambda ,args (block ,access-fn ,@body)) 'setf-method);(*setf-set* (setf-set))
	       (remprop ',access-fn 'setf-lambda)
	       (remprop ',access-fn 'setf-update-fn)
	       (si:putprop ',access-fn
			   ,(find-documentation (cdr rest))
			   'setf-documentation)
	       ',access-fn)))

(defmacro define-setf-expander (access-fn &rest rest)
  (declare (optimize (safety 2)))
  `(define-setf-method ,access-fn ,@rest))

;;; GET-SETF-METHOD.
;;; It just calls GET-SETF-METHOD-MULTIPLE-VALUE
;;;  and checks the number of the store variable.
(defun get-setf-method (form &optional env)
  (multiple-value-bind (vars vals stores store-form access-form)
      (get-setf-method-multiple-value form env)
    (unless (= (list-length stores) 1)
	    (error "Multiple store-variables are not allowed."))
    (values vars vals stores store-form access-form)))

(defun get-setf-expansion (form &optional env)
  (declare (optimize (safety 2)))
  (get-setf-method form env))

;;;; GET-SETF-METHOD-MULTIPLE-VALUE.

;; FIXME  when all is well, remove this and the setf tests in the pcl directory
(push :setf *features*)

(defun get-setf-method-multiple-value (form &optional env &aux tem); (*setf-set* (setf-set))
  (flet ((mvars (form) (mapcar (lambda (x) (declare (ignore x)) (setf-gensym)) (cdr form))))
    (cond ((symbolp form)
	   (let ((store (setf-gensym)))
	     (values nil nil (list store) `(setq ,form ,store) form)))
	  ((or (not (consp form)) (not (symbolp (car form))))
	   (error "Cannot get the setf-method of ~S." form))
	  ((multiple-value-bind
		 (t1 exp) (macroexpand form env)
	     (when exp (setq tem t1)))
	   (get-setf-method-multiple-value tem env))
	  ((get (car form) 'setf-method)
	   (apply (get (car form) 'setf-method) env (cdr form)))
	  ((or (get (car form) 'setf-update-fn)
	       (setq tem (get (car form) 'si::structure-access)))
	   (let ((vars (mvars form))
		 (store (setf-gensym)))
	     (values vars (cdr form) (list store)
	             (cond (tem (setf-structure-access (car vars) (car tem) (cdr tem) store))
			   ((let ((f (get (car form) 'setf-update-fn)))
			      `(,f ,@vars ,store))))
		     (cons (car form) vars))))
	  ((get (car form) 'setf-lambda)
	   (let* ((vars (mvars form))
		  (store (setf-gensym))
		  (f (get (car form) 'setf-lambda)))
	     (values vars (cdr form) (list store)
		     (funcall (apply f vars) store)
		     (cons (car form) vars))))
	  ((macro-function (car form))
	   (get-setf-method-multiple-value (macroexpand form env)))
	  (t
	   (let ((vars (mvars form))
		 (store (setf-gensym)))
	     (values vars (cdr form) (list store)
	             `(funcall
		       #'(setf ,(car form))
		       ,store ,@vars )
		     (cons (car form) vars)))))))


;;;; SETF definitions.

(defsetf car (x) (y) `(progn (rplaca ,x ,y) ,y))
(defsetf cdr (x) (y) `(progn (rplacd ,x ,y), y))
(defsetf caar (x) (y) `(progn (rplaca (car ,x) ,y) ,y))
(defsetf cdar (x) (y) `(progn (rplacd (car ,x) ,y) ,y))
(defsetf cadr (x) (y) `(progn (rplaca (cdr ,x) ,y) ,y))
(defsetf cddr (x) (y) `(progn (rplacd (cdr ,x) ,y) ,y))
(defsetf caaar (x) (y) `(progn (rplaca (caar ,x) ,y) ,y))
(defsetf cdaar (x) (y) `(progn (rplacd (caar ,x) ,y) ,y))
(defsetf cadar (x) (y) `(progn (rplaca (cdar ,x) ,y) ,y))
(defsetf cddar (x) (y) `(progn (rplacd (cdar ,x) ,y) ,y))
(defsetf caadr (x) (y) `(progn (rplaca (cadr ,x) ,y) ,y))
(defsetf cdadr (x) (y) `(progn (rplacd (cadr ,x) ,y) ,y))
(defsetf caddr (x) (y) `(progn (rplaca (cddr ,x) ,y) ,y))
(defsetf cdddr (x) (y) `(progn (rplacd (cddr ,x) ,y) ,y))
(defsetf caaaar (x) (y) `(progn (rplaca (caaar ,x) ,y) ,y))
(defsetf cdaaar (x) (y) `(progn (rplacd (caaar ,x) ,y) ,y))
(defsetf cadaar (x) (y) `(progn (rplaca (cdaar ,x) ,y) ,y))
(defsetf cddaar (x) (y) `(progn (rplacd (cdaar ,x) ,y) ,y))
(defsetf caadar (x) (y) `(progn (rplaca (cadar ,x) ,y) ,y))
(defsetf cdadar (x) (y) `(progn (rplacd (cadar ,x) ,y) ,y))
(defsetf caddar (x) (y) `(progn (rplaca (cddar ,x) ,y) ,y))
(defsetf cdddar (x) (y) `(progn (rplacd (cddar ,x) ,y) ,y))
(defsetf caaadr (x) (y) `(progn (rplaca (caadr ,x) ,y) ,y))
(defsetf cdaadr (x) (y) `(progn (rplacd (caadr ,x) ,y) ,y))
(defsetf cadadr (x) (y) `(progn (rplaca (cdadr ,x) ,y) ,y))
(defsetf cddadr (x) (y) `(progn (rplacd (cdadr ,x) ,y) ,y))
(defsetf caaddr (x) (y) `(progn (rplaca (caddr ,x) ,y) ,y))
(defsetf cdaddr (x) (y) `(progn (rplacd (caddr ,x) ,y) ,y))
(defsetf cadddr (x) (y) `(progn (rplaca (cdddr ,x) ,y) ,y))
(defsetf cddddr (x) (y) `(progn (rplacd (cdddr ,x) ,y) ,y))
(defsetf first (x) (y) `(progn (rplaca ,x ,y) ,y))
(defsetf second (x) (y) `(progn (rplaca (cdr ,x) ,y) ,y))
(defsetf third (x) (y) `(progn (rplaca (cddr ,x) ,y) ,y))
(defsetf fourth (x) (y) `(progn (rplaca (cdddr ,x) ,y) ,y))
(defsetf fifth (x) (y) `(progn (rplaca (cddddr ,x) ,y) ,y))
(defsetf sixth (x) (y) `(progn (rplaca (nthcdr 5 ,x) ,y) ,y))
(defsetf seventh (x) (y) `(progn (rplaca (nthcdr 6 ,x) ,y) ,y))
(defsetf eighth (x) (y) `(progn (rplaca (nthcdr 7 ,x) ,y) ,y))
(defsetf ninth (x) (y) `(progn (rplaca (nthcdr 8 ,x) ,y) ,y))
(defsetf tenth (x) (y) `(progn (rplaca (nthcdr 9 ,x) ,y) ,y))
(defsetf rest (x) (y) `(progn (rplacd ,x ,y) ,y))
(defsetf svref si:svset)
(defsetf elt si::elt-set)
(defsetf symbol-value set)
(defsetf symbol-function si::fset)
(defsetf macro-function (s &optional env) (v) `(let ((env ,env)) (declare (ignorable env)) (si:fset ,s (cons 'macro ,v)) ,v))
;; (defun aset-wrap (x &rest r &aux v)
;;   (declare (:dynamic-extent r)) 
;;   (setq r (nreverse r) v (pop r) r (nreverse r)) 
;;   (apply 'si:aset v x r))
(defsetf aref (x &rest r) (v) `(si::aset ,v ,x ,@r))
;(defsetf aref aset-wrap)
(defsetf get put-aux)
(defmacro put-aux (a b &rest l)
  `(si::sputprop ,a ,b (progn ,@l)))
;  `(si::sputprop ,a ,b ,(car (last l))))
(defsetf nth (n l) (v) `(progn (rplaca (nthcdr ,n ,l) ,v) ,v))
(defsetf char si::char-set)
(defsetf schar si::schar-set)
;(defsetf bit aset-wrap)
;(defsetf sbit aset-wrap)
(defsetf bit (x &rest r) (v) `(baset ,v ,x ,@r))
(defsetf sbit (x &rest r) (v) `(sbaset ,v ,x ,@r))
(defsetf fill-pointer c-set-adjvector-fillp)
;(defsetf symbol-plist si:set-symbol-plist)
(defsetf symbol-plist (x) (y) `(c-set-symbol-plist ,x ,y))
(defsetf gethash (k h &optional d) (v) `(progn ,d (si:hash-set ,k ,h ,v)))
(defsetf row-major-aref si::aset1)
(defsetf readtable-case si::set-readtable-case)


(define-setf-method getf (&environment env place indicator &optional default)
  (let ((itemp (setf-gensym))(store (setf-gensym))(def-temp (if default (setf-gensym))))
    (multiple-value-bind
	  (vars vals stores store-form access-form)
	(get-setf-method place env)
      (values `(,@vars ,itemp ,@(if default `(,def-temp)))
	      `(,@vals ,indicator ,@(if default `(,default)))
	      (list store)
	      `(let ((,(car stores) (si:put-f ,access-form ,store ,itemp)))
		 ,store-form
		 ,store)
	      `(getf ,access-form ,itemp ,@(if default `(,def-temp)))))))

(defsetf subseq (sequence1 start1 &optional end1)
		(sequence2)
  `(progn (replace ,sequence1 ,sequence2 :start1 ,start1 :end1 ,end1) ,sequence2))

(define-setf-method the (&environment env type form)
  (let ((store (setf-gensym)))
    (multiple-value-bind (vars vals stores store-form access-form)
	(get-setf-method form env)
      (values vars vals (list store)
	      `(let ((,(car stores) (the ,type ,store))) ,store-form)
	      `(the ,type ,access-form)))))

#|
(define-setf-method apply (&environment env fn &rest rest)
  (unless (and (consp fn) (eq (car fn) 'function) (symbolp (cadr fn))
	       (null (cddr fn)))
	  (error "Can't get the setf-method of ~S." fn))
  (multiple-value-bind (vars vals stores store-form access-form)
      (get-setf-expansion (cons (cadr fn) rest) env)
    (unless (eq (car (last store-form)) (car (last vars)))
            (error "Can't get the setf-method of ~S." fn))
    (values vars vals stores
	    `(apply #',(car store-form) ,@(cdr store-form))
	    `(apply #',(cadr fn) ,@(cdr access-form)))))
|#

(define-setf-method apply (&environment env fn &rest rest)
  (unless (and (consp fn)
               (or (eq (car fn) 'function) (eq (car fn) 'quote))
               (symbolp (cadr fn))
               (null (cddr fn)))
    (error "Can't get the setf-method of ~S." fn))
  (multiple-value-bind (vars vals stores store-form access-form)
      (get-setf-expansion (cons (cadr fn) rest) env)
    (cond ((eq (car (last store-form)) (car (last vars)))
           (values vars vals stores
                   `(apply #',(car store-form) ,@(cdr store-form))
                   `(apply #',(cadr fn) ,@(cdr access-form))))
          ((eq (car (last (butlast store-form))) (car (last vars)))
           (values vars vals stores
                   `(apply #',(car store-form)
                           ,@(cdr (butlast store-form 2))
                           (append ,(car (last (butlast store-form)))
                                   (list ,(car (last store-form)))))
                   `(apply #',(cadr fn) ,@(cdr access-form))))
          (t (error "Can't get the setf-method of ~S." fn)))))

(define-setf-method char-bit (&environment env char name)
  (let ((ntemp (setf-gensym))(store (setf-gensym)))
    (multiple-value-bind (temps vals stores store-form access-form)
	(get-setf-method char env)
      (values `(,ntemp ,@temps)
	      `(,name ,@vals)
	      (list store)
	      `(let ((,(first stores) (set-char-bit ,access-form ,ntemp ,store)))
	         ,store-form ,store)
	      `(char-bit ,access-form ,ntemp)))))

(define-setf-method ldb (&environment env bytespec int)
  (let ((btemp (setf-gensym))(store (setf-gensym)))
    (multiple-value-bind (temps vals stores store-form access-form)
	(get-setf-method int env)
      (values `(,btemp ,@temps)
	      `(,bytespec ,@vals)
	      (list store)
	      `(let ((,(first stores) (dpb ,store ,btemp ,access-form)))
	         ,store-form ,store)
	      `(ldb ,btemp ,access-form)))))

(define-setf-method mask-field (&environment env bytespec int)
  (let ((btemp (setf-gensym))(store (setf-gensym)))
    (multiple-value-bind (temps vals stores store-form access-form)
	(get-setf-method int env)
      (values `(,btemp ,@temps)
	      `(,bytespec ,@vals)
	      (list store)
	      `(let ((,(first stores) (deposit-field ,store ,btemp ,access-form)))
	         ,store-form ,store)
	      `(mask-field ,btemp ,access-form)))))

(defun setf-expand-values (places newvalue env)
  (let* ((syms (mapcar (lambda (x) (declare (ignore x)) (setf-gensym)) places))
	 (expns (mapcar (lambda (x y) (setf-expand-1 x y env)) places syms))
	 binds decls ctps alist
	 (setters (mapcar (lambda (x)
			    (cond ((when (consp x) (eq (car x) 'let*))
				   (mapc (lambda (x) (if (when (consp x) (member (cadr x) syms))
							 (push (cons (car x) (cadr x)) alist)
							 (push x binds)))
					 (cadr x))
				   (multiple-value-bind (doc dec ctp body) (parse-body-header (cddr x))
				     (declare (ignore doc));FIXME?
				     (setq decls (nconc decls dec) ctps (nconc ctps ctp))
				     `(progn ,@body)))
				  (x)))
			  expns)))
    `(let* ,(nreverse binds) ,@decls ,@ctps (multiple-value-bind ,syms ,newvalue (values ,@(sublis alist setters))))))

;;; The expansion function for SETF.
(defun setf-expand-1 (place newvalue env &aux g)
  (when (and (consp place) (eq (car place) 'the))
        (return-from setf-expand-1
          (setf-expand-1 (caddr place) `(the ,(cadr place) ,newvalue) env)))
  (when (and (consp place) (eq (car place) 'values))
    (return-from setf-expand-1 (setf-expand-values (cdr place) newvalue env)))
  (when (symbolp place)
        (return-from setf-expand-1 `(setq ,place ,newvalue)))
  (when (and (consp place)
	       (not (or (get (car place) 'setf-lambda)
			(get (car place) 'setf-update-fn))))
	  (multiple-value-setq (place g) (macroexpand place env))
	  (if g (return-from setf-expand-1 (setf-expand-1 place newvalue env))))
  (when (and (symbolp (car place)) (setq g (get (car place) 'setf-update-fn)))
    (return-from setf-expand-1 `(,g ,@(cdr place) ,newvalue)))
  (cond ((and (symbolp (car place))
	      (setq g (get (car place) 'structure-access)))
	 (return-from setf-expand-1
	   (setf-structure-access (cadr place) (car g) (cdr g) newvalue))))
	     
  (multiple-value-bind (vars vals stores store-form access-form)
      (get-setf-expansion place env)
    (declare (ignore access-form))
    `(let* ,(mapcar 'list (append vars stores) (append vals (list newvalue)))
       (declare (ignorable ,@vars))
       ,store-form)))

(defun setf-structure-access (struct type index newvalue)
  (case type
    (list `(setf (nth ,index ,struct) ,newvalue))
;    (list `(si:rplaca-nthcdr ,struct ,index ,newvalue))
    (vector `(si::elt-set ,struct ,index ,newvalue))
    (t `(str-refset ,struct ',type ,index ,newvalue))));si::structure-set

(defun setf-expand (l env)
  (cond ((endp l) nil)
        ((endp (cdr l)) (error "~S is an illegal SETF form." l))
        (t
         (cons (setf-expand-1 (car l) (cadr l) env)
               (setf-expand (cddr l) env)))))


;;; SETF macro.

;; (defun setf-helper (rest env)
;;   (setq rest (cdr rest))
;;   (cond ((endp rest) nil)
;; ;        ((endp (cdr rest)) (error "~S is an illegal SETF form." rest))
;;         ((endp (cddr rest)) (setf-expand-1 (car rest) (cadr rest) env))
;;         (t (cons 'progn (setf-expand rest env)))))

;; ;(setf (macro-function 'setf) 'setf-help)
;; (si::fset 'setf (cons 'macro (symbol-function 'setf-helper)))

(defmacro setf (&environment env &rest rest &aux (*gensym-counter* 0))
  (cond ((endp rest) nil)
;        ((endp (cdr rest)) (error "~S is an illegal SETF form." rest))
        ((endp (cddr rest)) (setf-expand-1 (car rest) (cadr rest) env))
        ((cons 'progn (setf-expand rest env)))))

;;; PSETF macro.

(defmacro psetf (&environment env &rest rest &aux (*gensym-counter* 0))
  (declare (optimize (safety 2)))
  (cond ((endp rest) nil)
        ((endp (cdr rest)) (error "~S is an illegal PSETF form." rest))
        ((endp (cddr rest))
         `(progn ,(setf-expand-1 (car rest) (cadr rest) env)
                 nil))
        (t
	 (do ((r rest (cddr r))
	      (pairs nil)
	      (store-forms nil))
	     ((endp r)
	      `(let* ,pairs
		 ,@(nreverse store-forms);FIXME put in ignorable decl here
		 nil))
	   (when (endp (cdr r)) (error "~S is an illegal PSETF form." rest))
	   (multiple-value-bind (vars vals stores store-form access-form)
	       (get-setf-expansion (car r) env)
             (declare (ignore access-form))
	     (setq store-forms (cons store-form store-forms))
	     (setq pairs
		   (nconc pairs
			  (mapcar 'list (append vars stores) (append vals (list (cadr r)))))))))))


;;; SHIFTF macro.
(defmacro shiftf (&environment env &rest rest &aux (*gensym-counter* 0))
  (declare (optimize (safety 2)))
  (do ((r rest (cdr r))
       (pairs nil)
       (stores nil)
       (store-forms nil)
       (g (setf-gensym))
       (access-forms nil))
      ((endp (cdr r))
       (setq stores (nreverse stores))
       (setq store-forms (nreverse store-forms))
       (setq access-forms (nreverse access-forms))
       `(let* ,(nconc pairs
		      (list (list g (car access-forms)))
		      (mapcar 'list stores (cdr access-forms))
		      (list (list (car (last stores)) (car r))))
	  ,@store-forms
	  ,g))
    (multiple-value-bind (vars vals stores1 store-form access-form)
	(get-setf-method (car r) env)
      (setq pairs (nconc pairs (mapcar 'list vars vals)))
      (setq stores (cons (car stores1) stores))
      (setq store-forms (cons store-form store-forms))
      (setq access-forms (cons access-form access-forms)))))


;;; ROTATEF macro.
(defmacro rotatef (&environment env &rest rest &aux  (*gensym-counter* 0))
  (declare (optimize (safety 2)))
  (do ((r rest (cdr r))
       (pairs nil)
       (stores nil)
       (store-forms nil)
       (access-forms nil))
      ((endp r)
       (setq stores (nreverse stores))
       (setq store-forms (nreverse store-forms))
       (setq access-forms (nreverse access-forms))
       (when store-forms
	 `(let* ,(nconc pairs
		      (mapcar 'list stores (cdr access-forms))
		      (list (list (car (last stores)) (car access-forms))))
	    ,@store-forms
	    nil
	    )))
    (multiple-value-bind (vars vals stores1 store-form access-form)
	(get-setf-method (car r) env)
      (setq pairs (nconc pairs (mapcar 'list vars vals)))
      (setq stores (cons (car stores1) stores))
      (setq store-forms (cons store-form store-forms))
      (setq access-forms (cons access-form access-forms)))))


(defmacro define-modify-macro (name lambda-list function
			       &optional doc-string
			       &aux (r (member '&rest lambda-list)))
  (declare (optimize (safety 2)))
  (let* ((update-form
	  ``((lambda (&rest ,rest-var)
	       (declare (dynamic-extent ,rest-var));GCL will not cons this list anyway, but to be safe
	       (apply ',',function ,access-form ,rest-var))
	     ,,@(mapcan (lambda (x) (unless (eq x '&optional) (list (if (atom x) x (car x)))))
			   (ldiff-nf lambda-list r));FIXME
	     ,@,(cadr r))))
    `(defmacro ,name (&environment env reference . ,lambda-list)
       ,@(when doc-string `(,doc-string))
       (let ((*gensym-counter* 0)(rest-var (setf-gensym)))
	 (when (symbolp reference)
           (return-from ,name
             (let ((access-form reference))
	       `(setq ,reference ,,update-form))))
	 (multiple-value-bind (vars vals stores store-form access-form)
	     (get-setf-expansion reference env)
	   `(let* ,(mapcar 'list (append vars stores) (append vals (list ,update-form)))
	      (declare (ignorable ,@vars))
	      ,store-form))))))



;;; Some macro definitions.

;;; (defmacro remf (&environment env place indicator)
;;;  (multiple-value-bind (vars vals stores store-form access-form)
;;;      (get-setf-method place env)
;;;    `(let* ,(mapcar #'list vars vals)
;;;       (multiple-value-bind (,(car stores) flag)
;;;           (si:rem-f ,access-form ,indicator)
;;;         ,store-form
;;;         flag))))

;;; This definition was obtained from SBCL
(defmacro remf (&environment env place indicator &aux  (*gensym-counter* 0))
  (declare (optimize (safety 2)))
  (let* ((ind-temp (setf-gensym))(local1 (setf-gensym))(local2 (setf-gensym)))
    (multiple-value-bind (dummies vals newval setter getter)
	(get-setf-method place env)
      (do* ((d dummies (cdr d))
            (v vals (cdr v))
            (let-list nil))
           ((null d)
            ;; See ANSI 5.1.3 for why we do out-of-order evaluation
            (push (list ind-temp indicator) let-list)
            (push (list (car newval) getter) let-list)
            `(let* ,(nreverse let-list)
	       (do ((,local1 ,(car newval) (cddr ,local1))
                    (,local2 nil ,local1))
                   ((atom ,local1) nil)
		 (cond ((atom (cdr ,local1))
			(error "Odd-length property list in REMF."))
		       ((eq (car ,local1) ,ind-temp)
			(cond (,local2
			       (rplacd (cdr ,local2) (cddr ,local1))
			       (return t))
			      (t (setq ,(car newval) (cddr ,(car newval)))
				 ,setter
				 (return t))))))))
	(push (list (car d) (car v)) let-list)))))

(define-modify-macro incf (&optional (delta 1)) +)
(define-modify-macro decf (&optional (delta 1)) -)

(defmacro push (&environment env item place &aux  (*gensym-counter* 0))
  (declare (optimize (safety 2)))
  (let ((myitem (setf-gensym)))
    (when (symbolp place)
      (return-from push `(let* ((,myitem ,item))
			   (setq ,place (cons ,myitem ,place)))))
    (multiple-value-bind (vars vals stores store-form access-form)
	(get-setf-method place env)
      `(let* ,(mapcar 'list (append (list myitem) vars stores) (append (list   item) vals (list (list 'cons myitem access-form))))
	 (declare (ignorable ,@vars))
	 ,store-form))))

(defmacro pushnew (&environment env item place &rest rest &aux  (*gensym-counter* 0))
  (declare (optimize (safety 2)))
  (let ((myitem (setf-gensym)))
    (cond ((symbolp place)
	   (return-from pushnew `(let* ((,myitem ,item))
				   (setq ,place (adjoin ,myitem ,place ,@rest))))))
    (multiple-value-bind (vars vals stores store-form access-form)
	(get-setf-method place env)
      `(let* ,(mapcar 'list (append (list myitem) vars stores)
		    (append (list item) vals  (list (list* 'adjoin myitem access-form rest))))
	 (declare (ignorable ,@vars))
	 ,store-form))))

(defmacro pop (&environment env place &aux  (*gensym-counter* 0))
    (declare (optimize (safety 2)))
    (when (symbolp place)
      (return-from pop
        (let ((temp (setf-gensym)))
          `(let ((,temp (car ,place)))
             (setq ,place (cdr ,place))
             ,temp))))
    (multiple-value-bind (vars vals stores store-form access-form)
	(get-setf-method place env)
      `(let* ,(mapcar 'list (append vars stores) (append vals (list (list 'cdr access-form))))
	 (declare (ignorable ,@vars))
	 (prog1 (car ,access-form)
           ,store-form))))

  (defun fdefinition (n)
  (declare (optimize (safety 2)))
  (let ((n (funid-sym n)))
    (if (fboundp n)
	(symbol-function n)
      (error 'undefined-function :name n))))

(defun (setf fdefinition) (def n)
  (declare (optimize (safety 2)))
  (check-type def function)
  (let ((n (funid-sym n)))
    (assert (not (special-operator-p n)))
    (setf (symbol-function n) def)))
