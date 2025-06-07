;;; CMPLAM  Lambda expression.
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

;;; During Pass1, a lambda-list
;;;
;;; (	{ var }*
;;; 	[ &optional { var | ( var [ initform [ svar ] ] ) }* ]
;;; 	[ &rest var ]
;;; 	[ &key { var | ( { var | ( kwd var ) } [initform [ svar ]])}*
;;; 		[&allow-other-keys]]
;;; 	[ &aux {var | (var [initform])}*]
;;; )
;;;
;;; is transformed into
;;;
;;; (	( { var }* )				; required
;;; 	( { (var initform svar) }* )		; optional
;;; 	{ var | nil }				; rest
;;; 	key-flag
;;; 	( { ( kwd-vv-index var initform svar) }* )	; key
;;; 	allow-other-keys-flag
;;; )
;;;
;;; where
;;; 	svar:	  nil		; means svar is not supplied
;;;	        | var
;;;
;;; &aux parameters will be embedded into LET*.
;;;
;;; c1lambda-expr receives
;;;	( lambda-list { doc | decl }* . body )
;;; and returns
;;;	( lambda info-object lambda-list' doc body' )
;;;
;;; Doc is NIL if no doc string is supplied.
;;; Body' is body possibly surrounded by a LET* (if &aux parameters are
;;; supplied) and an implicit block.




(defun wfs-error ()
  (error "This error is not supposed to occur: Contact Schelter ~
    ~%wfs@math.utexas.edu"))

(defun decls-from-procls (ll procls body)
  (cond ((or (null procls) (eq (car procls) '*)
	     (null ll) (member (car ll) '(&whole &optional &rest &key &environment))) nil)
	((eq (car procls) t)
	 (decls-from-procls (cdr ll) (cdr procls) body))
	(t
	 (cons (list (car procls) (or (if (atom (car ll)) (car ll) (caar ll))))
	       (decls-from-procls (cdr ll) (cdr procls) body)))))
	 
(defun c1lambda-expr (args &aux (regs (pop args)) requireds tv
			   doc body ss is ts other-decls (ovars *vars*)
			   (*vars* *vars*) narg (info (make-info)) ctps)


  (multiple-value-setq (body ss ts is other-decls doc ctps) (c1body args t));FIXME parse-body-header
  
  (mapc (lambda (x &aux (y (c1make-var x ss is ts))) 
	  (setf (var-mt y) nil)
	  (push-var y nil) (push y requireds)) regs)
  (when (member +nargs+ ts :key 'car)
    (setq narg (list (c1make-var +nargs+ ss is ts))))
  (setq tv (append narg requireds))

  (c1add-globals ss)
  (check-vdecl (mapcar 'var-name tv) ts is)
  
  (setq body (c1decl-body other-decls body))
  (ref-vars body requireds)
  (dolist (var requireds) (check-vref var))
  
  (dolist (v requireds)
    (when (var-p v)
      (unless (type>= (var-type v) (var-mt v))
	(setf (var-type v) (var-mt v)))));FIXME?
  (let ((*vars* ovars)) (add-info info (cadr body)))
  (cond (*compiler-new-safety*
	 (mapc (lambda (x) (setf (var-type x) #tt)) requireds)
	 (let ((i (cadr body)))
	   (setf (info-type i) (if (single-type-p (info-type i)) #tt #t*))))
	((mapc (lambda (l) (setf (var-type l) (type-and (var-type l) (nil-to-t (cdr (assoc (var-name l) ctps)))))) tv)));FIXME?
  
  `(lambda ,info ,(list (nreverse requireds) narg) ,doc ,body))


(defvar *rest-on-stack* nil)  ;; non nil means put rest arg on C stack.

(defun need-to-set-vs-pointers (lambda-list)
				;;; On entry to in-line lambda expression,
				;;; vs_base and vs_top must be set iff,
   (or *safe-compile*
       *compiler-check-args*
       (nth 1 lambda-list)	;;; optional,
       (nth 2 lambda-list)	;;; rest, or
       (nth 3 lambda-list)	;;; key-flag.
       ))
