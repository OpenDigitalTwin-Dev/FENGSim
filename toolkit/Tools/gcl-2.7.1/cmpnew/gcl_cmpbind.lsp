;;; CMPBIND  Variable Binding.
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

(si:putprop 'bds-bind 'set-bds-bind 'set-loc)

;;; Those functions that call the following binding functions should
;;; rebind the special variables,
;;; *vs*, *clink*, *ccb-vs*, and *unwind-exit*.

(defvar *new-env* nil)

(defun c2bind (var)
  (case (var-kind var)
        (LEXICAL
         (when (var-ref-ccb var)
	   (wt-nl)
	   (clink (var-ref var))
	   (setf (var-ref-ccb var) (ccb-vs-push))))
        (SPECIAL
	 (setq *bds-used* t)
         (wt-nl "bds_bind(" (vv-str (var-loc var)) ",") (wt-vs (var-ref var))
         (wt ");")
         (push 'bds-bind *unwind-exit*))
        (t
	 (wt-nl "V" (var-loc var) "=")
	 (wt (or (cdr (assoc (var-kind var) +to-c-var-alist+)) (baboon)))
	 (wt "(") (wt-vs (var-ref var)) (wt ");"))))

(defun c2bind-loc (var loc)
  (case (var-kind var)
        (LEXICAL
         (cond ((var-ref-ccb var)
                (wt-nl)
                (clink (var-ref var) loc)
                (setf (var-ref-ccb var) (ccb-vs-push)))
               (t
                (wt-nl) (wt-vs (var-ref var)) (wt "= " loc ";"))))
        (SPECIAL
	 (setq *bds-used* t)
         (wt-nl "bds_bind(" (vv-str (var-loc var)) "," loc ");")
         (push 'bds-bind *unwind-exit*))
        (t
	 (wt-nl "V" (var-loc var) "= ")
	 (let ((wtf (cdr (assoc (var-kind var) +wt-loc-alist+))))
	   (unless wtf (baboon))
	   (funcall wtf loc))
	 (wt ";"))))

(defun c2bind-init (var init)
  (case (var-kind var)
        (LEXICAL
         (cond ((var-ref-ccb var)
                (let* ((loc (list 'vs (var-ref var)))
		       (*value-to-go* loc))
		  (c2expr* init))
                (clink (var-ref var))
                (setf (var-ref-ccb var) (ccb-vs-push)))
               ((let ((*value-to-go* (list 'vs (var-ref var))))
                     (c2expr* init)))))
        (SPECIAL
         (let* ((loc `(cvar ,(cs-push t))) (*value-to-go* loc))
	   (c2expr* init)
	   (c2bind-loc var loc)))
	(t
	 (let ((*value-to-go* (list 'var var nil)))
	   (unless (assoc (var-kind var) +wt-loc-alist+) (baboon));FIXME???
	   (c2expr* init)))))
