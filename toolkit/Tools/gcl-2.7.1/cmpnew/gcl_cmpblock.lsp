;;; CMPBLOCK  Block and Return-from.
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

(si:putprop 'block 'c1block 'c1special)
(si:putprop 'block 'c2block 'c2)

(si:putprop 'return-from 'c1return-from 'c1special)
(si:putprop 'return-from 'c2return-from 'c2)

(defstruct (blk (:print-function (lambda (x s i) (s-print 'blk (blk-name x) (si::address x) s))))
           name			;;; Block name.
           ref			;;; Referenced or not.  T or NIL.
           ref-clb		;;; Cross local function reference.
           			;;; During Pass1, T or NIL.
           			;;; During Pass2, the vs-address for the
           			;;; block id, or NIL.
           ref-ccb		;;; Cross closure reference.
           			;;; During Pass1, T or NIL.
           			;;; During Pass2, the ccb-vs for the
           			;;; block id, or NIL.
           exit			;;; Where to return.  A label.
           value-to-go		;;; Where the value of the block to go.
           var			;;; The block name holder.  Used only in
           			;;; the error message.
	   type
           )

(si::freeze-defstruct 'blk)

(defvar *blocks* nil)

;;; During Pass 1, *blocks* holds a list of blk objects and the symbols 'CB'
;;; (Closure Boundary) and 'LB' (Level Boundary).  'CB' will be pushed on
;;; *blocks* when the compiler begins to process a closure.  'LB' will be
;;; pushed on *blocks* when *level* is incremented.


(defun ref-blks (form blks)
  (ref-obs form blks 
	   (lambda (x) (setf (blk-ref-ccb x) t))
	   (lambda (x) (setf (blk-ref-clb x) t))
	   (lambda (x) (setf (blk-ref x) t))))


(defun prune-mch (l &optional tag-conflict-p)
  (remove-if (lambda (x &aux (v (pop x))(tp (pop x))(st (pop x))(m (car x)))
	       (and (type<= (var-type v) tp)
		    (or (when tag-conflict-p (cdr st)) (subsetp (var-store v) st))
		    (if m (equal tp m) t)))
	     l))

(defvar *c1exit* nil)

(defun make-c1exit (n)
  (cons n (current-env)))

(defun c1block (args &aux (info (make-info))(*c1exit* (cons (make-c1exit (car args)) *c1exit*)))
  (when (endp args) (too-few-args 'block 1 0))
  (cmpck (not (symbolp (car args)))
         "The block name ~s is not a symbol." (car args))
  (let* ((blk (make-blk :name (car args) :ref nil :ref-ccb nil :ref-clb nil :exit *c1exit*
			:var (mapcan (lambda (x) (when (var-p x) (list (list x nil nil nil)))) *vars*)))
         (body (let ((*blocks* (cons blk *blocks*))) (c1progn (cdr args)))))
    (when (info-type (cadr body))
      (or-mch (prune-mch (blk-var blk))))
    (labels ((nb (b) (if (and (eq (car b) 'return-from) (eq blk (caddr b))) (nb (seventh b)) b)))
      (setq body (nb body)))
    (add-info info (cadr body))
    (setf (info-type info) (type-or1 (info-type (cadr body)) (blk-type blk)))
    (ref-blks body (list blk))
    (when (or (blk-ref-ccb blk) (blk-ref-clb blk))
      (set-volatile info))
    (when (info-type info)
      (mapc (lambda (x &aux (v (pop x))(tp (pop x))(st (pop x))(m (car x))
			 (tp (type-and tp (var-dt v))));FIXME, unnecessary?
	      (unless (and (type= tp (var-type v))
			   (subsetp st (var-store v)) (subsetp (var-store v) st)
			   (if m (equal m tp) t))
		(keyed-cmpnote (list (var-name v) 'block-set)
			       "Altering ~s at end of block ~s:~%   type from ~s to ~s,~%   store from ~s to ~s"
			       v (blk-name blk) (cmp-unnorm-tp (var-type v)) (cmp-unnorm-tp tp)
			       (var-store v) st)
		(do-setq-tp v '(blk-set) tp)
		(push-vbinds v st)))
	    (blk-var blk)))
    (cond ((or (blk-ref-ccb blk) (blk-ref-clb blk) (blk-ref blk))(list 'block info blk body))
	  (body))))


(defun c2block (blk body)
  (cond ((blk-ref-ccb blk) (c2block-ccb blk body))
        ((blk-ref-clb blk) (c2block-clb blk body))
        (t (c2block-local blk body))))

(defun c2block-local (blk body)
  (setf (blk-exit blk) *exit*)
  (setf (blk-value-to-go blk) *value-to-go*)
  (c2expr body))

(defun c2block-clb (blk body &aux (*vs* *vs*))
  (setf (blk-exit blk) *exit*)
  (setf (blk-value-to-go blk) *value-to-go*)
  (setf (blk-ref-clb blk) (vs-push))
  (wt-nl)
  (add-libc "setjmp")
  (setq *frame-used* t)
  (wt-vs (blk-ref-clb blk))
  (wt "=alloc_frame_id();")
  (wt-nl "frs_push(FRS_CATCH,") (wt-vs (blk-ref-clb blk)) (wt ");")
  (wt-nl "if(nlj_active)")
  (wt-nl "{nlj_active=FALSE;frs_pop();")
  (unwind-exit 'fun-val 'jump)
  (wt "}")
  (wt-nl "else{")
  (let ((*unwind-exit* (cons 'frame *unwind-exit*))) (c2expr body))
  (wt-nl "}")
  )

(defun c2block-ccb (blk body &aux (*vs* *vs*) (*clink* *clink*)
                                  (*ccb-vs* *ccb-vs*))
  (setf (blk-exit blk) *exit*)
  (setf (blk-value-to-go blk) *value-to-go*)
  (setf (blk-ref-clb blk) (vs-push))
  (setf (blk-var blk) (blk-name blk))
  (wt-nl) (wt-vs (blk-ref-clb blk)) (wt "=alloc_frame_id();")
  (wt-nl)
  (clink (blk-ref-clb blk))
  (setf (blk-ref-ccb blk) (ccb-vs-push))
  (add-libc "setjmp")
  (setq *frame-used* t)
  (wt-nl "frs_push(FRS_CATCH,") (wt-vs* (blk-ref-clb blk)) (wt ");")
  (wt-nl "if(nlj_active)")
  (wt-nl "{nlj_active=FALSE;frs_pop();")
  (unwind-exit 'fun-val 'jump)
  (wt "}")
  (wt-nl "else{")
  (let ((*unwind-exit* (cons 'frame *unwind-exit*))) (c2expr body))
  (wt-nl "}")
  )

(defun c1return-from (args &aux (name (car args)) ccb clb inner)
  (cond ((endp args) (too-few-args 'return-from 1 0))
        ((and (not (endp (cdr args))) (not (endp (cddr args))))
         (too-many-args 'return-from 2 (length args)))
        ((not (symbolp (car args)))
         "The block name ~s is not a symbol." (car args)))
  (dolist (blk *blocks* (cmperr "The block ~s is undefined." name))
    (case blk
	  (cb (setq ccb t inner (or inner 'cb)))
	  (lb (setq clb t inner (or inner 'lb)))
	  (t (when (when (eq (blk-name blk) name) (not (member blk *lexical-env-mask*)))
	       (let* ((*c1exit* (blk-exit blk))
		      (val (c1expr (cadr args)))
		      (c1fv (when ccb (c1inner-fun-var))))
		 (setf (blk-type blk) (type-or1 (blk-type blk) (info-type (cadr val))))
		 (when (info-type (cadr val)) (or-mch (prune-mch (blk-var blk))))
		 (return (list 'return-from
			       (let ((info (copy-info (cadr val))))
				 (setf (info-type info) nil)
				 (cond (ccb (pushnew blk (info-ref-ccb info)))
				       (clb (pushnew blk (info-ref-clb info)))
				       ((pushnew blk (info-ref info))))
				 (when c1fv (add-info info (cadr c1fv)))
				 info)
			       blk ccb clb c1fv val))))))));FIXME infer-tp here, or better in blk-var-null, etc.


(defun c2return-from (blk ccb clb c1fv val)
  (declare (ignore c1fv))
  (cond (ccb (c2return-ccb blk val))
        (clb (c2return-clb blk val))
        (t (c2return-local blk val))))

(defun c2return-local (blk val)
  (let ((*value-to-go* (blk-value-to-go blk))
        (*exit* (blk-exit blk)))
       (c2expr val)))

(defun c2return-clb (blk val)
  (let ((*value-to-go* 'top)) (c2expr* val))
  (wt-nl "unwind(frs_sch(")
  (if (blk-ref-ccb blk) (wt-vs* (blk-ref-clb blk)) (wt-vs (blk-ref-clb blk)))
  (wt "),Cnil);")
  (unwind-exit nil))

(defun c2return-ccb (blk val)
  (wt-nl "{frame_ptr fr;")
  (wt-nl "fr=frs_sch(") (wt-ccb-vs (blk-ref-ccb blk)) (wt ");")
  (wt-nl "if(fr==NULL) FEerror(\"The block ~s is missing.\",1," (vv-str (blk-var blk)) ");")
  (let ((*value-to-go* 'top)) (c2expr* val))
  (wt-nl "unwind(fr,Cnil);}")
  (unwind-exit nil))
