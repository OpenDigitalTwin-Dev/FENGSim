;;; CMPLABEL  Exit manager.
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

(defvar *last-label* 0)
(defvar *exit*)
(defvar *unwind-exit*)
(defvar *record-call-info* nil)

;;; *last-label* holds the label# of the last used label.
;;; *exit* holds an 'exit', which is
;;;	( label# . ref-flag ) or one of RETURNs (i.e. RETURN, RETURN-FIXNUM,
;;;	RETURN-CHARACTER, RETURN-LONG-FLOAT, RETURN-SHORT-FLOAT, or
;;;	RETURN-OBJECT).
;;; *unwind-exit* holds a list consisting of:
;;;	( label# . ref-flag ), one of RETURNs, TAIL-RECURSION-MARK, FRAME,
;;;	JUMP, BDS-BIND (each pushed for a single special binding), and
;;;	cvar (which holds the bind stack pointer used to unbind).

(defmacro next-label () `(cons (incf *last-label*) nil))

(defmacro next-label* () `(cons (incf *last-label*) t))

(defmacro wt-label (label)
  `(when (cdr ,label) (wt-nl "goto T" (car ,label) ";")(wt-nl1 "T" (car ,label) ":;")))

(defmacro wt-go (label)
  `(progn (rplacd ,label t) (wt "goto T" (car ,label) ";")(wt-nl)))


(defvar *restore-avma* nil)

(defun unwind-bds (bds-cvar bds-bind)
       (when (consp *inline-blocks*) (wt-nl "restore_avma; "))
       (when bds-cvar (wt-nl "bds_unwind(V" bds-cvar ");"))
       (dotimes (n bds-bind) (wt-nl "bds_unwind1;")))

(defun unwind-frames-bds (frames bds-cvar bds-bind)
  (dotimes (i frames) (wt-nl "frs_pop();"))
  (when (consp *inline-blocks*) (wt-nl "restore_avma; "))
  (when bds-cvar (wt-nl "bds_unwind(V" bds-cvar ");"))
  (dotimes (n bds-bind) (wt-nl "bds_unwind1;")))

(defun unwind-exit (loc &optional (jump-p nil) fname
                        &aux (*vs* *vs*) (bds-cvar nil) (bds-bind 0) type.wt (frames 0))
  (declare (fixnum bds-bind))
  (and *record-call-info* (record-call-info loc fname))
  (when (and (eq loc 'fun-val)
             (not (eq *value-to-go* 'return))
	     (not (rassoc *value-to-go* +return-alist+))
             (not (eq *value-to-go* 'top))
	     (not (multiple-values-p)));FIXME cleanup
        (wt-nl) (reset-top))
  (cond ((and (consp *value-to-go*) (eq (car *value-to-go*) 'jump-true))
         (set-jump-true loc (cadr *value-to-go*))
         (when (eq loc t) (return-from unwind-exit)))
        ((and (consp *value-to-go*) (eq (car *value-to-go*) 'jump-false))
         (set-jump-false loc (cadr *value-to-go*))
         (when (null loc) (return-from unwind-exit))))
  (dolist (ue *unwind-exit* (baboon))
   (cond
    ((consp ue)
     (cond ((eq ue *exit*)
	    (unless (and (consp *value-to-go*)
			 (or (eq (car *value-to-go*) 'jump-true)
			     (eq (car *value-to-go*) 'jump-false)))
	      (set-loc loc))
	    (unwind-frames-bds frames bds-cvar bds-bind)
	    (when jump-p
	      (when (consp *inline-blocks*) (wt-nl "restore_avma; "))
	      (wt-nl) (wt-go *exit*))
	    (return))
	   ;; Add (sup .var) handling in unwind-exit -- in
	   ;; c2multiple-value-prog1 and c2-multiple-value-call, apparently
	   ;; alone, c2expr-top is used to evaluate arguments, presumably to
	   ;; preserve certain states of the value stack for the purposes of
	   ;; retrieving the final results.  c2exprt-top rebinds sup, and
	   ;; vs_top in turn to the new sup, causing non-local exits to lose
	   ;; the true top of the stack vital for subsequent function
	   ;; evaluations.  We unwind this stack supremum variable change here
	   ;; when necessary.  CM 20040301
	   ((eq (car ue) 'sup)
	    (when (and ;; If we've pushed the sup, we've always reset vs_top, as we're
		       ;; using c2expr-top{*}.  Regardless then of whether we are
		       ;; explicitly unwinding a fun-val, we must reset the top, unless
		       ;; unless returning, when we rely on the returning code to leave
		       ;; the stack in the correct state, regardless of loc being a fun-val
		       ;; or otherwise.  We might need to reset when returning and loc is not
		       ;; fun-val, but this appears doubtful.  20040306 CM
		       ;; (eq loc 'fun-val)
		       (not (eq *value-to-go* 'return))
		       (not (rassoc *value-to-go* +return-alist+))
		       (not (eq *value-to-go* 'top)))
	      (wt-nl "sup=V" (cdr ue) ";")
	      (wt-nl)
	      (reset-top)))
	   ((setq jump-p t))))
    ((numberp ue) (setq bds-cvar ue bds-bind 0))
    ((eq ue 'bds-bind) (incf bds-bind))
    ((eq ue 'return)
     (unless (eq *exit* ue) (wfs-error))
     (set-loc loc)
     (unwind-frames-bds frames bds-cvar bds-bind)
     (wt-nl "return;")
     (return))
    ((eq ue 'frame) (incf frames))
    ((eq ue 'tail-recursion-mark))
    ((eq ue 'jump) (setq jump-p t))
    ((setq type.wt (assoc (car (rassoc ue +return-alist+)) +wt-loc-alist+))
     (unless (eq *exit* ue) (wfs-error))
     (cond (*mv-var*
	    (let* ((nv (cond ((and (consp fname) (eq (car fname) 'values)) (1- (cdr fname)))
			     ((or (not fname) (eq fname 'single-value)) 0)
			     ((abs (vald (get-return-type fname))))))
		   (nv (if (= nv (- multiple-values-limit 2)) 0 nv))
		   (fv (cs-push (car type.wt) t))
		   (lbs (mapcar (lambda (x) (declare (ignore x)) (cs-push t t)) (make-list (max 0 nv))))
		   (*value-to-go* (append (mapcar (lambda (x) (list 'cvar x)) (cons fv lbs)) '(trash))))
	      (wt-nl "{" (rep-type (car type.wt)) "V" fv ";")
	      (cond (lbs
		     (wt-nl "if (V" (var-loc *mv-var*) ") {")
		     (let ((i -1))
		       (mapc
			(lambda (x) 
			  (wt-nl "#define V" x " ((object *)V" (var-loc *mv-var*) ")[" (incf i) "]")) lbs))
		     (set-loc loc)
		     (mapc (lambda (x) (wt-nl "#undef V" x)) lbs)
		     (wt-nl "} else {")
		     (let ((*value-to-go* (list 'cvar fv))) (set-loc loc))
		     (wt-nl "}"))
		    ((set-loc loc)))
	      (when (or (eq loc 'fun-val) ;FIXME believe this is fixed now -- check;FIXME this can lead to a value stack leak on vs_top, e.g. typep with local mvfun tpi
			(and (consp loc)
			     (rassoc (car loc) +inline-types-alist+)
			     (flag-p (cadr loc) sets-vs-top)))
		(setq nv -2))
	      (unwind-frames-bds frames bds-cvar bds-bind)
	      (wt-nl "VMRV" *reservation-cmacro* "(V" fv "," nv ");}")))
	   ((let ((cvar (cs-push (car type.wt) t)))
	      (wt-nl "{" (rep-type (car type.wt)) "V" cvar " = ")
	      (funcall (cdr type.wt) loc)
	      (wt ";")
	      (unwind-frames-bds frames bds-cvar bds-bind)
	      (wt-nl "VMR" *reservation-cmacro* "(V" cvar ");}"))))
     (return))
    ((baboon)))))


(defun unwind-no-exit (exit &aux (bds-cvar nil) (bds-bind 0))
  (declare (fixnum bds-bind))
  (dolist (ue *unwind-exit* (baboon))
    (cond
       ((consp ue)
        (when (eq ue exit)
              (unwind-bds bds-cvar bds-bind)
              (return))
	;; Add (sup .var) handling in unwind-exit -- in
	;; c2multiple-value-prog1 and c2-multiple-value-call, apparently
	;; alone, c2expr-top is used to evaluate arguments, presumably to
	;; preserve certain states of the value stack for the purposes of
	;; retrieving the final results.  c2exprt-top rebinds sup, and
	;; vs_top in turn to the new sup, causing non-local exits to lose
	;; the true top of the stack vital for subsequent function
	;; evaluations.  We unwind this stack supremum variable change here
	;; when necessary.  CM 20040301
	(when (eq (car ue) 'sup)
	  (wt-nl "sup=V" (cdr ue) ";")
	  (wt-nl)
	  (reset-top)))
       ((numberp ue) (setq bds-cvar ue bds-bind 0))
       ((eq ue 'bds-bind) (incf bds-bind))
       ((or (eq ue 'return) (rassoc ue +return-alist+))
        (cond ((eq exit ue) (unwind-bds bds-cvar bds-bind)
                            (return))
              (t (baboon)))
        ;;; Never reached
        )
       ((eq ue 'frame) (wt-nl "frs_pop();"))
       ((eq ue 'tail-recursion-mark)
        (cond ((eq exit 'tail-recursion-mark) (unwind-bds bds-cvar bds-bind)
                                              (return)))
        ;;; Never reached
        )
       ((eq ue 'jump))
       (t (baboon))
       ;;; Never reached
       ))
  )

;;; Tail-recursion optimization for a function F is possible only if
;;;	1. the value of *DO-TAIL-RECURSION* is non-nil (this is default),
;;;	2. F receives only required parameters, and
;;;	3. no required parameter of F is enclosed in a closure.
;;;
;;; A recursive call (F e1 ... en) may be replaced by a loop only if
;;;	1. F is not declared as NOTINLINE,
;;;	2. n is equal to the number of required parameters of F,
;;;	3. the form is a normal function call (i.e. the arguments are
;;;	   pushed on the stack,
;;;	4. (F e1 ... en) is not surrounded by a form that causes dynamic
;;;	   binding (such as LET, LET*, PROGV),
;;;	5. (F e1 ... en) is not surrounded by a form that that pushes a frame
;;;	   onto the frame-stack (such as BLOCK and TAGBODY whose tags are
;;;	   enclosed in a closure, and CATCH),

;; (defun tail-recursion-possible ()
;;   (dolist (ue *unwind-exit* (baboon))
;;     (cond ((eq ue 'tail-recursion-mark) (return t))
;;           ((or (numberp ue) (eq ue 'bds-bind) (eq ue 'frame))
;;            (return nil))
;;           ((or (consp ue) (eq ue 'jump)))
;;           (t (baboon)))))
