;;; CMPMULT  Multiple-value-call and Multiple-value-prog1.
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

(si:putprop 'multiple-value-call 'c1multiple-value-call 'c1special)
(si:putprop 'multiple-value-call 'c2multiple-value-call 'c2)
(si:putprop 'multiple-value-prog1 'c1multiple-value-prog1 'c1special)
(si:putprop 'multiple-value-prog1 'c2multiple-value-prog1 'c2)
(si:putprop 'values 'c1values 'c1)
(si:putprop 'values 'c2values 'c2)
(si:putprop 'multiple-value-bind 'c1multiple-value-bind 'c1)
(si:putprop 'multiple-value-bind 'c2multiple-value-bind 'c2)

(defun nval (x)
  (cond ;((type>= #t(returns-exactly) x) 0)
	((single-type-p x) 1)
	((when (consp x) (eq (car x) 'returns-exactly)) (1- (length x)))))

(defun c1multiple-value-call (args
			      &aux (tsyms (load-time-value
					   (mapl (lambda (x) (setf (car x) (gensym "MV-CALL")))
						 (make-list 50)))))
  (when (endp args) (too-few-args 'multiple-value-call 1 0))
  (let* ((info (make-info))
	 (nargs (c1args args info))
	 (tps (mapcar (lambda (x) (info-type (cadr x))) (cdr nargs)))
	 (vals (mapcar 'nval tps))
	 (n (if (member nil vals) -1 (reduce '+ vals))))
    (cond ((endp (cdr args)) (c1funcall args))
	  ((and (>= (length tsyms) n 0) (inline-possible 'multiple-value-bind))
	   (let* ((syms (mapcar (lambda (x) (declare (ignore x)) (pop tsyms)) (make-list n)))
		  (r syms))
	     (c1expr
	      (reduce (lambda (x y) 
			(cond ((= 1 (length (car x)))
			       `(let ((,(caar x) ,(cadr x))) ,y))
			      (`(multiple-value-bind ,@x ,y))))
		      (mapcar (lambda (x y) (let* ((n (nval x)) syms)
					      (dotimes (i n) (push (pop r) syms))
					      (list (nreverse syms) y))) tps (cdr args))
		      :from-end t :initial-value `(funcall ,(car args) ,@syms)))))
	  ((list 'multiple-value-call info (pop nargs) nargs)))))


(defun c2multiple-value-call (funob forms &aux (*vs* *vs*) (loc (list 'vs (vs-push))) top sup)

  (let ((*value-to-go* loc))
    (c2expr* funob))
  
  (cond ((endp (cdr forms))
         (let ((*value-to-go* 'top)) (c2expr* (car forms))))
        ((setq top (cs-push t t))
         (setq sup (cs-push t t))
         (base-used)
	 ;; Add (sup .var) handling in unwind-exit -- in
	 ;; c2multiple-value-prog1 and c2-multiple-value-call, apparently
	 ;; alone, c2expr-top is used to evaluate arguments, presumably to
	 ;; preserve certain states of the value stack for the purposes of
	 ;; retrieving the final results.  c2exprt-top rebinds sup, and
	 ;; vs_top in turn to the new sup, causing non-local exits to lose
	 ;; the true top of the stack vital for subsequent function
	 ;; evaluations.  We unwind this stack supremum variable change here
	 ;; when necessary.  CM 20040301
         (wt-nl "{object *V" top "=base+" *vs* ",*V" sup "=sup;")
	 (dolist (form forms)
	   (let ((*value-to-go* 'top)
		 (*unwind-exit* (cons (cons 'sup sup) *unwind-exit*)))
	     (c2expr-top* form top))
	   (wt-nl "while(vs_base<vs_top)")
	   (wt-nl "{V" top "[0]=vs_base[0];V" top "++;vs_base++;}"))
         (wt-nl "vs_base=base+" *vs* ";vs_top=V" top ";sup=V" sup ";")))
	
  (if *compiler-push-events*
      (wt-nl "super_funcall(" loc ");")
    (if *super-funcall*
	(funcall *super-funcall* loc)
      (wt-nl "super_funcall_no_event(" loc ");")))
  (unwind-exit 'fun-val)

  (when (cdr forms)
    (wt "}")))

;; (defun c2multiple-value-call (funob forms &aux (*vs* *vs*) loc top sup)
;;   (cond ((endp (cdr forms))
;;          (setq loc (save-funob funob))
;;          (let ((*value-to-go* 'top)) (c2expr* (car forms)))
;;          (c2funcall funob 'args-pushed loc))
;;         (t
;;          (setq top (cs-push t t))
;;          (setq sup (cs-push t t))
;;          (setq loc (save-funob funob))
;;          (base-used)
;; 	 ;; Add (sup .var) handling in unwind-exit -- in
;; 	 ;; c2multiple-value-prog1 and c2-multiple-value-call, apparently
;; 	 ;; alone, c2expr-top is used to evaluate arguments, presumably to
;; 	 ;; preserve certain states of the value stack for the purposes of
;; 	 ;; retrieving the final results.  c2exprt-top rebinds sup, and
;; 	 ;; vs_top in turn to the new sup, causing non-local exits to lose
;; 	 ;; the true top of the stack vital for subsequent function
;; 	 ;; evaluations.  We unwind this stack supremum variable change here
;; 	 ;; when necessary.  CM 20040301
;;          (wt-nl "{object *V" top "=base+" *vs* ",*V" sup "=sup;")
;; 	 (dolist (form forms)
;; 	   (let ((*value-to-go* 'top)
;; 		 (*unwind-exit* (cons (cons 'sup sup) *unwind-exit*)))
;; 	     (c2expr-top* form top))
;; 	   (wt-nl "while(vs_base<vs_top)")
;; 	   (wt-nl "{V" top "[0]=vs_base[0];V" top "++;vs_base++;}"))
;;          (wt-nl "vs_base=base+" *vs* ";vs_top=V" top ";sup=V" sup ";")
;;          (c2funcall funob 'args-pushed loc)
;;          (wt "}"))))

(defun c1multiple-value-prog1 (args
			       &aux form info tp (tsyms (load-time-value
				       (mapl (lambda (x) (setf (car x) (gensym "MV-PROG1")))
					     (make-list 50)))))
  (when (endp args) (too-few-args 'multiple-value-prog1 1 0))
  (with-restore-vars
      (setq form (c1expr (car args)) info (copy-info (cadr form)) tp (info-type info))
    (unless (or (single-type-p tp)
		(and (consp tp) (eq (car tp) 'returns-exactly) (>= (length tsyms) (length (cdr tp)))))
      (keep-vars)))
  (cond ((single-type-p tp)
	 (let ((s (pop tsyms)))
	   (c1expr `(let ((,s ,(car args))) ,@(cdr args) ,s))))
	((and (consp tp) (eq (car tp) 'returns-exactly) (>= (length tsyms) (length (cdr tp))))
	 (let ((syms (mapcar (lambda (x) (declare (ignore x)) (pop tsyms)) (cdr tp))))
	   (c1expr `(multiple-value-bind (,@syms) ,(car args) ,@(cdr args) (values ,@syms)))))
	(t
	 (setq args (c1args (cdr args) info))
					;	       (setf (info-type info) (info-type (cadr form)))
	 (list 'multiple-value-prog1 info form args))))


;; (defun c1multiple-value-prog1 (args &aux (info (make-info)) form)
;;   (when (endp args) (too-few-args 'multiple-value-prog1 1 0))
;;   (setq form (c1expr* (car args) info))
;;   (let ((tp (info-type (cadr form))))
;;     (cond ((single-type-p tp) (let ((s (tmpsym))) (c1expr `(let ((,s ,(car args))) ,@(cdr args) ,s))))
;; 	  ((and (consp tp) (eq (car tp) 'returns-exactly))
;; 	   (let ((syms (mapcar (lambda (x) (declare (ignore x)) (tmpsym)) (cdr tp))))
;; 	     (c1expr `(multiple-value-bind (,@syms) ,(car args) ,@(cdr args) (values ,@syms)))))
;; 	  (t 
;; 	   (setq args (c1args (cdr args) info))
;; 	   (setf (info-type info) (info-type (cadr form)))
;; 	   (list 'multiple-value-prog1 info form args)))))

;; We may record information here when *value-to-go* = 'top
(defvar *top-data* nil)

(defun c2multiple-value-prog1 (form forms &aux (base (cs-push t t))
				               (top (cs-push t t))
					       (sup (cs-push t t))
					       top-data)
  (let ((*value-to-go* 'top)
	*top-data*)
    (c2expr* form)
    (setq top-data *top-data*))
  ;; Add (sup .var) handling in unwind-exit -- in
  ;; c2multiple-value-prog1 and c2-multiple-value-call, apparently
  ;; alone, c2expr-top is used to evaluate arguments, presumably to
  ;; preserve certain states of the value stack for the purposes of
  ;; retrieving the final results.  c2exprt-top rebinds sup, and
  ;; vs_top in turn to the new sup, causing non-local exits to lose
  ;; the true top of the stack vital for subsequent function
  ;; evaluations.  We unwind this stack supremum variable change here
  ;; when necessary.  CM 20040301
  (wt-nl "{object *V" top "=vs_top,*V" base "=vs_base,*V" sup "=sup;")
  (setq *sup-used* t)
  (wt-nl "vs_base=V" top ";")
  (dolist (form forms)
	    (let ((*value-to-go* 'trash)
		  (*unwind-exit* (cons (cons 'sup sup) *unwind-exit*)))
	      (c2expr-top* form top)))
  (wt-nl "vs_base=V" base ";vs_top=V" top ";sup=V" sup ";}")
  (unwind-exit 'fun-val nil (if top-data (car top-data))))

(defun c1values (args &aux (info (make-info))(a (mapcar (lambda (x) (c1expr* x info)) args)))

  (when (and a (not (cdr a)) (single-type-p (info-type (cadar a))))
      (return-from c1values (car a)))
  (setf (info-type info)
	(let ((x (mapcar (lambda (x) (coerce-to-one-value (info-type (cadr x)))) a)))
	  (if (unless (cdr x) x) (car x) (cons 'returns-exactly x))));FIXME
  (list 'values info a))

;; (defun c1values (args &aux (info (make-info)))
;;       (cond ((and args (not (cdr args)))
;; 	     (let ((nargs (c1args args info)))
;; 	       (if (type>= t (info-type (cadar nargs)))
;; 		   (c1expr (car args))
;; 		 (c1expr (let ((s (tmpsym))) `(let ((,s ,(car args))) ,s))))))
;; 	    (t  
;; 	     (setq args (c1args args info))
;; 	     (setf (info-type info) 
;; 		   (cmp-norm-tp 
;; 		    (cons 'returns-exactly
;; 			  (mapcar (lambda (x) (coerce-to-one-value (info-type (cadr x)))) args))))
;; 	     (list 'values info args))))

(defun c2values (forms)
  (let* ((*inline-blocks* 0)
	 (types (mapcar (lambda (x) (let ((x (coerce-to-one-value (info-type (cadr x))))) (if (type>= #tboolean x) t x))) forms))
	 (i -1)
	 ;FIXME all of this unnecessary, just avoid valp[i]=base[0]
	 (r (mapcar (lambda (x y &aux (x (when x (write-to-string (incf i))))) (strcat (rep-type y) " _t" x "=#" x ";")) (or forms (list (c1nil))) (or types (list #tnull))))
	 (i 0)
	 (s (mapcar (lambda (x &aux (x (when x (write-to-string (incf i))))) (strcat "@" x "(_t" x ")@")) (cdr forms)))
	 (s (strcat "({" (apply 'strcat (nconc r s)) "_t0;})"));FIXME
	 (s (cons s (mapcar 'inline-type (cdr types))))
	 (in (list (inline-type (car types)) (flags) s (inline-args forms types))))
    (unwind-exit in nil (cons 'values (length forms)))
    (close-inline-blocks)))


(defun c1multiple-value-bind (args &aux (info (make-info))
                                   (vars nil) (vnames nil) init-form
                                   ss is ts body other-decls
                                   (*vars* *vars*))
  (when (or (endp args) (endp (cdr args)))
    (too-few-args 'multiple-value-bind 2 (length args)))

  (when (and (caar args) (not (cdar args)))
    (return-from c1multiple-value-bind
		 (c1expr `(let ((,(caar args) ,(cadr args))) ,@(cddr args)))))

  (multiple-value-setq (body ss ts is other-decls) (c1body (cddr args) nil))

  (dolist (s (car args))
    (let ((v (c1make-var s ss is ts)))
      (push s vnames)
      (push v vars)))

  (c1add-globals (set-difference ss vnames))

  (setq init-form (c1arg (cadr args) info))

  (unless (let ((x (info-type (cadr init-form))))
	    (if (cmpt x) (not (member nil x)) x))
    (eliminate-src body)
    (return-from c1multiple-value-bind init-form))

  (when (single-type-p (info-type (cadr init-form)))
    (return-from c1multiple-value-bind
      (c1let-* (cons (cons (list (caar args) (cadr args)) (cdar args)) (cddr args)) t
	       (cons init-form (mapcar (lambda (x) (declare (ignore x)) (c1nil)) (cdar args))))))

  (setq vars (nreverse vars))
  (let* ((tp (info-type (second init-form)))
	 (tp (if (eq tp '*) (make-list (length vars) :initial-element t) (cdr tp))))
    (do ((v vars (cdr v)) (t1 tp (cdr t1)))
	((not v))
	(set-var-init-type (car v) (if t1 (car t1) #tnull))))

  (dolist (v vars) (push-var v init-form))

  (check-vdecl vnames ts is)

  (setq body (c1decl-body other-decls body))

  (add-info info (cadr body))
  (setf (info-type info) (info-type (cadr body)))

  (ref-vars body vars)
  (dolist (var vars) (check-vref var))

  ;; (let* ((*vars* ov));FIXME
  ;;   (print (setq fff (trim-vars vars (make-list (length vars) :initial-element init-form) body nil)))
  ;;   (break))

  (list 'multiple-value-bind info vars init-form body))

(defun max-stack-space (form) (abs (vald (info-type (cadr form)))))

(defun stack-space (form)
  (let* ((tp (info-type (cadr form)))
	 (vd (vald tp)))
    (cond ((< vd 0) (- vd))
	  ((equal tp #t(returns-exactly)) 0))))

(defvar *mvb-vals* nil)

(defvar *vals-set* nil)
(defun c2multiple-value-bind (vars init-form body
				   &aux (labels nil)
				   (*unwind-exit* *unwind-exit*)
				   (*vs* *vs*) (*clink* *clink*) (*ccb-vs* *ccb-vs*)
				   top-data lbs)

  (let* ((mv (make-var :type #tfixnum :kind 'lexical :loc (cs-push #tfixnum t)))
	 (nv (1- (length vars)))
	 (ns1 (stack-space init-form))
	 (ns (max nv (or ns1 (max-stack-space init-form))))
	 (*mvb-vals* t)
	 *vals-set*)
    (setf (var-kind mv) (c2var-kind mv) (var-space mv) nv (var-known-init mv) (or ns1 -1))
    (setq lbs
	  (mapcar (lambda (x)
		    (let ((kind (c2var-kind x))(f (eq x (car vars))))
		      (if kind (setf (var-kind x) (if f kind 'object)
				     (var-loc x) (cs-push (if f (var-type x) t) t))
			(setf (var-ref x) (vs-push) x (cs-push (if f (var-type x) t) t)))))
		  vars))
;    (wt-nl "{")
;    (wt-nl "int vals_set=0;")
    (when vars
	(wt-nl "register " (rep-type (var-type (car vars))) " V" (car lbs) ";")
	(wt-nl "object V" (var-loc mv) "[" ns "];"))
    (let ((i -1)) (mapc (lambda (x) (wt-nl "#define V" x " V" (var-loc mv) "[" (incf i) "]")) (cdr lbs)))
    (wt-nl);FIXME
    (dotimes (i (1+ (length vars))) (push (next-label) labels))
    
    (wt-nl "{")
    ;; (wt-nl "int vals_set=0;")
    (let ((*mv-var* mv)
	  (*value-to-go* (or (mapcar (lambda (x) (list 'cvar x)) lbs) 'trash))
	  *top-data*)
      (c2expr* init-form)
      (setq top-data *top-data*))
    
    (and *record-call-info* (record-call-info nil (car top-data)))

    (when lbs (unless *vals-set* (baboon)))

    ;; (wt-nl "if (!vals_set) {")

    ;; (setq labels (nreverse labels))
    ;; (do ((lb lbs (cdr lb))
    ;; 	 (lab labels (cdr lab)))
    ;; 	((endp lb)(reset-top)(wt-go (car lab)))
    ;;   (wt-nl "if(vs_base>=vs_top){")
    ;;   (reset-top)
    ;;   (wt-go (car lab)) 
    ;;   (wt "}")
    ;;   (set-cvar '(vs-base 0) (car lb))
    ;;   (when (cdr lb)
    ;; 	(wt-nl "vs_base++;")))
	   
    ;; (do ((lb lbs (cdr lb))
    ;; 	 (lab labels (cdr lab)))
    ;; 	((endp lb)(wt-label (car lab)))
    ;;   (wt-label (car lab))
    ;;   (set-cvar nil (car lb)))

    ;; (wt-nl "}}")

    (do ((vs vars (cdr vs)) (lb lbs (cdr lb)))
	((endp vs))
	(when (member (var-kind (car vs)) '(lexical special down))
	  (c2bind-loc (car vs) (list 'cvar (car lb))))))
  
  (c2expr body)
  (mapc (lambda (x) (wt-nl "#undef V" x)) (cdr lbs))
  (wt-nl "")
  (wt-nl "}"))

;; (defun c2multiple-value-bind (vars init-form body
;; 				   &aux (labels nil)
;; 				   (*unwind-exit* *unwind-exit*)
;; 				   (*vs* *vs*) (*clink* *clink*) (*ccb-vs* *ccb-vs*)
;; 				   top-data lbs)

;;   (multiple-value-check vars init-form)

;;   (let* ((mv (make-var :type #tfixnum :kind 'lexical :loc (cs-push #tfixnum t)))
;; 	 (nv (1- (length vars)))
;; 	 (ns1 (stack-space init-form))
;; 	 (ns (max nv (or ns1 (max-stack-space init-form))))
;; 	 (*mvb-vals* t))
;;     (setf (var-kind mv) (c2var-kind mv) (var-space mv) nv (var-known-init mv) (or ns1 -1))
;;     (setq lbs
;; 	  (mapcar (lambda (x)
;; 		    (let ((kind (c2var-kind x))(f (eq x (car vars))))
;; 		      (if kind (setf (var-kind x) (if f kind 'object)
;; 				     (var-loc x) (cs-push (if f (var-type x) t) t))
;; 			(setf (var-ref x) (vs-push) x (cs-push (if f (var-type x) t) t)))))
;; 		  vars))
;;     (wt-nl "{")
;; ;    (wt-nl "int vals_set=0;")
;;     (when vars
;; 	(wt-nl "register " (rep-type (var-type (car vars))) " V" (car lbs) ";")
;; 	(wt-nl "object V" (var-loc mv) "[" ns "];"))
;;     (let ((i -1)) (mapc (lambda (x) (wt-nl "#define V" x " V" (var-loc mv) "[" (incf i) "]")) (cdr lbs)))
;;     (wt-nl);FIXME
;;     (dotimes (i (1+ (length vars))) (push (next-label) labels))
    
;;     (wt-nl "{")
;;     (wt-nl "int vals_set=0;")
;;     (let ((*mv-var* mv)
;; 	  (*value-to-go* (or (mapcar (lambda (x) (list 'cvar x)) lbs) 'trash))
;; 	  *top-data*)
;;       (c2expr* init-form)
;;       (setq top-data *top-data*))
    
;;     (and *record-call-info* (record-call-info nil (car top-data)))

;;     (wt-nl "if (!vals_set) {")

;;     (setq labels (nreverse labels))
;;     (do ((lb lbs (cdr lb))
;; 	 (lab labels (cdr lab)))
;; 	((endp lb)(reset-top)(wt-go (car lab)))
;;       (wt-nl "if(vs_base>=vs_top){")
;;       (reset-top)
;;       (wt-go (car lab)) 
;;       (wt "}")
;;       (set-cvar '(vs-base 0) (car lb))
;;       (when (cdr lb)
;; 	(wt-nl "vs_base++;")))
	   
;;     (do ((lb lbs (cdr lb))
;; 	 (lab labels (cdr lab)))
;; 	((endp lb)(wt-label (car lab)))
;;       (wt-label (car lab))
;;       (set-cvar nil (car lb)))

;;     (wt-nl "}}")

;;     (do ((vs vars (cdr vs)) (lb lbs (cdr lb)))
;; 	((endp vs))
;; 	(when (member (var-kind (car vs)) '(lexical special down))
;; 	  (c2bind-loc (car vs) (list 'cvar (car lb))))))
  
;;   (c2expr body)
;;   (mapc (lambda (x) (wt-nl "#undef V" x)) (cdr lbs))
;;   (wt-nl "")
;;   (wt-nl "}"))
