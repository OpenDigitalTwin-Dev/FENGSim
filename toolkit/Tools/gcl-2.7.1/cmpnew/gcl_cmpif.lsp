;;; CMPIF  Conditionals.
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

(si:putprop 'if 'c1if 'c1special)
(si:putprop 'if 'c2if 'c2)

(si:putprop 'jump-true 'set-jump-true 'set-loc)
(si:putprop 'jump-false 'set-jump-false 'set-loc)

;; (si:putprop 'case 'c1case 'c1)
;; (si:putprop 'ecase 'c1ecase 'c1)
;; (si:putprop 'case 'c2case 'c2)

(defun note-branch-elimination (test-form val elim-form)
  (eliminate-src elim-form)
  (keyed-cmpnote (list 'branch-elimination test-form)
		 "Test form ~S is ~S,~%;; eliminating branch ~S~%" test-form val elim-form))

(defconstant +gen+ (make-var :name (gensym)))
;(defconstant +gen+ (gensym))

(defun tp-reduce (f1 f2 l1 l2)
  (labels ((c1 (c l2 &aux (d (cdr c))(m (cdr (or (assoc (car c) l2) (assoc +gen+ l2) '(nil t . t)))))
	       (cons (car c) (cons (funcall f1 (car m) (car d)) (funcall f2 (cdr m) (cdr d))))))
	  (remove-duplicates
	   (append
	    (mapcar (lambda (x) (c1 x l2)) l1)
	    (mapcar (lambda (x) (c1 x l1)) l2)) :key 'car)))

;; (defun tp-reduce (f1 f2 l1 l2 &optional r)
;;   (labels ((m (l1 l2) (cdr (or (assoc (caar l1) l2) (assoc +gen+ l2) 
;; 			       (when (eq (caar l1) +gen+) (car l1)) '(nil t . t))))
;; 	   (c (l1 l2 &aux (c (car l1))(d (cdr c))(m (m l1 l2)))
;; 	      (cons (car c) 
;; 		    (cons 
;; 		     (funcall f1 (car m) (car d))
;; 		     (funcall f2 (cdr m) (cdr d))))))
;; 	(cond (l1 (tp-reduce f1 f2 (cdr l1) l2 (cons (c l1 l2) r)))
;; 	      ((assoc (caar l2) r) (tp-reduce f1 f2 l1 (cdr l2) r))
;; 	      (l2 (tp-reduce f1 f2 l1 (cdr l2) (cons (c l2 r) r)))
;; 	      (r))))

(defconstant +bool-inf-op-list+ '((> . <=) (>= . <) (< . >=) (<= . >) (= . /=) (/= . =)))
(defconstant +bool-inf-sop-list+ '((> . <) (< . >) (<= . >=) (>= . <=) (= . =) (/= . /=)))

(defun comp-type-propagator (f t1 t2 &rest r)
  (let ((z (let ((r (num-type-rel f t1 t2)))
	     (cond ((car r) #t(member t))
		   ((cadr r) #t(member nil))
		   (#tboolean)))))
    (if r (type-or1 z (apply 'comp-type-propagator f t2 (car r) (cdr r))) z)))

(defun max-bnd (x y op &aux (nx (if (atom x) x (car x))) (ny (if (atom y) y (car y))))
  (cond ((or (eq x '*) (eq y '*)) '*)
	((= nx ny) (if (atom x) x y))
	((funcall op nx ny) x)
	(y)))

(defun real-bnds (t1) (num-type-bounds t1))

(defun two-tp-inf (fn t2 &aux (t2 (real-bnds (type-and #treal t2))))
  (case fn
	(= (cmp-norm-tp `(real ,(or (car t2) '*) ,(or (cadr t2) '*))))
	(/= (if (when (numberp (car t2)) (eql (car t2) (cadr t2)))
		(cmp-norm-tp `(and number (not (real ,@t2)))) #treal))
	(>  (cmp-norm-tp `(real ,(cond ((numberp (car t2)) (list (car t2))) ((car t2)) ('*)))))
	(>= (cmp-norm-tp `(real ,(or (car t2) '*))))
	(<  (cmp-norm-tp `(real * ,(cond ((numberp (cadr t2)) (list (cadr t2))) ((cadr t2)) ('*)))))
	(<= (cmp-norm-tp `(real * ,(or (cadr t2) '*))))))

(defmacro vl-name (x) `(var-name (car (third ,x))))
;(defmacro vl-type (x) `(var-type (car (third ,x))))  ; Won't work, ref might be across a function boundary
(defmacro vl-type (x) `(itp ,x))
(defmacro itp (x) `(info-type (second ,x)))
(defmacro vlp (x) `(and (eq 'var (car ,x)) (llvar-p (car (third ,x)))))
;(defmacro vlp (x) `(and (eq 'var (car ,x)) (eq (var-kind (car (third ,x))) 'lexical)))

;; (defun get-object-value (c1x)
;;   (when (and (eq 'location (car c1x)) (eq 'vv (caaddr c1x)))
;;     (values (gethash (cadr (caddr c1x)) *objects-rev*))))

;; (defvar *gen-nil* (list (cons +gen+ (cons nil t))))
;; (defvar *gen-t*   (list (cons +gen+ (cons t nil))))
;; (defvar *inferred-tps* nil)
;; (defvar *inferred-op* nil)
;; (defvar *inferred-iop* nil)

;; (defun fmla-chain (op iop fx fy &optional res)
;;   (let* ((*inferred-tps* res)
;; 	 (*inferred-op* op)
;; 	 (*inferred-iop* iop)
;; 	 (r (tp-reduce op iop fx fy))
;; 	 (r (if *inferred-tps* (tp-reduce op iop r *inferred-tps*) r)))
;;     (cond ((and (not (cdr fx)) (not (cdr fy))) r)
;; 	  ((equal r res) r)
;; 	  ((fmla-chain op iop fx fy r)))))

;; (defun intp (sym tp tf)
;;   (let* ((a (if tf 'cadr 'cddr))
;; 	 (itp (funcall a (assoc sym *inferred-tps*))))
;;     (if itp (funcall (if tf *inferred-op* *inferred-iop*) tp itp)
;;       tp)))

(defun tppra (tp arg f r)
  (let ((s (info-type (cadr arg))))
    (cons (type-and tp (two-tp-inf f s)) (type-and tp (two-tp-inf r s)))))

;; (defun tppra (tp arg f r)
;;   (let* ((x (info-type (cadr arg)))
;; 	 (s (cmp-norm-tp x))
;; 	 (sym (when (vlp arg) (vl-name arg))))
;;     (cons (type-and tp (two-tp-inf f (intp sym s t)))
;; 	  (type-and tp (two-tp-inf r (intp sym s nil))))))

(defun fmla-if1 (f tf ff)
  (let* ((nf (mapcar (lambda (x) (cons (car x) (cons (cddr x) (cadr x)))) f))    
	 (r1 (tp-reduce 'type-and 'type-or1 f  tf));FIXME rewrite to carry only desired branch
	 (r2 (tp-reduce 'type-and 'type-or1 nf ff))
	 (tr (tp-reduce 'type-or1 'type-and r1 r2))
	 (r1 (tp-reduce 'type-or1 'type-and nf tf))
	 (r2 (tp-reduce 'type-or1 'type-and f  ff))
	 (fr (tp-reduce 'type-and 'type-or1 r1 r2)))
    (mapc (lambda (x) (setf (cddr x) (cddr (assoc (car x) fr)))) tr)))

;; (defun fmla-if1 (f tf ff)
;;   (let* ((nf (mapcar (lambda (x) (cons (car x) (cons (cddr x) (cadr x)))) f))    
;; 	 (r1 (fmla-chain 'type-and 'type-or1 f  tf));FIXME rewrite to carry only desired branch
;; 	 (r2 (fmla-chain 'type-and 'type-or1 nf ff))
;; 	 (tr (fmla-chain 'type-or1 'type-and r1 r2))
;; 	 (r1 (fmla-chain 'type-or1 'type-and nf tf))
;; 	 (r2 (fmla-chain 'type-or1 'type-and f  ff))
;; 	 (fr (fmla-chain 'type-and 'type-or1 r1 r2))
;; 	 (tr (mapc (lambda (x) (setf (cddr x) (cddr (assoc (car x) fr)))) tr)));FIXME? check here?
;;     (delete +gen+ tr :key 'car)))

(defun fmla-if (f tf ff)
  (fmla-clean (fmla-if1 (fmla-infer-tp f) (fmla-infer-tp tf) (fmla-infer-tp ff))))

;; (defun fmla-if (f tf ff)
;;   (fmla-if1 (fmla-infer-tp f) (fmla-infer-tp tf) (fmla-infer-tp ff)))

;; (defun fmla-if (f tf ff)
;;   (let* ((f (fmla-infer-tp f))
;; 	 (r1 (fmla-chain 'type-and 'type-or1 f (fmla-infer-tp tf)))
;; 	 (f (mapcar (lambda (x) (cons (car x) (cons (cddr x) (cadr x)))) f))
;; 	 (r2 (fmla-chain 'type-and 'type-or1 f (fmla-infer-tp ff))))
;;     (delete +gen+ (fmla-chain 'type-or1 'type-and r1 r2) :key 'car)))

;; (defun fmla-switch (form &aux fm ntp ttp)
;;   (let ((c (caddr form)))
;;     (when (and (consp c) (eq (car c) 'inline))
;;       (let ((ca (caddr c)))
;; 	(when (eq ca 'tt3)
;; 	  (let* ((f (fifth c))
;; 		 (v (when (and (consp f) (eq (car f) 'let*)) (cadddr f)))
;; 		 (v (unless (cdr v) (when (and (consp (car v)) (eq (caar v) 'var)) (caaddr (car v)))))
;; 		 (tt (sixth form)))
;; 	    (do ((ints nil ints)) ((not (setq fm (pop tt))) (list* (var-name v) ttp ntp))
;; 		(cond ((tag-p fm) (push (tag-name fm) ints))
;; 		      ((and (consp fm) (eq (car fm) 'return-from))
;; 		       (let ((tp (info-type (cadr (sixth fm)))))
;; 			 (cond ((type>= #tnull tp) (setq ntp (type-or1 (ints-tt3 ints) ntp)
;; 							 ints nil))
;; 			       ((type>= #t(not null) tp)
;; 				(setq ttp (type-or1 (ints-tt3 ints) ttp) ints nil)))))))))))))

;(defun merge-fmla (x) x)

;; (defun fmla-infer-inline (f)
;;   (when (consp f)
;;   (case (car f)
;; 	((let let*) (sublis (mapcar 'cons 
;; 				    (mapcar 'var-name (third f)) 
;; 				    (mapcar (lambda (x) (when (and (consp x) (eq (car x) 'var))
;; 							  (var-name (car (third x))))) (fourth f)))
;; 			    (fmla-infer-inline (fifth f))))
;; 	(if (fmla-infer-inline (fourth f)));FIXME
;; 	(block 
;; 	 (merge-fmla (catch (third f) (fmla-infer-inline (fourth f)))))
;; 	(progn (fmla-infer-inline (car (last (third f)))))
;; 	(return-from 
;; 	 (throw (third f) (fmla-infer-inline (sixth f))))
;; 	(switch
;; 	 (mapc 'fmla-infer-inline (sixth f)))
;; 	(infer-tp (let ((tp (info-type (cadr (fifth f)))))
;; 		    (cond ((type>= #tnull tp) (list* (var-name (third f)) #tt (fourth f)))
;; 			  ((type>= #t(not null) tp) (list* (var-name (third f)) (fourth f) #tt))))))))

	
	
(defvar *infer-tags* nil)

(defun fmla-default (fmla &aux (tp (info-type (cadr fmla)))(nn (type-and tp #t(not null)))(n (type-and tp #tnull)))
  (unless (and nn n)
    (list (cons +gen+ (cons (when nn t) (when n t))))))

(defun fmla-clean (fmla)
  (delete +gen+ fmla :key 'car))

(defun fmla-infer-tp (fmla)
  (when (unless *compiler-new-safety* (listp fmla))
    (case (car fmla)
	  ((inline decl-body let let*) (fmla-infer-tp (car (last fmla))))
	  (block 
	   (let ((*infer-tags* (cons (cons (third fmla) (fmla-infer-tp (fourth fmla))) *infer-tags*)))
	     (labels ((fmla-walk (f)
				 (cond ((atom f))
				       ((when (eq (car f) 'return-from)
					  (eq (caddr f) (third fmla)))
					(fmla-infer-tp f))
				       (t (fmla-walk (car f)) (fmla-walk (cdr f))))))
		     (fmla-walk (fourth fmla)))
	     (fmla-clean (cdar *infer-tags*))))
	  (progn (fmla-infer-tp (car (last (third fmla)))))
	  (return-from 
	   (let ((x (assoc (third fmla) *infer-tags*)))
	     (when x
	       (let ((y (fmla-infer-tp (seventh fmla))))
		 (setf (cdr x) (fmla-if1 nil (cdr x) y))))))
	  (infer-tp (let* ((tp (info-type (cadr (fifth fmla))))
			   (vl (remove-if-not 'llvar-p (third fmla)))
			   (i (cond ((type>= #tnull tp) (cons nil (fourth fmla)));FIXME nil tp
				    ((type>= #t(not null) tp) (cons (fourth fmla) nil)))))
		      (nconc (when i (mapcar (lambda (x) (cons x i)) vl)) (fmla-infer-tp (fifth fmla)))));FIXME
	  (lit (mapcar (lambda (x) (list* x #t(not null) #tnull))
		       (local-aliases (get-top-var-binding (lit-bind fmla)) nil)))
	  (if (apply 'fmla-if (cddr fmla)))
	  (var (when (llvar-p (car (third fmla)))
		 (list (cons (car (third fmla)) (cons #t(not null) #tnull)))))
	  (setq (fmla-infer-tp (fourth fmla)));FIXME set var too, and in call global
	  (call-global
	   (let* ((fn (third fmla)) (rfn (cdr (assoc fn +bool-inf-op-list+)))
		  (sfn (cdr (assoc fn +bool-inf-sop-list+)))
		  (srfn (cdr (assoc sfn +bool-inf-op-list+)))
		  (args (if (eq (car fmla) 'inline) (fourth (fifth fmla)) (fourth fmla)))
		  (l (length args))
		  (pt (rassoc fn +cmp-type-alist+)));FIXME +cmp-type-alist+ (get fn 'si::predicate-type)
	     (cond ((and (= l 1) (vlp (first args)) pt) 
		    (list (cons (car (third (first args))) (cons (car pt) (tp-not (car pt))))))
		   ((and (= l 2) (eq fn 'typep) (vlp (first args))
			 (let ((tp (cmp-norm-tp (car (atomic-tp (info-type (cadr (second args))))))))
			   (when tp (list (cons (car (third (first args))) (cons tp (tp-not tp))))))))
		   ((and (= l 2) rfn)
		    (nconc
		     (when (vlp (first args))
		       (list (cons (car (third (first args)))
				   (tppra (vl-type (first args)) (second args) fn rfn))))
		     (when (eq 'lit (car (first args)))
		       (mapcar (lambda (x)
				 (cons x (tppra (vl-type (first args)) (second args) fn rfn)))
			       (local-aliases (get-top-var-binding (lit-bind (first args))) nil)))
		     (when (vlp (second args))
		       (list (cons (car (third (second args)))
				   (tppra (vl-type (second args)) (first args) sfn srfn))))
		     (when (eq 'lit (car (second args)))
		       (mapcar (lambda (x)
				 (cons x (tppra (vl-type (second args)) (first args) sfn srfn)))
			       (local-aliases (get-top-var-binding (lit-bind (second args))) nil)))))
		   ((fmla-default fmla)))))
	  (otherwise (fmla-default fmla)))))


(defvar *restore-vars* nil)

(defun restrict-type (v ot lt)
  (setf (var-type v) ot)
  (unless (type>= lt ot)
    (let ((nt (type-and ot lt)))
      (keyed-cmpnote (list 'type 'type-restriction (var-name v))
		     "restricting type of ~s to ~s~%" (var-name v) (cmp-unnorm-tp nt))
      (setf (var-type v) nt))))

(defun ignorable-pivot (pivot value)
  (let ((s (sgen "IGNORABLE-PIVOT")))
    `(let ((,s ,pivot))
       (declare (ignorable ,s))
       ,value)))

(defun fmla-is-changed (var fmla)
  (cond ((info-p fmla) (is-changed var fmla))
	((atom fmla) nil)
	((or (fmla-is-changed var (car fmla)) (fmla-is-changed var (cdr fmla))))))

;; (defun fmla-is-changed (name fmla)
;;   (cond ((info-p fmla) (let ((v (car (member name *vars* :key (lambda (x) (when (var-p x) (var-name x)))))))
;; 			 (is-changed v fmla)))
;; 	((atom fmla) nil)
;; 	((or (fmla-is-changed name (car fmla)) (fmla-is-changed name (cdr fmla))))))

(defun c1branch (tf r args info)
  (if (and (not tf) (endp (cddr args)))
      (list (c1nil) nil)
      (with-restore-vars ;FIXME eliminate if any variable restricts to nil
	(dolist (l r) (restrict-type (car l) (cadr l) (let ((l (caddr l))) (if tf (car l) (cdr l)))))
	(let (trv (b (c1expr* (if tf (cadr args) (caddr args)) info)))
	  (dolist (l *restore-vars*)
	    (push (if (var-p (car l))
		      (list (car l) (var-type (car l)) (var-store (car l)))
		      (progn (keyed-cmpnote (list 'type-mod-unwind) "Winding type ~s at end of branch" (car l))
			     (list (car l) (mcpt (car l)))))
		  trv))
	  (keep-warnings)
	  (list b trv)))))

(defun c-and (y x)
  (if (type>= #tnull (info-type (cadr y))) y
    (let ((x (fmla-c1expr x)))
      (list 'if (make-info :type (type-or1 (info-type (cadr x)) #tnull)) y x (c1nil)))))

(defun c-or (y x)
  (if (type>= #t(not null) (info-type (cadr y))) y
    (let ((x (fmla-c1expr x)))
      (list 'if (make-info :type (type-or1 (info-type (cadr x)) #t(member t))) y (c1t) x))))

(defun c-not (x)
  (let ((x (fmla-c1expr x)))
    (cond ((type>= #tnull (info-type (cadr x))) 
	   (list 'progn (make-info :type #t(member t)) (list x (c1t))))
	  ((type>= #t(not null) (info-type (cadr x))) 
	   (list 'progn (make-info :type #tnull) (list x (c1nil))))
	  ((list 'if (make-info :type #tboolean) x (c1nil) (c1t))))))


(defun fmla-c1expr (fmla)
  (case (car fmla)
	(fmla-and (reduce 'c-and (cdr fmla) :initial-value (c1t)))
	(fmla-or  (reduce 'c-or  (cdr fmla) :initial-value (c1nil)))
	(fmla-not (c-not (fmla-c1expr (cadr fmla))))
	(otherwise fmla)))

(defun maybe-progn-fmla (fmla args)
  (c1progn (list fmla args) (list (fmla-c1expr fmla) (c1expr args))))


(defun c1if (args &aux info f)
  (when (or (endp args) (endp (cdr args)))
        (too-few-args 'if 2 (length args)))
  (unless (or (endp (cddr args)) (endp (cdddr args)))
          (too-many-args 'if 3 (length args)))
  (setq f (c1fmla-constant (car args)))

  (case f
        ((t) 
	 (when (caddr args) (note-branch-elimination (car args) t (caddr args)))
	 (c1expr (cadr args)))
        ((nil) 
	 (note-branch-elimination (car args) nil (cadr args))
	 (if (endp (cddr args)) (c1nil) (c1expr (caddr args))))
        (otherwise
         (setq info (make-info))
	 (let* ((fmla (c1fmla f info))
		(inf (fmla-clean (fmla-infer-tp fmla)))
		(inf (remove-if (lambda (x) (fmla-is-changed (car x) fmla)) inf))
		(fmlae (fmla-eval-const fmla))
		(fmlae (if (notevery 'cadr inf) nil fmlae))
		(fmlae (if (notevery 'cddr inf) t   fmlae)))
	   (when inf 
	     (keyed-cmpnote (list* 'type-inference (mapcar (lambda (x) (var-name (car x))) inf))
			    "inferring types on form ~s, ~s"
			    f (mapcar (lambda (x)
					(list (pop x) (cmp-unnorm-tp (pop x)) (cmp-unnorm-tp x)))
				      inf)))
	   (if (not (eq fmlae 'boolean))

 	       (cond (fmlae 
  		      (when (caddr args) (note-branch-elimination (car args) t (caddr args)))
		      (maybe-progn-fmla fmla (cadr args)))
  		     (t (note-branch-elimination (car args) nil (cadr args)) 
			(maybe-progn-fmla fmla (caddr args))))
	     
	     (let (r)
	       (dolist (l inf)
		 (let ((v (car l)))
		   (when v
		     (push (list v (var-type v) (cdr l)) r))))
	       (unwind-protect

		   (let* ((tbl (c1branch t   r args info))
			  (fbl (c1branch nil r args info))
			  (tb (car tbl))
			  (fb (car fbl))
			  (tret (info-type (cadr tb)))
			  (fret (info-type (cadr fb)))
			  (trv (append (when tret (cadr tbl)) (when fret (cadr fbl)))))

		     (setf (info-type info) (type-or1 (info-type (cadr tb)) (info-type (cadr fb))))

		     (do (rv) ((not (setq rv (pop r))))
			 (setf (var-type (car rv)) (cadr rv))
			 (if fret
			     (unless tret
			       (do-setq-tp (car rv) nil (type-and (cdr (caddr rv)) (var-type (car rv)))))
			   (when tret
			     (do-setq-tp (car rv) nil (type-and (car (caddr rv)) (var-type (car rv)))))))

		     (or-branches trv)
		     (list 'if info fmla tb fb))

		 (dolist (l r)
		   (setf (var-type (car l)) (cadr l))))))))))


(defun t-and (x y)
  (cond ((eq x 'boolean) (when y 'boolean))
	((eq y 'boolean) (when x 'boolean))
	((and x y))))

(defun t-or (x y)
  (cond ((eq x 'boolean) (or (eq y t) 'boolean))
	((eq y 'boolean) (or (eq x t) 'boolean))
	((or x y))))

(defun t-not (x)
  (if (eq x 'boolean)
      'boolean
    (not x)))

(defun fmla-eval-const (fmla)
  (if *compiler-new-safety* 'boolean
    (case (car fmla)
	  (fmla-and (reduce (lambda (y x) (t-and (fmla-eval-const x) y)) (cdr fmla) :initial-value t))
	  (fmla-or (reduce (lambda (y x) (t-or (fmla-eval-const x) y)) (cdr fmla) :initial-value nil))
	  (fmla-not (t-not (fmla-eval-const (cdr fmla))))
	  ((t nil) (car fmla))
	  (otherwise (if (consp (car fmla)) 
			 (fmla-eval-const (car fmla)) 
		       (cond ((type>= #tnull (info-type (second fmla))) nil) ;FIXME
			     ((type>= #t(not null) (info-type (second fmla))) t)
			     ('boolean)))))))

(defun c1fmla-constant (fmla &aux f)
  (cond
   (*compiler-new-safety* fmla)
   ((consp fmla)
    (case (car fmla)
          (and (do ((fl (cdr fmla) (cdr fl)))
                   ((endp fl) t)
                   (declare (object fl))
                   (setq f (c1fmla-constant (car fl)))
                   (case f
                         ((t))
                         ((nil) (return nil))
                         (t (if (endp (cdr fl))
                                (return f)
                                  (return (list* 'and f (cdr fl))))))))
          (or (do ((fl (cdr fmla) (cdr fl)))
                  ((endp fl) nil)
                  (declare (object fl))
                  (setq f (c1fmla-constant (car fl)))
                  (case f
                        ((t) (return t))
                        ((nil))
                        (t (if (endp (cdr fl))
                               (return f)
                               (return (list* 'or f (cdr fl))))))))
          ((not null)
           (when (endp (cdr fmla)) (too-few-args 'not 1 0))
           (unless (endp (cddr fmla))
                   (too-many-args 'not 1 (length (cdr fmla))))
           (setq f (c1fmla-constant (cadr fmla)))
           (case f
                 ((t) nil)
                 ((nil) t)
                 (t (list 'not f))))
          (t fmla)))
   ((symbolp fmla) (if (constantp fmla)
                       (if (symbol-value fmla) t nil)
                       fmla))
   (t t)))

(defun fmla-tp (fmla)
  (case (car fmla)
	((fmla-and fmla-or)
	 (let ((tp (if (eq (car fmla) 'fmla-and) #tnull #t(not null)))
	       (z (mapcar 'fmla-tp (cdr fmla))))
	   (reduce (lambda (y x) (if (type>= tp y) y 
				   (type-or1 x (type-and tp y)))) (cdr z) :initial-value (car z))))
	(fmla-not (let ((tp (fmla-tp (cadr fmla)))) 
		    (cond ((type>= #tnull tp) #t(member t))
			  ((type>= #t(not null) tp) #tnull) 
			  (#tboolean))))
	(otherwise (info-type (cadr fmla)))))

;; (defun fmla-and-or (fmlac info tp)
;;   (let (r rp z)
;;     (dolist (x fmlac r)
;;       (with-restore-vars
;;        (setq z (c1fmla x info))
;;        (do (l) ((not (setq l (pop *restore-vars*)))) 
;; 	 (setf (var-type (car l)) (type-or1 (var-type (car l)) (cadr l)))))
;;       (setq rp (let ((tmp (cons z nil))) (if rp (cdr (rplacd rp tmp)) (setq r tmp))))
;;       (when (type>= tp (fmla-tp z))
;; 	(return r)))))

;; (defun c1fmla (fmla info &aux *c1exit*)
;;   (if (atom fmla) (c1expr* fmla info)
;;     (case (car fmla)
;; 	  (and (case (length (cdr fmla))
;; 		     (0 (c1t))
;; 		     (1 (c1fmla (cadr fmla) info))
;; 		     (t (cons 'FMLA-AND (fmla-and-or (cdr fmla) info #tnull)))))
;; 	  (or (case (length (cdr fmla))
;; 		    (0 (c1nil))
;; 		    (1 (c1fmla (cadr fmla) info))
;; 		    (t (cons 'FMLA-OR (fmla-and-or (cdr fmla) info #t(not null))))))
;; 	  ((not null)
;; 	   (when (endp (cdr fmla)) (too-few-args 'not 1 0))
;; 	   (unless (endp (cddr fmla))
;; 	     (too-many-args 'not 1 (length (cdr fmla))))
;; 	   (list 'FMLA-NOT (c1fmla (cadr fmla) info)))
;; 	  (t (let* ((cm (and (symbolp (car fmla)) (get (car fmla) 'si::compiler-macro-prop)))
;; 		    (cm (and cm (funcall cm fmla nil))))
;; 	       (cond ((and cm (not (eq cm fmla)))  (c1fmla cm info))
;; 		     ((let ((r (c1expr* fmla info))) 
;; 			(if (type>= #tboolean (info-type (cadr r))) r
;; 			  (let ((info (make-info :type #tboolean)))
;; 			    (add-info info (cadr r))
;; 			    (list 'if info 
;; 				  (list 'call-global info 'eq (list r (c1nil)))
;; 				  (c1nil) (c1t))))))))))))

(defconstant +fmla+ (list (make-c1exit (gensym))))

(defun exit-to-fmla-p nil
  (eq (last *c1exit*) +fmla+))

(defun co1or-arg-tp (arg)
  (let ((x (with-restore-vars (c1expr arg))))
    (if (member-if 'is-ttl-tag (info-ref (cadr x)))
	#tt (info-type (cadr x)))))

(defun co1or (fn args)
  (declare (ignore fn))
  (let* ((tp (when (and args (exit-to-fmla-p)) #t(member t)))
	 (arg (pop args))
	 (tp (or tp (co1or-arg-tp arg)))
	 (atp (atomic-tp (type-and tp #t(not null)))))
    (when (atomic-type-constant-value atp);FIXME make sure this is never a binding, FIXME ignorable-form?
      (c1expr (if args `(if ,arg ',(car atp) (or ,@args)) arg)))))

;; (defun co1or (fn args)
;;   (declare (ignore fn))
;;   (let* ((tp (when (and args (exit-to-fmla-p)) #t(member t)))
;; 	 (arg (pop args))
;; 	 (tp (or tp (info-type (cadr (with-restore-vars (c1expr arg))))))
;; 	 (atp (atomic-tp (type-and tp #t(not null)))))
;;     (when (atomic-type-constant-value atp);FIXME make sure this is never a binding
;;       (c1expr `(if ,arg ',(car atp) ,@(when args `((or ,@args))))))))

;; (defun co1or (fn args)
;;   (declare (ignore fn))
;;   (with-restore-vars
;;    (let* ((tp (when (and args (exit-to-fmla-p)) #t(member t)))
;; 	  (arg (pop args))
;; 	  (tp (or tp (info-type (cadr (c1expr arg)))))
;; 	  (atp (atomic-tp (type-and tp #t(not null)))))
;;      (when (atomic-type-constant-value atp)
;;        (keep-vars)
;;        (c1expr `(if ,arg ',(car atp) (or ,@args)))))))
(setf (get 'or 'co1special) 'co1or)

(defun c1fmla (fmla info &aux (*c1exit* +fmla+))
  (c1expr* fmla info))

(defun not-compiler-macro (form env)
  (declare (ignore env))
  `(if ,(cadr form) nil t))
(setf (get 'not 'si::compiler-macro-prop) 'not-compiler-macro)
(setf (get 'null 'si::compiler-macro-prop) 'not-compiler-macro)


(defun c2if (fmla form1 form2)
  (let* ((v *value-to-go*)
	 (rev (and (type>= #tnull (info-type (cadr form1)))
		   (type>= #t(not null) (info-type (cadr form2)))))
	 (reg (and (type>= #tnull (info-type (cadr form2)))
		   (type>= #t(not null) (info-type (cadr form1)))))
	 (vj (when (or rev reg) (and (consp v) (car (member (car v) '(jump-true jump-false))))))
	 (fj (eq vj (if rev 'jump-true 'jump-false)))
	 (Flabel (next-label))
;	 (Flabel (if vj (if fj (cadr v) (caddr v)) (next-label))) FIXME: This needs working side-effects propagation
	 (Tlabel (if vj (if fj (caddr v) (cadr v)) (next-label))))
    (let* ((*unwind-exit* (cons Flabel (cons Tlabel *unwind-exit*)))
	   (*exit* Tlabel))
      (CJF fmla Tlabel Flabel))
    (unless vj (wt-label Tlabel))
    (let ((*unwind-exit* (cons 'JUMP *unwind-exit*))) (c2expr form1))
    (wt-label Flabel)
;    (unless vj (wt-label Flabel))
    (c2expr form2)))

;; (defun c2if (fmla form1 form2
;;                   &aux (Tlabel (next-label)) Flabel)
;;   (cond ((and (eq (car form2) 'LOCATION);FIXME axe this
;;               (null (caddr form2))
;;               (eq *value-to-go* 'TRASH)
;; 	      (not (eq *exit* 'RETURN)))
;;          (let ((exit *exit*)
;;                (*unwind-exit* (cons Tlabel *unwind-exit*))
;;                (*exit* Tlabel))
;; 	   (CJF fmla Tlabel exit))
;;          (wt-label Tlabel)
;;          (c2expr form1))
;;         (t
;;          (setq Flabel (next-label))
;;          (let ((*unwind-exit* (cons Flabel (cons Tlabel *unwind-exit*)))
;;                (*exit* Tlabel))
;; 	   (CJF fmla Tlabel Flabel))
;;          (wt-label Tlabel)
;;          (let ((*unwind-exit* (cons 'JUMP *unwind-exit*))) (c2expr form1))
;;          (wt-label Flabel)
;;          (c2expr form2))))


(defun CJF (fmla Tlabel Flabel)
  (let ((*value-to-go* (list 'jump-false Flabel Tlabel))) (c2expr* fmla)))

(defun CJT (fmla Tlabel Flabel)
  (let ((*value-to-go* (list 'jump-true Tlabel Flabel))) (c2expr* fmla)))


;; (defun CJF (fmla Tlabel Flabel)
;;   (case (car fmla)
;;     (FMLA-AND (do ((fs (cdr fmla) (cdr fs)))
;;                   ((endp (cdr fs)) (CJF (car fs) Tlabel Flabel))
;;                   (declare (object fs))
;;                   (let* ((label (next-label))
;;                          (*unwind-exit* (cons label *unwind-exit*)))
;;                         (CJF (car fs) label Flabel)
;;                         (wt-label label))))
;;     (FMLA-OR (do ((fs (cdr fmla) (cdr fs)))
;;                  ((endp (cdr fs)) (CJF (car fs) Tlabel Flabel))
;;                  (declare (object fs))
;;                  (let* ((label (next-label))
;;                         (*unwind-exit* (cons label *unwind-exit*)))
;;                        (CJT (car fs) Tlabel label)
;;                        (wt-label label))))
;;     (FMLA-NOT (CJT (cadr fmla) Flabel Tlabel))
;;     (LOCATION
;;      (case (caddr fmla)
;;            ((t))
;;            ((nil) (unwind-no-exit Flabel) (wt-nl) (wt-go Flabel))
;;            (t (let ((*value-to-go* (list 'jump-false Flabel Tlabel)))
;; 		(c2expr* fmla)))))
;;     (OTHERWISE (let ((*value-to-go* (list 'jump-false Flabel Tlabel))) (c2expr* fmla)))))

;; (defun CJT (fmla Tlabel Flabel)
;;   (case (car fmla)
;;     (fmla-and (do ((fs (cdr fmla) (cdr fs)))
;;                   ((endp (cdr fs))
;;                    (CJT (car fs) Tlabel Flabel))
;;                   (declare (object fs))
;;                   (let* ((label (next-label))
;;                          (*unwind-exit* (cons label *unwind-exit*)))
;;                         (CJF (car fs) label Flabel)
;;                         (wt-label label))))
;;     (fmla-or (do ((fs (cdr fmla) (cdr fs)))
;;                  ((endp (cdr fs))
;;                   (CJT (car fs) Tlabel Flabel))
;;                  (declare (object fs))
;;                  (let* ((label (next-label))
;;                         (*unwind-exit* (cons label *unwind-exit*)))
;;                        (CJT (car fs) Tlabel label)
;;                        (wt-label label))))
;;     (fmla-not (CJF (cadr fmla) Flabel Tlabel))
;;     (LOCATION
;;      (case (caddr fmla)
;;            ((t) (unwind-no-exit Tlabel) (wt-nl) (wt-go Tlabel))
;;            ((nil))
;;            (t (let ((*value-to-go* (list 'jump-true Tlabel Flabel)))
;;                    (c2expr* fmla)))))
;;     (OTHERWISE (let ((*value-to-go* (list 'jump-true Tlabel Flabel))) (c2expr* fmla)))))

;;; If fmla is true, jump to Tlabel.  If false, do nothing.
;; (defun CJT (fmla Tlabel Flabel)
;;   (case (car fmla)
;;     (fmla-and (do ((fs (cdr fmla) (cdr fs)))
;;                   ((endp (cdr fs))
;;                    (CJT (car fs) Tlabel Flabel))
;;                   (declare (object fs))
;;                   (let* ((label (next-label))
;;                          (*unwind-exit* (cons label *unwind-exit*)))
;;                         (CJF (car fs) label Flabel)
;;                         (wt-label label))))
;;     (fmla-or (do ((fs (cdr fmla) (cdr fs)))
;;                  ((endp (cdr fs))
;;                   (CJT (car fs) Tlabel Flabel))
;;                  (declare (object fs))
;;                  (let* ((label (next-label))
;;                         (*unwind-exit* (cons label *unwind-exit*)))
;;                        (CJT (car fs) Tlabel label)
;;                        (wt-label label))))
;;     (fmla-not (CJF (cadr fmla) Flabel Tlabel))
;;     (LOCATION
;;      (case (caddr fmla)
;;            ((t) (unwind-no-exit Tlabel) (wt-nl) (wt-go Tlabel))
;;            ((nil))
;;            (t (let ((*value-to-go* (list 'jump-true Tlabel)))
;;                    (c2expr* fmla)))))
;;     (t (let ((*value-to-go* (list 'jump-true Tlabel))) (c2expr* fmla))))
;;   )

;; ;;; If fmla is false, jump to Flabel.  If true, do nothing.
;; (defun CJF (fmla Tlabel Flabel)
;;   (case (car fmla)
;;     (FMLA-AND (do ((fs (cdr fmla) (cdr fs)))
;;                   ((endp (cdr fs)) (CJF (car fs) Tlabel Flabel))
;;                   (declare (object fs))
;;                   (let* ((label (next-label))
;;                          (*unwind-exit* (cons label *unwind-exit*)))
;;                         (CJF (car fs) label Flabel)
;;                         (wt-label label))))
;;     (FMLA-OR (do ((fs (cdr fmla) (cdr fs)))
;;                  ((endp (cdr fs)) (CJF (car fs) Tlabel Flabel))
;;                  (declare (object fs))
;;                  (let* ((label (next-label))
;;                         (*unwind-exit* (cons label *unwind-exit*)))
;;                        (CJT (car fs) Tlabel label)
;;                        (wt-label label))))
;;     (FMLA-NOT (CJT (cadr fmla) Flabel Tlabel))
;;     (LOCATION
;;      (case (caddr fmla)
;;            ((t))
;;            ((nil) (unwind-no-exit Flabel) (wt-nl) (wt-go Flabel))
;;            (t (let ((*value-to-go* (list 'jump-false Flabel)))
;; 		(c2expr* fmla)))))
;;     (t (let ((*value-to-go* (list 'jump-false Flabel))) (c2expr* fmla))))
;;   )

;; (defun c1and (args)
;;   (cond ((endp args) (c1t))
;;         ((endp (cdr args)) (c1expr (car args)))
;;         ((let ((info (make-info))
;; 	       (nargs (append (mapcar (lambda (x) `(when ,x t)) (butlast args))
;; 			      (last args))))
;; 	   (list 'AND info (c1args nargs info))))))

;; (defun c2and (forms)
;;   (do ((forms forms (cdr forms)))
;;       ((endp (cdr forms))
;;        (c2expr (car forms)))
;;       (declare (object forms))
;;       (cond ((eq (caar forms) 'LOCATION)
;;              (case (caddar forms)
;;                    ((t))
;;                    ((nil) (unwind-exit nil 'JUMP))
;;                    (t (wt-nl "if(" (caddar forms) "==Cnil){")
;;                       (unwind-exit nil 'JUMP) (wt "}")
;;                       )))
;;             ((eq (caar forms) 'VAR)
;;              (wt-nl "if(")
;;              (wt-var (car (caddar forms)) (cadr (caddar forms)))
;;              (wt "==Cnil){")
;;              (unwind-exit nil 'jump) (wt "}"))
;;             (t
;;              (let* ((label (next-label))
;;                     (*unwind-exit* (cons label *unwind-exit*)))
;;                    (let ((*value-to-go* (list 'jump-true label)))
;;                         (c2expr* (car forms)))
;;                    (unwind-exit nil 'jump)
;;                    (wt-label label))))
;;       ))

;; (defun co1or (fn args &aux (arg (pop args)))
;;   (let* ((tp (info-type (cadr (c1expr arg))))
;; 	 (atp (atomic-tp (type-and tp #t(not null)))));(print (list arg args tp atp))(break)
;;     (when (and atp (c1constant-value (setq atp (car atp)) nil))
;;       (c1expr `(if ,arg ',atp (or ,@args))))))
;; (si:putprop 'or 'co1or 'co1special)


;; (defun c1or (args)
;;   (cond ((endp args) (c1nil))
;;         ((endp (cdr args)) (c1expr (car args)))
;;         (t (let ((info (make-info)))
;;                 (list 'OR info (c1args args info))))))

;; (defun c2or (forms &aux (*vs* *vs*) temp)
;;   (do ((forms forms (cdr forms))
;;        )
;;       ((endp (cdr forms))
;;        (c2expr (car forms)))
;;       (declare (object forms))
;;       (cond ((eq (caar forms) 'LOCATION)
;;              (case (caddar forms)
;;                    ((t) (unwind-exit t 'JUMP))
;;                    ((nil))
;;                    (t (wt-nl "if(" (caddar forms) "!=Cnil){")
;;                       (unwind-exit (caddar forms) 'JUMP) (wt "}"))))
;;             ((eq (caar forms) 'VAR)
;;              (wt-nl "if(")
;;              (wt-var (car (caddar forms)) (cadr (caddar forms)))
;;              (wt "!=Cnil){")
;;              (unwind-exit (cons 'VAR (caddar forms)) 'jump) (wt "}"))
;;             ((and (eq (caar forms) 'CALL-GLOBAL)
;;                   (get (caddar forms) 'predicate))
;;              (let* ((label (next-label))
;;                     (*unwind-exit* (cons label *unwind-exit*)))
;;                    (let ((*value-to-go* (list 'jump-false label)))
;;                         (c2expr* (car forms)))
;;                    (unwind-exit t 'jump)
;;                    (wt-label label)))
;;             (t
;;              (let* ((label (next-label))
;; 		    (*inline-blocks* 0)
;;                     (*unwind-exit* (cons label *unwind-exit*)))
;; 	           (setq temp (wt-c-push))
;;                    (let ((*value-to-go* temp)) (c2expr* (car forms)))
;;                    (wt-nl "if(" temp "==Cnil)") (wt-go label)
;;                    (unwind-exit temp 'jump)
;;                    (wt-label label)
;; 		   (close-inline-blocks)
;; 		   )))
;;       )
;;   )

(defun set-jump-true (loc label)
  (unless (null loc)
    (cond ((eq loc t))
          ((and (consp loc) (eq (car loc) 'INLINE-COND))
           (wt-nl "if(")
           (wt-inline-loc (caddr loc) (cadddr loc))
           (wt ")"))
          (t (wt-nl "if((" loc ")!=Cnil)")))
    (unless (eq loc t) (wt "{"))
    (unwind-no-exit label)
    (wt-nl) (wt-go label)
    (unless (eq loc t) (wt "}")))
  )

(defun set-jump-false (loc label)
  (unless (eq loc t)
    (cond ((null loc))
          ((and (consp loc) (eq (car loc) 'INLINE-COND))
           (wt-nl "if(!(")
           (wt-inline-loc (caddr loc) (cadddr loc))
           (wt "))"))
          (t (wt-nl "if((" loc ")==Cnil)")))
    (unless (null loc) (wt "{"))
    (unwind-no-exit label)
    (wt-nl) (wt-go label)
    (unless (null loc) (wt "}")))
  )

(defun c1ecase (args) (c1case args t))  

;;If the key is declared fixnum, then we convert a case statement to a switch,
;;so that we may see the benefit of a table jump.

(defun convert-case-to-switch (args)
  (let* ((sym (sgen "SWITCH"))
	 (op (pop args))
	 (args (mapcan (lambda (x &aux (k (pop x))(k (or (eq k 'otherwise) k))) 
			 (when k `(,@(if (listp k) k (list k)) (return-from ,sym (progn ,@x))))) args)))
    `(block ,sym (switch ,op ,@(if (member t args) args (nconc args `(t (return-from ,sym nil))))))))

;; (defun convert-case-to-switch (args default)
;;   (let ((sym (tmpsym)) body keys)
;;     (dolist (v (cdr args))
;; 	    (cond ((si::fixnump (car v)) (push  (car v) body))
;; 		  ((consp (car v))(dolist (w (car v)) (push w body)))
;; 		  ((member (car v) '(t otherwise))
;; 		   (and default
;; 			(cmperror "T or otherwise found in an ecase"))
;; 		   (push t body)))
;; 	    (push `(return-from ,sym (progn ,@ (cdr v))) body))
;;     (cond (default (push t body)
;; 	    (dolist (v (cdr args))
;; 	      (cond ((atom (car v)) (push (car v) keys))
;; 		    (t (setq keys (append (car v) keys)))))
;; 	    (push `(error "The key ~a for ECASE was not found in cases ~a" ,(car args) ',keys) body)))
;;     `(block ,sym (switch ,(car args) ,@(nreverse body)))))
	    
		  

(defun conv-kl (l s &aux (l (if (listp l) (remove-duplicates l) l)))
  (cond ((not l) nil)
	((atom l) `(= ,s ,l))
	((not (cdr l)) `(= ,s ,(car l)))
	((let* ((l (sort (copy-list l) '<))
		(n (car l))
		(x (car (last l)))
		(ll (let ((i (- n 1))) (mapl (lambda (x) (setf (car x) (incf i))) (make-list (length l))))))
	   (when (equal l ll)
	     `(<= ,n ,s ,x))))))

(define-compiler-macro case (&whole form &rest args)
  (if (when *compiler-in-use* (type>= #tfixnum (nil-to-t (info-type (cadr (with-restore-vars (c1arg (car args))))))))
      (let* ((s (pop args))
	     (oth (member-if (lambda (x &aux (x (car x))) (or (eq x t) (eq x 'otherwise))) args))
	     (rem (ldiff args oth))
	     (ff (when rem (conv-kl (caar rem) s))))
	(flet ((f (x) (let ((d (cdar x))) (if (cdr d) (cons 'progn d) (car d)))))
	      (cond ((unless (cdr rem) (when ff `(if ,ff ,(f rem) ,(f oth)))))
		    ((convert-case-to-switch (cdr form))))))
    form))
