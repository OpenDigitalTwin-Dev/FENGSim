;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defmacro typecase (keyform &rest clauses
		    &aux (sym (sgen "TYPECASE"))(key (if (symbolp keyform) keyform sym)))
  (declare (optimize (safety 2)))
  (labels ((l (x &aux (c (pop x))(tp (pop c))(fm (if (cdr c) (cons 'progn c) (car c)))(y (when x (l x))))
	     (if (or (eq tp t) (eq tp 'otherwise)) fm `(if (typep ,key ',tp) ,fm ,y))))
    (let ((x (l clauses)))
      (if (eq key keyform) x `(let ((,key ,keyform)) ,x)))))

(defmacro etypecase (keyform &rest clauses
		     &aux (sym (sgen "ETYPECASE"))(key (if (symbolp keyform) keyform sym)))
  (declare (optimize (safety 2)))
  (let* ((x `((t (error 'type-error :datum ,key :expected-type '(or ,@(mapcar 'car clauses))))))
	 (x `(typecase ,key ,@(append clauses x))))
    (if (eq key keyform) x `(let ((,key ,keyform)) ,x))))

(defmacro infer-tp (x y z) (declare (ignore x y)) z)

(defun mib (o l &optional f)
  (let* ((a (atom l))
	 (l (if a l (car l)))
	 (l (unless (eq '* l) l)))
    (when l
      (if (eq l 'unordered) `((isnan ,o))
	  (if f (if a `((<= ,l ,o)) `((< ,l ,o))) (if a `((<= ,o ,l)) `((< ,o ,l))))))))


(defun ?and-or (op x)
  (cond ((cdr x) (cons op x))
	((car x))
	((eq op 'and))))

(defun mibb (o tp)
  (?and-or 'and (nconc (mib o (car tp) t) (mib o (cadr tp)))))

(defun mdb (o tp)
  (let* ((b (car tp)))
    (cond ((not tp))
	  ((eq b '*))
	  ((not (listp b)) (or (eql b 1) `(eql (array-rank ,o) ,b)))
	  ((let ((l (length b))
		 (x (?and-or
		     'and
		     (let ((i -1))
		       (mapcan (lambda (x)
				 (incf i)
				 (unless (eq x '*) `((eql ,x (array-dimension ,o ,i))))) b)))))
	     (cond ((eql l 1) x)
		   ((eq x t) `(eql ,l (array-rank ,o)))
		   (`(when (eql ,l (array-rank ,o)) ,x))))))))


(defun msubt-and-or (and-or o tp y &optional res)
  (if tp
      (let ((x (msubt o (pop tp) y)))
	(if (eq x (eq and-or 'or)) x
	  (msubt-and-or and-or o tp y (if (eq x (eq and-or 'and)) res (cons x res)))))
    (?and-or and-or (nreverse res))))


(defvar *complex-part-types*
  (mapcar (lambda (x &aux (x (if (listp x) x (list x x))))
	    (list (cmp-norm-tp (cons 'complex* x)) (cmp-norm-tp (car x)) (cmp-norm-tp (cadr x))))
	(list* '(integer ratio) '(ratio integer) +range-types+)))

(defun complex-part-types (z)
  (lreduce (lambda (y x)
	     (if (tp-and z (pop x))
		 (mapcar 'tp-or x y)
	       y))
	   *complex-part-types* :initial-value (list nil nil)))

(defun and-form (x y)
  (when (and x y)
    (cond ((eq x t) y)
	  ((eq y t) x)
	  (`(when ,x ,y)))))

(defun msubt (o tp y &aux
		(tp (let ((x (cmp-norm-tp tp))) (or (tp>= x y) (when (tp-and x y) tp))))
		(otp (normalize-type tp));FIXME normalize, eg structure
                (lp (listp otp))(ctp (if lp (car otp) otp))(tp (when lp (cdr otp))))
  (case ctp
	((or and) (msubt-and-or ctp o tp y))
	(not (let ((x (msubt o (car tp) y))) (cond ((not x))((eq x t) nil)(`(not ,x)))))
	(satisfies `(,(car tp) ,o))
	(member (if (cdr tp) `(member ,o ',tp) `(eql ,o ',(car tp))))
	((t nil) ctp)
	(otherwise
	 (if (tp>= (case ctp ((proper-cons improper-cons) #tcons) (otherwise (cmp-norm-tp ctp))) y) ;FIXME
             (ecase ctp
		    (#.+range-types+ (mibb o tp))
		    (complex* (let* ((x (complex-part-types y))
				     (f (and-form (msubt 'r (car tp) (car x)) (msubt 'i (cadr tp) (cadr x)))))
				(if (consp f) `(let ((r (realpart ,o))(i (imagpart ,o))) ,f) f)))
		    ((simple-array non-simple-array) (mdb o (cdr tp)))
		    ((structure structure-object) (if tp `(mss (c-structure-def ,o) ',(car tp)) t))
		    ((std-instance funcallable-std-instance)
		     (if tp `(when (member (load-time-value (si-find-class ',(si-class-name (car tp)) nil))
					   (si-cpl-or-nil (si-class-of ,o)))
			       t)
			 t))
		    ((proper-cons improper-cons)
		     (and-form
		      (and-form (simple-type-case `(car ,o) (car tp)) (simple-type-case `(cdr ,o) (cadr tp)))
		      (if (eq ctp 'proper-cons)
			  (or (tp>= #tproper-list (cmp-norm-tp (cadr tp))) `(not (improper-consp ,o)))
			(or (tp>= #t(not proper-list) (cmp-norm-tp (cadr tp))) `(improper-consp ,o))))))
	   (progn (break) (simple-type-case o otp))))));;undecidable aggregation support


(defun branch (tpsff x f y &aux (q (cdr x))(x (car x))(z (cddr (assoc x tpsff))))
  (if q
      `((,(msubt f (tp-type q) y) ,(mkinfm f q z)))
    `((t ,(?-add 'progn z)))))


(defun branch1 (x tpsff f o &aux (y (lreduce 'tp-or (car x) :initial-value nil)))
  (let* ((z (mapcan (lambda (x) (branch tpsff x f y)) (cdr x)))
	 (s (lremove nil (mapcar 'cdr (cdr x))))
	 (z (if s (nconc z `((t ,(mkinfm f (tp-not (lreduce 'tp-or s :initial-value nil)) (cdar o))))) z)))
    (cons 'cond z)))

(defun mkinfm (f tp z &aux (z (?-add 'progn z)))
  (if (tp>= tp #tt) z `(infer-tp ,f ,tp ,z)))

(define-compiler-macro typecase (x &rest ff)
  (let* ((bind (unless (symbolp x) (list (list (gensym) x))));FIXME sgen?
	 (f (or (caar bind) x))
	 (o (member-if (lambda (x) (or (eq (car x) t) (eq (car x) 'otherwise))) ff));FIXME
	 (ff (if o (ldiff-nf ff o) ff))
	 (o (list (cons t (cdar o))))
	 (tps (mapcar 'cmp-norm-tp (mapcar 'car ff)))
	 (z nil) (tps (mapcar (lambda (x) (prog1 (tp-and x (tp-not z)) (setq z (tp-or x z)))) tps))
	 (tpsff (mapcan (lambda (x y) (when x (list (cons x y)))) tps ff))
	 (oth (unless (eq z t) (mkinfm f (tp-not z) (cdar o))))
	 (nb (>= (+ (length tpsff) (if oth 1 0)) 2))
	 (fm (if nb (let* ((c (calist2 (type-and-list (mapcar 'car tpsff))))
			   (fn (best-type-of c)))
		      `(case (,fn ,f)
			     ,@(branches f tpsff (cdr (assoc fn +rs+)) o c)
			     ,@(when oth `((otherwise ,oth)))))
	       (if z (mkinfm f (caar tpsff) (cddar tpsff)) oth))))
    (if (when nb bind) `(let ,bind ,fm) fm)))

(defun simple-type-case (x type)
  (funcall (get 'typecase 'compiler-macro-prop) `(typecase ,x (,type t)) nil))

(defun ?-add (x tp) (if (atom tp) tp (if (cdr tp) (cons x tp) (car tp))))

(defun branches (f tpsff fnl o c)
  (mapcar (lambda (x)
	    `(,(lremove-duplicates (mapcar (lambda (x) (cdr (assoc x fnl))) (car x)))
	      ,(mkinfm f (lreduce 'tp-or (car x) :initial-value nil) (list (branch1 x tpsff f o)))))
	  c))


(defun funcallable-symbol-function (x) (c-symbol-gfdef x))


(defconstant +xi+ (let* ((a (type-and-list (list (cmp-norm-tp `(and number (not immfix))))))
			 (rl (cdr (assoc 'tp8 +rs+)))
			 (i (lremove-duplicates (mapcar (lambda (x) (cdr (assoc (cadr x) rl))) a)))
;			 (mi (apply 'min i))
			 (xi (apply 'max i))
;			 (m (apply '+ i))
			 )
;		    (assert (= mi 1))
;		    (assert (= m (/ (* xi (1+ xi)) 2)))
		    xi))


(eval-when
 (compile eval)
 (defun mtp8b (tpi &aux (rl (cdr (assoc 'tp8 +rs+)))
		   (tp (lreduce 'tp-or
				(mapcar 'car
					(lremove-if-not
					 (lambda (x) (eql tpi (cdr x)))
					 rl))
				:initial-value nil)))
   `(infer-tp
     x ,tp
     (infer-tp
      y ,tp
      ,(let ((x (caar (member-if
		       (lambda (x &aux (z (assoc (cmp-norm-tp (cdr x)) rl :test 'tp<=)))
			 (eql tpi (cdr z)))
		       '((:fixnum . (and fixnum (not immfix)))
			 (:float . short-float)
			 (:double . long-float)
			 (:fcomplex . fcomplex)
			 (:dcomplex . dcomplex))))))
	 (if x `(,(intern (string-upcase (strcat "C-" x "-=="))) x y)
	   (cond ((tp<= tp (cmp-norm-tp 'bignum)) `(eql 0 (mpz_cmp x y)))
		 ((tp<= tp (cmp-norm-tp 'ratio))
		  `(and (eql (numerator x) (numerator y))
			(eql (denominator x) (denominator y))))
		 ((tp<= tp (cmp-norm-tp '(complex rational)))
		  `(and (eql (realpart x) (realpart y))
			(eql (imagpart x) (imagpart y))))
		 ((error "Unknown tp")))))))))
			   
#.`(defun num-comp (x y tp)
     (declare (fixnum tp))
     (case tp
	    ,@(let (r) (dotimes (i +xi+) (push `(,(1+ i) ,(mtp8b (1+ i))) r)) (nreverse r))))
(setf (get 'num-comp 'cmp-inline) t)

(defun eql (x y)
  (or (eq x y)
      (let ((tx (tp8 x)))
	(unless (zerop tx)
	  (let ((ty (tp8 y)))
	    (when (= tx ty)
	      (num-comp x y tx)))))))

(defun eql-with-tx (x y tx)
  (declare (fixnum tx))
  (or (eq x y)
      (unless (zerop tx)
	(let ((ty (tp8 y)))
	  (when (= tx ty)
	    (num-comp x y tx))))))
(setf (get 'eql-with-tx 'cmp-inline) t)
