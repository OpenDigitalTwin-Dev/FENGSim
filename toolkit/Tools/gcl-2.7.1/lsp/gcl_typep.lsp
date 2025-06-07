;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defun ib (o l &optional f)
  (let* ((a (atom l))
	 (l (if a l (car l)))
	 (l (unless (eq '* l) l)))
    (or (not l) (if (eq l 'unordered) (isnan o) (if f (if a (<= l o) (< l o)) (if a (<= o l) (< o l)))))))
(setf (get 'ib 'cmp-inline) t)

(defun db (o tp)
  (let* ((b (car tp))(i -1))
    (cond ((not tp))
	  ((eq b '*))
	  ((not (listp b)) (eql (c-array-rank o) b))
	  ((eql (length b) (c-array-rank o))
	   (not (member-if-not
		 (lambda (x)
		   (incf i)
		   (or (eq x '*) (eql x (array-dimension o i))))
		 b))))))

(defun dbv (o tp)
  (let* ((b (car tp))(b (if (listp b) (car b) b)))
     (cond ((not tp))
	   ((eq b '*))
	   ((eql (c-array-dim o) b)))))
(setf (get 'db 'cmp-inline) t)
(setf (get 'dbv 'cmp-inline) t)


(defun ibb (o tp)
  (and (ib o (car tp) t) (ib o (cadr tp))))
(setf (get 'ibb 'cmp-inline) t)

(defun sdata-includes (x)
  (when x (the (or s-data null) (*object (c-structure-self x) 4 nil nil))));FIXME s-data-name boostrap loop
(setf (get 'sdata-includes 'cmp-inline) t)
(defun sdata-included (x)
  (when x (the proper-list (*object (c-structure-self x) 3 nil nil))));FIXME s-data-name boostrap loop
(setf (get 'sdata-included 'cmp-inline) t)
(defun sdata-name (x)
  (when x (the symbol (*object (c-structure-self x) 0 nil nil))));FIXME s-data-name boostrap loop
(defun sdata-type (x)
  (when x (the symbol (*object (c-structure-self x) 16 nil nil))));FIXME s-data-name boostrap loop
(setf (get 'sdata-name 'cmp-inline) t)

(defun mss (o sn) (when o (or (eq (sdata-name o) sn) (mss (sdata-includes o) sn))))
(setf (get 'mss 'cmp-inline) t)

#.`(defun listp (x) ,(simple-type-case 'x 'list))

(defun valid-class-name (class &aux (name (si-class-name class)))
  (when (eq class (si-find-class name nil))
    name))
(setf (get 'valid-class-name 'cmp-inline) t)

(defun lookup-simple-typep-fn (name)
  (when (symbolp name) (get name 'simple-typep-fn)))
(defun lookup-typep-fn (name)
  (when (symbolp name) (get name 'typep-fn)))

(defmacro define-typep-fn (name lambda-list &rest body &aux (q (intern (string-concatenate (string name) "-TYPEP-FN"))))
  `(progn
     (defun ,q ,lambda-list ,@body)
     (setf (get ',name 'typep-fn) ',q (get ',q 'cmp-inline) t)))


(define-typep-fn or  (o tp) (when tp (or (typep o (pop tp)) (or-typep-fn o tp))))
(define-typep-fn and (o tp) (if tp (and (typep o (pop tp)) (and-typep-fn o tp)) t))
(define-typep-fn not (o tp) (not (when tp (typep o (car tp)))))

(defmacro define-compound-typep-fn (name (o tp) &rest body &aux (q (intern (string-concatenate (string name) "-TYPEP-FN"))))
  `(progn
     (defun ,q (,o ,tp) (declare (ignorable ,o ,tp)) (when ,(simple-type-case o name) ,@body))
     (setf (get ',name 'typep-fn) ',q (get ',q 'cmp-inline) t)))


#.`(progn
     ,@(mapcar (lambda (y) `(define-compound-typep-fn ,y (o tp) (ibb o tp)))
	       (append '(real float rational) +range-types+)))


(define-typep-fn unsigned-byte (o tp &aux (s (if tp (car tp) '*)))
  (typecase
   o
   (fixnum  (unless (minusp o) (or (eq s '*) (<= (integer-length o) s))))
   (integer (unless (minusp o) (or (eq s '*) (<= (integer-length o) s))))))

(define-typep-fn signed-byte (o tp &aux (s (if tp (car tp) '*)))
  (typecase
   o
   (fixnum  (or (eq s '*) (< (integer-length o) s)))
   (integer (or (eq s '*) (< (integer-length o) s)))))


#.`(progn
     ,@(mapcar (lambda (y) `(define-simple-typep-fn ,y)) +singleton-types+))


#.`(progn
     ,@(mapcan (lambda (x)
		 (mapcar (lambda (y)
			   `(deftype ,(cadr y) (&optional dims) `(,',(car x) ,',(car y) ,dims)))
			 (cdr x)))
	       +array-typep-alist+)
     ,@(mapcan (lambda (x)
		 (mapcar (lambda (y)
			   `(define-compound-typep-fn ,(cadr y) (o tp) (,(if (eq (car x) 'vector) 'dbv 'db) o tp)))
			 (cdr x)))
	       +array-typep-alist+)
     ,@(mapcan (lambda (x)
		 `((define-typep-fn ,(car x) (o tp)
		     (when (funcall (cddr (assoc (upgraded-array-element-type (if tp (car tp) '*))
						 (cdr (assoc ',(car x) +array-typep-alist+))))
				    o)
		       (,(if (eq (car x) 'vector) 'dbv 'db) o (cdr tp))))))
	       +array-typep-alist+))

(defun cmp-real-tp (x y)
  (when (member x +range-types+)
    (when (member y +range-types+)
      (if (eq x y) x
	(ecase x
	       (integer (ecase y (ratio 'integer-ratio)))
	       (ratio (ecase y (integer 'ratio-integer))))))))


(defconstant +complex*-typep-alist+
  (mapcar (lambda (x &aux (k (cmp-real-tp (if (listp x) (car x) x) (if (listp x) (cadr x) x)))
		     (q (intern (string-concatenate "COMPLEX*-" (string k)))))
	    (list* x k q (intern (string-concatenate (string q) "-SIMPLE-TYPEP-FN"))))
	  (list* '(integer ratio) '(ratio integer) +range-types+)))

#.`(progn
     ,@(mapcan (lambda (x)
		 `((deftype ,(caddr x) nil ',`(complex* ,(if (listp (car x)) (caar x) (car x)) ,(if (listp (car x)) (cadar x) (car x))))))
	       +complex*-typep-alist+))

(define-typep-fn complex* (o tp &aux (rtp (if tp (pop tp) '*))(itp (if tp (car tp) rtp))
			    (rctp (if (listp rtp) (car rtp) rtp))(ictp (if (listp itp) (car itp) itp))
			    (rdtp (when (listp rtp) (cdr rtp)))(idtp (when (listp itp) (cdr itp)))
			    (k (cmp-real-tp rctp ictp)))
  (if k
      (when (funcall (cdddr (rassoc k +complex*-typep-alist+ :key 'car)) o)
	(and (ibb (realpart o) rdtp) (ibb (imagpart o) idtp)))
    (when (complex*-simple-typep-fn o)
      (and (or (eq rtp '*) (typep (realpart o) rtp))
	   (or (eq itp '*) (typep (imagpart o) itp))))))

(define-typep-fn complex (o tp &aux (rtp (if tp (pop tp) '*))
			   (rlp (listp rtp))(rctp (if rlp (car rtp) rtp))(rdtp (when rlp (cdr rtp)))
			   (k (cmp-real-tp rctp rctp)))
  (if k
      (when (funcall (cdddr (rassoc k +complex*-typep-alist+ :key 'car)) o)
	(and (ibb (realpart o) rdtp) (ibb (imagpart o) rdtp)))
    (when (complex-simple-typep-fn o)
      (or (eq rtp '*) (and (typep (realpart o) rtp) (typep (imagpart o) rtp))))))


(define-compound-typep-fn structure (o tp)
  (if tp (mss (c-structure-def o) (car tp)) t))
(setf (get 'structure-object 'typep-fn) 'structure-typep-fn);FIXME

(define-compound-typep-fn std-instance (o tp)
  (if tp (when (member (car tp) (si-cpl-or-nil (si-class-of o))) t) t))
(define-compound-typep-fn funcallable-std-instance (o tp)
  (if tp (when (member (car tp) (si-cpl-or-nil (si-class-of o))) t) t))

(define-compound-typep-fn proper-cons (o tp)
  (if tp (and (typep (car o) (car tp)) (if (cdr tp) (typep (cdr o) (cadr tp)) t)) t))
(define-compound-typep-fn improper-cons (o tp)
  (if tp (and (typep (car o) (car tp)) (if (cdr tp) (typep (cdr o) (cadr tp)) t)) t))
(define-compound-typep-fn cons (o tp)
  (if tp (and (typep (car o) (car tp)) (if (cdr tp) (typep (cdr o) (cadr tp)) t)) t))

(define-typep-fn eql (o tp)
  (when tp (eql o (car tp))))

(define-typep-fn member (o tp)
  (when tp (when (member o tp) t)))

(define-simple-typep-fn t)
(define-simple-typep-fn nil)


(define-typep-fn satisfies (o tp)
  (funcall (car tp) o))


(defun typep (x type &optional env
		&aux (lp (listp type))(ctp (if lp (car type) type))(tp (when lp (cdr type)))
		(sfn (unless tp (lookup-simple-typep-fn ctp)))(fn (unless sfn (lookup-typep-fn ctp))))
  (declare (ignore env))
  (cond (sfn (when (funcall sfn x) t))
	(fn (when (funcall fn x tp) t))
	((case ctp (values t) (function tp) (otherwise (not (or (symbolp ctp) (si-classp ctp)))))
	 (error 'type-error :datum type :expected-type 'type-spec))
	((typep x (expand-deftype type)))))

(setq *typep-defined* t);FIXME


(defun array-offset (x)
  (typecase
   x
   ((and (array bit) adjustable-array) (c-array-offset x))
   (otherwise 0)))
(setf (get 'array-offset 'cmp-inline) t)



;; (defun open-stream-p (x)
;;   (declare (optimize (safety 1)))
;;   (typecase x (open-stream t)))

(defun input-stream-p (x)
  (declare (optimize (safety 1)))
  (etypecase
   x
   (broadcast-stream nil)
   (string-output-stream nil)
   (file-output-stream nil)
   (file-probe-stream nil)
   (synonym-stream (input-stream-p (symbol-value (synonym-stream-symbol x))))
   (stream t)))


;; (defun interactive-stream-p (x)
;;   (declare (optimize (safety 1)))
;;   (typecase x (interactive-stream t)))

(defun output-stream-p (x)
  (declare (optimize (safety 1)))
  (etypecase
   x
   (concatenated-stream nil)
   (string-input-stream nil)
   (file-input-stream nil)
   (file-probe-stream nil)
   (synonym-stream (output-stream-p (symbol-value (synonym-stream-symbol x))))
   (stream t)))

(defun floatp (x)
  (declare (optimize (safety 1)))
  (typecase x (float t)))

(defun numberp (x)
  (declare (optimize (safety 1)))
  (typecase x (number t)))

(defun characterp (x)
  (declare (optimize (safety 1)))
  (typecase x (character t)))

(defun readtablep (x)
  (declare (optimize (safety 1)))
  (typecase x (readtable t)))

(defun realp (x)
  (declare (optimize (safety 1)))
  (typecase x (real t)))

(defun integerp (x)
  (declare (optimize (safety 1)))
  (typecase x (integer t)))

(defun rationalp (x)
  (declare (optimize (safety 1)))
  (typecase x (rational t)))


(defun complexp (x)
  (declare (optimize (safety 1)))
  (typecase x (complex t)))

(defun bit-vector-p (x)
  (declare (optimize (safety 1)))
  (typecase x (bit-vector t)))

(defun simple-string-p (x)
  (declare (optimize (safety 1)))
  (typecase x (simple-string t)))

(defun simple-vector-p (x)
  (declare (optimize (safety 1)))
  (typecase x (simple-vector t)))

(defun streamp (x)
  (declare (optimize (safety 1)))
  (typecase x (stream t)))

(defun arrayp (x)
  (declare (optimize (safety 1)))
  (typecase x (array t)))

(defun vectorp (x)
  (declare (optimize (safety 1)))
  (typecase x (vector t)))

(defun packagep (x)
  (declare (optimize (safety 1)))
  (typecase x (package t)))

(defun simple-bit-vector-p (x)
  (declare (optimize (safety 1)))
  (typecase x (simple-bit-vector t)))

(defun random-state-p (x)
  (declare (optimize (safety 1)))
  (typecase x (random-state t)))
