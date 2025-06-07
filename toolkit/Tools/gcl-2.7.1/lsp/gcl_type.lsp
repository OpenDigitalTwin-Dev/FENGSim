;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(export '(cmp-norm-tp tp-p
	  cmp-unnorm-tp
;	  type-and type-or1 type>= type<=
	  tp-not tp-and tp-or tp<= tp>= tp= uniq-tp tsrch uniq-sig
	  atomic-tp tp-bnds object-tp
	  cmpt t-to-nil returs-exactly funcallable-symbol-function
	  infer-tp cnum creal long
	  sharp-t-reader +useful-types-alist+ +useful-type-list+ *useful-type-tree*))

(defun sharp-t-reader (stream subchar arg)
  (declare (ignore subchar arg))
  (let ((tp (cmp-norm-tp (read stream))))
    (if (constantp tp) tp `',tp)))
(set-dispatch-macro-character #\# #\t 'sharp-t-reader)


(defmacro cmpt (tp)  `(and (consp ,tp) (member (car ,tp) '(returns-exactly values))))

(defun t-to-nil (x) (unless (eq x t) x))
(setf (get 't-to-nil 'cmp-inline) t)

(let ((f (car (resolve-type `(or (array nil) ,@(mapcar 'car +r+))))))
  (unless (eq t f)
    (print (list "Representative types ill-defined" f))))

(progn
  . #.(let (y)
	(flet ((orthogonalize (x &aux (z y))
			      (setq y (car (resolve-type (list 'or x y))))
			      (car (resolve-type `(and ,x (not ,z))))))

	      (let* ((q1 (lremove
			  nil
			  (mapcar
			   #'orthogonalize
			   `((unsigned-byte 0)
			     ,@(mapcan (lambda (n &aux (m (1- n)))
					  (list `(unsigned-byte ,m) `(signed-byte ,n) `(unsigned-byte ,n)))
					'(2 4))
			     rnkind
			     ,@(mapcan (lambda (n &aux (m (1- n)))
					  (list `(unsigned-byte ,m) `(signed-byte ,n) `(unsigned-byte ,n)))
					'(8 16))
			     seqind
			     ,@(butlast
				(mapcan (lambda (n &aux (m (1- n)))
					  (list `(unsigned-byte ,m) `(signed-byte ,n) `(unsigned-byte ,n)))
					'(29 32 62 64)))
			     (and bignum (integer * -1))
			     (and bignum (integer 0))
			     ,@(mapcan (lambda (x)
					 (mapcar (lambda (y) (cons x y))
						 '((* (-1))(-1 -1) ((-1) (0)) (0 0) ((0) (1)) (1 1) ((1) *))))
				       '(ratio short-float long-float))
			     short-float long-float
			     proper-cons improper-cons (vector nil) (array nil);FIXME
			     ,@(lremove 'gsym (mapcar 'car +r+))))))
		     (q2 (mapcar
			  #'orthogonalize
			  (multiple-value-bind
			   (x y) (ceiling (1+ (length q1)) fixnum-length)
			   (let ((r '(gsym)))
			     (dotimes (i (- y) r) (push `(member ,(gensym)) r)))))))

		(unless (eq y t)
		  (print (list "Types ill-defined" y)))

		`((defconstant +btp-types1+ ',q1)
		  (defconstant +btp-types+  ',(append q1 q2)))))));; pad to fixnum-length with gensym


(defconstant +btp-length+ (length +btp-types+))

(defun make-btp (&optional (i 0)) (make-vector 'bit +btp-length+ nil nil nil 0 nil i))

(eval-when (compile eval)
  (defmacro mbtp-ltv nil `(load-time-value (make-btp))))

(deftype btp nil '(simple-array bit (#.+btp-length+)))

(defun btp-and (x y z)
  (declare (btp x y z));check-type?
  (bit-and x y z))
(defun btp-ior (x y z)
  (declare (btp x y z))
  (bit-ior x y z))
(defun btp-xor (x y z)
  (declare (btp x y z))
  (bit-xor x y z))
(defun btp-andc2 (x y z)
  (declare (btp x y z))
  (bit-andc2 x y z))
(defun btp-orc2 (x y z)
  (declare (btp x y z))
  (bit-orc2 x y z))
(defun btp-not (x y)
  (declare (btp x y))
  (bit-not x y))

(defvar *btps* (let ((i -1))
		 (mapcar (lambda (x &aux (z (make-btp)))
			   (setf (sbit z (incf i)) 1)
			   (list x (nprocess-type (normalize-type x)) z))
			 +btp-types+)))

(defvar *btpa* (let ((i -1)(z (make-vector t +btp-length+ nil nil nil 0 nil nil)))
		 (mapc (lambda (x) (setf (aref z (incf i)) x)) *btps*)
		 z))

(defvar *k-bv* (let ((i -1))
		   (lreduce (lambda (xx x &aux (z (assoc (caaar (cadr x)) xx)))
			      (unless z
				(push (setq z (cons (caaar (cadr x)) (make-btp))) xx))
			      (setf (sbit (cdr z) (incf i)) 1)
			      xx) *btps* :initial-value nil)))


(defvar *nil-tp* (make-btp))
(defvar *t-tp* (make-btp 1))

(defconstant +bit-words+ (ceiling +btp-length+ fixnum-length))


(defun copy-btp (tp &aux (n (make-btp))(ns (c-array-self n))(ts (c-array-self tp)))
  (dotimes (i +bit-words+ n)
    (*fixnum ns i t (*fixnum ts i nil nil))))

(defun btp-equal (x y &aux (xs (c-array-self x))(ys (c-array-self y)));FIXME inline?
  (dotimes (i +bit-words+ t)
    (unless (eql (*fixnum xs i nil nil) (*fixnum ys i nil nil))
      (return-from btp-equal nil))))

(defun copy-tp (x m tp d)
  (cond (tp (list* (copy-btp x) (copy-btp m) tp (let ((a (atomic-ntp tp))) (when a (list a)))))
	((unless (eql d 1)  (btp-equal x *nil-tp*)) nil)
	((unless (eql d -1) (btp-equal m *t-tp*))     t)
	((copy-btp x))))

(defun new-tp4 (k x m d z &aux (nz (unless (eql d -1) (ntp-not z))))
  (dotimes (i +btp-length+ (unless (btp-equal x m) z))
    (unless (zerop (sbit k i))
      (let ((a (aref *btpa* i)))
	(cond ((unless (eql d  1) (ntp-and?c2-nil-p (cadr a)  z nil))
	       (setf (sbit x i) 0))
	      ((unless (eql d -1) (ntp-and?c2-nil-p (cadr a) nz nil))
	       (setf (sbit m i) 1)))))))

(defun tp-mask (m1 x1 &optional m2 (x2 nil x2p)
		&aux (p1 (mbtp-ltv))(p2 (mbtp-ltv)))
    (btp-xor m1 x1 p1)
    (if x2p
	(btp-and p1 (btp-xor m2 x2 p2) p1)
      p1))

(defun atomic-type (tp)
  (when (consp tp)
    (case (car tp)
	  (#.+range-types+
	   (let* ((d (cdr tp))(dd (cadr d))(da (car d)))
	     (and (numberp da) (numberp dd) (eql da dd) d)))
	  ((member eql) (let ((d (cdr tp))) (unless (cdr d) d))))))

(defun singleton-listp (x) (unless (cdr x) (unless (eq t (car x)) x)))

(defun singleton-rangep (x) (when (singleton-listp x) (when (eql (caar x) (cdar x)) (car x))))

(defun singleton-kingdomp (x);sync with member-ld
  (case (car x)
	((proper-cons improper-cons)
	 (let ((x (cddar (singleton-listp (cdr x)))))
	   (when (car x) x)))
	(#.+range-types+ (singleton-rangep (cdr x)))
	(null '(nil));impossible if in +btp-types+
	(true (cdr x));impossible if in +btp-types+
	((structure std-instance funcallable-std-instance)
	 (when (singleton-listp (cdr x)) (unless (listp (cadr x)) (unless (s-class-p (cadr x)) (cdr x)))))
	(#.(mapcar 'cdr *all-array-types*) (when (singleton-listp (cdr x)) (when (arrayp (cadr x)) (cdr x))))
	(otherwise (when (singleton-listp (cdr x)) (unless (listp (cadr x)) (cdr x))))))

(defun atomic-ntp-array-dimensions (ntp)
  (unless (or (cadr ntp) (caddr ntp))
    (when (car ntp)
      (lreduce (lambda (&rest xy) (when (equal (car xy) (cadr xy)) (car xy)))
	       (car ntp)
	       :key (lambda (x)
		      (case (car x)
			(#.(mapcar 'cdr *all-array-types*)
			   (when (singleton-listp (cdr x))
			     (cond ((consp (cadr x))
				    (unless (eq 'rank (caadr x));(improper-consp (cadr x))
				      (unless (member-if 'symbolp (cadr x))
					(cdr x))))
				   ((arrayp (cadr x))
				    (list (array-dimensions (cadr x)))))))))))))


(defun atomic-tp-array-dimensions (tp)
  (when (consp tp)
    (atomic-ntp-array-dimensions (caddr tp))))

(defun atomic-ntp-array-rank (ntp)
  (unless (or (cadr ntp) (caddr ntp))
    (when (car ntp)
      (lreduce (lambda (&rest xy) (when (equal (car xy) (cadr xy)) (car xy)))
	       (car ntp)
	       :key (lambda (x)
		      (case (car x)
			(#.(mapcar 'cdr *all-array-types*)
			   (when (singleton-listp (cdr x))
			     (cond ((consp (cadr x))
				    (if (eq 'rank (caadr x))
					(cdadr x)
					(unless (member-if 'symbolp (cadr x))
					  (length (cadr x)))))
				   ((arrayp (cadr x))
				    (array-rank (cadr x))))))))))))

(defun atomic-tp-array-rank (tp)
  (when (consp tp)
    (atomic-ntp-array-rank (caddr tp))))

(defun atomic-ntp (ntp)
  (unless (cadr ntp)
    (when (singleton-listp (car ntp))
      (singleton-kingdomp (caar ntp)))))

(defun one-bit-btp (x &aux n)
  (dotimes (i +bit-words+ n)
    (let* ((y (*fixnum (c-array-self x) i nil nil))
	   . #.(let* ((m (mod +btp-length+ fixnum-length))(z (~ (<< -1 m))))
		 (unless (zerop m)
		   `((y (if (< i ,(1- +bit-words+)) y (& y ,z)))))))
      (unless (zerop y)
	(let* ((l (1- (integer-length y)))(l (if (minusp y) (1+ l) l)))
	  (if (unless n (eql y (<< 1 l)))
	      (setq n (+ (* i fixnum-length) (end-shft l)))
	    (return nil)))))))

(defun atomic-tp (tp)
  (unless (or (eq tp '*) (when (listp tp) (member (car tp) '(returns-exactly values))));FIXME
    (unless (eq tp t)
      (if (listp tp)
	  (fourth tp)
	(let ((i (one-bit-btp (xtp tp))))
	  (when i
	    (cadr (assoc i *atomic-btp-alist*))))))))

(defun object-index (x)
  (etypecase
   x
   (gsym #.(1- (length +btp-types+)))
   . #.(let ((i -1)) (mapcar (lambda (x) `(,x ,(incf i))) +btp-types1+))))


(defvar *cmp-verbose* nil)

(defvar *atomic-btp-alist* (let ((i -1))
			     (mapcan (lambda (x &aux (z (incf i)))
				       (when (atomic-type x)
					 (list (list z (cons (cadr x) (caddr x))))))
				     +btp-types+)))

(defun object-tp1 (x)
  (when *cmp-verbose* (print (list 'object-type x)))
  (let* ((i (object-index x))(z (caddr (svref *btpa* i))))
      (if (assoc i *atomic-btp-alist*) z
	(copy-tp z *nil-tp* (nprocess-type (normalize-type `(member ,x))) 0))))

(defvar *atomic-type-hash* (make-hash-table :test 'eql))

(defun hashable-atomp (thing &aux (pl (load-time-value (mapcar 'find-package '(:si :cl :keyword)))))
  (cond ((fixnump thing))
	((symbolp thing)
	 (member (symbol-package thing) pl))))

(defun object-tp (x &aux (h (hashable-atomp x)))
  (multiple-value-bind
   (f r) (when h (gethash x *atomic-type-hash*))
   (if r f
     (let ((z (object-tp1 x)))
       (when h (setf (gethash x *atomic-type-hash*) z))
       z))))


(defun comp-tp0 (type &aux (z (nprocess-type (normalize-type type)))
			(m (mbtp-ltv))(x (mbtp-ltv)))

  (when *cmp-verbose* (print (list 'computing type)))

  (btp-xor m m m)
  (btp-xor x x x)

  (when (cadr z)
    (btp-not m m)
    (btp-not x x))

  (if (caddr z)
      (if (cadr z) (btp-not m m) (btp-not x x))
      (dolist (k (car z))
	(let ((a (cdr (assoc (car k) *k-bv*))))
	  (if (cadr z)
	      (btp-andc2 m a m)
	      (btp-ior x a x)))))

  (copy-tp x m (new-tp4 (tp-mask m x) x m 0 z) 0))

(defvar *typep-defined* nil)

(defun comp-tp (type)
  (if (when *typep-defined* (atomic-type type));FIXME bootstrap NULL
      (object-tp (car (atomic-type (normalize-type type))));e.g. FLOAT coercion
    (comp-tp0 type)))

(defun btp-count (x &aux (j 0))
  (dotimes (i +bit-words+ j)
    (let* ((y (*fixnum (c-array-self x) i nil nil))
	   (q (logcount y)))
      (incf j (if (minusp y) (- fixnum-length q) q)))))

;(defun btp-count (x) (count-if-not 'zerop x))

(defun btp-type2 (x &aux (z +tp-t+))
  (dotimes (i +btp-length+ (ntp-not z))
    (unless (zerop (sbit x i))
      (setq z (ntp-and (ntp-not (cadr (aref *btpa* i))) z)))))

(defun btp-type1 (x)
  (car (nreconstruct-type (btp-type2 x))))

(defun btp-type (x &aux (n (>= (btp-count x) #.(ash +btp-length+ -1)))
		     (nn (mbtp-ltv)))
  (if n `(not ,(btp-type1 (btp-not x nn))) (btp-type1 x)))

;(defun btp-type (x) (btp-type1 x))


(defun tp-type (x)
  (when x
    (cond ((eq x t))
	  ((atom x) (btp-type x))
	  ((car (nreconstruct-type (caddr x)))))))

(defun num-bnd (x) (if (listp x) (car x) x))

(defun max-bnd (x y op &aux (nx (num-bnd x)) (ny (num-bnd y)))
  (cond ((or (eq x '*) (eq y '*)) '*)
	((eql nx ny) (if (atom x) x y))
	((funcall op nx ny) x)
	(y)))

(defun rng-bnd2 (y x &aux (mx (car x))(xx (cdr x))(my (car y))(xy (cdr y)))
  (let ((rm (max-bnd mx my '<))(rx (max-bnd xx xy '>)))
    (cond ((and (eql rm mx) (eql rx xx)) x)
	  ((and (eql rm my) (eql rx xy)) y)
	  ((cons rm rx)))))

(defun rng-bnd (y x) (if (cdr x) (if y (rng-bnd2 y x) x) y))

(defvar *btp-bnds*
  (let ((i -1))
    (mapcan (lambda (x)
	      (incf i)
	      (when (and (member (when (listp x) (car x)) +range-types+)
			 (caddr x));unordered
		`((,i ,(cons (cadr x) (caddr x))))))
	    +btp-types+)))

(defun list-merge-sort (l pred key)

  (labels ((ky (x) (if key (funcall key x) x)))
    (let* ((ll (length l)))
      (if (< ll 2) l
	  (let* ((i (ash ll -1))
		 (lf l)
		 (l1 (nthcdr (1- i) l))
		 (rt (prog1 (cdr l1) (rplacd l1 nil)))
		 (lf (list-merge-sort lf pred key))
		 (rt (list-merge-sort rt pred key)))
	    (do (l0 l1) ((not (and lf rt)) l0)
	      (cond ((funcall pred (ky (car rt)) (ky (car lf)))
		     (setq l1 (if l1 (cdr (rplacd l1 rt)) (setq l0 rt)) rt (cdr rt))
		     (unless rt (rplacd l1 lf)))
		    (t (setq l1 (if l1 (cdr (rplacd l1 lf)) (setq l0 lf)) lf (cdr lf))
		       (unless lf (rplacd l1 rt))))))))))


(defvar *btp-bnds<* (list-merge-sort (copy-list *btp-bnds*) (lambda (x y) (eq (max-bnd x y '<) x)) #'caadr))
(defvar *btp-bnds>* (list-merge-sort (copy-list *btp-bnds*) (lambda (x y) (eq (max-bnd x y '>) x)) #'cdadr))

(defun btp-bnds< (x)
  (dolist (l *btp-bnds<*)
    (unless (zerop (sbit x (car l)))
      (return (caadr l)))))

(defun btp-bnds> (x)
  (dolist (l *btp-bnds>*)
    (unless (zerop (sbit x (car l)))
      (return (cdadr l)))))

(defun btp-bnds (z)
  (let ((m (btp-bnds< z))(x (btp-bnds> z)))
    (when (and m x) (cons m x))))

(defun ntp-bnds (x)
  (lreduce (lambda (y x)
	     (lreduce 'rng-bnd
		      (when (member (car x) +range-types+)
			(if (eq (cadr x) t) (return-from ntp-bnds '(* . *))
			  (cdr x)))
		      :initial-value y))
	   (lreduce (lambda (y z)
		      (when (cadr x)
			(unless (assoc z y)
			  (push (list z t) y))) y)
		    +range-types+ :initial-value (car x))
	   :initial-value nil))

(defun tp-bnds (x)
  (when x
    (if (eq x t) '(* . *)
      (if (atom x) (btp-bnds x) (ntp-bnds (caddr x))))))

(defun xtp (tp) (if (listp tp) (car tp) tp))
(setf (get 'xtp 'cmp-inline) t)
(defun mtp (tp) (if (listp tp) (cadr tp) tp))
(setf (get 'mtp 'cmp-inline) t)

(defun ntp-op (op t1 t2)
  (ecase op
	 (and (ntp-and t1 t2))
	 (or (ntp-or t1 t2))))

(defun min-btp-type2 (x)
  (if (< (btp-count x) #.(ash +btp-length+ -1)) (btp-type2 x)
    (ntp-not (btp-type2 (btp-not x x)))))

(defun new-tp1 (op t1 t2 xp mp &aux (tmp (mbtp-ltv)))
  (cond
    ((atom t1)
     (unless (btp-equal xp mp)
       (if (eq op 'and)
	   (ntp-and (caddr t2) (min-btp-type2 (btp-orc2 t1 (xtp t2) tmp)))
	   (ntp-or (caddr t2) (min-btp-type2 (btp-andc2 t1 (mtp t2) tmp))))))
    ((atom t2) (new-tp1 op t2 t1 xp mp))
    ((new-tp4 (tp-mask (pop t1) (pop t1) (pop t2) (pop t2)) xp mp (if (eq op 'and) -1 1)
	      (ntp-op op (car t1) (car t2))))))


(defun cmp-tp-and (t1 t2 &aux (xp (mbtp-ltv))(mp (mbtp-ltv)))
  (btp-and (xtp t1) (xtp t2) xp)
  (cond ((when (atom t1) (btp-equal xp (xtp t2))) t2)
	((when (atom t2) (btp-equal xp (xtp t1))) t1)
	((and (atom t1) (atom t2)) (copy-tp xp xp nil -1))
	((btp-and (mtp t1) (mtp t2) mp)
	 (cond ((when (atom t1) (btp-equal mp t1)) t1)
	       ((when (atom t2) (btp-equal mp t2)) t2)
	       ((copy-tp xp mp (new-tp1 'and t1 t2 xp mp) -1))))))

(defun tp-and (t1 t2)
  (when (and t1 t2)
    (cond ((eq t1 t) t2)((eq t2 t) t1)
	  ((cmp-tp-and t1 t2)))))


(defun cmp-tp-or (t1 t2 &aux (xp (mbtp-ltv))(mp (mbtp-ltv)))
  (btp-ior (mtp t1) (mtp t2) mp)
  (cond ((when (atom t1) (btp-equal mp (mtp t2))) t2)
	((when (atom t2) (btp-equal mp (mtp t1))) t1)
	((and (atom t1) (atom t2)) (copy-tp mp mp nil 1))
	((btp-ior (xtp t1) (xtp t2) xp)
	 (cond ((when (atom t1) (btp-equal xp t1)) t1)
	       ((when (atom t2) (btp-equal xp t2)) t2)
	       ((copy-tp xp mp (new-tp1 'or t1 t2 xp mp) 1))))))

(defun tp-or (t1 t2)
  (cond ((eq t1 t))
	((eq t2 t))
	((not t1) t2)
	((not t2) t1)
	((cmp-tp-or t1 t2))))


(defun cmp-tp-not (tp)
  (if (atom tp)
      (btp-not tp (make-btp))
    (list (btp-not (cadr tp) (make-btp)) (btp-not (car tp) (make-btp)) (ntp-not (caddr tp)))))

(defun tp-not (tp)
  (unless (eq tp t)
    (or (not tp)
	(cmp-tp-not tp))))


(defun tp<= (t1 t2 &aux (p1 (mbtp-ltv))(p2 (mbtp-ltv)))
  (cond ((eq t2 t))
	((not t1))
	((or (not t2) (eq t1 t)) nil)
	((btp-equal *nil-tp* (btp-andc2 (xtp t1) (mtp t2) p1)))
	((btp-equal *nil-tp* (btp-andc2 p1 (btp-andc2 (xtp t2) (mtp t1) p2) p1))
	 (ntp-subtp (caddr t1) (caddr t2)))))

(defun tp>= (t1 t2) (tp<= t2 t1))

(defun tp= (t1 t2);(when (tp<= t1 t2) (tp<= t2 t1)))
  (cond ((or (if t1 (eq t1 t) t) (if t2 (eq t2 t) t)) (eq t1 t2))
	((and (atom t1) (atom t2)) (btp-equal t1 t2))
	((or (atom t1) (atom t2)) nil)
	((and (btp-equal (car t1) (car t2))
	      (btp-equal (cadr t1) (cadr t2))
	      (ntp-subtp (caddr t1) (caddr t2))
	      (ntp-subtp (caddr t2) (caddr t1))))))

(defun tp-p (x)
  (or (null x) (eq x t) (bit-vector-p x)
      (when (listp x)
	(and (bit-vector-p (car x))
	     (bit-vector-p (cadr x))
	     (consp (caddr x))))));FIXME

(defvar *nrm-hash* (make-hash-table :test 'eq))
(defvar *unnrm-hash* (make-hash-table :test 'eq))
(defvar *uniq-hash* (make-hash-table :test 'equal));FIXME type=?
(defvar *intindiv-hash* (make-hash-table :test 'equal))

(defun uniq-integer-individuals-type (type)
  (let ((type `(,(car type) ,@(list-merge-sort (copy-list (cdr type)) #'< nil))))
    (or (gethash type *intindiv-hash*)
	(setf (gethash type *intindiv-hash*) type))))

(defun hashable-typep (x)
  (or (when (symbolp x)
	(unless (is-standard-class (si-find-class x nil))
	  (let ((z (get x 's-data))) (if z (when (s-data-frozen z) x) x))))
      (when (listp x)
	(when (eq (car x) 'member)
	  (unless (member-if-not 'integerp (cdr x))
	    (uniq-integer-individuals-type x))))))

(defun comp-tp1 (x &aux (s (hashable-typep x)))
  (multiple-value-bind
   (r f) (when s (gethash s *nrm-hash*))
   (if f r
       (let* ((y (comp-tp x)))
	 (when (and s (unless (eq y t) y))
	   (setq y (or (gethash y *uniq-hash*) (setf (gethash y *uniq-hash*) y)))
	   (unless (gethash y *unnrm-hash*) (setf (gethash y *unnrm-hash*) s));e.g. first
	   (setf (gethash s *nrm-hash*) y))
	 y))))

(defun cmp-norm-tp (x)
  (cond ((if x (eq x t) t) x)
	((eq x '*) x)
	((when (listp x)
	   (case (car x)
		 ((returns-exactly values) (cons (car x) (mapcar 'cmp-norm-tp (cdr x)))))));FIXME
	((comp-tp1 x))))

(defun tp-type1 (x)
  (multiple-value-bind
	(r f) (gethash x *unnrm-hash*)
    (if f r
	(multiple-value-bind
	      (r f) (gethash (gethash x *uniq-hash*) *unnrm-hash*)
	  (if f r (tp-type x))))))

(defun cmp-unnorm-tp (x)
  (cond ((tp-p x) (tp-type1 x))
	((when (listp x)
	   (case (car x)
		 ((not returns-exactly values) (cons (car x) (mapcar 'cmp-unnorm-tp (cdr x)))))))
	(x)))






















(defconstant +rn+ '#.(mapcar (lambda (x) (cons (cmp-norm-tp (car x)) (cadr x))) +r+))


(defconstant +tfns1+ '(tp0 tp1 tp2 tp3 tp4 tp5 tp6 tp7 tp8))

(defconstant +rs+ (mapcar (lambda (x)
				 (cons x
				       (mapcar (lambda (y)
						 (cons (car y) (funcall x (eval (cdr y)))))
					       +rn+)))
			       +tfns1+))

(defconstant +kt+ (mapcar 'car +rn+))

(defun tps-ints (a rl)
  (lremove-duplicates (mapcar (lambda (x) (cdr (assoc (cadr x) rl))) a)))

(defun ints-tps (a rl)
  (lreduce (lambda (y x) (if (member (cdr x) a) (tp-or y (car x)) y)) rl :initial-value nil))


(eval-when
 (compile eval)
 (defun msym (x) (intern (string-concatenate (string x) "-TYPE-PROPAGATOR") :si)))

(defconstant +ktn+ (mapcar (lambda (x) (cons x (tp-not x))) +kt+))

(defun decidable-type-p (x)
  (or (atom x)
      (not (third (third x)))))

(defun type-and-list (tps)
  (mapcan (lambda (x &aux (q x))
	    (mapcan (lambda (y)
		      (unless (tp<= q (cdr y))
			`((,x ,(car y)
			      ,(cond ((tp<= (car y) x) (car y))
				     ((let ((x (tp-and (car y) x)))
					(when (decidable-type-p x)
					  x)))
				     (x))))))
		    +ktn+))
	  tps))

(defconstant +rq1+
  (mapcar (lambda (x)
	    (cons (pop x)
		  (lreduce (lambda (y x &aux (nx (tp-not (car x))))
			     (let ((z (rassoc (cdr x) y)))
			       (if z
				   (setf (car z) (tp-and nx (car z)) y y)
				 (cons (cons nx (cdr x)) y))))
			   x :initial-value nil)))
	  +rs+))

(defun norm-tp-ints (tp rl)
  (cmp-norm-tp
   (cons 'member
	 (lreduce (lambda (y x)
	     (if (tp<= tp (car x)) y (cons (cdr x) y)))
	   rl :initial-value nil))))


(progn;FIXME macrolet norm-tp-ints can only compile-file, not compile
  . #.(mapcar (lambda (x &aux (s (msym x)))
		`(let* ((rl (cdr (assoc ',x +rq1+))))
		   (defun ,s (f x)
		     (declare (ignore f))
		     (norm-tp-ints x rl))
		   (setf (get ',x 'type-propagator) ',s)
		   (setf (get ',x 'c1no-side-effects) t)))
	      +tfns1+))



(defun best-type-of (c)
  (let* ((r (lreduce 'set-difference c :key 'car :initial-value +kt+))
	 (tps (nconc (mapcar 'car c) (list r)))
	 (rs +rs+))
    (declare (special rs));FIXME to prevent unroll of +rs+
    (or (caar (member-if (lambda (x)
			   (let* ((x (cdr x))
				  (z (mapcan
				      (lambda (y)
					(lremove-duplicates
					 (mapcar (lambda (z) (cdr (assoc z x))) y)))
				      tps)))
			     (eq z (lremove-duplicates z))))
			 rs))
	(caar rs))))

(defun calist2 (a)
  (lreduce (lambda (y x &aux (z (rassoc (cdr x) y :test 'equal)));;aggregate identical subtypes, e.g. undecidable
	     (if z (setf (car z) (cons (caar x) (car z)) y y) (setf y (cons x y))))
	   (mapcar (lambda (x)
		     (cons (list x);; collect specified types intersecting with this tps
			   (mapcan (lambda (y &aux (q (caddr y)))
				     (when (eq x (cadr y))
				       (list (cons (car y) (unless (eq q x) q)))));;only subtypes smaller than tps
				   a)))
		   (lreduce (lambda (y x) (adjoin (cadr x) y)) a :initial-value nil));;unique tps
	   :initial-value nil))

(defconstant +useful-type-list+ `(nil
				  null
				  boolean keyword symbol
				  proper-cons cons proper-list list
				  simple-vector simple-string simple-bit-vector
				  string vector array
				  proper-sequence sequence
				  zero one
				  bit rnkind non-negative-char unsigned-char signed-char
				  non-negative-short unsigned-short signed-short
				  seqind seqbnd
				  non-negative-immfix immfix
				  non-negative-fixnum non-negative-bignum non-negative-integer
				  tractable-fixnum fixnum bignum integer rational
				  negative-short-float positive-short-float
				  non-negative-short-float non-positive-short-float
				  short-float
				  negative-long-float positive-long-float
				  non-negative-long-float non-positive-long-float
				  long-float
				  negative-float positive-float
				  non-negative-float non-positive-float
				  float
				  negative-real positive-real
				  non-negative-real non-positive-real
				  real
				  fcomplex dcomplex
				  complex-integer complex-ratio
				  complex-ratio-integer complex-integer-ratio
				  complex
				  number
				  character structure package hash-table function
				  t))
;; (defconstant +useful-types+ (mapcar 'cmp-norm-tp +useful-type-list+))
(defconstant +useful-types-alist+ (mapcar (lambda (x) (cons x (cmp-norm-tp x))) +useful-type-list+))

(defvar *useful-type-tree*
  (labels ((cons-count (f)
	     (cond ((atom f) 0)
		   ((+ 1 (cons-count (car f)) (cons-count (cdr f))))))
	   (group-useful-types (tp y)
	     (cons tp
		   (list-merge-sort
		    (mapcar (lambda (z) (group-useful-types (car z) (cdr z)))
			    (lreduce (lambda (y x)
				      (if (member-if (lambda (z) (member (car x) (cdr z))) y) y (cons x y)))
				    (list-merge-sort
				     (mapcar (lambda (z) (cons z (lremove z (lremove-if-not (lambda (x) (tp>= z x)) y)))) y)
				     #'> #'length)
				    :initial-value nil))
		    #'> #'cons-count))))
    (cdr (group-useful-types t (mapcan (lambda (x &aux (x (cdr x)))
					 (when x (unless (eq x t) (list x))))
				       +useful-types-alist+)))))


(defun tsrch (tp &optional (y *useful-type-tree*))
  (let ((x (member tp y :test 'tp<= :key 'car)))
    (when x
      (or (tsrch tp (cdar x)) (caar x)))))

(defvar *uniq-tp* (make-hash-table :test 'eq))

(defun uniq-tp (tp)
  (when tp
    (or (eq tp t)
	(let ((x (or (tsrch tp) t)))
	  (if (tp<= x tp) x
	      (let ((y (gethash x *uniq-tp*)))
		(car (or (member tp y :test 'tp=)
			 (setf (gethash x *uniq-tp*) (cons tp y))))))))))

(defvar *uniq-sig* (make-hash-table :test 'equal))

(defun uniq-sig (sig)
  (let ((x (list (mapcar (lambda (x) (if (eq x '*) x (uniq-tp x))) (car sig))
		 (cond ((cmpt (cadr sig)) (cons (caadr sig) (mapcar 'uniq-tp (cdadr sig))))
		       ((eq (cadr sig) '*) (cadr sig))
		       ((uniq-tp (cadr sig)))))))
    (or (gethash x *uniq-sig*) (setf (gethash x *uniq-sig*) x))))

(defun sig= (s1 s2)
  (labels ((s= (l1 l2)
	     (and (eql (length l1) (length l2))
		  (every (lambda (x y) (or (eq x y) (unless (or (symbolp x) (symbolp y)) (tp= x y)))) l1 l2))))
    (or (eq s1 s2)
	(and (s= (car s1) (car s2))
	     (if (or (cmpt (cadr s1)) (cmpt (cadr s2)))
		 (and (cmpt (cadr s1)) (cmpt (cadr s2)) (s= (cadr s1) (cadr s2)))
		 (s= (cdr s1) (cdr s2)))))))
