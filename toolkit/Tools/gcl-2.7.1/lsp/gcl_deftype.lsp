;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(export '(zero one negative-integer non-negative-integer tractable-fixnum
	  negative-short-float positive-short-float non-positive-short-float
	  non-negative-short-float negative-long-float positive-long-float
	  non-positive-long-float non-negative-long-float negative-float
	  positive-float non-positive-float non-negative-float negative-real
	  positive-real non-positive-real non-negative-real complex*
	  complex-integer complex-integer-ratio complex-ratio-integer seqind seqbnd
	  complex-ratio complex-short-float complex-long-float make-complex unordered));FIXME

(defun default-to-* (x)
  (let* ((z (member-if (lambda (x) (member x lambda-list-keywords)) x))
	 (y (ldiff-nf x z)))
    (if (member-if 'atom y)
	(nconc (mapcar (lambda (x) (if (atom x) `(,x '*) x)) y) z)
	x)))

(defun deftype-lambda-list (x)
  (let* ((y (cdr (member-if (lambda (x) (member x '(&optional &key))) x)))
	 (z (when y (deftype-lambda-list (default-to-* y)))))
    (if (eq y z) x (append (ldiff-nf x y) z))))


(defun no-reg-vars-p (lambda-list)
  (case (car lambda-list)
	((&whole &environment) (no-reg-vars-p (cddr lambda-list)))
	((nil) t)
	(otherwise (member (car lambda-list) lambda-list-keywords))))

(defun maybe-clear-tp (sym &aux (z (find-symbol "*NRM-HASH*")))
  (when z
    (when (boundp z)
      (multiple-value-bind
	    (r f) (gethash sym (symbol-value z))
	(declare (ignore r))
	(when f
	  (remhash sym (symbol-value z)))))))

(defvar *deftype-simple-typep-fns* nil)

(defmacro define-simple-typep-fn (name)
  (labels ((q (n) (intern (string-concatenate
			   (string n) "-SIMPLE-TYPEP-FN")))
	   (f (n &aux (q (q n)))
	     `(progn (defun ,q (o) (declare (ignorable o)) ,(simple-type-case 'o n))
		     (setf (get ',n 'simple-typep-fn) ',q
			   (get ',q 'cmp-inline) t))))
  (cond ((and (fboundp 'simple-type-case) (fboundp 'cmp-norm-tp))
	 `(progn
	    ,@(mapcar #'f (nreverse *deftype-simple-typep-fns*))
	    ,@(setq *deftype-simple-typep-fns* nil)
	    ,(f name)))
	((setq *deftype-simple-typep-fns* (cons name *deftype-simple-typep-fns*))
	 nil))))


(defmacro deftype (name lambda-list &rest body
		   &aux (lambda-list (deftype-lambda-list lambda-list))
		     (fun-name (gensym (string name))))
  ;; Replace undefaultized optional parameter X by (X '*).
  (declare (optimize (safety 2)))
  (multiple-value-bind
   (doc decls ctps body) (parse-body-header body)
   `(progn
	(eval-when (compile eval load)
		   (putprop ',name '(deftype ,name ,lambda-list ,@body) 'deftype-form)
		   (defmacro ,fun-name ,lambda-list ,@decls ,@ctps (block ,name ,@body))
		   (putprop ',name ',fun-name 'deftype-definition)
		   ;; (putprop ',name (defmacro ,fun-name ,lambda-list ,@decls ,@ctps (block ,name ,@body))
		   ;; 	    'deftype-definition)
		   (maybe-clear-tp ',name)
		   (putprop ',name ,doc 'type-documentation))
	,@(when (no-reg-vars-p lambda-list)
	    `((define-simple-typep-fn ,name)))
	',name)))

;;; Some DEFTYPE definitions.

(deftype function-designator nil
  `(or (and symbol (not boolean)) function))
(deftype extended-function-designator nil
  `(or function-designator (cons (member setf) (cons symbol null))))
(deftype hash-table nil
  `(or hash-table-eq hash-table-eql hash-table-equal hash-table-equalp))

;(deftype compiler::funcallable-symbol nil `(satisfies compiler::funcallable-symbol-p));FIXME

(defconstant +ifb+ (- (car (last (multiple-value-list (si::heap-report))))))
(defconstant +ifr+ (ash (- +ifb+)  -1))
(defconstant +ift+ (when (> #.+ifr+ 0) '(integer #.(- +ifr+) #.(1- +ifr+))))

;(deftype immfix () +ift+)
;(deftype bfix nil `(and fixnum (not immfix)))
(deftype eql-is-eq-tp nil
  `(or #.+ift+ (not number)))
(deftype equal-is-eq-tp nil
  `(or #.+ift+ (not (or cons string bit-vector pathname number))))
(deftype equalp-is-eq-tp nil
  `(not (or array hash-table structure cons string  bit-vector pathname number)))

(deftype non-negative-byte (&optional s)
  `(unsigned-byte ,(if (eq s '*) s (1- s))))
(deftype negative-byte (&optional s)
  (normalize-type `(integer  ,(if (eq s '*) s (- (ash 1 (1- s)))) -1)))
(deftype signed-byte (&optional s &aux (n (if (eq s '*) 0 (ash 1 (1- s)))))
  (normalize-type `(integer ,(if (zerop n) s (- n)) ,(if (zerop n) s (1- n)))))
(deftype unsigned-byte (&optional s)
  (normalize-type `(integer 0 ,(if (eq s '*) s (1- (ash 1 s))))))

(deftype non-negative-char nil
  `(non-negative-byte ,char-length))
(deftype negative-char nil
  `(negative-byte ,char-length))
(deftype signed-char nil
  `(signed-byte ,char-length))
(deftype unsigned-char nil
  `(unsigned-byte ,char-length))
(deftype char nil
  `(signed-char))

(deftype non-negative-short nil
  `(non-negative-byte ,short-length))
(deftype negative-short nil
  `(negative-byte ,short-length))
(deftype signed-short nil
  `(signed-byte ,short-length))
(deftype unsigned-short nil
  `(unsigned-byte ,short-length))
(deftype short nil
  `(signed-short))

(deftype non-negative-int nil
  `(non-negative-byte ,int-length))
(deftype negative-int nil
  `(negative-byte ,int-length))
(deftype signed-int nil
  `(signed-byte ,int-length))
(deftype unsigned-int nil
  `(unsigned-byte ,int-length))
(deftype int nil
  `(signed-int))

(deftype non-negative-fixnum nil
  `(non-negative-byte ,fixnum-length))
(deftype negative-fixnum nil
  `(negative-byte ,fixnum-length))
(deftype signed-fixnum nil
  `(signed-byte ,fixnum-length))
(deftype unsigned-fixnum nil
  `(unsigned-byte ,fixnum-length))

(deftype non-negative-lfixnum nil
`(non-negative-byte ,lfixnum-length))
(deftype negative-lfixnum nil
`(negative-byte ,lfixnum-length))
(deftype signed-lfixnum nil
`(signed-byte ,lfixnum-length))
(deftype unsigned-lfixnum nil
`(unsigned-byte ,lfixnum-length))
(deftype lfixnum nil
`(signed-lfixnum))

(deftype fcomplex nil
  `(complex short-float))
(deftype dcomplex nil
  `(complex long-float))

(deftype string (&optional size)
  `(array character (,size)))
(deftype base-string (&optional size)
  `(array base-char (,size)))
(deftype bit-vector (&optional size)
  `(array bit (,size)))

(deftype simple-vector (&optional size)
  `(simple-array t (,size)))
(deftype simple-string (&optional size)
  `(simple-array character (,size)))
(deftype simple-base-string (&optional size)
  `(simple-array base-char (,size)))
(deftype simple-bit-vector (&optional size)
  `(simple-array bit (,size)))

(deftype cons (&optional car cdr)
  `(or (proper-cons ,car ,cdr) (improper-cons ,car ,cdr)))

(deftype proper-cons (&whole w &optional car cdr
		      &aux (a (normalize-type (if (eq car '*) t car)))
			(d (normalize-type (if (eq cdr '*) t cdr))))
  (cond ((and (eq a car) (eq d cdr)) w)
	((and a d) `(,(car w) ,a ,d))))

(setf (get 'improper-cons 'deftype-definition) (get 'proper-cons 'deftype-definition))

(deftype function-name nil
  `(or symbol (proper-cons (member setf) (proper-cons symbol null))))
(deftype function-identifier nil
  `(or function-name (proper-cons (member lambda) t)));;FIXME? t?

(deftype list nil
  `(or cons null))
(deftype sequence nil
  `(or list vector))

(deftype extended-char nil
  nil)
(deftype base-char nil
  `(or standard-char non-standard-base-char))
(deftype character nil
  `(or base-char extended-char))

(deftype stream nil
  `(or broadcast-stream concatenated-stream echo-stream
       file-stream string-stream synonym-stream two-way-stream))
(deftype file-stream nil
  `(or file-input-stream file-output-stream file-io-stream file-probe-stream))
(deftype path-stream nil
  `(or file-stream file-synonym-stream))
(deftype pathname-designator nil
  `(or pathname string path-stream))
(deftype synonym-stream nil
  `(or file-synonym-stream non-file-synonym-stream))
(deftype string-stream nil
  `(or string-input-stream string-output-stream))

(deftype input-stream nil
  `(and stream (satisfies  input-stream-p)))
(deftype output-stream nil
  `(and stream (satisfies  output-stream-p)))

;(deftype bignum nil `(and integer (not fixnum)))
(deftype non-negative-bignum nil
  `(and non-negative-byte (not non-negative-fixnum)))
(deftype negative-bignum nil
  `(and negative-byte (not negative-fixnum)))

(defconstant most-negative-immfix (or (cadr +ift+) 1))
(defconstant most-positive-immfix (or (caddr +ift+) -1))

(deftype rational (&optional low high)
  `(or (integer ,low ,high) (ratio ,low ,high)))

(deftype float (&optional low high)
  `(or (short-float ,low ,high) (long-float ,low ,high)))
(deftype single-float (&optional low high)
  `(long-float ,low ,high))
(deftype double-float (&optional low high)
  `(long-float ,low ,high))
(deftype real (&optional low high)
  `(or (rational ,low ,high) (float ,low ,high)))
(deftype number nil
  `(or real complex))
(deftype atom nil
  `(not cons))
(deftype compiled-function nil
  `(or funcallable-std-instance non-standard-object-compiled-function))
(deftype function (&rest r)
  (declare (ignore r))
  `(or compiled-function interpreted-function))

(deftype string-designator    nil `(or string symbol character (integer 0 255)))



(defun ctp-num-bnd (x tp inc &aux (a (atom x))(nx (if a x (car x))))
  (flet ((f (b)
	   (when (fboundp 'fpe::break-on-floating-point-exceptions);FIXME
	     (fpe::break-on-floating-point-exceptions :suspend t))
	   (let ((z (float nx b)))
	     (when (fboundp 'fpe::break-on-floating-point-exceptions)
	       (fpe::break-on-floating-point-exceptions :suspend nil))
	     (if (eql z nx) x (if a z (list z))))))
  (case tp
    (integer
     (let ((nx (if (unless a (integerp (rational nx))) (+ nx inc) nx)))
       (if (> inc 0) (ceiling nx) (floor nx))))
    (ratio
     (let ((z (rational nx)))
       (if (eql z nx) (if (integerp x) (list x) x)
	   (if a z (list z)))))
    (short-float (f 0.0s0))
    (long-float (f 0.0)))))

(defun ctp-bnd (x tp inc)
  (if (eq x '*) x (ctp-num-bnd x tp inc)))

(defun bnd-chk (l h &aux (nl (if (listp l) (car l) l))(nh (if (listp h) (car h) h)))
  (or (eq l '*) (eq h '*) (< nl nh) (and (eql l h) (eql nl nh) (eql l nl))))

(defun bnd-exp (tp w low high &aux (l (ctp-bnd low tp 1)) (h (ctp-bnd high tp -1)))
  (when (bnd-chk l h)
    (if (and (eql l (cadr w)) (eql h (caddr w)))
	w
	`(,tp ,l ,h))))

(deftype integer (&whole w &optional low high)
  (bnd-exp 'integer w low high))

(deftype ratio (&whole w &optional low high)
  (bnd-exp 'ratio w low high))

(deftype short-float (&whole w &optional low (high '* hp))
  (if (and (eq low 'unordered) (not hp)) w ;This unnecessary extension is simpler than
                                           ;(and short-float (not (or (short-float 0) (short-float * 0))))
      (bnd-exp 'short-float w low high)))

(deftype long-float (&whole w &optional low (high '* hp))
  (if (and (eq low 'unordered) (not hp)) w
      (bnd-exp 'long-float w low high)))


(deftype zero nil `(integer 0 0))
(deftype one nil `(integer 1 1))
(deftype non-negative-integer nil `(integer 0))
(deftype negative-integer nil `(integer * (0)))
(deftype tractable-fixnum nil `(integer ,(- most-positive-fixnum) ,most-positive-fixnum))
(deftype negative-short-float nil `(short-float * (0.0)))
(deftype positive-short-float nil `(short-float (0.0)))
(deftype non-positive-short-float nil `(short-float * 0.0))
(deftype non-negative-short-float nil `(short-float 0.0))
(deftype negative-long-float nil `(long-float * (0.0)))
(deftype positive-long-float nil `(long-float (0.0)))
(deftype non-positive-long-float nil `(long-float * 0.0))
(deftype non-negative-long-float nil `(long-float 0.0))
(deftype negative-float nil `(float * (0.0)))
(deftype positive-float nil `(float (0.0)))
(deftype non-positive-float nil `(float * 0.0))
(deftype non-negative-float nil `(float 0.0))
(deftype negative-real nil `(real * (0.0)))
(deftype positive-real nil `(real (0.0)))
(deftype non-positive-real nil `(real * 0.0))
(deftype non-negative-real nil `(real 0.0))

(deftype double nil 'long-float)

(deftype unadjustable-array nil
  `(or simple-string simple-bit-vector simple-vector))
(deftype adjustable-array nil
  `(and array (not unadjustable-array)))
(deftype adjustable-vector nil
  `(and vector (not unadjustable-array)))
(deftype matrix (&optional et dims)
  `(and (array ,et ,dims) (not vector)))

(deftype simple-array (&whole w &optional et dims)
  (if (eq et '*)
      `(or ,@(mapcar (lambda (x) `(simple-array ,x ,dims)) (cons nil +array-types+)))
      (let* ((e (upgraded-array-element-type et))(d (or dims 0)))
  	(if (and (eq (cadr w) e) (eq (caddr w) d)) w `(simple-array ,e ,d)))))

(deftype non-simple-array (&whole w &optional et dims
				  &aux (ets '(character t bit))
				  (d (cond ((eq dims '*) dims)
					   ((eql dims 1) '*)
					   ((atom dims) nil)
					   ((cdr dims) nil)
					   ((eq (car dims) '*) '*)
					   (dims))))
  (when d
    (if (eq et '*)
	(?or (mapcar (lambda (x) `(non-simple-array ,x ,d)) ets))
      (let* ((e (upgraded-array-element-type et)))
	(when (member e ets)
	  (if (and (eq (cadr w) e) (eq (caddr w) d)) w `(non-simple-array ,e ,d)))))))

(deftype array (&optional et dims) `(or (simple-array ,et ,dims) (non-simple-array ,et ,dims)))

(deftype true nil
  `(member t))
(deftype null nil
  `(member nil))
(deftype boolean nil
  `(or true null))

(deftype symbol nil
  `(or boolean keyword gsym))

(defconstant +ctps+ (mapcar (lambda (x)
			      (list x
				    (intern
				     (string-concatenate
				      "COMPLEX-"
				      (if (consp x)
					  (string-concatenate (string (pop x)) "-" (string (car x)))
				      (string x))))))
			    (cons '(integer ratio) (cons '(ratio integer) +complex-types+))));FIXME

#.`(progn
     ,@(mapcar (lambda (x &aux (s (cadr x))(x (car x)))
		 `(deftype ,s (&optional l h)
		    ,(if (consp x)
			 ``(complex* (,',(pop x) ,l ,h) (,',(car x) ,l ,h))
		       ``(complex (,',x ,l ,h)))))
	       +ctps+))

(defun ?or (x) (if (cdr x) (cons 'or x) (car x)))

(deftype complex (&optional rp) `(complex* ,rp))

(defun ncs (rp &aux (rp (if (eq rp '*) 'real rp)))
  (mapcar (lambda (x) (cons x (car (resolve-type `(and ,x ,rp))))) +range-types+))

(defun make-complex* (r i)
  (when (and (cdr r) (cdr i)) `((complex* ,(cdr r) ,(cdr i)))))

(deftype complex* (&optional rp (ip rp) &aux (rr (ncs rp))(ri (ncs ip)));FIXME upgraded
  (?or (nconc
	(make-complex* (assoc 'integer rr) (assoc 'ratio ri))
	(make-complex* (assoc 'ratio rr) (assoc 'integer ri))
	(mapcan (lambda (x) (make-complex* (assoc x rr) (assoc x ri)))  +range-types+))))
;; &whole w
;; (if (or (equal w x) (member w x :test 'equal));FIXME
;; 	w x)))

(deftype pathname nil
  `(or non-logical-pathname logical-pathname))

(deftype proper-sequence nil
  `(or vector proper-list))

(deftype proper-list nil
  `(or null proper-cons))

(deftype not-type nil
  'null)


(deftype type-spec nil '(or (and symbol (not (eql values)))
			    (proper-cons (and symbol (not (member values function))))
			    (and std-instance (satisfies si-classp))))

(deftype ftype-spec nil
  `(cons (member function)
	 (cons (satisfies arg-list-type-p)
	       (cons (satisfies values-list-type-p)
		     null))))

(deftype fpvec nil
  `(and adjustable-vector (satisfies array-has-fill-pointer-p)))


(deftype vector (&optional et size)
  `(array ,et (,size)))

(deftype non-negative-immfix nil
  `(non-negative-byte ,(1+ (integer-length most-positive-immfix))))
(deftype immfix nil
  #.(when (plusp most-positive-immfix) `'(signed-byte ,(1+ (integer-length most-positive-immfix)))));FIXME check null
(deftype fixnum nil
  `(signed-byte #.(1+ (integer-length most-positive-fixnum))))
(deftype bfix nil
  `(and fixnum (not immfix)))
(deftype bignum nil
  `(or (integer * ,(1- most-negative-fixnum)) (integer ,(1+ most-positive-fixnum) *)))

(deftype function-type-spec nil
  `(cons (member function) t));fixme
(deftype full-type-spec nil
  `(or type-spec function-type-spec))

(deftype seqind nil
  `(integer 0 ,(- array-dimension-limit 2)))
(deftype seqbnd nil
  `(integer 0 ,(1- array-dimension-limit)))
(deftype rnkind nil
  `(integer 0 ,(1- array-rank-limit)))
(deftype mod (n)
  `(integer 0 ,(1- n)))

(deftype bit nil
  `(mod 2))


(defun all-eq (x y)
  (cond ((not (and x y)) t)
	((eq (car x) (car y)) (all-eq (cdr x) (cdr y)))))

(defun and-or-norm (op w r &aux (n (mapcar 'normalize-type r)))
  (if (all-eq r n) w (cons op n)))

(deftype or (&whole w &rest r)
  (when r (and-or-norm 'or w r)))

(deftype and (&whole w &rest r &aux (r (if r r '(t))))
  (and-or-norm 'and w r))

(deftype not (&whole w &rest r)
  (and-or-norm 'not w r));x and-or-flatten

(deftype satisfies (&whole w pred &aux (tp (get pred 'predicate-type)));Note: guard against infinite recursion
  (if tp (normalize-type tp) w))

(deftype eql (&rest r)
  (when r
    (unless (cdr r)
      `(member ,@r))))

(deftype member (&whole w &rest r)
  (when r w))

(deftype cnum nil `(or fixnum float fcomplex dcomplex))
(deftype creal nil `(and real cnum))
(deftype long nil 'fixnum)

(deftype key-test-type nil `(or null function-designator))
