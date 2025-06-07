;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defconstant +vtps+ (mapcar (lambda (x) (list x (intern (string-concatenate "VECTOR-"  (string x))))) +array-types+))
(defconstant +atps+ (mapcar (lambda (x) (list x (intern (string-concatenate "ARRAY-"   (string x))))) +array-types+))
(defconstant +vtpsn+ `((nil vector-nil) ,@+vtps+))
(defconstant +atpsn+ `((nil array-nil) ,@+atps+))


(defun real-rep (x)
  (case x (integer 1) (ratio 1/2) (short-float 1.0s0) (long-float 1.0)))

(defun complex-rep (x)
  (let* ((s (symbolp x))
	 (r (real-rep (if s x (car x))))
	 (i (real-rep (if s x (cadr x)))))
    (complex r i)))

(defun make-string-output-stream (&key (element-type 'character))
  (declare (optimize (safety 1))(ignore element-type))
  (make-string-output-stream-int))

(defconstant +r+ `(,@(when (plusp most-positive-immfix) `((immfix 1)))
		   (bfix  most-positive-fixnum)
		   (bignum (1+ most-positive-fixnum))
		   (ratio 1/2)
		   (short-float 1.0s0)
		   (long-float 1.0)
		   ,@(mapcar (lambda (x &aux (v (complex-rep (car x))))
			       `(,(cadr x) ,v)) +ctps+)
		   (standard-char #\a)
		   (non-standard-base-char #\Return)
		   (structure (make-dummy-structure))
		   (std-instance (set-d-tt 1 (make-dummy-structure)))
		   (funcallable-std-instance (set-d-tt 1 (lambda nil nil)))
		   (non-logical-pathname (init-pathname nil nil nil nil nil nil ""))
		   (logical-pathname (set-d-tt 1 (init-pathname nil nil nil nil nil nil "")))
		   (hash-table-eq (make-hash-table :test 'eq))
		   (hash-table-eql (make-hash-table :test 'eql))
		   (hash-table-equal (make-hash-table :test 'equal))
		   (hash-table-equalp (make-hash-table :test 'equalp))
		   (package *package*)
		   (file-input-stream (let ((s (open-int "/dev/null" :input 'character nil nil nil nil :default))) (close s) s))
		   (file-output-stream (let ((s (open-int "/dev/null" :output 'character nil nil nil nil :default))) (close s) s))
		   (file-io-stream (let ((s (open-int "/dev/null" :io 'character nil nil nil nil :default))) (close s) s))
		   (file-probe-stream (let ((s (open-int "/dev/null" :probe 'character nil nil nil nil :default))) (close s) s))
		   (file-synonym-stream (let* ((*standard-output* (open-int "/dev/null" :output 'character nil nil nil nil :default))) (close *standard-output*)  (make-synonym-stream '*standard-output*)))
		   (non-file-synonym-stream *debug-io*);FIXME
		   (broadcast-stream (make-broadcast-stream))
		   (concatenated-stream (make-concatenated-stream))
		   (two-way-stream *terminal-io*)
		   (echo-stream (make-echo-stream *standard-output* *standard-output*))
		   (string-input-stream (make-string-input-stream-int (make-vector 'character 0 t 0 nil 0 nil nil) 0 0))
		   (string-output-stream (make-string-output-stream));FIXME user defined, socket
		   (random-state (make-random-state)) 
		   (readtable (standard-readtable)) 
		   (non-standard-object-compiled-function (function eq))
		   (interpreted-function (set-d-tt 2 (lambda nil nil)))
		   ,@(mapcar (lambda (x)
			       `((simple-array ,(car x) 1)
				 (make-vector ',(car x) 1 nil nil nil 0 nil nil))) +vtps+)
		   ,@(mapcar (lambda (x)
			       `((matrix ,(car x))
				 (make-array1 ',(car x) nil nil nil 0 '(1 1) t))) +atps+)
		   ((non-simple-array character)
		    (make-vector 'character 1 t nil nil 0 nil nil))
		   ((non-simple-array bit)
		    (make-vector 'bit 1 t nil nil 0 nil nil))
		   ((non-simple-array t)
		    (make-vector 't 1 t nil nil 0 nil nil))
		   ((vector nil) (set-d-tt 16 (make-vector 't 1 t nil nil 0 nil nil)));FIXME
		   ((matrix nil) (set-d-tt 16 (make-array1 't nil nil nil 0 '(1 1) t)));FIXME
                   (spice (alloc-spice))
		   (cons '(1))
		   (keyword :a)
		   (null nil)
		   (true t)
		   (gsym 'a)))

(defconstant +tfns1+ '(tp0 tp1 tp2 tp3 tp4 tp5 tp6 tp7 tp8))

(defconstant +tfnsx+
  '#.(let ((x (lreduce (lambda (y x)
			 (if (> (cadr x) (cadr y)) x y))
		       (mapcar (lambda (x &aux (z (lremove-duplicates
						   (mapcar (lambda (q)
							     (funcall x (eval (cadr q))))
							   +r+))))
				 (list x (length z) (lreduce 'min z) (lreduce 'max z)))
			       +tfns1+) :initial-value (list nil 0))))
       (unless (eql (cadr x) (length +r+))
	 (print (list "type-of functions too general" x (length +r+))))
       x))


(defconstant +type-of-dispatch+
  (make-vector t #.(1+ (- (cadddr +tfnsx+) (caddr +tfnsx+))) nil nil nil 0 nil nil))

(defmacro tp7-ind (x) `(- (#.(car +tfnsx+) ,x) #.(caddr +tfnsx+)))

(defun array-type-of (array-tp array)
  (list array-tp (nth (c-array-elttype array) +array-types+) (array-dimensions array)))

(defun simple-array-type-of (array) (array-type-of 'simple-array array))
(defun non-simple-array-type-of (array) (array-type-of 'non-simple-array array))

(defun integer-type-of (x) `(integer ,x ,x))
(defun ratio-type-of (x) `(ratio ,x ,x))
(defun short-float-type-of (x) `(short-float ,x ,x))
(defun long-float-type-of (x) `(long-float ,x ,x))

(defun complex-type-of (cmp)
  (declare (complex cmp));FIXME
  `(complex* ,(type-of (realpart cmp)) ,(type-of (imagpart cmp))))

(defun structure-type-of (str)
  (sdata-name (c-structure-def str)))

(defun valid-class-name (class &aux (name (si-class-name class)))
  (when (eq class (si-find-class name nil))
    name))
(setf (get 'valid-class-name 'cmp-inline) t)

(defun std-object-type-of (x)
  (let* ((c (si-class-of x))) (or (valid-class-name c) c)))

(defun cons-type-of (x);recurse?
  (if (improper-consp x) 'improper-cons 'proper-cons))

(mapc (lambda (x)
	(setf (aref +type-of-dispatch+ (tp7-ind (eval (cadr x))))
	      (let* ((x (car x))(x (if (listp x) (car x) x)))
		(case
		 x
		 ((immfix bfix bignum) #'integer-type-of)
		 (#.(mapcar 'cadr +ctps+) #'complex-type-of)
		 ((structure simple-array non-simple-array cons ratio short-float long-float)
		  (symbol-function (intern (string-concatenate (string x) "-TYPE-OF"))))
		 (matrix #'simple-array-type-of)
		 ((std-instance funcallable-std-instance) #'std-object-type-of)
		 (otherwise x)))))
      +r+)
		
(defun type-of (x &aux (z (aref +type-of-dispatch+ (tp7-ind x))))
  (if (functionp z) (values (funcall z x)) z))
