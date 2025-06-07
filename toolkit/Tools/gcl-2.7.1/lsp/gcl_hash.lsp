;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defun make-hash-table (&key (test 'eql)
			     (size *default-hash-table-size*)
			     (rehash-size *default-hash-table-rehash-size*)
			     (rehash-threshold *default-hash-table-rehash-threshold*))
  (the hash-table (make-hash-table-int test size rehash-size rehash-threshold nil)))

(defun hash-table-p (x)
  (declare (optimize (safety 1)))
  (typecase x (hash-table t)))

(defun htent-key (e) (*fixnum e 0 nil nil))
(setf (get 'htent-key 'compiler::cmp-inline) t)
(defun htent-value (e) (*object e 1 nil nil))
(setf (get 'htent-value 'compiler::cmp-inline) t)
(defun set-htent-key (e y) (*fixnum e 0 t y))
(setf (get 'set-htent-key 'compiler::cmp-inline) t)
(defun set-htent-value (e y) (*object e 1 t y))
(setf (get 'set-htent-value 'compiler::cmp-inline) t)

(defun gethash (x y &optional z)
  (declare (optimize (safety 1)))
  (check-type y hash-table)
  (let ((e (gethash-int x y)))
    (if (eql +objnull+ (htent-key e))
	(values z nil)
      (values (htent-value e) t))))

(defun gethash1 (x y)
  (declare (optimize (safety 1)))
  (check-type y hash-table)
  (let ((e (gethash-int x y)))
    (if (eql +objnull+ (htent-key e))
	nil
      (htent-value e))))

(defun maphash (f h)
  (declare (optimize (safety 1)))
  (check-type h hash-table)
  (let ((n (hash-table-size h)))
    (dotimes (i n)
      (let* ((e (hashtable-self h i))
	     (k (htent-key e)))
	(unless (eql +objnull+ k)
	  (funcall f (nani k) (htent-value e)))))))

(defun remhash (x y)
  (declare (optimize (safety 1)))
  (check-type y hash-table)
  (let ((e (gethash-int x y)))
    (unless (eql +objnull+ (htent-key e))
      (set-htent-key e +objnull+)
      (c-set-hashtable-nent y (1- (c-hashtable-nent y)))
      t)))

(defun clrhash (h)
  (declare (optimize (safety 1)))
  (check-type h hash-table)
  (let ((n (hash-table-size h)))
    (dotimes (i n)
      (let ((e (hashtable-self h i)))
	(set-htent-key e +objnull+)
	(set-htent-value e (nani +objnull+))));FIXNE?
    (c-set-hashtable-nent h 0)
    h))

(defun sxhash (x)
  (declare (optimize (safety 1)))
  (typecase x
	    (symbol (c-symbol-hash x))
	    (otherwise (hash-equal x 0))))

(defun hash-set (k h v)
  (declare (optimize (safety 1)))
  (check-type h hash-table)
  (let ((n (c-hashtable-nent h)))
    (when (>= (1+ n) (c-hashtable-max_ent h))
      (extend-hashtable h))
    (let ((e (gethash-int k h)))
      ;touch hashtable header; ;FIXME GBC
      (c-set-hashtable-nent h (if (eql +objnull+ (htent-key e)) (1+ n) n))
      (set-htent-key e (address k))
      (set-htent-value e v))))
(setf (get 'hash-set 'compiler::cmp-inline) t)

(setf (symbol-function 'hash-table-count) (symbol-function 'c-hashtable-nent))
(setf (symbol-function 'hash-table-size)  (symbol-function 'c-hashtable-size))
(setf (symbol-function 'hash-table-rehash-size)  (symbol-function 'c-hashtable-rhsize))
(setf (symbol-function 'hash-table-rehash-threshold)  (symbol-function 'c-hashtable-rhthresh))

(defun hash-table-test (h)
  (declare (optimize (safety 1)))
  (check-type h hash-table)
  (aref #(eq eql equal equalp) (c-hashtable-test h)))

(defun hash-table-eq-p (x)
  (declare (optimize (safety 1)))
  (typecase x (hash-table (eq 'eq (hash-table-test x)))))

(defun hash-table-eql-p (x)
  (declare (optimize (safety 1)))
  (typecase x (hash-table (eq 'eql (hash-table-test x)))))

(defun hash-table-equal-p (x)
  (declare (optimize (safety 1)))
  (typecase x (hash-table (eq 'equal (hash-table-test x)))))

(defun hash-table-equalp-p (x)
  (declare (optimize (safety 1)))
  (typecase x (hash-table (eq 'equalp (hash-table-test x)))))

