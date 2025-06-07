;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 09:45:22 1998
;;;; Contains: Testing of CL Features related to "CONS", part 17

(in-package :cl-test)

(declaim (optimize (safety 3)))

(defun rev-assoc-list (x)
  (cond
   ((null x) nil)
   ((null (car x))
    (cons nil (rev-assoc-list (cdr x))))
   (t
    (acons (cdar x) (caar x) (rev-assoc-list (cdr x))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; rassoc

(deftest rassoc.1
  (rassoc nil nil)
  nil)

(deftest rassoc.2
  (rassoc nil '(nil))
  nil)

(deftest rassoc.3
  (rassoc nil (rev-assoc-list '(nil (nil . 2) (a . b))))
  (2 . nil))

(deftest rassoc.4
  (rassoc nil '((a . b) (c . d)))
  nil)

(deftest rassoc.5
  (rassoc 'a '((b . a)))
  (b . a))

(deftest rassoc.6
  (rassoc 'a (rev-assoc-list '((:a . b) (#:a . c) (a . d) (a . e) (z . f))))
  (d . a))

(deftest rassoc.7
  (let* ((x (copy-tree (rev-assoc-list '((a . b) (b . c) (c . d)))))
	 (xcopy (make-scaffold-copy x))
	 (result (rassoc 'b x)))
    (and
     (eqt result (second x))
     (check-scaffold-copy x xcopy)))
  t)

(deftest rassoc.8
  (rassoc 1 (rev-assoc-list '((0 . a) (1 . b) (2 . c))))
  (b . 1))

(deftest rassoc.9
  (rassoc (copy-seq "abc")
	  (rev-assoc-list '((abc . 1) ("abc" . 2) ("abc" . 3))))
  nil)

(deftest rassoc.10
  (rassoc (copy-list '(a))
	  (copy-tree (rev-assoc-list '(((a) b) ((a) (c))))))
  nil)

(deftest rassoc.11
  (let ((x (list 'a 'b)))
    (rassoc x
	    (rev-assoc-list `(((a b) c) (,x . d) (,x . e) ((a b) 1)))))
  (d a b))


(deftest rassoc.12
  (rassoc #\e
	  (copy-tree
	   (rev-assoc-list '(("abefd" . 1) ("aevgd" . 2) ("edada" . 3))))
	  :key #'(lambda (x) (char x 1)))
  (2 . "aevgd"))

(deftest rassoc.13
  (rassoc nil
	  (copy-tree
	   (rev-assoc-list
	    '(((a) . b) ( nil . c ) ((nil) . d))))
	  :key #'car)
  (c))

(deftest rassoc.14
  (rassoc (copy-seq "abc")
	  (copy-tree
	   (rev-assoc-list
	    '((abc . 1) ("abc" . 2) ("abc" . 3))))
	  :test #'equal)
  (2 . "abc"))

(deftest rassoc.15
  (rassoc (copy-seq "abc")
	  (copy-tree
	   (rev-assoc-list
	    '((abc . 1) ("abc" . 2) ("abc" . 3))))
	  :test #'equalp)
  (2 . "abc"))

(deftest rassoc.16
  (rassoc (copy-list '(a))
	  (copy-tree
	   (rev-assoc-list '(((a) b) ((a) (c)))))
	  :test #'equal)
  ((b) a))

(deftest rassoc.17
  (rassoc (copy-seq "abc")
	  (copy-tree
	   (rev-assoc-list
	    '((abc . 1) (a . a) (b . b) ("abc" . 2) ("abc" . 3))))
	  :test-not (complement #'equalp))
  (2 . "abc"))

(deftest rassoc.18
  (rassoc 'a 
	  (copy-tree
	   (rev-assoc-list
	    '((a . d)(b . c))))
	  :test-not #'eq)
  (c . b))

(deftest rassoc.19
  (rassoc 'a
	  (copy-tree
	   (rev-assoc-list
	    '((a . d)(b . c))))
	  :test (complement #'eq))
  (c . b))

(deftest rassoc.20
  (rassoc "a"
	  (copy-tree
	   (rev-assoc-list
	    '(("" . 1) (a . 2) ("A" . 6) ("a" . 3) ("A" . 5))))
	  :key #'(lambda (x) (and (stringp x) (string-downcase x)))
	  :test #'equal)
  (6 . "A"))

(deftest rassoc.21
  (rassoc "a"
	  (copy-tree
	   (rev-assoc-list
	    '(("" . 1) (a . 2) ("A" . 6) ("a" . 3) ("A" . 5))))
	  :key #'(lambda (x) (and (stringp x) x))
	  :test #'equal)
  (3 . "a"))

(deftest rassoc.22
  (rassoc "a"
	  (copy-tree
	   (rev-assoc-list
	    '(("" . 1) (a . 2) ("A" . 6) ("a" . 3) ("A" . 5))))
	  :key #'(lambda (x) (and (stringp x) (string-downcase x)))
	  :test-not (complement #'equal))
  (6 . "A"))

(deftest rassoc.23
  (rassoc "a"
	  (copy-tree
	   (rev-assoc-list
	    '(("" . 1) (a . 2) ("A" . 6) ("a" . 3) ("A" . 5))))
	  :key #'(lambda (x) (and (stringp x) x))
	  :test-not (complement #'equal))
  (3 . "a"))

;; Check that it works when test returns a true value
;; other than T

(deftest rassoc.24
  (rassoc 'a
	  (copy-tree
	   (rev-assoc-list
	    '((b . 1) (a . 2) (c . 3))))
	  :test #'(lambda (x y) (and (eqt x y) 'matched)))
  (2 . a))

;; Check that the order of the arguments to :test is correct

(deftest rassoc.25
  (block fail
    (rassoc 'a '((1 . b) (2 . c) (3 . a))
	    :test #'(lambda (x y)
		      (unless (eqt x 'a) (return-from fail 'fail))
		      (eqt x y))))
  (3 . a))

;;; Order of argument evaluation

(deftest rassoc.order.1
  (let ((i 0) x y)
    (values
     (rassoc (progn (setf x (incf i)) 'c)
	     (progn (setf y (incf i)) '((1 . a) (2 . b) (3 . c) (4 . c))))
     i x y))
  (3 . c) 2 1 2)

(deftest rassoc.order.2
  (let ((i 0) x y z)
    (values
     (rassoc (progn (setf x (incf i)) 'c)
	     (progn (setf y (incf i)) '((1 . a) (2 . b) (3 . c) (4 . c)))
	     :test (progn (setf z (incf i)) #'eql))
     i x y z))
  (3 . c) 3 1 2 3)

(deftest rassoc.order.3
  (let ((i 0) x y)
    (values
     (rassoc (progn (setf x (incf i)) 'c)
	    (progn (setf y (incf i)) '((1 . a) (2 . b) (3 . c) (4 . c)))
	    :test #'eql)
     i x y))
  (3 . c) 2 1 2)

(deftest rassoc.order.4
  (let ((i 0) x y z w)
    (values
     (rassoc (progn (setf x (incf i)) 'c)
	    (progn (setf y (incf i)) '((1 . a) (2 . b) (3 . c) (4 . c)))
	    :key (progn (setf z (incf i)) #'identity)
	    :key (progn (setf w (incf i)) #'not))
     i x y z w))
  (3 . c) 4 1 2 3 4)

;;; Keyword tests

(deftest rassoc.allow-other-keys.1
  (rassoc 'b '((1 . a) (2 . b) (3 . c)) :bad t :allow-other-keys t)
  (2 . b))

(deftest rassoc.allow-other-keys.2
  (rassoc 'b '((1 . a) (2 . b) (3 . c)) :allow-other-keys t :bad t)
  (2 . b))

(deftest rassoc.allow-other-keys.3
  (rassoc 'a '((1 . a) (2 . b) (3 . c)) :allow-other-keys t :bad t
	  :test-not #'eql)
  (2 . b))

(deftest rassoc.allow-other-keys.4
  (rassoc 'b '((1 . a) (2 . b) (3 . c)) :allow-other-keys t)
  (2 . b))

(deftest rassoc.allow-other-keys.5
  (rassoc 'b '((1 . a) (2 . b) (3 . c)) :allow-other-keys nil)
  (2 . b))

(deftest rassoc.keywords.6
  (rassoc 'b '((1 . a) (2 . b) (3 . c))
	  :test #'eql :test (complement #'eql))
  (2 . b))

;;; Error tests

(deftest rassoc.error.1
  (classify-error (rassoc))
  program-error)

(deftest rassoc.error.2
  (classify-error (rassoc nil))
  program-error)

(deftest rassoc.error.3
  (classify-error (rassoc nil nil :bad t))
  program-error)

(deftest rassoc.error.4
  (classify-error (rassoc nil nil :key))
  program-error)

(deftest rassoc.error.5
  (classify-error (rassoc nil nil 1 1))
  program-error)

(deftest rassoc.error.6
  (classify-error (rassoc nil nil :bad t :allow-other-keys nil))
  program-error)

(deftest rassoc.error.7
  (classify-error (rassoc 'a '((b . a)(c . d)) :test #'identity))
  program-error)

(deftest rassoc.error.8
  (classify-error (rassoc 'a '((b . a)(c . d)) :test-not #'identity))
  program-error)

(deftest rassoc.error.9
  (classify-error (rassoc 'a '((b . a)(c . d)) :key #'cons))
  program-error)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; rassoc-if

(deftest rassoc-if.1
    (let* ((x (rev-assoc-list '((1 . a) (3 . b) (6 . c) (7 . d))))
	   (xcopy (make-scaffold-copy x))
	   (result (rassoc-if #'evenp x)))
      (and
       (check-scaffold-copy x xcopy)
       (eqt result (third x))
       result))
  (c . 6))

(deftest rassoc-if.2
  (let* ((x (rev-assoc-list '((1 . a) (3 . b) (6 . c) (7 . d))))
	 (xcopy (make-scaffold-copy x))
	 (result (rassoc-if #'oddp x :key #'1+)))
    (and
     (check-scaffold-copy x xcopy)
     (eqt result (third x))
     result))
  (c . 6))

(deftest rassoc-if.3
    (let* ((x (rev-assoc-list '((1 . a) nil (3 . b) (6 . c) (7 . d))))
	   (xcopy (make-scaffold-copy x))
	   (result (rassoc-if #'evenp x)))
      (and
       (check-scaffold-copy x xcopy)
       (eqt result (fourth x))
       result))
  (c . 6))

(deftest rassoc-if.4
    (rassoc-if #'null
	       (rev-assoc-list '((a . b) nil (c . d) (nil . e) (f . g))))
  (e))

;;; Order of argument evaluation

(deftest rassoc-if.order.1
  (let ((i 0) x y)
    (values
     (rassoc-if (progn (setf x (incf i)) #'null)
		(progn (setf y (incf i))
		       '((1 . a) (2 . b) (17) (4 . d))))
     i x y))
  (17) 2 1 2)

(deftest rassoc-if.order.2
  (let ((i 0) x y z)
    (values
     (rassoc-if (progn (setf x (incf i)) #'null)
		(progn (setf y (incf i))
		       '((1 . a) (2 . b) (17) (4 . d)))
		:key (progn (setf z (incf i)) #'null))
     i x y z))
  (1 . a) 3 1 2 3)


;;; Keyword tests

(deftest rassoc-if.allow-other-keys.1
  (rassoc-if #'null '((1 . a) (2) (3 . c)) :bad t :allow-other-keys t)
  (2))

(deftest rassoc-if.allow-other-keys.2
  (rassoc-if #'null '((1 . a) (2) (3 . c)) :allow-other-keys t :bad t)
  (2))

(deftest rassoc-if.allow-other-keys.3
  (rassoc-if #'identity '((1 . a) (2) (3 . c)) :allow-other-keys t :bad t
	  :key 'not)
  (2))

(deftest rassoc-if.allow-other-keys.4
  (rassoc-if #'null '((1 . a) (2) (3 . c)) :allow-other-keys t)
  (2))

(deftest rassoc-if.allow-other-keys.5
  (rassoc-if #'null '((1 . a) (2) (3 . c)) :allow-other-keys nil)
  (2))

(deftest rassoc-if.keywords.6
  (rassoc-if #'identity '((1 . a) (2) (3 . c)) :key #'not :key #'identity)
  (2))

;;; Error tests

(deftest rassoc-if.error.1
  (classify-error (rassoc-if))
  program-error)

(deftest rassoc-if.error.2
  (classify-error (rassoc-if #'null))
  program-error)

(deftest rassoc-if.error.3
  (classify-error (rassoc-if #'null nil :bad t))
  program-error)

(deftest rassoc-if.error.4
  (classify-error (rassoc-if #'null nil :key))
  program-error)

(deftest rassoc-if.error.5
  (classify-error (rassoc-if #'null nil 1 1))
  program-error)

(deftest rassoc-if.error.6
  (classify-error (rassoc-if #'null nil :bad t :allow-other-keys nil))
  program-error)

(deftest rassoc-if.error.7
  (classify-error (rassoc-if #'cons '((a . b)(c . d))))
  program-error)

(deftest rassoc-if.error.8
  (classify-error (rassoc-if #'car '((a . b)(c . d))))
  type-error)

(deftest rassoc-if.error.9
  (classify-error (rassoc-if #'identity '((a . b)(c . d)) :key #'cons))
  program-error)

(deftest rassoc-if.error.10
  (classify-error (rassoc-if #'identity '((a . b)(c . d)) :key #'car))
  type-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; rassoc-if-not

(deftest rassoc-if-not.1
    (let* ((x (rev-assoc-list '((1 . a) (3 . b) (6 . c) (7 . d))))
	   (xcopy (make-scaffold-copy x))
	   (result (rassoc-if-not #'oddp x)))
      (and
       (check-scaffold-copy x xcopy)
       (eqt result (third x))
       result))
  (c . 6))

(deftest rassoc-if-not.2
  (let* ((x (rev-assoc-list '((1 . a) (3 . b) (6 . c) (7 . d))))
	 (xcopy (make-scaffold-copy x))
	 (result (rassoc-if-not #'evenp x :key #'1+)))
    (and
     (check-scaffold-copy x xcopy)
     (eqt result (third x))
     result))
  (c . 6))

(deftest rassoc-if-not.3
    (let* ((x (rev-assoc-list '((1 . a) nil (3 . b) (6 . c) (7 . d))))
	   (xcopy (make-scaffold-copy x))
	   (result (rassoc-if-not #'oddp x)))
      (and
       (check-scaffold-copy x xcopy)
       (eqt result (fourth x))
       result))
  (c . 6))

(deftest rassoc-if-not.4
    (rassoc-if-not #'identity 
		   (rev-assoc-list '((a . b) nil (c . d) (nil . e) (f . g))))
  (e))

;;; Order of argument evaluation

(deftest rassoc-if-not.order.1
  (let ((i 0) x y)
    (values
     (rassoc-if-not (progn (setf x (incf i)) #'identity)
		    (progn (setf y (incf i))
			   '((1 . a) (2 . b) (17) (4 . d))))
     i x y))
  (17) 2 1 2)

(deftest rassoc-if-not.order.2
  (let ((i 0) x y z)
    (values
     (rassoc-if-not (progn (setf x (incf i)) #'identity)
		    (progn (setf y (incf i))
			   '((1 . a) (2 . b) (17) (4 . d)))
		    :key (progn (setf z (incf i)) #'null))
     i x y z))
  (1 . a) 3 1 2 3)

;;; Keyword tests

(deftest rassoc-if-not.allow-other-keys.1
  (rassoc-if-not #'identity '((1 . a) (2) (3 . c)) :bad t :allow-other-keys t)
  (2))

(deftest rassoc-if-not.allow-other-keys.2
  (rassoc-if-not #'values '((1 . a) (2) (3 . c)) :allow-other-keys t :bad t)
  (2))

(deftest rassoc-if-not.allow-other-keys.3
  (rassoc-if-not #'not '((1 . a) (2) (3 . c)) :allow-other-keys t :bad t
	  :key 'not)
  (2))

(deftest rassoc-if-not.allow-other-keys.4
  (rassoc-if-not #'identity '((1 . a) (2) (3 . c)) :allow-other-keys t)
  (2))

(deftest rassoc-if-not.allow-other-keys.5
  (rassoc-if-not #'identity '((1 . a) (2) (3 . c)) :allow-other-keys nil)
  (2))

(deftest rassoc-if-not.allow-other-keys.6
  (rassoc-if-not #'identity '((1 . a) (2) (3 . c)) :allow-other-keys t
		 :allow-other-keys nil :bad t)
  (2))

(deftest rassoc-if-not.keywords.7
  (rassoc-if-not #'identity '((1 . a) (2) (3 . c)) :key #'not :key nil)
  (1 . a))

;;; Error tests

(deftest rassoc-if-not.error.1
  (classify-error (rassoc-if-not))
  program-error)

(deftest rassoc-if-not.error.2
  (classify-error (rassoc-if-not #'null))
  program-error)

(deftest rassoc-if-not.error.3
  (classify-error (rassoc-if-not #'null nil :bad t))
  program-error)

(deftest rassoc-if-not.error.4
  (classify-error (rassoc-if-not #'null nil :key))
  program-error)

(deftest rassoc-if-not.error.5
  (classify-error (rassoc-if-not #'null nil 1 1))
  program-error)

(deftest rassoc-if-not.error.6
  (classify-error (rassoc-if-not #'null nil :bad t :allow-other-keys nil))
  program-error)

(deftest rassoc-if-not.error.7
  (classify-error (rassoc-if-not #'cons '((a . b)(c . d))))
  program-error)

(deftest rassoc-if-not.error.8
  (classify-error (rassoc-if-not #'car '((a . b)(c . d))))
  type-error)

(deftest rassoc-if-not.error.9
  (classify-error (rassoc-if-not #'identity '((a . b)(c . d)) :key #'cons))
  program-error)

(deftest rassoc-if-not.error.10
  (classify-error (rassoc-if-not #'identity '((a . b)(c . d)) :key #'car))
  type-error)
