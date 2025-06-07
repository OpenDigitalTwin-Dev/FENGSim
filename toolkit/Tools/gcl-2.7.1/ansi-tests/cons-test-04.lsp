;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 07:33:20 1998
;;;; Contains:  Testing of CL Features related to "CONS", part 4

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; push
;;;  There will be a separate test suite
;;;    for ACCESSORS x SETF-like macros

;;; See also places.lsp

(deftest push.1
  (let ((x nil))
    (push 'a x))
  (a))

(deftest push.2
  (let ((x 'b))
    (push 'a x)
    (push 'c x))
  (c a . b))

(deftest push.3
  (let ((x (copy-tree '(a))))
    (push x x)
    (and
     (eqt (car x) (cdr x))
     x))
  ((a) a))

(deftest push.order.1
  (let ((x (list nil)) (i 0) a b)
    (values
     (push (progn (setf a (incf i)) 'z)
	   (car (progn (setf b (incf i)) x)))
     x
     i a b))
  (z) ((z)) 2 1 2)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; pop

(deftest pop.1
  (let ((x (copy-tree '(a b c))))
    (let ((y (pop x)))
      (list x y)))
  ((b c) a))

(deftest pop.2
  (let ((x nil))
    (let ((y (pop x)))
      (list x y)))
  (nil nil))

;;; Confirm argument is executed just once.
(deftest pop.order.1
  (let ((i 0)
	(a (vector (list 'a 'b 'c))))
    (pop (aref a (progn (incf i) 0)))
    (values a i))
  #((b c)) 1)

(deftest push-and-pop
  (let* ((x (copy-tree '(a b)))
	 (y x))
    (push 'c x)
    (and
     (eqt (cdr x) y)
     (pop x)))
  c)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; pushnew

;;; See also places.lsp

(deftest pushnew.1
  (let ((x nil))
    (let ((y (pushnew 'a x)))
      (and
       (eqt x y)
       (equal x '(a))
       t)))
  t)

(deftest pushnew.2
  (let* ((x (copy-tree '(b c d a k f q)))
	 (y (pushnew 'a x)))
    (and
     (eqt x y)
     x))
  (b c d a k f q))

(deftest pushnew.3
  (let* ((x (copy-tree '(1 2 3 4 5 6 7 8)))
	 (y (pushnew 7 x)))
    (and
     (eqt x y)
     x))
  (1 2 3 4 5 6 7 8))

(deftest pushnew.4
  (let* ((x (copy-tree '((a b) 1 "and" c d e)))
	 (y (pushnew (copy-tree '(c d)) x
		     :test 'equal)))
    (and (eqt x y)
	 x))
  ((c d) (a b) 1 "and" c d e))

(deftest pushnew.5
  (let* ((x (copy-tree '((a b) 1 "and" c d e)))
	 (y (pushnew (copy-tree '(a b)) x
		     :test 'equal)))
    (and
     (eqt x y)
     x))
  ((a b) 1 "and" c d e))

(deftest pushnew.6
  (let* ((x (copy-tree '((a b) (c e) (d f) (g h))))
	 (y (pushnew (copy-tree '(d i)) x :key #'car))
	 (z (pushnew (copy-tree '(z 10)) x :key #'car)))
    (and (eqt y (cdr z))
	 (eqt z x)
	 x))
  ((z 10) (a b) (c e) (d f) (g h)))

(deftest pushnew.7
  (let* ((x (copy-tree '(("abc" 1) ("def" 2) ("ghi" 3))))
	 (y (pushnew (copy-tree '("def" 4)) x
		     :key #'car :test #'string=))
	 (z (pushnew (copy-tree '("xyz" 10))
		     x
		     :key #'car :test #'string=)))
    (and
     (eqt y (cdr x))
     (eqt x z)
     x))
  (("xyz" 10) ("abc" 1) ("def" 2) ("ghi" 3)))

(deftest pushnew.8
  (let* ((x (copy-tree '(("abc" 1) ("def" 2) ("ghi" 3))))
	 (y (pushnew (copy-tree '("def" 4)) x
		     :key #'car :test-not (complement #'string=)))
	 (z (pushnew (copy-tree '("xyz" 10)) x
		     :key #'car :test-not (complement #'string=))))
    (and
     (eqt y (cdr x))
     (eqt x z)
     x))
  (("xyz" 10) ("abc" 1) ("def" 2) ("ghi" 3)))

(deftest pushnew.9
  (let* ((x (copy-tree '(("abc" 1) ("def" 2) ("ghi" 3))))
	 (y (pushnew (copy-tree '("def" 4)) x
		     :key 'car :test-not (complement #'string=)))
	 (z (pushnew (copy-tree '("xyz" 10)) x
		     :key 'car :test-not (complement #'string=))))
    (and
     (eqt y (cdr x))
     (eqt x z)
     x))
  (("xyz" 10) ("abc" 1) ("def" 2) ("ghi" 3)))

;; Check that a NIL :key argument is the same as no key argument at all
(deftest pushnew.10
  (let* ((x (list 'a 'b 'c 'd))
	 (result (pushnew 'z x :key nil)))
    result)
  (z a b c d))

;; Check that a NIL :key argument is the same as no key argument at all
(deftest pushnew.11
  (let* ((x (copy-tree '((a b) 1 "and" c d e)))
	 (y (pushnew (copy-tree '(a b)) x
		     :test 'equal :key nil)))
    (and
     (eqt x y)
     x))
  ((a b) 1 "and" c d e))

(deftest pushnew.12
  (let ((i 0) x y z (d '(b c)))
    (values
     (pushnew (progn (setf x (incf i)) 'a)
	      d
	      :key (progn (setf y (incf i)) #'identity)
	      :test (progn (setf z (incf i)) #'eql))
     d i x y z))
  (a b c) (a b c)
  3 1 2 3)

(deftest pushnew.13
  (let ((i 0) x y z (d '(b c)))
    (values
     (pushnew (progn (setf x (incf i)) 'a)
	      d
	      :key (progn (setf y (incf i)) #'identity)
	      :test-not (progn (setf z (incf i)) (complement #'eql)))
     d i x y z))
  (a b c) (a b c)
  3 1 2 3)

(deftest pushnew.14
  (let ((i 0) x y z (d '(b c)))
    (values
     (pushnew (progn (setf x (incf i)) 'a)
	      d
	      :test (progn (setf z (incf i)) #'eql)
	      :key (progn (setf y (incf i)) #'identity))
     d i x y z))
  (a b c) (a b c)
  3 1 3 2)

(deftest pushnew.15
  (let ((i 0) x y z (d '(b c)))
    (values
     (pushnew (progn (setf x (incf i)) 'a)
	      d
	      :test-not (progn (setf z (incf i)) (complement #'eql))
	      :key (progn (setf y (incf i)) #'identity))
     d i x y z))
  (a b c) (a b c)
  3 1 3 2)

(deftest pushnew.error.1
  (classify-error
   (let ((x '(a b)))
     (pushnew 'c x :test #'identity)))
  program-error)

(deftest pushnew.error.2
  (classify-error
   (let ((x '(a b)))
     (pushnew 'c x :test-not #'identity)))
  program-error)

(deftest pushnew.error.3
  (classify-error
   (let ((x '(a b)))
     (pushnew 'c x :key #'cons)))
  program-error)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; adjoin

(deftest adjoin.1
  (adjoin 'a nil)
  (a))

(deftest adjoin.2
  (adjoin nil nil)
  (nil))

(deftest adjoin.3
  (adjoin 'a '(a))
  (a))

;; Check that a NIL :key argument is the same as no key argument at all
(deftest adjoin.4
  (adjoin 'a '(a) :key nil)
  (a))

(deftest adjoin.5
  (adjoin 'a '(a) :key #'identity)
  (a))

(deftest adjoin.6
  (adjoin 'a '(a) :key 'identity)
  (a))

(deftest adjoin.7
  (adjoin (1+ 11) '(4 3 12 2 1))
  (4 3 12 2 1))

;; Check that the test is EQL, not EQ (by adjoining a bignum)
(deftest adjoin.8
  (adjoin (1+ 999999999999) '(4 1 1000000000000 3816734 a "aa"))
  (4 1 1000000000000 3816734 a "aa"))

(deftest adjoin.9
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a))
  ("aaa" aaa "AAA" "aaa" #\a))

(deftest adjoin.10
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a) :test #'equal)
  (aaa "AAA" "aaa" #\a))

(deftest adjoin.11
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a) :test 'equal)
  (aaa "AAA" "aaa" #\a))

(deftest adjoin.12
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a)
	  :test-not (complement #'equal))
  (aaa "AAA" "aaa" #\a))

(deftest adjoin.14
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a)
	  :test #'equal :key #'identity)
  (aaa "AAA" "aaa" #\a))

(deftest adjoin.15
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a)
	  :test 'equal :key #'identity)
  (aaa "AAA" "aaa" #\a))

;; Test that a :key of NIL is the same as no key at all
(deftest adjoin.16
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a)
	  :test #'equal :key nil)
  (aaa "AAA" "aaa" #\a))

;; Test that a :key of NIL is the same as no key at all
(deftest adjoin.17
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a)
	  :test 'equal :key nil)
  (aaa "AAA" "aaa" #\a))

;; Test that a :key of NIL is the same as no key at all
(deftest adjoin.18
  (adjoin (copy-seq "aaa") '(aaa "AAA" "aaa" #\a)
	  :test-not (complement #'equal) :key nil)
  (aaa "AAA" "aaa" #\a))

(deftest adjoin.order.1
  (let ((i 0) w x y z)
    (values
     (adjoin (progn (setf w (incf i)) 'a)
	     (progn (setf x (incf i)) '(b c d a e))
	     :key (progn (setf y (incf i)) #'identity)
	     :test (progn (setf z (incf i)) #'eql))
     i w x y z))
  (b c d a e)
  4 1 2 3 4)

(deftest adjoin.order.2
  (let ((i 0) w x y z p)
    (values
     (adjoin (progn (setf w (incf i)) 'a)
	     (progn (setf x (incf i)) '(b c d e))
	     :test-not (progn (setf y (incf i)) (complement #'eql))
	     :key (progn (setf z (incf i)) #'identity)
	     :key (progn (setf p (incf i)) nil))
     i w x y z p))
  (a b c d e)
  5 1 2 3 4 5)

(deftest adjoin.allow-other-keys.1
  (adjoin 'a '(b c) :bad t :allow-other-keys t)
  (a b c))

(deftest adjoin.allow-other-keys.2
  (adjoin 'a '(b c) :allow-other-keys t :foo t)
  (a b c))

(deftest adjoin.allow-other-keys.3
  (adjoin 'a '(b c) :allow-other-keys t)
  (a b c))

(deftest adjoin.allow-other-keys.4
  (adjoin 'a '(b c) :allow-other-keys nil)
  (a b c))

(deftest adjoin.allow-other-keys.5
  (adjoin 'a '(b c) :allow-other-keys t :allow-other-keys nil 'bad t)
  (a b c))

(deftest adjoin.repeat-key
  (adjoin 'a '(b c) :test #'eq :test (complement #'eq))
  (a b c))					      

(deftest adjoin.error.1
  (classify-error (adjoin))
  program-error)

(deftest adjoin.error.2
  (classify-error (adjoin 'a))
  program-error)

(deftest adjoin.error.3
  (classify-error (adjoin 'a '(b c) :bad t))
  program-error)

(deftest adjoin.error.4
  (classify-error (adjoin 'a '(b c) :allow-other-keys nil :bad t))
  program-error)

(deftest adjoin.error.5
  (classify-error (adjoin 'a '(b c) 1 2))
  program-error)

(deftest adjoin.error.6
  (classify-error (adjoin 'a '(b c) :test))
  program-error)

(deftest adjoin.error.7
  (classify-error (adjoin 'a '(b c) :test #'identity))
  program-error)

(deftest adjoin.error.8
  (classify-error (adjoin 'a '(b c) :test-not #'identity))
  program-error)

(deftest adjoin.error.9
  (classify-error (adjoin 'a '(b c) :key #'cons))
  program-error)
