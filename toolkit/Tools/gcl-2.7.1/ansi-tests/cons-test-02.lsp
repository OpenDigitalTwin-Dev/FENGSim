;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Sat Mar 28 07:30:50 1998
;;;; Contains: Testing of CL Features related to "CONS", part 2

(in-package :cl-test)

(declaim (optimize (safety 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; copy-tree

;; Try copy-tree on a tree containing elements of various kinds
(deftest copy-tree.1
  (let ((x (cons 'a (list (cons 'b 'c)
			  (cons 1 1.2)
			  (list (list "abcde"
				      (make-array '(10) :initial-element (cons 'e 'f)))
				'g)))))
    (let ((y (copy-tree x)))
      (check-cons-copy x y)))
  t)

;; Try copy-tree on *universe*
(deftest copy-tree.2
  (let* ((x (copy-list *universe*))
	 (y (copy-tree x)))
    (check-cons-copy x y))
  t)

(deftest copy-tree.order.1
  (let ((i 0))
    (values
     (copy-tree (progn (incf i) '(a b c)))
     i))
  (a b c) 1)

(deftest copy-tree.error.1
  (classify-error (copy-tree))
  program-error)

(deftest copy-tree.error.2
  (classify-error (copy-tree 'a 'b))
  program-error)

;;;

(deftest sublis.1
  (check-sublis '((a b) g (d e 10 g h) 15 . g)
		'((e . e2) (g . 17)))
  ((a b) 17 (d e2 10 17 h) 15 . 17))

(deftest sublis.2
  (check-sublis '(f6 10 (f4 (f3 (f1 a b) (f1 a p)) (f2 a b)))
		'(((f1 a b) . (f2 a b)) ((f2 a b) . (f1 a b)))
		:test #'equal)
  (f6 10 (f4 (f3 (f2 a b) (f1 a p)) (f1 a b))))

(deftest sublis.3
  (check-sublis '(10 ((10 20 (a b c) 30)) (((10 20 30 40))))
		'((30 . "foo")))
  (10 ((10 20 (a b c) "foo")) (((10 20 "foo" 40)))))

(deftest sublis.4
  (check-sublis (sublis
		 (copy-tree '((a . 2) (b . 4) (c . 1)))
		 (copy-tree '(a b c d e (a b c a d b) f)))
		'((t . "yes"))
		:key #'(lambda (x) (and (typep x 'integer)
					(evenp x))))
  ("yes" "yes" 1 d e ("yes" "yes" 1 "yes" d "yes") f))

(deftest sublis.5
  (check-sublis '("fee" (("fee" "Fie" "foo"))
		  fie ("fee" "fie"))
		`((,(copy-seq "fie") . #\f)))
  ("fee" (("fee" "Fie" "foo")) fie ("fee" "fie")))

(deftest sublis.6
  (check-sublis '("fee" fie (("fee" "Fie" "foo") 1)
		  ("fee" "fie"))
		`((,(copy-seq "fie") . #\f))
		:test 'equal)
  ("fee" fie (("fee" "Fie" "foo") 1) ("fee" #\f)))

(deftest sublis.7
  (check-sublis '(("aa" a b)
		  (z "bb" d)
		  ((x . "aa")))
		`((,(copy-seq "aa") . 1)
		  (,(copy-seq "bb") . 2))
		:test 'equal
		:key #'(lambda (x) (if (consp x) (car x)
				     '*not-present*)))
  (1 (z . 2) ((x . "aa"))))

;; Check that a null key arg is ignored.

(deftest sublis.8
  (check-sublis 
   '(1 2 a b)
   '((1 . 2) (a . b))
   :key nil)
  (2 2 b b))

;;; Order of argument evaluation
(deftest sublis.order.1
  (let ((i 0) w x y z)
    (values
     (sublis
      (progn (setf w (incf i))
	     '((a . z)))
      (progn (setf x (incf i))
	     (copy-tree '(a b c d)))
      :test (progn (setf y (incf i)) #'eql)
      :key (progn (setf z (incf i)) #'identity))
     i w x y z))
  (z b c d)
  4 1 2 3 4)

(deftest sublis.order.2
  (let ((i 0) w x y z)
    (values
     (sublis
      (progn (setf w (incf i))
	     '((a . z)))
      (progn (setf x (incf i))
	     (copy-tree '(a b c d)))
      :key (progn (setf y (incf i)) #'identity)
      :test-not (progn (setf z (incf i)) (complement #'eql))
      )
     i w x y z))
  (z b c d)
  4 1 2 3 4)


;;; Keyword tests

(deftest sublis.allow-other-keys.1
  (sublis nil 'a :bad t :allow-other-keys t)
  a)

(deftest sublis.allow-other-keys.2
  (sublis nil 'a :allow-other-keys t :bad t)
  a)

(deftest sublis.allow-other-keys.3
  (sublis nil 'a :allow-other-keys t)
  a)

(deftest sublis.allow-other-keys.4
  (sublis nil 'a :allow-other-keys nil)
  a)

(deftest sublis.allow-other-keys.5
  (sublis nil 'a :allow-other-keys t :allow-other-keys t :bad t)
  a)

(deftest sublis.keywords.6
  (sublis '((1 . a)) (list 0 1 2) :key #'(lambda (x) (if (numberp x) (1+ x) x))
	  :key #'identity)
  (a 1 2))


;; Argument error cases

(deftest sublis.error.1
  (classify-error (sublis))
  program-error)

(deftest sublis.error.2
  (classify-error (sublis nil))
  program-error)

(deftest sublis.error.3
  (classify-error (sublis nil 'a :test))
  program-error)

(deftest sublis.error.4
  (classify-error (sublis nil 'a :bad-keyword t))
  program-error)

(deftest sublis.error.5
  (classify-error (sublis '((a . 1) (b . 2))
			  (list 'a 'b 'c 'd)
			  :test #'identity))
  program-error)

(deftest sublis.error.6
  (classify-error (sublis '((a . 1) (b . 2))
			  (list 'a 'b 'c 'd)
			  :key #'cons))
  program-error)

(deftest sublis.error.7
  (classify-error (sublis '((a . 1) (b . 2))
			  (list 'a 'b 'c 'd)
			  :test-not #'identity))
  program-error)

;; nsublis

(deftest nsublis.1
  (check-nsublis '((a b) g (d e 10 g h) 15 . g)
		 '((e . e2) (g . 17)))
  ((a b) 17 (d e2 10 17 h) 15 . 17))

(deftest nsublis.2
  (check-nsublis '(f6 10 (f4 (f3 (f1 a b) (f1 a p)) (f2 a b)))
		 '(((f1 a b) . (f2 a b)) ((f2 a b) . (f1 a b)))
		 :test #'equal)
  (f6 10 (f4 (f3 (f2 a b) (f1 a p)) (f1 a b))))

(deftest nsublis.3
  (check-nsublis '(10 ((10 20 (a b c) 30)) (((10 20 30 40))))
		 '((30 . "foo")))
  (10 ((10 20 (a b c) "foo")) (((10 20 "foo" 40)))))

(deftest nsublis.4
  (check-nsublis
   (nsublis (copy-tree '((a . 2) (b . 4) (c . 1)))
	    (copy-tree '(a b c d e (a b c a d b) f)))
   '((t . "yes"))
   :key #'(lambda (x) (and (typep x 'integer)
			   (evenp x))))
  ("yes" "yes" 1 d e ("yes" "yes" 1 "yes" d "yes") f))

(deftest nsublis.5
  (check-nsublis '("fee" (("fee" "Fie" "foo"))
		   fie ("fee" "fie"))
		 `((,(copy-seq "fie") . #\f)))
  ("fee" (("fee" "Fie" "foo")) fie ("fee" "fie")))

(deftest nsublis.6
  (check-nsublis '("fee" fie (("fee" "Fie" "foo") 1)
		   ("fee" "fie"))
		 `((,(copy-seq "fie") . #\f))
		 :test 'equal)
  ("fee" fie (("fee" "Fie" "foo") 1) ("fee" #\f)))

(deftest nsublis.7
  (check-nsublis '(("aa" a b)
		   (z "bb" d)
		   ((x . "aa")))
		 `((,(copy-seq "aa") . 1)
		   (,(copy-seq "bb") . 2))
		 :test 'equal
		 :key #'(lambda (x) (if (consp x) (car x)
				      '*not-present*)))
  (1 (z . 2) ((x . "aa"))))

(deftest nsublis.8
  (nsublis nil 'a :bad-keyword t :allow-other-keys t)
  a)

;; Check that a null key arg is ignored.

(deftest nsublis.9
  (check-nsublis 
   '(1 2 a b)
   '((1 . 2) (a . b))
   :key nil)
  (2 2 b b))

;;; Order of argument evaluation
(deftest nsublis.order.1
  (let ((i 0) w x y z)
    (values
     (nsublis
      (progn (setf w (incf i))
	     '((a . z)))
      (progn (setf x (incf i))
	     (copy-tree '(a b c d)))
      :test (progn (setf y (incf i)) #'eql)
      :key (progn (setf z (incf i)) #'identity))
     i w x y z))
  (z b c d)
  4 1 2 3 4)

(deftest nsublis.order.2
  (let ((i 0) w x y z)
    (values
     (nsublis
      (progn (setf w (incf i))
	     '((a . z)))
      (progn (setf x (incf i))
	     (copy-tree '(a b c d)))
      :key (progn (setf y (incf i)) #'identity)
      :test-not (progn (setf z (incf i)) (complement #'eql))
      )
     i w x y z))
  (z b c d)
  4 1 2 3 4)

;;; Keyword tests

(deftest nsublis.allow-other-keys.1
  (nsublis nil 'a :bad t :allow-other-keys t)
  a)

(deftest nsublis.allow-other-keys.2
  (nsublis nil 'a :allow-other-keys t :bad t)
  a)

(deftest nsublis.allow-other-keys.3
  (nsublis nil 'a :allow-other-keys t)
  a)

(deftest nsublis.allow-other-keys.4
  (nsublis nil 'a :allow-other-keys nil)
  a)

(deftest nsublis.allow-other-keys.5
  (nsublis nil 'a :allow-other-keys t :allow-other-keys t :bad t)
  a)

(deftest nsublis.keywords.6
  (nsublis '((1 . a)) (list 0 1 2)
	   :key #'(lambda (x) (if (numberp x) (1+ x) x))
	   :key #'identity)
  (a 1 2))

;; Argument error cases

(deftest nsublis.error.1
  (classify-error (nsublis))
  program-error)

(deftest nsublis.error.2
  (classify-error (nsublis nil))
  program-error)

(deftest nsublis.error.3
  (classify-error (nsublis nil 'a :test))
  program-error)

(deftest nsublis.error.4
  (classify-error (nsublis nil 'a :bad-keyword t))
  program-error)

(deftest nsublis.error.5
  (classify-error (nsublis '((a . 1) (b . 2))
			   (list 'a 'b 'c 'd)
			   :test #'identity))
  program-error)

(deftest nsublis.error.6
  (classify-error (nsublis '((a . 1) (b . 2))
			   (list 'a 'b 'c 'd)
			   :key #'cons))
  program-error)

(deftest nsublis.error.7
  (classify-error (nsublis '((a . 1) (b . 2))
			   (list 'a 'b 'c 'd)
			   :test-not #'identity))
  program-error)

;;;;;;

(deftest sublis.shared
  (let* ((shared-piece (list 'a 'b))
	 (a (list shared-piece shared-piece)))
    (check-sublis a '((a . b) (b . a))))
  ((b a) (b a)))

(defvar *subst-tree-1* '(10 (30 20 10) (20 10) (10 20 30 40)))

(deftest subst.1
  (check-subst "Z" 30 (copy-tree *subst-tree-1*))
  (10 ("Z" 20 10) (20 10) (10 20 "Z" 40)))

(deftest subst.2
  (check-subst "A" 0 (copy-tree *subst-tree-1*))
  (10 (30 20 10) (20 10) (10 20 30 40)))

(deftest subst.3
  (check-subst "Z" 100 (copy-tree *subst-tree-1*) :test-not #'eql)
  "Z")

(deftest subst.4
  (check-subst 'grape 'dick
	       '(melville wrote (moby dick)))
  (melville wrote (moby grape)))

(deftest subst.5
  (check-subst 'cha-cha-cha 'nil '(melville wrote (moby dick)))
  (melville wrote (moby dick . cha-cha-cha) . cha-cha-cha))

(deftest subst.6
  (check-subst
   '(1 2) '(foo . bar)
   '((foo . baz) (foo . bar) (bar . foo) (baz foo . bar))
   :test #'equal)
  ((foo . baz) (1 2) (bar . foo) (baz 1 2)))

(deftest subst.7
  (check-subst
   'foo "aaa"
   '((1 . 2) (4 . 5) (6 7 8 9 10 (11 12)))
   :key #'(lambda (x) (if (and (numberp x) (evenp x))
			  "aaa"
			nil))
   :test #'string=)
  ((1 . foo) (foo . 5) (foo 7 foo 9 foo (11 foo))))

(deftest subst.8
  (check-subst
   'foo nil
   '((1 . 2) (4 . 5) (6 7 8 9 10 (11 12)))
   :key #'(lambda (x) (if (and (numberp x) (evenp x))
			  (copy-seq "aaa")
			nil))
   :test-not #'equal)
  ((1 . foo) (foo . 5) (foo 7 foo 9 foo (11 foo))))

(deftest subst.9
  (check-subst 'a 'b
	       (copy-tree '(a b c d a b))
	       :key nil)
  (a a c d a a))

;;; Order of argument evaluation
(deftest subst.order.1
  (let ((i 0) v w x y z)
    (values
     (subst (progn (setf v (incf i)) 'b)
	    (progn (setf w (incf i)) 'a)
	    (progn (setf x (incf i)) (copy-tree '((10 a . a) a b c ((a)) z)))
	    :key (progn (setf y (incf i)) #'identity)
	    :test (progn (setf z (incf i)) #'eql))
     i v w x y z))
  ((10 b . b) b b c ((b)) z)
  5 1 2 3 4 5)

(deftest subst.order.2
  (let ((i 0) v w x y z)
    (values
     (subst (progn (setf v (incf i)) 'b)
	    (progn (setf w (incf i)) 'a)
	    (progn (setf x (incf i)) (copy-tree '((10 a . a) a b c ((a)) z)))
	    :test-not (progn (setf y (incf i)) (complement #'eql))
	    :key (progn (setf z (incf i)) #'identity)
	    )
     i v w x y z))
  ((10 b . b) b b c ((b)) z)
  5 1 2 3 4 5)



;;; Keyword tests for subst

(deftest subst.allow-other-keys.1
  (subst 'a 'b (list 'a 'b 'c) :bad t :allow-other-keys t)
  (a a c))

(deftest subst.allow-other-keys.2
  (subst 'a 'b (list 'a 'b 'c) :allow-other-keys t)
  (a a c))

(deftest subst.allow-other-keys.3
  (subst 'a 'b (list 'a 'b 'c) :allow-other-keys nil)
  (a a c))

(deftest subst.allow-other-keys.4
  (subst 'a 'b (list 'a 'b 'c) :allow-other-keys t :bad t)
  (a a c))

(deftest subst.allow-other-keys.5
  (subst 'a 'b (list 'a 'b 'c) :allow-other-keys t :allow-other-keys nil
	 :bad t)
  (a a c))

(deftest subst.keywords.6
  (subst 'a 'b (list 'a 'b 'c) :test #'eq :test (complement #'eq))
  (a a c))


;;; Tests for subst-if, subst-if-not
  
(deftest subst-if.1
  (check-subst-if 'a #'consp '((100 1) (2 3) (4 3 2 1) (a b c)))
  a)

(deftest subst-if-not.1
  (check-subst-if-not '(x) 'consp '(1 (1 2) (1 2 3) (1 2 3 4)))
  ((x)
   ((x) (x) x)
   ((x) (x) (x) x)
   ((x) (x) (x) (x) x)
   x))

(deftest subst-if.2
  (check-subst-if 17 (complement #'listp) '(a (a b) (a c d) (a nil e f g)))
  (17 (17 17) (17 17 17) (17 nil 17 17 17)))

(deftest subst-if.3
  (check-subst-if '(z)
		  (complement #'consp)
		  '(a (a b) (c d e) (f g h i)))
  ((z)
   ((z) (z) z)
   ((z) (z) (z) z)
   ((z) (z) (z) (z) z)
   z))

(deftest subst-if-not.2
  (check-subst-if-not 'a (complement #'listp)
		      '((100 1) (2 3) (4 3 2 1) (a b c)))
  a)

(deftest subst-if.4
  (check-subst-if 'b #'identity '((100 1) (2 3) (4 3 2 1) (a b c))
		  :key #'listp)
  b)

(deftest subst-if-not.3
  (check-subst-if-not 'c #'identity
		      '((100 1) (2 3) (4 3 2 1) (a b c))
		      :key (complement #'listp))
  c)

(deftest subst-if.5
  (check-subst-if 4 #'(lambda (x) (eql x 1))
		  '((1 3) (1) (1 10 20 30) (1 3 x y))
		  :key #'(lambda (x)
			   (and (consp x)
				(car x))))
  (4 4 4 4))

(deftest subst-if-not.4
  (check-subst-if-not
   40
   #'(lambda (x) (not (eql x 17)))
   '((17) (17 22) (17 22 31) (17 21 34 54))
   :key #'(lambda (x)
	    (and (consp x)
		 (car x))))
  (40 40 40 40))

(deftest subst-if.6
  (check-subst-if 'a  #'(lambda (x) (eql x 'b))
		  '((a) (b) (c) (d))
		  :key nil)
  ((a) (a) (c) (d)))

(deftest subst-if-not.5
  (check-subst-if-not 'a  #'(lambda (x) (not (eql x 'b)))
		      '((a) (b) (c) (d))
		      :key nil)
  ((a) (a) (c) (d)))

(deftest subst-if.7
  (let ((i 0) w x y z)
    (values
     (subst-if
      (progn (setf w (incf i)) 'a)
      (progn (setf x (incf i)) #'(lambda (x) (eql x 'b)))
      (progn (setf y (incf i)) (copy-list '(1 2 a b c)))
      :key (progn (setf z (incf i)) #'identity))
     i w x y z))
  (1 2 a a c)
  4 1 2 3 4)

(deftest subst-if-not.7
  (let ((i 0) w x y z)
    (values
     (subst-if-not
      (progn (setf w (incf i)) 'a)
      (progn (setf x (incf i)) #'(lambda (x) (not (eql x 'b))))
      (progn (setf y (incf i)) (copy-list '(1 2 a b c)))
      :key (progn (setf z (incf i)) #'identity))
     i w x y z))
  (1 2 a a c)
  4 1 2 3 4)
  
	       

;;; Keyword tests for subst-if

(deftest subst-if.allow-other-keys.1
  (subst-if 'a #'null nil :bad t :allow-other-keys t)
  a)

(deftest subst-if.allow-other-keys.2
  (subst-if 'a #'null nil :allow-other-keys t)
  a)

(deftest subst-if.allow-other-keys.3
  (subst-if 'a #'null nil :allow-other-keys nil)
  a)

(deftest subst-if.allow-other-keys.4
  (subst-if 'a #'null nil :allow-other-keys t :bad t)
  a)

(deftest subst-if.allow-other-keys.5
  (subst-if 'a #'null nil :allow-other-keys t :allow-other-keys nil :bad t)
  a)

(deftest subst-if.keywords.6
  (subst-if 'a #'null nil :key nil :key (constantly 'b))
  a)

;;; Keywords tests for subst-if-not

(deftest subst-if-not.allow-other-keys.1
  (subst-if-not 'a #'identity nil :bad t :allow-other-keys t)
  a)

(deftest subst-if-not.allow-other-keys.2
  (subst-if-not 'a #'identity nil :allow-other-keys t)
  a)

(deftest subst-if-not.allow-other-keys.3
  (subst-if-not 'a #'identity nil :allow-other-keys nil)
  a)

(deftest subst-if-not.allow-other-keys.4
  (subst-if-not 'a #'identity nil :allow-other-keys t :bad t)
  a)

(deftest subst-if-not.allow-other-keys.5
  (subst-if-not 'a #'identity nil :allow-other-keys t :allow-other-keys nil :bad t)
  a)

(deftest subst-if-not.keywords.6
  (subst-if-not 'a #'identity nil :key nil :key (constantly 'b))
  a)


(defvar *nsubst-tree-1* '(10 (30 20 10) (20 10) (10 20 30 40)))

(deftest nsubst.1
  (check-nsubst "Z" 30 (copy-tree *nsubst-tree-1*))
  (10 ("Z" 20 10) (20 10) (10 20 "Z" 40)))

(deftest nsubst.2
  (check-nsubst "A" 0 (copy-tree *nsubst-tree-1*))
  (10 (30 20 10) (20 10) (10 20 30 40)))

(deftest nsubst.3
  (check-nsubst "Z" 100 (copy-tree *nsubst-tree-1*) :test-not #'eql)
  "Z")

(deftest nsubst.4
  (check-nsubst 'grape 'dick
		'(melville wrote (moby dick)))
  (melville wrote (moby grape)))

(deftest nsubst.5
  (check-nsubst 'cha-cha-cha 'nil '(melville wrote (moby dick)))
  (melville wrote (moby dick . cha-cha-cha) . cha-cha-cha))

(deftest nsubst.6
  (check-nsubst
   '(1 2) '(foo . bar)
   '((foo . baz) (foo . bar) (bar . foo) (baz foo . bar))
   :test #'equal)
  ((foo . baz) (1 2) (bar . foo) (baz 1 2)))

(deftest nsubst.7
  (check-nsubst
   'foo "aaa"
   '((1 . 2) (4 . 5) (6 7 8 9 10 (11 12)))
   :key #'(lambda (x) (if (and (numberp x) (evenp x))
			  "aaa"
			nil))
   :test #'string=)
  ((1 . foo) (foo . 5) (foo 7 foo 9 foo (11 foo))))

(deftest nsubst.8
  (check-nsubst
   'foo nil
   '((1 . 2) (4 . 5) (6 7 8 9 10 (11 12)))
   :key #'(lambda (x) (if (and (numberp x) (evenp x))
			  (copy-seq "aaa")
			nil))
   :test-not #'equal)
  ((1 . foo) (foo . 5) (foo 7 foo 9 foo (11 foo))))

(deftest nsubst.9
  (check-nsubst 'a 'b
		(copy-tree '(a b c d a b))
		:key nil)
  (a a c d a a))

;;; Order of argument evaluation
(deftest nsubst.order.1
  (let ((i 0) v w x y z)
    (values
     (nsubst (progn (setf v (incf i)) 'b)
	     (progn (setf w (incf i)) 'a)
	     (progn (setf x (incf i)) (copy-tree '((10 a . a) a b c ((a)) z)))
	     :key (progn (setf y (incf i)) #'identity)
	     :test (progn (setf z (incf i)) #'eql))
     i v w x y z))
  ((10 b . b) b b c ((b)) z)
  5 1 2 3 4 5)

(deftest nsubst.order.2
  (let ((i 0) v w x y z)
    (values
     (nsubst (progn (setf v (incf i)) 'b)
	     (progn (setf w (incf i)) 'a)
	     (progn (setf x (incf i)) (copy-tree '((10 a . a) a b c ((a)) z)))
	     :test-not (progn (setf y (incf i)) (complement #'eql))
	     :key (progn (setf z (incf i)) #'identity)
	     )
     i v w x y z))
  ((10 b . b) b b c ((b)) z)
  5 1 2 3 4 5)

;;; Keyword tests for nsubst

(deftest nsubst.allow-other-keys.1
  (nsubst 'a 'b (list 'a 'b 'c) :bad t :allow-other-keys t)
  (a a c))

(deftest nsubst.allow-other-keys.2
  (nsubst 'a 'b (list 'a 'b 'c) :allow-other-keys t)
  (a a c))

(deftest nsubst.allow-other-keys.3
  (nsubst 'a 'b (list 'a 'b 'c) :allow-other-keys nil)
  (a a c))

(deftest nsubst.allow-other-keys.4
  (nsubst 'a 'b (list 'a 'b 'c) :allow-other-keys t :bad t)
  (a a c))

(deftest nsubst.allow-other-keys.5
  (nsubst 'a 'b (list 'a 'b 'c) :allow-other-keys t :allow-other-keys nil
	 :bad t)
  (a a c))

(deftest nsubst.keywords.6
  (nsubst 'a 'b (list 'a 'b 'c) :test #'eq :test (complement #'eq))
  (a a c))

;;; Tests for nsubst-if, nsubst-if-not

(deftest nsubst-if.1
    (check-nsubst-if 'a #'consp '((100 1) (2 3) (4 3 2 1) (a b c)))
  a)

(deftest nsubst-if-not.1
  (check-nsubst-if-not '(x) 'consp '(1 (1 2) (1 2 3) (1 2 3 4)))
  ((x)
   ((x) (x) x)
   ((x) (x) (x) x)
   ((x) (x) (x) (x) x)
   x))

(deftest nsubst-if.2
  (check-nsubst-if 17 (complement #'listp) '(a (a b) (a c d) (a nil e f g)))
  (17 (17 17) (17 17 17) (17 nil 17 17 17)))

(deftest nsubst-if.3
  (check-nsubst-if '(z)
		   (complement #'consp)
		   '(a (a b) (c d e) (f g h i)))
  ((z)
   ((z) (z) z)
   ((z) (z) (z) z)
   ((z) (z) (z) (z) z)
   z))

(deftest nsubst-if-not.2
  (check-nsubst-if-not 'a (complement #'listp)
		       '((100 1) (2 3) (4 3 2 1) (a b c)))
  a)

(deftest nsubst-if.4
  (check-nsubst-if 'b #'identity '((100 1) (2 3) (4 3 2 1) (a b c))
		   :key #'listp)
  b)

(deftest nsubst-if-not.3
  (check-nsubst-if-not 'c #'identity
		       '((100 1) (2 3) (4 3 2 1) (a b c))
		       :key (complement #'listp))
  c)

(deftest nsubst-if.5
  (check-nsubst-if 4 #'(lambda (x) (eql x 1))
		   '((1 3) (1) (1 10 20 30) (1 3 x y))
		   :key #'(lambda (x)
			    (and (consp x)
				 (car x))))
  (4 4 4 4))

(deftest nsubst-if-not.4
  (check-nsubst-if-not
   40
   #'(lambda (x) (not (eql x 17)))
   '((17) (17 22) (17 22 31) (17 21 34 54))
   :key #'(lambda (x)
	    (and (consp x)
		 (car x))))
  (40 40 40 40))

(deftest nsubst-if.6
  (check-nsubst-if 'a  #'(lambda (x) (eql x 'b))
		   '((a) (b) (c) (d))
		   :key nil)
  ((a) (a) (c) (d)))

(deftest nsubst-if-not.5
  (check-nsubst-if-not 'a  #'(lambda (x) (not (eql x 'b)))
		       '((a) (b) (c) (d))
		       :key nil)
  ((a) (a) (c) (d)))

(deftest nsubst-if.7
  (nsubst-if 'a #'null nil :bad t :allow-other-keys t)
  a)

(deftest nsubst-if-not.6
  (nsubst-if-not 'a #'null nil :bad t :allow-other-keys t)
  nil)

(deftest nsubst-if.8
  (let ((i 0) w x y z)
    (values
     (nsubst-if
      (progn (setf w (incf i)) 'a)
      (progn (setf x (incf i)) #'(lambda (x) (eql x 'b)))
      (progn (setf y (incf i)) (copy-list '(1 2 a b c)))
      :key (progn (setf z (incf i)) #'identity))
     i w x y z))
  (1 2 a a c)
  4 1 2 3 4)

(deftest nsubst-if-not.7
  (let ((i 0) w x y z)
    (values
     (nsubst-if-not
      (progn (setf w (incf i)) 'a)
      (progn (setf x (incf i)) #'(lambda (x) (not (eql x 'b))))
      (progn (setf y (incf i)) (copy-list '(1 2 a b c)))
      :key (progn (setf z (incf i)) #'identity))
     i w x y z))
  (1 2 a a c)
  4 1 2 3 4)

;;; Keyword tests for nsubst-if

(deftest nsubst-if.allow-other-keys.1
  (nsubst-if 'a #'null nil :bad t :allow-other-keys t)
  a)

(deftest nsubst-if.allow-other-keys.2
  (nsubst-if 'a #'null nil :allow-other-keys t)
  a)

(deftest nsubst-if.allow-other-keys.3
  (nsubst-if 'a #'null nil :allow-other-keys nil)
  a)

(deftest nsubst-if.allow-other-keys.4
  (nsubst-if 'a #'null nil :allow-other-keys t :bad t)
  a)

(deftest nsubst-if.allow-other-keys.5
  (nsubst-if 'a #'null nil :allow-other-keys t :allow-other-keys nil :bad t)
  a)

(deftest nsubst-if.keywords.6
  (nsubst-if 'a #'null nil :key nil :key (constantly 'b))
  a)

;;; Keywords tests for nsubst-if-not

(deftest nsubst-if-not.allow-other-keys.1
  (nsubst-if-not 'a #'identity nil :bad t :allow-other-keys t)
  a)

(deftest nsubst-if-not.allow-other-keys.2
  (nsubst-if-not 'a #'identity nil :allow-other-keys t)
  a)

(deftest nsubst-if-not.allow-other-keys.3
  (nsubst-if-not 'a #'identity nil :allow-other-keys nil)
  a)

(deftest nsubst-if-not.allow-other-keys.4
  (nsubst-if-not 'a #'identity nil :allow-other-keys t :bad t)
  a)

(deftest nsubst-if-not.allow-other-keys.5
  (nsubst-if-not 'a #'identity nil :allow-other-keys t :allow-other-keys nil :bad t)
  a)

(deftest nsubst-if-not.keywords.6
  (nsubst-if-not 'a #'identity nil :key nil :key (constantly 'b))
  a)

;;; Error cases

;;; subst
(deftest subst.error.1
  (classify-error (subst))
  program-error)

(deftest subst.error.2
  (classify-error (subst 'a))
  program-error)

(deftest subst.error.3
  (classify-error (subst 'a 'b))
  program-error)

(deftest subst.error.4
  (classify-error (subst 'a 'b nil :foo nil))
  program-error)

(deftest subst.error.5
  (classify-error (subst 'a 'b nil :test))
  program-error)

(deftest subst.error.6
  (classify-error (subst 'a 'b nil 1))
  program-error)

(deftest subst.error.7
  (classify-error (subst 'a 'b nil :bad t :allow-other-keys nil))
  program-error)

(deftest subst.error.8
  (classify-error (subst 'a 'b (list 'a 'b) :test #'identity))
  program-error)

(deftest subst.error.9
  (classify-error (subst 'a 'b (list 'a 'b) :test-not #'identity))
  program-error)

(deftest subst.error.10
  (classify-error (subst 'a 'b (list 'a 'b) :key #'equal))
  program-error)

;;; nsubst
(deftest nsubst.error.1
  (classify-error (nsubst))
  program-error)

(deftest nsubst.error.2
  (classify-error (nsubst 'a))
  program-error)

(deftest nsubst.error.3
  (classify-error (nsubst 'a 'b))
  program-error)

(deftest nsubst.error.4
  (classify-error (nsubst 'a 'b nil :foo nil))
  program-error)

(deftest nsubst.error.5
  (classify-error (nsubst 'a 'b nil :test))
  program-error)

(deftest nsubst.error.6
  (classify-error (nsubst 'a 'b nil 1))
  program-error)

(deftest nsubst.error.7
  (classify-error (nsubst 'a 'b nil :bad t :allow-other-keys nil))
  program-error)

(deftest nsubst.error.8
  (classify-error (nsubst 'a 'b (list 'a 'b) :test #'identity))
  program-error)

(deftest nsubst.error.9
  (classify-error (nsubst 'a 'b (list 'a 'b) :test-not #'identity))
  program-error)

(deftest nsubst.error.10
  (classify-error (nsubst 'a 'b (list 'a 'b) :key #'equal))
  program-error)

;;; subst-if
(deftest subst-if.error.1
  (classify-error (subst-if))
  program-error)

(deftest subst-if.error.2
  (classify-error (subst-if 'a))
  program-error)

(deftest subst-if.error.3
  (classify-error (subst-if 'a #'null))
  program-error)

(deftest subst-if.error.4
  (classify-error (subst-if 'a #'null nil :foo nil))
  program-error)

(deftest subst-if.error.5
  (classify-error (subst-if 'a #'null nil :test))
  program-error)

(deftest subst-if.error.6
  (classify-error (subst-if 'a #'null nil 1))
  program-error)

(deftest subst-if.error.7
  (classify-error (subst-if 'a #'null nil :bad t :allow-other-keys nil))
  program-error)

(deftest subst-if.error.8
  (classify-error (subst-if 'a #'null (list 'a nil 'c) :key #'cons))
  program-error)

;;; subst-if-not
(deftest subst-if-not.error.1
  (classify-error (subst-if-not))
  program-error)

(deftest subst-if-not.error.2
  (classify-error (subst-if-not 'a))
  program-error)

(deftest subst-if-not.error.3
  (classify-error (subst-if-not 'a #'null))
  program-error)

(deftest subst-if-not.error.4
  (classify-error (subst-if-not 'a #'null nil :foo nil))
  program-error)

(deftest subst-if-not.error.5
  (classify-error (subst-if-not 'a #'null nil :test))
  program-error)

(deftest subst-if-not.error.6
  (classify-error (subst-if-not 'a #'null nil 1))
  program-error)

(deftest subst-if-not.error.7
  (classify-error (subst-if-not 'a #'null nil
				:bad t :allow-other-keys nil))
  program-error)

(deftest subst-if-not.error.8
  (classify-error (subst-if-not 'a #'null (list 'a nil 'c) :key #'cons))
  program-error)

;;; nsubst-if
(deftest nsubst-if.error.1
  (classify-error (nsubst-if))
  program-error)

(deftest nsubst-if.error.2
  (classify-error (nsubst-if 'a))
  program-error)

(deftest nsubst-if.error.3
  (classify-error (nsubst-if 'a #'null))
  program-error)

(deftest nsubst-if.error.4
  (classify-error (nsubst-if 'a #'null nil :foo nil))
  program-error)

(deftest nsubst-if.error.5
  (classify-error (nsubst-if 'a #'null nil :test))
  program-error)

(deftest nsubst-if.error.6
  (classify-error (nsubst-if 'a #'null nil 1))
  program-error)

(deftest nsubst-if.error.7
  (classify-error (nsubst-if 'a #'null nil :bad t :allow-other-keys nil))
  program-error)

(deftest nsubst-if.error.8
  (classify-error (nsubst-if 'a #'null (list 'a nil 'c) :key #'cons))
  program-error)


;;; nsubst-if-not
(deftest nsubst-if-not.error.1
  (classify-error (nsubst-if-not))
  program-error)

(deftest nsubst-if-not.error.2
  (classify-error (nsubst-if-not 'a))
  program-error)

(deftest nsubst-if-not.error.3
  (classify-error (nsubst-if-not 'a #'null))
  program-error)

(deftest nsubst-if-not.error.4
  (classify-error (nsubst-if-not 'a #'null nil :foo nil))
  program-error)

(deftest nsubst-if-not.error.5
  (classify-error (nsubst-if-not 'a #'null nil :test))
  program-error)

(deftest nsubst-if-not.error.6
  (classify-error (nsubst-if-not 'a #'null nil 1))
  program-error)

(deftest nsubst-if-not.error.7
  (classify-error (nsubst-if-not 'a #'null nil
				 :bad t :allow-other-keys nil))
  program-error)

(deftest nsubst-if-not.error.8
  (classify-error (nsubst-if-not 'a #'null (list 'a nil 'c) :key #'cons))
  program-error)

