;-*- Mode:     Lisp -*-
;;;; Author:   Paul Dietz
;;;; Created:  Mon Oct 21 22:58:00 2002
;;;; Contains: Tests for iteration forms other than LOOP

(in-package :cl-test)

;;; Confirm that most macros exist

(defparameter *iteration-macros*
  '(do do* dotimes dolist loop))

(deftest iteration-macros
  (remove-if #'macro-function *iteration-macros*)
  nil)

;;; Tests of DO

(deftest do.1
  (do ((i 0 (1+ i)))
      ((>= i 10) i))
  10)

(deftest do.2
  (do ((i 0 (1+ j))
       (j 0 (1+ i)))
      ((>= i 10) (+ i j)))
  20)

(deftest do.3
  (let ((x nil))
    (do ((i 0 (1+ i)))
	((>= i 10) x)
      (push i x)))
  (9 8 7 6 5 4 3 2 1 0))

(deftest do.4
  (let ((x nil))
    (do ((i 0 (1+ i)))
	((>= i 10) x)
      (declare (fixnum i))
      (push i x)))
  (9 8 7 6 5 4 3 2 1 0))

(deftest do.5
  (do ((i 0 (1+ i)))
      (nil)
    (when (> i 10) (return i)))
  11)

;;; Zero iterations
(deftest do.6
  (do ((i 0 (+ i 10)))
      ((> i -1) i)
    (return 'bad))
  0)

;;; Tests of go tags
(deftest do.7
  (let ((x nil))
    (do ((i 0 (1+ i)))
	((>= i 10) x)
      (go around)
      small
      (push 'a x)
      (go done)
      big
      (push 'b x)
      (go done)
      around
      (if (> i 4) (go big) (go small))
      done))
  (b b b b b a a a a a))

;;; No increment form
(deftest do.8
  (do ((i 0 (1+ i))
       (x nil))
      ((>= i 10) x)
    (push 'a x))
  (a a a a a a a a a a))

;;; No do locals
(deftest do.9
  (let ((i 0))
    (do ()
	((>= i 10) i)
      (incf i)))
  10)

;;; Return of no values
(deftest do.10
  (do ((i 0 (1+ i)))
      ((> i 10) (values))))

;;; Return of two values
(deftest do.11
  (do ((i 0 (1+ i)))
      ((> i 10) (values i (1+ i))))
  11 12)

;;; The results* list is an implicit progn
(deftest do.12
  (do ((i 0 (1+ i)))
      ((> i 10) (incf i) (incf i) i))
  13)

(deftest do.13
  (do ((i 0 (1+ i)))
      ((> i 10)))
  nil)

;; Special var
(deftest do.14
  (let ((x 0))
    (flet ((%f () (locally (declare (special i))
			   (incf x i))))
      (do ((i 0 (1+ i)))
	  ((>= i 10) x)
	(declare (special i))
	(%f))))
  45)

;;; Confirm that the variables in successive iterations are
;;; identical
(deftest do.15
  (mapcar #'funcall
	  (let ((x nil))
	    (do ((i 0 (1+ i)))
		((= i 5) x)
	      (push #'(lambda () i) x))))
  (5 5 5 5 5))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;; Tests of DO*

(deftest do*.1
  (do* ((i 0 (1+ i)))
      ((>= i 10) i))
  10)

(deftest do*.2
  (do* ((i 0 (1+ j))
        (j 0 (1+ i)))
      ((>= i 10) (+ i j)))
  23)

(deftest do*.3
  (let ((x nil))
    (do* ((i 0 (1+ i)))
	((>= i 10) x)
      (push i x)))
  (9 8 7 6 5 4 3 2 1 0))

(deftest do*.4
  (let ((x nil))
    (do* ((i 0 (1+ i)))
	((>= i 10) x)
      (declare (fixnum i))
      (push i x)))
  (9 8 7 6 5 4 3 2 1 0))

(deftest do*.5
  (do* ((i 0 (1+ i)))
      (nil)
    (when (> i 10) (return i)))
  11)

;;; Zero iterations
(deftest do*.6
  (do* ((i 0 (+ i 10)))
      ((> i -1) i)
    (return 'bad))
  0)

;;; Tests of go tags
(deftest do*.7
  (let ((x nil))
    (do* ((i 0 (1+ i)))
	((>= i 10) x)
      (go around)
      small
      (push 'a x)
      (go done)
      big
      (push 'b x)
      (go done)
      around
      (if (> i 4) (go big) (go small))
      done))
  (b b b b b a a a a a))

;;; No increment form
(deftest do*.8
  (do* ((i 0 (1+ i))
       (x nil))
      ((>= i 10) x)
    (push 'a x))
  (a a a a a a a a a a))

;;; No do* locals
(deftest do*.9
  (let ((i 0))
    (do* ()
	((>= i 10) i)
      (incf i)))
  10)

;;; Return of no values
(deftest do*.10
  (do* ((i 0 (1+ i)))
      ((> i 10) (values))))

;;; Return of two values
(deftest do*.11
  (do* ((i 0 (1+ i)))
      ((> i 10) (values i (1+ i))))
  11 12)

;;; The results* list is an implicit progn
(deftest do*.12
  (do* ((i 0 (1+ i)))
      ((> i 10) (incf i) (incf i) i))
  13)

(deftest do*.13
  (do* ((i 0 (1+ i)))
      ((> i 10)))
  nil)

;; Special var
(deftest do*.14
  (let ((x 0))
    (flet ((%f () (locally (declare (special i))
			   (incf x i))))
      (do* ((i 0 (1+ i)))
	   ((>= i 10) x)
	(declare (special i))
	(%f))))
  45)

;;; Confirm that the variables in successive iterations are
;;; identical
(deftest do*.15
  (mapcar #'funcall
	  (let ((x nil))
	    (do* ((i 0 (1+ i)))
		((= i 5) x)
	      (push #'(lambda () i) x))))
  (5 5 5 5 5))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;; Tests for DOLIST

(deftest dolist.1
  (let ((count 0))
    (dolist (x '(a b nil d)) (incf count))
    count)
  4)

(deftest dolist.2
  (let ((count 0))
    (dolist (x '(a nil c d) count) (incf count)))
  4)

(deftest dolist.3
  (let ((count 0))
    (dolist (x nil count) (incf count)))
  0)

(deftest dolist.4
  (let ((y nil))
    (flet ((%f () (locally (declare (special e))
			   (push e y))))
      (dolist (e '(a b c) (reverse y))
	(declare (special e))
	(%f))))
  (a b c))

;;; Tests that it's a tagbody
(deftest dolist.5
  (let ((even nil)
	(odd nil))
    (dolist (i '(1 2 3 4 5 6 7 8) (values (reverse even)
					  (reverse odd)))
      (when (evenp i) (go even))
      (push i odd)
      (go done)
      even
      (push i even)
      done))
  (2 4 6 8)
  (1 3 5 7))

;;; Test that bindings are not normally special
(deftest dolist.6
  (let ((i 0) (y nil))
    (declare (special i))
    (flet ((%f () i))
      (dolist (i '(1 2 3 4))
	(push (%f) y)))
    y)
  (0 0 0 0))

;;; Test multiple return values

(deftest dolist..7
  (dolist (x '(a b) (values))))

(deftest dolist.8
  (let ((count 0))
    (dolist (x '(a b c) (values count count))
      (incf count)))
  3 3)

;;; Test ability to return, and the scope of the implicit
;;; nil block
(deftest dolist.9
  (block nil
    (eqlt (dolist (x '(a b c))
	    (return 1))
	  1))
  t)

(deftest dolist.10
  (block nil
    (eqlt (dolist (x '(a b c))
	    (return-from nil 1))
	  1))
  t)

(deftest dolist.11
  (block nil
    (dolist (x (return 1)))
    2)
  2)

(deftest dolist.12
  (block nil
    (dolist (x '(a b) (return 1)))
    2)
  2)

;;; Check that binding of element var is visible in the result form
(deftest dolist.13
  (dolist (e '(a b c) e))
  nil)

(deftest dolist.14
  (let ((e 1))
    (dolist (e '(a b c) (setf e 2)))
    e)
  1)

(deftest dolist.15
  (let ((x nil))
    (dolist (e '(a b c d e f))
      (push e x)
      (when (eq e 'c) (return x))))
  (c b a))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Tests for DOTIMES

(deftest dotimes.1
  (dotimes (i 10))
  nil)

(deftest dotimes.2
  (dotimes (i 10 'a))
  a)

(deftest dotimes.3
  (dotimes (i 10 (values))))

(deftest dotimes.3a
  (dotimes (i 10 (values 'a 'b 'c)))
  a b c)

(deftest dotimes.4
  (let ((x nil))
    (dotimes (i 5 x) (push i x)))
  (4 3 2 1 0))

(deftest dotimes.5
  (let ((x nil))
    (dotimes (i 0 x) (push i x)))
  nil)

(deftest dotimes.6
  (let ((x nil))
    (dotimes (i -1 x) (push i x)))
  nil)

(deftest dotimes.7
  (let ((x nil))
    (dotimes (i (1- most-negative-fixnum) x) (push i x)))
  nil)

;;; Implicit nil block has the right scope
(deftest dotimes.8
  (block nil
    (dotimes (i (return 1)))
    2)
  2)

(deftest dotimes.9
  (block nil
    (dotimes (i 10 (return 1)))
    2)
  2)

(deftest dotimes.10
  (block nil
    (dotimes (i 10) (return 1))
    2)
  2)

(deftest dotimes.11
  (let ((x nil))
    (dotimes (i 10)
      (push i x)
      (when (= i 5) (return x))))
  (5 4 3 2 1 0))

;;; Check there's an implicit tagbody
(deftest dotimes.12
  (let ((even nil)
	(odd nil))
    (dotimes (i 8 (values (reverse even)
			  (reverse odd)))
      (when (evenp i) (go even))
      (push i odd)
      (go done)
      even
      (push i even)
      done))
  (0 2 4 6)
  (1 3 5 7))

;;; Check that at the time the result form is evaluated,
;;; the index variable is set to the number of times the loop
;;; was executed.

(deftest dotimes.13
  (let ((i 100))
    (dotimes (i 10 i)))
  10)

(deftest dotimes.14
  (let ((i 100))
    (dotimes (i 0 i)))
  0)

(deftest dotimes.15
  (let ((i 100))
    (dotimes (i -1 i)))
  0)

;;; Check that the variable is not bound in the count form
(deftest dotimes.16
  (let ((i nil))
    (values
     i
     (dotimes (i (progn (setf i 'a) 10) i))
     i))
  nil 10 a)

;;; Check special variable decls
(deftest dotimes.17
  (let ((i 0) (y nil))
    (declare (special i))
    (flet ((%f () i))
      (dotimes (i 4)
	(push (%f) y)))
    y)
  (0 0 0 0))

(deftest dotimes.18
  (let ((i 0) (y nil))
    (declare (special i))
    (flet ((%f () i))
      (dotimes (i 4)
	(declare (special i))
	(push (%f) y)))
    y)
  (3 2 1 0))




