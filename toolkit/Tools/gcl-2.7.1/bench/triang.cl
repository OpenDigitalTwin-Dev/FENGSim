;; $Header$
;; $Locker$

;;; TRIANG -- Board game benchmark.

(declaim (special board seq a b c))
(defvar answer)
(defvar final)

(defun triang-setup ()
  (setq board (make-array 16 :initial-element 1))
  (setq seq (make-array 14 :initial-element 0))
  (setq a
    (make-array
      37
      :initial-contents
      '(1 2 4 3 5 6 1 3 6 2 5 4 11 12 13 7 8 4 4 7 11 8 12
	13 6 10 15 9 14 13 13 14 15 9 10 6 6)))
  (setq b (make-array
	    37 :initial-contents
	    '(2 4 7 5 8 9 3 6 10 5 9 8 12 13 14 8 9 5
	      2 4 7 5 8 9 3 6 10 5 9 8 12 13 14 8 9 5 5)))
  (setq c (make-array
	    37 :initial-contents
	    '(4 7 11 8 12 13 6 10 15 9 14 13 13 14 15 9 10 6
	      1 2 4 3 5 6 1 3 6 2 5 4 11 12 13 7 8 4 4)))
  (setf (svref board 5) 0))

(defun last-position ()
  (do ((i 1 (the fixnum (+ i 1))))
      ((= i 16) 0)
    (declare (fixnum i))
    (if (eq 1 (svref board i))
	(return i))))

(defun try (i depth)
  (declare (fixnum i depth))
  (cond ((= depth 14) 
	 (let ((lp (last-position)))
	   (unless (member lp final :test #'eq)
	     (push lp final)))
	 (push (cdr (simple-vector-to-list seq))
	       answer) t) 		; this is a hack to replace LISTARRAY
	((and (eq 1 (svref board (svref a i)))
	      (eq 1 (svref board (svref b i)))
	      (eq 0 (svref board (svref c i))))
	 (setf (svref board (svref a i)) 0)
	 (setf (svref board (svref b i)) 0)
	 (setf (svref board (svref c i)) 1)
	 (setf (svref seq depth) i)
	 (do ((j 0 (the fixnum (+ j 1)))
	      (depth (the fixnum (+ depth 1))))
	     ((or (= j 36)
		  (try j depth)) ())
	   (declare (fixnum j depth)))
	 (setf (svref board (svref a i)) 1) 
	 (setf (svref board (svref b i)) 1)
	 (setf (svref board (svref c i)) 0) ())))

(defun simple-vector-to-list (seq)
  (do ((i (- (length seq) 1) (1- i))
       (res))
      ((< i 0)
       res)
    (declare (fixnum i))
    (push (svref seq i) res)))
		
(defun gogogo (i)
  (let ((answer ())
	(final ()))
    (try i 1)))

(defun testtriang ()
  (triang-setup)
  (print (time (gogogo 22))))
