;; $Header$
;; $Locker$

#+excl
(eval-when (compile) (setq comp::register-use-threshold 6))

(declaim (ftype (function (fixnum fixnum fixnum) fixnum) tak))

(defun tak (x y z)
  (declare (fixnum x y z))
  (cond ((not (< y x)) z)
	(t
	 (tak
	   (tak (the fixnum (1- x)) y z)
	   (tak (the fixnum (1- y)) z x)
	   (tak (the fixnum (1- z)) x y)))))

(defun testtak ()
  (print (time
	   (progn (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)
		  (tak 18 12 6)))))

#+excl (eval-when (compile) (setq comp::register-use-threshold 3))
