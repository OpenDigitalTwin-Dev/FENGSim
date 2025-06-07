;; Copyright (C) 2024 Camm Maguire
;; (in-package 'lisp)
;; (export '(macro-function))
(in-package :si)

(defun macro-function (x &optional env &aux l)
  (declare (optimize (safety 2)))
  (check-type x symbol)
  (check-type env proper-list)
  (cond ((setq l (cdr (assoc x (cadr env)))) (when (eq (car l) 'macro) (cadr l)))
	((unless (zerop (c-symbol-mflag x)) (c-symbol-gfdef x)))))

(defun special-operator-p (x)
  (declare (optimize (safety 1)))
  (check-type x symbol)
  (if (member x '(locally symbol-macrolet)) t (/= (address nil) (c-symbol-sfdef x))))

(defun find-symbol (s &optional (p *package*) &aux r)
  (declare (optimize (safety 1)))
  (check-type s string)
  (check-type p (or package string symbol character))
  (labels ((inb (h p) (package-internal p (mod h (c-package-internal_size p))))
	   (exb (h p) (package-external p (mod h (c-package-external_size p))))
	   (coerce-to-package 
	    (p)
	    (cond ((packagep p) p)
		  ((find-package p))
		  (t 
		   (cerror "Input new package" 'package-error
			   :package p 
			   :format-control "~a is not a package"
			   :format-arguments (list p)) 
		   (coerce-to-package (eval (read))))))
	   (cns (s b) (member-if (lambda (x) (declare (symbol x)) (string= x s)) b)))
	(let* ((p (coerce-to-package p))
	       (h (pack-hash s)))
	  (cond ((setq r (cns s (inb h p)))
		 (values (car r) :internal))
		((setq r (cns s (exb h p)))
		 (values (car r) :external))
		((dolist (p (c-package-uselist p))
		   (when (setq r (cns s (exb h p)))
		     (return r)))
		 (values (car r) :inherited))
		(t (values nil nil))))))


(defun symbol-value (s)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (if (boundp s) (c-symbol-dbind s)
    (error 'unbound-variable :name s)))

(defun boundp (s)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (not (eq (nani +objnull+) (c-symbol-dbind s))))


(defun symbol-name (s)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (c-symbol-name s))

(defun symbol-function (s)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (or (let ((x (c-symbol-sfdef s)))
	(when (nani x) (cons 'special x)))
      (let ((x (c-symbol-gfdef s)))
	(when (eql (address x) +objnull+)
	  (error 'undefined-function :name s))
	(if (zerop (c-symbol-mflag s)) x (cons 'macro x)))))

(defun remprop (s i)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (remf (symbol-plist s) i))

(defun makunbound (s)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (c-set-symbol-dbind s (nani +objnull+))
  s)

(defun set (s y)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (c-set-symbol-dbind s y))

#-pre-gcl
(defun get (s y &optional d)
  (declare (optimize (safety 1)))
  (check-type s symbol)
  (getf (symbol-plist s) y d))

#-pre-gcl(defun symbolp (x) (if x (typecase x (symbol t)) t))
#+pre-gcl(defun symbolp (x) (typecase x (list (not x)) (symbol t)))
(defun keywordp (x) (typecase x (keyword t)))

(setf (symbol-function 'symbol-plist)   (symbol-function 'c-symbol-plist))
(setf (symbol-function 'symbol-package) (symbol-function 'c-symbol-hpack))
