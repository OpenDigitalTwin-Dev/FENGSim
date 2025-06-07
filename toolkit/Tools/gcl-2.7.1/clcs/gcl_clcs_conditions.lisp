;; Copyright (C) 2024 Camm Maguire
;;; -*- Mode: Lisp; Syntax: Common-Lisp; Package: "CONDITIONS"; Base: 10 -*-

;(in-package "CONDITIONS" :USE '(:cl #+(and clos (not pcl)) "CLOS" #+pcl "PCL"))

(in-package :conditions)

(defun slot-sym (base slot)
  (values (intern (concatenate 'string (string base) "-" (string slot)))))

(defun coerce-to-fn (x y)
  (cond ((stringp x) `(lambda (c s) (declare (ignore c)) (write-string ,x s)))
	((symbolp x) x)
	((atom x) nil)
	((eq (car x) 'lambda) x)
	((stringp (car x))
	 `(lambda (c s) 
	    (declare (ignorable c))
	    (call-next-method)
	    (format s ,(car x) ,@(mapcar (lambda (st) `(if (slot-boundp c ',st) (,(slot-sym y st) c) 'unbound)) (cdr x)))))))

(defun default-report (x)
  `(lambda (c s) (call-next-method) (format s "~s " ',x)))

(defmacro define-condition (name parent-list slot-specs &rest options)
  (unless (or parent-list (eq name 'condition))
	  (setq parent-list (list 'condition)))
  (let* ((report-function nil)
	 (default-initargs nil)
	 (documentation nil))
    (declare (ignore documentation))
    (do ((o options (cdr o)))
	((null o))
      (let ((option (car o)))
	(case (car option)
	  (:report (setq report-function (coerce-to-fn (cadr option) name)))
	  (:default-initargs (setq default-initargs option)) 
	  (:documentation (setq documentation (cadr option)))
	  (otherwise (cerror "ignore this define-condition option."
			     "invalid define-condition option: ~s" option)))))
    `(progn
       (eval-when (compile)
	 (setq pcl::*defclass-times* '(compile load eval)))
       ,(if default-initargs
       `(defclass ,name ,parent-list ,slot-specs ,default-initargs)
       `(defclass ,name ,parent-list ,slot-specs))
       (eval-when (compile load eval)
;	 (setf (get ',name 'documentation) ',documentation)
	 (setf (get ',name 'si::s-data) nil))
      ,@(when report-function
	   `((defmethod print-object ((x ,name) stream)
	       (if *print-escape*
		   (call-next-method)
		   (,report-function x stream)))))
      ',name)))

(eval-when (compile load eval)
  (define-condition condition nil nil))

(defmethod pcl::make-load-form ((object condition) &optional env)
  (declare (ignore env))
  (error "~@<default ~s method for ~s called.~@>" 'pcl::make-load-form object))

(mapc 'pcl::proclaim-incompatible-superclasses '((condition pcl::metaobject)))

(defun conditionp (object) (typep object 'condition))

(defun is-condition (x) (conditionp x))
(defun is-warning (x) (typep x 'warning))

(defmethod print-object ((x condition) stream)
  (let ((y (class-name (class-of x))))
    (if *print-escape* 
	(format stream "#<~s.~d>" y (unique-id x))
      (format stream "~a: " y))));(type-of x)

(defun make-condition (type &rest slot-initializations)
  ;; (when (and (consp type) (eq (car type) 'or))
  ;;   (return-from make-condition (apply 'make-condition (cadr type) slot-initializations)))
					;FIXME
  (unless (condition-class-p type)
    (error 'simple-type-error
	   :datum type
	   :expected-type '(satisfies condition-class-p)
	   :format-control "not a condition type: ~s"
	   :format-arguments (list type)))
  (apply 'make-instance type slot-initializations))

