;; Copyright (C) 2024 Camm Maguire
;;; -*- Mode: Lisp; Syntax: Common-Lisp; Package: "CONDITIONS"; Base: 10 -*-

(in-package :si)

(defvar *restarts* nil)
(defvar *restart-condition* nil)

(defmacro restart-bind (bindings &body forms)
  (declare (optimize (safety 2)))
  `(let ((*restarts* 
	  (list* ,@(mapcar (lambda (x) `(cons (make-restart :name ',(pop x) :function ,(pop x) ,@x) *restart-condition*)) bindings)
		 *restarts*)))
	 ,@forms))


(defmacro with-condition-restarts (condition-form restarts-form &body body)
  (declare (optimize (safety 1)))
  (let ((n-cond (gensym)))
    `(let* ((,n-cond ,condition-form)
	    (*restarts* (nconc (mapcar (lambda (x) (cons x ,n-cond)) ,restarts-form) *restarts*)))
       ,@body)))

(defun condition-pass (condition restart &aux b (f (restart-test-function restart)))
  (when (if f (funcall f condition) t)
    (mapc (lambda (x) 
	    (when (eq (pop x) restart)
	      (if (if condition (eq x condition) t)
		  (return-from condition-pass t)
		(setq b (or b x))))) *restarts*)
    (not b)))

(defvar *kcl-top-restarts* nil)

(defun make-kcl-top-restart (quit-tag)
  (make-restart :name 'gcl-top-restart
		:function (lambda () (throw (car (list quit-tag)) quit-tag))
		:report-function 
		(lambda (stream) 
		    (let ((b-l (if (eq quit-tag si::*quit-tag*)
				   si::*break-level*
				   (car (or (find quit-tag si::*quit-tags*
						  :key #'cdr)
					    '(:not-found))))))
		      (cond ((eq b-l :not-found)
			     (format stream "Return to ? level."))
			    ((null b-l)
			     (format stream "Return to top level."))
			    (t
			     (format stream "Return to break level ~D."
				     (length b-l))))))))

(defun find-kcl-top-restart (quit-tag)
  (cdr (or (assoc quit-tag *kcl-top-restarts*)
	   (car (push (cons quit-tag (make-kcl-top-restart quit-tag))
		      *kcl-top-restarts*)))))

(defun kcl-top-restarts ()
  (let* (;(old-tags (ldiff si::*quit-tags* (member nil si::*quit-tags* :key 'cdr)))
	 (old-tags si::*quit-tags*)
	 (old-tags (mapcan (lambda (e) (when (cdr e) (list (cdr e)))) old-tags))
	 (tags (if si::*quit-tag* (cons si::*quit-tag* old-tags) old-tags))
	 (restarts (mapcar 'find-kcl-top-restart tags)))
    (setq *kcl-top-restarts* (mapcar 'cons tags restarts))
    restarts))

(defun compute-restarts (&optional condition)
  (remove-if-not (lambda (x) (condition-pass condition x)) (nconc (mapcar 'car *restarts*) (kcl-top-restarts))))

(defun find-restart (name &optional condition &aux (sn (symbolp name)))
  (car (member name (compute-restarts condition) :key (lambda (x) (if sn (restart-name x) x)))))

(defun transform-keywords (&key report interactive test 
				&aux rr (report (if (stringp report) `(lambda (s) (write-string ,report s)) report)))
  (macrolet ((do-setf (x y)
		      `(when ,x 
			 (setf (getf rr ,y) (list 'function ,x)))))
	    (do-setf report :report-function)
	    (do-setf interactive :interactive-function)
	    (do-setf test :test-function)
	    rr))

(defun rewrite-restart-case-clause (r &aux (name (pop r))(ll (pop r)))
  (labels ((l (r) (if (member (car r) '(:report :interactive :test)) (l (cddr r)) r)))
	  (let ((rd (l r)))
	    (list* name (gensym) (apply 'transform-keywords (ldiff-nf r rd)) ll rd))))


(defun restart-case-expression-condition (expression env c &aux (e (macroexpand expression env))(n (when (listp e) (pop e))))
  (case n
	(cerror (let ((ca (pop e))) `((process-error ,(pop e) (list ,@e)) (,n ,ca ,c))))
	(error `((process-error ,(pop e) (list ,@e)) (,n ,c)))
	(warn `((process-error ,(pop e) (list ,@e) 'simple-warning) (,n ,c)))
	(signal `((coerce-to-condition ,(pop e) (list ,@e) 'simple-condition ',n) (,n ,c)))))


(defmacro restart-case (expression &body clauses &environment env)
  (declare (optimize (safety 2)))
  (let* ((block-tag (gensym))(args (gensym))(c (gensym))
	 (data (mapcar 'rewrite-restart-case-clause clauses))
	 (e (restart-case-expression-condition expression env c)))
    `(block 
      ,block-tag
      (let* (,args (,c ,(car e)) (*restart-condition* ,c))
	(tagbody
	 (restart-bind
	  ,(mapcar (lambda (x) `(,(pop x) (lambda (&rest r) (setq ,args r) (go ,(pop x))) ,@(pop x))) data)
	  (return-from ,block-tag ,(or (cadr e) expression)))
	 ,@(mapcan (lambda (x &aux (x (cdr x)))
		     `(,(pop x) (return-from ,block-tag (apply (lambda ,(progn (pop x)(pop x)) ,@x) ,args)))) data))))))


(defvar *unique-id-table* (make-hash-table))
(defvar *unique-id-count* -1)

(defun unique-id (obj)
  "generates a unique integer id for its argument."
  (or (gethash obj *unique-id-table*)
      (setf (gethash obj *unique-id-table*) (incf *unique-id-count*))))

(defun restart-print (restart stream depth)
  (declare (ignore depth))
  (if *print-escape*
      (format stream "#<~s.~d>" (type-of restart) (unique-id restart))
      (restart-report restart stream)))

(defstruct (restart (:print-function restart-print))
  name
  function
  report-function
  interactive-function
  (test-function (lambda (c) (declare (ignore c)) t)))

(defun restart-report (restart stream &aux (f (restart-report-function restart)))
  (if f (funcall f stream)
    (format stream "~s" (or (restart-name restart) restart))))

(defun invoke-restart (restart &rest values)
  (let ((real-restart (or (find-restart restart)
			  (error 'control-error :format-control "restart ~s is not active." :format-arguments (list restart)))))
       (apply (restart-function real-restart) values)))

(defun invoke-restart-interactively (restart)
  (let ((real-restart (or (find-restart restart)
			  (error "restart ~s is not active." restart))))
    (apply (restart-function real-restart)
	   (let ((interactive-function (restart-interactive-function real-restart)))
	     (when interactive-function
		 (funcall interactive-function))))))


(defmacro with-simple-restart ((restart-name format-control &rest format-arguments)
			       &body forms)
  (declare (optimize (safety 1)))
  `(restart-case (progn ,@forms)
     (,restart-name nil
        :report (lambda (stream) (format stream ,format-control ,@format-arguments))
	(values nil t))))

(defun abort (&optional condition)
  "Transfers control to a restart named abort, signalling a control-error if
   none exists."
  (invoke-restart (find-restart 'abort condition))
  (error 'abort-failure))


(defun muffle-warning (&optional condition)
  "Transfers control to a restart named muffle-warning, signalling a
   control-error if none exists."
  (invoke-restart (find-restart 'muffle-warning condition)))

(macrolet ((define-nil-returning-restart (name args doc)
	     (let ((restart (gensym)))
	       `(defun ,name (,@args &optional condition)
		  ,doc
		  (declare (optimize (safety 1)))
		  (let ((,restart (find-restart ',name condition))) (when ,restart (invoke-restart ,restart ,@args)))))))

  (define-nil-returning-restart continue nil
    "Transfer control to a restart named continue, returning nil if none exists.")
  
  (define-nil-returning-restart store-value (value)
    "Transfer control and value to a restart named store-value, returning nil if
   none exists.")
  
  (define-nil-returning-restart use-value (value)
    "Transfer control and value to a restart named use-value, returning nil if
   none exists."))

(defun show-restarts (&aux (i 0))
  (mapc (lambda (x)
	  (format *debug-io* "~& ~4d ~a ~a ~%"
		  (incf i)
		  (cond ((eq x *debug-abort*) "(abort)") ((eq x *debug-continue*) "(continue)") (""))
		  x)) *debug-restarts*)
  nil)
