;; Copyright (C) 2024 Camm Maguire
;; -*-Lisp-*-
(in-package :si)

(macrolet 
 ((make-conditionp (condition &aux (n (intern (concatenate 'string (string condition) "P"))))
    `(defun ,n (x &aux (z (load-time-value nil)))
       (setq z (or z (let ((x (si-find-class ',condition nil))) (unless (symbolp x) x))))
       (when z (typep x z))))
  (make-condition-classp (class &aux (n (intern (concatenate 'string (string class) "-CLASS-P"))))
    `(defun ,n (x &aux (s (load-time-value nil)) (z (load-time-value nil)))
       (setq s (or s (let ((x (si-find-class 'standard-class nil))) (unless (symbolp x) x))))
       (setq z (or z (let ((x (si-find-class ',class nil))) (unless (symbolp x) x))))
       (when (and s z)
	 (let ((x (if (symbolp x) (si-find-class x nil) x)))
	   (when (and x (typep x s))
	     (member z (si-cpl-or-nil x))))))))
  (make-conditionp condition)
  (make-conditionp warning)
  (make-condition-classp condition)
  (make-condition-classp simple-condition))
 
(defun si-make-condition (tp &rest args &aux (z (load-time-value nil)))
  (setq z (or z (when (fboundp 'make-condition) (symbol-function 'make-condition))))
  (when z (values (apply z tp args))))

(defun coerce-to-condition (datum arguments default-type function-name)
  (cond ((conditionp datum)
	 (if arguments
	     (cerror "ignore the additional arguments."
		     'simple-type-error
		     :datum arguments
		     :expected-type 'null
		     :format-control "you may not supply additional arguments ~
				     when giving ~s to ~s."
		     :format-arguments (list datum function-name)))
	 datum)
        ((condition-class-p datum)
	 (apply #'si-make-condition datum arguments))
        ((when (condition-class-p default-type) (or (stringp datum) (functionp datum)))
	 (si-make-condition default-type :format-control datum :format-arguments arguments))
	((coerce-to-string datum arguments))))

(defvar *handler-clusters* nil)
(defvar *break-on-signals* nil)

(defmacro handler-bind (bindings &body forms)
  (declare (optimize (safety 2)))
  `(let ((*handler-clusters*
	  (cons (list ,@(mapcar (lambda (x) `(cons ',(car x) ,(cadr x))) bindings))
		*handler-clusters*)))
     ,@forms))

(defmacro handler-case (form &rest cases)
  (declare (optimize (safety 2)))
  (let ((no-error-clause (assoc ':no-error cases)))
    (if no-error-clause
	(let ((normal-return (gensym)) (error-return  (gensym)))
	  `(block ,error-return
	     (multiple-value-call (lambda ,@(cdr no-error-clause))
	       (block ,normal-return
		 (return-from ,error-return
		   (handler-case (return-from ,normal-return ,form)
		     ,@(remove no-error-clause cases)))))))
	(let ((block (gensym))(var (gensym))
	      (tcases (mapcar (lambda (x) (cons (gensym) x)) cases)))
	  `(block ,block
	     (let (,var)
	       (declare (ignorable ,var))
	       (tagbody
		 (handler-bind ,(mapcar (lambda (x &aux (tag (pop x))(type (pop x))(ll (car x)))
					  (list type `(lambda (x)
							,(if ll `(setq ,var x) `(declare (ignore x)))
							(go ,tag))))
					tcases)
			       (return-from ,block ,form))
		  ,@(mapcan (lambda (x &aux (tag (pop x))(type (pop x))(ll (pop x))(body x))
			      (declare (ignore type))
			      (list tag `(return-from ,block (let ,(when ll `((,(car ll) ,var))) ,@body))))
			   tcases))))))))

(defmacro ignore-errors (&rest forms)
  `(handler-case (progn ,@forms)
     (error (condition) (values nil condition))))

(defun signal (datum &rest arguments)
  (declare (optimize (safety 1)))
  (let ((*handler-clusters* *handler-clusters*)
	(condition (coerce-to-condition datum arguments 'simple-condition 'signal)))
    (if (typep condition *break-on-signals*)
	(break "~a~%break entered because of *break-on-signals*." condition))
    (unless (stringp condition)
      (do nil ((not *handler-clusters*))
	(dolist (handler (pop *handler-clusters*))
	  (when (typep condition (car handler));FIXME, might string-match condition w handler in non-ansi here.
	    (funcall (cdr handler) condition)))))
    nil))

(defvar *debugger-hook* nil)
(defvar *debug-level* 1)
(defvar *debug-restarts* nil)
(defvar *debug-abort* nil)
(defvar *debug-continue* nil)
(defvar *abort-restarts* nil)

(defun break-level-invoke-restart (n)
  (cond ((when (plusp n) (< n (+ (length *debug-restarts*) 1)))
	 (invoke-restart-interactively (nth (1- n) *debug-restarts*)))
	((format t "~&no such restart."))))

(defun fun-name (fun) (sixth (c-function-plist fun)))

(defun find-ihs (s i &optional (j i))
  (cond ((eq (ihs-fname i) s) i)
	((and (> i 0) (find-ihs s (1- i) j)))
	(j)))

(defmacro without-interrupts (&rest forms)
  `(let (*quit-tag* *quit-tags* *restarts*)
     ,@forms))

(defun process-args (args &optional fc fa others);FIXME do this without consing, could be oom
  (cond ((not args) (nconc (nreverse others) (when fc (list (apply 'format nil fc fa)))))
	((eq (car args) :format-control)
	 (process-args (cddr args) (cadr args) fa others))
	((eq (car args) :format-arguments)
	 (process-args (cddr args) fc (cadr args) others))
	((process-args (cdr args) fc fa (cons (car args) others)))))

(defun coerce-to-string (datum args) 
  (cond ((stringp datum)
	 (if args 
	     (let ((*print-pretty* nil)(*print-readably* nil)
		   (*print-level* *debug-print-level*)
		   (*print-length* *debug-print-level*)
		   (*print-case* :upcase))
	       (apply 'format nil datum args))
	   datum))
	((symbolp datum)
	 (let* ((args (process-args args))
		(fn (member :function-name args))
		(args (if fn (nconc (ldiff args fn) (cddr fn)) args)))
	   (string-concatenate
	    (or (cadr fn) "")
	    (substitute
	     #\^ #\~
	     (coerce-to-string
	      (apply 'string-concatenate  datum (if args ": " "") (make-list (length args) :initial-element " ~a"))
	      args)))))
	("unknown error")))

(defun put-control-string (strm strng)
  (when (tty-stream-p strm)
    (let ((pos (c-stream-int strm)))
      (format strm strng)
      (c-set-stream-int strm pos))))

(defvar *error-color* "92")

(defun error-format (control &rest arguments)
  (put-control-string *error-output* (concatenate 'string (string 27) "[1;" *error-color* "m"))
  (apply 'format *error-output* control arguments)
  (put-control-string *error-output* (concatenate 'string (string 27) "[0m")))

(defun warn (datum &rest arguments);FIXME? &aux (*sig-fn-name* (or *sig-fn-name* (get-sig-fn-name))))
  (declare (optimize (safety 2)))
  (let ((c (process-error datum arguments 'simple-warning)))
    (check-type c (or string (satisfies warningp)) "a warning condition")
    (when *break-on-warnings*
      (break "~A~%break entered because of *break-on-warnings*." c))
    (restart-case
     (signal c)
     (muffle-warning nil :report "Skip warning."  (return-from warn nil)))
    (error-format "~&~a~%" c)
    (force-output *error-output*)
    nil))
(putprop 'cerror t 'compiler::cmp-notinline)

(dolist (l '(break cerror error universal-error-handler ihs-top get-sig-fn-name next-stack-frame check-type-symbol))
  (setf (get l 'dbl-invisible) t))

(defvar *sig-fn-name* nil)

(defun get-sig-fn-name (&aux (p (ihs-top))(p (next-stack-frame p)))
  (when p (ihs-fname p)))

(defun process-error (datum args &optional (default-type 'simple-error))
  (let ((internal (cond ((simple-condition-class-p datum)
			 (find-symbol (concatenate 'string "INTERNAL-" (string datum)) :conditions))
			((condition-class-p datum)
			 (find-symbol (concatenate 'string "INTERNAL-SIMPLE-" (string datum)) :conditions)))))
    (coerce-to-condition (or internal datum) (if internal (append args (list :function-name *sig-fn-name*)) args) default-type 'process-error)))

(defun universal-error-handler (n cp fn cs es &rest args &aux (*sig-fn-name* fn))
  (declare (ignore es))
  (if cp (apply #'cerror cs n args) (apply #'error n args)))

(defun cerror (continue-string datum &rest args &aux (*sig-fn-name* (or *sig-fn-name* (get-sig-fn-name))))
  (values 
   (with-simple-restart 
    (continue continue-string args)
    (apply #'error datum args))))
(putprop 'cerror t 'compiler::cmp-notinline)


(defun error (datum &rest args &aux (*sig-fn-name* (or *sig-fn-name* (get-sig-fn-name))))
  (let ((c (process-error datum args))(q (or *quit-tag* +top-level-quit-tag+)))
    (signal c)
    (invoke-debugger c)
    (throw q q)))
(putprop 'error t 'compiler::cmp-notinline)
  

(defun invoke-debugger (condition)

  (when *debugger-hook*
	(let ((hook *debugger-hook*) *debugger-hook*)
	  (funcall hook condition hook)))

  (maybe-clear-input)
  
  (let ((correctable (find-restart 'continue))
	*print-pretty*
	(*print-level* *debug-print-level*)
	(*print-length* *debug-print-level*)
	(*print-case* :upcase))
    (terpri *error-output*)
    (error-format (if (and correctable *break-enable*) "Correctable error:" "Error:"))
    (let ((*indent-formatted-output* t))
      (when (stringp condition) (error-format condition)))
    (terpri *error-output*)
    (if (> (length *link-array*) 0)
	(error-format "Fast links are on: do (si::use-fast-links nil) for debugging~%"))
    (error-format "Signalled by ~:@(~S~).~%" (or *sig-fn-name* "an anonymous function"))
    (when (and correctable *break-enable*)
      (error-format "~&If continued: "))
    (force-output *error-output*)
    (when (and correctable *break-enable*)
      (funcall (restart-report-function correctable) *debug-io*))
    (when *break-enable* (break-level condition))))


(defun dbl-eval (- &aux (break-command t))
  (let ((val-list (multiple-value-list
		   (cond 
		    ((keywordp -) (break-call - nil 'break-command))
		    ((and (consp -) (keywordp (car -))) (break-call (car -) (cdr -) 'break-command))
		    ((integerp -) (break-level-invoke-restart -))     
		    (t (setq break-command nil) (evalhook - nil nil *break-env*))))))
    (cons break-command val-list)))

(defun dbl-rpl-loop (p-e-p)

  (setq +++ ++ ++ + + -)

  (if *no-prompt*
      (setq *no-prompt* nil)
    (format *debug-io* "~&~a~a>~{~*>~}"
	    (if p-e-p "" "dbl:")
	    (if (eq *package* (find-package 'user)) "" (package-name *package*))
	    *break-level*))

  (setq - (dbl-read *debug-io* nil *top-eof*))
  (when (eq - *top-eof*) (bye -1))
  (let* ((ev (dbl-eval -))
	 (break-command (car ev))
	 (values (cdr ev)))
    (unless (and break-command (eq (car values) :resume))
      (setq /// // // / / values *** ** ** * * (car /))
      (fresh-line *debug-io*)
      (dolist (val /)
	(prin1 val *debug-io*)
	(terpri *debug-io*))
      (dbl-rpl-loop p-e-p))))

(defun do-break-level (at env p-e-p debug-level); break-level

  (unless
      (with-simple-restart
       (abort "Return to debug level ~D." debug-level)

       (catch-fatal 1)
       (setq *interrupt-enable* t)
       (cond (p-e-p
	      (format *debug-io* "~&~A~2%" at)
	      (set-current)
	      (setq *no-prompt* nil)
	      (show-restarts))
	     ((set-back at env)))

       (not (catch 'step-continue (dbl-rpl-loop p-e-p))))

    (terpri *debug-io*)
    (break-current)
    (do-break-level at env p-e-p debug-level)))


(defun break-level (at &optional env)
  (let* ((p-e-p (unless (listp at) t))
         (+ +) (++ ++) (+++ +++)
         (- -)
         (* *) (** **) (*** ***)
         (/ /) (// //) (/// ///)
	 (debug-level *debug-level*)
	 (*quit-tags* (cons (cons *break-level* *quit-tag*) *quit-tags*))
	 *quit-tag*
	 (*break-level* (if p-e-p (cons t *break-level*) *break-level*))
	 (*ihs-base* (1+ *ihs-top*))
	 (*ihs-top* (ihs-top))
	 (*frs-base* (or (sch-frs-base *frs-top* *ihs-base*) (1+ (frs-top))))
	 (*frs-top*  (frs-top))
	 (*current-ihs* *ihs-top*)
	 (*debug-level* (1+ *debug-level*))
	 (*debug-restarts* (compute-restarts))
	 (*debug-abort* (find-restart 'abort))
	 (*debug-continue* (find-restart 'continue))
	 (*abort-restarts* (remove-if-not (lambda (x) (eq 'abort (restart-name x))) *debug-restarts*))
	 (*readtable* (or *break-readtable* *readtable*))
	 *break-env* *read-suppress*)
    
      (do-break-level at env p-e-p debug-level)))

(putprop 'break-level t 'compiler::cmp-notinline)

(defun break (&optional format-string &rest args &aux message (*sig-fn-name* (or *sig-fn-name* (get-sig-fn-name))))

  (let ((*print-pretty* nil)
	(*print-level* 4)
	(*print-length* 4)
	(*print-case* :upcase))
    (terpri *error-output*)
    (cond (format-string
	   (error-format "~&Break: ")
	   (let ((*indent-formatted-output* t))
	     (apply 'error-format format-string args))
	   (terpri *error-output*)
	   (setq message (apply 'format nil format-string args)))
	  (t (error-format "~&Break.~%")
	     (setq message "")))
    (force-output *error-output*))
  (with-simple-restart 
   (continue "Return from break.")
   (break-level message))
  nil)
(putprop 'break t 'compiler::cmp-notinline)
