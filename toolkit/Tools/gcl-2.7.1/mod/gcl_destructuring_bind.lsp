;; Copyright (C) 2024 Camm Maguire
(in-package :si)

(defmacro destructuring-bind (ll f &rest body)
  (compiler::blla ll nil f body))

;; ;;;; From CMULISP

;; ;;;; From defmacro.lisp

;; ;;;; Some variable definitions.

;; ;;; Variables for amassing the results of parsing a defmacro.  Declarations
;; ;;; in DEFMACRO are the reason this isn't as easy as it sounds.
;; ;;;

;; ;; (in-package 'lisp)

;; ;; (export '(destructuring-bind))

;; (in-package :si)

;; (defvar *arg-tests* ()
;;   "A list of tests that do argument counting at expansion time.")

;; (defvar *system-lets* nil)
;; ;(defvar *system-lets* ()
;; ;  "Let bindings that are done to make lambda-list parsing possible.")

;; (defvar *user-lets* ()
;;   "Let bindings that the user has explicitly supplied.")

;; (defvar *default-default* nil
;;   "Unsupplied optional and keyword arguments get this value defaultly.")

;; ;; Temps that we introduce and might not reference.
;; (defvar *ignorable-vars*)

;; ;;;; Stuff to parse DEFMACRO, MACROLET, DEFINE-SETF-METHOD, and DEFTYPE.

;; ;;; We save space in macro definitions by callig this function.
;; ;;;
;; (defun do-arg-count-error (error-kind name arg lambda-list minimum maximum)
;;   (error "Error in do-arg-count-error: ~S ~S ~S ~S ~S ~S~%"
;; 	 error-kind
;; 	 name
;; 	 arg
;; 	 lambda-list
;; 	 minimum
;; 	 maximum))

;; (defun push-let-binding (variable path systemp &optional condition
;; 				  (init-form *default-default*))
;;   (let ((let-form (if condition
;; 		      `(,variable (if ,condition ,path ,init-form))
;; 		      `(,variable ,path))))
;;     (if systemp
;; 	(push let-form *system-lets*)
;; 	(push let-form *user-lets*))))

;; (defun defmacro-error (problem kind name)
;; ; FIXME check this
;;   (declare (ignore kind))
;;   (error 'type-error :datum problem :expected-type name))

;; (defun push-sub-list-binding (variable path object name error-kind error-fun)
;;   (let ((var (gensym "TEMP-")))
;;     (push `(,variable
;; 	    (let ((,var ,path))
;; 	      (if (listp ,var)
;; 		  ,var
;; 		(,error-fun "destructuring-bind-error: kind ~s name ~s object ~s ll ~s~%"
;; 		       ',error-kind ',name ',var ',object))))
;; 	  *system-lets*)))

;; (defun push-optional-binding (value-var init-form supplied-var condition path
;; 					name error-kind error-fun)
;;   (unless supplied-var
;;     (setf supplied-var (gensym "SUPLIEDP-")))
;;   (push-let-binding supplied-var condition t)
;;   (cond ((consp value-var)
;; 	 (let ((whole-thing (gensym "OPTIONAL-SUBLIST-")))
;; 	   (push-sub-list-binding whole-thing
;; 				  `(if ,supplied-var ,path ,init-form)
;; 				  value-var name error-kind error-fun)
;; 	   (parse-defmacro-lambda-list value-var whole-thing name
;; 				       error-kind error-fun)))
;; 	((symbolp value-var)
;; 	 (push-let-binding value-var path nil supplied-var init-form))
;; 	(t
;; 	 (error "Illegal optional variable name: ~S" value-var))))

;; (defun make-keyword (symbol)
;;   "Takes a non-keyword symbol, symbol, and returns the corresponding keyword."
;;   (intern (symbol-name symbol) (find-package "KEYWORD")))

;; ;;;; From macros.lisp

;; ;;; Parse-Body  --  Public
;; ;;;
;; ;;;    Parse out declarations and doc strings, *not* expanding macros.
;; ;;; Eventually the environment arg should be flushed, since macros can't expand
;; ;;; into declarations anymore.
;; ;;;
;; (defun parse-body (body environment &optional (doc-string-allowed t))
;;   "This function is to parse the declarations and doc-string out of the body of
;;   a defun-like form.  Body is the list of stuff which is to be parsed.
;;   Environment is ignored.  If Doc-String-Allowed is true, then a doc string
;;   will be parsed out of the body and returned.  If it is false then a string
;;   will terminate the search for declarations.  Three values are returned: the
;;   tail of Body after the declarations and doc strings, a list of declare forms,
;;   and the doc-string, or NIL if none."
;;   (declare (ignore environment))
;;   (let ((decls ())
;; 	(doc nil))
;;     (do ((tail body (cdr tail)))
;; 	((endp tail)
;; 	 (values tail (nreverse decls) doc))
;;       (let ((form (car tail)))
;; 	(cond ((and (stringp form) (cdr tail))
;; 	       (if doc-string-allowed
;; 		   (setq doc form
;; 			 ;; Only one doc string is allowed.
;; 			 doc-string-allowed nil)
;; 		   (return (values tail (nreverse decls) doc))))
;; 	      ((not (and (consp form) (symbolp (car form))))
;; 	       (return (values tail (nreverse decls) doc)))
;; 	      ((eq (car form) 'declare)
;; 	       (push form decls))
;; 	      (t
;; 	       (return (values tail (nreverse decls) doc))))))))

;; (defun lookup-keyword (keyword key-list)
;;   (do ((remaining key-list (cddr remaining)))
;;       ((endp remaining))
;;     (when (eq keyword (car remaining))
;;       (return (cadr remaining)))))

;; (defun parse-defmacro-lambda-list
;;        (lambda-list arg-list-name name error-kind error-fun
;; 		    &optional top-level env-illegal env-arg-name wholep)
;;   (let ((path (if top-level `(cdr ,arg-list-name) arg-list-name))
;; 	(now-processing :required)
;; 	(maximum 0)
;; 	(minimum 0)
;; 	(keys ())
;; 	rest-name restp allow-other-keys-p env-arg-used)
;;     ;; This really strange way to test for '&whole is neccessary because member
;;     ;; does not have to work on dotted lists, and dotted lists are legal
;;     ;; in lambda-lists.
;;     (when (and (do ((list lambda-list (cdr list)))
;; 		   ((atom list) nil)
;; 		 (when (eq (car list) '&whole) (return t)))
;; 	       (not (eq (car lambda-list) '&whole)))
;;       (error "&Whole must appear first in ~S lambda-list." error-kind))
;;     (do ((rest-of-args lambda-list (cdr rest-of-args)))
;; 	((atom rest-of-args)
;; 	 (cond ((null rest-of-args) nil)
;; 	       ;; Varlist is dotted, treat as &rest arg and exit.
;; 	       (t (push-let-binding rest-of-args path nil)
;; 		  (setf restp t))))
;;       (let ((var (car rest-of-args)))
;; 	(cond ((eq var '&environment)
;; 	       (cond (env-illegal
;; 		      (error "&Environment not valid with ~S." error-kind))
;; 		     ((not top-level)
;; 		      (error "&Environment only valid at top level of ~
;; 		      lambda-list.")))
;; 	       (cond ((and (cdr rest-of-args) (symbolp (cadr rest-of-args)))
;; 		      (setf rest-of-args (cdr rest-of-args))
;; 		      (push-let-binding (car rest-of-args) env-arg-name nil)
;; 		      (setf env-arg-used t))
;; 		     (t
;; 		      (defmacro-error "&ENVIRONMENT" error-kind name))))
;; 	      ((eq var '&body)
;; 	       (cond ((and (cdr rest-of-args) (symbolp (cadr rest-of-args)))
;; 		      (setf rest-of-args (cdr rest-of-args))
;; 		      (setf restp t)
;; 		      (push-let-binding (car rest-of-args) path nil))
;; 		     ;;
;; 		     ;; This branch implements an incompatible extension to
;; 		     ;; Common Lisp.  In place of a symbol following &body,
;; 		     ;; there may be a list of up to three elements which will
;; 		     ;; be bound to the body, declarations, and doc-string of
;; 		     ;; the body.
;; 		     ((and (cdr rest-of-args)
;; 			   (consp (cadr rest-of-args))
;; 			   (symbolp (caadr rest-of-args)))
;; 		      (setf rest-of-args (cdr rest-of-args))
;; 		      (setf restp t)
;; 		      (let ((body-name (caar rest-of-args))
;; 			    (declarations-name (cadar rest-of-args))
;; 			    (doc-string-name (caddar rest-of-args))
;; 			    (parse-body-values (gensym)))
;; 			(push-let-binding
;; 			 parse-body-values
;; 			 `(multiple-value-list
;; 			   (parse-body ,path ,env-arg-name
;; 				       ,(not (null doc-string-name))))
;; 			 t)
;; 			(setf env-arg-used t)
;; 			(when body-name
;; 			  (push-let-binding body-name
;; 					    `(car ,parse-body-values) nil))
;; 			(when declarations-name
;; 			  (push-let-binding declarations-name
;; 					    `(cadr ,parse-body-values) nil))
;; 			(when doc-string-name
;; 			  (push-let-binding doc-string-name
;; 					    `(caddr ,parse-body-values) nil))))
;; 		     (t
;; 		      (defmacro-error (symbol-name var) error-kind name))))
;; 	      ((eq var '&rest)
;; 	       (setf restp t)
;; 	       (setf now-processing :rest))	      
;; 	      ((eq var '&whole)
;; 	       (setf now-processing :whole))
;; 	      ((eq var '&optional)
;; 	       (setf now-processing :optionals))
;; 	      ((eq var '&key)
;; 	       (setf now-processing :keywords)
;; 	       (setf rest-name (gensym "KEYWORDS-"))
;; 	       (push rest-name *ignorable-vars*)
;; 	       (setf restp t)
;; 	       (push-let-binding rest-name path t))
;; 	      ((eq var '&allow-other-keys)
;; 	       (setf allow-other-keys-p t))
;; 	      ((eq var '&aux)
;; 	       (setf now-processing :auxs))
;; 	      ((listp var)
;; 	       (case now-processing
;; 		 (:required
;; 		  (let ((sub-list-name (gensym "SUBLIST-")))
;; 		    (push-sub-list-binding sub-list-name `(car ,path) var
;; 					   name error-kind error-fun)
;; 		    (parse-defmacro-lambda-list var sub-list-name name
;; 						error-kind error-fun))
;; 		  (setf path `(cdr ,path))
;; 		  (incf minimum)
;; 		  (incf maximum))
;; 		 (:rest
;; 		  (let ((sub-list-name (gensym "SUBLIST-")))
;; 		    (push-sub-list-binding sub-list-name path var
;; 					   name error-kind error-fun)
;; 		    (parse-defmacro-lambda-list var sub-list-name name
;; 						error-kind error-fun)))
;; 		 (:whole
;; 		  (let ((sub-list-name (gensym "SUBLIST-")))
;; 		    (push-sub-list-binding sub-list-name arg-list-name var
;; 					   name error-kind error-fun)
;; 		    (parse-defmacro-lambda-list var sub-list-name name
;; 						error-kind error-fun nil nil nil t)
;; 		    (setf now-processing :required)))
;; 		 (:optionals
;; 		  (when (> (length var) 3)
;; 		    (cerror "Ignore extra noise."
;; 			    "More than variable, initform, and suppliedp ~
;; 			    in &optional binding - ~S"
;; 			    var))
;; 		  (push-optional-binding (car var) (cadr var) (caddr var)
;; 					 `(not (null ,path)) `(car ,path)
;; 					 name error-kind error-fun)
;; 		  (setf path `(cdr ,path))
;; 		  (incf maximum))
;; 		 (:keywords
;; 		  (let* ((keyword-given (consp (car var)))
;; 			 (variable (if keyword-given
;; 				       (cadar var)
;; 				       (car var)))
;; 			 (keyword (if keyword-given
;; 				      (caar var)
;; 				      (make-keyword variable)))
;; 			 (supplied-p (caddr var)))
;; 		    (push-optional-binding variable (cadr var) supplied-p
;; 					   `(keyword-supplied-p ',keyword
;; 								,rest-name)
;; 					   `(lookup-keyword ',keyword
;; 							    ,rest-name)
;; 					   name error-kind error-fun)
;; 		    (push keyword keys)))
;; 		 (:auxs (push-let-binding (car var) (cadr var) nil))))
;; 	      ((symbolp var)
;; 	       (case now-processing
;; 		 (:required
;; 		  (incf minimum)
;; 		  (incf maximum)
;; 		  (push-let-binding var `(car ,path) nil)
;; 		  (setf path `(cdr ,path)))
;; 		 (:rest
;; 		  (push-let-binding var path nil))
;; 		 (:whole
;; 		  (push-let-binding var arg-list-name nil)
;; 		  (setf now-processing :required))
;; 		 (:optionals
;; 		  (incf maximum)
;; 		  (push-let-binding var `(car ,path) nil `(not (null ,path)))
;; 		  (setf path `(cdr ,path)))
;; 		 (:keywords
;; 		  (let ((key (make-keyword var)))
;; 		    (push-let-binding var `(lookup-keyword ,key ,rest-name)
;; 				      nil)
;; 		    (push key keys)))
;; 		 (:auxs
;; 		  (push-let-binding var nil nil))))
;; 	      (t
;; 	       (error "Non-symbol in lambda-list - ~S." var)))))
;;     ;; Generate code to check the number of arguments, unless dotted
;;     ;; in which case length will not work.
;;     (unless (or restp wholep)
;;        (push `(unless (<= ,minimum
;; 			  (length (the list ,(if top-level
;; 						 `(cdr ,arg-list-name)
;; 					       arg-list-name)))
;; 			  ,@(unless restp
;; 				    (list maximum)))
;; 		      ,(let ((arg (if top-level
;; 				      `(cdr ,arg-list-name)
;; 				    arg-list-name)))
;; 			 (if (eq error-fun 'error)
;; 			     `(do-arg-count-error ',error-kind ',name ,arg
;; 						  ',lambda-list ,minimum
;; 						  ,(unless restp maximum))
;; 			   `(,error-fun 'defmacro-ll-arg-count-error
;; 				 :kind ',error-kind
;; 				 ,@(when name `(:name ',name))
;; 				 :argument ,arg
;; 				 :lambda-list ',lambda-list
;; 				 :minimum ,minimum
;; 				 ,@(unless restp `(:maximum ,maximum))))))
;; 	     *arg-tests*))
;;     (if keys
;; 	(let ((problem (gensym "KEY-PROBLEM-"))
;; 	      (info (gensym "INFO-")))
;; 	  (push `(multiple-value-bind
;; 		     (,problem ,info)
;; 		     (verify-keywords ,rest-name ',keys ',allow-other-keys-p)
;; 		   (when ,problem
;; 		     (,error-fun
;; 		      'defmacro-ll-broken-key-list-error
;; 		      :kind ',error-kind
;; 		      ,@(when name `(:name ',name))
;; 		      :problem ,problem
;; 		      :info ,info)))
;; 		*arg-tests*)))
;;     (values env-arg-used minimum (if (null restp) maximum nil))))

;; ;;; PARSE-DEFMACRO returns, as multiple-values, a body, possibly a declare
;; ;;; form to put where this code is inserted, and the documentation for the
;; ;;; parsed body.
;; ;;;
;; (defun parse-defmacro (lambda-list arg-list-name code name error-kind
;; 				   &key (annonymousp nil)
;; 				   (doc-string-allowed t)
;; 				   ((:environment env-arg-name))
;; 				   ((:default-default *default-default*))
;; 				   (error-fun 'error))
;;   "Returns as multiple-values a parsed body, any local-declarations that
;;    should be made where this body is inserted, and a doc-string if there is
;;    one."
;;   (multiple-value-bind (body declarations documentation)
;; 		       (parse-body code nil doc-string-allowed)
;;     (let* ((*arg-tests* ())
;; 	   (*user-lets* ())
;; 	   (*system-lets* ())
;; 	   (*ignorable-vars* ()))
;;       (multiple-value-bind
;; 	  (env-arg-used minimum maximum)
;; 	  (parse-defmacro-lambda-list lambda-list arg-list-name name
;; 				      error-kind error-fun (not annonymousp)
;; 				      nil env-arg-name)
;; 	(values
;; 	 `(let* ,(nreverse *system-lets*)
;; 	   ,@(when *ignorable-vars*
;; 	       `((declare (ignorable ,@*ignorable-vars*))))
;; 	    ,@*arg-tests*
;; 	    (let* ,(nreverse *user-lets*)
;; 	      ,@declarations
;; 	      ,@body))
;; 	 `(,@(when (and env-arg-name (not env-arg-used))
;; 	       `((declare (ignore ,env-arg-name)))))
;; 	 documentation
;; 	 minimum
;; 	 maximum)))))





;; (defun verify-keywords (key-list valid-keys allow-other-keys)
;;   (do ((already-processed nil)
;;        (unknown-keyword nil)
;;        (remaining key-list (cddr remaining)))
;;       ((null remaining)
;;        (if (and unknown-keyword
;; 		(not allow-other-keys)
;; 		(not (lookup-keyword :allow-other-keys key-list)))
;; 	   (values :unknown-keyword (list unknown-keyword valid-keys))
;; 	   (values nil nil)))
;;     (cond ((not (and (consp remaining) (listp (cdr remaining))))
;; 	   (return (values :dotted-list key-list)))
;; 	  ((null (cdr remaining))
;; 	   (return (values :odd-length key-list)))
;; 	  #+nil ;; Not ANSI compliant to disallow duplicate keywords.
;; 	  ((member (car remaining) already-processed)
;; 	   (return (values :duplicate (car remaining))))
;; 	  ((or (eq (car remaining) :allow-other-keys)
;; 	       (member (car remaining) valid-keys))
;; 	   (push (car remaining) already-processed))
;; 	  (t
;; 	   (setf unknown-keyword (car remaining))))))


;; ;;;
;; (defun keyword-supplied-p (keyword key-list)
;;   (do ((remaining key-list (cddr remaining)))
;;       ((endp remaining))
;;     (when (eq keyword (car remaining))
;;       (return t))))












;; ;;;; Destructuring-bind

;; (defmacro destructuring-bind (lambda-list arg-list &rest body)
;;   "Bind the variables in LAMBDA-LIST to the contents of ARG-LIST."
;;   (declare (optimize (safety 1)))
;;   (let* ((arg-list-name (gensym "ARG-LIST-")))
;;     (multiple-value-bind
;; 	(body local-decls)
;; 	(parse-defmacro lambda-list arg-list-name body nil 'destructuring-bind
;; 			:annonymousp t :doc-string-allowed nil)
;;       `(let ((,arg-list-name ,arg-list))
;; 	 ,@local-decls
;; 	 ,body))))

