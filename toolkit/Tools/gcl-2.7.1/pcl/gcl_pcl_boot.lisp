;;;-*-Mode: LISP; Package:(PCL LISP 1000); Base:10; Syntax:Common-lisp -*-
;;;
;;; *************************************************************************
;;; Copyright (c) 1985, 1986, 1987, 1988, 1989, 1990 Xerox Corporation.
;;; Copyright (c) 2024 Camm Maguire
;;; All rights reserved.
;;;
;;; Use and copying of this software and preparation of derivative works
;;; based upon this software are permitted.  Any distribution of this
;;; software or derivative works must comply with all applicable United
;;; States export control laws.
;;; 
;;; This software is made available AS IS, and Xerox Corporation makes no
;;; warranty about the software, its performance or its conformity to any
;;; specification.
;;; 
;;; Any person obtaining a copy of this software is requested to send their
;;; name and post office or electronic mail address to:
;;;   CommonLoops Coordinator
;;;   Xerox PARC
;;;   3333 Coyote Hill Rd.
;;;   Palo Alto, CA 94304
;;; (or send Arpanet mail to CommonLoops-Coordinator.pa@Xerox.arpa)
;;;
;;; Suggestions, comments and requests for improvements are also welcome.
;;; *************************************************************************
;;;

(in-package :pcl)

#|

The CommonLoops evaluator is meta-circular.  

Most of the code in PCL is methods on generic functions, including most of
the code that actually implements generic functions and method lookup.

So, we have a classic bootstrapping problem.   The solution to this is to
first get a cheap implementation of generic functions running, these are
called early generic functions.  These early generic functions and the
corresponding early methods and early method lookup are used to get enough
of the system running that it is possible to create real generic functions
and methods and implement real method lookup.  At that point (done in the
file FIXUP) the function fix-early-generic-functions is called to convert
all the early generic functions to real generic functions.

The cheap generic functions are built using the same funcallable-instance
objects real generic-functions are made out of.  This means that as PCL
is being bootstrapped, the cheap generic function objects which are being
created are the same objects which will later be real generic functions.
This is good because:
  - we don't cons garbage structure
  - we can keep pointers to the cheap generic function objects
    during booting because those pointers will still point to
    the right object after the generic functions are all fixed
    up



This file defines the defmethod macro and the mechanism used to expand it.
This includes the mechanism for processing the body of a method.  defmethod
basically expands into a call to load-defmethod, which basically calls
add-method to add the method to the generic-function.  These expansions can
be loaded either during bootstrapping or when PCL is fully up and running.

An important effect of this structure is it means we can compile files with
defmethod forms in them in a completely running PCL, but then load those files
back in during bootstrapping.  This makes development easier.  It also means
there is only one set of code for processing defmethod.  Bootstrapping works
by being sure to have load-method be careful to call only primitives which
work during bootstrapping.

|#

(proclaim '(notinline make-a-method
		      add-named-method		      
		      ensure-generic-function-using-class
		      add-method
		      remove-method
		      ))

(defvar *early-functions*
	'((make-a-method early-make-a-method
			 real-make-a-method)
	  (add-named-method early-add-named-method
			    real-add-named-method)
	  ))

;;;
;;; For each of the early functions, arrange to have it point to its early
;;; definition.  Do this in a way that makes sure that if we redefine one
;;; of the early definitions the redefinition will take effect.  This makes
;;; development easier.
;;;
;;; The function which generates the redirection closure is pulled out into
;;; a separate piece of code because of a bug in ExCL which causes this not
;;; to work if it is inlined.
;;;
(eval-when (load eval)

(defun redirect-early-function-internal (real early);(print (list  'baz real early))
  (setf (gdefinition real)
	(set-function-name
	 (lambda (&rest args)
	     (apply (the function (symbol-function early)) args))
	 real)))

(dolist (fns *early-functions*)
  (let ((name (car fns))
	(early-name (cadr fns)))
    (redirect-early-function-internal name early-name)))

)


;;;
;;; *generic-function-fixups* is used by fix-early-generic-functions to
;;; convert the few functions in the bootstrap which are supposed to be
;;; generic functions but can't be early on.
;;; 
(defvar *generic-function-fixups*
    '((add-method
	((generic-function method)		        ;lambda-list
	 (standard-generic-function method)	        ;specializers
	 real-add-method))			        ;method-function
      (remove-method
	((generic-function method)
	 (standard-generic-function method)
	 real-remove-method))
      (get-method
        ((generic-function qualifiers specializers &optional (errorp t))
	 (standard-generic-function t t)
	 real-get-method))
      (ensure-generic-function-using-class
	((generic-function function-specifier
			   &key generic-function-class environment
			   &allow-other-keys)
	 (generic-function t)
	 real-ensure-gf-using-class--generic-function)
        ((generic-function function-specifier
			   &key generic-function-class environment
			   &allow-other-keys)
	 (null t)
	 real-ensure-gf-using-class--null))
      (make-method-lambda
       ((proto-generic-function proto-method lambda-expression environment)
	(standard-generic-function standard-method t t)
	real-make-method-lambda))
      (make-method-initargs-form
       ((proto-generic-function proto-method lambda-expression lambda-list environment)
	(standard-generic-function standard-method t t t)
	real-make-method-initargs-form))
      (compute-effective-method
       ((generic-function combin applicable-methods)
	(generic-function standard-method-combination t)
	standard-compute-effective-method))
      ))



;;;  FIXME need GCL support for these kernel:* functions

;;;
;;;
;;;
;;;
;;; ANSI 3.4.2, Generic Function Lambda Lists
;;;

;(defun parse-generic-function-lambda-list (lambda-list)
  ;; This is like kernel:parse-lambda-list, but returns an additional
  ;; value AUXP which is true if LAMBDA-LIST contains any &aux keyword.
;  (multiple-value-bind (required optional restp rest keyp keys
;				 allow-other-keys-p aux morep
;				 more-context more-count)
;      (kernel:parse-lambda-list lambda-list)
;    (values required optional restp rest keyp keys allow-other-keys-p
;	    (or aux (member '&aux lambda-list :test #'eq)) aux
;	    morep more-context more-count)))

;(defun check-generic-function-lambda-list (function-specifier lambda-list)
;  (multiple-value-bind (required optional restp rest keyp keys
;				 allow-other-keys-p auxp aux morep
;				 more-context more-count)
;      (parse-generic-function-lambda-list lambda-list)
;    (declare (ignore restp rest keyp aux allow-other-keys-p more-context
;		     more-count))
;    (labels ((lambda-list-error (format-control &rest format-arguments)
;	       (simple-program-error "Generic function ~A:~%~?"
;				     function-specifier
;				     format-control format-arguments))
;	     (check-required-parameter (parameter)
;	       (unless (symbolp parameter)
;		 (lambda-list-error
;		  "Invalid generic function parameter name ~A"
;		  parameter)))
;	     (check-key-or-optional-parameter (parameter)
;	       (unless (or (symbolp parameter)
;			   (and (consp parameter)
;				(symbolp (car parameter))))
;		 (lambda-list-error
;		  "Invalid generic function parameter name: ~A"
;		  parameter))
;	       (when (and (consp parameter)
;			  (not (null (cdr parameter))))
;		 (lambda-list-error
;		  "Optional and key parameters of generic functions~%~
;                   may not have default values or supplied-p ~
;                   parameters: ~A" parameter))))
;      (when morep
;	(lambda-list-error
;	 "&MORE not allowed in generic function lambda lists"))
;      (when auxp
;	(lambda-list-error
;	 "&AUX not allowed in generic function lambda lists"))
;      (mapc #'check-required-parameter required)
;      (mapc #'check-key-or-optional-parameter optional)
;      (mapc #'check-key-or-optional-parameter keys))))

(defmacro defgeneric (function-specifier lambda-list &body options)
;  (check-generic-function-lambda-list function-specifier lambda-list)
  (expand-defgeneric function-specifier lambda-list options))

(defun expand-defgeneric (function-specifier lambda-list options)
  (let ((initargs ())
	(methods ()))
   (labels ((loose (format-control &rest format-arguments)
		    (error 'program-error
		     (format nil "~~@<Generic function ~~s: ~?.~~@:>" format-control format-arguments)
		     function-specifier))
	     (duplicate-option (name) (loose "The option ~S appears more than once." name))
	     (check-declaration (declaration-specifiers)
	       (loop for specifier in declaration-specifiers
		     when (and (consp specifier)
			       (member (car specifier) '(special ftype function inline notinline declaration))) do
				       (loose "Declaration specifier ~S is not allowed" specifier)))
	     (check-argument-precedence-order (precedence)
	       (let ((required (ldiff lambda-list
				      (member-if (lambda (x) (member x '(&optional &rest &key &allow-other-keys &aux))) lambda-list))))
		 (when (set-difference required precedence)
		   (loose "Argument precedence order must list all ~
                           required parameters and only those: ~s"
			  precedence))
		 (when (/= (length (remove-duplicates precedence))
			   (length precedence))
		   (loose "Duplicate parameter names in argument ~
                           precedence order: ~s"
			  precedence))))
	     (initarg (key &optional (new nil new-supplied-p))
		      (if new-supplied-p
			  (setf (getf initargs key) new)
			(getf initargs key))))
      (when (and (symbolp function-specifier)
		 (special-operator-p function-specifier))
	(loose "Special operators cannot be made generic functions"))
      (dolist (option options)
	(case (car option)
	  (:argument-precedence-order
	   (when (initarg :argument-precedence-order)
	     (duplicate-option :argument-precedence-order))
	   (check-argument-precedence-order (cdr option))
	   (initarg :argument-precedence-order `',(cdr option)))
	  (declare
	   (check-declaration (cdr option))
	   (initarg :declarations (append (cdr option) (initarg :declarations))))
	  (:documentation
	   (if (initarg :documentation)
	       (duplicate-option :documentation)
	     (initarg :documentation `',(cadr option))))
	  (:method-combination
	   (if (initarg :method-combination)
	       (duplicate-option :method-combination)
	     (initarg :method-combination `',(cdr option))))
	  (:generic-function-class
	   (if (initarg :generic-function-class)
	       (duplicate-option :generic-function-class)
	     (initarg :generic-function-class `',(cadr option))))
	  (:method-class
	   (if (initarg :method-class)
	       (duplicate-option :method-class)
	     (initarg :method-class `',(cadr option))))
	  (:method
	   (push `(defmethod ,function-specifier ,@(cdr option)) methods))
	  (t (loose "Unsupported option ~S." option))))
      
      (let ((declarations (initarg :declarations)))
	(when declarations (initarg :declarations `',declarations))))
;    (tell-compiler-about-gf function-specifier lambda-list)
  `(progn
     (proclaim-defgeneric ',function-specifier ',lambda-list)
     ,(make-top-level-form
       `(defgeneric ,function-specifier)
			   *defgeneric-times*
			   `(load-defgeneric ',function-specifier ',lambda-list ,@initargs))
     ,@(when methods
	 `((set-initial-methods (mapcar 'eval ',methods) (function ,function-specifier))))
     `,(function ,function-specifier))))

(defun set-initial-methods (methods gf)
  (setf (generic-function-initial-methods gf) methods))

(defun load-defgeneric (function-specifier lambda-list &rest initargs)
   (when (fboundp function-specifier)
    (warn "Redefining ~s" function-specifier)
    (let ((fun (fdefinition function-specifier)))
      (when (generic-function-p fun)
	(mapc (lambda (x) (remove-method fun x)) (generic-function-initial-methods fun))
	(setf (generic-function-initial-methods fun) '()))))
   (apply #'ensure-generic-function
	 function-specifier
	 :lambda-list lambda-list
	 :definition-source `((defgeneric ,function-specifier)
			      ,(load-truename))
	 initargs))


;;;
;;;
;;;
(defmacro DEFMETHOD (&rest args &environment env)
  (declare (arglist name
		    {method-qualifier}*
		    specialized-lambda-list
		    &body body))
  (multiple-value-bind (name qualifiers lambda-list body)
      (parse-defmethod args)
    (multiple-value-bind (proto-gf proto-method)
	(prototypes-for-make-method-lambda name)
      (expand-defmethod name proto-gf proto-method
			qualifiers lambda-list body env))))

(defun prototypes-for-make-method-lambda (name)
  (if (not (eq *boot-state* 'complete))	  
      (values nil nil)
      (let ((gf? (and (gboundp name)
		      (gdefinition name))))
	(if (or (null gf?)
		(not (generic-function-p gf?)))
	    (values (class-prototype (find-class 'standard-generic-function))
		    (class-prototype (find-class 'standard-method)))
	    (values gf?
		    (class-prototype (or (generic-function-method-class gf?)
					 (find-class 'standard-method))))))))

;;;
;;; takes a name which is either a generic function name or a list specifying
;;; a setf generic function (like: (SETF <generic-function-name>)).  Returns
;;; the prototype instance of the method-class for that generic function.
;;;
;;; If there is no generic function by that name, this returns the default
;;; value, the prototype instance of the class STANDARD-METHOD.  This default
;;; value is also returned if the spec names an ordinary function or even a
;;; macro.  In effect, this leaves the signalling of the appropriate error
;;; until load time.
;;;
;;; NOTE that during bootstrapping, this function is allowed to return NIL.
;;; 
(defun method-prototype-for-gf (name)      
  (let ((gf? (and (gboundp name)
		  (gdefinition name))))
    (cond ((neq *boot-state* 'complete) nil)
	  ((or (null gf?)
	       (not (generic-function-p gf?)))	        ;Someone else MIGHT
						        ;error at load time.
	   (class-prototype (find-class 'standard-method)))
	  (t
	    (class-prototype (or (generic-function-method-class gf?)
				 (find-class 'standard-method)))))))


(defvar *optimize-asv-funcall-p* nil)
(defvar *asv-readers*)
(defvar *asv-writers*)
(defvar *asv-boundps*)

(defun expand-defmethod (name proto-gf proto-method qualifiers lambda-list body env)
  (let ((*make-instance-function-keys* nil)
	(*optimize-asv-funcall-p* t)
	(*asv-readers* nil) (*asv-writers* nil) (*asv-boundps* nil))
    (declare (special *make-instance-function-keys*))
    (multiple-value-bind (method-lambda unspecialized-lambda-list specializers)
	(add-method-declarations name qualifiers lambda-list body env)
      (multiple-value-bind (method-function-lambda initargs)
	  (make-method-lambda proto-gf proto-method method-lambda env)
	(let ((initargs-form (make-method-initargs-form 
			      proto-gf proto-method
			      method-function-lambda initargs env)))
;	  (tell-compiler-about-gf name lambda-list)
	  `(progn
	     (proclaim-defgeneric ',name ',lambda-list)
	     ,@(when *make-instance-function-keys*
		 `((get-make-instance-functions ',*make-instance-function-keys*)))
	     ,@(when (or *asv-readers* *asv-writers* *asv-boundps*)
		 `((initialize-internal-slot-gfs*
		    ',*asv-readers* ',*asv-writers* ',*asv-boundps*)))
	     ,(make-defmethod-form name qualifiers specializers
				   unspecialized-lambda-list
				   (if proto-method
				       (class-name (class-of proto-method))
				       'standard-method)
				   initargs-form
	                           (getf (getf initargs :plist)
				         :pv-table-symbol))))))))

(defun interned-symbol-p (x)
  (and (symbolp x) (symbol-package x)))

(defun make-defmethod-form (name qualifiers specializers
				 unspecialized-lambda-list method-class-name
				 initargs-form &optional pv-table-symbol)
  (let (fn fn-lambda)
    (if (and (interned-symbol-p (if (consp name)
				    (and (eq (car name) 'setf) (cadr name))
				    name))
	     (every #'interned-symbol-p qualifiers)
	     (every (lambda (s)
			(if (consp s)
			    (and (eq (car s) 'eql) 
				 (constantp (cadr s))
				 (let ((sv (eval (cadr s))))
				   (or (interned-symbol-p sv)
				       (integerp sv)
				       (and (characterp sv)
					    (standard-char-p sv)))))
			    (interned-symbol-p s)))
		    specializers)
	     (consp initargs-form)
	     (eq (car initargs-form) 'list*)
	     (memq (cadr initargs-form) '(:function :fast-function))
	     (consp (setq fn (caddr initargs-form)))
	     (eq (car fn) 'function)
	     (consp (setq fn-lambda (cadr fn)))
	     (eq (car fn-lambda) 'lambda))
	(let* ((specls (mapcar (lambda (specl)
				   (if (consp specl)
				       `(,(car specl) ,(eval (cadr specl)))
				       specl))
			       specializers))
	       (mname `(,(if (eq (cadr initargs-form) :function)
			     'method 'fast-method)
			,name ,@qualifiers ,specls))
	       (mname-sym (intern (let ((*print-pretty* nil))
				    (format nil "~S" mname)))))
	  `(eval-when ,*defmethod-times*
	    (defun ,mname-sym ,(cadr fn-lambda)
	      ,@(cddr fn-lambda))
	    ,(make-defmethod-form-internal 
	      name qualifiers `',specls
	      unspecialized-lambda-list method-class-name
	      `(list* ,(cadr initargs-form) #',mname-sym ,@(cdddr initargs-form))
	      pv-table-symbol)))
	(make-top-level-form 
	 `(defmethod ,name ,@qualifiers ,specializers)
	 *defmethod-times*
	 (make-defmethod-form-internal 
	  name qualifiers 
	  `(list ,@(mapcar (lambda (specializer)
			       (if (consp specializer)
				   ``(,',(car specializer) ,,(cadr specializer))
				   `',specializer))
		    specializers))
	  unspecialized-lambda-list method-class-name
	  initargs-form
	  pv-table-symbol)))))

(defun make-defmethod-form-internal (name qualifiers specializers-form
					  unspecialized-lambda-list method-class-name
					  initargs-form &optional pv-table-symbol)
  `(load-defmethod
    ',method-class-name
    ',name
    ',qualifiers
    ,specializers-form
    ',unspecialized-lambda-list
    ,initargs-form
    ;;Paper over a bug in KCL by passing the cache-symbol
    ;;here in addition to in the list.
    ',pv-table-symbol))

(defmacro make-method-function (method-lambda &environment env)
  (make-method-function-internal method-lambda env))

(defun make-method-function-internal (method-lambda &optional env)
  (multiple-value-bind (proto-gf proto-method)
	(prototypes-for-make-method-lambda nil)
    (multiple-value-bind (method-function-lambda initargs)
	(make-method-lambda proto-gf proto-method method-lambda env)
      (make-method-initargs-form proto-gf proto-method
				 method-function-lambda initargs env))))

(defun add-method-declarations (name qualifiers lambda-list body env)
  (multiple-value-bind (parameters unspecialized-lambda-list specializers)
      (parse-specialized-lambda-list lambda-list)
    (declare (ignore parameters))
    (multiple-value-bind (documentation declarations real-body)
	(extract-declarations body env)
      (values `(lambda ,unspecialized-lambda-list
		 ,@(when documentation `(,documentation))
		 (declare (method-name ,(list name qualifiers specializers)))
		 (declare (method-lambda-list ,@lambda-list))
		 ,@declarations
		 ,@real-body)
	      unspecialized-lambda-list specializers))))

(defun real-make-method-initargs-form (proto-gf proto-method 
				       method-lambda initargs env)
  (declare (ignore proto-gf proto-method))
  (unless (and (consp method-lambda) (eq (car method-lambda) 'lambda))
    (error "The method-lambda argument to make-method-function, ~S,~
            is not a lambda form" method-lambda))
  (make-method-initargs-form-internal method-lambda initargs env))

(unless (fboundp 'make-method-initargs-form)
  (setf (gdefinition 'make-method-initargs-form)
	(symbol-function 'real-make-method-initargs-form)))

(defun real-make-method-lambda (proto-gf proto-method method-lambda env)
  (declare (ignore proto-gf proto-method))
  (make-method-lambda-internal method-lambda env))

(defun make-method-lambda-internal (method-lambda &optional env)
  (unless (and (consp method-lambda) (eq (car method-lambda) 'lambda))
    (error "The method-lambda argument to make-method-lambda, ~S,~
            is not a lambda form" method-lambda))
  (multiple-value-bind (documentation declarations real-body)
      (extract-declarations (cddr method-lambda) env)
    (let* ((name-decl (get-declaration 'method-name declarations))
	   (sll-decl (get-declaration 'method-lambda-list declarations))
	   (method-name (when (consp name-decl) (car name-decl)))
	   (generic-function-name (when method-name (car method-name)))
	   (specialized-lambda-list (or sll-decl (cadr method-lambda))))
      (multiple-value-bind (parameters lambda-list specializers)
	  (parse-specialized-lambda-list specialized-lambda-list)
	(let* ((required-parameters
		(mapcar (lambda (r s) (declare (ignore s)) r)
			parameters
			specializers))
	       (specialized-parameters
	        (loop for s in specialized-lambda-list
		      for p in required-parameters
		      when (listp s) collect p))
	       (slots (mapcar #'list required-parameters))
	       (calls (list nil))
	       (class-declarations
		`(declare
		  ,@(mapcan (lambda (a s)
			      (when (and (symbolp s) (neq s t))
				(list `(class ,a ,s))))
;				(list `(type ,s ,a))))
			    parameters
			    specializers)))
	       (method-lambda
		  ;; Remove the documentation string and insert the
		  ;; appropriate class declarations.  The documentation
		  ;; string is removed to make it easy for us to insert
		  ;; new declarations later, they will just go after the
		  ;; cadr of the method lambda.  The class declarations
		  ;; are inserted to communicate the class of the method's
		  ;; arguments to the code walk.
		  `(lambda ,lambda-list
		     (declare (ignorable ,@specialized-parameters))
		     ,class-declarations
		     ,@declarations
		     (block ,(if (listp generic-function-name)
				 (cadr generic-function-name)
				 generic-function-name)
		       ,@real-body)))
	       (constant-value-p (and (null (cdr real-body))
				      (constantp (car real-body))))
	       (constant-value (and constant-value-p
				    (eval (car real-body))))
	       (plist (if (and constant-value-p
			       (or (typep constant-value '(or number character))
				   (and (symbolp constant-value)
					(symbol-package constant-value))))
			  (list :constant-value constant-value)
			  ()))
	       (walked-lambda (walk-method-lambda method-lambda
						  required-parameters env 
						  slots calls)))
	      (multiple-value-bind (ignore walked-declarations walked-lambda-body)
		  (extract-declarations (cddr walked-lambda))
		(declare (ignore ignore))
		(when (some #'cdr slots)
		  (multiple-value-bind (slot-name-lists call-list)
		      (slot-name-lists-from-slots slots calls)
		    (let ((pv-table-symbol (make-symbol "pv-table")))
		      (setq plist 
			    `(,@(when slot-name-lists 
				  `(:slot-name-lists ,slot-name-lists))
			      ,@(when call-list
				  `(:call-list ,call-list))
			      :pv-table-symbol ,pv-table-symbol
			      ,@plist))
		      (setq walked-lambda-body
			    `((pv-binding (,required-parameters
					   ,slot-name-lists
					   ,pv-table-symbol)
			       ,@walked-lambda-body))))))
		(when (and (memq '&key lambda-list)
			   (not (memq '&allow-other-keys lambda-list)))
		  (let ((aux (memq '&aux lambda-list)))
		    (setq lambda-list (nconc (ldiff lambda-list aux)
					     (list '&allow-other-keys)
					     aux))))
		(values `(lambda (.method-args. .next-methods.)
			   (simple-lexical-method-functions
			       (,lambda-list .method-args. .next-methods.
                                :method-name-declaration ,name-decl)
			     ,@walked-declarations
			     ,@walked-lambda-body))
			`(,@(when plist 
			      `(:plist ,plist))
			  ,@(when documentation 
			      `(:documentation ,documentation))))))))))
		 
(unless (fboundp 'make-method-lambda)
  (setf (gdefinition 'make-method-lambda)
	(symbol-function 'real-make-method-lambda)))

(defmacro simple-lexical-method-functions 
  ((lambda-list method-args next-methods &rest lmf-options) 
   &body body)
  `(progn
     ,method-args ,next-methods
     (bind-simple-lexical-method-macros (,method-args ,next-methods)
       (bind-lexical-method-functions (,@lmf-options)
         (bind-args (,lambda-list ,method-args)
	   ,@body)))))

(defmacro fast-lexical-method-functions ((lambda-list next-method-call args rest-arg &rest lmf-options)
					 &body body)
  `(bind-fast-lexical-method-macros
    (,args ,rest-arg ,next-method-call)
    (bind-lexical-method-functions
     (,@lmf-options)
     (bind-args
      (,lambda-list (list* ,@args ,rest-arg))
      ,@body))))

(defun call-no-next-method (method-name-declaration &rest args)
  (destructuring-bind (name qualifiers specializers)
      (car method-name-declaration)
    (let ((method (find-method (gdefinition name) qualifiers specializers)))
      (apply #'no-next-method (method-generic-function method) method args))))

(defun %check-cnm-args (cnm-args orig-args method-name-declaration)
  (declare (optimize (speed 3) (safety 0) (debug 0))
           (type list cnm-args orig-args))
  ;; 1. Check for no arguments.
  (when cnm-args
    (let* ((gf (gdefinition (caar method-name-declaration)));(method-generic-function (car method-cell))
           (nreq (generic-function-nreq gf)))
      (declare (type (integer 0 #.call-arguments-limit) nreq))
      ;; 2. Consider required arguments pairwise: if the old and new
      ;; arguments in all pairs are EQL, the applicable methods must
      ;; be the same. This takes care of the relatively common case of
      ;; twiddling with &KEY arguments.
      (unless (do ((orig orig-args (cdr orig))
                   (args cnm-args (cdr args))
                   (n nreq (1- n)))
                  ((zerop n) t)
                (declare (type (integer 0 #.call-arguments-limit) n))
                (unless (eql (car orig) (car args))
                  (return nil)))
        ;; 3. Only then make a cnm args checker and do the full check.
        ;; Disabled until problems with EQL specializers and method
        ;; "shadowing" are worked out.
        #+(or) (let ((result (%use-cnm-checker gf nreq cnm-args orig-args)))
                 (when result
                   (destructuring-bind (cnm-methods . orig-methods) result
                     (error "~@<The set of methods ~S applicable to argument~P ~
                             ~{~S~^, ~} to call-next-method is different from ~
                             the set of methods ~S applicable to the original ~
                             method argument~P ~{~S~^, ~}.~@:>"
                            cnm-methods (length cnm-args) cnm-args
                            orig-methods (length orig-args) orig-args))))
        (let ((orig-methods (compute-applicable-methods gf orig-args))
              (cnm-methods (compute-applicable-methods gf cnm-args)))
          (unless (equal orig-methods cnm-methods)
           (error "~@<The set of methods ~S applicable to argument~P ~
                   ~{~S~^, ~} to call-next-method is different from ~
                   the set of methods ~S applicable to the original ~
                   method argument~P ~{~S~^, ~}.~@:>"
                   cnm-methods (length cnm-args) cnm-args
                   orig-methods (length orig-args) orig-args)))))))

(defmacro bind-simple-lexical-method-macros 
  ((method-args next-methods) &body body)
  `(macrolet ((call-next-method-bind (&body body)
		`(let ((.next-method. (car ,',next-methods))
		       (,',next-methods (cdr ,',next-methods)))
		   .next-method. ,',next-methods
		   ,@body))
	      (call-next-method-body (method-name-declaration cnm-args)
		`(progn
		   (%check-cnm-args cnm-args ,',method-args ',method-name-declaration)
		   (if .next-method.
		     (funcall (if (std-instance-p .next-method.)
				  (method-function .next-method.)
				.next-method.) ; for early methods
			      (or ,cnm-args ,',method-args)
		              ,',next-methods)
		     (apply #'call-no-next-method ',method-name-declaration
                            (or ,cnm-args ,',method-args)))))
	      (next-method-p-body ()
	        `(not (null .next-method.))))
     ,@body))

(defstruct method-call
  (function #'identity :type function)
  call-method-args)

#+cmu
(declaim (ext:freeze-type method-call))

(defmacro invoke-method-call1 (function args cm-args)
  `(let ((.function. ,function)
	 (.args. ,args)
	 (.cm-args. ,cm-args))
     (if (and .cm-args. (null (cdr .cm-args.)))
	 (funcall .function. .args. (car .cm-args.))
	 (apply .function. .args. .cm-args.))))

(defmacro invoke-method-call (method-call restp &rest required-args+rest-arg)
  `(invoke-method-call1 (method-call-function ,method-call)
                        ,(if restp
			     `(list* ,@required-args+rest-arg)
			     `(list ,@required-args+rest-arg))
                        (method-call-call-method-args ,method-call)))

(defstruct fast-method-call
  (function #'identity :type function)
  pv-cell
  next-method-call
  arg-info)

#+cmu
(declaim (ext:freeze-type fast-method-call))

#-akcl
(defmacro fmc-funcall (fn pv-cell next-method-call &rest args)
  `(funcall ,fn ,pv-cell ,next-method-call ,@args))

(defmacro invoke-fast-method-call (method-call &rest required-args+rest-arg)
  `(fmc-funcall (fast-method-call-function ,method-call)
                (fast-method-call-pv-cell ,method-call)
                (fast-method-call-next-method-call ,method-call)
                ,@required-args+rest-arg))

(defstruct fast-instance-boundp
  (index 0 :type fixnum))

#+cmu
(declaim (ext:freeze-type fast-instance-boundp))

(eval-when (compile load eval)
(defvar *allow-emf-call-tracing-p* nil)
(defvar *enable-emf-call-tracing-p* #-testing nil #+testing t)
)

(defvar *emf-call-trace-size* 200)
(defvar *emf-call-trace* nil)
(defvar emf-call-trace-index 0)

(defun show-emf-call-trace ()
  (when *emf-call-trace*
    (let ((j emf-call-trace-index)
	  (*enable-emf-call-tracing-p* nil))
      (format t "~&(The oldest entries are printed first)~%")
      (dotimes (i *emf-call-trace-size*)
	(declare (fixnum i))
	(let ((ct (aref *emf-call-trace* j)))
	  (when ct (print ct)))
	(incf j)
	(when (= j *emf-call-trace-size*)
	  (setq j 0))))))

(defun trace-emf-call-internal (emf format args)
  (unless *emf-call-trace*
    (setq *emf-call-trace* (make-array *emf-call-trace-size*)))
  (setf (aref *emf-call-trace* emf-call-trace-index)
	(list* emf format args))
  (incf emf-call-trace-index)
  (when (= emf-call-trace-index *emf-call-trace-size*)
    (setq emf-call-trace-index 0)))

(defmacro trace-emf-call (emf format args)
  (when *allow-emf-call-tracing-p*
    `(when *enable-emf-call-tracing-p*
       (trace-emf-call-internal ,emf ,format ,args))))

(defmacro invoke-effective-method-function-fast
    (emf restp &rest required-args+rest-arg)
  `(progn
     (trace-emf-call ,emf ,restp (list ,@required-args+rest-arg))
     (invoke-fast-method-call ,emf ,@required-args+rest-arg)))

(defmacro invoke-effective-method-function (emf restp &rest required-args+rest-arg)
  (unless (constantp restp)
    (error "The restp argument to invoke-effective-method-function is not constant"))
  (setq restp (eval restp))
  `(progn
     (trace-emf-call ,emf ,restp (list ,@required-args+rest-arg))
     (cond ((typep ,emf 'fast-method-call)
	    (invoke-fast-method-call ,emf ,@required-args+rest-arg))
	   ,@(when (and (null restp) (= 1 (length required-args+rest-arg)))
	       `(((typep ,emf 'fixnum)
		  (let* ((.slots. (get-slots-or-nil
				   ,(car required-args+rest-arg)))
			 (value (when .slots. (%instance-ref .slots. ,emf))))
		    (if (eq value ',*slot-unbound*)
			(slot-unbound-internal ,(car required-args+rest-arg)
					       ,emf)
			value)))))
	   ,@(when (and (null restp) (= 2 (length required-args+rest-arg)))
	       `(((typep ,emf 'fixnum)
		  (let ((.new-value. ,(car required-args+rest-arg))
			(.slots. (get-slots-or-nil
				  ,(car required-args+rest-arg))))
		    (when .slots. ; just to avoid compiler wranings
		      (setf (%instance-ref .slots. ,emf) .new-value.))))))
	   #|| 
	   ,@(when (and (null restp) (= 1 (length required-args+rest-arg)))
	       `(((typep ,emf 'fast-instance-boundp)
		  (let ((.slots. (get-slots-or-nil
				  ,(car required-args+rest-arg))))
		    (and .slots.
			 (not (eq (%instance-ref
				   .slots. (fast-instance-boundp-index ,emf))
				  ',*slot-unbound*)))))))
	   ||#
	   (t
	    (etypecase ,emf
	      (method-call
	       (invoke-method-call ,emf ,restp ,@required-args+rest-arg))
	      (function
	       ,(if restp
		    `(apply (the function ,emf) ,@required-args+rest-arg)
		    `(funcall (the function ,emf)
			      ,@required-args+rest-arg))))))))

(defun invoke-emf (emf args)
  (trace-emf-call emf t args)
  (etypecase emf
    (fast-method-call
     (let* ((arg-info (fast-method-call-arg-info emf))
	    (restp (cdr arg-info))
	    (nreq (car arg-info)))
       (if restp
	   (let* ((rest-args (nthcdr nreq args))
		  (req-args (ldiff args rest-args)))
	     (apply (fast-method-call-function emf)
		    (fast-method-call-pv-cell emf)
		    (fast-method-call-next-method-call emf)
		    (nconc req-args (list rest-args))))
	   (cond ((null args)
		  (if (eql nreq 0) 
		      (invoke-fast-method-call emf)
		      (error 'program-error :format-control "wrong number of args")))
		 ((null (cdr args))
		  (if (eql nreq 1) 
		      (invoke-fast-method-call emf (car args))
		      (error 'program-error :format-control "wrong number of args")))
		 ((null (cddr args))
		  (if (eql nreq 2) 
		      (invoke-fast-method-call emf (car args) (cadr args))
		      (error 'program-error :format-control "wrong number of args")))
		 (t
		  (apply (fast-method-call-function emf)
			 (fast-method-call-pv-cell emf)
			 (fast-method-call-next-method-call emf)
			 args))))))
    (method-call 
     (apply (method-call-function emf)
	    args
	    (method-call-call-method-args emf)))
    (fixnum 
     (cond ((null args) (error "1 or 2 args expected"))
	   ((null (cdr args))
	    (let ((value (%instance-ref (get-slots (car args)) emf)))
	      (if (eq value *slot-unbound*)
		  (slot-unbound-internal (car args) emf)
		  value)))
	   ((null (cddr args))
	    (setf (%instance-ref (get-slots (cadr args)) emf)
		  (car args)))
	   (t (error "1 or 2 args expected"))))
    (fast-instance-boundp
     (if (or (null args) (cdr args))
	 (error "1 arg expected")
	 (not (eq (%instance-ref (get-slots (car args)) 
				 (fast-instance-boundp-index emf))
		  *slot-unbound*))))
    (function
     (apply emf args))))

(defmacro bind-fast-lexical-method-macros 
  ((args rest-arg next-method-call) &body body)
  `(macrolet ((call-next-method-bind (&body body)
		`(let () ,@body))
	      (call-next-method-body (method-name-declaration cnm-args)
		`(progn
		   (%check-cnm-args cnm-args (list ,@',args) ',method-name-declaration)
		   (if ,',next-method-call
		     ,(if (and (null ',rest-arg)
			       (consp cnm-args)
			       (eq (car cnm-args) 'list))
			  `(invoke-effective-method-function
			    ,',next-method-call nil
			    ,@(cdr cnm-args))
			  (let ((call `(invoke-effective-method-function
					,',next-method-call 
					,',(not (null rest-arg))
					,@',args 
					,@',(when rest-arg `(,rest-arg)))))
			    `(if ,cnm-args
				 (bind-args ((,@',args 
					      ,@',(when rest-arg
						    `(&rest ,rest-arg)))
					     ,cnm-args)
					    ,call)
				 ,call)))
                     ,(if (and (null ',rest-arg)
                               (consp cnm-args)
                               (eq (car cnm-args) 'list))
                          `(call-no-next-method ',method-name-declaration
                                                ,@(cdr cnm-args))
                          `(call-no-next-method ',method-name-declaration
                                                ,@',args
                                                ,@',(when rest-arg
                                                      `(,rest-arg)))))))
	      (next-method-p-body ()
	        `(not (null ,',next-method-call))))
     ,@body))

(defmacro bind-lexical-method-functions 
    ((&key method-name-declaration) &body body)
    `(call-next-method-bind
      (flet ((call-next-method (&rest cnm-args)
	       (declare (dynamic-extent cnm-args))
	       (call-next-method-body ,method-name-declaration cnm-args))
	     (next-method-p () (next-method-p-body)))
	(declare (ignorable #'call-next-method #'next-method-p))
	,@body)))

(defmacro bind-args ((lambda-list args) &body body)
  `(destructuring-bind ,lambda-list ,args ,@body))

(defun get-key-arg-tail (keyword list)
  (loop for (key . tail) on list by #'cddr
	when (null tail) do
	  (error 'program-error :format-control "Odd number of keyword arguments in ~S" :format-arguments (list list))
	when (eq key keyword)
	  return tail))

(defun walk-method-lambda (method-lambda required-parameters env slots calls)
  (flet ((walk-function (form context env)
			(cond ((not (eq context :eval)) form)
			      ((not (listp form)) form)
			      ((and (or (eq (car form) 'slot-value)
					(eq (car form) 'set-slot-value)
					(eq (car form) 'slot-boundp))
				    (constantp (caddr form)))
			       (let ((parameter
				      (can-optimize-access form required-parameters env)))
				 (ecase (car form)
				   (slot-value
				    (optimize-slot-value     slots parameter form))
				   (set-slot-value
				    (optimize-set-slot-value slots parameter form))
				   (slot-boundp
				    (optimize-slot-boundp    slots parameter form)))))
			      ((and (eq (car form) 'apply)
				    (consp (cadr form))
				    (eq (car (cadr form)) 'function)
				    (generic-function-name-p (cadr (cadr form))))
			       (optimize-generic-function-call 
				form required-parameters env slots calls))
			      ((and (or (symbolp (car form))
					(and (consp (car form))
					     (eq (caar form) 'setf)))
				    (generic-function-name-p (car form)))
			       (optimize-generic-function-call 
				form required-parameters env slots calls))
			      ((and (eq (car form) 'asv-funcall)
				    *optimize-asv-funcall-p*)
			       (case (fourth form)
				 (reader (push (third form) *asv-readers*))
				 (writer (push (third form) *asv-writers*))
				 (boundp (push (third form) *asv-boundps*)))
			       `(,(second form) ,@(cddddr form)))
			      (t form))))
    
    (walk-form method-lambda env #'walk-function)))

(defun generic-function-name-p (name)
  (and (or (symbolp name)
	   (and (consp name)
		(eq (car name) 'setf)
		(consp (cdr name))
		(symbolp (cadr name))
		(null (cddr name))))
       (gboundp name)
       (if (eq *boot-state* 'complete)
	   (standard-generic-function-p (gdefinition name))
	   (funcallable-instance-p (gdefinition name)))))



(defvar *method-function-plist* (make-hash-table :test #'eq))
(defvar *mf1* nil) (defvar *mf1p* nil) (defvar *mf1cp* nil)
(defvar *mf2* nil) (defvar *mf2p* nil) (defvar *mf2cp* nil)

(defun method-function-plist (method-function)
  (unless (eq method-function *mf1*)
    (rotatef *mf1* *mf2*)
    (rotatef *mf1p* *mf2p*)
    (rotatef *mf1cp* *mf2cp*))
  (unless (or (eq method-function *mf1*) (null *mf1cp*))
    (setf (gethash *mf1* *method-function-plist*) *mf1p*))
  (unless (eq method-function *mf1*)
    (setf *mf1* method-function
	  *mf1cp* nil
	  *mf1p* (gethash method-function *method-function-plist*)))
  *mf1p*)

(defun (setf method-function-plist)
       (val method-function)
  (unless (eq method-function *mf1*)
    (rotatef *mf1* *mf2*)
    (rotatef *mf1cp* *mf2cp*)
    (rotatef *mf1p* *mf2p*))
  (unless (or (eq method-function *mf1*) (null *mf1cp*))
    (setf (gethash *mf1* *method-function-plist*) *mf1p*))
  (setf *mf1* method-function
	*mf1cp* t
	*mf1p* val))

(defun method-function-get (method-function key &optional default)
  (getf (method-function-plist method-function) key default))

(defun (setf method-function-get)
       (val method-function key)
  (setf (getf (method-function-plist method-function) key) val))


(defun method-function-pv-table (method-function)
  (method-function-get method-function :pv-table))

(defun method-function-method (method-function)
  (method-function-get method-function :method))

(defmacro method-function-closure-generator (method-function)
  `(method-function-get ,method-function 'closure-generator))

(defun load-defmethod (class name quals specls ll initargs &optional pv-table-symbol)
  (setq initargs (copy-tree initargs))
  (let ((method-spec (or (getf initargs :method-spec)
			 (make-method-spec name quals specls))))
    (setf (getf initargs :method-spec) method-spec)
    (load-defmethod-internal class name quals specls ll initargs pv-table-symbol)))

(defun load-defmethod-internal
    (method-class gf-spec qualifiers specializers lambda-list 
		  initargs pv-table-symbol)
  (when pv-table-symbol
    (setf (getf (getf initargs :plist) :pv-table-symbol)
	  pv-table-symbol))
  (let ((method (apply #'add-named-method
		       gf-spec qualifiers specializers lambda-list
		       :definition-source `((defmethod ,gf-spec
						,@qualifiers
					      ,specializers)
					    ,(load-truename))
		       initargs)))
    (unless (or (eq method-class 'standard-method)
		(eq (find-class method-class nil) (class-of method)))
      (format *error-output*
	      "~&At the time the method with qualifiers ~:S and~%~
               specializers ~:S on the generic function ~S~%~
               was compiled, the method-class for that generic function was~%~
               ~S.  But, the method class is now ~S, this~%~
               may mean that this method was compiled improperly.~%"
	      qualifiers specializers gf-spec
	      method-class (class-name (class-of method))))
    method))

(defun make-method-spec (gf-spec qualifiers unparsed-specializers)
  `(method ,gf-spec ,@qualifiers ,unparsed-specializers))

(defun initialize-method-function (initargs &optional return-function-p method)
  (let* ((mf (getf initargs :function))
	 (method-spec (getf initargs :method-spec))
	 (plist (getf initargs :plist))
	 (pv-table-symbol (getf plist :pv-table-symbol))
	 (pv-table nil)
	 (mff (getf initargs :fast-function)))
    (flet ((set-mf-property (p v)
	     (when mf
	       (setf (method-function-get mf p) v))
	     (when mff
	       (setf (method-function-get mff p) v))))
      (when method-spec
	(when mf
	  (setq mf (set-function-name mf method-spec)))
	(when mff
	  (let ((name `(,(or (get (car method-spec) 'fast-sym)
			     (setf (get (car method-spec) 'fast-sym)
				   (intern (format nil "FAST-~A"
						   (car method-spec))
					   *the-pcl-package*)))
			 ,@(cdr method-spec))))
	    (set-function-name mff name)
	    (unless mf
	      (set-mf-property :name name)))))
      (when plist
	(let ((snl (getf plist :slot-name-lists))
	      (cl (getf plist :call-list)))
	  (when (or snl cl)
	    (setq pv-table (intern-pv-table :slot-name-lists snl
					    :call-list cl))
	    (when pv-table (set pv-table-symbol pv-table))
	    (set-mf-property :pv-table pv-table)))    
	(loop (when (null plist) (return nil))
	      (set-mf-property (pop plist) (pop plist)))      
	(when method
	  (set-mf-property :method method))    
	(when return-function-p
	  (or mf (method-function-from-fast-function mff)))))))



(defun analyze-lambda-list (lambda-list)
  ;;(declare (values nrequired noptional keysp restp allow-other-keys-p
  ;;                 keywords keyword-parameters))
  (flet ((parse-keyword-argument (arg)
	   (if (listp arg)
	       (if (listp (car arg))
		   (caar arg)
		   (make-keyword (car arg)))
	       (make-keyword arg))))
    (let ((nrequired 0)
	  (noptional 0)
	  (keysp nil)
	  (restp nil)
	  (allow-other-keys-p nil)
	  (keywords ())
	  (keyword-parameters ())
	  (state 'required))
      (dolist (x lambda-list)
	(if (memq x lambda-list-keywords)
	    (case x
	      (&optional         (setq state 'optional))
	      (&key              (setq keysp 't
				       state 'key))
	      (&allow-other-keys (setq allow-other-keys-p 't))
	      (&rest             (setq restp 't
				       state 'rest))
	      (&aux              (return t))
	      (otherwise
		(error "Encountered the non-standard lambda list keyword ~S." x)))
	    (ecase state
	      (required  (incf nrequired))
	      (optional  (incf noptional))
	      (key       (push (parse-keyword-argument x) keywords)
			 (push x keyword-parameters))
	      (rest      ()))))
      (values nrequired noptional keysp restp allow-other-keys-p
	      (reverse keywords)
	      (reverse keyword-parameters)))))

(defun keyword-spec-name (x)
  (let ((key (if (atom x) x (car x))))
    (if (atom key)
	(intern (symbol-name key) (find-package "KEYWORD"))
	(car key))))

(defun ftype-declaration-from-lambda-list (lambda-list #+cmu name)
  (multiple-value-bind (nrequired noptional keysp restp allow-other-keys-p
				  keywords keyword-parameters)
      (analyze-lambda-list lambda-list)
    (declare (ignore keyword-parameters))
    (let* (#+cmu (old (c::info function type name))
	   #+cmu (old-ftype (if (c::function-type-p old) old nil))
	   #+cmu (old-restp (and old-ftype (c::function-type-rest old-ftype)))
	   #+cmu (old-keys (and old-ftype
				(mapcar #'c::key-info-name
					(c::function-type-keywords old-ftype))))
	   #+cmu (old-keysp (and old-ftype (c::function-type-keyp old-ftype)))
	   #+cmu (old-allowp (and old-ftype (c::function-type-allowp old-ftype)))
	   (keywords #+cmu (union old-keys (mapcar #'keyword-spec-name keywords))
		     #-cmu (mapcar #'keyword-spec-name keywords)))
      `(function ,(append (make-list nrequired :initial-element t)
			  (when (plusp noptional)
			    (append '(&optional)
				    (make-list noptional :initial-element t)))
			  (when (or restp #+cmu old-restp)
			    '(&rest t))
			  (when (or keysp #+cmu old-keysp)
			    (append '(&key)
				    (mapcar (lambda (key)
						`(,key t))
					    keywords)
				    (when (or allow-other-keys-p #+cmu old-allowp)
				      '(&allow-other-keys)))))
		 *))))

(defun proclaim-defgeneric (spec lambda-list)
  #-cmu (declare (ignore lambda-list))
  (when (consp spec)
    (setq spec (get-setf-function-name (cadr spec))))
  (let (#+cmu
	(decl `(ftype ,(ftype-declaration-from-lambda-list lambda-list #+cmu spec)
		      ,spec)))
    #+cmu (proclaim decl)
;;FIXME  proclaim setf compiled closures too
    #+kcl (when (symbolp spec)
	    (setf (get spec 'compiler::proclaimed-closure) t))))

;;;; Early generic-function support
;;;
;;;
(defvar *early-generic-functions* ())

(defun ensure-generic-function (function-specifier
				&rest all-keys
				&key environment
				&allow-other-keys)
  (declare (ignore environment))
  (let* ((function-specifier (or (get (si::funid-sym function-specifier) 'si::traced) function-specifier))
	 (existing (and (gboundp function-specifier) (gdefinition function-specifier))))
    (when (and existing
	       (eq *boot-state* 'complete)
	       (null (generic-function-p existing)))
      (generic-clobbers-function function-specifier)
      (setq existing nil))
    (apply #'ensure-generic-function-using-class
	   existing function-specifier all-keys)))

(defun generic-clobbers-function (function-specifier)
  (cerror "Discard the existing definition." 'program-error 
	  :format-control
	  "~S already names an ordinary function or a macro.~%~
	If you want to replace it with a generic function, you should remove~%~
        the existing definition beforehand.~%" :format-arguments (list function-specifier))
  (fmakunbound function-specifier))

(defvar *sgf-wrapper* 
  (#+cmu17 boot-make-wrapper #-cmu17 make-wrapper
   (early-class-size 'standard-generic-function)
   #+cmu17 'standard-generic-function))

(defvar *sgf-slots-init*
  (map 'vector
       (lambda (canonical-slot)
	   (if (memq (getf canonical-slot :name) '(arg-info source))
	       *slot-unbound*
	       (let ((initfunction (getf canonical-slot :initfunction)))
		 (if initfunction
		     (funcall initfunction)
		     *slot-unbound*))))
       (early-collect-inheritance 'standard-generic-function)))

(defvar *sgf-method-class-index* 
  (bootstrap-slot-index 'standard-generic-function 'method-class))

(defun early-gf-p (x)
  (and (fsc-instance-p x)
       (eq (instance-ref (get-slots x) *sgf-method-class-index*)
	   *slot-unbound*)))

(defvar *sgf-methods-index* 
  (bootstrap-slot-index 'standard-generic-function 'methods))

(defmacro early-gf-methods (gf)
  `(instance-ref (get-slots ,gf) *sgf-methods-index*))

(defvar *sgf-arg-info-index* 
  (bootstrap-slot-index 'standard-generic-function 'arg-info))

(defmacro early-gf-arg-info (gf)
  `(instance-ref (get-slots ,gf) *sgf-arg-info-index*))

(defvar *sgf-dfun-state-index* 
  (bootstrap-slot-index 'standard-generic-function 'dfun-state))

(defstruct (arg-info
	     (:conc-name nil)
	     (:constructor make-arg-info ()))
  (arg-info-lambda-list :no-lambda-list)
  arg-info-precedence
  arg-info-metatypes
  arg-info-number-optional
  arg-info-key/rest-p
  arg-info-keywords ;nil         no keyword or rest allowed
	            ;(k1 k2 ..)  each method must accept these keyword arguments
	            ;T           must have &key or &rest

  gf-info-simple-accessor-type ; nil, reader, writer, boundp
  (gf-precompute-dfun-and-emf-p nil) ; set by set-arg-info

  gf-info-static-c-a-m-emf
  (gf-info-c-a-m-emf-std-p t)
  gf-info-fast-mf-p)

#+cmu
(declaim (ext:freeze-type arg-info))

(defun arg-info-valid-p (arg-info)
  (not (null (arg-info-number-optional arg-info))))

(defun arg-info-applyp (arg-info)
  (or (plusp (the fixnum (arg-info-number-optional arg-info)))
      (arg-info-key/rest-p arg-info)))

(defun arg-info-number-required (arg-info)
  (length (arg-info-metatypes arg-info)))

(defun arg-info-nkeys (arg-info)
  (count-if (lambda (x) (neq x t)) (arg-info-metatypes arg-info)))

;;; Keep pages clean by not setting if the value is already the same.
(defmacro esetf (pos val)
  (let ((valsym (gensym "value")))
    `(let ((,valsym ,val))
       (unless (equal ,pos ,valsym)
	 (setf ,pos ,valsym)))))

(defun generic-function-nreq (gf)
  (let* ((arg-info (if (early-gf-p gf)
                       (early-gf-arg-info gf)
                       (gf-arg-info gf)));safe
         (metatypes (arg-info-metatypes arg-info)))
    (declare (list metatypes))
    (length metatypes)))

(defun set-arg-info (gf &key new-method (lambda-list nil lambda-list-p)
			argument-precedence-order)
  (let* ((arg-info (if (eq *boot-state* 'complete)
		       (gf-arg-info gf)
		       (early-gf-arg-info gf)))
	 (methods (if (eq *boot-state* 'complete)
		      (generic-function-methods gf)
		      (early-gf-methods gf)))
	 (was-valid-p (integerp (arg-info-number-optional arg-info)))
	 (first-p (and new-method (null (cdr methods)))))
    (when (and (not lambda-list-p) methods)      
      (setq lambda-list (gf-lambda-list gf)))
    (when (or lambda-list-p
	      (and first-p (eq (arg-info-lambda-list arg-info) :no-lambda-list)))
      (multiple-value-bind (nreq nopt keysp restp allow-other-keys-p keywords)
	  (analyze-lambda-list lambda-list)
	(when (and methods (not first-p))
	  (let ((gf-nreq (arg-info-number-required arg-info))
		(gf-nopt (arg-info-number-optional arg-info))
		(gf-key/rest-p (arg-info-key/rest-p arg-info)))
	    (unless (and (= nreq gf-nreq)
			 (= nopt gf-nopt)
			 (eq (or keysp restp) gf-key/rest-p))
	      (error "The lambda-list ~S is incompatible with ~
                     existing methods of ~S."
		     lambda-list gf))))
	(when lambda-list-p
	  (esetf (arg-info-lambda-list arg-info) lambda-list))
	(when (or lambda-list-p argument-precedence-order
		  (null (arg-info-precedence arg-info)))
	  (esetf (arg-info-precedence arg-info)
		 (compute-precedence lambda-list nreq
				     argument-precedence-order)))
	(esetf (arg-info-metatypes arg-info) (make-list nreq))
	(esetf (arg-info-number-optional arg-info) nopt)
	(esetf (arg-info-key/rest-p arg-info) (not (null (or keysp restp))))
	(esetf (arg-info-keywords arg-info) 
	       (if lambda-list-p
		   (if allow-other-keys-p t keywords)
		   (arg-info-key/rest-p arg-info)))))
    (when new-method
      (check-method-arg-info gf arg-info new-method))
    (set-arg-info1 gf arg-info new-method methods was-valid-p first-p)
    arg-info))

(defun check-method-arg-info (gf arg-info method)
  (multiple-value-bind (nreq nopt keysp restp allow-other-keys-p keywords)
      (analyze-lambda-list (if (consp method)
			       (early-method-lambda-list method)
			       (method-lambda-list method)))
    (flet ((lose (format-control &rest format-args)
	     (error 'program-error :format-control
			     "Attempt to add the method ~S to the generic function ~S.~%~
                   But ~?" :format-arguments (list method gf format-control format-args)))
	   (compare (x y)
	     (if (> x y) "more" "fewer")))
      (let ((gf-nreq (arg-info-number-required arg-info))
	    (gf-nopt (arg-info-number-optional arg-info))
	    (gf-key/rest-p (arg-info-key/rest-p arg-info))
	    (gf-keywords (arg-info-keywords arg-info)))
	(unless (= nreq gf-nreq)
	  (lose "the method has ~A required arguments than the generic function."
		(compare nreq gf-nreq)))
	(unless (= nopt gf-nopt)
	  (lose "the method has ~S optional arguments than the generic function."
		(compare nopt gf-nopt)))
	(unless (eq (or keysp restp) gf-key/rest-p)
	  (lose "the method and generic function differ in whether they accept~%~
                  rest or keyword arguments."))
	(when (consp gf-keywords)
	  (unless (or (and restp (not keysp))
		      allow-other-keys-p
		      (every (lambda (k) (memq k keywords)) gf-keywords))
	    (lose "the method does not accept each of the keyword arguments~%~
                 ~S." gf-keywords)))))))

(defun set-arg-info1 (gf arg-info new-method methods was-valid-p first-p)  
  (let* ((existing-p (and methods (cdr methods) new-method))
	 (nreq (length (arg-info-metatypes arg-info)))
	 (metatypes (if existing-p
			(arg-info-metatypes arg-info)
			(make-list nreq)))
	 (type (if existing-p
		   (gf-info-simple-accessor-type arg-info)
		   nil)))
    (when (arg-info-valid-p arg-info)
      (dolist (method (if new-method (list new-method) methods))
	(let* ((specializers (if (or (eq *boot-state* 'complete)
				     (not (consp method)))
				 (method-specializers method)
				 (early-method-specializers method t)))
	       (class (if (or (eq *boot-state* 'complete) (not (consp method)))
			  (class-of method)
			  (early-method-class method)))
	       (new-type (when (and class
				    (or (not (eq *boot-state* 'complete))
					(eq (generic-function-method-combination gf)
					    *standard-method-combination*)))
			   (cond ((eq class *the-class-standard-reader-method*)
				  'reader)
				 ((eq class *the-class-standard-writer-method*)
				  'writer)
				 ((eq class *the-class-standard-boundp-method*)
				  'boundp)))))
	  (setq metatypes (mapcar #'raise-metatype metatypes specializers))
	  (setq type (cond ((null type) new-type)
			   ((eq type new-type) type)
			   (t nil)))))
      (esetf (arg-info-metatypes arg-info) metatypes)
      (esetf (gf-info-simple-accessor-type arg-info) type)))
  (when (or (not was-valid-p) first-p)
    (multiple-value-bind (c-a-m-emf std-p)
	(if (early-gf-p gf)
	    (values t t)
	    (compute-applicable-methods-emf gf))
      (esetf (gf-info-static-c-a-m-emf arg-info) c-a-m-emf)
      (esetf (gf-info-c-a-m-emf-std-p arg-info) std-p)
      (unless (gf-info-c-a-m-emf-std-p arg-info)
	(esetf (gf-info-simple-accessor-type arg-info) t))))
  (unless was-valid-p
    (let ((name (if (eq *boot-state* 'complete)
		    (generic-function-name gf)
		    (early-gf-name gf))))
      (esetf (gf-precompute-dfun-and-emf-p arg-info) 
	     (let* ((sym (if (atom name) name (cadr name)))
		    (pkg-list (cons *the-pcl-package* 
				    (package-use-list *the-pcl-package*))))
	       (and sym (symbolp sym)
		    (not (null (memq (symbol-package sym) pkg-list)))
		    (not (find #\space (symbol-name sym))))))))
  (esetf (gf-info-fast-mf-p arg-info)
	 (or (not (eq *boot-state* 'complete))
	     (let* ((method-class (generic-function-method-class gf))
		    (methods (compute-applicable-methods 
			      #'make-method-lambda
			      (list gf (class-prototype method-class)
				    '(lambda) nil))))
	       (and methods (null (cdr methods))
		    (let ((specls (method-specializers (car methods))))
		      (and (classp (car specls))
			   (eq 'standard-generic-function
			       (class-name (car specls)))
			   (classp (cadr specls))
			   (eq 'standard-method
			       (class-name (cadr specls)))))))))
  arg-info)

;;;
;;; This is the early definition of ensure-generic-function-using-class.
;;; 
;;; The static-slots field of the funcallable instances used as early generic
;;; functions is used to store the early methods and early discriminator code
;;; for the early generic function.  The static slots field of the fins
;;; contains a list whose:
;;;    CAR    -   a list of the early methods on this early gf
;;;    CADR   -   the early discriminator code for this method
;;;    
(defun ensure-generic-function-using-class (existing spec &rest keys
					    &key (lambda-list nil lambda-list-p)
					    argument-precedence-order
					    &allow-other-keys)
  (declare (ignore keys))
  (cond ((and existing (early-gf-p existing))
	 existing)
	((assoc spec *generic-function-fixups* :test #'equal)
	 (if existing
	     (make-early-gf spec lambda-list lambda-list-p existing
			    argument-precedence-order)
	     (error "The function ~S is not already defined" spec)))
	(existing
	 (error "~S should be on the list ~S" spec '*generic-function-fixups*))
	(t
	 (pushnew spec *early-generic-functions* :test #'equal)
	 (make-early-gf spec lambda-list lambda-list-p nil
	                argument-precedence-order))))

(defun make-early-gf (spec &optional lambda-list lambda-list-p function
			   argument-precedence-order)
  (let ((fin (allocate-funcallable-instance *sgf-wrapper* *sgf-slots-init*)))
    (set-funcallable-instance-function 
     fin 
     (or function
	 (if (eq spec 'print-object)
	     (fin-lambda-fn (instance stream)
	       (printing-random-thing (instance stream)
		 (format stream "std-instance")))
	     (fin-lambda-fn (&rest args)
	       (declare (ignore args))
	       (warn "The function of the funcallable-instance ~S~
                      has not been set" fin)
	       (values-list (make-list (random multiple-values-limit)))))))
    (setf (gdefinition spec) fin)
    (bootstrap-set-slot 'standard-generic-function fin 'name spec)
    (bootstrap-set-slot 'standard-generic-function fin 'source (load-truename))
    (set-function-name fin spec)
    (let ((arg-info (make-arg-info)))
      (setf (early-gf-arg-info fin) arg-info)
      (when lambda-list-p
	(proclaim-defgeneric spec lambda-list)
	(if argument-precedence-order
	    (set-arg-info fin
	                  :lambda-list lambda-list
			  :argument-precedence-order argument-precedence-order)
	    (set-arg-info fin :lambda-list lambda-list))))
    fin))

(defun set-dfun (gf &optional dfun cache info)
  (when cache
    (setf (cache-owner cache) gf))
  (let ((new-state (if (and dfun (or cache info))
		       (list* dfun cache info)
		       dfun)))
    (if (eq *boot-state* 'complete)
	(setf (gf-dfun-state gf) new-state)
	(setf (instance-ref (get-slots gf) *sgf-dfun-state-index*) new-state)))
  dfun)

(defun gf-dfun-cache (gf)
  (let ((state (if (eq *boot-state* 'complete)
		   (gf-dfun-state gf)
		   (instance-ref (get-slots gf) *sgf-dfun-state-index*))))
    (typecase state
      (function nil)
      (cons (cadr state)))))

(defun gf-dfun-info (gf)
  (let ((state (if (eq *boot-state* 'complete)
		   (gf-dfun-state gf)
		   (instance-ref (get-slots gf) *sgf-dfun-state-index*))))
    (typecase state
      (function nil)
      (cons (cddr state)))))

(defvar *sgf-name-index* 
  (bootstrap-slot-index 'standard-generic-function 'name))

(defun early-gf-name (gf)
  (instance-ref (get-slots gf) *sgf-name-index*))

(defun gf-lambda-list (gf)
  (let ((arg-info (if (eq *boot-state* 'complete)
		      (gf-arg-info gf)
		      (early-gf-arg-info gf))))
    (if (eq :no-lambda-list (arg-info-lambda-list arg-info))
	(let ((methods (if (eq *boot-state* 'complete)
			   (generic-function-methods gf)
			   (early-gf-methods gf))))
	  (if (null methods)
	      (progn
		(warn "No way to determine the lambda list for ~S." gf)
		nil)
	      (let* ((method (car (last methods))))
		(if (consp method)
		    (early-method-lambda-list method)
		    (method-lambda-list method)))))
	(arg-info-lambda-list arg-info))))

(defmacro real-ensure-gf-internal (gf-class all-keys env)
  `(progn
     (cond ((symbolp ,gf-class)
	    (setq ,gf-class (find-class ,gf-class t ,env)))
	   ((classp ,gf-class))
	   (t
	    (error "The :GENERIC-FUNCTION-CLASS argument (~S) was neither a~%~
                    class nor a symbol that names a class."
		   ,gf-class)))
     (remf ,all-keys :generic-function-class)
     (remf ,all-keys :environment)
     (let ((combin (getf ,all-keys :method-combination '.shes-not-there.)))
       (unless (eq combin '.shes-not-there.)
	 (setf (getf ,all-keys :method-combination)
	       (find-method-combination (class-prototype ,gf-class)
					(car combin)
					(cdr combin)))))
     (let ((method-class (getf ,all-keys :method-class '.shes-not-there.)))
       (unless (eq method-class '.shes-not-there.)
	 (setf (getf ,all-keys :method-class)
	       (if (classp method-class) method-class (find-class method-class t ,env)))))))
     
(defun real-ensure-gf-using-class--generic-function
       (existing
	function-specifier
	&rest all-keys
	&key environment (lambda-list nil lambda-list-p)
	     (generic-function-class 'standard-generic-function gf-class-p)
	&allow-other-keys)
  (real-ensure-gf-internal generic-function-class all-keys environment)
  (unless (or (null gf-class-p)
	      (eq (class-of existing) generic-function-class))
    (change-class existing generic-function-class))
  (prog1
      (apply #'reinitialize-instance existing all-keys)
    (when lambda-list-p
      (proclaim-defgeneric function-specifier lambda-list))))

(defun real-ensure-gf-using-class--null
       (existing
	function-specifier
	&rest all-keys
	&key environment (lambda-list nil lambda-list-p)
	     (generic-function-class 'standard-generic-function)
	&allow-other-keys)
  (declare (ignore existing))
  (real-ensure-gf-internal generic-function-class all-keys environment)
  (prog1
      (setf (gdefinition function-specifier)
	    (apply #'make-instance generic-function-class 
		   :name function-specifier all-keys))
    (when lambda-list-p
      (proclaim-defgeneric function-specifier lambda-list))))



(defun get-generic-function-info (gf)
  ;; values   nreq applyp metatypes nkeys arg-info
  (multiple-value-bind (applyp metatypes arg-info)
      (let* ((arg-info (if (early-gf-p gf)
			   (early-gf-arg-info gf)
			   (gf-arg-info gf)))
	     (metatypes (arg-info-metatypes arg-info)))
	(values (arg-info-applyp arg-info)
		metatypes
		arg-info))
;      (print (list 'baz gf (early-gf-p gf) arg-info (arg-info-applyp arg-info)))
    (values (length metatypes) applyp metatypes
	    (count-if (lambda (x) (neq x t)) metatypes)
	    arg-info)))

(defun early-make-a-method (class qualifiers arglist specializers initargs doc
			    &optional slot-name)
  (initialize-method-function initargs)
  (let ((parsed ())
	(unparsed ()))
    ;; Figure out whether we got class objects or class names as the
    ;; specializers and set parsed and unparsed appropriately.  If we
    ;; got class objects, then we can compute unparsed, but if we got
    ;; class names we don't try to compute parsed.
    ;; 
    ;; Note that the use of not symbolp in this call to every should be
    ;; read as 'classp' we can't use classp itself because it doesn't
    ;; exist yet.
    (if (every (lambda (s) (not (symbolp s))) specializers)
	(setq parsed specializers
	      unparsed (mapcar (lambda (s)
				   (if (eq s t) t (class-name s)))
			       specializers))
	(setq unparsed specializers
	      parsed ()))
    (list :early-method		  ;This is an early method dammit!
	  
	  (getf initargs :function)
	  (getf initargs :fast-function)
	  
	  parsed                  ;The parsed specializers.  This is used
				  ;by early-method-specializers to cache
				  ;the parse.  Note that this only comes
				  ;into play when there is more than one
				  ;early method on an early gf.
	  
	  (list class             ;A list to which real-make-a-method
		qualifiers        ;can be applied to make a real method
		arglist           ;corresponding to this early one.
		unparsed
		initargs
		doc
		slot-name)
	  )))

(defun real-make-a-method
       (class qualifiers lambda-list specializers initargs doc
	&optional slot-name)
  (setq specializers (parse-specializers specializers))
  (apply #'make-instance class 
	 :qualifiers qualifiers
	 :lambda-list lambda-list
	 :specializers specializers
	 :documentation doc
	 :slot-name slot-name
	 :allow-other-keys t
	 initargs))

(defun early-method-function (early-method)
  (values (cadr early-method) (caddr early-method)))

(defun early-method-class (early-method)
  (find-class (car (fifth early-method))))

(defun early-method-standard-accessor-p (early-method)
  (let ((class (first (fifth early-method))))
    (or (eq class 'standard-reader-method)
        (eq class 'standard-writer-method)
        (eq class 'standard-boundp-method))))

(defun early-method-standard-accessor-slot-name (early-method)
  (seventh (fifth early-method)))

;;;
;;; Fetch the specializers of an early method.  This is basically just a
;;; simple accessor except that when the second argument is t, this converts
;;; the specializers from symbols into class objects.  The class objects
;;; are cached in the early method, this makes bootstrapping faster because
;;; the class objects only have to be computed once.
;;; NOTE:
;;;  the second argument should only be passed as T by early-lookup-method.
;;;  this is to implement the rule that only when there is more than one
;;;  early method on a generic function is the conversion from class names
;;;  to class objects done.
;;;  the corresponds to the fact that we are only allowed to have one method
;;;  on any generic function up until the time classes exist.
;;;  
(defun early-method-specializers (early-method &optional objectsp)
  (if (and (listp early-method)
	   (eq (car early-method) :early-method))
      (cond ((eq objectsp t)
	     (or (fourth early-method)
		 (setf (fourth early-method)
		       (mapcar #'find-class (cadddr (fifth early-method))))))
	    (t
	     (cadddr (fifth early-method))))
      (error "~S is not an early-method." early-method)))

(defun early-method-qualifiers (early-method)
  (cadr (fifth early-method)))

(defun early-method-lambda-list (early-method)
  (caddr (fifth early-method)))

(defun early-add-named-method (generic-function-name
			       qualifiers
			       specializers
			       arglist
			       &rest initargs)
  (let* ((gf (ensure-generic-function generic-function-name))
	 (existing
	   (dolist (m (early-gf-methods gf))
	     (when (and (equal (early-method-specializers m) specializers)
			(equal (early-method-qualifiers m) qualifiers))
	       (return m))))
	 (new (make-a-method 'standard-method
			     qualifiers
			     arglist
			     specializers
			     initargs
			     ())))
    (when existing (remove-method gf existing))
    (add-method gf new)))

;;;
;;; This is the early version of add-method.  Later this will become a
;;; generic function.  See fix-early-generic-functions which has special
;;; knowledge about add-method.
;;;
(defun add-method (generic-function method)
  (when (not (fsc-instance-p generic-function))
    (error "Early add-method didn't get a funcallable instance."))
  (when (not (and (listp method) (eq (car method) :early-method)))
    (error "Early add-method didn't get an early method."))
  (push method (early-gf-methods generic-function))
  (set-arg-info generic-function :new-method method)
  (unless (assoc (early-gf-name generic-function) *generic-function-fixups*
		 :test #'equal)
    (update-dfun generic-function)))

;;;
;;; This is the early version of remove method.
;;;
(defun remove-method (generic-function method)
  (when (not (fsc-instance-p generic-function))
    (error "Early remove-method didn't get a funcallable instance."))
  (when (not (and (listp method) (eq (car method) :early-method)))
    (error "Early remove-method didn't get an early method."))
  (setf (early-gf-methods generic-function)
	(remove method (early-gf-methods generic-function)))
  (set-arg-info generic-function)
  (unless (assoc (early-gf-name generic-function) *generic-function-fixups*
		 :test #'equal)
    (update-dfun generic-function)))

;;;
;;; And the early version of get-method.
;;;
(defun get-method (generic-function qualifiers specializers
				    &optional (errorp t))
  (if (early-gf-p generic-function)
      (or (dolist (m (early-gf-methods generic-function))
	    (when (and (or (equal (early-method-specializers m nil)
				  specializers)
			   (equal (early-method-specializers m t)
				  specializers))
		       (equal (early-method-qualifiers m) qualifiers))
	      (return m)))
	  (if errorp
	      (error "Can't get early method.")
	      nil))
      (real-get-method generic-function qualifiers specializers errorp)))

(defvar *fegf-debug-p* nil)

(defun fix-early-generic-functions (&optional (noisyp *fegf-debug-p*))
  (let ((accessors nil))
    ;; Rearrange *early-generic-functions* to speed up fix-early-generic-functions.
    (dolist (early-gf-spec *early-generic-functions*)
      (when (every #'early-method-standard-accessor-p
		   (early-gf-methods (gdefinition early-gf-spec)))
	(push early-gf-spec accessors)))
    (dolist (spec (nconc accessors
			 '(accessor-method-slot-name
			   generic-function-methods
			   method-specializers
			   specializerp
			   specializer-type
			   specializer-class
			   slot-definition-location
			   slot-definition-name
			   class-slots
			   gf-arg-info
			   class-precedence-list
			   slot-boundp-using-class
			   (setf slot-value-using-class)
			   slot-value-using-class
			   structure-class-p
			   standard-class-p
			   funcallable-standard-class-p
			   specializerp)))
      (setq *early-generic-functions* 
	    (cons spec (delete spec *early-generic-functions* :test #'equal))))

    (dolist (early-gf-spec *early-generic-functions*)
      (when noisyp (format t "~&~S..." early-gf-spec))
      (let* ((gf (gdefinition early-gf-spec))
	     (methods (mapcar (lambda (early-method)
				  (let ((args (copy-list (fifth early-method))))
				    (setf (fourth args)
					  (early-method-specializers early-method t))
				    (apply #'real-make-a-method args)))
			      (early-gf-methods gf))))
	(setf (generic-function-method-class gf) *the-class-standard-method*)
	(setf (generic-function-method-combination gf) *standard-method-combination*)
	(set-methods gf methods)))
	  
    (dolist (fns *early-functions*)
      (setf (gdefinition (car fns)) (symbol-function (caddr fns))))
      
    (dolist (fixup *generic-function-fixups*)
      (let* ((fspec (car fixup))
	     (gf (gdefinition fspec))
	     (methods (mapcar (lambda (method)
				  (let* ((lambda-list (first method))
					 (specializers (second method))
					 (method-fn-name (third method))
					 (fn-name (or method-fn-name fspec))
					 (fn (symbol-function fn-name))
					 (initargs 
					  (list :function
						(set-function-name
						 (lambda (args next-methods)
						     (declare (ignore next-methods))
						     (apply fn args))
						 `(call ,fn-name)))))
				    (declare (type function fn))
				    (make-a-method 'standard-method
						   ()
						   lambda-list
						   specializers
						   initargs
						   nil)))
			      (cdr fixup))))
	(setf (generic-function-method-class gf) *the-class-standard-method*)
	(setf (generic-function-method-combination gf) *standard-method-combination*)
	(set-methods gf methods)))))


;;;
;;; parse-defmethod is used by defmethod to parse the &rest argument into
;;; the 'real' arguments.  This is where the syntax of defmethod is really
;;; implemented.
;;; 
(defun parse-defmethod (cdr-of-form)
  (declare (list cdr-of-form))
  (let ((name (pop cdr-of-form))
	(qualifiers ())
	(spec-ll ()))
    (loop (if (and (car cdr-of-form) (atom (car cdr-of-form)))
	      (push (pop cdr-of-form) qualifiers)
	      (return (setq qualifiers (nreverse qualifiers)))))
    (setq spec-ll (pop cdr-of-form))
    (values name qualifiers spec-ll cdr-of-form)))

(defun parse-specializers (specializers)
  (declare (list specializers))
  (flet ((parse (spec)
	   (let ((result (specializer-from-type spec)))
	     (if (specializerp result)
		 result
		 (if (symbolp spec)
		     (error "~S used as a specializer,~%~
                             but is not the name of a class."
			    spec)
		     (error "~S is not a legal specializer." spec))))))
    (mapcar #'parse specializers)))

(defun unparse-specializers (specializers-or-method)
  (if (listp specializers-or-method)
      (flet ((unparse (spec)
	       (if (specializerp spec)
                   (let ((type (specializer-type spec)))
                     (if (and (consp type)
                              (eq (car type) 'class))
                         (let* ((class (cadr type))
                                (class-name (class-name class)))
                           (if (eq class (find-class class-name nil))
                               class-name
                               type))
                         type))
		   (error "~S is not a legal specializer." spec))))
	(mapcar #'unparse specializers-or-method))
      (unparse-specializers (method-specializers specializers-or-method))))

(defun parse-method-or-spec (spec &optional (errorp t))
  ;;(declare (values generic-function method method-name))
  (let (gf method name temp)
    (if (method-p spec)	
	(setq method spec
	      gf (method-generic-function method)
	      temp (and gf (generic-function-name gf))
	      name (if temp
		       (intern-function-name
			 (make-method-spec temp
					   (method-qualifiers method)
					   (unparse-specializers
					     (method-specializers method))))
		       (make-symbol (format nil "~S" method))))
	(multiple-value-bind (gf-spec quals specls)
	    (parse-defmethod spec)
	  (and (setq gf (and (or errorp (gboundp gf-spec))
			     (gdefinition gf-spec)))
	       (let ((nreq (compute-discriminating-function-arglist-info gf)))
		 (setq specls (append (parse-specializers specls)
				      (make-list (- nreq (length specls))
						 :initial-element
						 *the-class-t*)))
		 (and 
		   (setq method (get-method gf quals specls errorp))
		   (setq name
			 (intern-function-name (make-method-spec gf-spec
								 quals
								 specls))))))))
    (values gf method name)))



(defun extract-parameters (specialized-lambda-list)
  (multiple-value-bind (parameters ignore1 ignore2)
      (parse-specialized-lambda-list specialized-lambda-list)
    (declare (ignore ignore1 ignore2))
    parameters))

(defun extract-lambda-list (specialized-lambda-list)
  (multiple-value-bind (ignore1 lambda-list ignore2)
      (parse-specialized-lambda-list specialized-lambda-list)
    (declare (ignore ignore1 ignore2))
    lambda-list))

(defun extract-specializer-names (specialized-lambda-list)
  (multiple-value-bind (ignore1 ignore2 specializers)
      (parse-specialized-lambda-list specialized-lambda-list)
    (declare (ignore ignore1 ignore2))
    specializers))

(defun extract-required-parameters (specialized-lambda-list)
  (multiple-value-bind (ignore1 ignore2 ignore3 required-parameters)
      (parse-specialized-lambda-list specialized-lambda-list)
    (declare (ignore ignore1 ignore2 ignore3))
    required-parameters))

(defun parse-specialized-lambda-list (arglist &optional post-keyword)
  ;;(declare (values parameters lambda-list specializers required-parameters))
  (check-type arglist list)
  (let ((arg (car arglist)))
    (cond ((null arglist) (values nil nil nil nil))
	  ((eq arg '&aux)
	   (values nil arglist nil))
	  ((memq arg lambda-list-keywords)
	   (unless (memq arg '(&optional &rest &key &allow-other-keys &aux))
	     ;; Warn about non-standard lambda-list-keywords, but then
	     ;; go on to treat them like a standard lambda-list-keyword
	     ;; what with the warning its probably ok.
	     (warn "Unrecognized lambda-list keyword ~S in arglist.~%~
                    Assuming that the symbols following it are parameters,~%~
                    and not allowing any parameter specializers to follow~%~
                    to follow it."
		   arg))
	   ;; When we are at a lambda-list-keyword, the parameters don't
	   ;; include the lambda-list-keyword; the lambda-list does include
	   ;; the lambda-list-keyword; and no specializers are allowed to
	   ;; follow the lambda-list-keywords (at least for now).
	   (multiple-value-bind (parameters lambda-list)
	       (parse-specialized-lambda-list (cdr arglist) t)
	     (values parameters
		     (cons arg lambda-list)
		     ()
		     ())))
	  (post-keyword
	   ;; After a lambda-list-keyword there can be no specializers.
	   (multiple-value-bind (parameters lambda-list)
	       (parse-specialized-lambda-list (cdr arglist) t)	       
	     (values (cons (if (listp arg) (car arg) arg) parameters)
		     (cons arg lambda-list)
		     ()
		     ())))
	  (t
	   (multiple-value-bind (parameters lambda-list specializers required)
	       (parse-specialized-lambda-list (cdr arglist))
	     (values (cons (if (listp arg) (car arg) arg) parameters)
		     (cons (if (listp arg) (car arg) arg) lambda-list)
		     (cons (if (and (listp arg) (listp (cdr arg))) (cadr arg) t) specializers)
		     (cons (if (listp arg) (car arg) arg) required)))))))


(eval-when (load eval)
  (setq *boot-state* 'early))


#-cmu ;; CMUCL Has a real symbol-macrolet
(progn
(defmacro symbol-macrolet (bindings &body body &environment env)
  (let ((specs (mapcar (lambda (binding)
			 (when (or (si::specialp (car binding)) (constantp (car binding)))
			   (error 'program-error));FIXME
			 (list (car binding)
			       (variable-lexical-p (car binding) env)
			       (cadr binding)))
		       bindings)))
    (do ((l body (cdr l)))
	((or (atom (car l)) (not (eq (caar l) 'declare))))
      (dolist (ll (cdar l))
	(when (and (consp ll) (eq (car ll) 'special) (some (lambda (x) (assoc x bindings)) (cdr ll)))
	  (error 'program-error))));FIXME
    (walk-form `(let nil ,@body)
	       env
	       (lambda (f c e) (expand-symbol-macrolet-internal specs f c e)))))

(defun expand-symbol-macrolet-internal (specs form context env)
  (let ((entry nil))
    (cond ((not (eq context :eval)) form)
	  ((symbolp form)
	   (if (and (setq entry (assoc form specs))
		    (eq (cadr entry) (variable-lexical-p form env)))
	       (caddr entry)
	       form))
	  ((not (listp form)) form)
	  ((member (car form) '(setq setf psetq psetf));FIXME -- other setf forms?  or real symbol-macrolet
	   ;; Have to be careful.  We must only convert the form to a SETF
	   ;; form when we convert one of the 'logical' variables to a form
	   ;; Otherwise we will get looping in implementations where setf
	   ;; is a macro which expands into setq.
	   (let ((kind (car form)))
	     (labels ((scan-setf (tail)
				 (when tail
				   (walker::relist*
				    tail
				    (if (and (setq entry (assoc (car tail) specs))
					     (eq (cadr entry) (variable-lexical-p (car tail) env)))
					(progn (setq kind (if (eq kind 'psetf) kind 'setf))
					       (caddr entry))
				      (car tail))
				    (cadr tail)
				    (scan-setf (cddr tail))))))
	       (let (new-tail)
		 (setq new-tail (scan-setf (cdr form)))
		 (let ((r (walker::recons form kind new-tail))) (if (eq (car form) 'psetq) `(progn ,r nil) r))))))
	  ((eq (car form) 'multiple-value-setq)
	   (let* ((vars (cadr form))
		  (vars (mapcar (lambda (x) 
				  (let ((y (assoc x specs)))
				    (if (and y (eq (cadr y) (variable-lexical-p x env))) (walker::macroexpand-all (caddr y) env) x))) vars))
		  (gensyms (mapcar (lambda (i) (declare (ignore i)) (gensym)) vars))
		  (pls (mapcan 'cdr (copy-tree (remove-if-not 'consp vars))))
		  (psyms (mapcar (lambda (x) (declare (ignore x)) (gensym)) pls))
		  (ppsyms psyms)
		  (vars (mapcar (lambda (x) (if (atom x) x (cons (car x) (mapcar (lambda (x) (declare (ignore x)) (pop ppsyms)) (cdr x))))) vars)))
	     `(let* ,(mapcar 'list psyms pls)
		(multiple-value-bind 
		 ,gensyms 
		 ,(caddr form)
		 .,(reverse (mapcar (lambda (v g) `(setf ,v ,g)) vars gensyms))))))
	  ((eq (car form) 'restart-case)
	   (let* ((nf (cadr form))
		  (nf (let ((y (assoc nf specs)))
			(if (and y (eq (cadr y) (variable-lexical-p nf env))) (walker::macroexpand-all (caddr y) env) nf))))
	     (if (eq nf (cadr form)) form
	       `(,(car form) ,nf ,@(cddr form)))))
	  ((eq (car form) 'symbol-macrolet) (walker::macroexpand-all form env));(macroexpand form env)
	  ((eq 'si::macro (cadar (assoc (car form) env :key 'car)));(not (eq (car form) 'symbol-macrolet))
	   (let ((kind (car form)))
	     (labels ((scan-setf (tail)
				 (when tail
				   (walker::relist*
				    tail
				    (if (and (setq entry (assoc (car tail) specs))
					     (eq (cadr entry) (variable-lexical-p (car tail) env)))
					(caddr entry)
				      (car tail))
				    (scan-setf (cdr tail))))))
	       (walker::recons form kind (scan-setf (cdr form))))))
	  (form)))))

;; (defun expand-symbol-macrolet-internal (specs form context env)
;;   (let ((entry nil))
;;     (cond ((not (eq context :eval)) form)
;; 	  ((symbolp form)
;; 	   (if (and (setq entry (assoc form specs))
;; 		    (eq (cadr entry) (variable-lexical-p form env)))
;; 	       (caddr entry)
;; 	       form))
;; 	  ((not (listp form)) form)
;; 	  ((member (car form) '(setq setf psetq psetf));FIXME -- other setf forms?  or real symbol-macrolet
;; 	   ;; Have to be careful.  We must only convert the form to a SETF
;; 	   ;; form when we convert one of the 'logical' variables to a form
;; 	   ;; Otherwise we will get looping in implementations where setf
;; 	   ;; is a macro which expands into setq.
;; 	   (let ((kind (car form)))
;; 	     (labels ((scan-setf (tail)
;; 				 (if (null tail)
;; 				     nil
;; 				   (walker::relist*
;; 				    tail
;; 				    (if (and (setq entry (assoc (car tail) specs))
;; 					     (eq (cadr entry) (variable-lexical-p (car tail) env)))
;; 					(progn (setq kind (if (eq kind 'psetf) kind 'setf))
;; 					       (caddr entry))
;; 				      (car tail))
;; 				    (cadr tail)
;; 				    (scan-setf (cddr tail))))))
;; 	       (let (new-tail)
;; 		 (setq new-tail (scan-setf (cdr form)))
;; 		 (let ((r (walker::recons form kind new-tail))) (if (eq (car form) 'psetq) `(progn ,r nil) r))))))
;; 	  ((eq (car form) 'multiple-value-setq)
;; 	   (let* ((vars (cadr form))
;; 		  (vars (mapcar (lambda (x) 
;; 				  (let ((y (assoc x specs)))
;; 				    (if (and y (eq (cadr y) (variable-lexical-p x env))) (walker::macroexpand-all (caddr y) env) x))) vars))
;; 		  (gensyms (mapcar (lambda (i) (declare (ignore i)) (gensym)) vars))
;; 		  (pls (mapcan 'cdr (copy-tree (remove-if-not 'consp vars))))
;; 		  (psyms (mapcar (lambda (x) (declare (ignore x)) (gensym)) pls))
;; 		  (ppsyms psyms)
;; 		  (vars (mapcar (lambda (x) (if (atom x) x (cons (car x) (mapcar (lambda (x) (declare (ignore x)) (pop ppsyms)) (cdr x))))) vars)))
;; 	     `(let* ,(mapcar 'list psyms pls)
;; 		(multiple-value-bind 
;; 		 ,gensyms 
;; 		 ,(caddr form)
;; 		 .,(reverse (mapcar (lambda (v g) `(setf ,v ,g)) vars gensyms))))))
;; 	  ((eq (car form) 'conditions::restart-case)
;; 	   (let* ((nf (cadr form))
;; 		  (nf (let ((y (assoc nf specs)))
;; 			(if (and y (eq (cadr y) (variable-lexical-p nf env))) (walker::macroexpand-all (caddr y) env) nf))))
;; 	     (if (eq nf (cadr form)) form
;; 	       `(,(car form) ,nf ,@(cddr form)))))
;; 	  (t form)))))

(defmacro with-slots (slots instance &body body)
  (declare (optimize (safety 2)))
  (let ((in (gensym)))
    `(let ((,in ,instance))
       (declare (ignorable ,in))
       ,@(let ((instance (if (and (consp instance) (eq (car instance) 'the))
                             (third instance)
                             instance)))
	   (and (symbolp instance)
                `((declare (variable-rebinding ,in ,instance)))))
       ,in
       (symbol-macrolet ,(mapcar (lambda (slot-entry)
				     (let ((variable-name 
					    (if (symbolp slot-entry)
						slot-entry
						(car slot-entry)))
					   (slot-name
					    (if (symbolp slot-entry)
						slot-entry
						(cadr slot-entry))))
				       `(,variable-name
					  (slot-value ,in ',slot-name))))
				 slots)
			,@body))))


(defmacro with-accessors (slots instance &body body)
  (let ((in (gensym)))
    `(let ((,in ,instance))
       (declare (ignorable ,in))
       ,@(let ((instance (if (and (consp instance) (eq (car instance) 'the))
                             (third instance)
                             instance)))
	   (and (symbolp instance)
                `((declare (variable-rebinding ,in ,instance)))))
       ,in
       (symbol-macrolet ,(mapcar (lambda (slot-entry)
				   (let ((variable-name (car slot-entry))
					 (accessor-name (cadr slot-entry)))
				     `(,variable-name
				        (,accessor-name ,in))))
			       slots)
          ,@body))))



