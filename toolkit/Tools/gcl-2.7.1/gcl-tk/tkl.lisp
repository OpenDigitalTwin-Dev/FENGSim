;; Copyright (C) 1994 W. Schelter
;; Copyright (C) 2024 Camm Maguire

;; This file is part of GNU Common Lisp, herein referred to as GCL
;;
;; GCL is free software; you can redistribute it and/or modify it under
;;  the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
;; the Free Software Foundation; either version 2, or (at your option)
;; any later version.
;; 
;; GCL is distributed in the hope that it will be useful, but WITHOUT
;; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
;; FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
;; License for more details.
;; 



(eval-when (load eval compile)
(in-package "TK")
)

(eval-when (compile) 
(proclaim '(ftype (function (t fixnum fixnum) fixnum) set-message-header
		  get-number-string))
(proclaim '(ftype (function (t t fixnum) t) store-circle))
(proclaim '(ftype (function (t fixnum) t) get-circle))
(proclaim '(ftype (function (t fixnum fixnum fixnum) fixnum)
		  push-number-string))
)

(defvar *tk-package* (find-package "TK"))

(eval-when (compile eval load)

(defconstant *header* '(magic1 magic2 type flag body-length nil nil msg-index nil nil))

;;enum print_arglist_codes {..};
(defvar *print-arglist-codes*
  '(
    normal
    no_leading_space
    join_follows
    end_join
    begin_join
    begin_join_no_leading_space
    no_quote
    no_quote_no_leading_space
    no_quote_downcase
    no_quotes_and_no_leading_space

    ))

(defconstant *mtypes*
  '( m_not_used
     m_create_command
     m_reply
     m_call
     m_tcl_command
     m_tcl_command_wait_response
     m_tcl_clear_connection  
     m_tcl_link_text_variable
     m_set_lisp_loc
     m_tcl_set_text_variable
     m_tcl_unlink_text_variable
     m_lisp_eval
     m_lisp_eval_wait_response
     ))

(defconstant *magic1* #\)
(defconstant *magic2* #\A)


(defvar *some-fixnums* (make-array 3 :element-type 'fixnum))
(defmacro msg-index () `(the fixnum
			    (aref (the (array fixnum) *some-fixnums*) 0)))
;;; (defmacro safe-car (x)
;;;   (cond ((symbolp x) `(if (consp ,x) (car ,x) (if (null ,x) nil
;;; 						(not-a-cons ,x))))
;;; 	(t (let ((sym (gensym)))
;;; 	     `(let ((,sym ,x))
;;; 		(safe-car ,sym))))))
;;; (defmacro safe-cdr (x)
;;;   (cond ((symbolp x) `(if (consp ,x) (cdr ,x) (if (null ,x) nil
;;; 						(not-a-cons ,x))))
;;; 	(t (let ((sym (gensym)))
;;; 	     `(let ((,sym ,x))
;;;		(safe-cdr ,sym))))))


(defun desetq-consp-check (val)
  (or (consp val) (error "~a is not a cons" val)))

(defun desetq1 (form val)
  (cond ((symbolp form)
	 (cond (form			;(push form *desetq-binds*)
		`(setf ,form ,val))))
	((consp form)
	 `(progn
	    (desetq-consp-check ,val)
	    ,(desetq1 (car form) `(car ,val))
	    ,@ (if (consp (cdr form))
		   (list(desetq1 (cdr form) `(cdr ,val)))
		 (and (cdr form) `((setf ,(cdr form) (cdr ,val)))))))
	(t (error ""))))

(defmacro desetq (form val)
  (cond ((atom val) (desetq1 form val))
	(t (let ((value (gensym)))
	     `(let ((,value ,val)) , (desetq1 form value))))))
(defmacro while (test &body body)
  `(sloop while ,test do ,@ body))

)

;(defmacro nth-value (n form)
;  `(multiple-value-bind ,(make-list (+ n 1) :initial-element 'a) ,form  a))

(defvar *tk-command* nil)

(defvar *debugging* nil)
(defvar *break-on-errors* nil)

(defvar *tk-connection* nil )

;; array of functions to be invoked from lisp.
(defvar *call-backs* (make-array 20 :fill-pointer 0 :adjustable t ))

;;array of message half read. Ie read header but not body.
(defvar *pending* nil)

;;circular array for replies,requests esp for debugging
;; replies is used for getting replies.
(defvar *replies* (make-array (expt 2 7)) "circle of replies to requests in *requests*")

;; these are strings
(defvar *requests* (make-array (expt 2 7)))

;; these are lisp forms
(defvar *request-forms* (make-array 40))


(defvar *read-buffer* (make-array 400 :element-type 'standard-char
				  :fill-pointer 0 :static t))

(defvar *text-variable-locations*
  (make-array 10 :fill-pointer 0 :adjustable t))




(defmacro pos (flag lis)
  (or
   (member flag (symbol-value lis))
   (error "~a is not in ~a" flag lis))
  (position flag (symbol-value lis)))

  



;;; (defun p1 (a &aux tem)
;;;   ;;Used for putting  A into a string for sending a command to TK
;;;   (cond
;;;     ((and (symbolp a) (setq tem (get a 'tk-print)))
;;;      (format *tk-command* tem))
;;;     ((keywordp a)
;;;      (format *tk-command* "-~(~a~)" a))
;;;     ((numberp a)
;;;      (format *tk-command* "~a" a))
;;;     ((stringp a)
;;;      (format *tk-command* "\"~a\"" a))
;;;     ((and (consp a)(eq (car a) 'a))
;;;      (format *tk-command* "~a" (cdr a)))
;;;     ((and (consp a)(eq (car a) 'd))
;;;      (format *tk-command* "~(~a~)" (cdr a)))
;;;     ((and (symbolp a)
;;; 	  (eql (aref (symbol-name a) 0)
;;; 	       #\.))
;;;      (format *tk-command* "~(~a~)" a))
;;;    (t (error "unrecognized term ~s" a))))


(defvar *command-strings*
  (sloop for i below 2 collect
       (make-array 200 :element-type 'standard-char :fill-pointer 0 :adjustable t)))

(defvar *string-streams* (list (make-string-input-stream "") (make-string-input-stream "")))

(defmacro with-tk-command (&body body)
  `(let ((tk-command (grab-tk-command)) (*command-strings* *command-strings*))
     (declare (string tk-command))
     ,@ body))

(defun grab-tk-command( &aux x)
  ;; keep a list of available *command-strings* and grab one
  (cond
   ((cdr *command-strings*))
   (t 
    (setq x (list (make-array 70
			      :element-type 'character
			      :fill-pointer 0 :adjustable t))
	  )
    (or *command-strings* (error "how??"))

    (setq *command-strings* (nconc *command-strings* x))))
  (let ((x (car *command-strings*)))
    (setq  *command-strings* (cdr *command-strings*))
    (setf (fill-pointer x ) #.(length *header*))
    x
    ))

(defun print-to-string (str x code)
  (cond ((consp x)
	 (cond ((eq (car x) 'a)
		(setq x (cdr x)
		      code (pos no_quote *print-arglist-codes*)))
	       ((eq (car x) 'd)
		(setq x (cdr x)
		      code (pos no_quote_downcase *print-arglist-codes*)))
	       (t (error "bad arg ~a" x)))))
  (while (null (si::print-to-string1 str x code))
    (cond ((typep x 'bignum)
	   (setq x (format nil "~a" x)))
	  (t (setq str (adjust-array str
				     (the fixnum
					  (+ (the fixnum
						  (array-total-size str))
					     (the fixnum
						  (+ 
						   (if (stringp x)
						       (length (the string x))
						     0)
					      70))))
				     :fill-pointer (fill-pointer str)
				     :element-type 'string-char)))))
  str)

(defmacro pp (x code)
  (let ((u `(pos ,code *print-arglist-codes*)))
  `(print-to-string tk-command ,x ,u)))

(defun print-arglist (to-string l &aux v in-join x)
;;      (sloop for v in l do (p :| | v))
  (while l
    (setq v (cdr l))
    (setq x (car l))
    (cond
     ((eql (car v) ': )
      (print-to-string to-string x
		       (if in-join
			   (pos join_follows *print-arglist-codes*)
			 (pos begin_join *print-arglist-codes*)))
      (setq in-join t)
      (setq v (cdr v)))
     (in-join
      (print-to-string to-string x (pos end_join *print-arglist-codes*))
      (setq in-join nil))
     (t;; code == (pos normal *print-arglist-codes*)
      (print-to-string to-string x (pos normal *print-arglist-codes*))))

    (setq l v)
    ))
     
(defmacro p (&rest l)
  `(progn ,@ (sloop for v in l collect `(p1 ,v))))

(defvar *send-and-wait* nil "If not nil, then wait for answer and check result")

(defun tk-call (fun &rest l &aux result-type)
  (with-tk-command
   (pp fun no_leading_space)
   (setq result-type (prescan-arglist l nil nil))
   (print-arglist tk-command l)
   (cond (result-type
	  (call-with-result-type tk-command result-type))
	 (t  (send-tcl-cmd *tk-connection* tk-command nil)
	     (values)))))

(defun tk-do (str &rest l &aux )
  (with-tk-command
       (pp str no_quotes_and_no_leading_space)
       ;; leading keyword printed without '-' at beginning.
       (while l
	 (pp (car l) no_quotes_and_no_leading_space)
	 (setq l (cdr l)))
       (call-with-result-type tk-command 'string)))

(defun tk-do-no-wait (str &aux (n (length str)))
  (with-tk-command
   (si::copy-array-portion str  tk-command 0  #.(length *header*) n)
   (setf (fill-pointer tk-command) (the fixnum (+ n  #.(length *header*))))
   (let ()
     (send-tcl-cmd *tk-connection* tk-command nil))))

(defun fsubseq (s &optional (b 0) (e (length s)))
  (make-array (- e b) :element-type (array-element-type s) :displaced-to s :displaced-index-offset b :fill-pointer (- e b)))

(defun send-tcl-cmd (c str send-and-wait )
  ;(notice-text-variables)
  (or send-and-wait (setq send-and-wait *send-and-wait*))
 ; (setq send-and-wait t)
  (vector-push-extend (code-char 0) str)
  (let ((msg-id (set-message-header str
				    (if send-and-wait
					(pos m_tcl_command_wait_response *mtypes*)
				      (pos m_tcl_command *mtypes*))
				    (the fixnum
					 (- (length str)
					    #.(length *header*))))))
    
    (cond (send-and-wait
	   (if *debugging*
	       (store-circle *requests* (fsubseq str #.(length *header*))
			     msg-id))
	   (store-circle *replies* nil  msg-id)
	   (execute-tcl-cmd c str))
	  (t (store-circle *requests* nil msg-id)
	   (write-to-connection c str)))))

  
(defun send-tcl-create-command (c str)
  (vector-push-extend (code-char 0) str)
  (set-message-header str (pos m_create_command *mtypes*)
		      (- (length str) #.(length *header*)))
  (write-to-connection c str))

(defun write-to-connection (con string &aux tem)
  (let* ((*sigusr1* t)
	 ;; dont let us get interrupted while writing!!
	 (n (length string))
	 (fd (caar con))
	 (m 0))
    (declare (Fixnum n m))
    (or con (error "Trying to write to non open connection "))
    (if *debugging* (describe-message string))
    (or (typep fd 'string)
	(error "~a is not a connection" con))
    (setq m (si::our-write fd string n))
    (or (eql m n) (error "Failed to write ~a bytes to file descriptor ~a" n fd))
    (setq tem *sigusr1*)
    ;; a signal at this instruction would not be noticed...since it
    ;; would set *sigusr1* to :received but that would be too late for tem
    ;; since the old value will be popped off the binding stack at the next 'paren'
    )
  (cond ((eq tem :received)
	 (read-and-act nil)))
  t)


(defun coerce-string (a)
  (cond ((stringp a) a)
	((fixnump a) (format nil "~a" a))
	((numberp a) (format nil "~,2f" (float a)))
        ((keywordp a)
	 (format nil "-~(~a~)" a))
	((symbolp a)
	 (format nil "~(~a~)" a))
	(t (error "bad type"))))
;;2 decimals

(defun my-conc (a b)
  (setq a (coerce-string a))
  (setq b (coerce-string b))
  (concatenate 'string a b ))

;; In an arglist   'a : b' <==> (tk-conc a b)
;; eg:   1  : "b" <==> "1b"
;        "c" : "b" <==> "cb"
;        'a  : "b" <==> "ab"
;       '.a  : '.b  <==> ".a.b"
;       ':ab : "b"  <==> "abb"

;;Convenience for concatenating symbols, strings, numbers
;;  (tk-conc '.joe.bill ".frame.list yview " 3) ==> ".joe.bill.frame.list yview 3"
(defun tk-conc (&rest l)
  (declare (:dynamic-extent l))
  (let ((tk-command
	 (make-array 30 :element-type 'standard-char
		     :fill-pointer 0 :adjustable t)))
    (cond ((null l))
	  (t (pp (car l) no_quote_no_leading_space)))
    (setq l (cdr l))
    (while (cdr l)
      (pp (car l) join_follows) (setq l (cdr l)))
    (and l (pp (car l) no_quote_no_leading_space))
    tk-command
    ))


;;; (defun verify-list (l)
;;;   (loop
;;;    (cond ((null l)(return t))
;;; 	 ((consp l) (setq l (cdr l)))
;;; 	 (t (error "not a true list ~s"l)))))

;;; (defun prescan-arglist (l pathname name-caller &aux result-type)
;;;   (let ((v l) tem prev a b  c)
;;;     (verify-list l)
;;;     (sloop while v
;;;        do
;;;        (cond
;;; 	((keywordp (car v))
;;; 	 (setq a (car v))
;;; 	 (setq c (cdr v))
;;; 	 (setq b (car c) c (cadr c))
;;; 	 (cond ((eq a :bind)
;;; 		(cond ((setq tem (cdddr v))
;;; 		       (or (eq (cadr tem) ': )
;;; 			   (setf (car tem)
;;; 				 (tcl-create-command (car tem)
;;; 						     nil 
;;; 						     t))))))
;;; 	       ((eq c ': ))
;;; 	       ((member a'(:yscroll :command
;;; 				    :xscroll
;;; 				    :yscrollcommand
;;; 				    :xscrollcommand
;;; 				    :scrollcommand
;;; 				    ))
;;; 		(cond ((setq tem (cdr v))
;;; 		       (setf (car tem)
;;; 			     (tcl-create-command (car tem)
;;; 						 (or (get a 'command-arg)
						     
;;; 						     (get name-caller
;;; 							  'command-arg))
;;; 						 nil)))))
;;; 	       ((eq (car v) :return)
;;; 		(setf result-type (cadr v))
;;; 		(cond (prev
;;; 		       (setf (cdr prev) (cddr v)))
;;; 		      (t (setf (car v) '(a . ""))
;;; 			 (setf (cdr v) (cddr v)))))
;;; 	       ((eq (car v) :textvariable)
;;; 		(setf (second v) (link-variable b 'string)))
;;; 	       ((member (car v) '(:value :onvalue :offvalue))
;;; 		(let* ((va (get pathname 'variable))
;;; 		       (type (get va 'linked-variable-type))
;;; 		       (fun (cdr (get type
;;; 				 'coercion-functions))))
;;; 		  (or va
;;; 		      (error
;;; 		       "Must specify :variable before :value so that we know the type"))
;;; 		  (or fun (error "No coercion-functions for type ~s" type))
;;; 		  (setf (cadr v) (funcall fun b))))
;;; 	       ((eq (car v) :variable)
;;; 		(let ((va (second v))
;;; 		      (type (cond ((eql name-caller 'checkbutton) 'boolean)
;;; 			     (t 'string))))
;;; 		  (cond ((consp va)
;;; 			 (desetq (type va) va)
;;; 			 (or (symbolp va)
;;; 			     (error "should be :variable (type symbol)"))))
;;; 		  (setf (get pathname 'variable) va)
;;; 		  (setf (second v)
;;; 		      (link-variable   va type))))
;;; 		)))
;;;        (setq prev v)      
;;;        (setq v (cdr v))
;;;        ))
;;;   result-type
;;;   )


(defun prescan-arglist (l pathname name-caller &aux result-type)
  (let ((v l) tem prev a )
;    (verify-list l) ; unnecessary all are from &rest args.
; If pathname supplied, then this should be an alternating list
;; of keywords and values.....
    (sloop while v
       do 	 (setq a (car v))
       (cond
	((keywordp a)
	 (cond
	  ((eq (car v) :return)
	   (setf result-type (cadr v))
	   (cond (prev
		  (setf (cdr prev) (cddr v)))
		 (t (setf (car v) '(a . ""))
		    (setf (cdr v) (cddr v)))))
	  ((setq tem (get a 'prescan-function))
	   (funcall tem a v pathname name-caller)))))
       (setq prev v)
       (setq v (cdr v)))
    result-type))

(eval-when (compile eval load)
(defun set-prescan-function (fun &rest l)
  (dolist (v l) (setf (get v 'prescan-function) fun)))
)
	 
	  
(set-prescan-function 'prescan-bind :bind)
(defun prescan-bind
       (x  v pathname name-caller &aux tem)
      name-caller pathname x
      (cond ((setq tem (cdddr v))
	     (or
	      (keywordp (car tem))
	      (eq (cadr tem) ': )
		 (setf (car tem)
		       (tcl-create-command (car tem)
					   nil 
					   t))))))

(set-prescan-function 'prescan-command :yscroll :command
		      :postcommand
		      :xscroll
		      :yscrollcommand
		      :xscrollcommand
		      :scrollcommand)

(defun prescan-command (x v pathname name-caller &aux tem arg)
  x pathname
  (setq arg (cond (( member v     '(:xscroll
				    :yscrollcommand
				    :xscrollcommand
				    :scrollcommand))
		   
		   'aaaa)
		  ((get name-caller 'command-arg))))
  (cond ((setq tem (cdr v))
	 (cond ((eq (car tem) :return ) :return)
	       (t
		(setf (car tem)
		      (tcl-create-command (car tem) arg nil)))))))
  
(defun prescan-value (a v pathname name-caller)
  a name-caller
  (let* ((va (get pathname ':variable))
	 (type (get va 'linked-variable-type))
	 (fun (cdr (get type
			'coercion-functions))))
    (or va
	(error
	 "Must specify :variable before :value so that we know the type"))
    (or fun (error "No coercion-functions for type ~s" type))
    (setq v (cdr v))
    (if v
	(setf (car v) (funcall fun (car v))))))

(set-prescan-function 'prescan-value :value :onvalue :offvalue)

(set-prescan-function
 #'(lambda (a v pathname name-caller)
     a
     (let ((va (second v))
	   (type (cond ((eql name-caller 'checkbutton) 'boolean)
		       (t 'string))))
       (cond ((consp va)
	      (desetq (type va) va)
	      (or (symbolp va)
		  (error "should be :variable (type symbol)"))))
       (cond (va
	      (setf (get pathname a) va)
	      (setf (second v)
		    (link-variable   va type))))))
 :variable :textvariable)

(defun make-widget-instance (pathname widget)
  ;; ??make these not wait for response unless user is doing debugging..
  (or (symbolp pathname) (error "must give a symbol"))
  #'(lambda ( &rest l &aux result-type (option (car l)))
      (declare (:dynamic-extent l))
      (setq result-type (prescan-arglist l pathname  widget))
      (if (and *break-on-errors* (not result-type))
	  (store-circle *request-forms*
			(cons pathname (copy-list l))
			(msg-index)))
      (with-tk-command
       (pp pathname no_leading_space)
       ;; the leading keyword gets printed with no leading -
       (or (keywordp option)
	   (error "First arg to ~s must be an option keyword not ~s"
		  pathname option ))
       (pp option no_quote)
       (setq l (cdr l))
       ;(print (car l))
       (cond ((and (keywordp (car l))
		   (not (eq option :configure))
		   (not (eq option :config))
		   (not (eq option :itemconfig))
		   (not (eq option :cget))
		   (not (eq option :postscript))
			)
	      (pp (car l) no_quote)
	      (setq l (cdr l))))
       (print-arglist tk-command l)
       (cond (result-type
	      (call-with-result-type tk-command result-type))
	    (t  (send-tcl-cmd *tk-connection* tk-command nil)
		(values))))))

(defmacro def-widget (widget &key (command-arg 'sssss))
  `(eval-when (compile eval load)
    (setf (get ',widget 'command-arg) ',command-arg)
    (defun ,widget (pathname &rest l)(declare (:dynamic-extent l))
      (widget-function ',widget pathname l))))

     
;; comand-arg "asaa" means pass second arg back as string, and others not quoted
  ;; ??make these always wait for response
  ;; since creating a window failure is likely to cause many failures.
(defun widget-function (widget pathname l )
  (or (symbolp pathname)
      (error "First arg to ~s must be a symbol not ~s" widget pathname))
  (if *break-on-errors*
      (store-circle *request-forms* (cons pathname (copy-list l))
		    (msg-index)))
  (prescan-arglist l pathname widget)
  (with-tk-command
   (pp widget no_leading_space)
   (pp pathname normal)
   (print-arglist tk-command l )
   (multiple-value-bind (res success)
			(send-tcl-cmd *tk-connection* tk-command t)
			(if success
			    (setf (symbol-function pathname)
				  (make-widget-instance pathname widget))
			  (error
			   "Cant define ~(~a~) pathnamed ~(~a~): ~a"
			   widget pathname res)))
   pathname))
(def-widget button)
(def-widget listbox)
(def-widget scale :command-arg a)
(def-widget canvas)
(def-widget menu)
(def-widget scrollbar)
(def-widget checkbutton)
(def-widget menubutton)
(def-widget text)
(def-widget entry)
(def-widget message)
(def-widget frame)
(def-widget label)
(def-widget |image create photo|)
(def-widget |image create bitmap|)
(def-widget radiobutton)
(def-widget toplevel)

(defmacro def-control (name &key print-name before)
  (cond ((null print-name )(setq print-name name))
	(t  (setq print-name (cons 'a print-name))))
  `(defun ,name (&rest l)
     ,@ (if before `((,before ',print-name l)))
     (control-function ',print-name l)))

(defun call-with-result-type (tk-command result-type)
  (multiple-value-bind
   (res suc)
   (send-tcl-cmd *tk-connection* tk-command t)
   (values (if result-type (coerce-result res result-type) res)
	   suc)))

(defun control-function (name l &aux result-type)
      ;(store-circle *request-forms* (cons name l) (msg-index))
      (setq result-type (prescan-arglist l nil name))
      (with-tk-command
       (pp name normal)
       ;; leading keyword printed without '-' at beginning. 
       (cond ((keywordp (car l))
	      (pp (car l) no_quote)
	      (setq l (cdr l))))
       (print-arglist tk-command l)
       (call-with-result-type tk-command result-type)))


(dolist (v
  '( |%%| |%#| |%a| |%b| |%c| |%d| |%f| |%h| |%k| |%m| |%o| |%p| |%s| |%t|
     |%v| |%w| |%x| |%y| |%A| |%B| |%D| |%E| |%K| |%N| |%R| |%S| |%T| |%W| |%X| |%Y|))
  (progn   (setf (get v 'event-symbol)
		 (symbol-name v))
	   (or (member v '(|%d| |%m| |%p| |%K| ;|%W|
			   |%A|))
	       (setf (get v 'event-symbol)
		     (cons (get v 'event-symbol) 'fixnum )))))

(defvar *percent-symbols-used* nil)
(defun get-per-cent-symbols (expr)
  (cond ((atom expr)
	 (and (symbolp expr) (get expr 'event-symbol)
	      (pushnew expr *percent-symbols-used*)))
	(t (get-per-cent-symbols (car expr))
	   (setq expr (cdr expr))
	   (get-per-cent-symbols expr))))


(defun reserve-call-back ( &aux ind)
  (setq ind (fill-pointer *call-backs*))
  (vector-push-extend nil *call-backs* )
  ind)

;; The command arg:
;; For bind windowSpec SEQUENCE COMMAND
;;  COMMAND is called when the event SEQUENCE occurs to windowSpec.
;;    If COMMAND is a symbol or satisfies (functionp COMMAND), then
;;  it will be funcalled.   The number of args supplied in this
;;  case is determined by the widget... for example a COMMAND for the
;;  scale widget will be supplied exactly 1 argument.
;;    If COMMAND is a string then this will be passed to the graphics
;;  interpreter with no change, 
;;  This allows invoking of builtin functionality, without bothering the lisp process.
;;    If COMMAND is a lisp expression to eval, and it may reference
;;  details of the event via the % constructs eg:  %K refers to the keysym
;;  of the key pressed (case of BIND only).   A function whose body is the
;;  form, will actually be  constructed which takes as args all the % variables
;;  actually appearing in the form.  The body of the function will be the form.
;;  Thus (print (list |%w| %W) would turn into #'(lambda(|%w| %W) (print (list |%w| %W)))
;;  and when invoked it would be supplied with the correct args.  

(defvar *arglist* nil)
(defun tcl-create-command (command  arg-data allow-percent-data)
  (with-tk-command
   (cond ((or (null command) (equal command ""))
	  (return-from tcl-create-command ""))
	 ((stringp command)
	  (return-from tcl-create-command command)))
   (let (*percent-symbols-used* tem ans  name ind)
     (setq ind  (reserve-call-back))
     (setq name (format nil "callback_~d" ind))
     ;; install in tk the knowledge that callback_ind will call back to here.
     ;; and tell it arg types expected.
     ;; the percent commands are handled differently
     (push-number-string tk-command ind #.(length *header*) 3)
     (setf (fill-pointer tk-command) #.(+ (length *header*) 3))
     (if arg-data (pp arg-data no_leading_space))
     (send-tcl-create-command *tk-connection* tk-command)
     (if (and arg-data allow-percent-data) (error "arg data and percent data not allowed"))
     (cond ((or (symbolp command)
		(functionp command)))
	   (allow-percent-data
	    (get-per-cent-symbols command)
	    (and *percent-symbols-used* (setq ans ""))
	    (sloop for v in *percent-symbols-used* 
	       do (setq tem (get v 'event-symbol))
	       (cond ((stringp tem)
		      (setq ans (format nil "~a \"~a\"" ans tem)))
		     ((eql (cdr tem) 'fixnum)
		      (setq ans (format nil "~a ~a" ans (car tem))))
		     (t (error "bad arg"))))
	    (if ans (setq ans (concatenate 'string "{(" ans ")}")))
	    (setq command (eval `(lambda ,*percent-symbols-used* ,command)))
	    (if ans (setq name (concatenate 'string "{"name " " ans"}"))))
	   (t (setq command (eval `(lambda (&rest *arglist*) ,command)))))
     (setf (aref *call-backs* ind)  command)
     ;; the command must NOT appear as "{[...]}" or it will be eval'd. 
     (cons 'a name)
     )))
   
(defun bind (window-spec &optional sequence command type)
  "command may be a function name, or an expression which
 may involve occurrences of elements of *percent-symbols*
 The expression will be evaluated in an enviroment in which
 each of the % symbols is bound to the value of the corresponding
 event value obtained from TK."
  (cond ((equal sequence :return)
	 (setq sequence nil)
	 (setq command nil)))
  (cond ((equal command :return)
	 (or (eq type 'string)
	     (tkerror "bind only returns type string"))
	 (setq command nil))
	(command
	 (setq command  (tcl-create-command command nil t))))
  (with-tk-command
   (pp 'bind no_leading_space)
   (pp window-spec normal)
   (and sequence (pp sequence normal))
   (and command (pp command normal))
   (send-tcl-cmd *tk-connection* tk-command (or (null sequence)(null command)))))

(defmacro tk-connection-fd (x) `(caar ,x))

(def-control after)
(def-control exit)
(def-control lower)
(def-control place)
(def-control send)
(def-control tkvars)
(def-control winfo)
(def-control focus)
(def-control option)
(def-control raise)
(def-control tk)
;; problem on waiting.  Waiting for dialog to kill self
;; wont work because the wait blocks even messages which go
;; to say to kill...
;; must use
;; (grab :set :global .fo)
;; and sometimes the gcltkaux gets blocked and cant accept input when
;; in grabbed state...
(def-control tkwait)
(def-control wm)
(def-control destroy :before destroy-aux)
(def-control grab)
(def-control pack)
(def-control selection)
(def-control tkerror)
(def-control update)
(def-control tk-listbox-single-select :print-name "tk_listboxSingleSelect")
(def-control tk-menu-bar :print-name "tk_menuBar")
(def-control tk-dialog :print-name "tk_dialog")
(def-control get_tag_range)

(def-control lsearch)
(def-control lindex)


(defun tk-wait-til-exists (win)
  (tk-do (tk-conc "if ([winfo exists " win " ]) { } else {tkwait visibility " win "}")))

(defun destroy-aux (name  l)
  name
  (dolist (v l)
	  (cond ((stringp v))
		((symbolp v) 
		 (dolist (prop '(:variable :textvariable))
			 (remprop v prop))
		 (fmakunbound v)
		 )
		(t (error "not a pathname : ~s" v))))
	  
  )

(defvar *default-timeout* (* 100 internal-time-units-per-second))

(defun execute-tcl-cmd (connection cmd)
  (let  (id tem (time *default-timeout*))
    (declare (fixnum  time))
    (setq id (get-number-string cmd  (pos msg-index *header*) 3))
    (store-circle *replies* nil  id)
    (write-to-connection connection cmd)
    (loop
     (cond ((setq tem (get-circle *replies* id))
	    (cond ((or (car tem) (null *break-on-errors*))
		   (return-from execute-tcl-cmd  (values (cdr tem) (car tem))))
		  (t (cerror "Type :r to continue" "Cmd failed: ~a : ~a "
			     (subseq cmd (length *header*)
				    (- (length cmd) 1)
				    )
			    (cdr tem))
		     (return (cdr tem))
		     ))))
     (cond ((> (si::check-state-input
		(tk-connection-fd connection) 10) 0)
	    (read-and-act id)
	    ))
     (setq time (- time 10))
     (cond ((< time 0)
	    (cerror ":r resumes waiting for *default-timeout*"
		    "Did not get a reply for cmd ~a" cmd)
	    (setq time *default-timeout*)
	    )))))

(defun push-number-string (string number ind  bytes )
  (declare (fixnum ind number bytes))
  ;; a number #xabcdef is stored "<ef><cd><ab>" where <ef> is (code-char #xef)
  (declare (string string))
  (declare (fixnum  number bytes ))
  (sloop while (>= bytes 1) do
     (setf (aref string ind)
	   (the character (code-char
				  (the fixnum(logand number 255)))))
     (setq ind (+ ind 1))
     (setq bytes (- bytes 1))
;     (setq number (* number 256))
     (setq number (ash number -8))
     nil))

(defun get-number-string (string  start  bytes &aux (number 0))
  ;; a number #xabcdef is stored "<ef><cd><ab>" where <ef> is (code-char #xef)
  (declare (string string))
  (declare (fixnum  number bytes start))
  (setq start (+ start (the fixnum (- bytes 1))))
  (sloop while (>= bytes 1) do
     (setq number (+ number (char-code (aref string start))))
     (setq start (- start 1) bytes (- bytes 1))
     (cond ((> bytes 0) (setq number (ash number 8)))
	   (t (return number)))))


(defun quit () (tkdisconnect) (bye))

(defun debugging (x)
  (setq *debugging* x))
	
(defmacro dformat (&rest l)
  `(if *debugging* (dformat1 ,@l)))
(defun dformat1 (&rest l)
  (declare (:dynamic-extent l))
  (format *debug-io* "~%Lisp:")
  (apply 'format *debug-io* l))

(defvar *sigusr1* nil)
;;??NOTE NOTE we need to make it so that if doing code inside an interrupt,
;;then we do NOT do a gc for relocatable.   This will kill US.
;;One hack would be that if relocatable is low or cant be grown.. then
;;we just set a flag which says run our sigusr1 code at the next cons...
;;and dont do anything here.  Actually we can always grow relocatable via sbrk,
;;so i think it is ok.....??......

(defun system::sigusr1-interrupt (x)
  x
  (cond (*sigusr1*
	 (setq *sigusr1* :received))
	(*tk-connection*
	 (let ((*sigusr1* t))
	   (dformat "Received SIGUSR1. ~a"
		    (if (> (si::check-state-input 
			    (tk-connection-fd *tk-connection*) 0) 0) ""
		      "No Data left there."))
	   ;; we put 4 here to wait for a bit just in case
	   ;; data comes
	   (si::check-state-input 
			    (tk-connection-fd *tk-connection*) 4 )
	   (read-and-act nil)))))
(setf (symbol-function 'si::SIGIO-INTERRUPT) (symbol-function 'si::sigusr1-interrupt))


(defun store-circle (ar reply id)
  (declare (type (array t) ar)
	   (fixnum id))
  (setf (aref ar (the fixnum (mod id (length ar)))) reply))

(defun get-circle (ar  id)
  (declare (type (array t) ar)
	   (fixnum id))
  (aref ar (the fixnum (mod id (length ar)))))

(defun decode-response (str &aux reply-from )
  (setq reply-from (get-number-string str
			      #.(+ 1 (length *header*))
			      3))
  (values
   (fsubseq str #.(+ 4 (length *header*)))
   (eql (aref str #.(+ 1 (length *header*))) #\0)
   reply-from
   (get-circle *requests* reply-from)))

(defun describe-message (vec)

  (let ((body-length (get-number-string vec  (pos body-length *header*) 3))
	(msg-index (get-number-string vec  (pos msg-index *header*) 3))
	(mtype (nth (char-code (aref vec (pos type *header*))) *mtypes*))
	success from-id requ
	)
    (format t "~%Msg-id=~a, type=~a, leng=~a, " msg-index mtype body-length)
    (case mtype
      (m_reply
       (setq from-id (get-number-string vec #.(+ 1  (length *header*))
					3))
       (setq success (eql (aref vec #.(+ 0  (length *header*)))
			  #\0))
       (setq requ (get-circle *requests* from-id))
       (format t "result-code=~a[bod:~s](form msg ~a)[hdr:~s]"
	       success
	       (subseq vec #.(+ 4 (length *header*)))
	       from-id
		       (subseq vec 0 (length *header*))
	       )
       )
      ((m_create_command m_call
         m_lisp_eval
	m_lisp_eval_wait_response)
       (let ((islot (get-number-string vec #.(+ 0 (length *header*)) 3)))
	 (format t "islot=~a(callback_~a), arglist=~s" islot  islot
		 (subseq vec #.(+ 3 (length *header*))))))
      ((m_tcl_command m_tcl_command_wait_response 
		      M_TCL_CLEAR_CONNECTION
		      )
       (format t "body=[~a]"  (subseq vec (length *header*)) ))
      ((m_tcl_set_text_variable)
       (let* ((bod (subseq vec (length *header*)))
	      (end (position (code-char 0) bod))
	      (var (subseq bod 0 end)))
	 (format t "name=~s,val=[~a],body=" var (subseq bod (+ 1 end)
							(- (length bod) 1))
		 bod)))
      ((m_tcl_link_text_variable
	m_tcl_unlink_text_variable
	m_set_lisp_loc)

       (let (var (islot (get-number-string vec #.(+ 0 (length *header*)) 3)))
	 (format t "array_slot=~a,name=~s,type=~s body=[~a]" islot
		 (setq var (aref *text-variable-locations* islot))
		 (get var 'linked-variable-type)
		 (subseq vec #.(+ 3 (length *header*))))))
      
      (otherwise (error "unknown message type ~a [~s]" mtype vec )))))

(defun clear-tk-connection ()
  ;; flush both sides of connection and discard any partial command.
  (cond
   (*tk-connection*
    (si::clear-connection-state (car (car *tk-connection*)))
    (setq *pending* nil)
    (with-tk-command
     (set-message-header tk-command (pos m_tcl_clear_connection *mtypes*) 0)
     (write-to-connection *tk-connection* tk-command))
    )))

(defun read-tk-message (ar connection timeout &aux 
			   (n-read 0))
  (declare (fixnum timeout n-read)
	   (string ar))
  (cond (*pending*
	 (read-message-body *pending* connection timeout)))
	 
  (setq n-read(si::our-read-with-offset (tk-connection-fd  connection)
					ar 0 #.(length *header*) timeout))
  (setq *pending* ar)
  (cond ((not  (eql n-read #.(length *header*)))
	 (cond ((< n-read 0)
		(tkdisconnect)
		(cerror ":r to resume "
			"Read got an error, have closed connection"))
	       (t 	       (error "Bad tk message"))))
	(t
	 (or (and 
	      (eql (aref ar (pos magic1 *header*)) *magic1*)
	      (eql (aref ar (pos magic2 *header*)) *magic2*))
	     (error "Bad magic"))
	 (read-message-body ar connection timeout))))

(defun read-message-body (ar connection timeout &aux (m 0) (n-read 0))
  (declare (fixnum m n-read))
  (setq m (get-number-string ar (pos body-length *header*) 3))
  (or (>= (array-total-size ar) (the fixnum (+ m #.(length *header*))))
      (setq ar (adjust-array ar (the fixnum (+ m 40)))))
  (cond (*pending*
	 (setq n-read (si::our-read-with-offset (tk-connection-fd connection)
						ar
				     #.(length *header*) m  timeout))
	 (setq *pending* nil)
	 (or (eql n-read m)
	     (error "Failed to read ~a bytes" m))
	 (setf (fill-pointer ar) (the fixnum (+ m #.(length *header*))))))
  (if *debugging* (describe-message ar))
  ar)

(defun tkdisconnect ()
  (cond (*tk-connection*
	 (si::close-sd (caar *tk-connection*))
	 (si::close-fd (cadr *tk-connection*))))
  (setq *sigusr1* t);; disable it...
  (setq *pending* nil)
  (setf *tk-connection* nil)
  
  )

(defun read-and-act (id)
  id
  (when
   *tk-connection*
   (let* ((*sigusr1* t) tem fun string)
     (with-tk-command
      (tagbody
       TOP
       (or (> (si::check-state-input 
	       (tk-connection-fd *tk-connection*) 0) 0)
	   (return-from read-and-act))
       (setq string (read-tk-message tk-command *tk-connection* *default-timeout*))

       (let ((type (char-code (aref string (pos type *header*))))
	     from-id success)
	 (case
	  type
	  (#.(pos m_reply *mtypes*)
	     (setq from-id (get-number-string tk-command #.(+ 1  (length *header*))
					      3))
	     (setq success (eql (aref tk-command  #.(+ 0  (length *header*)))
				#\0))
	     (cond ((and (not success)
			 *break-on-errors*
			 (not (get-circle *requests* from-id)))
		    (cerror
		     ":r to resume ignoring"
		     "request ~s failed: ~s"
		     (or (get-circle *request-forms* from-id) "")
		     (subseq tk-command #.(+ 4 (length *header*))))))
				
	     (store-circle *replies*
			   (cons success
				 (if (eql (length tk-command) #.(+ 4 (length *header*))) ""
				   (fsubseq tk-command #.(+ 4 (length *header*)))))
			   from-id))
	  (#.(pos m_call *mtypes*)
	     ;; Can play a game of if read-and-act called with request-id:
	     ;; When we send a request which waits for an m_reply, we note
	     ;; at SEND time, the last message id received from tk.   We
	     ;; dont process any funcall's with lower id than this id,
	     ;; until after we get the m_reply back from tk.
	     (let ((islot
		    (get-number-string tk-command #.(+ 0 (length *header*))3))
		   (n (length tk-command)))
	       (declare (fixnum islot n))
	       (setq tem (our-read-from-string tk-command
					        #.(+ 0 (length *header*)3)))
	       (or (< islot (length *call-backs*))
		   (error "out of bounds call back??"))
	       (setq fun (aref (the (array t) *call-backs*) islot))
	       (cond ((equal n #.(+ 3 (length *header*)))
		      (funcall fun))
		     (t
		      (setq tem (our-read-from-string
				 tk-command
				 #.(+ 3(length *header*))))
		      (cond ((null tem) (funcall fun))
			    ((consp tem) (apply fun tem))
			    (t (error "bad m_call message ")))))))
	  (#.(pos m_set_lisp_loc *mtypes*)
	     (let* ((lisp-var-id (get-number-string tk-command #.(+ 0  (length *header*))
						    3))
		    (var (aref *text-variable-locations* lisp-var-id))
		    (type (get var 'linked-variable-type))
		    val)
	       (setq val (coerce-result (fsubseq tk-command  #.(+ 3 (length *header*))) type))
	       (setf (aref *text-variable-locations* (the fixnum
							  ( + lisp-var-id 1)))
		     val)
	       (set var val)))
	  (otherwise (format t "Unknown response back ~a" tk-command)))
	       
	 (if (eql *sigusr1* :received)
	     (dformat  "<<received signal while reading>>"))
	 (go TOP)
	 ))))))

(defun our-read-from-string (string start)
  (let* ((s (car *string-streams*))
	 (*string-streams* (cdr *string-streams*)))
    (or s (setq s (make-string-input-stream "")))
    (assert (array-has-fill-pointer-p string))
    (setf (fill-pointer string) start)
    (si::c-set-stream-object0 s string)
    (read s nil nil)))


(defun atoi (string)
  (if (numberp string) string
    (our-read-from-string string 0)))


(defun conc (a b &rest l &aux tem)
  (declare (:dynamic-extent l))
  (sloop
     do
     (or (symbolp a) (error "not a symbol ~s" a))
;     (or (symbolp b) (error "not a symbol ~s" b))
     (cond ((setq tem (get a b)))
	   (t (setf (get a b)
		    (setq tem (intern (format nil "~a~a" a b)
				      *tk-package*
				      )))))
     while l
     do
     (setq a  tem b (car l) l (cdr l)))
  tem)

     


(defun dpos (x)  (wm :geometry x "+60+25"))

(defun string-list (x)
  (let ((tk-command
	 (make-array 30 :element-type 'standard-char :fill-pointer 0 :adjustable t)))
    (string-list1 tk-command x)
    tk-command))

(defun string-list1 (tk-command l &aux x)
  ;; turn a list into a tk list
    (desetq (x . l) l)
    (pp x no_leading_space)
    (while l
      (desetq (x . l) l)
      (cond ((atom x)
	     (pp x normal))
	    ((consp x)
	     (pp "{" no_quote)
	     (string-list1 tk-command x)
	     (pp '} no_leading_space)))))

(defun list-string (x &aux
		      (brace-level 0)
		      skipping (ch #\space)
		      (n (length x))
		      )
  (declare (Fixnum brace-level n)
	   (string x)
	   (character ch))
  (if (eql n 0) (return-from list-string nil)) 
  (sloop for i below n
     with beg = 0 and ans
     do (setq ch (aref x i))
     (cond
      ((eql ch #\space)
       (cond (skipping nil)
	     ((eql brace-level 0)
	      (if (> i beg)
		  (setq ans (cons (fsubseq x beg i) ans)))
	      
	      (setq beg (+ i 1))
		       )))
      (t (cond (skipping (setq skipping nil)
			 (setq beg i)))
       (case ch
       (#\{ (cond ((eql brace-level 0)
		   (setq beg (+ i 1))))
	    (incf brace-level))
       (#\} (cond ((eql brace-level 1)
		   (setq ans (cons (fsubseq x beg i) ans))
		   (setq skipping t)))
	    (incf brace-level -1)))))
     finally
     (unless skipping
	     (setq ans (cons (fsubseq x beg i) ans)))
     (return (nreverse ans))
     ))

;; unless keyword :integer-value, :string-value, :list-strings, :list-forms
;; (foo :return 'list)  "ab 2 3" --> (ab 2 3)
;; (foo :return 'list-strings)  "ab 2 3" --> ("ab" "2" "3")  ;;ie 
;; (foo :return 'string)  "ab 2 3" --> "ab 2 3"
;; (foo :return 't)  "ab 2 3" --> AB
;; (foo :return 'boolean)  "1" --> t

  
(defun coerce-result (string key)
  (case key
    (list (our-read-from-string (tk-conc "("string ")") 0))
    (string string)
    (number (our-read-from-string string 0))
    ((t) (our-read-from-string string 0))
    (t (let ((funs (get key 'coercion-functions)))
	 (cond ((null funs)
		(error "Undefined coercion for type ~s" key)))
	 (funcall (car funs) string)))))

;;convert "2c" into screen units or points or something...

;; If loc is suitable for handing to setf,  then
;; (setf loc (coerce-result val type)
;; (radio-button

(defvar *unbound-var* "<unbound>")

(defun link-variable (var type)
  (let* ((i 0)
	 (ar  *text-variable-locations*)
	 (n (length ar))
	   tem
	 )
    (declare (fixnum i n)
	     (type (array (t)) ar))
    (cond ((stringp var)
	   (return-from link-variable var))
	  ((symbolp var))
	  ((and (consp var)
		(consp (cdr var)))
	   (setq type (car var))
	   (setq var (cadr var))))
    (or (and (symbolp type)
	     (get type 'coercion-functions))
	(error "Need coercion functions for type ~s" type))
    (or (symbolp var) (error "illegal text variable ~s" var))
    (setq tem (get var 'linked-variable-type))
    (unless (if (and tem (not (eq tem type)))
		(format t "~%;;Warning: ~s had type ~s, is being changed to type ~s"
			var tem type
			)))
    (setf (get var 'linked-variable-type) type)
    (while (< i n)
      (cond ((eq (aref ar i) var)
	     (return-from link-variable var))
	    ((null (aref ar i))
	     (return nil))
	    (t   (setq i (+ i 2)))))
;; i is positioned at the write place
    (cond ((= i n)
	   (vector-push-extend nil ar)
	   (vector-push-extend nil ar)))
    (setf (aref ar i) var)
    (setf (aref ar (the fixnum (+ i 1)))
		(if (boundp var)
		    (symbol-value var)
		  *unbound-var*))
    (with-tk-command
     (push-number-string tk-command i #.(length *header*) 3)
     (setf (fill-pointer tk-command) #. (+ 3  (length *header*)))
     (pp var no_quotes_and_no_leading_space)
     (vector-push-extend (code-char 0) tk-command)
     (set-message-header tk-command (pos m_tcl_link_text_variable *mtypes*)
			 (- (length tk-command) #.(length *header*)))
     (write-to-connection *tk-connection* tk-command)))
  (notice-text-variables)
  var)

(defun unlink-variable (var )
  (let* ((i 0)
	 (ar  *text-variable-locations*)
	 (n (length ar))

	 )
    (declare (fixnum i n)
	     (type (array (t)) ar))
    (while (< i n)
      (cond ((eq (aref ar i) var)
	     (setf (aref ar i) nil)
	     (setf (aref ar (+ i 1)) nil)
	     (return nil)
	     )
	    (t   (setq i (+ i 2)))))
    
    (cond ((< i n)
	   (with-tk-command
	    (push-number-string tk-command i #.(length *header*) 3)
	    (setf (fill-pointer tk-command) #. (+ 3  (length *header*)))
	    (pp var no_quotes_and_no_leading_space)
	    (vector-push-extend (code-char 0) tk-command)
	    (set-message-header tk-command (pos m_tcl_unlink_text_variable *mtypes*)
				(- (length tk-command) #.(length *header*)))
	    (write-to-connection *tk-connection* tk-command))
	   var))))
  
(defun notice-text-variables ()
  (let* ((i 0)
	 (ar  *text-variable-locations*)
	 (n (length ar))
	  tem var type
	 )
    (declare (fixnum i n)
	     (type (array (t)) ar))
    (tagbody
     (while (< i n)
       (unless (or (not (boundp (setq var  (aref ar i))))
		   (eq (setq tem (symbol-value var))
		       (aref ar (the fixnum (+ i 1)))))
	       (setf (aref ar (the fixnum (+ i 1))) tem)
	       (setq type (get var 'linked-variable-type))
	       (with-tk-command
		;(push-number-string tk-command i #.(length *header*) 3)
		;(setf (fill-pointer tk-command) #. (+ 3  (length *header*)))
		(pp var no_quote_no_leading_space)
		(vector-push (code-char 0) tk-command )
		(case type
		  (string (or (stringp tem) (go error)))
		  (number (or (numberp tem) (go error)))
		  ((t) (setq tem (format nil "~s" tem )))
		  (t 
		   (let ((funs (get type 'coercion-functions)))
		     (or funs (error "no writer for type ~a" type))
		     (setq tem (funcall (cdr funs) tem)))))
		(pp tem no_quotes_and_no_leading_space)
		(vector-push (code-char 0) tk-command )
		(set-message-header tk-command (pos m_tcl_set_text_variable *mtypes*)
				    (- (length tk-command) #.(length *header*)))
		(write-to-connection *tk-connection* tk-command)))
       (setq i (+ i 2)))
     (return-from notice-text-variables)
     error
     (error "~s has value ~s which is not of type ~s" (aref ar i)
	    tem type)
     )))
(defmacro setk (&rest l)
  `(prog1 (setf ,@ l)
    (notice-text-variables)))

(setf (get 'boolean 'coercion-functions)
      (cons #'(lambda (x &aux (ch (aref x 0)))
		(cond ((eql ch #\0) nil)
		      ((eql ch #\1) t)
		      (t (error "non boolean value ~s" x))))
	    #'(lambda (x) (if x "1" "0"))))

(setf (get 't 'coercion-functions)
      (cons #'(lambda (x) (our-read-from-string x 0))
	    #'(lambda (x) (format nil "~s" x))))

(setf (get 'string 'coercion-functions)
      (cons #'(lambda (x)
		(cond ((stringp x) x)
		      (t (format nil "~s" x))))
	    'identity))


(setf (get 'list-strings 'coercion-functions)
      (cons 'list-string 'list-to-string))
(defun list-to-string  (l &aux (x l) v (start t))
  (with-tk-command
   (while x
     (cond ((consp x)
	    (setq v (car  x)))
	   (t (error "Not a true list ~s" l)))
     (cond (start (pp v no_leading_space) (setq start nil))
	   (t (pp v normal)))
     (setf x (cdr x)))
   (fsubseq tk-command #.(length *header*))))



(defvar *tk-library* nil)
(defun tkconnect (&key host can-rsh gcltksrv (display (si::getenv "DISPLAY"))
		       (args  "")
			    &aux hostid  (loopback "127.0.0.1"))
  (if *tk-connection*  (tkdisconnect))
  (or display (error "DISPLAY not set"))
  (or *tk-library* (setq *tk-library* (si::getenv "TK_LIBRARY")))
  (or gcltksrv
      (setq 	gcltksrv
	 (cond (host "gcltksrv")
	       ((si::getenv "GCL_TK_SERVER"))
	       ((probe-file (tk-conc si::*lib-directory* "gcl-tk/gcltksrv")))
	       (t (error "Must setenv GCL_TK_SERVER ")))))
  (let ((pid (if host  -1 (si::getpid)))
	(tk-socket  (si::open-named-socket 0))
	)
    (cond ((not host) (setq hostid loopback))
	  (host (setq hostid (si::hostname-to-hostid (si::gethostname)))))
    (or hostid (error "Can't find my address"))
    (setq tk-socket (si::open-named-socket 0))
    (if (pathnamep gcltksrv) (setq gcltksrv (namestring gcltksrv)))
    (let ((command 
	   (tk-conc   gcltksrv " " hostid " "
		       (cdr tk-socket) " "
		        pid " " display " "
			args
			)))
      (print command)
      (cond ((not host) (si::system command))
	    (can-rsh
	      (si::system (tk-conc "rsh " host " "   command
			        " < /dev/null &")))
	    (t (format t "Waiting for you to invoke GCL_TK_SERVER,
on ~a as in: ~s~%" host command )))
      (let ((ar *text-variable-locations*))
	(declare (type (array (t)) ar)) 
	(sloop for i below (length ar) by 2
	       do (remprop (aref ar i) 'linked-variable-type)))
      (setf (fill-pointer *text-variable-locations*) 0)
      (setf (fill-pointer *call-backs*) 0)

      (setq *tk-connection* (si::accept-socket-connection tk-socket ))
      (if (eql pid -1)
	  (si::SET-SIGIO-FOR-FD  (car (car *tk-connection*))))
      (setf *sigusr1* nil)
      (tk-do (tk-conc "source "  si::*lib-directory* "gcl-tk/gcl.tcl"))
      )))


  
(defun children (win)
  (let ((ans (list-string (winfo :children win))))
    (cond ((null ans) win)
	  (t (cons win (mapcar 'children ans))))))


;; read nth item from a string in



(defun nth-a (n string &optional (separator #\space) &aux (j 0) (i 0)
		(lim (length string)) ans)
  (declare (fixnum j n i lim))
  (while (< i lim)
    (cond ((eql j n)
	   (setq ans (our-read-from-string string i))
	   (setq i lim))
	  ((eql (aref string i) separator)
	   (setq j (+ j 1))))
    (setq i (+ i 1)))
  ans)



(defun set-message-header(vec mtype body-length &aux (m (msg-index)) )
  (declare (fixnum mtype body-length m)
	   (string vec) )
  (setf (aref vec (pos magic1 *header*)) *magic1*)
  (setf (aref vec (pos magic2 *header*)) *magic2*)
;  (setf (aref vec (pos flag *header*)) (code-char (make-flag flags)))
  (setf (aref vec (pos type *header*)) (code-char mtype))
  (push-number-string vec body-length (pos body-length *header*) 3)
  (push-number-string vec  m (pos msg-index *header*) 3)
  (setf (msg-index) (the fixnum (+ m 1)))
  m)

(defun get-autoloads (&optional (lis (directory "*.lisp")) ( out "index.lsp")
				&aux *paths*
				)
  (declare (special *paths*))
  (with-open-file
   (st out :direction :output)
   (format st "~%(in-package ~s)" (package-name *package*))
   (dolist (v lis) (get-file-autoloads v st))
   (format st "~%(in-package ~s)" (package-name *package*))
   (format st "~2%~s" `(setq si::*load-path* (append ',*paths* si::*load-path*)))

   ))


		  
(defun get-file-autoloads (file &optional (out t)
				&aux (eof '(nil))
				(*package* *package*)
				saw-package
				name  )
  (declare (special *paths*))
  (setq name (pathname-name (pathname file)))
  (with-open-file
   (st file)
   (if (boundp '*paths*)
       (pushnew (namestring (make-pathname :directory
					   (pathname-directory
					    (truename st))))
		*paths* :test 'equal))
   (sloop for tem = (read st nil eof)
	  while (not (eq tem eof))
	  do (cond ((and (consp tem) (eq (car tem) 'defun))
		    (or saw-package
			(format t "~%;;Warning:(in ~a) a defun not preceded by package declaration" file))
		    (format out "~%(~s '~s '|~a|)"
			    'si::autoload
			    (second tem) name))
		   ((and (consp tem) (eq (car tem) 'in-package))
		    (setq saw-package t)
		    (or (equal (find-package (second tem)) *package*)
			(format out "~%~s" tem))
		    (eval tem))
		   ))))

;; execute form return values as usual unless error
;; occurs in which case if symbol set-var is supplied, set it
;; to the tag, returning the tag.
(defmacro myerrorset (form &optional set-var)
 `(let ((*break-enable* nil)(*debug-io* si::*null-io*)
	(*error-output* si::*null-io*))
    (multiple-value-call 'error-set-help ',set-var
     (si::error-set ,form))))

(defun error-set-help (var tag &rest l)
  (cond (tag (if var (set var tag))) ;; got an error
	(t (apply 'values l))))

;;; Local Variables: ***
;;; mode:lisp ***
;;; version-control:t ***
;;; comment-column:0 ***
;;; comment-start: ";;; "  ***
;;; End: ***


       
       
       
