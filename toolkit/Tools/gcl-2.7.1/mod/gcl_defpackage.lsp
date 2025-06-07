;;; -*- Mode: LISP; Syntax: Common-lisp; Base: 10; Package: (DEFPACKAGE :COLON-MODE :EXTERNAL) -*-
;;;
;;;				 THE BOEING COMPANY
;;;			      BOEING COMPUTER SERVICES
;;;			       RESEARCH AND TECHNOLOGY
;;;				  COMPUTER SCIENCE
;;;			      P.O. BOX 24346, MS 7L-64
;;;			       SEATTLE, WA 98124-0346
;;;
;;;
;;; Copyright (c) 1990, 1991 The Boeing Company, All Rights Reserved.
;;; Copyright (c) 2024 Camm Maguire
;;;
;;; Permission is granted to any individual or institution to use,
;;; copy, modify, and distribute this software, provided that this
;;; complete copyright and permission notice is maintained, intact, in
;;; all copies and supporting documentation and that modifications are
;;; appropriately documented with date, author and description of the
;;; change.
;;;
;;; Stephen L. Nicoud (snicoud@boeing.com) provides this software "as
;;; is" without express or implied warranty by him or The Boeing
;;; Company.
;;;
;;; This software is distributed in the hope that it will be useful,
;;; but WITHOUT ANY WARRANTY.  No author or distributor accepts
;;; responsibility to anyone for the consequences of using it or for
;;; whether it serves any particular purpose or works at all.
;;;
;;;	Author:	Stephen L. Nicoud
;;;
;;; -----------------------------------------------------------------
;;;
;;;	Read-Time Conditionals used in this file.
;;;
;;;	#+LISPM
;;;	#+EXCL
;;;	#+SYMBOLICS
;;;	#+TI
;;; 
;;; -----------------------------------------------------------------

;;; -----------------------------------------------------------------
;;;
;;;	DEFPACKAGE - This files attempts to define a portable
;;;	implementation for DEFPACKAGE, as defined in "Common LISP, The
;;;	Language", by Guy L. Steele, Jr., Second Edition, 1990, Digital
;;;	Press.
;;;
;;;	Send comments, suggestions, and/or questions to:
;;;
;;;		Stephen L Nicoud <snicoud@boeing.com>
;;;
;;;	An early version of this file was tested in Symbolics Common
;;;	Lisp (Genera 7.2 & 8.0 on a Symbolics 3650 Lisp Machine),
;;;	Franz's Allegro Common Lisp (Release 3.1.13 on a Sun 4, SunOS
;;;	4.1), and Sun Common Lisp (Lucid Common Lisp 3.0.2 on a Sun 3,
;;;	SunOS 4.1).
;;;
;;;	91/5/23 (SLN) - Since the initial testing, modifications have
;;;	been made to reflect new understandings of what DEFPACKAGE
;;;	should do.  These new understandings are the result of
;;;	discussions appearing on the X3J13 and Common Lisp mailing
;;;	lists.  Cursory testing was done on the modified version only
;;;	in Allegro Common Lisp (Release 3.1.13 on a Sun 4, SunOS 4.1).
;;;
;;; -----------------------------------------------------------------

(unless (find-package :defpackage)
  (make-package :defpackage :use '(:cl)))
(in-package :defpackage)

(export '(defpackage))

(use-package :SLOOP)

(defmacro DEFPACKAGE (name &rest options)
  (declare (optimize (safety 1)))
  "DEFPACKAGE - DEFINED-PACKAGE-NAME {OPTION}*			[Macro]

   This creates a new package, or modifies an existing one, whose name is
   DEFINED-PACKAGE-NAME.  The DEFINED-PACKAGE-NAME may be a string or a 
   symbol; if it is a symbol, only its print name matters, and not what
   package, if any, the symbol happens to be in.  The newly created or 
   modified package is returned as the value of the DEFPACKAGE form.

   Each standard OPTION is a list of keyword (the name of the option)
   and associated arguments.  No part of a DEFPACKAGE form is evaluated.
   Except for the :SIZE and :DOCUMENTATION options, more than one option 
   of the same kind may occur within the same DEFPACKAGE form.

  Valid Options:
	(:documentation		string)
	(:size			integer)
	(:nicknames		{package-name}*)
	(:shadow		{symbol-name}*)
	(:shadowing-import-from	package-name {symbol-name}*)
	(:use			{package-name}*)
	(:import-from		package-name {symbol-name}*)
	(:intern		{symbol-name}*)
	(:export		{symbol-name}*)
	(:export-from		{package-name}*)

  [Note: :EXPORT-FROM is an extension to DEFPACKAGE.
	 If a symbol is interned in the package being created and
	 if a symbol with the same print name appears as an external
	 symbol of one of the packages in the :EXPORT-FROM option,
	 then the symbol is exported from the package being created.

	 :DOCUMENTATION is an extension to DEFPACKAGE.

	 :SIZE is used only in Genera and Allegro.]"
  (sloop for option in options 
	 unless (member 
		 (first option) 
		 '(:documentation :size :nicknames :shadow
				  :shadowing-import-from :use :import-from
				  :intern :export :export-from))
	 do (cerror "Proceed, ignoring this option." "~s is not a valid option." option))
  (let ((name (string name)))
    (labels ((option-test (arg1 arg2) (when (consp arg2) (equal (car arg2) arg1)))
	     (option-values-list (option options)
				 (sloop for result = (member option options
							     :test #'option-test)
					then (member option (rest result)
						     :test #'option-test)
					until (null result) when result collect
					(rest (first result))))
	     (option-values (option options)
			    (sloop for result  = (member option options :test #'option-test)
				   then (member option (rest result) :test #'option-test)
				   until (null result) when result append
				   (rest (first result)))))
	    (sloop for option in '(:size :documentation)
		   when (<= 2 (count option options :key 'car))
		   do (error 'program-error :format-control "DEFPACKAGE option ~s specified more than once."
			     :format-arguments (list option)))
	    (setq name (string name))
	    (let ((nicknames (mapcar 'string (option-values :nicknames options)))
		  (documentation (first (option-values :documentation options)))
					;		(size (first (option-values :size options))) FIXME?  size support in gcl
		  (shadowed-symbol-names (mapcar 'string (option-values :shadow options)))
		  (interned-symbol-names (mapcar 'string (option-values :intern options)))
		  (exported-symbol-names (mapcar 'string (option-values :export options)))
		  (shadowing-imported-from-symbol-names-list 
		   (sloop for list in (option-values-list :shadowing-import-from options)
			  collect (cons (string (first list)) (mapcar 'string (rest list)))))
		  (imported-from-symbol-names-list 
		   (sloop for list in (option-values-list :import-from options)
			  collect (cons (string (first list)) (mapcar 'string (rest list)))))
		  (exported-from-package-names 
		   (mapcar 'string (option-values :export-from options))))
	      (flet ((find-duplicates 
		      (&rest lists)
		      (let (results)
			(sloop for list in lists
			       for more on (cdr lists)
			       for i from 1
			       do
			       (sloop for elt in list
				      as entry = (find elt results :key 'car :test 'string=)
				      unless (member i entry)
				      do
				      (sloop for l2 in more
					     for j from (1+ i)
					     do
					     (if (member elt l2 :test 'string=)
						 (if entry
						     (nconc entry (list j))
						   (setq entry 
							 (car (push 
							       (list elt i j) results))))))))
			results)))
		    (sloop for duplicate in 
			   (find-duplicates 
			    shadowed-symbol-names 
			    interned-symbol-names
			    (sloop for list in shadowing-imported-from-symbol-names-list 
				   append (rest list))
			    (sloop for list in imported-from-symbol-names-list 
				   append (rest list)))
			   do
			   (error 
			    'program-error
			    :format-control "The symbol ~s cannot coexist in these lists:~{ ~s~}" 
			    :format-arguments 
			    (list (first duplicate)
				  (sloop for num in (rest duplicate)
					 collect 
					 (case num 
					       (1 :SHADOW)
					       (2 :INTERN)
					       (3 :SHADOWING-IMPORT-FROM)
					       (4 :IMPORT-FROM))))))
		    (sloop for duplicate in 
			   (find-duplicates exported-symbol-names interned-symbol-names)
			   do
			   (error 
			    'program-error
			    :format-control "The symbol ~s cannot coexist in these lists:~{ ~s~}" 
			    :format-arguments 
			    (list (first duplicate)
				  (sloop for num in 
					 (rest duplicate) 
					 collect (case num 
						       (1 :EXPORT)
						       (2 :INTERN)))))))
	      `(eval-when (load eval compile)
			  (if (find-package ,name)
			      (progn (rename-package ,name ,name)
				     ,@(when nicknames 
					 `((rename-package ,name ,name ',nicknames)))
				     ,@(when (not (null (member :use options :key 'car)))
					 `((unuse-package 
					    (package-use-list (find-package ,name)) ,name))))
			    (make-package 
			     ,name 
			     :use 'nil 
			     :nicknames 
			     ',nicknames))
			  ,@(progn
			      `((setf (get ',(intern name :keyword) 
					   'si::package-documentation) 
				      ,documentation))
			      )
			  (let ((*package* (find-package ,name)))
			    ,@(when SHADOWed-symbol-names 
				`((SHADOW (mapcar 'intern ',SHADOWed-symbol-names))))
			    ,@(when SHADOWING-IMPORTed-from-symbol-names-list
				(mapcar (lambda (list)
					  `(SHADOWING-IMPORT 
					    (mapcar (lambda (symbol) 
						      (multiple-value-bind (sym fnd) (find-symbol symbol ,(first list))
									   (unless fnd
									     (specific-correctable-error 
									      :package-error
									      "A package error occurred on ~S: ~S." ,(first list) 
									      (format nil "~%Symbol ~a not present" symbol)))
									   (intern symbol ,(first list))))
					; FIXME better error messages
						    ',(rest list))))
					SHADOWING-IMPORTed-from-symbol-names-list))
			    (USE-PACKAGE ',(if (member :USE options :test #'option-test)
					       (mapcar 'string (option-values :USE options))
					     "CL"))
			    ,@(when IMPORTed-from-symbol-names-list
				(mapcar (lambda (list) 
					  `(IMPORT (mapcar (lambda (symbol) 
							     (multiple-value-bind (sym fnd) (find-symbol symbol ,(first list))
										  (unless fnd
										    (specific-correctable-error 
										     :package-error
										     "A package error occurred on ~S: ~S." ,(first list) 
										     (format nil "~%Symbol ~a not present" symbol)))
										  (intern symbol ,(first list))))
					; FIXME better error messages
							   ',(rest list))))
					IMPORTed-from-symbol-names-list))
			    ,@(when INTERNed-symbol-names 
				`((mapcar 'INTERN ',INTERNed-symbol-names)))
			    ,@(when EXPORTed-symbol-names 
				`((EXPORT (mapcar 'intern ',EXPORTed-symbol-names))))
			    ,@(when EXPORTed-from-package-names
				`((dolist (package ',EXPORTed-from-package-names)
				    (do-external-symbols 
				     (symbol (find-package package))
				     (when (nth 1 (multiple-value-list 
						   (find-symbol (string symbol))))
				       (EXPORT (list (intern (string symbol)))))))))
			    )
			  (find-package ,name))))))
  

(provide :defpackage)
(pushnew :defpackage *features*)

(eval-when (load)
  (in-package :USER)
  (unintern 'defpackage 'user)
  (use-package "DEFPACKAGE"))

;;;; ------------------------------------------------------------
;;;;	End of File
;;;; ------------------------------------------------------------

