;; Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
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
;; You should have received a copy of the GNU Library General Public License 
;; along with GCL; see the file COPYING.  If not, write to the Free Software
;; Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.


;;;;    assert.lsp


(in-package :si)

(defun check-type-symbol (symbol value type &optional type-string 
				 &aux (type-string (when type-string (concatenate 'string ": need a " type-string))))
  (restart-case 
   (cerror "Check type again." 'type-error :datum value :expected-type type)
   (store-value (v) 
		:report (lambda (stream) (format stream "Supply a new value of ~s. ~a" symbol (or type-string "")))
		:interactive read-evaluated-form
		(setf value v)))
  (if (typep value type) value (check-type-symbol symbol value type type-string)))

(defmacro check-type (place typespec &optional string)
  (declare (optimize (safety 2)))
  `(progn (,(if (symbolp place) 'setq 'setf) ,place 
	   (the ,typespec (if (typep ,place ',typespec) ,place (check-type-symbol ',place ,place ',typespec ',string)))) nil))

(defun read-evaluated-form nil
  (format *query-io* "~&type a form to be evaluated:~%")
  (list (eval (read *query-io*))))

(defun assert-places (places values string &rest args)
  (declare (dynamic-extent args))
  (restart-case
   (apply 'cerror "Repeat assertion." string args)
   (store-value (&rest r)
		:report (lambda (stream) (format stream "Supply a new values for ~s (old values are ~s)." places values))
		:interactive (lambda nil
			       (mapcar (lambda (x)
					 (format *query-io* "~&type a form to be evaluated for ~s:~%" x)
					 (eval (read *query-io*)))
				       places))
		:test (lambda (c) (declare (ignore c)) places)
		(declare (dynamic-extent r))
		(values-list r))))

(defmacro assert (test-form &optional places string &rest args)
  (declare (dynamic-extent args))
  `(do nil (,test-form nil)
     (multiple-value-setq
	 ,places
       (apply 'assert-places ',places (list ,@places)
	      ,@(if string `(,string (list ,@args)) `("The assertion ~:@(~S~) failed." ',test-form nil))))))

(defmacro ctypecase (keyform &rest clauses &aux (key (sgen "CTYPECASE")))
  (declare (optimize (safety 2)))
;  (check-type clauses (list-of proper-list))
  `(do nil (nil)
    (typecase ,keyform
      ,@(mapcar (lambda (l)
		  `(,(car l) (return (progn ,@(subst key keyform (cdr l))))))
		clauses))
    (check-type ,keyform (or ,@(mapcar 'car clauses)))))
