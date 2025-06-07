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


;;;;    packlib.lsp
;;;;
;;;;                    package routines


;; (in-package 'lisp)


;; (export '(find-all-symbols do-symbols do-external-symbols do-all-symbols with-package-iterator))
;; (export '(apropos apropos-list))

(in-package :system)

;;
;; This slightly slower version uses less invocation history stack space
;;
;; (defmacro with-package-iterator ((name packlist key &rest keys) &rest body
;; 				 &aux (*gensym-counter* 0)
;; 				 (pl (sgen "WPI-PL")) (ql (sgen "WPI-QL"))
;; 				 (ilim (sgen "WPI-ILIM")) (elim (sgen "WPI-ELIM"))
;; 				 (p (sgen "WPI-P")) (q (sgen "WPI-Q")) (l (sgen "WPI-L"))
;; 				 (a (sgen "WPI-A")) (x (sgen "WPI-X")) (y (sgen "WPI-Y")))
;;   (declare (optimize (safety 2)))
;;   (let (int ext inh)
;;     (dolist (key (cons key keys))
;;       (ecase key
;; 	     (:internal  (setq int t))
;; 	     (:external  (setq ext t))
;; 	     (:inherited (setq inh t))))
;;     `(let* ((,pl ,packlist) ,p ,q ,ql (,x 0) (,y 0) (,ilim 0) (,elim 0) ,l ,a)
;;        (declare ((integer 0 1048573) ,x ,y ,ilim ,elim) (ignorable ,x ,y ,ilim ,elim))
;;        (labels 
;; 	((match (s l) (member-if (lambda (x) (declare (symbol x)) (string= s x)) l))
;; 	 (iematch (s p h) (or (match s (package-internal p (mod h (package-internal_size p))))
;; 			      (match s (package-external p (mod h (package-external_size p))))))
;; 	 (next-var nil 
;; 		   (tagbody 
;; 		    :top
;; 		    (cond ,@(when (or int ext) `(((when (eq ,q ,p) ,l) (return-from next-var (prog1 ,l (pop ,l))))))
;; 			  ,@(when inh `(((unless (eq ,q ,p) ,l) 
;; 					 (let* ((v (prog1 ,l (pop ,l))) (s (symbol-name (car v))) (h (pack-hash s)))
;; 					   (when (iematch s ,p h) (go :top))
;; 					   (return-from next-var (progn (setq ,a :inherited) v))))))
;; 			  ,@(when int `(((and (eq ,q ,p) (< ,x ,ilim)) (setq ,l (package-internal ,q ,x) ,a :internal ,x (1+ ,x)) (go :top))))
;; 			  ,@(when (or ext inh) `(((< ,y ,elim) (setq ,l (package-external ,q ,y) ,a :external ,y (1+ ,y)) (go :top))))
;; 			  (,ql 
;; 			   (setq ,x 0 ,y 0 ,q (if (listp ,ql) (pop ,ql) (prog1 ,ql (setq ,ql nil))))
;; 			   (multiple-value-setq (,elim ,ilim) (package-size ,q))
;; 			   (go :top))
;; 			  (,pl 
;; 			   (setq ,p (coerce-to-package (if (listp ,pl) (pop ,pl) (prog1 ,pl (setq ,pl nil))))
;; 				 ,ql ,(if inh `(cons ,p (package-use-list ,p)) p))
;; 			   (go :top)))))
;; 	 (,name nil (let ((f (next-var))) (values f (car f) ,a ,p))))
;; 	,@body))))

(defmacro with-package-iterator ((name packlist key &rest keys) &rest body
				 &aux (*gensym-counter* 0)
				 (pl (sgen "WPI-PL")) (ql (sgen "WPI-QL"))
				 (ilim (sgen "WPI-ILIM")) (elim (sgen "WPI-ELIM"))
				 (p (sgen "WPI-P")) (q (sgen "WPI-Q")) (l (sgen "WPI-L"))
				 (a (sgen "WPI-A")) (x (sgen "WPI-X")) (y (sgen "WPI-Y")))
  (declare (optimize (safety 1)))
  (let (int ext inh)
    (dolist (key (cons key keys))
      (ecase key
	     (:internal  (setq int t))
	     (:external  (setq ext t))
	     (:inherited (setq inh t))))
    `(let* ((,pl ,packlist) ,p ,q ,ql (,x 0) (,y 0) (,ilim 0) (,elim 0) ,l ,a)
       (declare ((integer 0 1048573) ,x ,y ,ilim ,elim) (ignorable ,x ,y ,ilim ,elim))
       (labels 
	((match (s l) (member-if (lambda (x) (declare (symbol x)) (string= s x)) l))
	 (inh-match (&aux (v (prog1 ,l (pop ,l))) (s (symbol-name (car v))) (h (pack-hash s)))
		    (cond ((match s (package-internal ,p (mod h (package-internal_size ,p)))) (next-var))
			  ((match s (package-external ,p (mod h (package-external_size ,p)))) (next-var))
			  ((setq ,a :inherited) v)))
	 (next-var nil 
		   (cond ,@(when (or int ext) `(((when (eq ,q ,p) ,l) (prog1 ,l (pop ,l)))))
			 ,@(when inh `(((unless (eq ,q ,p) ,l) (inh-match))))
			 ,@(when int `(((and (eq ,q ,p) (< ,x ,ilim)) (setq ,l (package-internal ,q ,x) ,a :internal ,x (1+ ,x)) (next-var))))
			 ,@(when (or ext inh) `(((< ,y ,elim) (setq ,l (package-external ,q ,y) ,a :external ,y (1+ ,y)) (next-var))))
			 (,ql 
			  (setq ,x 0 ,y 0 ,q (if (listp ,ql) (pop ,ql) (prog1 ,ql (setq ,ql nil))))
			  (multiple-value-setq (,elim ,ilim) (package-size ,q))
			  (next-var))
			 (,pl 
			  (setq ,p (coerce-to-package (if (listp ,pl) (pop ,pl) (prog1 ,pl (setq ,pl nil))))
				,ql ,(if inh `(cons ,p (package-use-list ,p)) p))
			  (next-var))))
	 (,name nil (let ((f (next-var))) (values f (car f) ,a ,p))))
	 (declare (ignorable #'inh-match))
	,@body))))

;; (defmacro with-package-iterator ((name packlist key &rest keys) &rest body
;; 				 &aux (*gensym-counter* 0)
;; 				 (pl (sgen "WPI-PL")) (ql (sgen "WPI-QL"))
;; 				 (ilim (sgen "WPI-ILIM")) (elim (sgen "WPI-ELIM"))
;; 				 (p (sgen "WPI-P")) (q (sgen "WPI-Q")) (l (sgen "WPI-L"))
;; 				 (a (sgen "WPI-A")) (x (sgen "WPI-X")) (y (sgen "WPI-Y")))
;;   (declare (optimize (safety 2)))
;;   (let (int ext inh)
;;     (dolist (key (cons key keys))
;;       (ecase key
;; 	     (:internal  (setq int t))
;; 	     (:external  (setq ext t))
;; 	     (:inherited (setq inh t))))
;;     `(let* ((,pl ,packlist) ,p ,q ,ql (,x 0) (,y 0) (,ilim 0) (,elim 0) ,l ,a)
;;        (declare ((integer 0 1048573) ,x ,y ,ilim ,elim) (ignorable ,x ,y ,ilim ,elim))
;;        (labels 
;; 	((match (s l) (member-if (lambda (x) (declare (symbol x)) (string= s x)) l))
;; 	 (inh-match (&aux (v (prog1 ,l (pop ,l))) (s (symbol-name (car v))) (h (pack-hash s)))
;; 		    (cond ((match s (package-internal ,p (mod h (package-internal_size ,p)))) (next-var))
;; 			  ((match s (package-external ,p (mod h (package-external_size ,p)))) (next-var))
;; 			  ((setq ,a :inherited) v)))
;; 	 (next-var nil 
;; 		   (tagbody
;; 		    :top
;; 		    (cond ,@(when (or int ext) `(((when (eq ,q ,p) ,l) (return-from next-var (prog1 ,l (pop ,l))))))
;; 			  ,@(when inh `(((unless (eq ,q ,p) ,l) (return-from next-var (inh-match)))))
;; 			  ,@(when int `(((and (eq ,q ,p) (< ,x ,ilim)) (setq ,l (package-internal ,q ,x) ,a :internal ,x (1+ ,x)) (go :top))))
;; 			  ,@(when (or ext inh) `(((< ,y ,elim) (setq ,l (package-external ,q ,y) ,a :external ,y (1+ ,y)) (go :top))))
;; 			  (,ql 
;; 			   (setq ,x 0 ,y 0 ,q (if (listp ,ql) (pop ,ql) (prog1 ,ql (setq ,ql nil))))
;; 			   (multiple-value-setq (,elim ,ilim) (package-size ,q))
;; 			   (go :top))
;; 			  (,pl 
;; 			   (setq ,p (coerce-to-package (if (listp ,pl) (pop ,pl) (prog1 ,pl (setq ,pl nil))))
;; 				 ,ql ,(if inh `(cons ,p (package-use-list ,p)) p))
;; 			   (go :top)))))
;; 	 (,name nil (let ((f (next-var))) (values f (car f) ,a ,p))))
;; 	,@body))))



(eval-when 
 (eval compile)
 (defmacro do-symbols1 ((var package result-form &rest keys) body
			&aux (*gensym-counter* 0)(m (sgen "DS-M"))(f (sgen "DS-F")))
  `(multiple-value-bind
    (doc declarations check-types body)
    (parse-body-header ,body)
    (declare (ignore doc))
    `(with-package-iterator 
      (,',f ,,package ,,@keys)
      (do (,',m ,,var) ((not (multiple-value-setq (,',m ,,var) (,',f))) ,,result-form)
	  (declare (ignorable ,',m) (symbol ,,var))
	  ,@declarations
	  ,@check-types
	  ,@body)))))

(defmacro do-symbols ((var &optional (package '*package*) result-form) &rest body)
  (do-symbols1 (var package result-form :internal :external :inherited) body))

(defmacro do-external-symbols ((var &optional (package '*package*) result-form) &rest body)
  (do-symbols1 (var package result-form :external) body))

(defmacro do-all-symbols ((var &optional result-form) &rest body)
  (do-symbols1 (var '(list-all-packages) result-form :internal :external :inherited) body))

(defun find-all-symbols (sd)
  (declare (optimize (safety 1)))
  (check-type sd string-designator)
  (setq sd (string  sd))
  (mapcan (lambda (p)
	    (multiple-value-bind 
	     (s i)
	     (find-symbol sd p)
	     (when (or (eq i :internal) (eq i :external))
		 (list s))))
          (list-all-packages)))

;; (defun substringp (sub str)
;;   (do ((i (- (length str) (length sub)))
;;        (l (length sub))
;;        (j 0 (1+ j)))
;;       ((> j i) nil)
;;     (when (string-equal sub str :start2 j :end2 (+ j l))
;;           (return t))))


(defun print-symbol-apropos (symbol)
  (prin1 symbol)
  (when (fboundp symbol)
        (if (special-operator-p symbol)
            (princ "  Special form")
            (if (macro-function symbol)
                (princ "  Macro")
                (princ "  Function"))))
  (when (boundp symbol)
        (if (constantp symbol)
            (princ "  Constant: ")
            (princ "  has value: "))
        (prin1 (symbol-value symbol)))
  (terpri))

(defun apropos-list (string &optional package &aux list (package (or package (list-all-packages))))
  (declare (optimize (safety 1)))
  (setq string (string string))
  (do-symbols (symbol package list) ;FIXME?
	       (when (search string (string symbol) :test 'char-equal)
		  (push symbol list)))
  (stable-sort list 'string< :key 'symbol-name))

(defun apropos (string &optional package)
  (declare (optimize (safety 1)))
  (dolist (symbol (apropos-list string package))
    (print-symbol-apropos symbol))
  (values))

(defun package-name (p) 
  (c-package-name (si::coerce-to-package p)))

(defun make-package (name &key nicknames use)
  (declare (optimize (safety 1)))
  (check-type name string-designator)
  (check-type nicknames proper-list)
  (check-type use proper-list)
  (make-package-int name nicknames use))
