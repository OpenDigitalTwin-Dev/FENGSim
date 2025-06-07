;;; Thu Aug 12 14:22:09 1993 by Mark Kantrowitz <mkant@cs.cmu.edu>
;;; make-defpackage.lisp -- 1961 bytes

;;; ****************************************************************
;;; Make a Defpackage Form From Package State **********************
;;; ****************************************************************

(in-package :si)

(defun make-defpackage-form (package-name)
  "Given a package, returns a defpackage form that could recreate the 
   current state of the package, more or less."
  (let ((package (find-package package-name)))
    (let* ((name (package-name package))
	   (nicknames (package-nicknames package))
	   (package-use-list (package-use-list package))
	   (use-list (mapcar #'package-name package-use-list))
	   (externs nil)
	   (shadowed-symbols (package-shadowing-symbols package))
	   (imports nil)
	   (shadow-imports nil)
	   (pure-shadow nil) 
	   (pure-import nil))
      (do-external-symbols (sym package) (push (symbol-name sym) externs))
      (do-symbols (sym package)
	(unless (or (eq package (symbol-package sym)) 
		    (find (symbol-package sym) package-use-list))
	  (push sym imports)))
      (setq shadow-imports (intersection shadowed-symbols imports))
      (setq pure-shadow (set-difference shadowed-symbols shadow-imports))
      (setq pure-import (set-difference imports shadow-imports))
      `(defpackage ,name
	   ,@(when nicknames `((:nicknames ,@nicknames)))
	   ,@(when use-list `((:use ,@use-list)))
	   ,@(when externs `((:export ,@externs)))
	   ;; skip :intern
	   ,@(when pure-shadow 
	       `((:shadow ,@(mapcar #'symbol-name pure-shadow))))
	   ,@(when shadow-imports
	       (mapcar #'(lambda (symbol)
			   `((:shadowing-import-from 
			      ,(package-name (symbol-package symbol))
			      ,(symbol-name symbol))))
		       shadow-imports))
	   ,@(when pure-import 
	       (mapcar #'(lambda (symbol)
			   `((:import-from
			      ,(package-name (symbol-package symbol))
			      ,(symbol-name symbol))))
		       pure-import))))))

;;; *EOF*
