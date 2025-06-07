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


;;;;    module.lsp
;;;;
;;;;                            module routines


;; (in-package 'lisp)

;; (export '(*modules* provide require))
;; (export '(documentation variable function structure type setf compiler-macro))

(in-package :system)


(defvar *modules* nil)


(defun provide (module-name)
  (declare (optimize (safety 1)))
  (check-type module-name string-designator)
  (pushnew (string module-name) *modules* :test 'string=))

(defun list-of-pathname-designators-p (x)
  (not (member-if-not (lambda (x) (typep x 'pathname-designator)) x)))

(defun default-module-pathlist (module-name)
  (list (make-pathname :name (string module-name)
		       :directory (append (pathname-directory (pathname *system-directory*))
					  (list :up "modules")))))

(defun require (module-name &optional (pl (default-module-pathlist module-name))
		&aux (*default-pathname-defaults* (make-pathname))
		  (pl1 (if (listp pl) pl (list pl))));FIXME ansi-test modules.7
  (declare (optimize (safety 1)))
  (check-type module-name string-designator)
  (check-type pl1 (and proper-list (satisfies list-of-pathname-designators-p)))
  (unless (member (string module-name) *modules* :test 'string=)
    (mapc 'load pl1)))

(defun software-type nil nil)
(defun software-version nil nil)

(defvar *doc-strings* (make-hash-table :test 'eq));FIXME weak

(defun real-documentation (object doc-type)
  (declare (optimize (safety 1)))
  (check-type doc-type (member variable function structure type setf compiler-macro method-combination t))
  (getf (gethash object *doc-strings*) doc-type))

(defun set-documentation (object doc-type value)
  (declare (optimize (safety 1)))
  (check-type doc-type (member variable function structure type setf compiler-macro method-combination t))
  (setf (getf (gethash object *doc-strings*) doc-type) value))



(defun find-documentation (body)
  (if (or (endp body) (endp (cdr body)))
      nil
      (let ((form (macroexpand (car body))))
        (if (stringp form)
            form
            (if (and (consp form)
                     (eq (car form) 'declare))
                (find-documentation (cdr body))
                nil)))))
