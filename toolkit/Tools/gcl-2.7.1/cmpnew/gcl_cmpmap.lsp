;;; CMPMAP  Map functions.
;;;
;; Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa

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


(in-package :compiler)

(defun push-changed-vars (locs funob &aux (locs1 nil) (forms (list funob)))
  (dolist (loc locs (reverse locs1))
          (if (and (consp loc)
                   (eq (car loc) 'VAR)
                   (args-info-changed-vars (cadr loc) forms))
              (let ((temp (list 'VS (vs-push))))
                   (wt-nl temp "= " loc ";")
                   (push temp locs1))
              (push loc locs1))))
