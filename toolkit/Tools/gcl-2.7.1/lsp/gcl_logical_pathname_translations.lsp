(in-package :si)

(defvar *pathname-logical* nil)

(defun (setf logical-pathname-translations) (v k)
  (declare (optimize (safety 1)))
  (check-type v list)
  (check-type k string)
  (setf (cdr (or (assoc k *pathname-logical* :test 'string-equal) (car (push (cons k t) *pathname-logical*)))) ;(cons k nil)
	(mapcar (lambda (x) (list (parse-namestring (car x) k) (parse-namestring (cadr x)))) v)))

;(defsetf logical-pathname-translations (x) (y) `(setf-logical-pathname-translations ,y ,x))
(remprop 'logical-pathname-translations 'si::setf-update-fn)

(defun logical-pathname-translations (k)
  (declare (optimize (safety 1)))
  (check-type k string)
  (cdr (assoc k *pathname-logical* :test 'string-equal)))


(defun load-logical-pathname-translations (k)
  (declare (optimize (safety 1)))
  (unless (logical-pathname-translations k)
    (error "No translations found for ~s" k)))

(defun logical-pathname-host-p (host)
  (when host
    (logical-pathname-translations host)))
