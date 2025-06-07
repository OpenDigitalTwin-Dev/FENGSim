(in-package :si)

(defun eval-feature (x)
  (cond ((atom x) (member x *features*))
        ((eq (car x) :and)
         (dolist (x (cdr x) t) (unless (eval-feature x) (return nil))))
        ((eq (car x) :or)
         (dolist (x (cdr x) nil) (when (eval-feature x) (return t))))
        ((eq (car x) :not)
	 (not (eval-feature (cadr x))))
	(t (error "~S is not a feature expression." x))))


(defun sharp-+-reader (stream subchar arg)
  (declare (ignore subchar arg))
  (if (eval-feature (let ((*read-suppress* nil) 
			  (*read-base* 10.)
			  (*package* (load-time-value (find-package 'keyword))))
		      (read stream t nil t)))
      (values (read stream t nil t))
    (let ((*read-suppress* t)) (read stream t nil t) (values))))

(set-dispatch-macro-character #\# #\+ 'sharp-+-reader)
(set-dispatch-macro-character #\# #\+ 'sharp-+-reader
                              (si::standard-readtable))

(defun sharp---reader (stream subchar arg)
  (declare (ignore subchar arg))
  (if (eval-feature (let ((*read-suppress* nil)
			  (*read-base* 10.)
			  (*package* (load-time-value (find-package 'keyword))))
		      (read stream t nil t)))
      (let ((*read-suppress* t)) (read stream t nil t) (values))
    (values (read stream t nil t))))

(set-dispatch-macro-character #\# #\- 'sharp---reader)
(set-dispatch-macro-character #\# #\- 'sharp---reader
                              (si::standard-readtable))
