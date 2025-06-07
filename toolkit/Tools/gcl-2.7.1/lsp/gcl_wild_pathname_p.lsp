(in-package :si)

(defun wild-namestring-p (x)
  (when (stringp x) (>= (string-match #v"(\\*|\\?|\\[|\\{)" x) 0)))

(defun wild-dir-element-p (x)
  (or (eq x :wild) (eq x :wild-inferiors) (wild-namestring-p x)))

(defun wild-path-element-p (x)
  (or (eq x :wild) (wild-namestring-p x)))

#.`(defun wild-pathname-p (pd &optional f)
     (declare (optimize (safety 1)))
     (check-type pd pathname-designator)
     (check-type f (or null (member ,@+pathname-keys+)))
     (case f
       ((nil) (or (wild-namestring-p (namestring pd))
		  (when (typep pd 'pathname);FIXME stream
		    (eq :wild (pathname-version pd)))))
       ;; ((nil) (if (stringp pd) (wild-namestring-p pd)
       ;; 		(let ((p (pathname pd)))
       ;; 		  (when (member-if (lambda (x) (wild-pathname-p p x)) +pathname-keys+) t))))
       ((:host :device) nil)
       (:directory (when (member-if 'wild-dir-element-p (pathname-directory pd)) t))
       (:name (wild-path-element-p (pathname-name pd)))
       (:type (wild-path-element-p (pathname-type pd)))
       (:version (wild-path-element-p (pathname-version pd)))))
    
