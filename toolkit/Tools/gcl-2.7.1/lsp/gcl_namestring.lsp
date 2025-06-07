(in-package :si)

(defun namestring (x)
  (declare (optimize (safety 1)))
  (check-type x pathname-designator)
  (typecase x
    (string x)
    (pathname (c-pathname-namestring x))
    (stream (namestring (c-stream-object1 x)))))

(defun file-namestring (x &aux (px (pathname x)))
  (declare (optimize (safety 1)))
  (check-type x pathname-designator)
  (namestring (make-pathname :name (pathname-name px) :type (pathname-type px) :version (pathname-version px))))

(defun directory-namestring (x &aux (px (pathname x)))
  (declare (optimize (safety 1)))
  (check-type x pathname-designator)
  (namestring (make-pathname :directory (pathname-directory px))))

(defun host-namestring (x &aux (px (pathname x)))
  (declare (optimize (safety 1)))
  (check-type x pathname-designator)
  (or (pathname-host px) ""))

#.`(defun enough-namestring (x &optional (def *default-pathname-defaults*) &aux (px (pathname x))(pdef (pathname def)))
     (declare (optimize (safety 1)))
     (check-type x pathname-designator)
     (check-type def pathname-designator)
     ,(labels ((new? (k &aux (f (intern (concatenate 'string "PATHNAME-" (string k)) :si)))
		     `(let ((k (,f px))) (unless (equal k (,f pdef)) k))))
	`(namestring (make-pathname
	  ,@(mapcan (lambda (x) (list x (new? x))) +pathname-keys+)))))

(defun faslink (file name &aux (pfile (namestring (merge-pathnames (make-pathname :type "o") (pathname file))))(*package* *package*));FIXME
  (declare (optimize (safety 1)))
  (check-type file pathname-designator)
  (check-type name string)
  (faslink-int pfile name))
