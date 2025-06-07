(in-package :si)

(defun set-path-stream-name (x y)
  (check-type x pathname-designator)
  (typecase x
    (synonym-stream (set-path-stream-name (symbol-value (synonym-stream-symbol x)) y))
    (stream (c-set-stream-object1 x y))))

(defun rename-file (f n &aux (pf (pathname f))(pn (merge-pathnames n pf nil))
		      (tpf (truename pf))(nf (namestring tpf))
		      (tpn (translate-logical-pathname pn))(nn (namestring tpn)))
  (declare (optimize (safety 1)))
  (check-type f pathname-designator)
  (check-type n (and pathname-designator (not stream)))
  (unless (rename nf nn)
    (error 'file-error :pathname pf :format-control "Cannot rename ~s to ~s." :format-arguments (list nf nn)))
  (set-path-stream-name f pn)
  (values pn tpf (truename tpn)))

(defun user-homedir-pathname (&optional (host :unspecific hostp))
  (declare (optimize (safety 1)))
  (check-type host (or string list (eql :unspecific)))
  (unless hostp
    (pathname (home-namestring "~"))))

(defun delete-file (f &aux (pf (truename f))(nf (namestring pf)))
  (declare (optimize (safety 1)))
  (check-type f pathname-designator)
  (unless (if (eq :directory (stat nf)) (rmdir nf) (unlink nf))
    (error 'file-error :pathname (pathname nf) :format-control "Cannot delete pathname."))
  t)

(defun file-write-date (spec)
  (declare (optimize (safety 1)))
  (check-type spec pathname-designator)
  (multiple-value-bind
      (tp sz tm) (stat (namestring (truename spec)))
    (declare (ignore tp sz))
    (+ tm (* (+ 17 (* 70 365)) (* 24 60 60)))))

  
(defun file-author (spec)
  (declare (optimize (safety 1)))
  (check-type spec pathname-designator)
  (multiple-value-bind
      (tp sz tm uid) (stat (namestring (truename spec)))
    (declare (ignore tp sz tm))
    (uid-to-name uid)))

