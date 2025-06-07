(in-package :si)

(defun to-regexp (x &optional (rp t) &aux (px (pathname x))(lp (typep px 'logical-pathname)))
  (to-regexp-or-namestring (mlp px) rp lp))

(deftype compiled-regexp nil `(vector unsigned-char))

(defun pathname-match-p (p w &aux (s (namestring p)))
  (declare (optimize (safety 1)))
  (check-type p pathname-designator)
  (check-type w (or compiled-regexp pathname-designator))
  (and (zerop (string-match (if (typep w 'compiled-regexp) w (to-regexp w)) s))
       (eql (match-end 0) (length s))))

