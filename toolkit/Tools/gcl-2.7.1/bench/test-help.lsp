(defvar *repeats* '(("destru" 4)("destru-mod" 4)
		    ("fprint" 4)("fread" 4)("tprint" 4)
		    ("tak-mod" 4)("tak" 4)("takl" 4)("stak" 4)("takr" 4)
		    ("fft" 10)("fft-mod" 10)
		    ("traverse" 0.1)("triang-mod" 0.1)("triang" 0.1)))


(defun do-test (file output &optional
			      (n (or (cadr (assoc (pathname-name file) *repeats* :test 'equal)) 1))
			      (scale 100))
  (load file)
  (let* ((file (pathname-name file))
	 (pos (position #\- file)))
    (let ((command (intern (string-upcase (format nil "TEST~a" (if pos (subseq file 0 pos) file))))))
      (let ((start    (get-internal-run-time)))
	(with-open-file (s "/dev/null" :direction :output :if-exists :append)
	  (let ((*trace-output* s)(*standard-output* s))
	    (dotimes (i (truncate (* n scale))) (funcall command))))
	(setq start (- (get-internal-run-time) start))
	(setq start (float start));(setq start (/ (float start) n))
	(with-open-file
	    (st output :direction :output :if-exists :append :if-does-not-exist :create)
	  (format st "~:@(~a~)~,12t~,3f~%"
		  file
		  (/ start (float internal-time-units-per-second)))
	  (force-output st)
	  )))))
