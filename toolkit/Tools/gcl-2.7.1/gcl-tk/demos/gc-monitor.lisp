
;; bug in aix c compiler on optimize??
#+aix3 (eval-when (compile) (proclaim '(optimize (speed 0))))

(in-package "TK")

(defvar *gc-monitor-types*
  '(cons fixnum string si::relocatable-blocks stream))

(defvar *special-type-background* "red")

(defun make-one-graph (top type)
  (let* ((f (conc top '.type type)))
    (setf (get type 'frame) f)
    (setf (get type 'canvas) (conc top '.canvas type))
    (frame f )
    (canvas (get type 'canvas) :relief "sunken" :width "8c" :height ".4c")
    (label (conc f '.data))
    (button (conc f '.label)  :text (string-capitalize (symbol-name type))
	    :background "gray90"
	    :command `(draw-status ',type t))
    (pack (conc f '.label) (conc f '.data)  :side "left" :anchor "w" :padx "4m")
    (pack f :side "top" :anchor "w"  :padx "1c")
    (pack (get type 'canvas)  :side "top" :expand 1 :pady "2m")
    ))

(defvar *prev-special-type* nil)

(defvar *time-to-stay-on-type* 0)


(defvar *values-array* (make-array 20 :fill-pointer 0))
(defun push-multiple-values (&rest l)
   (declare (:dynamic-extent l))
   (dolist (v l) (vector-push-extend v *values-array*)))

(defun draw-status (special-type &optional clicked)
  (setf (fill-pointer *values-array*) 0)
  (let ((max-size 0) (ar *values-array*) (i 0) (width 7.0s0)
	(ht ".15c"))
    (declare (si::seqind max-size) (short-float width)(type (array (t)) ar))
    (dolist (v *gc-monitor-types*)
      (let ((fp (fill-pointer *values-array*))
	    )
	(multiple-value-call 'push-multiple-values (si::allocated v))
	(setq max-size (max max-size (aref ar (the si::seqind (+ fp 1)))))))
					;  (nfree npages maxpage nppage gccount nused)
    (dolist (v *gc-monitor-types*)
      (let* ((nfree (aref ar i))
	     (npages (aref ar (setq i(+ i 1))))
	     (maxpage (aref ar (setq i(+ i 1))))
	     (nppage (aref ar (setq i(+ i 1))))
	     (gccount (aref ar (setq i (+ i 1))))
	     (nused   (aref ar (setq i (+ i 1))))
	     (wid (/ (the short-float(* npages width)) max-size))
	     (f (get v 'frame))
	     (tot (* npages nppage))
	     (width-used (the short-float
			      (/ (the short-float
				      (* wid (the si::seqind
						  (- tot
						     (the si::seqind nfree)))))
				 tot))))
	(declare (si::seqind nppage npages  tot)
		 (short-float  wid))
	(setq i (+ i 1))
    	(funcall (get v 'canvas) :delete "graph")
	(funcall (get v 'canvas) :create "line"
		 0 ht
		 width-used : "c" ht
		 :width "3m" :tag "graph" :fill "red")
	(funcall  (get v 'canvas) :create "line" 
		  width-used : "c" ht
		  wid : "c" ht
		  :width "3m" :tag "graph" :fill "aquamarine4" )
	(funcall (conc f '.data) :configure :text
		 gccount	: " gc's for " :|| npages :
		 " pages (used=" :|| nused : ")")
	(cond ((eql special-type v)
	       (cond
		(clicked
		 (let ((n (* max-size 2)))
		   (.gc.amount :configure :length "8c"
			       :label "Allocate: " : (or special-type "")
			       :tickinterval (truncate n 4) :to n)
		   (.gc.amount :set  npages)

		   )))))))
    (set-label-background *prev-special-type* "pink")

    (setq *prev-special-type* special-type)
    (set-label-background special-type *special-type-background*)
    )
  )

  

(defun do-allocation ()
  (when *prev-special-type*
    (si::allocate *prev-special-type*
	      (.gc.amount :get :return 'number)
	      t)
    (draw-status *prev-special-type*)))
       
(defun set-label-background (type colour)
  (and (get type 'frame)
       (let ((label (conc (get type 'frame) '.label)))
	 (funcall label :configure :background colour))))
	 

(defun mkgcmonitor()
  (let (si::*after-gbc-hook*)
    (toplevel '.gc)
    (wm :title '.gc "GC Monitor")
    (wm :title '.gc "GC")
    (or (> (read-from-string (winfo :depth '.gc)) 1)
	(setq *special-type-background* "white"))
    (message '.gc.msg :font :Adobe-times-medium-r-normal--*-180* :aspect 400
	     :text
	     "GC monitor displays after each garbage collection the amount of space used (red) and free (green) of the types in the list *gc-monitor-types*.   Clicking on a type makes its size appear on the scale at the bottom, and double clicking on the scale causes actual allocation!")
    (pack '.gc.msg :side "top")
    (dolist (v *gc-monitor-types*)
      (make-one-graph '.gc v)
      )
    (.gc :configure :borderwidth 4 :relief "ridge")
    ;; it is important to create the frame first, so that
    ;; it is earlier... and the others will show.
    (frame '.gc.ff)
    (button '.gc.ok :text "QUIT"
	    :command `(progn   (setq si::*after-gbc-hook* nil)
			       (destroy '.gc)))
    
    (scale '.gc.amount :label "Amount :" :width ".3c"
	   :orient "horizontal" :to 100)
    (pack '.gc.amount)
    (button '.gc.reset :text "RESET Number Used"
	    :command '(progn (dolist (v *gc-monitor-types*)
				     (set-label-background v "gray90"))
			     (si::reset-number-used)
			     (draw-status *prev-special-type*)))
    (button '.gc.update :text "Update"
	    :command '(draw-status *prev-special-type*))

    (pack '.gc.ok '.gc.reset '.gc.update :expand 1 :fill "x"
	  :in '.gc.ff :padx 3 :pady 2 :side 'left)

    (pack '.gc.ff :expand 1 :fill "x")
    (bind '.gc.amount "<Double-ButtonPress-1>"
			 'do-allocation)


    
    (draw-status nil))
  (setq si::*after-gbc-hook* 'draw-status)
  )
