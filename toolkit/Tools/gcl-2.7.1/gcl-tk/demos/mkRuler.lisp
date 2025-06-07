;;# mkRuler w
;;
;; Create a canvas demonstration consisting of a ruler.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.
;; This file implements a canvas widget that displays a ruler with tab stops
;; that can be set individually.  The only procedure that should be invoked
;; from outside the file is the first one, which creates the canvas.

(in-package "TK")

(defun mkRuler (&optional (w '.ruler)) 
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Ruler Demonstration")
    (wm :iconname w "Ruler")
    (setq c (conc w '.c))

    (message (conc w '.msg) :font :Adobe-Times-Medium-R-Normal-*-180-* :width "13c" 
	    :relief "raised" :bd 2 :text "This canvas widget shows a mock-up of a ruler.  You can create tab stops by dragging them out of the well to the right of the ruler.  You can also drag existing tab stops.  (if :you drag a tab stop far enough up or down so that it turns dim, it will be deleted when you release the mouse button.")
    (canvas c :width "14.8c" :height "2.5c" :relief "raised")
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.msg) (conc w '.c) :side "top" :fill "x")
    (pack (conc w '.ok) :side "bottom" :pady 5)
    (setf *v* (gensym))
    (setf (get *v* 'grid) '.25c)
    (setf (get *v* 'left) (winfo :fpixels c "1c" :return t))
    (setf (get *v* 'right) (winfo :fpixels c "13c" :return t))
    (setf (get *v* 'top) (winfo :fpixels c "1c" :return t))
    (setf (get *v* 'bottom) (winfo :fpixels c "1.5c" :return t))
    (setf (get *v* 'size) (winfo :fpixels c '.2c :return t))
    (setf (get *v* 'normalStyle) '(:fill "black"))
    (if (> (read-from-string (winfo :depth c)) 1)
	(progn 
	  (setf (get *v* 'activeStyle) '(:fill "red" :stipple ""))
	  (setf (get *v* 'deleteStyle)
		`(:stipple "@" : ,*tk-library* :"/demos/bitmaps/grey.25" 
			   :fill "red"))
	  );;else 
      (progn 
	(setf (get *v* 'activeStyle) '(:fill "black" :stipple "" ))
	(setf (get *v* 'deleteStyle)
	      `(:stipple "@" : ,*tk-library* : "/demos/bitmaps/grey.25"
			 :fill "black"))
	))

    (funcall c :create "line" "1c" "0.5c" "1c" "1c" "13c" "1c" "13c" "0.5c" :width 1)
    (dotimes
     (i  12)
     (let (( x (+ i 1)))
       (funcall c :create "line" x :"c" "1c" x :"c" "0.6c" :width 1)
       (funcall c :create "line" x :".25c" "1c"  x :".25c" "0.8c" :width 1)
       (funcall c :create "line"  x :".5c" "1c"  x :".5c" "0.7c" :width 1)
       (funcall c :create "line"  x :".75c" "1c"  x :".75c" "0.8c" :width 1)
       (funcall c :create "text"  x :".15c" '.75c :text i :anchor "sw")
       ))
    (funcall c :addtag "well" "withtag"
	     (funcall c :create "rect" "13.2c" "1c" "13.8c" "0.5c" 
		      :outline "black" :fill
		      (nth 4 (funcall c :config :background
				      :return 'list-strings))))
    (funcall c :addtag "well" "withtag"
	     (rulerMkTab c (winfo :pixels c "13.5c" :return t)
			 (winfo :pixels c '.65c :return t)))

    (funcall c :bind "well" "<1>" `(rulerNewTab ',c |%x| |%y|))
    (funcall c :bind "tab" "<1>" `(demo_selectTab  ',c |%x| |%y|))
    (bind c "<B1-Motion>" `(rulerMoveTab ',c |%x| |%y|))
    (bind c "<Any-ButtonRelease-1>" `(rulerReleaseTab ',c))
)

(defun rulerMkTab (c x y) 

    (funcall c :create "polygon" x y (+ x (get *v* 'size))
	     (+ y (get *v* 'size))
	     (- x (get *v* 'size))
	     (+ y (get *v* 'size))
	     :return 'string
	     )

)

(defun rulerNewTab (c x y) 

    (funcall c :addtag "active" "withtag" (rulerMkTab c x y))
    (funcall c :addtag "tab" "withtag" "active")
    (setf (get *v* 'x) x)
    (setf (get *v* 'y) y)
    (rulerMoveTab c x y)
)
(defvar *recursive* nil)
;; prevent recursive calls
(defun rulerMoveTab (c x y &aux cx cy (*recursive* *recursive*) )
  (cond (*recursive* (return-from rulerMoveTab))
	(t (setq *recursive* t)))
  (if (equal (funcall c :find "withtag" "active" :return 'string) "")
      (return-from rulerMoveTab nil))
  (setq cx (funcall c :canvasx x (get *v* 'grid) :return t))
  (setq cy (funcall c :canvasy y :return t))
  (if (<  cx  (get *v* 'left))(setq cx (get *v* 'left)))
  (if (> cx  (get *v* 'right))(setq cx (get *v* 'right)))

  (if (and (>= cy (get *v* 'top)) (<= cy (get *v* 'bottom)))
      (progn 
	(setq cy (+ 2 (get *v* 'top)))
	(apply c :itemconf "active" (get *v* 'activestyle)))
    
    (progn 
      (setq cy (- cy (get *v* 'size) 2))
      (apply c :itemconf "active"(get *v* 'deletestyle)))
    )
  (funcall c :move "active" (- cx (get *v* 'x))
	   (- cy (get *v* 'y)) )
  (setf (get *v* 'x) cx)
  (setf (get *v* 'y) cy)
  )

(defun demo_selectTab (c x y) 

    (setf (get *v* 'x) (funcall c :canvasx x (get *v* 'grid) :return t))
    (setf (get *v* 'y) (+ 2  (get *v* 'top)))
    (funcall c :addtag "active" "withtag" "current")
    (apply  c :itemconf "active" (get *v* 'activeStyle))
    (funcall c :raise "active")
)

(defun rulerReleaseTab (c )

    (if (equal (funcall c :find "withtag" "active" :return 'string)
	       "") (return-from rulerReleaseTab nil))

    (if (not (eql (get *v* 'y) (+ 2 (get *v* 'top))))
	(funcall c :delete "active")
     (progn
	(apply c :itemconf "active" (get *v* 'normalStyle))
	(funcall c :dtag "active")
    )
))
