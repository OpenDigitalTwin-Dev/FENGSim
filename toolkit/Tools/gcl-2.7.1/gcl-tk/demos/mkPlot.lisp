(in-package "TK")
;;# mkPlot w
;;
;; Create a top-level window containing a canvas displaying a simple
;; graph with data points that can be moved interactively.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(defun mkPlot ( &optional (w '.plot ) &aux c font x y item)
    (toplevel w )
    (dpos w)
    (wm :title w  "Plot Demonstration " : w)
    (wm :iconname w "Plot")
    (setq c (conc w '.c))

    (message (conc w '.msg) :font :Adobe-Times-Medium-R-Normal-*-180-* :width 400 
	    :bd 2 :relief "raised" :text "This window displays a canvas widget containing a simple 2-dimensional plot.  You can doctor the data by dragging any of the points with mouse button 1.")
    (canvas c :relief "raised" :width 450 :height 300)
    (button (conc w '.ok) :text "OK" :command  (tk-conc "destroy " w))
    (pack (conc w '.msg) (conc w '.c) :side "top" :fill "x")
    (pack (conc w '.ok) :side "bottom" :pady 5)

    (setq font :Adobe-helvetica-medium-r-*-180-*)

    (funcall c :create "line" 100 250 400 250 :width 2)
    (funcall c :create "line" 100 250 100 50 :width 2)
    (funcall c :create "text" 225 20 :text "A Simple Plot" :font font :fill "brown")
    
    (sloop for i to 10 do 
	(setq x (+ 100 (* i 30)))
	(funcall c :create "line" x 250 x 245 :width 2)
	(funcall c :create "text" x 254 :text (* 10 i) :anchor "n" :font font))

    (sloop for i to 5 do  
	(setq y (- 250 (* i 40)))
	(funcall c :create "line" 100 y 105 y :width 2)
	(funcall c :create "text" 96 y :text  (* i 50) : ".0" :anchor "e" :font font))

    (sloop for point in '((12 56) (20 94) (33 98) (32 120) (61 180)
			  (75 160) (98 223))
       do
       (setq x (+ 100  (* 3 (nth 0 point))))
       (setq y (- 250 (truncate (* 4 (nth 1 point)) 5)))
       (setq item (funcall c :create "oval" (- x 6) (- y 6) 
			   (+ x 6) (+ y 6) :width 1 :outline "black" 
			   :fill "SkyBlue2" :return 'string ))
       (funcall c :addtag "point" "withtag" item)
    )
    

    (funcall c :bind "point" "<Any-Enter>"  c : " itemconfig current -fill red")
    (funcall c :bind "point" "<Any-Leave>"   c : " itemconfig current -fill SkyBlue2")
    (funcall c :bind "point" "<1>"  `(plotdown ',c |%x| |%y|))
    (funcall c :bind "point" "<ButtonRelease-1>"  c : " dtag selected")
    (bind c "<B1-Motion>" `(plotmove ',c |%x| |%y|))
)

(defvar plotlastX 0)
(defvar plotlastY 0)

(defun plotDown (w x y) 
    (funcall w :dtag "selected")
    (funcall w :addtag "selected" "withtag" "current")
    (funcall w :raise "current")
    (setq plotlastY y)
    (setq plotlastX x)
)

(defun plotMove (w x y &aux )
  (let ((oldx  plotlastX)
	(oldy  plotlastY))
    ;; Note plotmove may be called recursively... since
    ;; the funcall may call something which calls this.
    ;; so we must set the global plotlastx before the funcall..
    (setq plotlastx x)
    (setq plotlastY y) 
    (funcall w :move "selected" (- x oldx) (- y oldy))
    )
  )

