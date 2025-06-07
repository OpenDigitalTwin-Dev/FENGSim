(in-package "TK")
;;# mkVScale w
;;
;; Create a top-level window that displays a vertical scale.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(defun mkVScale (&optional (w '.vscale ))
;    (catch {destroy w})
    (toplevel w)
    (dpos w)
    (wm :title w "Vertical Scale Demonstration")
    (wm :iconname w "Scale")
    (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 300 
	    :text "A bar and a vertical scale are displayed below.  If you click or drag mouse button 1 in the scale, you can change the height of the bar.  Click the OK button when you're finished.")
    (frame (conc w '.frame) :borderwidth 10)
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.msg) (conc w '.frame) (conc w '.ok))

    (scale (conc w '.frame.scale) :orient "vertical" :length 280 :from 0 :to 250 
	    :command #'(lambda (height)
			 ; (print height)
                          (setHeight  (conc w '.frame.right.inner) height)) 
            :tickinterval 50 
	    :bg "Bisque1")
    (frame (conc w '.frame.right) :borderwidth 15)
    (frame (conc w '.frame.right.inner) :width 40 :height 20 :relief "raised" 
	    :borderwidth 2 :bg "SteelBlue1")
    (pack (conc w '.frame.scale) :side "left" :anchor "ne")
    (pack (conc w '.frame.right) :side "left" :anchor "nw")
    (funcall (conc w '.frame.scale) :set 20)


    (pack (conc w '.frame.right.inner) :expand "yes" :anchor "nw")
)

(defun setHeight (w height) 
    (funcall w :config :width 40 :height height)
)
