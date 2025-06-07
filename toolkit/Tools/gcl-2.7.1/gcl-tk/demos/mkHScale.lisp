;;# mkHScale w
;;
;; Create a top-level window that displays a horizontal scale.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(in-package "TK")

(defun mkHScale (&optional (w '.scale2)) 
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Horizontal Scale Demonstration")
    (wm :iconname w "Scale")
    (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 300 
	    :text "A bar and a horizontal scale are displayed below.  (if :you click or drag mouse button 1 in the scale, you can change the width of the bar.  Click the \"OK\" button when you're finished.")
    (frame (conc w '.frame) :borderwidth 10)
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.msg) (conc w '.frame) (conc w '.ok) :side "top" :fill "x")

    (frame (conc w '.frame.top) :borderwidth 15)
    (scale (conc w '.frame.scale) :orient "horizontal" :length 280 :from 0 :to 250 
	    :command (tk-conc "setWidth " w ".frame.top.inner") :tickinterval 50 
	    :bg "Bisque1")
    (frame (conc w '.frame.top.inner) :width 20 :height 40 :relief "raised" :borderwidth 2 
	    :bg "SteelBlue1")
    (pack (conc w '.frame.top) :side "top" :expand "yes" :anchor "sw")
    (pack (conc w '.frame.scale) :side "bottom" :expand "yes" :anchor "nw")


    (pack (conc w '.frame.top.inner) :expand "yes" :anchor "sw")
    (funcall (conc w '.frame.scale) :set 20)
)

(defun setWidth (w width) 
    (funcall w :config  :width ${width} :height 40)
)
