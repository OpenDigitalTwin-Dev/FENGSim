;;# mkCanvText w
;;
;; Create a top-level window containing a canvas displaying a text
;; string and allowing the string to be edited and re-anchored.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.
(in-package "TK")
(defun mkCanvText ({w .ctext}) 
    (catch {destroy w})
    (toplevel w)
    (dpos w)
    (wm :title w "Canvas Text Demonstration")
    (wm :iconname w "Text")
    (setq c (conc w '.c))

    (message (conc w '.msg) :font -Adobe-Times-Medium-R-Normal-*-180-* :width 420 
	    :relief "raised" :bd 2 :text "This window displays a string of text to demonstrate the text facilities of canvas widgets.  You can point, click, and type.  You can also select and then delete with Control-d.  You can copy the selection with Control-v.  You can click in the boxes to adjust the position of the text relative to its positioning point or change its justification.")
    (canvas c :relief "raised" :width 500 :height 400)
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.msg) :side "top" :fill "both")
    (pack (conc w '.c) :side "top" :expand "yes" :fill "both")
    (pack (conc w '.ok) :side "bottom" :pady 5 :anchor "center")

    (setq font :Adobe-helvetica-medium-r-*-240-*)

    (funcall c :create rectangle 245 195 255 205 :outline "black" :fill "red")

    ;; First, create the text item and give it bindings so it can be edited.
    
    (funcall c :addtag text withtag (funcall c create text 250 200 :text "This is just a string of text to demonstrate the text facilities of canvas widgets. You can point, click, and type.  You can also select and then delete with Control-d." :width 440 :anchor "n" :font font :justify "left"))
    (funcall c :bind text "<1>" (textB1Press  c  |%x| |%y|))
    (funcall c :bind text "<B1-Motion>" (textB1Move  c  %x %y))
    (funcall c :bind text "<Shift-1>" (tk-conc c " select adjust current @%x,%y"))
    (funcall c :bind text "<Shift-B1-Motion>" (funcall 'textB1Move  c  |%x| |%y|))
    (funcall c :bind text "<KeyPress>" (tk-conc c " insert text insert %A"))
    (funcall c :bind text "<Shift-KeyPress>" (tk-conc c " insert text insert %A"))
    (funcall c :bind text "<Return>" (tk-conc c " insert text insert \\n"))
    (funcall c :bind text "<Control-h>" (funcall 'textBs c))
    (funcall c :bind text "<Delete>" (funcall  'textBs  c))
    (funcall c :bind text "<Control-d>" (tk-conc c " dchars text sel.first sel.last"))
    (funcall c :bind text "<Control-v>" (tk-conc c " insert text insert \[selection get\]"))

    ;; Next, create some items that allow the text's anchor position
    ;; to be edited.

    (setq x 50)
    (setq y 50)
    (setq color LightSkyBlue1)
    (mkTextConfig c x y :anchor "se" color)
    (mkTextConfig c (+ x 30) y :anchor "s" color)
    (mkTextConfig c (+ x 60) y :anchor "sw" color)
    (mkTextConfig c x (+ y 30) :anchor "e" color)
    (mkTextConfig c (+ x 30) (+ y 30) :anchor "center" color)
    (mkTextConfig c (+ x 60) (+ y 30) :anchor "w" color)
    (mkTextConfig c x (+ y 60) :anchor "ne" color)
    (mkTextConfig c (+ x 30) (+ y 60) :anchor "n" color)
    (mkTextConfig c (+ x 60) (+ y 60) :anchor "nw" color)
    (setq item (funcall c create rect (+ x 40) (+ y 40) (+ x 50) (+ y 50) 
	    :outline "black" :fill "red"))
    (funcall c :bind item "<1>" (tk-conc c " itemconf text :anchor ")center"")
     (funcall c :create text (+ x 45) (- y 5) :text "{Text Position}" :anchor "s" 
	    :font -Adobe-times-medium-r-normal--*-240-* :fill "brown")

    ;; Lastly, create some items that allow the text's justification to be
    ;; changed.
    
    (setq x 350)
    (setq y 50)
    (setq color SeaGreen2)
    (mkTextConfig c x y :justify "left" color)
    (mkTextConfig c (+ x 30) y :justify "center" color)
    (mkTextConfig c (+ x 60) y :justify "right" color)
    (funcall c :create text (+ x 45) (- y 5) :text "Justification" :anchor "s" 
	    :font -Adobe-times-medium-r-normal--*-240-* :fill "brown")

    (funcall c :bind config "<Enter>" (tk-conc "textEnter " c))
    (funcall c :bind config "<Leave>" (tk-conc c " itemconf current :fill \$textConfigFill"))
)

(defun mkTextConfig (w x y option value color) 
    (setq item (funcall w create rect x y (+ x 30) (+ y 30) 
	    :outline "black" :fill color :width 1))
    (funcall w :bind item "<1>" (tk-conc w " itemconf text " option " " value))
    (funcall w :addtag "config" "withtag" item)
)

(setq textConfigFill "")

(defun textEnter (w) 
    (global :textConfigFill)
    (setq textConfigFill [lindex (funcall w :itemconfig "current" :fill) 4])
    (funcall w :itemconfig "current" :fill "black")
)

(defun textB1Press (w x y) 
    (funcall w :icursor "current" (aT x y))
    (funcall w :focus "current")
    (focus w)
    (funcall w :select "from" "current" (aT x y))
)

(defun textB1Move (w x y) 
    (funcall w :select "to current" (aT x y))
)

(defun textBs (w &aux char) 
    (setq char  (atoi (funcall w :index "text" "insert")) - 1)
    (if (>= char  0) (funcall w :dchar "text" char))
)
